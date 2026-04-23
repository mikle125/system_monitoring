import cv2
import json
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
from collections import OrderedDict, defaultdict
import time
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
import threading
import queue
import logging
import signal
import sys
import os
import base64
import uuid

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebRTCYOLOAnnotator:
    def __init__(self, flask_port=3000):
        """
        Серверный аннотатор с использованием WebRTC для захвата видео с камеры пользователя
        """
        try:
            self.model = YOLO('best.pt')
        except:
            # Используем стандартную модель YOLOv8 если best.pt не найден
            self.model = YOLO('yolov8n.pt')
            logger.warning("Модель best.pt не найдена, используется yolov8n.pt")
        
        # Словарь для перевода поз в прилагательные
        self.pose_adjectives = {
            'standing': 'стоящий',
            'sitting': 'сидящий',
            'lying': 'лежащий',
            'falling': 'падающий',
            'walking': 'идущий',
            'running': 'бегущий',
            'jumping': 'прыгающий',
            'crouching': 'приседающий',
            'bending': 'наклоняющийся',
            'kneeling': 'стоящий на коленях',
            'squatting': 'сидящий на корточках',
            'reaching': 'тянущийся',
            'raising': 'поднимающий',
            'pointing': 'указывающий',
            'holding': 'держащий',
            'carrying': 'несущий',
            'pushing': 'толкающий',
            'pulling': 'тянущий',
            'lifting': 'поднимающий',
            'throwing': 'бросающий',
            'catching': 'ловящий',
            'kicking': 'пинающий',
            'punching': 'бьющий',
            'waving': 'машущий',
            'clapping': 'хлопающий',
            'dancing': 'танцующий',
            'exercising': 'занимающийся спортом',
            'resting': 'отдыхающий',
            'sleeping': 'спящий',
            'working': 'работающий',
            'reading': 'читающий',
            'writing': 'пишущий',
            'typing': 'печатающий',
            'talking': 'разговаривающий',
            'listening': 'слушающий',
            'watching': 'смотрящий',
            'eating': 'едящий',
            'drinking': 'пьющий',
            'cooking': 'готовящий',
            'cleaning': 'убирающий',
            'washing': 'моющий',
            'brushing': 'чистящий',
            'combing': 'расчесывающий',
            'shaving': 'бреющийся',
            'showering': 'принимающий душ',
            'bathing': 'купающийся',
            'swimming': 'плывущий',
            'cycling': 'едущий на велосипеде',
            'driving': 'ведущий машину',
            'riding': 'едущий верхом',
            'flying': 'летящий',
            'sailing': 'плывущий под парусом',
            'rowing': 'гребущий',
            'skiing': 'катающийся на лыжах',
            'skating': 'катающийся на коньках',
            'snowboarding': 'катающийся на сноуборде',
            'surfing': 'серфингист',
            'climbing': 'лезущий',
            'hiking': 'идущий в поход',
            'camping': 'кемпингуемый',
            'fishing': 'рыбачащий',
            'hunting': 'охотящийся'
        }
        
        self.output_file = Path('annotations.json')
        self.annotations = OrderedDict()
        
        # Параметры для оптимизации
        self.prev_objects = None
        self.position_threshold = 50
        self.iou_threshold = 0.3
        
        # Очередь для кадров
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Статистика
        self.stats = {
            'total_frames': 0,
            'saved_frames': 0,
            'total_objects': 0,
            'fps': 0,
            'start_time': time.time(),
            'object_counts': {},
            'detection_history': [],
            'active_clients': 0,
            'daily_activity': defaultdict(lambda: defaultdict(int)),  # Для хранения активности по дням
            'weekly_distribution': defaultdict(lambda: defaultdict(float))  # Для недельного распределения
        }
        
        # Контроль работы
        self.running = True
        self.flask_port = flask_port
        self.pause_annotation = False
        
        # Клиенты
        self.clients = {}
        
        # Папки
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Настройки по умолчанию
        self.settings = {
            'confidence': 0.5,
            'show_boxes': True,
            'show_labels': True,
            'show_conf': True,
            'box_color': '#3b82f6',
            'text_color': '#ffffff',
            'box_thickness': 2,
            'font_size': 12,
            'save_interval': 300,  # 5 минут
            'max_fps': 30,
            'detection_mode': 'balanced',  # fast, balanced, accurate
            'use_russian_poses': True  # Использовать русские прилагательные для поз
        }
        
        # Запуск Flask
        self.flask_thread = threading.Thread(target=self.start_flask_server)
        self.flask_thread.daemon = True
        self.flask_thread.start()
        
        # Обработка сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Сервер запущен на порту: {flask_port}")
    
    def translate_pose(self, pose_name):
        """Перевод названия позы на русский язык в виде прилагательного"""
        if self.settings.get('use_russian_poses', True):
            # Приводим к нижнему регистру для поиска
            pose_lower = pose_name.lower().strip()
            
            # Прямой перевод из словаря
            if pose_lower in self.pose_adjectives:
                return self.pose_adjectives[pose_lower]
            
            # Обработка составных названий
            for eng, rus in self.pose_adjectives.items():
                if eng in pose_lower:
                    return rus
            
            # Если не нашли, возвращаем исходное с русским окончанием
            # Добавляем окончание в зависимости от контекста
            if pose_lower.endswith('ing'):
                # Заменяем ing на 'щий'
                base = pose_lower[:-3]
                if base.endswith('t'):
                    return base + 'ящий'
                elif base.endswith('k'):
                    return base + 'ающий'
                else:
                    return base + 'ающий'
            
            return pose_name + ' (поза)'
        return pose_name
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        logger.info(f"Получен сигнал {signum}, завершение...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Очистка ресурсов"""
        self._save_to_json(final=True)
        logger.info("Ресурсы освобождены")
    
    def calculate_weekly_distribution(self):
        """Вычисление недельного распределения активности"""
        weekly_distribution = {}
        week_days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        
        # Получаем данные за последние 7 дней
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).date()
            day_name = week_days[date.weekday()]
            
            if date.isoformat() in self.stats['daily_activity']:
                day_data = self.stats['daily_activity'][date.isoformat()]
                total = sum(day_data.values())
                
                # Вычисляем общую активность за день в процентах
                if total > 0:
                    weekly_distribution[day_name] = min(100, total / 10)  # Нормализуем
                else:
                    weekly_distribution[day_name] = 0
            else:
                weekly_distribution[day_name] = 0
        
        return weekly_distribution
    
    def start_flask_server(self):
        """Запуск Flask сервера"""
        app = Flask(__name__)
        
        # Современный HTML интерфейс с Bootstrap 5
        HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Мониторинг</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Custom CSS -->
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --dark-bg: #0f172a;
            --dark-card: rgba(30, 41, 59, 0.8);
            --light-bg: #f8fafc;
            --light-card: rgba(255, 255, 255, 0.8);
            --glass-bg: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --accent-color: #3b82f6;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
        }
        
        [data-bs-theme="dark"] {
            background-color: var(--dark-bg) !important;
            color: var(--text-primary) !important;
        }
        
        [data-bs-theme="light"] {
            background-color: var(--light-bg) !important;
            color: #1e293b !important;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        
        /* Glass effect */
        .glass {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .glass-dark {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .glass-light {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .btn-glass {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: inherit;
            transition: all 0.3s ease;
        }
        
        .btn-glass:hover {
            background: rgba(255, 255, 255, 0.25);
            border-color: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        .btn-glass-primary {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
        }
        
        .navbar-glass {
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .navbar-glass[data-bs-theme="light"] {
            background: rgba(248, 250, 252, 0.9);
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .card-glass {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            transition: all 0.3s ease;
        }
        
        [data-bs-theme="light"] .card-glass {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .card-glass:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        .video-container-wrapper {
            position: relative;
            border-radius: 20px;
            overflow: hidden;
            background: #000;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            border: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        #webcamCanvas {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        
        .video-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to bottom, rgba(0,0,0,0.8) 0%, transparent 100%);
            padding: 1.5rem;
            z-index: 10;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(21, 128, 61, 0.2);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 50px;
            backdrop-filter: blur(10px);
        }
        
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-active { background: #22c55e; }
        .status-paused { background: #ef4444; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .stat-card {
            padding: 1.5rem;
            text-align: center;
            border-radius: 16px;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        
        [data-bs-theme="dark"] .stat-value {
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        [data-bs-theme="light"] .stat-value {
            background: linear-gradient(135deg, #1e40af, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        [data-bs-theme="dark"] .stat-label {
            color: var(--text-secondary);
        }
        
        [data-bs-theme="light"] .stat-label {
            color: #64748b;
        }
        
        .object-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            font-size: 0.875rem;
            margin: 4px;
        }
        
        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            transition: background 0.2s ease;
        }
        
        [data-bs-theme="dark"] .detection-item {
            border-bottom: 1px solid var(--border-color);
        }
        
        [data-bs-theme="light"] .detection-item {
            border-bottom: 1px solid #e2e8f0;
        }
        
        .detection-item:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        [data-bs-theme="light"] .detection-item:hover {
            background: rgba(0, 0, 0, 0.05);
        }
        
        .progress-bar-custom {
            height: 6px;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        [data-bs-theme="dark"] .progress-bar-custom {
            background: rgba(255, 255, 255, 0.1);
        }
        
        [data-bs-theme="light"] .progress-bar-custom {
            background: rgba(0, 0, 0, 0.1);
        }
        
        .progress-custom {
            height: 100%;
            background: var(--primary-gradient);
            transition: width 0.3s ease;
        }
        
        .control-panel {
            position: sticky;
            top: 20px;
            z-index: 100;
        }
        
        .settings-panel {
            background: rgba(30, 41, 59, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }
        
        [data-bs-theme="light"] .settings-panel {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .form-range::-webkit-slider-thumb {
            background: var(--accent-color);
        }
        
        .form-range::-moz-range-thumb {
            background: var(--accent-color);
        }
        
        .form-check-input:checked {
            background-color: var(--accent-color);
            border-color: var(--accent-color);
        }
        
        .empty-state {
            text-align: center;
            padding: 3rem;
        }
        
        [data-bs-theme="dark"] .empty-state {
            color: var(--text-secondary);
        }
        
        [data-bs-theme="light"] .empty-state {
            color: #64748b;
        }
        
        .empty-state i {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
        
        .chart-container {
            position: relative;
            height: 200px;
            margin: 1rem 0;
        }
        
        .chart-container-large {
            position: relative;
            height: 300px;
            margin: 1rem 0;
        }
        
        .floating-controls {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 1000;
        }
        
        .floating-btn {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .floating-btn:hover {
            transform: scale(1.1) translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        }
        
        .color-picker {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            cursor: pointer;
        }
        
        .nav-tabs-glass {
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        [data-bs-theme="light"] .nav-tabs-glass {
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .nav-tabs-glass .nav-link {
            color: inherit;
            border: none;
            background: transparent;
            border-radius: 12px 12px 0 0;
            margin-bottom: -1px;
            padding: 12px 24px;
        }
        
        .nav-tabs-glass .nav-link.active {
            background: rgba(255, 255, 255, 0.1);
            border-bottom: 2px solid var(--accent-color);
        }
        
        [data-bs-theme="light"] .nav-tabs-glass .nav-link.active {
            background: rgba(0, 0, 0, 0.05);
        }
        
        .tab-pane {
            padding: 1.5rem 0;
        }
        
        .slider-with-value {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .slider-with-value input {
            flex: 1;
        }
        
        .slider-value {
            min-width: 60px;
            text-align: right;
            font-weight: 600;
        }
        
        .pose-badge {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 5px;
        }
        
        @media (max-width: 768px) {
            .floating-controls {
                bottom: 1rem;
                right: 1rem;
            }
            
            .stat-value {
                font-size: 1.5rem;
            }
            
            .video-container-wrapper {
                border-radius: 12px;
            }
            
            .chart-container-large {
                height: 250px;
            }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        }
        
        [data-bs-theme="light"] ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.05);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }
        
        [data-bs-theme="light"] ::-webkit-scrollbar-thumb {
            background: rgba(0, 0, 0, 0.2);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        [data-bs-theme="light"] ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 0, 0, 0.3);
        }
        
        /* Animations */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .glow {
            animation: glow 2s infinite alternate;
        }
        
        @keyframes glow {
            from { box-shadow: 0 0 20px rgba(59, 130, 246, 0.5); }
            to { box-shadow: 0 0 30px rgba(59, 130, 246, 0.8); }
        }
        
        .chart-title {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .stats-section {
            margin-top: 1rem;
        }
    </style>
</head>
<body data-bs-theme="dark">
    <!-- Navigation -->
    <nav class="navbar navbar-glass navbar-expand-lg">
        <div class="container-fluid">
            <a class="navbar-brand d-flex align-items-center gap-2" href="#">
                <div class="p-2 rounded glass" style="background: linear-gradient(135deg, #667eea, #764ba2);">
                    <i class="bi bi-person-standing text-white"></i>
                </div>
                <span class="fw-bold">Мониторинг</span>
            </a>
            
            <div class="d-flex align-items-center gap-3">
                <div class="status-badge glass">
                    <span class="status-indicator status-active" id="statusIndicator"></span>
                    <span id="statusText" class="small fw-medium">Подключение...</span>
                </div>
                <button class="btn btn-glass btn-sm" onclick="toggleTheme()" id="themeToggle">
                    <i class="bi bi-moon-stars"></i>
                </button>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container-fluid py-4">
        <div class="row g-4">
            <!-- Left Column - Video Feed -->
            <div class="col-xl-8 col-lg-7">
                <!-- Video Container -->
                <div class="card-glass p-0 mb-4">
                    <div class="video-container-wrapper">
                        <div class="video-overlay">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="d-flex align-items-center gap-3">
                                    <h5 class="mb-0 text-white"><i class="bi bi-camera-video"></i> Видеопоток</h5>
                                    <span class="badge glass" style="background: rgba(0,0,0,0.5);">
                                        <i class="bi bi-lightning-charge"></i> <span id="fpsDisplay">0</span> FPS
                                    </span>
                                </div>
                                <div id="currentObjects" class="d-flex flex-wrap gap-2"></div>
                            </div>
                        </div>
                        <video id="webcamVideo" autoplay playsinline class="w-100" style="max-height: 70vh; object-fit: contain;"></video>
                        <canvas id="webcamCanvas" class="w-100"></canvas>
                    </div>
                </div>

                <!-- Controls -->
                <div class="row g-3 mb-4">
                    <div class="col-md-3 col-6">
                        <button class="btn btn-glass-primary w-100 d-flex align-items-center justify-content-center gap-2" onclick="startWebcam()" id="startBtn">
                            <i class="bi bi-camera-video"></i> <span>Старт</span>
                        </button>
                    </div>
                    <div class="col-md-3 col-6">
                        <button class="btn btn-glass w-100 d-flex align-items-center justify-content-center gap-2" onclick="togglePause()" id="pauseBtn">
                            <i class="bi bi-pause-circle"></i> <span>Пауза</span>
                        </button>
                    </div>
                    <div class="col-md-3 col-6">
                        <button class="btn btn-glass w-100 d-flex align-items-center justify-content-center gap-2" onclick="takeSnapshot()">
                            <i class="bi bi-camera"></i> <span>Снимок</span>
                        </button>
                    </div>
                    <div class="col-md-3 col-6">
                        <button class="btn btn-glass w-100 d-flex align-items-center justify-content-center gap-2" onclick="saveSession()">
                            <i class="bi bi-save"></i> <span>Сохранить</span>
                        </button>
                    </div>
                </div>

                <!-- Recent Detections -->
                <div class="card-glass mb-4">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="card-title mb-0"><i class="bi bi-person-arms-up"></i> Обнаруженные позы</h5>
                            <span class="badge glass" style="background: rgba(0,0,0,0.3);" id="detectionCount">0</span>
                        </div>
                        <div class="detection-list" id="detectionsList" style="max-height: 200px; overflow-y: auto;">
                            <div class="empty-state">
                                <i class="bi bi-eye-slash"></i>
                                <p class="mb-0">Нет обнаружений</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column - Sidebar -->
            <div class="col-xl-4 col-lg-5">
                <div class="control-panel">
                    <!-- Activity Graphs Section -->
                    <div class="card-glass mb-4">
                        <div class="card-body">
                            <h5 class="card-title mb-3"><i class="bi bi-graph-up"></i> Статистика активности</h5>
                            
                            <!-- Daily Activity -->
                            <div class="chart-title">Активность за сегодня (%)</div>
                            <div class="chart-container-large">
                                <canvas id="dailyActivityChart"></canvas>
                            </div>
                            
                            <!-- Weekly Distribution -->
                            <div class="chart-title mt-4">Распределение за неделю (%)</div>
                            <div class="chart-container-large">
                                <canvas id="weeklyDistributionChart"></canvas>
                            </div>
                        </div>
                    </div>

                    <!-- Simple Stats -->
                    <div class="row g-3 mb-4">
                        <div class="col-md-6">
                            <div class="card-glass stat-card">
                                <div class="stat-value" id="totalFrames">0</div>
                                <div class="stat-label">Обнаружено объектов</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card-glass stat-card">
                                <div class="stat-value" id="savedFrames">0</div>
                                <div class="stat-label">Сохранено снимков</div>
                            </div>
                        </div>
                    </div>

                    <!-- Object Distribution -->
                    <div class="card-glass">
                        <div class="card-body">
                            <h5 class="card-title mb-3"><i class="bi bi-pie-chart"></i> Распределение поз</h5>
                            <div id="objectDistribution" class="mb-3"></div>
                            <div class="chart-container">
                                <canvas id="objectsChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Settings Modal -->
    <div class="modal fade" id="settingsModal" tabindex="-1">
        <div class="modal-dialog modal-lg modal-dialog-centered">
            <div class="modal-content settings-panel">
                <div class="modal-header border-0 pb-0">
                    <h5 class="modal-title"><i class="bi bi-sliders"></i> Настройки</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <ul class="nav nav-tabs nav-tabs-glass mb-4" id="settingsTabs">
                        <li class="nav-item">
                            <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#basic-tab">
                                <i class="bi bi-gear"></i> Основные
                            </button>
                        </li>
                        <li class="nav-item">
                            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#visual-tab">
                                <i class="bi bi-palette"></i> Визуальные
                            </button>
                        </li>
                        <li class="nav-item">
                            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#export-tab">
                                <i class="bi bi-download"></i> Экспорт
                            </button>
                        </li>
                    </ul>
                    
                    <div class="tab-content">
                        <!-- Basic Settings -->
                        <div class="tab-pane fade show active" id="basic-tab">
                            <div class="mb-4">
                                <label class="form-label">Чувствительность: <span id="confidenceValue" class="fw-bold">0.5</span></label>
                                <div class="slider-with-value">
                                    <input type="range" class="form-range" id="confidenceSlider" min="0.1" max="0.9" step="0.05" value="0.5">
                                    <span class="slider-value" id="confidencePercent">50%</span>
                                </div>
                            </div>
                            <div class="mb-4">
                                <label class="form-label">Частота обработки</label>
                                <div class="slider-with-value">
                                    <input type="range" class="form-range" id="fpsSlider" min="1" max="30" step="1" value="10">
                                    <span class="slider-value"><span id="fpsValue">10</span> FPS</span>
                                </div>
                            </div>
                            <div class="mb-4">
                                <label class="form-label">Режим работы</label>
                                <select class="form-select glass" id="detectionMode">
                                    <option value="fast">Быстрый</option>
                                    <option value="balanced" selected>Сбалансированный</option>
                                    <option value="accurate">Точный</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="autoSave" checked>
                                    <label class="form-check-label" for="autoSave">Автосохранение</label>
                                </div>
                            </div>
                            <div class="mb-3">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="useRussianPoses" checked>
                                    <label class="form-check-label" for="useRussianPoses">
                                        <i class="bi bi-translate"></i> Русские прилагательные для поз
                                    </label>
                                </div>
                                <small class="text-muted d-block mt-1">Отображать позы как: стоящий, сидящий, лежащий, падающий и т.д.</small>
                            </div>
                        </div>
                        
                        <!-- Visual Settings -->
                        <div class="tab-pane fade" id="visual-tab">
                            <div class="row g-4">
                                <div class="col-md-6">
                                    <label class="form-label">Цвет рамок</label>
                                    <div class="d-flex align-items-center gap-3">
                                        <input type="color" class="color-picker" id="boxColor" value="#3b82f6">
                                        <span id="boxColorText" class="small">#3b82f6</span>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Цвет текста</label>
                                    <div class="d-flex align-items-center gap-3">
                                        <input type="color" class="color-picker" id="textColor" value="#ffffff">
                                        <span id="textColorText" class="small">#ffffff</span>
                                    </div>
                                </div>
                            </div>
                            <div class="mt-4">
                                <label class="form-label">Толщина рамок</label>
                                <div class="slider-with-value">
                                    <input type="range" class="form-range" id="boxThickness" min="1" max="5" step="1" value="2">
                                    <span class="slider-value"><span id="thicknessValue">2</span> px</span>
                                </div>
                            </div>
                            <div class="mt-4">
                                <label class="form-label">Размер шрифта</label>
                                <div class="slider-with-value">
                                    <input type="range" class="form-range" id="fontSize" min="10" max="24" step="1" value="12">
                                    <span class="slider-value"><span id="fontValue">12</span> px</span>
                                </div>
                            </div>
                            <div class="mt-4">
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input" type="checkbox" id="showBoxes" checked>
                                    <label class="form-check-label" for="showBoxes">Показывать рамки</label>
                                </div>
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input" type="checkbox" id="showLabels" checked>
                                    <label class="form-check-label" for="showLabels">Показывать метки</label>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Export Settings -->
                        <div class="tab-pane fade" id="export-tab">
                            <div class="mb-4">
                                <label class="form-label">Формат экспорта</label>
                                <select class="form-select glass" id="exportFormat">
                                    <option value="json">JSON</option>
                                    <option value="csv">CSV</option>
                                    <option value="xml">XML</option>
                                </select>
                            </div>
                            <div class="mb-4">
                                <button class="btn btn-glass w-100 mb-2" onclick="exportAnnotations()">
                                    <i class="bi bi-download"></i> Экспорт данных
                                </button>
                                <button class="btn btn-glass w-100 mb-2" onclick="clearAnnotations()">
                                    <i class="bi bi-trash"></i> Очистить данные
                                </button>
                                <button class="btn btn-glass w-100" onclick="resetStatistics()">
                                    <i class="bi bi-arrow-clockwise"></i> Сбросить статистику
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer border-0">
                    <button type="button" class="btn btn-glass" data-bs-dismiss="modal">Отмена</button>
                    <button type="button" class="btn btn-glass-primary" onclick="applySettings()">Применить настройки</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Floating Action Buttons -->
    <div class="floating-controls">
        <button class="floating-btn btn-glass-primary" onclick="downloadAnnotations()" data-bs-toggle="tooltip" title="Экспорт данных">
            <i class="bi bi-download"></i>
        </button>
        <button class="floating-btn glass" data-bs-toggle="modal" data-bs-target="#settingsModal" title="Настройки">
            <i class="bi bi-sliders"></i>
        </button>
    </div>

    <!-- Toast Container -->
    <div class="toast-container position-fixed bottom-0 end-0 p-3"></div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        // Global variables
        let cameraStream = null;
        let isProcessing = false;
        let clientId = null;
        let frameInterval = null;
        let charts = {};
        let settings = {
            confidence: 0.5,
            showBoxes: true,
            showLabels: true,
            fps: 10,
            autoSave: true,
            detectionMode: 'balanced',
            boxColor: '#3b82f6',
            textColor: '#ffffff',
            boxThickness: 2,
            fontSize: 12,
            useRussianPoses: true
        };
        
        // Chart colors
        const chartColorsDark = {
            primary: 'rgba(59, 130, 246, 0.8)',
            secondary: 'rgba(147, 51, 234, 0.8)',
            success: 'rgba(34, 197, 94, 0.8)',
            warning: 'rgba(245, 158, 11, 0.8)',
            danger: 'rgba(239, 68, 68, 0.8)',
            info: 'rgba(6, 182, 212, 0.8)',
            grid: 'rgba(255, 255, 255, 0.1)',
            text: 'rgba(248, 250, 252, 0.8)'
        };
        
        const chartColorsLight = {
            primary: 'rgba(30, 64, 175, 0.8)',
            secondary: 'rgba(124, 58, 237, 0.8)',
            success: 'rgba(21, 128, 61, 0.8)',
            warning: 'rgba(180, 83, 9, 0.8)',
            danger: 'rgba(185, 28, 28, 0.8)',
            info: 'rgba(8, 145, 178, 0.8)',
            grid: 'rgba(0, 0, 0, 0.1)',
            text: 'rgba(30, 41, 59, 0.8)'
        };
        
        // Function to show camera error
        function showCameraError(error) {
            console.error('Camera error:', error);
            
            let errorMessage = 'Неизвестная ошибка';
            
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Доступ к камере запрещен. Пожалуйста, разрешите доступ к камере в настройках браузера.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'Камера не найдена. Убедитесь, что камера подключена.';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Камера используется другим приложением.';
            } else if (error.name === 'OverconstrainedError') {
                errorMessage = 'Запрошенные параметры камеры не поддерживаются.';
            } else {
                errorMessage = `Ошибка: ${error.message || error.name}`;
            }
            
            alert(`❌ ${errorMessage}`);
        }
        
        // Initialize charts
        function initCharts() {
            const objectsCtx = document.getElementById('objectsChart').getContext('2d');
            const dailyCtx = document.getElementById('dailyActivityChart').getContext('2d');
            const weeklyCtx = document.getElementById('weeklyDistributionChart').getContext('2d');
            
            const colors = document.documentElement.getAttribute('data-bs-theme') === 'dark' 
                ? chartColorsDark : chartColorsLight;
            
            // Daily Activity Chart (Pie)
            charts.daily = new Chart(dailyCtx, {
                type: 'pie',
                data: {
                    labels: ['Загрузка...'],
                    datasets: [{
                        data: [100],
                        backgroundColor: [colors.primary],
                        borderColor: colors.grid,
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: colors.text,
                                padding: 20,
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return context.label + ': ' + context.parsed + '%';
                                }
                            }
                        }
                    }
                }
            });
            
            // Weekly Distribution Chart (Bar)
            charts.weekly = new Chart(weeklyCtx, {
                type: 'bar',
                data: {
                    labels: ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
                    datasets: [{
                        label: 'Активность (%)',
                        data: [0, 0, 0, 0, 0, 0, 0],
                        backgroundColor: colors.primary,
                        borderColor: colors.primary.replace('0.8', '1'),
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: colors.grid },
                            ticks: { 
                                color: colors.text,
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        },
                        x: {
                            grid: { color: colors.grid },
                            ticks: { color: colors.text }
                        }
                    }
                }
            });
            
            // Objects Chart (Line)
            charts.objects = new Chart(objectsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Объекты',
                        data: [],
                        borderColor: colors.success,
                        backgroundColor: colors.success.replace('0.8', '0.2'),
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: colors.grid },
                            ticks: { color: colors.text }
                        },
                        x: {
                            display: false
                        }
                    }
                }
            });
        }
        
        // Update chart colors based on theme
        function updateChartColors() {
            const colors = document.documentElement.getAttribute('data-bs-theme') === 'dark' 
                ? chartColorsDark : chartColorsLight;
            
            Object.keys(charts).forEach(chartName => {
                const chart = charts[chartName];
                if (chart) {
                    // Update grid colors
                    if (chart.options.scales) {
                        if (chart.options.scales.y) {
                            chart.options.scales.y.grid.color = colors.grid;
                            chart.options.scales.y.ticks.color = colors.text;
                        }
                        if (chart.options.scales.x) {
                            chart.options.scales.x.grid.color = colors.grid;
                            chart.options.scales.x.ticks.color = colors.text;
                        }
                    }
                    
                    // Update legend colors
                    if (chart.options.plugins?.legend?.labels) {
                        chart.options.plugins.legend.labels.color = colors.text;
                    }
                    
                    chart.update();
                }
            });
        }
        
        // Load settings from localStorage
        function loadSettings() {
            const saved = localStorage.getItem('visionai_settings');
            if (saved) {
                settings = { ...settings, ...JSON.parse(saved) };
            }
            updateSettingsUI();
        }
        
        // Save settings to localStorage
        function saveSettings() {
            localStorage.setItem('visionai_settings', JSON.stringify(settings));
        }
        
        // Update settings UI
        function updateSettingsUI() {
            // Update sliders and values
            document.getElementById('confidenceSlider').value = settings.confidence;
            document.getElementById('confidenceValue').textContent = settings.confidence.toFixed(2);
            document.getElementById('confidencePercent').textContent = Math.round(settings.confidence * 100) + '%';
            
            document.getElementById('fpsSlider').value = settings.fps;
            document.getElementById('fpsValue').textContent = settings.fps;
            
            document.getElementById('detectionMode').value = settings.detectionMode;
            document.getElementById('autoSave').checked = settings.autoSave;
            document.getElementById('useRussianPoses').checked = settings.useRussianPoses !== false;
            
            // Visual settings
            document.getElementById('boxColor').value = settings.boxColor;
            document.getElementById('boxColorText').textContent = settings.boxColor;
            document.getElementById('textColor').value = settings.textColor;
            document.getElementById('textColorText').textContent = settings.textColor;
            document.getElementById('boxThickness').value = settings.boxThickness;
            document.getElementById('thicknessValue').textContent = settings.boxThickness;
            document.getElementById('fontSize').value = settings.fontSize;
            document.getElementById('fontValue').textContent = settings.fontSize;
            document.getElementById('showBoxes').checked = settings.showBoxes;
            document.getElementById('showLabels').checked = settings.showLabels;
        }
        
        // Generate client ID
        function generateClientId() {
            return 'client_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
        
        async function startWebcam() {
            try {
                // Остановка предыдущего потока
                if (cameraStream) {
                    cameraStream.getTracks().forEach(track => track.stop());
                }
                
                // ГИБКИЕ настройки захвата
                const constraints = {
                    video: {
                        width: { 
                            min: 320,
                            ideal: 1280,
                            max: 1920
                        },
                        height: { 
                            min: 240,
                            ideal: 720,
                            max: 1080
                        },
                        frameRate: { 
                            min: 10,
                            ideal: 30,
                            max: 60
                        },
                        facingMode: { ideal: "user" }
                    },
                    audio: false
                };
                
                // Получение потока
                cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
                
                const video = document.getElementById('webcamVideo');
                video.srcObject = cameraStream;
                
                // Обновление кнопки
                const startBtn = document.getElementById('startBtn');
                startBtn.innerHTML = '<i class="bi bi-stop-circle"></i> <span>Стоп</span>';
                startBtn.className = 'btn btn-glass btn-danger';
                startBtn.onclick = stopWebcam;
                
                // Генерация ID клиента
                clientId = generateClientId();
                
                // Запуск обработки кадров
                startFrameProcessing();
                
                showToast('Камера запущена', 'success');
                
            } catch (error) {
                console.error('Ошибка доступа к камере:', error);
                showCameraError(error);
            }
        }
        
        // Stop webcam
        function stopWebcam() {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
                
                if (frameInterval) {
                    clearInterval(frameInterval);
                    frameInterval = null;
                }
                
                // Update UI
                const startBtn = document.getElementById('startBtn');
                startBtn.innerHTML = '<i class="bi bi-camera-video"></i> <span>Старт</span>';
                startBtn.classList.remove('btn-danger');
                startBtn.classList.add('btn-glass-primary');
                startBtn.onclick = startWebcam;
                
                // Clear canvas
                const canvas = document.getElementById('webcamCanvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Очистка текущих объектов
                document.getElementById('currentObjects').innerHTML = '';
                document.getElementById('detectionCount').textContent = '0';
                
                showToast('Камера остановлена', 'info');
            }
        }
        
        // Start frame processing
        function startFrameProcessing() {
            const video = document.getElementById('webcamVideo');
            const canvas = document.getElementById('webcamCanvas');
            const ctx = canvas.getContext('2d');
            
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            };
            
            // Set processing interval based on settings
            const interval = 1000 / settings.fps;
            if (frameInterval) clearInterval(frameInterval);
            frameInterval = setInterval(() => {
                if (video.readyState === video.HAVE_ENOUGH_DATA && !isProcessing) {
                    processFrame(video, canvas, ctx);
                }
            }, interval);
        }
        
        // Process frame
        async function processFrame(video, canvas, ctx) {
            isProcessing = true;
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            try {
                const response = await fetch('/api/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: imageData,
                        client_id: clientId,
                        settings: settings
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    if (settings.showBoxes && data.annotations) {
                        drawBoundingBoxes(ctx, data.annotations);
                    }
                    updateCurrentObjects(data.annotations || []);
                }
            } catch (error) {
                console.error('Frame processing error:', error);
            } finally {
                isProcessing = false;
            }
        }
        
        // Draw bounding boxes
        function drawBoundingBoxes(ctx, annotations) {
            ctx.lineWidth = settings.boxThickness;
            annotations.forEach(ann => {
                const { x1, y1, x2, y2, label, confidence } = ann;
                
                // Draw box
                ctx.strokeStyle = settings.boxColor;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                
                // Draw label if enabled
                if (settings.showLabels) {
                    ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
                    let text = label;
                    if (confidence) {
                        text += ` ${(confidence * 100).toFixed(0)}%`;
                    }
                    ctx.font = `bold ${settings.fontSize}px Segoe UI`;
                    const textWidth = ctx.measureText(text).width;
                    
                    ctx.fillRect(x1, y1 - (settings.fontSize + 10), textWidth + 10, settings.fontSize + 10);
                    ctx.fillStyle = settings.textColor;
                    ctx.fillText(text, x1 + 5, y1 - 5);
                }
            });
        }
        
        // Update current objects display
        function updateCurrentObjects(annotations) {
            const container = document.getElementById('currentObjects');
            const count = annotations.length;
            
            document.getElementById('detectionCount').textContent = count;
            
            if (count === 0) {
                container.innerHTML = '<span class="text-muted">Объекты не обнаружены</span>';
                return;
            }
            
            const classCounts = {};
            annotations.forEach(ann => {
                classCounts[ann.label] = (classCounts[ann.label] || 0) + 1;
            });
            
            let html = '';
            for (const [label, count] of Object.entries(classCounts)) {
                html += `<span class="object-badge">${label}: ${count}</span>`;
            }
            
            container.innerHTML = html;
        }
        
        // Update statistics
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                // Update stats
                document.getElementById('totalFrames').textContent = data.total_objects.toLocaleString();
                document.getElementById('savedFrames').textContent = data.saved_frames.toLocaleString();
                document.getElementById('fpsDisplay').textContent = data.fps.toFixed(1);
                
                // Update status
                const indicator = document.getElementById('statusIndicator');
                const statusText = document.getElementById('statusText');
                
                if (data.is_paused) {
                    indicator.className = 'status-indicator status-paused';
                    statusText.textContent = 'Пауза';
                    document.getElementById('pauseBtn').innerHTML = '<i class="bi bi-play-circle"></i> <span>Продолжить</span>';
                } else {
                    indicator.className = 'status-indicator status-active';
                    statusText.textContent = 'Активно';
                    document.getElementById('pauseBtn').innerHTML = '<i class="bi bi-pause-circle"></i> <span>Пауза</span>';
                }
                
                // Update detections list
                updateDetectionsList(data.recent_detections || []);
                
                // Update object distribution
                updateObjectDistribution(data.object_counts || {});
                
                // Update charts
                updateCharts(data);
                
            } catch (error) {
                console.error('Stats update error:', error);
            }
        }
        
        // Update charts
        function updateCharts(data) {
            const colors = document.documentElement.getAttribute('data-bs-theme') === 'dark' 
                ? chartColorsDark : chartColorsLight;
            
            // Update objects chart (line)
            if (charts.objects) {
                const timeLabel = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
                charts.objects.data.labels.push(timeLabel);
                charts.objects.data.datasets[0].data.push(data.total_objects || 0);
                
                if (charts.objects.data.labels.length > 15) {
                    charts.objects.data.labels.shift();
                    charts.objects.data.datasets[0].data.shift();
                }
                
                charts.objects.update();
            }
            
            // Update daily activity chart (pie)
            if (charts.daily && data.daily_activity) {
                const dailyData = data.daily_activity;
                const total = Object.values(dailyData).reduce((a, b) => a + b, 0);
                
                if (total > 0) {
                    const labels = Object.keys(dailyData);
                    const percentages = labels.map(label => 
                        Math.round((dailyData[label] / total) * 100)
                    );
                    
                    // Use predefined colors
                    const colorPalette = [
                        colors.primary, colors.secondary, colors.success, 
                        colors.warning, colors.danger, colors.info
                    ];
                    
                    charts.daily.data.labels = labels;
                    charts.daily.data.datasets[0].data = percentages;
                    charts.daily.data.datasets[0].backgroundColor = labels.map((_, i) => 
                        colorPalette[i % colorPalette.length]
                    );
                    
                    charts.daily.update();
                }
            }
            
            // Update weekly distribution chart (bar)
            if (charts.weekly && data.weekly_distribution) {
                const weekDays = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];
                const percentages = weekDays.map(day => 
                    Math.round(data.weekly_distribution[day] || 0)
                );
                
                charts.weekly.data.datasets[0].data = percentages;
                charts.weekly.update();
            }
        }
        
        // Update detections list
        function updateDetectionsList(detections) {
            const list = document.getElementById('detectionsList');
            
            if (detections.length === 0) {
                list.innerHTML = `
                    <div class="empty-state">
                        <i class="bi bi-eye-slash"></i>
                        <p class="mb-0">Нет обнаружений</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            detections.slice(-5).reverse().forEach(detection => {
                const time = new Date(detection.timestamp).toLocaleTimeString('ru-RU');
                html += `
                    <div class="detection-item fade-in">
                        <div>
                            <span class="fw-medium">${detection.label}</span>
                            <span class="pose-badge small">${detection.confidence}%</span>
                            <div class="small text-muted">уверенность</div>
                        </div>
                        <span class="text-muted small">${time}</span>
                    </div>
                `;
            });
            
            list.innerHTML = html;
        }
        
        // Update object distribution
        function updateObjectDistribution(objectCounts) {
            const container = document.getElementById('objectDistribution');
            
            if (Object.keys(objectCounts).length === 0) {
                container.innerHTML = `
                    <div class="empty-state py-3">
                        <p class="mb-0 small text-muted">Нет данных</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            const sorted = Object.entries(objectCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5);
            
            sorted.forEach(([label, count]) => {
                const total = Object.values(objectCounts).reduce((a, b) => a + b, 0);
                const percentage = total > 0 ? (count / total) * 100 : 0;
                
                html += `
                    <div class="mb-3 fade-in">
                        <div class="d-flex justify-content-between mb-1">
                            <span class="small">${label}</span>
                            <span class="small fw-medium">${count}</span>
                        </div>
                        <div class="progress-bar-custom">
                            <div class="progress-custom" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        // Toggle pause
        async function togglePause() {
            try {
                await fetch('/api/toggle_pause', { method: 'POST' });
                updateStats();
            } catch (error) {
                console.error('Pause toggle error:', error);
            }
        }
        
        // Save session
        async function saveSession() {
            try {
                const response = await fetch('/api/save_session', { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    showToast(data.message || 'Данные сохранены!', 'success');
                } else {
                    showToast(data.error || 'Ошибка сохранения', 'error');
                }
            } catch (error) {
                console.error('Save session error:', error);
                showToast('Ошибка соединения с сервером', 'error');
            }
        }
        
        // Download annotations
        function downloadAnnotations() {
            window.open('/api/download_annotations', '_blank');
        }
        
        // Export annotations with custom format
        async function exportAnnotations() {
            try {
                const format = document.getElementById('exportFormat').value;
                
                const response = await fetch('/api/export_annotations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        format: format,
                        include_images: true,
                        include_metadata: true,
                        include_statistics: true
                    })
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `annotations_${new Date().toISOString().slice(0,10)}.${format}`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    showToast('Данные экспортированы', 'success');
                }
            } catch (error) {
                showToast('Ошибка экспорта', 'error');
            }
        }
        
        // Clear annotations
        async function clearAnnotations() {
            if (confirm('Вы уверены, что хотите очистить все данные?')) {
                try {
                    const response = await fetch('/api/clear_annotations', { method: 'POST' });
                    const data = await response.json();
                    if (data.success) {
                        showToast('Данные очищены', 'success');
                        updateStats();
                    }
                } catch (error) {
                    showToast('Ошибка очистки', 'error');
                }
            }
        }
        
        // Reset statistics
        async function resetStatistics() {
            if (confirm('Вы уверены, что хотите сбросить статистику?')) {
                try {
                    const response = await fetch('/api/reset_stats', { method: 'POST' });
                    const data = await response.json();
                    if (data.success) {
                        showToast('Статистика сброшена', 'success');
                        updateStats();
                    }
                } catch (error) {
                    showToast('Ошибка сброса', 'error');
                }
            }
        }
        
        // Take snapshot
        function takeSnapshot() {
            const canvas = document.getElementById('webcamCanvas');
            const link = document.createElement('a');
            link.download = `snapshot_${Date.now()}.png`;
            link.href = canvas.toDataURL();
            link.click();
            showToast('Снимок сохранен', 'success');
        }
        
        // Apply settings
        async function applySettings() {
            try {
                // Basic settings
                settings.confidence = parseFloat(document.getElementById('confidenceSlider').value);
                settings.fps = parseInt(document.getElementById('fpsSlider').value);
                settings.detectionMode = document.getElementById('detectionMode').value;
                settings.autoSave = document.getElementById('autoSave').checked;
                settings.useRussianPoses = document.getElementById('useRussianPoses').checked;
                
                // Visual settings
                settings.boxColor = document.getElementById('boxColor').value;
                settings.textColor = document.getElementById('textColor').value;
                settings.boxThickness = parseInt(document.getElementById('boxThickness').value);
                settings.fontSize = parseInt(document.getElementById('fontSize').value);
                settings.showBoxes = document.getElementById('showBoxes').checked;
                settings.showLabels = document.getElementById('showLabels').checked;
                
                // Update UI values
                document.getElementById('confidenceValue').textContent = settings.confidence.toFixed(2);
                document.getElementById('confidencePercent').textContent = Math.round(settings.confidence * 100) + '%';
                document.getElementById('fpsValue').textContent = settings.fps;
                document.getElementById('boxColorText').textContent = settings.boxColor;
                document.getElementById('textColorText').textContent = settings.textColor;
                document.getElementById('thicknessValue').textContent = settings.boxThickness;
                document.getElementById('fontValue').textContent = settings.fontSize;
                
                // Save to localStorage
                saveSettings();
                
                // Send to server
                await fetch('/api/update_settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });
                
                // Restart frame processing if camera is running
                if (cameraStream) {
                    if (frameInterval) {
                        clearInterval(frameInterval);
                    }
                    startFrameProcessing();
                }
                
                const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
                modal.hide();
                
                showToast('Настройки применены', 'success');
                
            } catch (error) {
                console.error('Settings apply error:', error);
                showToast('Ошибка применения настроек', 'error');
            }
        }
        
        // Show toast notification
        function showToast(message, type = 'info') {
            const toastContainer = document.querySelector('.toast-container');
            const toastId = 'toast-' + Date.now();
            
            const toast = document.createElement('div');
            toast.className = `toast align-items-center text-bg-${type} border-0`;
            toast.id = toastId;
            toast.setAttribute('role', 'alert');
            toast.innerHTML = `
                <div class="d-flex">
                    <div class="toast-body d-flex align-items-center">
                        ${type === 'success' ? '<i class="bi bi-check-circle-fill me-2"></i>' : ''}
                        ${type === 'error' ? '<i class="bi bi-exclamation-circle-fill me-2"></i>' : ''}
                        ${type === 'warning' ? '<i class="bi bi-exclamation-triangle-fill me-2"></i>' : ''}
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            `;
            
            toastContainer.appendChild(toast);
            const bsToast = new bootstrap.Toast(toast, {
                autohide: true,
                delay: 3000
            });
            bsToast.show();
            
            toast.addEventListener('hidden.bs.toast', () => {
                toast.remove();
            });
        }
        
        // Toggle theme
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-bs-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-bs-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Update button icon
            const btn = document.getElementById('themeToggle');
            btn.innerHTML = newTheme === 'dark' ? 
                '<i class="bi bi-moon-stars"></i>' : 
                '<i class="bi bi-sun"></i>';
            
            // Update chart colors
            updateChartColors();
            
            // Update glass effect
            updateGlassEffects();
        }
        
        // Update glass effects based on theme
        function updateGlassEffects() {
            const theme = document.documentElement.getAttribute('data-bs-theme');
            const glassElements = document.querySelectorAll('.glass');
            
            glassElements.forEach(el => {
                if (theme === 'dark') {
                    el.classList.remove('glass-light');
                    el.classList.add('glass-dark');
                } else {
                    el.classList.remove('glass-dark');
                    el.classList.add('glass-light');
                }
            });
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Load saved theme
            const savedTheme = localStorage.getItem('theme') || 'dark';
            document.documentElement.setAttribute('data-bs-theme', savedTheme);
            
            // Update theme button
            const themeBtn = document.getElementById('themeToggle');
            themeBtn.innerHTML = savedTheme === 'dark' ? 
                '<i class="bi bi-moon-stars"></i>' : 
                '<i class="bi bi-sun"></i>';
            
            // Load settings
            loadSettings();
            
            // Initialize charts
            initCharts();
            
            // Update glass effects
            updateGlassEffects();
            
            // Initialize tooltips
            const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
            tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));
            
            // Update stats every second
            setInterval(updateStats, 1000);
            
            // Auto-save if enabled
            if (settings.autoSave) {
                setInterval(saveSession, 300000); // 5 minutes
            }
            
            // Color picker events
            document.getElementById('boxColor').addEventListener('input', (e) => {
                document.getElementById('boxColorText').textContent = e.target.value;
            });
            
            document.getElementById('textColor').addEventListener('input', (e) => {
                document.getElementById('textColorText').textContent = e.target.value;
            });
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {
                if (cameraStream) {
                    cameraStream.getTracks().forEach(track => track.stop());
                }
                if (frameInterval) {
                    clearInterval(frameInterval);
                }
            });
        });
    </script>
</body>
</html>
"""
        
        @app.route('/')
        def index():
            return render_template_string(HTML_PAGE)
        
        @app.route('/api/process_frame', methods=['POST'])
        def process_frame():
            """Обработка кадра от клиента"""
            try:
                data = request.json
                image_data = data['image']
                client_id = data.get('client_id', 'unknown')
                client_settings = data.get('settings', {})
                
                # Обновление настроек из клиента
                if 'useRussianPoses' in client_settings:
                    self.settings['use_russian_poses'] = client_settings['useRussianPoses']
                
                # Обновление информации о клиенте
                self.clients[client_id] = {
                    'last_activity': time.time(),
                    'frame_count': self.clients.get(client_id, {}).get('frame_count', 0) + 1
                }
                
                # Очистка неактивных клиентов
                current_time = time.time()
                inactive = [cid for cid, client in self.clients.items() 
                           if current_time - client['last_activity'] > 30]
                for cid in inactive:
                    del self.clients[cid]
                
                # Декодирование изображения
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return jsonify({'success': False, 'error': 'Не удалось декодировать изображение'})
                
                # Обновление статистики
                self.stats['total_frames'] += 1
                self.stats['fps'] = len(self.clients) * 10
                self.stats['active_clients'] = len(self.clients)
                
                # Сохранение кадра
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Детекция объектов
                annotations = []
                if not self.pause_annotation:
                    confidence = client_settings.get('confidence', 0.5)
                    detection_mode = client_settings.get('detection_mode', 'balanced')
                    
                    # Настройки модели в зависимости от режима
                    model_args = {
                        'verbose': False,
                        'conf': confidence
                    }
                    
                    if detection_mode == 'fast':
                        model_args['half'] = False
                        model_args['device'] = 'cpu'
                    elif detection_mode == 'accurate':
                        model_args['iou'] = 0.3
                        model_args['agnostic_nms'] = True
                    
                    results = self.model(frame, **model_args)
                    result = results[0]
                    
                    if result.boxes is not None:
                        boxes = result.boxes.cpu().numpy()
                        
                        current_objects = OrderedDict()
                        for i in range(len(boxes)):
                            box = boxes[i]
                            conf = box.conf[0]
                            
                            if conf > confidence:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cls_id = int(box.cls[0])
                                label = self.model.names[cls_id]
                                
                                # Переводим позу в русское прилагательное если включено
                                display_label = self.translate_pose(label) if self.settings.get('use_russian_poses', True) else label
                                
                                obj_id = f"{label}_{i}_{self.stats['total_frames']}"
                                
                                current_objects[obj_id] = {
                                    'label': label,
                                    'display_label': display_label,
                                    'class_id': cls_id,
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2,
                                    'confidence': float(conf)
                                }
                                
                                # Для возврата клиенту (используем отображаемую метку)
                                annotations.append({
                                    'label': display_label,
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2,
                                    'confidence': float(conf)
                                })
                                
                                # Обновление статистики (используем оригинальную метку для подсчета)
                                self.stats['object_counts'][label] = self.stats['object_counts'].get(label, 0) + 1
                        
                        # Проверка на сохранение
                        if self.has_significant_changes(current_objects):
                            self.stats['saved_frames'] += 1
                            
                            frame_annotation = {
                                'frame_number': self.stats['total_frames'],
                                'saved_index': self.stats['saved_frames'],
                                'timestamp': datetime.now().isoformat(),
                                'objects': current_objects,
                                'client_id': client_id,
                                'settings': client_settings
                            }
                            
                            self.annotations[f"frame_{self.stats['saved_frames']}"] = frame_annotation
                            self.prev_objects = current_objects.copy()
                            
                            # Обновление истории
                            self.stats['detection_history'].append({
                                'timestamp': datetime.now().isoformat(),
                                'object_count': len(current_objects)
                            })
                            
                            if len(self.stats['detection_history']) > 100:
                                self.stats['detection_history'] = self.stats['detection_history'][-100:]
                            
                            # Обновление активности за день
                            today = datetime.now().date().isoformat()
                            for obj in current_objects.values():
                                self.stats['daily_activity'][today][obj['display_label']] = \
                                    self.stats['daily_activity'][today].get(obj['display_label'], 0) + 1
                
                # Обновление общего количества объектов
                self.stats['total_objects'] = sum(len(frame['objects']) 
                                                 for frame in self.annotations.values())
                
                return jsonify({
                    'success': True,
                    'annotations': annotations,
                    'frame_number': self.stats['total_frames']
                })
                
            except Exception as e:
                logger.error(f"Ошибка обработки кадра: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/stats')
        def get_stats():
            """Получение статистики"""
            recent_detections = self.get_recent_detections(10)
            
            # Вычисление ежедневной активности в процентах
            daily_activity_percent = {}
            today = datetime.now().date().isoformat()
            if today in self.stats['daily_activity']:
                today_data = self.stats['daily_activity'][today]
                total_today = sum(today_data.values())
                if total_today > 0:
                    for label, count in today_data.items():
                        daily_activity_percent[label] = round((count / total_today) * 100, 1)
            
            # Вычисление недельного распределения
            weekly_distribution = self.calculate_weekly_distribution()
            
            stats_data = {
                'total_frames': self.stats['total_frames'],
                'saved_frames': self.stats['saved_frames'],
                'total_objects': self.stats['total_objects'],
                'fps': self.stats['fps'],
                'object_counts': self.stats['object_counts'],
                'recent_detections': recent_detections,
                'detection_history': self.stats['detection_history'][-20:],
                'is_paused': self.pause_annotation,
                'active_clients': len(self.clients),
                'daily_activity': daily_activity_percent,
                'weekly_distribution': weekly_distribution,
                'settings': self.settings
            }
            return jsonify(stats_data)
        
        @app.route('/api/take_snapshot', methods=['POST'])
        def take_snapshot():
            """Создание скриншота на сервере"""
            try:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"snapshot_{timestamp}.jpg"
                        filepath = self.screenshots_dir / filename
                        
                        cv2.imwrite(str(filepath), self.latest_frame)
                        return jsonify({'success': True, 'filename': filename})
                return jsonify({'success': False, 'error': 'Нет доступных кадров'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/download_annotations')
        def download_annotations():
            """Скачивание аннотаций"""
            if self.annotations:
                annotations_data = self.prepare_annotations_data()
                return Response(
                    json.dumps(annotations_data, indent=2, ensure_ascii=False),
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=annotations.json'}
                )
            return jsonify({'error': 'Нет аннотаций'}), 404
        
        @app.route('/api/export_annotations', methods=['POST'])
        def export_annotations():
            """Экспорт аннотаций в разных форматах"""
            try:
                data = request.json
                format = data.get('format', 'json')
                
                if not self.annotations:
                    return jsonify({'error': 'Нет аннотаций'}), 404
                
                annotations_data = self.prepare_annotations_data(
                    include_images=True,
                    include_metadata=True,
                    include_statistics=True
                )
                
                if format == 'json':
                    content = json.dumps(annotations_data, indent=2, ensure_ascii=False)
                    mimetype = 'application/json'
                    ext = 'json'
                elif format == 'csv':
                    # Преобразование в CSV
                    import csv
                    import io
                    
                    output = io.StringIO()
                    writer = csv.writer(output)
                    
                    # Заголовок
                    writer.writerow(['Кадр', 'Объект', 'X1', 'Y1', 'X2', 'Y2', 'Уверенность', 'Время'])
                    
                    # Данные
                    for frame_id, frame in annotations_data.get('frames', {}).items():
                        for obj_id, obj in frame.get('objects', {}).items():
                            writer.writerow([
                                frame_id,
                                obj.get('display_label', obj['label']),
                                obj['x1'],
                                obj['y1'],
                                obj['x2'],
                                obj['y2'],
                                f"{obj['confidence']:.2f}",
                                frame.get('timestamp', '')
                            ])
                    
                    content = output.getvalue()
                    mimetype = 'text/csv'
                    ext = 'csv'
                elif format == 'xml':
                    # Преобразование в XML
                    import xml.etree.ElementTree as ET
                    
                    root = ET.Element('annotations')
                    
                    metadata = ET.SubElement(root, 'metadata')
                    ET.SubElement(metadata, 'export_date').text = annotations_data.get('metadata', {}).get('export_date', '')
                    ET.SubElement(metadata, 'total_frames').text = str(len(self.annotations))
                    
                    frames_elem = ET.SubElement(root, 'frames')
                    for frame_id, frame in annotations_data.get('frames', {}).items():
                        frame_elem = ET.SubElement(frames_elem, 'frame', id=frame_id)
                        ET.SubElement(frame_elem, 'timestamp').text = frame.get('timestamp', '')
                        
                        objects_elem = ET.SubElement(frame_elem, 'objects')
                        for obj_id, obj in frame.get('objects', {}).items():
                            obj_elem = ET.SubElement(objects_elem, 'object', id=obj_id)
                            ET.SubElement(obj_elem, 'label').text = obj.get('display_label', obj['label'])
                            ET.SubElement(obj_elem, 'x1').text = str(obj.get('x1', 0))
                            ET.SubElement(obj_elem, 'y1').text = str(obj.get('y1', 0))
                            ET.SubElement(obj_elem, 'x2').text = str(obj.get('x2', 0))
                            ET.SubElement(obj_elem, 'y2').text = str(obj.get('y2', 0))
                            ET.SubElement(obj_elem, 'confidence').text = str(obj.get('confidence', 0))
                    
                    content = ET.tostring(root, encoding='unicode', method='xml')
                    mimetype = 'application/xml'
                    ext = 'xml'
                else:
                    return jsonify({'error': 'Формат не поддерживается'}), 400
                
                return Response(
                    content,
                    mimetype=mimetype,
                    headers={'Content-Disposition': f'attachment; filename=annotations.{ext}'}
                )
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/save_session', methods=['POST'])
        def save_session():
            """Сохранение сессии"""
            try:
                success = self._save_to_json()
                if success:
                    return jsonify({'message': f'Сохранено {len(self.annotations)} кадров', 'success': True})
                return jsonify({'error': 'Ошибка сохранения', 'success': False}), 500
            except Exception as e:
                logger.error(f"Ошибка сохранения сессии: {e}")
                return jsonify({'error': str(e), 'success': False}), 500
        
        @app.route('/api/toggle_pause', methods=['POST'])
        def toggle_pause():
            """Переключение паузы"""
            self.pause_annotation = not self.pause_annotation
            return jsonify({'paused': self.pause_annotation})
        
        @app.route('/api/update_settings', methods=['POST'])
        def update_settings():
            """Обновление настроек"""
            try:
                data = request.json
                # Обновляем только нужные настройки
                for key in ['confidence', 'show_boxes', 'show_labels', 
                           'box_color', 'text_color', 'box_thickness', 'font_size', 'detection_mode', 'useRussianPoses']:
                    if key in data:
                        # Конвертируем camelCase в snake_case для сервера
                        server_key = key
                        if key == 'useRussianPoses':
                            server_key = 'use_russian_poses'
                        self.settings[server_key] = data[key]
                
                return jsonify({'message': 'Настройки обновлены', 'settings': self.settings})
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @app.route('/api/clear_annotations', methods=['POST'])
        def clear_annotations():
            """Очистка аннотаций"""
            try:
                self.annotations.clear()
                self.prev_objects = None
                return jsonify({'success': True, 'message': 'Данные очищены'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/reset_stats', methods=['POST'])
        def reset_stats():
            """Сброс статистики"""
            try:
                self.stats = {
                    'total_frames': 0,
                    'saved_frames': 0,
                    'total_objects': 0,
                    'fps': 0,
                    'start_time': time.time(),
                    'object_counts': {},
                    'detection_history': [],
                    'active_clients': len(self.clients),
                    'daily_activity': defaultdict(lambda: defaultdict(int)),
                    'weekly_distribution': defaultdict(lambda: defaultdict(float))
                }
                return jsonify({'success': True, 'message': 'Статистика сброшена'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        logger.info(f"🌐 Сервер запущен: http://localhost:{self.flask_port}")
        logger.info("   Откройте этот адрес в браузере для использования системы")
        
        app.run(host='0.0.0.0', port=self.flask_port, debug=False, threaded=True, use_reloader=False)
    
    def get_recent_detections(self, count=10):
        """Получение последних обнаружений"""
        recent = []
        frames = list(self.annotations.values())[-10:]
        
        for frame in frames:
            for obj in frame['objects'].values():
                # Используем отображаемую метку если есть
                display_label = obj.get('display_label', obj['label'])
                recent.append({
                    'label': display_label,
                    'confidence': round(obj['confidence'] * 100, 1),
                    'timestamp': frame['timestamp']
                })
                if len(recent) >= count:
                    return recent
        return recent
    
    def calculate_iou(self, box1, box2):
        """Вычисление IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def has_significant_changes(self, current_objects):
        """Проверка на значительные изменения"""
        if self.pause_annotation or self.prev_objects is None:
            return True
        
        if len(current_objects) != len(self.prev_objects):
            return True
        
        # Проверка классов
        current_labels = set(obj['label'] for obj in current_objects.values())
        prev_labels = set(obj['label'] for obj in self.prev_objects.values())
        if current_labels != prev_labels:
            return True
        
        # Проверка положения
        for obj_id, curr_obj in current_objects.items():
            if obj_id in self.prev_objects:
                prev_obj = self.prev_objects[obj_id]
                curr_box = (curr_obj['x1'], curr_obj['y1'], curr_obj['x2'], curr_obj['y2'])
                prev_box = (prev_obj['x1'], prev_obj['y1'], prev_obj['x2'], prev_obj['y2'])
                
                iou = self.calculate_iou(curr_box, prev_box)
                if iou < self.iou_threshold:
                    return True
                
                # Расстояние между центрами
                curr_center = ((curr_obj['x1'] + curr_obj['x2']) // 2, 
                              (curr_obj['y1'] + curr_obj['y2']) // 2)
                prev_center = ((prev_obj['x1'] + prev_obj['x2']) // 2, 
                              (prev_obj['y1'] + prev_obj['y2']) // 2)
                
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                
                if distance > self.position_threshold:
                    return True
        
        return False
    
    def prepare_annotations_data(self, include_images=True, include_metadata=True, include_statistics=True):
        """Подготовка данных для экспорта"""
        data = {}
        
        if include_metadata:
            data['metadata'] = {
                'project': 'Vision AI - Анализатор поз',
                'export_date': datetime.now().isoformat(),
                'total_frames': len(self.annotations),
                'total_objects': self.stats['total_objects'],
                'model': str(self.model),
                'settings': self.settings,
                'pose_translation_enabled': self.settings.get('use_russian_poses', True)
            }
        
        if include_statistics:
            data['statistics'] = {
                'total_frames': self.stats['total_frames'],
                'saved_frames': self.stats['saved_frames'],
                'total_objects': self.stats['total_objects'],
                'object_counts': dict(self.stats['object_counts']),
                'daily_activity': dict(self.stats['daily_activity']),
                'weekly_distribution': dict(self.stats['weekly_distribution'])
            }
        
        frames_data = {}
        for frame_id, frame in self.annotations.items():
            frame_copy = frame.copy()
            if not include_images:
                frame_copy.pop('image_data', None)
            frames_data[frame_id] = frame_copy
        
        data['frames'] = frames_data
        
        return data
    
    def run(self):
        """Основной цикл"""
        logger.info("🚀 Vision AI - Анализатор поз запущен")
        logger.info("   Откройте браузер по указанному адресу")
        logger.info("   Для выхода нажмите Ctrl+C")
        
        try:
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Завершение работы...")
        
        finally:
            self.cleanup()
            logger.info("Сервер остановлен")
    
    def _save_to_json(self, final=False):
        """Сохранение в JSON"""
        try:
            if self.annotations:
                data = self.prepare_annotations_data()
                filename = 'autosave_annotations.json' if not final else self.output_file
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"Сохранено {len(self.annotations)} кадров в {filename}")
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
        return False

def main():
    """Точка входа"""
    print("\n" + "="*60)
    print("🚀 VISION AI - АНАЛИЗАТОР ПОЗ")
    print("="*60)
    print("Серверная система распознавания поз с переводом на русский язык")
    print("\n✨ Основные функции:")
    print("  • Обнаружение поз в реальном времени")
    print("  • Автоматический перевод поз в прилагательные")
    print("  • Статистика активности за день и неделю")
    print("  • Современный дизайн с графиками")
    print("  • Настройка параметров обнаружения")
    print("  • Экспорт данных в разных форматах")
    print("\n🎮 Инструкция:")
    print("  1. Откройте браузер (Chrome/Firefox/Edge)")
    print("  2. Перейдите по адресу который появится после запуска")
    print("  3. Нажмите 'Старт' для начала работы с камерой")
    print("  4. Обнаруженные позы будут отображаться как прилагательные:")
    print("     стоящий, сидящий, лежащий, падающий и т.д.")
    print("  5. Используйте кнопку настроек (правый нижний угол)")
    print("  6. В настройках можно отключить перевод на русский")
    print("="*60)
    
    try:
        port = 5050
        annotator = WebRTCYOLOAnnotator(flask_port=port)
        annotator.run()
    except ValueError:
        print("Ошибка: порт должен быть числом")
    except Exception as e:
        print(f"Ошибка запуска: {e}")
    finally:
        print("\n👋 До свидания!")

if __name__ == "__main__":
    main()
