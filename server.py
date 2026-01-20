import cv2
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from collections import OrderedDict
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
            'active_clients': 0
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
        
        # Запуск Flask
        self.flask_thread = threading.Thread(target=self.start_flask_server)
        self.flask_thread.daemon = True
        self.flask_thread.start()
        
        # Обработка сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Сервер запущен на порту: {flask_port}")
    
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
    
    def start_flask_server(self):
        """Запуск Flask сервера"""
        app = Flask(__name__)
        
        # Минималистичный HTML интерфейс
        HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision AI Annotator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 1.8rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .video-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .video-container {
            position: relative;
            width: 100%;
            background: black;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        
        .video-container video,
        .video-container canvas {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .video-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9rem;
            display: flex;
            justify-content: space-between;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 12px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .btn-primary {
            background: #3498db;
            color: white;
        }
        
        .btn-success {
            background: #2ecc71;
            color: white;
        }
        
        .btn-warning {
            background: #f39c12;
            color: white;
        }
        
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        
        .btn-info {
            background: #1abc9c;
            color: white;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .stat-card h3 {
            font-size: 1.5rem;
            margin: 10px 0;
            color: #2c3e50;
        }
        
        .stat-card p {
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .card h3 {
            margin-bottom: 15px;
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        
        .detection-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .detection-item {
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
            display: flex;
            justify-content: space-between;
        }
        
        .detection-item:last-child {
            border-bottom: none;
        }
        
        .object-badge {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin: 2px;
        }
        
        .object-distribution {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .object-item {
            flex: 1;
            min-width: 120px;
            background: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        
        .object-item span {
            display: block;
            font-weight: bold;
            font-size: 1.2rem;
            color: #2c3e50;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-active {
            background: #2ecc71;
            box-shadow: 0 0 10px #2ecc71;
        }
        
        .status-paused {
            background: #e74c3c;
            box-shadow: 0 0 10px #e74c3c;
        }
        
        .camera-selection {
            margin-bottom: 20px;
        }
        
        .camera-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        
        .camera-btn {
            flex: 1;
            min-width: 150px;
            background: #ecf0f1;
            border: 2px solid transparent;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .camera-btn:hover {
            background: #bdc3c7;
        }
        
        .camera-btn.active {
            background: #3498db;
            color: white;
            border-color: #2980b9;
        }
        
        .progress-bar {
            height: 6px;
            background: #ecf0f1;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .progress {
            height: 100%;
            background: #3498db;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Заголовок -->
        <div class="header">
            <h1>🚀 Vision AI Annotator</h1>
            <div>
                <span class="status-indicator" id="statusIndicator"></span>
                <span id="statusText">Активно</span>
            </div>
        </div>
        
        <div class="main-content">
            <!-- Левая колонка -->
            <div class="video-section">
                <!-- Выбор камеры -->
                <div class="camera-selection">
                    <h3>📷 Выбор камеры</h3>
                    <div class="camera-list" id="cameraList">
                        <button class="camera-btn active" onclick="selectCamera('default')">
                            Камера по умолчанию
                        </button>
                    </div>
                </div>
                
                <!-- Видео поток -->
                <h3>🎥 Видео поток</h3>
                <div class="video-container">
                    <video id="webcamVideo" autoplay playsinline></video>
                    <canvas id="webcamCanvas"></canvas>
                    <div class="video-overlay">
                        <div id="videoStats">
                            FPS: <span id="fpsDisplay">0</span> | 
                            Кадров: <span id="frameCount">0</span> | 
                            Объектов: <span id="objectCount">0</span>
                        </div>
                        <div id="currentObjects"></div>
                    </div>
                </div>
                
                <!-- Управление -->
                <div class="controls">
                    <button class="btn btn-success" onclick="startWebcam()" id="startBtn">
                        ▶ Запустить камеру
                    </button>
                    <button class="btn btn-warning" onclick="togglePause()" id="pauseBtn">
                        ⏸ Пауза
                    </button>
                    <button class="btn btn-info" onclick="takeSnapshot()">
                        📷 Скриншот
                    </button>
                    <button class="btn btn-primary" onclick="saveSession()">
                        💾 Сохранить
                    </button>
                    <button class="btn btn-primary" onclick="downloadAnnotations()">
                        📥 Экспорт
                    </button>
                    <button class="btn" onclick="showSettings()" style="background: #9b59b6; color: white;">
        ⚙ Настройки
                    </button>
                </div>
                
                <!-- Статистика -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <p>Кадров обработано</p>
                        <h3 id="totalFrames">0</h3>
                    </div>
                    <div class="stat-card">
                        <p>Кадров сохранено</p>
                        <h3 id="savedFrames">0</h3>
                    </div>
                    <div class="stat-card">
                        <p>Всего объектов</p>
                        <h3 id="totalObjects">0</h3>
                    </div>
                    <div class="stat-card">
                        <p>Активные клиенты</p>
                        <h3 id="activeClients">0</h3>
                    </div>
                </div>
            </div>
            
            <!-- Правая колонка -->
            <div class="sidebar">
                <!-- Последние обнаружения -->
                <div class="card">
                    <h3>🎯 Последние обнаружения</h3>
                    <div class="detection-list" id="detectionsList">
                        <div class="detection-item">
                            <span>Нет обнаружений</span>
                            <span>--:--:--</span>
                        </div>
                    </div>
                </div>
                
                <!-- Распределение объектов -->
                <div class="card">
                    <h3>📊 Распределение объектов</h3>
                    <div class="object-distribution" id="objectDistribution">
                        <div class="object-item">
                            <span>0</span>
                            <small>Нет данных</small>
                        </div>
                    </div>
                </div>
                
                <!-- Настройки -->
                <div class="card" id="settingsPanel" style="display: none;">
                    <h3>⚙ Настройки</h3>
                    <div style="margin-bottom: 15px;">
                        <label>Порог уверенности: <span id="confidenceValue">0.5</span></label>
                        <input type="range" id="confidenceSlider" min="0.1" max="0.9" step="0.1" value="0.5" 
                               oninput="updateConfidence(this.value)" style="width: 100%;">
                    </div>
                    <div style="margin-bottom: 15px;">
                        <label>
                            <input type="checkbox" id="showBoxes" checked onchange="toggleBoxes()">
                            Показывать рамки
                        </label>
                    </div>
                    <div>
                        <button class="btn btn-primary" onclick="applySettings()" style="width: 100%;">
                            Применить настройки
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Глобальные переменные
        let cameraStream = null;
        let selectedCamera = 'default';
        let isProcessing = false;
        let clientId = null;
        let frameInterval = null;
        let settings = {
            confidence: 0.5,
            showBoxes: true
        };
        
        // Генерация ID клиента
        function generateClientId() {
            return 'client_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
        
        // Загрузка доступных камер
        async function loadCameras() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                const cameraList = document.getElementById('cameraList');
                
                cameraList.innerHTML = '';
                
                videoDevices.forEach((device, index) => {
                    const btn = document.createElement('button');
                    btn.className = 'camera-btn';
                    btn.textContent = device.label || `Камера ${index + 1}`;
                    btn.onclick = () => selectCamera(device.deviceId, btn);
                    
                    if (index === 0) {
                        btn.classList.add('active');
                        selectedCamera = device.deviceId;
                    }
                    
                    cameraList.appendChild(btn);
                });
                
                if (videoDevices.length === 0) {
                    cameraList.innerHTML = '<p style="color: #e74c3c;">Камеры не найдены</p>';
                }
            } catch (error) {
                console.error('Ошибка загрузки камер:', error);
                document.getElementById('cameraList').innerHTML = 
                    '<p style="color: #e74c3c;">Ошибка доступа к камерам</p>';
            }
        }
        
        // Выбор камеры
        function selectCamera(deviceId, element) {
            // Сброс активного класса
            document.querySelectorAll('.camera-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Установка активного класса
            if (element) {
                element.classList.add('active');
            }
            
            selectedCamera = deviceId;
            
            // Перезапуск камеры если она уже запущена
            if (cameraStream) {
                startWebcam();
            }
        }
        
        // Запуск веб-камеры
        async function startWebcam() {
            try {
                // Остановка предыдущего потока
                if (cameraStream) {
                    cameraStream.getTracks().forEach(track => track.stop());
                }
                
                // Настройки захвата
                const constraints = {
                    video: {
                        deviceId: selectedCamera !== 'default' ? { exact: selectedCamera } : undefined,
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        frameRate: { ideal: 30 }
                    },
                    audio: false
                };
                
                // Получение потока
                cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
                const video = document.getElementById('webcamVideo');
                video.srcObject = cameraStream;
                
                // Обновление кнопки
                document.getElementById('startBtn').innerHTML = '⏹ Остановить';
                document.getElementById('startBtn').className = 'btn btn-danger';
                document.getElementById('startBtn').onclick = stopWebcam;
                
                // Генерация ID клиента
                clientId = generateClientId();
                
                // Запуск обработки кадров
                startFrameProcessing();
                
            } catch (error) {
                console.error('Ошибка доступа к камере:', error);
                alert(`Ошибка доступа к камере: ${error.message}`);
            }
        }
        
        // Остановка веб-камеры
        function stopWebcam() {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
                
                if (frameInterval) {
                    clearInterval(frameInterval);
                    frameInterval = null;
                }
                
                // Обновление кнопки
                document.getElementById('startBtn').innerHTML = '▶ Запустить камеру';
                document.getElementById('startBtn').className = 'btn btn-success';
                document.getElementById('startBtn').onclick = startWebcam;
                
                // Очистка canvas
                const canvas = document.getElementById('webcamCanvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }
        
        // Запуск обработки кадров
        function startFrameProcessing() {
            const video = document.getElementById('webcamVideo');
            const canvas = document.getElementById('webcamCanvas');
            const ctx = canvas.getContext('2d');
            
            // Установка размеров canvas
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            };
            
            // Интервал обработки кадров (10 FPS)
            frameInterval = setInterval(() => {
                if (video.readyState === video.HAVE_ENOUGH_DATA && !isProcessing) {
                    processFrame(video, canvas, ctx);
                }
            }, 100);
        }
        
        // Обработка кадра
        async function processFrame(video, canvas, ctx) {
            isProcessing = true;
            
            // Рисуем кадр на canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Получаем изображение в base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            try {
                // Отправляем на сервер
                const response = await fetch('/api/process_frame', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageData,
                        client_id: clientId,
                        settings: settings
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Отображение рамок если нужно
                    if (settings.showBoxes && data.annotations) {
                        drawBoundingBoxes(ctx, data.annotations);
                    }
                    
                    // Обновление текущих объектов
                    updateCurrentObjects(data.annotations || []);
                }
            } catch (error) {
                console.error('Ошибка обработки кадра:', error);
            } finally {
                isProcessing = false;
            }
        }
        
        // Отрисовка рамок
        function drawBoundingBoxes(ctx, annotations) {
            annotations.forEach(ann => {
                const { x1, y1, x2, y2, label, confidence } = ann;
                
                // Рисуем прямоугольник
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                
                // Рисуем подпись
                ctx.fillStyle = '#00FF00';
                ctx.font = '14px Arial';
                const text = `${label} ${(confidence * 100).toFixed(1)}%`;
                
                // Фон для текста
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                const textWidth = ctx.measureText(text).width;
                ctx.fillRect(x1, y1 - 20, textWidth + 10, 20);
                
                // Текст
                ctx.fillStyle = '#00FF00';
                ctx.fillText(text, x1 + 5, y1 - 5);
            });
        }
        
        // Обновление текущих объектов
        function updateCurrentObjects(annotations) {
            const currentObjectsDiv = document.getElementById('currentObjects');
            const objectCount = annotations.length;
            
            document.getElementById('objectCount').textContent = objectCount;
            
            if (objectCount === 0) {
                currentObjectsDiv.innerHTML = '<span style="color: #95a5a6;">Нет объектов</span>';
                return;
            }
            
            // Группировка по классам
            const classCounts = {};
            annotations.forEach(ann => {
                classCounts[ann.label] = (classCounts[ann.label] || 0) + 1;
            });
            
            let html = '';
            for (const [label, count] of Object.entries(classCounts)) {
                html += `<span class="object-badge">${label}: ${count}</span>`;
            }
            
            currentObjectsDiv.innerHTML = html;
        }
        
        // Обновление статистики
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                // Обновление статистики
                document.getElementById('totalFrames').textContent = data.total_frames;
                document.getElementById('savedFrames').textContent = data.saved_frames;
                document.getElementById('totalObjects').textContent = data.total_objects;
                document.getElementById('activeClients').textContent = data.active_clients || 0;
                document.getElementById('fpsDisplay').textContent = data.fps.toFixed(1);
                document.getElementById('frameCount').textContent = data.total_frames;
                
                // Обновление статуса
                const indicator = document.getElementById('statusIndicator');
                const statusText = document.getElementById('statusText');
                
                if (data.is_paused) {
                    indicator.className = 'status-indicator status-paused';
                    statusText.textContent = 'Пауза';
                    document.getElementById('pauseBtn').innerHTML = '▶ Возобновить';
                } else {
                    indicator.className = 'status-indicator status-active';
                    statusText.textContent = 'Активно';
                    document.getElementById('pauseBtn').innerHTML = '⏸ Пауза';
                }
                
                // Обновление последних обнаружений
                updateDetectionsList(data.recent_detections || []);
                
                // Обновление распределения объектов
                updateObjectDistribution(data.object_counts || {});
                
            } catch (error) {
                console.error('Ошибка обновления статистики:', error);
            }
        }
        
        // Обновление списка обнаружений
        function updateDetectionsList(detections) {
            const list = document.getElementById('detectionsList');
            
            if (detections.length === 0) {
                list.innerHTML = '<div class="detection-item"><span>Нет обнаружений</span><span>--:--:--</span></div>';
                return;
            }
            
            let html = '';
            detections.slice(-8).reverse().forEach(detection => {
                const time = detection.timestamp.split('T')[1].split('.')[0];
                html += `
                    <div class="detection-item">
                        <span>${detection.label} (${detection.confidence}%)</span>
                        <span>${time}</span>
                    </div>
                `;
            });
            
            list.innerHTML = html;
        }
        
        // Обновление распределения объектов
        function updateObjectDistribution(objectCounts) {
            const container = document.getElementById('objectDistribution');
            
            if (Object.keys(objectCounts).length === 0) {
                container.innerHTML = '<div class="object-item"><span>0</span><small>Нет данных</small></div>';
                return;
            }
            
            let html = '';
            const sorted = Object.entries(objectCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 6);
            
            sorted.forEach(([label, count]) => {
                const total = Object.values(objectCounts).reduce((a, b) => a + b, 0);
                const percentage = total > 0 ? (count / total) * 100 : 0;
                
                html += `
                    <div class="object-item">
                        <span>${count}</span>
                        <small>${label}</small>
                        <div class="progress-bar">
                            <div class="progress" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        // Пауза/возобновление
        async function togglePause() {
            try {
                await fetch('/api/toggle_pause', { method: 'POST' });
                updateStats();
            } catch (error) {
                console.error('Ошибка переключения паузы:', error);
            }
        }
        
        // Сохранение сессии
        async function saveSession() {
            try {
                const response = await fetch('/api/save_session', { method: 'POST' });
                const data = await response.json();
                alert(data.message || 'Сессия сохранена!');
            } catch (error) {
                alert('Ошибка сохранения сессии');
            }
        }
        
        // Экспорт аннотаций
        function downloadAnnotations() {
            window.open('/api/download_annotations', '_blank');
        }
        
        // Скриншот
        function takeSnapshot() {
            const canvas = document.getElementById('webcamCanvas');
            const link = document.createElement('a');
            link.download = `snapshot_${Date.now()}.png`;
            link.href = canvas.toDataURL();
            link.click();
        }
        
        // Показать настройки
        function showSettings() {
            const panel = document.getElementById('settingsPanel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }
        
        // Обновление порога уверенности
        function updateConfidence(value) {
            settings.confidence = parseFloat(value);
            document.getElementById('confidenceValue').textContent = value;
        }
        
        // Переключение отображения рамок
        function toggleBoxes() {
            settings.showBoxes = document.getElementById('showBoxes').checked;
        }
        
        // Применение настроек
        async function applySettings() {
            try {
                await fetch('/api/update_settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });
                alert('Настройки применены!');
                document.getElementById('settingsPanel').style.display = 'none';
            } catch (error) {
                alert('Ошибка применения настроек');
            }
        }
        
        // Инициализация
        document.addEventListener('DOMContentLoaded', () => {
            // Загрузка камер
            loadCameras();
            
            // Обновление статистики каждую секунду
            setInterval(updateStats, 1000);
            
            // Первоначальное обновление
            updateStats();
            
            // Очистка при закрытии
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
                
                # Сохранение кадра
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Детекция объектов
                annotations = []
                if not self.pause_annotation:
                    confidence = client_settings.get('confidence', 0.5)
                    
                    results = self.model(frame, verbose=False, conf=confidence)
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
                                
                                obj_id = f"{label}_{i}_{self.stats['total_frames']}"
                                
                                current_objects[obj_id] = {
                                    'label': label,
                                    'class_id': cls_id,
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2,
                                    'confidence': float(conf)
                                }
                                
                                # Для возврата клиенту
                                annotations.append({
                                    'label': label,
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2,
                                    'confidence': float(conf)
                                })
                                
                                # Обновление статистики
                                self.stats['object_counts'][label] = self.stats['object_counts'].get(label, 0) + 1
                        
                        # Проверка на сохранение
                        if self.has_significant_changes(current_objects):
                            self.stats['saved_frames'] += 1
                            
                            frame_annotation = {
                                'frame_number': self.stats['total_frames'],
                                'saved_index': self.stats['saved_frames'],
                                'timestamp': datetime.now().isoformat(),
                                'objects': current_objects,
                                'client_id': client_id
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
            
            stats_data = {
                'total_frames': self.stats['total_frames'],
                'saved_frames': self.stats['saved_frames'],
                'total_objects': self.stats['total_objects'],
                'fps': self.stats['fps'],
                'object_counts': self.stats['object_counts'],
                'recent_detections': recent_detections,
                'detection_history': self.stats['detection_history'][-20:],
                'is_paused': self.pause_annotation,
                'active_clients': len(self.clients)
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
        
        @app.route('/api/save_session', methods=['POST'])
        def save_session():
            """Сохранение сессии"""
            success = self._save_to_json()
            if success:
                return jsonify({'message': f'Сохранено {len(self.annotations)} кадров'})
            return jsonify({'error': 'Ошибка сохранения'}), 500
        
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
                if 'confidence' in data:
                    # Настройки применяются для каждого клиента отдельно
                    pass
                if 'iou_threshold' in data:
                    self.iou_threshold = float(data['iou_threshold'])
                return jsonify({'message': 'Настройки обновлены'})
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        logger.info(f"🌐 Сервер запущен: http://localhost:{self.flask_port}")
        logger.info("   Откройте этот адрес в браузере для использования системы")
        
        app.run(host='0.0.0.0', port=self.flask_port, debug=False, threaded=True, use_reloader=False)
    
    def get_recent_detections(self, count=10):
        """Получение последних обнаружений"""
        recent = []
        frames = list(self.annotations.values())[-10:]
        
        for frame in frames:
            for obj in frame['objects'].values():
                recent.append({
                    'label': obj['label'],
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
    
    def prepare_annotations_data(self):
        """Подготовка данных для экспорта"""
        return {
            'metadata': {
                'project': 'Vision AI Annotator',
                'export_date': datetime.now().isoformat(),
                'total_frames': len(self.annotations),
                'total_objects': self.stats['total_objects']
            },
            'statistics': self.stats,
            'frames': dict(self.annotations)
        }
    
    def run(self):
        """Основной цикл"""
        logger.info("🚀 Vision AI Annotator запущен")
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
    print("🚀 VISION AI ANNOTATOR - WebRTC Version")
    print("="*60)
    print("Серверная система аннотации объектов")
    print("\n✨ Особенности:")
    print("  • Использует веб-камеру пользователя через браузер")
    print("  • Не требует установки камеры на сервере")
    print("  • Современный интерфейс с адаптивным дизайном")
    print("  • Экспорт аннотаций в JSON")
    print("  • Поддержка нескольких клиентов")
    print("\n🎮 Инструкция:")
    print("  1. Откройте браузер (Chrome/Firefox/Edge)")
    print("  2. Перейдите по адресу который появится после запуска")
    print("  3. Разрешите доступ к веб-камере")
    print("  4. Выберите камеру из списка")
    print("  5. Нажмите 'Запустить камеру'")
    print("  6. Обнаруженные объекты будут отображаться в реальном времени")
    print("="*60)
    
    try:
        port = int(input(f"Введите порт [по умолчанию 3000]: ") or "3000")
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