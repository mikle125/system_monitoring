import cv2
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from collections import OrderedDict
import time
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request, send_file
import threading
import queue
import logging
import signal
import sys
import os
from werkzeug.utils import secure_filename

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfessionalYOLOAnnotator:
    def __init__(self, output_file='annotations.json', flask_port=3000):
        """
        Профессиональный аннотатор с современным веб-интерфейсом
        
        Args:
            output_file: путь к выходному JSON файлу
            flask_port: порт для Flask сервера
        """
        self.model = YOLO('best.pt')
        self.output_file = Path(output_file)
        self.annotations = OrderedDict()
        
        # Параметры для оптимизации
        self.prev_objects = None
        self.position_threshold = 50
        self.iou_threshold = 0.3
        
        # Очередь для кадров
        self.frame_queue = queue.Queue(maxsize=30)
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
            'hourly_stats': {}
        }
        
        # Контроль работы
        self.running = True
        self.flask_port = flask_port
        self.pause_annotation = False
        
        # Камера
        self.current_camera_index = 0
        self.available_cameras = self._get_available_cameras()
        
        # Открытие камеры
        self.cap = cv2.VideoCapture(self.current_camera_index)
        if not self.cap.isOpened():
            logger.error(f"Не удалось открыть камеру {self.current_camera_index}")
            raise Exception(f"Не удалось открыть камеру {self.current_camera_index}")
        
        # Настройка камеры для лучшей производительности
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Папка для скриншотов
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Запуск Flask
        self.flask_thread = threading.Thread(target=self.start_flask_server)
        self.flask_thread.daemon = True
        self.flask_thread.start()
        
        # Обработка сигналов для корректного завершения
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Инициализация завершена. Порт: {flask_port}")
        logger.info(f"Доступные камеры: {self.available_cameras}")
    
    def _get_available_cameras(self):
        """Получить список доступных камер"""
        available_cameras = []
        for i in range(5):  # Проверяем первые 5 индексов
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append({
                        'index': i,
                        'name': f'Камера {i}',
                        'resolution': f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}'
                    })
                cap.release()
        return available_cameras
    
    def switch_camera(self, camera_index):
        """Переключение камеры"""
        try:
            # Закрываем текущую камеру
            if hasattr(self, 'cap'):
                self.cap.release()
            
            # Открываем новую камеру
            self.current_camera_index = camera_index
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Не удалось открыть камеру {camera_index}")
                return False
            
            # Настройка параметров
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Сброс предыдущих объектов
            self.prev_objects = None
            
            logger.info(f"Переключено на камеру {camera_index}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при переключении камеры: {e}")
            return False
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        logger.info(f"Получен сигнал {signum}, завершение работы...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Очистка ресурсов"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        self._save_to_json(final=True)
        logger.info("Ресурсы освобождены")
    
    def start_flask_server(self):
        """Запуск Flask сервера с современным интерфейсом"""
        app = Flask(__name__)
        
        # Современный HTML интерфейс с Bootstrap 5
        HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision AI Annotator</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-color: #4361ee;
            --secondary-color: #3a0ca3;
            --success-color: #4cc9f0;
            --light-bg: #f8f9fa;
            --card-shadow: 0 4px 20px rgba(0,0,0,0.08);
            --border-radius: 16px;
        }
        
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: var(--border-radius);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: var(--card-shadow);
        }
        
        .stat-card {
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border: none;
            border-radius: 12px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(67, 97, 238, 0.3);
        }
        
        .video-container {
            position: relative;
            border-radius: var(--border-radius);
            overflow: hidden;
            background: #000;
        }
        
        .video-overlay {
            position: absolute;
            top: 15px;
            left: 15px;
            right: 15px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 0.9rem;
        }
        
        .object-badge {
            display: inline-block;
            padding: 4px 12px;
            margin: 3px;
            background: var(--primary-color);
            color: white;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .camera-thumbnail {
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 3px solid transparent;
        }
        
        .camera-thumbnail:hover {
            transform: scale(1.03);
        }
        
        .camera-thumbnail.active {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.2);
        }
        
        .detection-item {
            padding: 12px;
            border-bottom: 1px solid rgba(0,0,0,0.1);
            transition: background 0.2s;
        }
        
        .detection-item:hover {
            background: rgba(67, 97, 238, 0.05);
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .status-active {
            background: rgba(76, 201, 240, 0.2);
            color: var(--success-color);
        }
        
        .status-paused {
            background: rgba(255, 107, 107, 0.2);
            color: #ff6b6b;
        }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <!-- Заголовок -->
        <div class="row mb-4">
            <div class="col">
                <div class="glass-card p-4">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h1 class="h3 mb-1 fw-bold">🚀 Vision AI Annotator</h1>
                            <p class="text-muted mb-0">Профессиональная система аннотации объектов в реальном времени</p>
                        </div>
                        <div class="d-flex align-items-center gap-3">
                            <div id="statusBadge" class="status-badge status-active">
                                <i class="bi bi-record-circle me-2"></i>
                                <span id="statusText">АКТИВНО</span>
                            </div>
                            <button class="btn btn-primary" onclick="downloadAnnotations()">
                                <i class="bi bi-download me-2"></i>Экспорт
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row g-4">
            <!-- Основное видео и управление -->
            <div class="col-lg-8">
                <div class="glass-card p-4">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h5 class="mb-0 fw-bold">
                            <i class="bi bi-camera-video me-2"></i>Основной поток
                        </h5>
                        <div class="btn-group">
                            <button id="pauseBtn" class="btn btn-outline-primary" onclick="togglePause()">
                                <i class="bi bi-pause-circle me-2"></i>Пауза
                            </button>
                            <button class="btn btn-outline-primary" onclick="takeSnapshot()">
                                <i class="bi bi-camera me-2"></i>Скриншот
                            </button>
                            <button class="btn btn-outline-primary" onclick="saveSession()">
                                <i class="bi bi-save me-2"></i>Сохранить
                            </button>
                        </div>
                    </div>
                    
                    <div class="video-container mb-4">
                        <img id="video" src="/video" class="w-100">
                        <div class="video-overlay d-flex justify-content-between">
                            <div>
                                <span id="fpsDisplay">FPS: 0</span> | 
                                <span id="frameCount">Кадров: 0</span> | 
                                <span id="objectCount">Объектов: 0</span>
                            </div>
                            <div id="currentObjects" class="text-end"></div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-4">
                            <div class="glass-card p-3 text-center stat-card">
                                <h6 class="text-muted mb-2">Кадров обработано</h6>
                                <h3 id="totalFrames" class="fw-bold mb-0">0</h3>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="glass-card p-3 text-center stat-card">
                                <h6 class="text-muted mb-2">Кадров сохранено</h6>
                                <h3 id="savedFrames" class="fw-bold mb-0">0</h3>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="glass-card p-3 text-center stat-card">
                                <h6 class="text-muted mb-2">Всего объектов</h6>
                                <h3 id="totalObjects" class="fw-bold mb-0">0</h3>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Выбор камеры и обнаруженные объекты -->
            <div class="col-lg-4">
                <!-- Выбор камеры -->
                <div class="glass-card p-4 mb-4">
                    <h5 class="fw-bold mb-3">
                        <i class="bi bi-camera me-2"></i>Выбор камеры
                    </h5>
                    <div id="cameraList" class="row g-2">
                        <!-- Камеры будут загружены через JS -->
                    </div>
                </div>
                
                <!-- Последние обнаружения -->
                <div class="glass-card p-4 mb-4">
                    <h5 class="fw-bold mb-3">
                        <i class="bi bi-bullseye me-2"></i>Последние обнаружения
                    </h5>
                    <div id="detectionsList" style="max-height: 300px; overflow-y: auto;">
                        <!-- Обнаружения будут загружены через JS -->
                    </div>
                </div>
                
                <!-- Распределение объектов -->
                <div class="glass-card p-4">
                    <h5 class="fw-bold mb-3">
                        <i class="bi bi-pie-chart me-2"></i>Распределение объектов
                    </h5>
                    <div class="mb-3">
                        <canvas id="objectDistributionChart" height="200"></canvas>
                    </div>
                    <div id="objectList" class="mt-3">
                        <!-- Список объектов будет загружен через JS -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Графики и статистика -->
        <div class="row mt-4">
            <div class="col-lg-6">
                <div class="glass-card p-4">
                    <h5 class="fw-bold mb-3">
                        <i class="bi bi-graph-up me-2"></i>Объекты за последний час
                    </h5>
                    <canvas id="objectsOverTimeChart" height="250"></canvas>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="glass-card p-4">
                    <h5 class="fw-bold mb-3">
                        <i class="bi bi-bar-chart me-2"></i>Статистика обнаружений
                    </h5>
                    <canvas id="detectionStatsChart" height="250"></canvas>
                    <div class="mt-3" id="statsInfo"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Модальное окно настроек -->
    <div class="modal fade" id="settingsModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"><i class="bi bi-gear me-2"></i>Настройки</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label class="form-label">Порог уверенности</label>
                        <input type="range" class="form-range" id="confidenceThreshold" min="0.1" max="0.9" step="0.1" value="0.5">
                        <div class="text-end">
                            <span id="confidenceValue">0.5</span>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">IOU Threshold</label>
                        <input type="range" class="form-range" id="iouThreshold" min="0.1" max="0.9" step="0.1" value="0.3">
                        <div class="text-end">
                            <span id="iouValue">0.3</span>
                        </div>
                    </div>
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="autoSave" checked>
                        <label class="form-check-label" for="autoSave">Автосохранение каждые 10 кадров</label>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Закрыть</button>
                    <button type="button" class="btn btn-primary" onclick="applySettings()">Применить</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        let charts = {};
        let updateInterval;
        let cameraThumbnails = {};
        
        // Инициализация при загрузке
        document.addEventListener('DOMContentLoaded', function() {
            loadCameras();
            updateStats();
            initCharts();
            
            // Обновление статистики каждую секунду
            updateInterval = setInterval(updateStats, 1000);
            
            // Инициализация слайдеров
            document.getElementById('confidenceThreshold').addEventListener('input', function(e) {
                document.getElementById('confidenceValue').textContent = e.target.value;
            });
            
            document.getElementById('iouThreshold').addEventListener('input', function(e) {
                document.getElementById('iouValue').textContent = e.target.value;
            });
        });
        
        function loadCameras() {
            fetch('/api/cameras')
                .then(response => response.json())
                .then(data => {
                    const cameraList = document.getElementById('cameraList');
                    cameraList.innerHTML = '';
                    
                    data.cameras.forEach(camera => {
                        const cameraHTML = `
                            <div class="col-6">
                                <div class="camera-thumbnail-container position-relative">
                                    <img src="/camera_preview/${camera.index}" 
                                         class="camera-thumbnail ${camera.index === data.current_camera ? 'active' : ''}"
                                         onclick="switchCamera(${camera.index})"
                                         alt="${camera.name}"
                                         data-camera-index="${camera.index}">
                                    <div class="position-absolute bottom-0 start-0 end-0 p-2 text-white text-center bg-dark bg-opacity-75">
                                        <small>${camera.name}</small>
                                    </div>
                                </div>
                            </div>
                        `;
                        cameraList.innerHTML += cameraHTML;
                        
                        // Сохраняем ссылку на изображение для обновления
                        cameraThumbnails[camera.index] = document.querySelector(`img[data-camera-index="${camera.index}"]`);
                    });
                    
                    // Обновление миниатюр каждые 3 секунды
                    setInterval(updateCameraPreviews, 3000);
                });
        }
        
        function updateCameraPreviews() {
            Object.keys(cameraThumbnails).forEach(index => {
                const img = cameraThumbnails[index];
                if (img) {
                    // Добавляем timestamp для предотвращения кэширования
                    img.src = `/camera_preview/${index}?t=${Date.now()}`;
                }
            });
        }
        
        function switchCamera(cameraIndex) {
            fetch('/api/switch_camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ camera_index: cameraIndex })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Обновляем активную камеру
                    document.querySelectorAll('.camera-thumbnail').forEach(img => {
                        img.classList.remove('active');
                    });
                    cameraThumbnails[cameraIndex].classList.add('active');
                    
                    // Сбрасываем статистику
                    resetStats();
                } else {
                    alert('Ошибка переключения камеры: ' + data.error);
                }
            });
        }
        
        function resetStats() {
            document.getElementById('totalFrames').textContent = '0';
            document.getElementById('savedFrames').textContent = '0';
            document.getElementById('totalObjects').textContent = '0';
            document.getElementById('fpsDisplay').textContent = 'FPS: 0';
            document.getElementById('frameCount').textContent = 'Кадров: 0';
            document.getElementById('objectCount').textContent = 'Объектов: 0';
        }
        
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    // Обновление основной статистики
                    document.getElementById('totalFrames').textContent = data.total_frames.toLocaleString();
                    document.getElementById('savedFrames').textContent = data.saved_frames.toLocaleString();
                    document.getElementById('totalObjects').textContent = data.total_objects.toLocaleString();
                    document.getElementById('fpsDisplay').textContent = `FPS: ${data.fps.toFixed(1)}`;
                    document.getElementById('frameCount').textContent = `Кадров: ${data.total_frames}`;
                    
                    // Обновление статуса
                    const statusBadge = document.getElementById('statusBadge');
                    const statusText = document.getElementById('statusText');
                    
                    if (data.is_paused) {
                        statusBadge.className = 'status-badge status-paused';
                        statusText.textContent = 'ПАУЗА';
                        document.getElementById('pauseBtn').innerHTML = '<i class="bi bi-play-circle me-2"></i>Возобновить';
                    } else {
                        statusBadge.className = 'status-badge status-active';
                        statusText.textContent = 'АКТИВНО';
                        document.getElementById('pauseBtn').innerHTML = '<i class="bi bi-pause-circle me-2"></i>Пауза';
                    }
                    
                    // Обновление текущих объектов
                    updateCurrentObjects(data.current_objects || {});
                    
                    // Обновление последних обнаружений
                    updateDetectionsList(data.recent_detections || []);
                    
                    // Обновление списка объектов
                    updateObjectList(data.object_counts || {});
                    
                    // Обновление графиков
                    updateCharts(data);
                })
                .catch(error => {
                    console.error('Ошибка получения статистики:', error);
                });
        }
        
        function updateCurrentObjects(objects) {
            const currentObjectsDiv = document.getElementById('currentObjects');
            const objectCount = Object.keys(objects).length;
            
            document.getElementById('objectCount').textContent = `Объектов: ${objectCount}`;
            
            if (objectCount === 0) {
                currentObjectsDiv.innerHTML = '<span class="text-muted">Нет объектов</span>';
                return;
            }
            
            // Группируем объекты по классам
            const classCounts = {};
            Object.values(objects).forEach(obj => {
                classCounts[obj.label] = (classCounts[obj.label] || 0) + 1;
            });
            
            let html = '';
            for (const [label, count] of Object.entries(classCounts)) {
                html += `<span class="object-badge">${label}: ${count}</span>`;
            }
            
            currentObjectsDiv.innerHTML = html;
        }
        
        function updateDetectionsList(detections) {
            const detectionsList = document.getElementById('detectionsList');
            
            if (detections.length === 0) {
                detectionsList.innerHTML = '<div class="text-center text-muted py-4">Нет обнаружений</div>';
                return;
            }
            
            let html = '';
            detections.forEach(detection => {
                const time = detection.timestamp.split('T')[1].split('.')[0];
                html += `
                    <div class="detection-item">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <span class="fw-bold">${detection.label}</span>
                                <small class="text-muted ms-2">${detection.confidence}%</small>
                            </div>
                            <small class="text-muted">${time}</small>
                        </div>
                    </div>
                `;
            });
            
            detectionsList.innerHTML = html;
        }
        
        function updateObjectList(objectCounts) {
            const objectList = document.getElementById('objectList');
            
            if (Object.keys(objectCounts).length === 0) {
                objectList.innerHTML = '<div class="text-muted text-center">Нет данных</div>';
                return;
            }
            
            let html = '<div class="row">';
            const sortedObjects = Object.entries(objectCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 8); // Показываем топ-8
            
            sortedObjects.forEach(([label, count]) => {
                const percentage = (count / Object.values(objectCounts).reduce((a, b) => a + b, 0)) * 100;
                html += `
                    <div class="col-6 mb-2">
                        <div class="d-flex justify-content-between">
                            <span>${label}</span>
                            <span class="fw-bold">${count} <small class="text-muted">(${percentage.toFixed(1)}%)</small></span>
                        </div>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar" role="progressbar" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            
            objectList.innerHTML = html;
        }
        
        function initCharts() {
            // Chart 1: Распределение объектов (круговая диаграмма)
            const ctx1 = document.getElementById('objectDistributionChart').getContext('2d');
            charts.distribution = new Chart(ctx1, {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            '#4361ee', '#3a0ca3', '#4cc9f0', '#7209b7', 
                            '#f72585', '#560bad', '#4895ef', '#3f37c9'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom',
                        }
                    }
                }
            });
            
            // Chart 2: Объекты во времени (линейный график)
            const ctx2 = document.getElementById('objectsOverTimeChart').getContext('2d');
            charts.timeline = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Объектов',
                        data: [],
                        borderColor: '#4361ee',
                        backgroundColor: 'rgba(67, 97, 238, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Количество объектов'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Время'
                            }
                        }
                    }
                }
            });
            
            // Chart 3: Статистика обнаружений (столбчатая диаграмма)
            const ctx3 = document.getElementById('detectionStatsChart').getContext('2d');
            charts.stats = new Chart(ctx3, {
                type: 'bar',
                data: {
                    labels: ['Человек', 'Автомобиль', 'Стул', 'Стол', 'Телефон', 'Ноутбук'],
                    datasets: [{
                        label: 'Обнаружено',
                        data: [0, 0, 0, 0, 0, 0],
                        backgroundColor: 'rgba(67, 97, 238, 0.7)',
                        borderColor: '#4361ee',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Количество'
                            }
                        }
                    }
                }
            });
        }
        
        function updateCharts(data) {
            // Обновление круговой диаграммы
            if (data.object_counts) {
                const labels = Object.keys(data.object_counts);
                const counts = Object.values(data.object_counts);
                
                charts.distribution.data.labels = labels;
                charts.distribution.data.datasets[0].data = counts;
                charts.distribution.update();
            }
            
            // Обновление временного графика
            if (data.detection_history && data.detection_history.length > 0) {
                const history = data.detection_history.slice(-20); // Последние 20 точек
                const labels = history.map(h => {
                    const date = new Date(h.timestamp);
                    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
                });
                const counts = history.map(h => h.object_count);
                
                charts.timeline.data.labels = labels;
                charts.timeline.data.datasets[0].data = counts;
                charts.timeline.update();
            }
            
            // Обновление статистики (пример для конкретных классов)
            if (data.object_counts) {
                const commonLabels = ['person', 'car', 'chair', 'dining table', 'cell phone', 'laptop'];
                const counts = commonLabels.map(label => data.object_counts[label] || 0);
                
                // Обновляем только если есть изменения
                if (JSON.stringify(charts.stats.data.datasets[0].data) !== JSON.stringify(counts)) {
                    charts.stats.data.datasets[0].data = counts;
                    charts.stats.update();
                }
            }
        }
        
        function togglePause() {
            fetch('/api/toggle_pause', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    // Статус обновится при следующем запросе статистики
                });
        }
        
        function saveSession() {
            fetch('/api/save_session', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Сессия сохранена!');
                });
        }
        
        function downloadAnnotations() {
            window.location.href = '/api/download_annotations';
        }
        
        function takeSnapshot() {
            fetch('/api/take_snapshot', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`Скриншот сохранён: ${data.filename}`);
                    } else {
                        alert('Ошибка сохранения скриншота');
                    }
                });
        }
        
        function applySettings() {
            const confidence = document.getElementById('confidenceThreshold').value;
            const iou = document.getElementById('iouThreshold').value;
            const autoSave = document.getElementById('autoSave').checked;
            
            fetch('/api/update_settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    confidence: parseFloat(confidence),
                    iou_threshold: parseFloat(iou),
                    auto_save: autoSave
                })
            })
            .then(response => response.json())
            .then(data => {
                alert('Настройки применены!');
                // Закрываем модальное окно
                bootstrap.Modal.getInstance(document.getElementById('settingsModal')).hide();
            });
        }
        
        // Функция для открытия модального окна настроек
        function openSettings() {
            const modal = new bootstrap.Modal(document.getElementById('settingsModal'));
            modal.show();
        }
    </script>
</body>
</html>
"""
        
        @app.route('/')
        def index():
            return render_template_string(HTML_PAGE, port=self.flask_port)
        
        @app.route('/video')
        def video_feed():
            """Видеопоток MJPEG"""
            def generate():
                while self.running:
                    try:
                        frame_data = self.frame_queue.get(timeout=1.0)
                        if frame_data is not None:
                            # Компрессия для быстрой передачи
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                            ret, jpeg = cv2.imencode('.jpg', frame_data, encode_param)
                            if ret:
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + 
                                      jpeg.tobytes() + b'\r\n')
                    except queue.Empty:
                        # Отправляем пустой кадр при отсутствии данных
                        continue
                    except Exception as e:
                        logger.error(f"Ошибка видео потока: {e}")
                        break
            
            return Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/camera_preview/<int:camera_index>')
        def camera_preview(camera_index):
            """Предпросмотр камеры"""
            try:
                # Создаем временный захват для предпросмотра
                if camera_index == self.current_camera_index:
                    # Используем основной поток если это текущая камера
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            frame = self.latest_frame.copy()
                        else:
                            # Создаем черный кадр если нет данных
                            frame = np.zeros((240, 320, 3), dtype=np.uint8)
                else:
                    # Для других камер создаем отдельный захват
                    temp_cap = cv2.VideoCapture(camera_index)
                    if temp_cap.isOpened():
                        ret, frame = temp_cap.read()
                        temp_cap.release()
                        if not ret:
                            frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    else:
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                
                # Изменяем размер для миниатюры
                frame = cv2.resize(frame, (320, 240))
                ret, jpeg = cv2.imencode('.jpg', frame)
                return Response(jpeg.tobytes(), mimetype='image/jpeg')
            except Exception as e:
                # Возвращаем черный кадр при ошибке
                black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                ret, jpeg = cv2.imencode('.jpg', black_frame)
                return Response(jpeg.tobytes(), mimetype='image/jpeg')
        
        @app.route('/api/stats')
        def get_stats():
            """Получение статистики в формате JSON"""
            # Получаем последние объекты
            current_objects = {}
            if hasattr(self, 'prev_objects') and self.prev_objects:
                current_objects = self.prev_objects
            
            # Формируем историю обнаружений (последние 50 записей)
            detection_history = []
            if self.stats.get('detection_history'):
                detection_history = self.stats['detection_history'][-50:]
            
            stats_data = {
                'total_frames': self.stats['total_frames'],
                'saved_frames': self.stats['saved_frames'],
                'total_objects': self.stats['total_objects'],
                'fps': self.stats['fps'],
                'start_time': self.stats['start_time'],
                'object_counts': self.stats['object_counts'],
                'current_objects': current_objects,
                'recent_detections': self.get_recent_detections(10),
                'detection_history': detection_history,
                'is_paused': self.pause_annotation,
                'queue_size': self.frame_queue.qsize(),
                'current_camera': self.current_camera_index
            }
            return jsonify(stats_data)
        
        @app.route('/api/cameras')
        def get_cameras():
            """Получение списка доступных камер"""
            return jsonify({
                'cameras': self.available_cameras,
                'current_camera': self.current_camera_index
            })
        
        @app.route('/api/switch_camera', methods=['POST'])
        def api_switch_camera():
            """Переключение камеры"""
            try:
                data = request.json
                camera_index = int(data.get('camera_index', 0))
                
                success = self.switch_camera(camera_index)
                if success:
                    return jsonify({'success': True, 'message': f'Камера переключена на {camera_index}'})
                else:
                    return jsonify({'success': False, 'error': 'Не удалось переключить камеру'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/take_snapshot', methods=['POST'])
        def take_snapshot():
            """Создание скриншота"""
            try:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"snapshot_{timestamp}.jpg"
                        filepath = self.screenshots_dir / filename
                        
                        # Сохраняем кадр
                        cv2.imwrite(str(filepath), self.latest_frame)
                        
                        # Добавляем аннотации если есть
                        if self.prev_objects:
                            annotated_frame = self.latest_frame.copy()
                            for obj in self.prev_objects.values():
                                x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, f"{obj['label']}: {obj['confidence']:.2f}",
                                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            annotated_filename = f"snapshot_annotated_{timestamp}.jpg"
                            annotated_filepath = self.screenshots_dir / annotated_filename
                            cv2.imwrite(str(annotated_filepath), annotated_frame)
                        
                        return jsonify({'success': True, 'filename': filename})
                    else:
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
                    headers={'Content-Disposition': 'attachment; filename=vision_ai_annotations.json'}
                )
            return jsonify({'error': 'No annotations available'}), 404
        
        @app.route('/api/save_session', methods=['POST'])
        def save_session():
            """Сохранение текущей сессии"""
            success = self._save_to_json()
            if success:
                return jsonify({'message': f'Session saved with {len(self.annotations)} frames'})
            return jsonify({'error': 'Failed to save session'}), 500
        
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
                    # Можно добавить обновление порога уверенности модели
                    pass
                if 'iou_threshold' in data:
                    self.iou_threshold = float(data['iou_threshold'])
                return jsonify({'message': 'Settings updated'})
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        logger.info(f"🌐 Веб-интерфейс доступен по http://localhost:{self.flask_port}")
        logger.info(f"   📊 Статистика: http://localhost:{self.flask_port}/api/stats")
        logger.info(f"   📥 Аннотации: http://localhost:{self.flask_port}/api/download_annotations")
        
        app.run(host='0.0.0.0', port=self.flask_port, debug=False, threaded=True, use_reloader=False)
    
    def get_recent_detections(self, count=10):
        """Получение последних обнаруженных объектов"""
        recent = []
        frames = list(self.annotations.values())[-10:]  # Последние 10 кадров
        
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
        
        # Координаты пересечения
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def has_significant_changes(self, current_objects):
        """Проверка на значительные изменения"""
        if self.pause_annotation:
            return False
        
        if self.prev_objects is None:
            return True
        
        if len(current_objects) != len(self.prev_objects):
            return True
        
        # Проверка классов
        current_labels = set(obj['label'] for obj in current_objects.values())
        prev_labels = set(obj['label'] for obj in self.prev_objects.values())
        if current_labels != prev_labels:
            return True
        
        # Проверка положения объектов
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
        """Подготовка данных аннотаций для экспорта"""
        return {
            'metadata': {
                'project': 'Vision AI Annotator',
                'version': '2.0',
                'export_date': datetime.now().isoformat(),
                'total_frames': len(self.annotations),
                'total_objects': self.stats['total_objects'],
                'session_duration_seconds': time.time() - self.stats['start_time'],
                'camera_index': self.current_camera_index,
                'settings': {
                    'model': 'yolov8n.pt',
                    'confidence_threshold': 0.5,
                    'iou_threshold': self.iou_threshold,
                    'position_threshold': self.position_threshold
                }
            },
            'statistics': self.stats,
            'frames': dict(self.annotations)
        }
    
    def run(self):
        """Основной цикл обработки"""
        logger.info("🚀 Запуск Vision AI Annotator")
        logger.info("   Нажмите Ctrl+C для выхода")
        logger.info("   Пауза/возобновление через веб-интерфейс")
        
        frame_count = 0
        saved_frame_count = 0
        last_fps_time = time.time()
        fps_frames = 0
        last_history_update = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Не удалось получить кадр с камеры")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                fps_frames += 1
                timestamp = datetime.now().isoformat()
                
                # Расчет FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.stats['fps'] = fps_frames / (current_time - last_fps_time)
                    fps_frames = 0
                    last_fps_time = current_time
                
                # Детекция объектов (если не на паузе)
                current_objects = OrderedDict()
                
                if not self.pause_annotation:
                    results = self.model(frame, verbose=False, conf=0.5)
                    result = results[0]
                    
                    if result.boxes is not None:
                        boxes = result.boxes.cpu().numpy()
                        
                        for i in range(len(boxes)):
                            box = boxes[i]
                            conf = box.conf[0]
                            
                            if conf > 0.5:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cls_id = int(box.cls[0])
                                label = self.model.names[cls_id]
                                
                                obj_id = f"{label}_{i}_{frame_count}"
                                
                                current_objects[obj_id] = {
                                    'label': label,
                                    'class_id': cls_id,
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2,
                                    'confidence': float(conf),
                                    'width': x2 - x1,
                                    'height': y2 - y1,
                                    'center_x': (x1 + x2) // 2,
                                    'center_y': (y1 + y2) // 2
                                }
                                
                                # Обновление статистики объектов
                                self.stats['object_counts'][label] = self.stats['object_counts'].get(label, 0) + 1
                
                # Проверка на сохранение
                should_save = self.has_significant_changes(current_objects)
                
                if should_save and not self.pause_annotation:
                    saved_frame_count += 1
                    
                    frame_annotation = {
                        'frame_number': frame_count,
                        'saved_index': saved_frame_count,
                        'timestamp': timestamp,
                        'objects': current_objects
                    }
                    
                    self.annotations[f"frame_{saved_frame_count}"] = frame_annotation
                    
                    # Автосохранение каждые 10 кадров
                    if saved_frame_count % 10 == 0:
                        self._save_to_json()
                    
                    self.prev_objects = current_objects.copy()
                
                # Обновление истории обнаружений (каждые 5 секунд)
                if current_time - last_history_update >= 5:
                    self.stats['detection_history'].append({
                        'timestamp': timestamp,
                        'object_count': len(current_objects),
                        'objects': list(current_objects.keys())
                    })
                    # Ограничиваем историю 100 записями
                    if len(self.stats['detection_history']) > 100:
                        self.stats['detection_history'] = self.stats['detection_history'][-100:]
                    last_history_update = current_time
                
                # Обновление статистики
                self.stats['total_frames'] = frame_count
                self.stats['saved_frames'] = saved_frame_count
                self.stats['total_objects'] = sum(len(frame['objects']) 
                                                 for frame in self.annotations.values())
                
                # Отображение (если не в режиме только веб)
                if not self.pause_annotation and 'results' in locals():
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame.copy()
                
                # Добавление информации на кадр
                color = (0, 255, 0) if not self.pause_annotation else (0, 0, 255)
                status = "ACTIVE" if not self.pause_annotation else "PAUSED"
                
                cv2.putText(annotated_frame, f"Status: {status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(annotated_frame, f"Frames: {frame_count} ({saved_frame_count} saved)", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"Objects: {len(current_objects)}", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"FPS: {self.stats['fps']:.1f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"Camera: {self.current_camera_index}", (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(annotated_frame, f"Web UI: http://localhost:{self.flask_port}", 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Отправка кадра в веб-интерфейс
                try:
                    display_frame = cv2.resize(annotated_frame, (854, 480))
                    
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    if display_frame is not None:
                        self.frame_queue.put_nowait(display_frame)
                        
                        with self.frame_lock:
                            self.latest_frame = display_frame.copy()
                except Exception as e:
                    logger.debug(f"Ошибка очереди кадров: {e}")
                
                # Отображение в локальном окне
                cv2.imshow('Vision AI Annotator - Local View', annotated_frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' или ESC
                    self.running = False
                    break
                elif key == ord(' '):  # Пробел для паузы
                    self.pause_annotation = not self.pause_annotation
                    logger.info(f"Пауза: {self.pause_annotation}")
                elif key == ord('s'):  # Принудительное сохранение
                    self._save_to_json()
                    logger.info("Принудительное сохранение выполнено")
                elif key == ord('c'):  # Переключение камеры
                    self.current_camera_index = (self.current_camera_index + 1) % max(len(self.available_cameras), 1)
                    self.switch_camera(self.current_camera_index)
        
        except KeyboardInterrupt:
            logger.info("Прервано пользователем")
        
        except Exception as e:
            logger.error(f"Ошибка в основном цикле: {e}")
        
        finally:
            self.cleanup()
            
            # Финальная статистика
            logger.info("\n" + "="*50)
            logger.info("СЕССИЯ ЗАВЕРШЕНА")
            logger.info("="*50)
            logger.info(f"Всего обработано кадров: {frame_count}")
            logger.info(f"Сохранено кадров: {saved_frame_count}")
            logger.info(f"Всего объектов: {self.stats['total_objects']}")
            logger.info(f"Обнаружено классов: {len(self.stats['object_counts'])}")
            logger.info(f"Эффективность: {saved_frame_count/frame_count*100:.1f}%")
            logger.info(f"Средний FPS: {self.stats['fps']:.1f}")
            logger.info(f"Файл аннотаций: {self.output_file}")
            logger.info(f"Скриншоты сохранены в: {self.screenshots_dir}")
            logger.info(f"Веб-интерфейс был доступен по: http://localhost:{self.flask_port}")
            logger.info("="*50)
    
    def _save_to_json(self, final=False):
        """Сохранение аннотаций в JSON файл"""
        try:
            if self.annotations:
                annotations_data = self.prepare_annotations_data()
                
                filename = self.output_file if final else f"autosave_{self.output_file}"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(annotations_data, f, indent=2, ensure_ascii=False)
                
                if final:
                    logger.info(f"Финальное сохранение: {len(self.annotations)} кадров в {filename}")
                else:
                    logger.debug(f"Автосохранение: {len(self.annotations)} кадров")
                
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
        
        return False

def main():
    """Точка входа в приложение"""
    print("\n" + "="*60)
    print("🚀 VISION AI ANNOTATOR v2.0")
    print("="*60)
    print("Профессиональная система аннотации объектов в реальном времени")
    print("\n✨ Новые возможности:")
    print("  • Выбор и переключение камер в реальном времени")
    print("  • Работающие графики со статистикой")
    print("  • Современный интерфейс на Bootstrap 5")
    print("  • Предпросмотр всех доступных камер")
    print("  • Скриншоты с аннотациями")
    print("\n🎮 Управление:")
    print("  • Пробел - пауза/возобновление")
    print("  • S - принудительное сохранение")
    print("  • C - переключение камеры")
    print("  • Q или ESC - выход")
    print("="*60)
    
    try:
        port = int(input(f"Введите порт для веб-интерфейса [по умолчанию 3000]: ") or "3000")
        
        annotator = ProfessionalYOLOAnnotator(
            output_file='vision_ai_annotations.json',
            flask_port=port
        )
        
        annotator.run()
        
    except ValueError:
        print("Ошибка: порт должен быть числом")
    except Exception as e:
        print(f"Ошибка запуска: {e}")
    finally:
        print("\n👋 До свидания!")

if __name__ == "__main__":
    main()