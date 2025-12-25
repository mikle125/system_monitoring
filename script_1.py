import cv2
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from collections import OrderedDict
import time
import numpy as np

class OptimizedFrameByFrameYOLOAnnotator:
    def __init__(self, output_file='annotations.json'):
        """
        Инициализация аннотатора с оптимизированным сохранением
        
        Args:
            output_file: путь к выходному JSON файлу
        """
        self.model = YOLO('yolov8n.pt')
        self.output_file = Path(output_file)
        self.annotations = OrderedDict()  # Все аннотации за сессию
        
        # Параметры для определения изменений
        self.prev_objects = None  # Объекты на предыдущем кадре
        self.position_threshold = 50  # Порог изменения положения (пиксели)
        self.iou_threshold = 0.3  # Порог IoU для определения изменений
        
        # Открытие веб-камеры
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise Exception("Не удалось открыть веб-камеру")
    
    def calculate_iou(self, box1, box2):
        """Вычисление Intersection over Union (IoU) между двумя bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Координаты пересечения
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Площадь пересечения
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Площадь каждого бокса
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Площадь объединения
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def has_significant_changes(self, current_objects):
        """
        Проверка на значительные изменения между кадрами
        
        Returns:
            bool: True если есть значительные изменения
        """
        if self.prev_objects is None:
            return True  # Первый кадр всегда сохраняем
        
        # Проверка изменения количества объектов
        if len(current_objects) != len(self.prev_objects):
            return True
        
        # Проверка изменения классов объектов
        current_labels = set(obj['label'] for obj in current_objects.values())
        prev_labels = set(obj['label'] for obj in self.prev_objects.values())
        
        if current_labels != prev_labels:
            return True
        
        # Проверка значительного изменения положения объектов
        for obj_id, curr_obj in current_objects.items():
            if obj_id in self.prev_objects:
                prev_obj = self.prev_objects[obj_id]
                
                # Проверка изменения положения
                curr_box = (curr_obj['x1'], curr_obj['y1'], curr_obj['x2'], curr_obj['y2'])
                prev_box = (prev_obj['x1'], prev_obj['y1'], prev_obj['x2'], prev_obj['y2'])
                
                # Проверка IoU
                iou = self.calculate_iou(curr_box, prev_box)
                if iou < self.iou_threshold:
                    return True
                
                # Проверка смещения центра
                curr_center = ((curr_obj['x1'] + curr_obj['x2']) // 2, 
                              (curr_obj['y1'] + curr_obj['y2']) // 2)
                prev_center = ((prev_obj['x1'] + prev_obj['x2']) // 2, 
                              (prev_obj['y1'] + prev_obj['y2']) // 2)
                
                # Евклидово расстояние между центрами
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                
                if distance > self.position_threshold:
                    return True
        
        return False  # Значительных изменений нет
    
    def run(self):
        """
        Основной цикл с оптимизированным сохранением
        """
        print("Запуск оптимизированной аннотации")
        print("Кадры сохраняются только при значительных изменениях")
        print("Нажмите 'q' для выхода")
        print("Нажмите 's' для принудительного сохранения текущего кадра")
        
        frame_count = 0
        saved_frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp = datetime.now().isoformat()
                
                # Детекция объектов
                results = self.model(frame, verbose=False)
                result = results[0]
                
                # Данные для текущего кадра
                current_objects = OrderedDict()
                
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        box = boxes[i]
                        conf = box.conf[0]
                        
                        if conf > 0.5:  # Порог уверенности
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            label = self.model.names[cls_id]
                            
                            # Уникальный ID объекта в кадре
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
                
                # Проверяем, нужно ли сохранять этот кадр
                should_save = self.has_significant_changes(current_objects)
                
                # Принудительное сохранение по клавише 's'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    should_save = True
                    print(f"Принудительное сохранение кадра {frame_count}")
                
                if should_save:
                    saved_frame_count += 1
                    
                    # Сохраняем данные кадра
                    frame_annotation = {
                        'frame_number': frame_count,
                        'saved_index': saved_frame_count,
                        'timestamp': timestamp,
                        'objects': current_objects
                    }
                    
                    self.annotations[f"frame_{saved_frame_count}"] = frame_annotation
                    
                    # Сохраняем в файл
                    self._save_to_json()
                    
                    # Обновляем предыдущие объекты
                    self.prev_objects = current_objects.copy()
                
                # Отображение результатов
                annotated_frame = result.plot()
                
                # Добавление информации на экран
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Saved Frames: {saved_frame_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Objects: {len(current_objects)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Расчет FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Индикатор сохранения
                status_color = (0, 255, 0) if should_save else (0, 0, 255)
                status_text = "SAVED" if should_save else "SKIPPED"
                cv2.putText(annotated_frame, f"Status: {status_text}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.imshow('Optimized YOLO Annotation', annotated_frame)
                
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nПрервано пользователем")
        
        finally:
            # Финализируем сохранение
            self._save_to_json(final=True)
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Статистика
            total_objects = sum(len(frame['objects']) 
                              for frame in self.annotations.values())
            print(f"\nИтоги сессии:")
            print(f"  Обработано кадров: {frame_count}")
            print(f"  Сохранено кадров: {saved_frame_count}")
            print(f"  Сжатие: {saved_frame_count/frame_count*100:.1f}% от общего числа кадров")
            print(f"  Обнаружено объектов: {total_objects}")
            print(f"  Файл сохранен: {self.output_file}")
    
    def _save_to_json(self, final=False):
        """Сохранение всех аннотаций в один JSON файл"""
        if self.annotations:
            session_data = {
                'session_info': {
                    'total_processed_frames': len(self.annotations),
                    'total_objects': sum(len(frame['objects']) 
                                       for frame in self.annotations.values()),
                    'save_time': datetime.now().isoformat(),
                    'settings': {
                        'position_threshold': self.position_threshold,
                        'iou_threshold': self.iou_threshold,
                        'confidence_threshold': 0.5
                    }
                },
                'frames': self.annotations
            }
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            if final:
                print(f"Финальное сохранение: {len(self.annotations)} кадров в {self.output_file}")
            else:
                print(f"Автосохранение: {len(self.annotations)} кадров")
            
            return True
        return False
    
    def print_statistics(self):
        """Вывод статистики по сохраненным данным"""
        if not self.annotations:
            print("Нет сохраненных данных")
            return
        
        print("\nСтатистика сохраненных данных:")
        print(f"Всего сохранено кадров: {len(self.annotations)}")
        
        # Статистика по объектам
        object_counts = {}
        for frame_data in self.annotations.values():
            for obj in frame_data['objects'].values():
                label = obj['label']
                object_counts[label] = object_counts.get(label, 0) + 1
        
        print("Объекты по категориям:")
        for label, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")

# Упрощенная версия с одним JSON файлом
class SimpleYOLOAnnotator:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.annotations = OrderedDict()
        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0
        
    def run(self):
        print("Простая аннотация с сохранением в один JSON файл")
        print("Нажмите 'q' для выхода")
        print("Нажмите 's' для сохранения текущего кадра")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                timestamp = datetime.now().isoformat()
                
                # Детекция
                results = self.model(frame, verbose=False)
                result = results[0]
                
                # Подготовка данных для кадра
                frame_data = {
                    'frame_number': self.frame_count,
                    'timestamp': timestamp,
                    'objects': OrderedDict()
                }
                
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        conf = box.conf[0]
                        
                        if conf > 0.5:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            label = self.model.names[cls_id]
                            
                            obj_id = f"{label}_{i}_{self.frame_count}"
                            
                            frame_data['objects'][obj_id] = {
                                'label': label,
                                'class_id': cls_id,
                                'bbox': {
                                    'x1': x1,
                                    'y1': y1,
                                    'x2': x2,
                                    'y2': y2
                                },
                                'confidence': float(conf),
                                'center': {
                                    'x': (x1 + x2) // 2,
                                    'y': (y1 + y2) // 2
                                }
                            }
                
                # Всегда сохраняем в память
                self.annotations[f"frame_{self.frame_count}"] = frame_data
                
                # Отображение
                annotated_frame = result.plot()
                
                # Информация
                cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Objects: {len(frame_data['objects'])}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "All frames in memory", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Simple YOLO Annotation', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_to_json()
        
        except KeyboardInterrupt:
            print("\nПрервано пользователем")
        
        finally:
            self.save_to_json(final=True)
            self.cap.release()
            cv2.destroyAllWindows()
            
            total_objects = sum(len(frame['objects']) 
                              for frame in self.annotations.values())
            print(f"\nСохранено {len(self.annotations)} кадров с {total_objects} объектами")
    
    def save_to_json(self, final=False):
        """Сохранение всех данных в один JSON файл"""
        if self.annotations:
            session_data = {
                'metadata': {
                    'total_frames': len(self.annotations),
                    'total_objects': sum(len(frame['objects']) 
                                       for frame in self.annotations.values()),
                    'created_at': datetime.now().isoformat(),
                    'model': 'yolov8n.pt'
                },
                'frames': self.annotations
            }
            
            filename = 'annotations.json' if final else 'annotations_autosave.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"Сохранено в {filename}: {len(self.annotations)} кадров")

def main():
    """Основная функция запуска"""
    print("Выберите режим:")
    print("1. Оптимизированное сохранение (только при изменениях)")
    print("2. Простое сохранение (все кадры в один файл)")
    print("3. Быстрый тест")
    
    try:
        choice = input("Введите номер (1/2/3): ").strip()
        
        if choice == '1':
            annotator = OptimizedFrameByFrameYOLOAnnotator()
            annotator.run()
            annotator.print_statistics()
        elif choice == '2':
            annotator = SimpleYOLOAnnotator()
            annotator.run()
        elif choice == '3':
            quick_test()
        else:
            print("Неверный выбор. Запускаю режим 2 по умолчанию")
            annotator = SimpleYOLOAnnotator()
            annotator.run()
    except KeyboardInterrupt:
        print("\nПрограмма завершена")

def quick_test():
    """Быстрый тест"""
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    
    print("Быстрый тест - 50 кадров")
    print("Нажмите 'q' для досрочного выхода")
    
    annotations = OrderedDict()
    frame_count = 0
    
    try:
        for _ in range(50):  # Ограничиваем 50 кадрами для теста
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = datetime.now().isoformat()
            
            results = model(frame, verbose=False)
            result = results[0]
            
            frame_data = {
                'frame': frame_count,
                'timestamp': timestamp,
                'objects': OrderedDict()
            }
            
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    conf = box.conf[0]
                    
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        
                        obj_id = f"{label}_{i}"
                        frame_data['objects'][obj_id] = {
                            'label': label,
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2]
                        }
            
            annotations[f"frame_{frame_count}"] = frame_data
            
            # Отображение
            annotated_frame = result.plot()
            cv2.putText(annotated_frame, f"Test Frame: {frame_count}/50", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Quick Test', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Ошибка: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Сохраняем все в один файл
        if annotations:
            output_data = {
                'test_info': {
                    'frames': len(annotations),
                    'timestamp': datetime.now().isoformat()
                },
                'frames': annotations
            }
            
            with open('quick_test.json', 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            total_objects = sum(len(frame['objects']) for frame in annotations.values())
            print(f"\nТест завершен. Сохранено {len(annotations)} кадров с {total_objects} объектами в quick_test.json")

if __name__ == "__main__":
    main()