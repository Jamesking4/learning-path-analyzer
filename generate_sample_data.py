"""
Скрипт для генерации тестовых данных LMS
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sample_data(num_students=50, days=30, output_file='data/sample_large_log.csv'):
    """Генерация синтетических данных LMS"""
    
    # Параметры генерации
    courses = ['course_101', 'course_102', 'course_201']
    modules = [f'module_{i}' for i in range(1, 6)]
    event_types = [
        'login', 'logout', 'content_view', 'content_download',
        'assignment_submit', 'quiz_attempt', 'exam_start',
        'forum_post', 'forum_reply', 'forum_view',
        'message_send', 'resource_access'
    ]
    
    # База данных студентов
    students = [f'student_{i:04d}' for i in range(1001, 1001 + num_students)]
    
    records = []
    start_date = datetime(2024, 1, 1)
    
    print(f"🎯 Generating data for {num_students} students over {days} days...")
    
    # Генерация данных для каждого студента
    for student_id in students:
        # Случайное количество активностей для студента
        num_activities = np.random.poisson(lam=40)
        
        for _ in range(num_activities):
            # Случайное смещение времени
            time_offset = timedelta(
                days=np.random.randint(0, days),
                hours=np.random.randint(8, 20),
                minutes=np.random.randint(0, 60)
            )
            
            event_time = start_date + time_offset
            event_type = random.choice(event_types)
            course = random.choice(courses)
            module = random.choice(modules)
            
            # Генерация оценки для оценочных событий
            grade = None
            activity_duration = 0
            
            if 'assignment' in event_type or 'quiz' in event_type:
                # Базовый балл с вариацией
                base_grade = np.random.normal(75, 15)
                grade = max(0, min(100, round(base_grade, 1)))
                activity_duration = np.random.exponential(60)
            elif 'forum' in event_type or 'content' in event_type:
                activity_duration = np.random.exponential(20)
            else:
                activity_duration = np.random.exponential(5)
            
            # Ограничение длительности
            activity_duration = min(activity_duration, 180)
            
            record = {
                'student_id': student_id,
                'event_type': event_type,
                'event_time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
                'module': module,
                'course': course,
                'grade': grade,
                'activity_duration': round(activity_duration, 1)
            }
            
            records.append(record)
    
    # Создание DataFrame
    df = pd.DataFrame(records)
    
    # Сортировка по времени
    df['event_time'] = pd.to_datetime(df['event_time'])
    df = df.sort_values('event_time')
    
    # Создаем директорию если её нет
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Сохранение в CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Generated {len(df)} records for {num_students} students")
    print(f"📁 Saved to {output_file}")
    
    # Основная статистика
    print(f"\n📊 Dataset statistics:")
    print(f"   Total records: {len(df)}")
    print(f"   Unique students: {df['student_id'].nunique()}")
    print(f"   Time range: {df['event_time'].min()} to {df['event_time'].max()}")
    print(f"   Event types: {df['event_type'].nunique()}")
    
    return df

def generate_small_sample():
    """Генерация маленького примера данных"""
    print("📦 Generating small sample dataset...")
    
    # Пример данных для тестов
    data = {
        'student_id': ['1001', '1001', '1001', '1002', '1002', '1003'],
        'event_type': ['login', 'assignment_submit', 'forum_post', 
                      'login', 'quiz_attempt', 'content_view'],
        'event_time': ['2024-01-15 09:30:00', '2024-01-15 11:45:00', 
                      '2024-01-16 14:20:00', '2024-01-15 09:35:00',
                      '2024-01-17 10:15:00', '2024-01-18 15:30:00'],
        'module': ['module_1', 'module_1', 'module_1', 
                  'module_1', 'module_1', 'module_2'],
        'course': ['course_101', 'course_101', 'course_101',
                  'course_101', 'course_101', 'course_102'],
        'grade': [None, 85, None, None, 92, None],
        'activity_duration': [0, 120, 15, 0, 45, 30]
    }
    
    df = pd.DataFrame(data)
    output_file = 'data/sample_log.csv'
    
    # Создаем директорию если её нет
    os.makedirs('data', exist_ok=True)
    
    # Сохранение
    df.to_csv(output_file, index=False)
    print(f"✅ Small sample saved to {output_file}")
    print(f"   Records: {len(df)}")
    
    return df

if __name__ == "__main__":
    print("=" * 50)
    print("🎓 Learning Path Analyzer - Data Generator")
    print("=" * 50)
    
    # Создаем директорию data
    os.makedirs('data', exist_ok=True)
    
    # Генерация данных
    generate_small_sample()
    generate_sample_data(
        num_students=100,
        days=90,
        output_file='data/sample_large_log.csv'
    )
    
    print("\n🎉 All datasets generated successfully!")
