"""
Модуль для генерации рекомендаций на основе анализа данных
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any


class RecommendationEngine:
    """Двигатель рекомендаций для оптимизации обучения"""
    
    def __init__(self, config):
        self.config = config
        self.thresholds = config['analysis']
        
    def generate_personalized_recommendations(self, df, student_id, top_n=None):
        """Генерация персонализированных рекомендаций для студента"""
        if top_n is None:
            top_n = self.thresholds['top_n_recommendations']
        
        # Фильтрация данных студента
        student_data = df[df['student_id'] == student_id]
        
        if student_data.empty:
            return ["No data available for this student"]
        
        recommendations = []
        
        # 1. Рекомендации на основе активности
        activity_recs = self._analyze_activity_patterns(student_data)
        recommendations.extend(activity_recs)
        
        # 2. Рекомендации на основе успеваемости
        if 'grade' in student_data.columns:
            grade_recs = self._analyze_performance(student_data)
            recommendations.extend(grade_recs)
        
        # 3. Рекомендации на основе временных паттернов
        time_recs = self._analyze_time_patterns(student_data)
        recommendations.extend(time_recs)
        
        # 4. Рекомендации на основе сравнения с успешными студентами
        comparison_recs = self._compare_with_successful_students(df, student_id)
        recommendations.extend(comparison_recs)
        
        # Удаление дубликатов и ограничение количества
        unique_recs = list(dict.fromkeys(recommendations))
        return unique_recs[:top_n]
    
    def generate_general_recommendations(self, df):
        """Генерация общих рекомендаций для всех студентов"""
        recommendations = []
        
        # Анализ общих паттернов
        overall_stats = self._calculate_overall_statistics(df)
        
        # 1. Рекомендации по активности
        avg_events = overall_stats.get('avg_events_per_student', 0)
        if avg_events < 10:
            recommendations.append(
                "Increase overall course engagement - aim for at least 10 activities per week"
            )
        
        # 2. Рекомендации по типам активностей
        event_dist = overall_stats.get('event_distribution', {})
        if event_dist.get('social', 0) < event_dist.get('assessment', 0) * 0.5:
            recommendations.append(
                "Encourage more forum participation and peer collaboration"
            )
        
        # 3. Рекомендации по времени
        time_patterns = self._analyze_overall_time_patterns(df)
        if time_patterns.get('weekend_activity', 0) < 0.1:
            recommendations.append(
                "Consider distributing learning activities more evenly throughout the week"
            )
        
        # 4. Рекомендации для преподавателей
        if 'grade' in df.columns:
            grade_stats = overall_stats.get('grade_stats', {})
            if grade_stats.get('std', 0) > 20:
                recommendations.append(
                    "Consider additional support for students with grades below 70%"
                )
        
        return recommendations[:self.thresholds['top_n_recommendations']]
    
    def _analyze_activity_patterns(self, student_data):
        """Анализ паттернов активности студента"""
        recommendations = []
        
        # Подсчет различных типов активностей
        if 'event_category' in student_data.columns:
            event_counts = student_data['event_category'].value_counts()
            
            # Проверка социальной активности
            social_events = event_counts.get('social', 0)
            if social_events < 3:
                recommendations.append(
                    "Increase participation in forum discussions and peer collaboration"
                )
            
            # Проверка взаимодействия с контентом
            content_events = event_counts.get('content_interaction', 0)
            if content_events < 5:
                recommendations.append(
                    "Spend more time reviewing course materials and resources"
                )
        
        # Анализ длительности активности
        if 'activity_duration' in student_data.columns:
            avg_duration = student_data['activity_duration'].mean()
            if avg_duration > 120:
                recommendations.append(
                    "Break study sessions into shorter, more frequent intervals (30-60 minutes)"
                )
            elif avg_duration < 30:
                recommendations.append(
                    "Increase study session duration to at least 30 minutes for better retention"
                )
        
        return recommendations
    
    def _analyze_performance(self, student_data):
        """Анализ успеваемости студента"""
        recommendations = []
        
        grade_data = student_data[student_data['grade'] > 0]['grade']
        
        if len(grade_data) > 0:
            avg_grade = grade_data.mean()
            grade_std = grade_data.std()
            
            if avg_grade < self.thresholds['min_grade_threshold']:
                recommendations.append(
                    f"Seek additional help - current average grade ({avg_grade:.1f}%) is below threshold"
                )
            
            if grade_std > 15 and len(grade_data) > 3:
                recommendations.append(
                    "Work on consistency - grades vary significantly between assignments"
                )
            
            # Анализ улучшения/ухудшения
            if len(grade_data) >= 3:
                grades_sorted = grade_data.sort_index().values
                trend = self._calculate_trend(grades_sorted)
                
                if trend < -0.1:
                    recommendations.append(
                        "Performance trend is declining - consider reviewing study strategies"
                    )
                elif trend > 0.1:
                    recommendations.append(
                        "Great improvement trend! Continue with current strategies"
                    )
        
        return recommendations
    
    def _analyze_time_patterns(self, student_data):
        """Анализ временных паттернов"""
        recommendations = []
        
        if 'hour' in student_data.columns:
            # Предпочтительное время дня
            peak_hour = student_data['hour'].mode()
            if len(peak_hour) > 0:
                hour = peak_hour[0]
                if 22 <= hour <= 24 or 0 <= hour <= 4:
                    recommendations.append(
                        "Consider studying during daylight hours for better concentration"
                    )
                elif 14 <= hour <= 17:
                    recommendations.append(
                        "Good study time detected - afternoon sessions are effective for most learners"
                    )
        
        if 'day_of_week' in student_data.columns:
            # Распределение по дням недели
            weekend_ratio = len(student_data[student_data['day_of_week'] >= 5]) / len(student_data)
            if weekend_ratio > 0.5:
                recommendations.append(
                    "Balance study time more evenly throughout the week"
                )
            elif weekend_ratio < 0.1:
                recommendations.append(
                    "Consider some weekend review sessions for better retention"
                )
        
        # Регулярность
        if 'date' in student_data.columns:
            unique_days = student_data['date'].nunique()
            total_days = (student_data['date'].max() - student_data['date'].min()).days + 1
            
            if total_days > 7:  # Если курс длится больше недели
                regularity = unique_days / total_days
                if regularity < 0.5:
                    recommendations.append(
                        "Increase regularity - aim to engage with the course at least every other day"
                    )
        
        return recommendations
    
    def _compare_with_successful_students(self, df, student_id):
        """Сравнение с успешными студентами"""
        recommendations = []
        
        # Идентификация успешных студентов
        successful = self._identify_successful_students(df)
        
        if successful.empty:
            return recommendations
        
        # Данные текущего студента
        student_data = df[df['student_id'] == student_id]
        
        # Сравнение метрик
        comparisons = []
        
        # Частота активностей
        student_freq = len(student_data)
        successful_freq = len(df[df['student_id'].isin(successful.index)]) / len(successful)
        
        if student_freq < successful_freq * 0.7:
            comparisons.append("activity frequency")
        
        # Социальная активность
        if 'event_category' in df.columns:
            student_social = len(student_data[student_data['event_category'] == 'social'])
            successful_social = len(
                df[(df['student_id'].isin(successful.index)) & 
                   (df['event_category'] == 'social')]
            ) / len(successful)
            
            if student_social < successful_social * 0.5:
                comparisons.append("forum participation")
        
        if comparisons:
            recommendations.append(
                f"Increase {', '.join(comparisons)} to match patterns of successful students"
            )
        
        return recommendations
    
    def _identify_successful_students(self, df):
        """Идентификация успешных студентов"""
        if 'grade' not in df.columns:
            return pd.DataFrame()
        
        # Студенты с оценками выше порога
        student_grades = df[df['grade'] > 0].groupby('student_id')['grade'].mean()
        successful = student_grades[student_grades >= self.thresholds['min_grade_threshold']]
        
        return successful
    
    def _calculate_overall_statistics(self, df):
        """Расчет общей статистики"""
        stats = {
            'total_students': df['student_id'].nunique(),
            'total_events': len(df),
            'avg_events_per_student': len(df) / df['student_id'].nunique() if df['student_id'].nunique() > 0 else 0
        }
        
        if 'event_category' in df.columns:
            stats['event_distribution'] = df['event_category'].value_counts().to_dict()
        
        if 'grade' in df.columns:
            grade_data = df[df['grade'] > 0]['grade']
            if len(grade_data) > 0:
                stats['grade_stats'] = {
                    'mean': float(grade_data.mean()),
                    'median': float(grade_data.median()),
                    'std': float(grade_data.std()),
                    'min': float(grade_data.min()),
                    'max': float(grade_data.max())
                }
        
        return stats
    
    def _analyze_overall_time_patterns(self, df):
        """Анализ общих временных паттернов"""
        patterns = {}
        
        if 'day_of_week' in df.columns:
            weekend_events = len(df[df['day_of_week'] >= 5])
            patterns['weekend_activity'] = weekend_events / len(df) if len(df) > 0 else 0
        
        if 'hour' in df.columns:
            patterns['peak_hours'] = df['hour'].mode().tolist()
        
        return patterns
    
    def _calculate_trend(self, values):
        """Расчет тренда в данных"""
        if len(values) < 2:
            return 0
        
        # Простой линейный тренд
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Нормализация
        if np.std(values) > 0:
            return slope / np.std(values)
        return 0
    
    def save_recommendations(self, recommendations, file_path):
        """Сохранение рекомендаций в файл"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Recommendations saved to {file_path}")