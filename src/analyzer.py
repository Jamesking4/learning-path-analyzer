"""
Модуль для анализа данных и выявления закономерностей
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime


class LearningAnalyzer:
    """Анализатор образовательных данных"""

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def calculate_basic_metrics(self, df):
        """Расчет базовых метрик"""
        metrics = {
            "total_students": df["student_id"].nunique() if not df.empty else 0,
            "total_events": len(df) if not df.empty else 0,
            "avg_events_per_student": (
                len(df) / df["student_id"].nunique()
                if not df.empty and df["student_id"].nunique() > 0
                else 0
            ),
        }

        # Обработка time_range для пустых данных
        if (
            not df.empty
            and "event_time" in df.columns
            and df["event_time"].notna().any()
        ):
            metrics["time_range"] = {
                "start": df["event_time"].min().strftime("%Y-%m-%d"),
                "end": df["event_time"].max().strftime("%Y-%m-%d"),
            }
        else:
            metrics["time_range"] = {"start": "N/A", "end": "N/A"}

        if "event_category" in df.columns and not df.empty:
            metrics["event_distribution"] = (
                df["event_category"].value_counts().to_dict()
            )
        else:
            metrics["event_distribution"] = {}

        if "grade" in df.columns and not df.empty:
            grade_data = df[df["grade"] > 0]["grade"]
            if len(grade_data) > 0:
                metrics["grade_stats"] = {
                    "mean": float(grade_data.mean()),
                    "median": float(grade_data.median()),
                    "std": float(grade_data.std()),
                    "min": float(grade_data.min()),
                    "max": float(grade_data.max()),
                }

        return metrics

    def calculate_correlations(self, df):
        """Расчет корреляций между активностью и успеваемостью"""
        if "grade" not in df.columns:
            print("⚠️ No grade data available for correlation analysis")
            return pd.DataFrame()

        # Агрегация данных по студентам
        student_data = self._prepare_student_data(df)

        # Выбор числовых колонок для корреляции
        numeric_cols = student_data.select_dtypes(include=[np.number]).columns

        # Расчет матрицы корреляций
        correlation_matrix = student_data[numeric_cols].corr(method="pearson")

        # Фильтрация корреляций с оценками
        if "grade_mean" in correlation_matrix.columns:
            grade_correlations = correlation_matrix["grade_mean"].sort_values(
                ascending=False
            )
            print("\n📊 Top correlations with grades:")
            for feature, corr in grade_correlations.head(10).items():
                if feature != "grade_mean" and abs(corr) > 0.1:
                    print(f"   {feature}: {corr:.3f}")

        return correlation_matrix

    def _prepare_student_data(self, df):
        """Подготовка данных для анализа на уровне студентов"""
        # Агрегация по студентам
        agg_funcs = {"event_time": "count", "grade": ["mean", "max", "min", "std"]}

        # Добавление агрегаций по категориям событий
        if "event_category" in df.columns:
            event_dummies = pd.get_dummies(df["event_category"], prefix="event")
            df = pd.concat([df, event_dummies], axis=1)

            for col in event_dummies.columns:
                agg_funcs[col] = "sum"

        # Агрегация данных
        student_data = df.groupby("student_id").agg(agg_funcs)

        # Упрощение мультииндекса
        student_data.columns = [
            "_".join(col).strip() for col in student_data.columns.values
        ]

        # Добавление временных метрик
        student_data["activity_days"] = df.groupby("student_id")["date"].nunique()

        # Нормализация данных
        numeric_cols = student_data.select_dtypes(include=[np.number]).columns
        student_data[numeric_cols] = self.scaler.fit_transform(
            student_data[numeric_cols]
        )

        return student_data.reset_index()

    def cluster_students(self, df, n_clusters=None):
        """Кластеризация студентов по стилям обучения"""
        if n_clusters is None:
            n_clusters = self.config["analysis"]["clustering_n_clusters"]

        # Подготовка данных
        student_data = self._prepare_student_data(df)

        if len(student_data) < n_clusters:
            print(f"⚠️ Not enough students for {n_clusters} clusters")
            return pd.DataFrame()

        # Выбор признаков для кластеризации
        feature_cols = [
            col
            for col in student_data.columns
            if col not in ["student_id"]
            and student_data[col].dtype in [np.float64, np.int64]
        ]

        # Кластеризация K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        student_data["cluster"] = kmeans.fit_predict(student_data[feature_cols])

        # Анализ кластеров
        cluster_stats = student_data.groupby("cluster")[feature_cols].mean()

        print(f"\n🎯 Student clustering results ({n_clusters} clusters):")
        for cluster_id in range(n_clusters):
            cluster_size = (student_data["cluster"] == cluster_id).sum()
            print(f"   Cluster {cluster_id}: {cluster_size} students")

        return student_data[["student_id", "cluster"]]

    def analyze_time_patterns(self, df):
        """Анализ временных паттернов активности"""
        patterns = {}

        # Почасовое распределение
        df["hour"] = df["event_time"].dt.hour
        hourly_pattern = df.groupby("hour").size()
        patterns["hourly_distribution"] = hourly_pattern.to_dict()

        # Распределение по дням недели
        df["day_of_week"] = df["event_time"].dt.dayofweek
        daily_pattern = df.groupby("day_of_week").size()
        patterns["daily_distribution"] = daily_pattern.to_dict()

        # Активность по типам событий в течение дня
        if "event_category" in df.columns:
            event_hourly = (
                df.groupby(["hour", "event_category"]).size().unstack(fill_value=0)
            )
            patterns["event_by_hour"] = event_hourly.to_dict()

        return patterns

    def identify_learning_patterns(self, df):
        """Выявление успешных учебных паттернов"""
        successful_students = self._identify_successful_students(df)

        if successful_students.empty:
            return {}

        # Анализ паттернов успешных студентов
        patterns = {}

        # Получаем ID успешных студентов
        successful_ids = successful_students.index.tolist()

        # Фильтруем исходные данные
        successful_df = df[df["student_id"].isin(successful_ids)]

        if successful_df.empty:
            return patterns

        # Частота активностей (используем count вместо event_time_count)
        patterns["activity_frequency"] = len(successful_df) / len(successful_ids)

        # Распределение по категориям событий
        if "event_category" in successful_df.columns:
            event_dist = successful_df["event_category"].value_counts(normalize=True)
            patterns["event_distribution"] = event_dist.to_dict()

        # Временные паттерны
        if "hour" in successful_df.columns:
            patterns["preferred_hours"] = successful_df["hour"].mode().tolist()

        if "day_of_week" in successful_df.columns:
            patterns["preferred_days"] = successful_df["day_of_week"].mode().tolist()

        # Длительность активности
        if "activity_duration" in successful_df.columns:
            patterns["avg_duration"] = successful_df["activity_duration"].mean()

        return patterns

    def _identify_successful_students(self, df, threshold=None):
        """Идентификация успешных студентов"""
        if threshold is None:
            threshold = self.config["analysis"]["min_grade_threshold"]

        if "grade" not in df.columns:
            return pd.DataFrame()

        # Средняя оценка по студенту
        student_grades = (
            df[df["grade"] > 0].groupby("student_id")["grade"].agg(["mean", "count"])
        )

        # Фильтрация студентов с достаточным количеством оценок
        min_grades = 3  # минимальное количество оценок
        valid_students = student_grades[student_grades["count"] >= min_grades]

        # Успешные студенты
        successful = valid_students[valid_students["mean"] >= threshold]

        return successful.reset_index()

    def save_results(self, results, file_path):
        """Сохранение результатов анализа"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Results saved to {file_path}")
