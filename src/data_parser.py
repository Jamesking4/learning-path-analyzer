"""
Модуль для парсинга и обработки логов LMS
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os


class LogParser:
    """Парсер логов LMS систем"""

    def __init__(self, config):
        self.config = config
        self.event_types = config["events"]

    def parse_csv(self, file_path):
        """Парсинг CSV файла с логами"""
        try:
            # Чтение CSV файла
            df = pd.read_csv(file_path, parse_dates=["event_time"])

            # Проверка необходимых колонок
            required_columns = ["student_id", "event_type", "event_time"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Очистка данных
            df = self._clean_data(df)

            # Извлечение дополнительных признаков
            df = self._extract_features(df)

            # Категоризация событий
            df = self._categorize_events(df)

            print(f"✅ Successfully parsed {len(df)} records")
            return df

        except Exception as e:
            print(f"❌ Error parsing CSV: {e}")
            raise

    def _clean_data(self, df):
        """Очистка и предобработка данных"""
        # Удаление дубликатов
        df = df.drop_duplicates()

        # Заполнение пропущенных значений
        if "grade" in df.columns:
            df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
            df["grade"] = df["grade"].fillna(0)

        if "activity_duration" in df.columns:
            df["activity_duration"] = pd.to_numeric(
                df["activity_duration"], errors="coerce"
            )
            df["activity_duration"] = df["activity_duration"].fillna(0)

        # Удаление записей с некорректными датами
        df = df[df["event_time"].notna()]

        return df

    def _extract_features(self, df):
        """Извлечение дополнительных признаков из данных"""
        # Временные признаки
        df["hour"] = df["event_time"].dt.hour
        df["day_of_week"] = df["event_time"].dt.dayofweek
        df["day_name"] = df["event_time"].dt.day_name()
        df["week_number"] = df["event_time"].dt.isocalendar().week
        df["month"] = df["event_time"].dt.month
        df["date"] = df["event_time"].dt.date

        # Признаки активности
        df["is_assessment"] = df["event_type"].str.contains(
            "assignment|quiz|exam", case=False
        )
        df["is_social"] = df["event_type"].str.contains(
            "forum|comment|reply", case=False
        )
        df["is_content"] = df["event_type"].str.contains(
            "view|download|read", case=False
        )

        # Длительность активности (если доступна)
        if "activity_duration" in df.columns:
            df["activity_category"] = pd.cut(
                df["activity_duration"],
                bins=[0, 5, 30, 60, 120, float("inf")],
                labels=["very_short", "short", "medium", "long", "very_long"],
            )

        return df

    def _categorize_events(self, df):
        """Категоризация событий по типам"""

        def categorize_event(event_type):
            event_lower = str(event_type).lower()

            if any(
                login_event in event_lower
                for login_event in self.event_types["login_events"]
            ):
                return "login"
            elif any(
                content_event in event_lower
                for content_event in self.event_types["content_events"]
            ):
                return "content_interaction"
            elif any(
                assessment_event in event_lower
                for assessment_event in self.event_types["assessment_events"]
            ):
                return "assessment"
            elif any(
                social_event in event_lower
                for social_event in self.event_types["social_events"]
            ):
                return "social"
            elif any(
                important_event in event_lower
                for important_event in self.event_types["important_events"]
            ):
                return "milestone"
            else:
                return "other"

        df["event_category"] = df["event_type"].apply(categorize_event)
        return df

    def filter_by_timeframe(self, df, timeframe):
        """Фильтрация данных по временному периоду"""
        try:
            if "-" in timeframe:
                # Формат YYYY-MM
                year, month = map(int, timeframe.split("-"))
                mask = (df["event_time"].dt.year == year) & (
                    df["event_time"].dt.month == month
                )
                return df[mask]
            else:
                # Только год
                year = int(timeframe)
                return df[df["event_time"].dt.year == year]
        except:
            print(f"⚠️ Invalid timeframe format: {timeframe}. Using all data.")
            return df

    def aggregate_student_data(self, df):
        """Агрегация данных по студентам"""
        student_stats = (
            df.groupby("student_id")
            .agg(
                {
                    "event_time": ["count", "min", "max"],
                    "grade": ["mean", "max", "min", "count"],
                    "activity_duration": (
                        "sum" if "activity_duration" in df.columns else "count"
                    ),
                }
            )
            .round(2)
        )

        # Упрощение мультииндекса
        student_stats.columns = [
            "_".join(col).strip() for col in student_stats.columns.values
        ]
        student_stats = student_stats.reset_index()

        # Расчет дополнительных метрик
        student_stats["activity_range_days"] = (
            pd.to_datetime(student_stats["event_time_max"])
            - pd.to_datetime(student_stats["event_time_min"])
        ).dt.days

        # Распределение по категориям событий
        event_counts = (
            df.groupby(["student_id", "event_category"]).size().unstack(fill_value=0)
        )
        student_stats = student_stats.merge(event_counts, on="student_id", how="left")

        return student_stats

    def parse_csv_from_dataframe(self, df):
        """Парсинг данных из DataFrame (для тестов)"""
        # Очистка данных
        df = self._clean_data(df)

        # Извлечение дополнительных признаков
        df = self._extract_features(df)

        # Категоризация событий
        df = self._categorize_events(df)

        return df
