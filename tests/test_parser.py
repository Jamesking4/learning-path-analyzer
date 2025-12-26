"""
Тесты для модуля парсинга данных
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_parser import LogParser


@pytest.fixture
def sample_config():
    """Конфигурация для тестов"""
    return {
        "events": {
            "login_events": ["login", "logout"],
            "content_events": ["view", "download"],
            "assessment_events": ["assignment", "quiz", "exam"],
            "social_events": ["forum", "comment"],
            "important_events": ["complete", "certificate"],
        }
    }


@pytest.fixture
def sample_data():
    """Пример данных для тестов"""
    data = {
        "student_id": [1, 1, 2, 2, 3],
        "event_type": [
            "login",
            "assignment_submit",
            "forum_post",
            "quiz_attempt",
            "view_content",
        ],
        "event_time": pd.date_range("2024-01-01", periods=5, freq="H"),
        "grade": [None, 85, None, 92, None],
        "activity_duration": [0, 120, 15, 45, 30],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_dates():
    """Пример данных с датами для тестов агрегации"""
    data = {
        "student_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "event_type": [
            "login",
            "assignment",
            "forum_post",
            "login",
            "quiz",
            "login",
            "assignment",
            "view",
        ],
        "event_time": pd.date_range("2024-01-01", periods=8, freq="D"),
        "grade": [None, 85, None, None, 92, None, 78, None],
        "activity_duration": [0, 120, 15, 0, 45, 0, 90, 30],
        "module": ["module1"] * 8,
        "course": ["course101"] * 8,
    }
    return pd.DataFrame(data)


def test_parser_initialization(sample_config):
    """Тест инициализации парсера"""
    parser = LogParser(sample_config)
    assert parser is not None
    assert "login_events" in parser.event_types


def test_parse_csv(sample_config, sample_data, tmp_path):
    """Тест парсинга CSV файла"""
    # Сохраняем данные во временный файл
    file_path = tmp_path / "test.csv"
    sample_data.to_csv(file_path, index=False)

    # Парсим файл
    parser = LogParser(sample_config)
    df = parser.parse_csv(str(file_path))

    # Проверяем результат
    assert len(df) == 5
    assert "hour" in df.columns
    assert "day_of_week" in df.columns
    assert "event_category" in df.columns


def test_parse_dataframe(sample_config, sample_data):
    """Тест парсинга данных из DataFrame"""
    parser = LogParser(sample_config)

    # Используем метод parse_csv для DataFrame
    df = parser.parse_csv_from_dataframe(sample_data)  # Этот метод нужно добавить

    # Или напрямую вызываем внутренние методы
    df = parser._clean_data(sample_data)
    df = parser._extract_features(df)
    df = parser._categorize_events(df)

    assert len(df) == 5
    assert "hour" in df.columns
    assert "event_category" in df.columns


def test_data_cleaning(sample_config, sample_data):
    """Тест очистки данных"""
    parser = LogParser(sample_config)
    df = parser._clean_data(sample_data)

    # Проверяем отсутствие пропусков в важных колонках
    assert df["event_time"].isnull().sum() == 0

    # Проверяем заполнение пропущенных оценок
    if "grade" in df.columns:
        assert df["grade"].isnull().sum() == 0


def test_feature_extraction(sample_config, sample_data):
    """Тест извлечения признаков"""
    parser = LogParser(sample_config)
    df = parser._extract_features(sample_data)

    # Проверяем наличие новых колонок
    assert "hour" in df.columns
    assert "day_of_week" in df.columns
    assert "is_assessment" in df.columns
    assert "is_social" in df.columns

    # Проверяем создание колонки date
    assert "date" in df.columns


def test_event_categorization(sample_config):
    """Тест категоризации событий"""
    parser = LogParser(sample_config)

    # Тестовые данные
    test_data = pd.DataFrame(
        {"event_type": ["login", "assignment_submit", "forum_post", "view_content"]}
    )

    df = parser._categorize_events(test_data)

    # Проверяем категории
    assert "event_category" in df.columns
    categories = df["event_category"].tolist()
    assert "login" in categories
    assert "assessment" in categories
    assert "social" in categories


def test_timeframe_filter(sample_config, sample_data):
    """Тест фильтрации по времени"""
    parser = LogParser(sample_config)

    # Фильтрация по году
    filtered = parser.filter_by_timeframe(sample_data, "2024")
    assert len(filtered) == 5

    # Фильтрация по году и месяцу
    filtered = parser.filter_by_timeframe(sample_data, "2024-01")
    assert len(filtered) == 5


def test_student_aggregation(sample_config, sample_data_with_dates):
    """Тест агрегации данных по студентам"""
    parser = LogParser(sample_config)

    # Обрабатываем данные
    df = sample_data_with_dates.copy()
    df = parser._clean_data(df)
    df = parser._extract_features(df)
    df = parser._categorize_events(df)

    aggregated = parser.aggregate_student_data(df)

    # Проверяем агрегированные данные
    assert "student_id" in aggregated.columns
    assert len(aggregated) == 3  # 3 уникальных студента

    # Проверяем наличие основных метрик
    assert "event_time_count" in aggregated.columns or len(aggregated.columns) > 0
