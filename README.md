# 🎓 Learning Path Analyzer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/yourusername/learning-path-analyzer/actions/workflows/analyze.yml/badge.svg)](https://github.com/yourusername/learning-path-analyzer/actions)
[![Code Coverage](https://codecov.io/gh/yourusername/learning-path-analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/learning-path-analyzer)

**Система анализа образовательных траекторий студентов на основе логов LMS** (Moodle, Canvas и др.)

## 📊 Описание проекта

Learning Path Analyzer — это инструмент для анализа активности студентов в системах управления обучением (LMS). Система автоматически анализирует логи, выявляет закономерности между типами активностей и успеваемостью, предоставляет персонализированные рекомендации для оптимизации учебного процесса.

### 🎯 Ключевые возможности:
- **Анализ корреляций** между активностью и успеваемостью
- **Кластеризация студентов** по стилям обучения
- **Визуализация паттернов** активности
- **Персонализированные рекомендации** для студентов
- **Автоматическая генерация отчетов** (HTML, графики)
- **CI/CD пайплайн** для регулярного анализа

## 🏗️ Структура проекта

```
learning-path-analyzer/
├── src/                    # Исходный код
│   ├── data_parser.py     # Парсинг CSV логов
│   ├── analyzer.py        # Анализ корреляций
│   ├── visualizer.py      # Визуализация результатов
│   └── recommender.py     # Генерация рекомендаций
├── tests/                 # Модульные тесты
│   ├── test_parser.py
│   ├── test_analyzer.py
│   └── test_visualizer.py
├── data/                  # Примеры данных
│   ├── sample_log.csv     # Пример входных данных
│   └── sample_large_log.csv
├── .github/workflows/     # GitHub Actions CI/CD
│   └── analyze.yml
├── reports/               # Автоматически генерируемые отчеты
├── docs/                  # Документация
├── requirements.txt       # Зависимости Python
├── config.yaml           # Конфигурация проекта
├── main.py               # Основной скрипт
└── README.md             # Эта документация
```

## 🚀 Быстрый старт

### 1. Клонирование и настройка
```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/learning-path-analyzer.git
cd learning-path-analyzer

# Создайте виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установите зависимости
pip install -r requirements.txt
```

### 2. Запуск примера
```bash
# Запустите анализ на примере данных
python main.py --input data/sample_log.csv --output reports/

# Откройте сгенерированный отчет
# (файл будет в reports/analysis_report.html)
```

### 3. Запуск тестов
```bash
# Запустите все тесты
pytest tests/ -v

# Запустите тесты с покрытием кода
pytest tests/ --cov=src --cov-report=html
```

## 📊 Примеры использования

### Пример 1: Базовый анализ
```python
from src.data_parser import LogParser
from src.analyzer import LearningAnalyzer

# Загрузка и парсинг данных
parser = LogParser(config)
df = parser.parse_csv("data/sample_log.csv")

# Анализ данных
analyzer = LearningAnalyzer(config)
metrics = analyzer.calculate_basic_metrics(df)
correlations = analyzer.calculate_correlations(df)

print(f"Всего студентов: {metrics['total_students']}")
print(f"Всего событий: {metrics['total_events']}")
```

### Пример 2: Командная строка
```bash
# Анализ с фильтрацией по времени
python main.py -i data.csv -o reports/ --timeframe "2024-01"

# Анализ конкретного студента
python main.py -i data.csv -o reports/ --student-id 1001

# Только визуализация
python main.py -i data.csv --visualize-only
```

### Пример 3: Генерация отчетов
```bash
# Полный анализ с HTML отчетом
python main.py --input lms_data.csv --output analysis_results/ --export-json

# Отчет будет содержать:
# - Графики корреляций
# - Кластеры студентов
# - Временные паттерны
# - Рекомендации
```

## 📈 Примеры вывода

### 1. Корреляционный анализ
```
📊 Top correlations with grades:
   forum_participation: +0.42
   regular_activity: +0.38
   content_review: +0.35
   assignment_early_submission: +0.28
```

### 2. Кластеризация студентов
```
🎯 Student clustering results (4 clusters):
   Cluster 0: 25 students (Active Collaborators)
   Cluster 1: 18 students (Independent Learners)
   Cluster 2: 12 students (Assessment-Focused)
   Cluster 3: 8 students (Minimal Engagers)
```

### 3. Рекомендации
```
📋 Recommendations for student ID: 1001
--------------------------------------------------
1. Увеличьте активность на форуме (текущий: 2 поста/неделю)
2. Оптимальное время для занятий: 10:00-12:00
3. Рекомендуется делать больше попыток прохождения тестов
4. Равномерно распределяйте нагрузку в течение недели
```

## 📁 Формат входных данных

### Пример CSV файла:
```csv
student_id,event_type,event_time,module,course,grade,activity_duration
1001,login,2024-01-15 09:30:00,module_1,course_101,,0
1001,assignment_submit,2024-01-15 11:45:00,module_1,course_101,85,120
1001,forum_post,2024-01-16 14:20:00,module_1,course_101,,15
1001,quiz_attempt,2024-01-17 10:15:00,module_1,course_101,92,45
1002,login,2024-01-15 09:35:00,module_1,course_101,,0
1002,assignment_submit,2024-01-16 10:00:00,module_1,course_101,78,90
```

### Поддерживаемые типы событий:
- `login` / `logout` - вход/выход из системы
- `assignment_submit` - сдача задания
- `quiz_attempt` / `exam_start` - тестирование
- `forum_post` / `forum_reply` - обсуждения
- `content_view` / `download` - работа с материалами