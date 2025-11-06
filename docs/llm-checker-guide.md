# Инструкции для LLM-проверяльщика творческого задания

## Назначение

Этот документ содержит инструкции для LLM-системы автоматической проверки творческих программных проектов студентов. Используется в связке с основной рубрикой (15 баллов).

---

## Входные данные для проверки

Проверяльщик получает:
1. **URL репозитория GitHub/GitVerse** студента
2. **README.md** файл проекта
3. **Структура папок** репозитория (output `tree` команды)
4. **Наличие/отсутствие** ключевых файлов (`.gitignore`, `requirements.txt`, `.github/workflows/`)
5. **Статус GitHub Actions** (logs последних запусков)
6. **Основной Python файл** проекта (первый 50 строк и весь код если < 200 строк)

---

## КАТЕГОРИЯ 1: ПОЛЕЗНОСТЬ ЗАДАЧИ (4 балла)

### Алгоритм проверки:

**Шаг 1: Определить тему проекта**
- Прочитать название репозитория и README
- Выделить основную проблему, которую решает проект
- Определить предметную область

**Шаг 2: Оценить релевантность к ИИ в образовании / обработке данных**
- Проверить наличие связей с:
  - Образовательными данными (student performance, learning analytics, assessment)
  - Обработкой текста (NLP, text analysis, language processing)
  - Рекомендательными системами (recommendation engines, personalization)
  - Визуализацией или анализом данных
  - Автоматизацией образовательных процессов

**Шаг 3: Определить практическую ценность**

| Полезность | Индикаторы |
|-------------|-----------|
| **4 балла** | Решает конкретную проблему, видно практическое применение, используются реальные или реалистичные данные, есть потенциал для использования в других проектах |
| **3 балла** | Решает ясную проблему, может быть полезна в образовательном контексте, область применения определена |
| **2 балла** | Имеет образовательную ценность, но применение ограничено или слишком узкое |
| **1 балл** | Задача тривиальна, минимальное практическое применение |
| **0 баллов** | Задача отсутствует или не ясна из README |

### Примеры для референса:

**4 балла (Отличная):**
- Анализатор успеваемости студентов на основе логов LMS
- Система рекомендаций учебных материалов через collaborative filtering
- Парсер образовательных курсов с агрегацией контента по темам
- Инструмент для анализа заданий на предмет когнитивной сложности (через NLP)

**3 балла (Хорошая):**
- Валидатор качества задач по кодингу (но без глубокого анализа)
- Классификатор типов ошибок в студенческом коде
- Генератор вариантов тестовых данных

**1-2 балла (Низкая):**
- Еще один TODO app
- Простой калькулятор
- Counter для чего-либо без контекста

---

## КАТЕГОРИЯ 2: ОФОРМЛЕНИЕ РЕПОЗИТОРИЯ (3 балла)

### Автоматическая проверка (checkpoints):

```python
checks = {
    ".gitignore": {
        "exists": True/False,
        "quality": "complete" | "partial" | "missing",  # Проверить наличие типовых исключений
        "python_patterns": ["*.pyc", "__pycache__", ".env", ".venv"]
    },
    "requirements.txt": {
        "exists": True/False,
        "has_dependencies": len(lines) > 0,
        "valid_format": all(lines match "package==version" or "package>=version")
    },
    "structure": {
        "has_src_folder": True/False,
        "has_tests_folder": True/False,
        "has_data_folder": True/False,
        "root_clutter": number_of_files_in_root,  # Не должно быть > 5-7
        "folders_count": number_of_meaningful_folders  # > 2 = хорошо
    },
    "unwanted_files": [
        ".pyc files in repo": True/False,
        "__pycache__ in repo": True/False,
        ".env in repo": True/False,
        "venv/ in repo": True/False,
        "node_modules/ in repo": True/False
    ]
}
```

### Алгоритм оценки:

**3 балла** — ВСЕ условия выполнены:
- ✓ `.gitignore` существует и покрывает основные типы файлов
- ✓ `requirements.txt` существует и содержит зависимости
- ✓ Структура имеет минимум 3 логичные папки
- ✓ Нет нежелательных файлов (`.pyc`, `__pycache__` и т.д.)

**2 балла** — Большинство условий выполнены:
- ✓ `.gitignore` и `requirements.txt` существуют (может быть неполный .gitignore)
- ~ Структура есть, но может быть лучше организована
- ~ Один-два нежелательных файла могут быть в repo

**1 балл** — Минимальный уровень:
- ~ `.gitignore` или `requirements.txt` неполные или неправильные
- ~ Структура папок не очень логична
- ~ Несколько проблем с организацией

**0 баллов** — Критические ошибки:
- ✗ `.gitignore` или `requirements.txt` отсутствуют
- ✗ Весь код в корне репо
- ✗ Множество нежелательных файлов

---

## КАТЕГОРИЯ 3: РАБОТОСПОСОБНОСТЬ + CI/CD (4 балла)

### Проверка работоспособности кода:

```
workflow_status = get_latest_github_actions_logs()
last_run_status = workflow_status["conclusion"]  # "success" | "failure" | "neutral"

if last_run_status == "success":
    code_score = 3-4  # Базовые баллы за работоспособность
else:
    code_score = 0-1  # Код не работает
```

### Проверка наличия CI/CD:

```
workflow_exists = file_exists(".github/workflows/*.yml")
if workflow_exists:
    workflow_content = read_workflow_file()
    checks = {
        "trigger_on_push": "on: [push]" in workflow_content or "on: push" in workflow_content,
        "trigger_on_pull_request": "on: [pull_request]" in workflow_content,
        "has_tests": "pytest" in workflow_content or "unittest" in workflow_content or "test" in workflow_content.lower(),
        "has_linting": "flake8" in workflow_content or "black" in workflow_content or "pylint" in workflow_content,
        "has_badge": "![" in readme and "actions" in readme.lower()  # Badge для статуса
    }
else:
    workflow_exists = False
```

### Алгоритм оценки:

| Баллы | Условия |
|-------|---------|
| **4** | Код работает (последний CI/CD success) + workflow с PEP8 проверками + тесты + badge |
| **3** | Код работает + workflow существует с PEP8 проверками (но нет тестов) |
| **2** | Код работает + workflow есть, но без полноценных проверок (только базовый lint) |
| **1** | Код работает, но CI/CD отсутствует или не работает |
| **0** | Код не работает (last CI/CD run = failure) |

### Особые случаи:

- Если `last run` имеет статус "skipped" или нет запусков → предположить, что workflow не настроен правильно → максимум 2 балла
- Если в workflow есть `on: schedule:` (запуск по расписанию) → + 0.5 балла за креатив (но это учитывается в категории 5)

---

## КАТЕГОРИЯ 4: КАЧЕСТВО README (2 балла)

### Проверяемые секции:

```
required_sections = [
    "project_description",      # Что это такое, зачем нужно
    "installation",             # Как установить
    "usage",                    # Как использовать
    "structure" or "organization",  # Структура проекта (не обязательно, +)
    "requirements",             # Зависимости, версия Python и т.д.
    "examples"                  # Примеры кода / примеры выполнения
]

quality_metrics = {
    "has_code_examples": count_of_code_blocks > 0,
    "has_output_examples": any("output" in lowercase or ">>>" in content),
    "readability": spelling_errors < 3,
    "structure": number_of_headers > 0
}
```

### Алгоритм оценки:

**2 балла** — Отличный README:
- ✓ Четыре или больше основных секций
- ✓ Примеры кода с ожидаемым выводом
- ✓ Ясно написано, без опечаток
- ✓ Легко следовать инструкциям

**1 балл** — Удовлетворительный README:
- ~ Основные секции есть (описание, установка, использование)
- ~ Примеры есть, но могут быть не полные
- ~ Понятно, но не идеально структурирован

**0 баллов** — Плохой README:
- ✗ README отсутствует или содержит только название
- ✗ Нет информации о том, как использовать проект
- ✗ Примеры отсутствуют или не работают

---

## КАТЕГОРИЯ 5: КРЕАТИВНОЕ ИСПОЛЬЗОВАНИЕ CI/CD (2 балла)

### Признаки креативного использования:

```
creative_patterns = {
    "scheduled_workflows": "cron:" in workflow_content,
    "artifact_uploads": "upload-artifact" in workflow_content,
    "workflow_dispatch": "workflow_dispatch:" in workflow_content,
    "data_updates": any([
        "commit" in workflow_content and ("data" in workflow_content or "generate" in workflow_content),
        "git config" in workflow_content,
        "git add" in workflow_content
    ]),
    "notifications": any([
        "slack" in workflow_content.lower(),
        "mail" in workflow_content.lower(),
        "notify" in workflow_content.lower()
    ]),
    "multi_python_versions": "strategy:" in workflow_content and "python-version:" in workflow_content,
    "deployment": "deploy" in workflow_content.lower() or "github-pages" in workflow_content
}

is_creative = sum(creative_patterns.values()) >= 1
```

### Алгоритм оценки:

**2 балла** — Отличное творческое использование:
- ✓ Workflow делает что-то большее, чем просто проверка кода
- ✓ Используется минимум одна из продвинутых техник (schedule, artifact upload, workflow_dispatch, data updates, deployment)
- ✓ Это имеет практическую ценность для проекта
- ✓ Задокументировано в README

**1 балл** — Хорошее использование:
- ~ Workflow существует и работает
- ~ Делает что-то полезное, но не очень креативно
- ~ Или используется только одна базовая техника (типа schedule), но хорошо реализовано

**0 баллов** — Стандартное использование:
- ✗ Workflow используется только для проверки PEP8
- ✗ GitHub Actions отсутствует вообще

---

## Итоговая логика подсчета баллов

```python
def calculate_score(category_scores):
    """
    Категория 1 (Полезность): 0-4 баллов
    Категория 2 (Оформление): 0-3 баллов
    Категория 3 (Работоспособность + CI/CD): 0-4 баллов
    Категория 4 (README): 0-2 балла
    Категория 5 (Креатив с CI/CD): 0-2 балла
    ---
    Итого: 0-15 баллов
    """
    total = sum(category_scores.values())
    
    rating = {
        (13, 15): "Отлично (A)",
        (11, 12): "Хорошо (B)",
        (9, 10): "Удовлетворительно (C)",
        (7, 8): "Требуется доработка (D)",
        (0, 6): "Не зачтено (F)"
    }
    
    return total, rating[total]
```

---

## Примеры проверки реальных проектов

### Пример 1: "Student Performance Analyzer" (ожидаемый результат: 13-14 баллов)

**Полезность (4 баллы):** ✓ Решает реальную проблему анализа успеваемости студентов
**Оформление (3 балла):** ✓ Хорошая структура с `/src`, `/data`, `/tests`, правильный `.gitignore`
**Работоспособность (4 балла):** ✓ Все тесты проходят, есть workflow с pytest и flake8
**README (2 балла):** ✓ Хорошее описание, примеры с выводом
**Креатив CI/CD (2 балла):** ✓ Workflow с scheduled update данных и artifact upload
**Итого: 15 баллов**

### Пример 2: "Simple TODO App with Tests" (ожидаемый результат: 7-8 баллов)

**Полезность (1 балл):** ~ Задача тривиальна, но есть тесты
**Оформление (2 балла):** ~ Структура есть, но .gitignore неполный
**Работоспособность (3 балла):** ✓ Код работает, есть workflow с proверкой, но нет тестов в workflow
**README (2 балла):** ✓ Нормальное описание
**Креатив CI/CD (0 баллов):** ✗ Только стандартная lint проверка
**Итого: 8 баллов**

---

## Дополнительные подсказки для LLM

1. **При анализе README:** используйте Markdown парсер для подсчета заголовков и кода блоков
2. **При анализе workflow:** ищите ключевые слова (pytest, flake8, schedule, cron, artifact, dispatch)
3. **При анализе структуры:** используйте `os.walk()` логику для подсчета папок и глубины
4. **При анализе работоспособности:** смотрите на последний CI/CD run и его логи, не только на статус
5. **При оценке текста:** проверяйте опечатки только в основных секциях (не считайте ошибки в примерах кода)

