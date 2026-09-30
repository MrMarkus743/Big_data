# Theory of Big Data Processing

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Distributed Engine](https://img.shields.io/badge/engine-Dask-orange.svg)
![Graph Tech](https://img.shields.io/badge/graphs-Neo4j%20%7C%20NetworkX-emerald.svg)
![Code Style](https://img.shields.io/badge/code%20style-PEP%208-black.svg)

> Учебный репозиторий по дисциплине «Теория обработки больших данных». Практические работы по рекомендательным алгоритмам, графовым базам данных, сетевому анализу и параллельным вычислениям.

---

## О дисциплине

В курсе изучаются основные подходы к анализу и обработке данных в условиях ограничений памяти и вычислительных ресурсов:

* **Рекомендательные системы:** расчет сходства пользователей и объектов, формирование подборок Top-N рекомендаций.
* **Графовые базы данных и сети:** хранение и выборка связанных данных без тяжелых JOIN-запросов (Neo4j / Cypher), анализ метрик графов в NetworkX.
* **Параллельная обработка:** работа с таблицами через Dask (ленивые вычисления, разбиение данных на партиции).
* ...

---

## Tech Stack

* **Language:** Python 3.10+
* **Distributed Computing:** Dask (DataFrame, Delayed)
* **Graph Technologies:** Neo4j (Cypher), NetworkX
* **Data Processing:** Pandas, NumPy
* **Tools:** Visual Studio Code, Git, Docker 

---

## Repository Structure

```text
Big_data/
├── 01_recomendation_systems/       # Практика 1: Рекомендательные системы
│   ├── data/                      
│   │   └── data.csv               # Датасет оценок
│   ├── README.md                  # Отчет и разбор метрик качества
│   ├── solution.py                # Скрипт расчета рекомендаций на Python
│   └── task.md                    # Условия задания из методички
├── 02_graph_databases/            # Практика 2: Графовые базы данных (Neo4j)
│   ├── queries.cypher             # DDL и аналитические запросы на языке Cypher
│   ├── README.md                  # Отчет, схема графа и ответы к защите
│   └── task.md                    # Задание к лабораторной работе
├── 03_network_analysis/           # Практика 3: Анализ графовых сетей (NetworkX)
│   ├── data/                      # Входной граф взаимодействий
│   │   └── variant_1.csv
│   ├── network_metrics.csv        # Рассчитанные топологические метрики по узлам
│   ├── README.md                  # Отчет и анализ ключевых вершин сети
│   ├── solution.py                # Скрипт расчета центральностей и визуализации
│   └── task.md                    # Задание к практическому занятию
├── 04_distributed_dask/           # Практика 4: Параллельные вычисления в Dask
│   ├── data/                      # Временные ряды урожайности
│   │   └── urozhaynost_zernovykh_30_regionov_100_let.xlsx
│   ├── images/                    # Визуализация результатов
│   │   └── crop_yield_moving_average.png
│   ├── README.md                  # Отчет по скользящему окну и профилированию
│   ├── solution.py                # Пайплайн обработки и бенчмарк скорости
│   └── task.md                    # Условия расчета и оптимизации кода
├── materials/                     # Исходные файлы кафедры (PDF, DOCX)
│   ├── 01_PZ_8_Rekomendatelnye_sistemy.pdf
│   ├── 02_4_1_Grafovye_BD.pdf
│   ├── 03_4_2_Analiz_seti_sredstvami_NetworkX.docx
│   └── 04_PZ_Skolzaschaya_Dask.docx
├── .gitignore                     # Исключение виртуального окружения (venv), кэша и логов
└── README.md                      # Главная документация репозитория
```

---

## Практические модули

| Модуль | Технологии | Что делаем | Входные данные |
| :--- | :--- | :--- | :--- |
| **01. Рекомендации** | Python, Pandas, Scikit-Learn | Считаем сходство (Cosine) и факторизацию (SVD) | `data.csv` |
| **02. Графовые БД** | Neo4j, Cypher | Создаем узлы, связи и делаем многошаговые выборки | Предметная область |
| **03. Анализ сетей** | NetworkX, Matplotlib | Считаем метрики связности, центральности и строим граф | `variant_1.csv` |
| **04. Dask** | Dask DataFrame | Параллельный расчет скользящего среднего и бенчмарк | `.xlsx` файл урожайности |

---

## Assignment Workflow & Standards

Каждая практическая работа выполняется по единому регламенту:

1. **Requirements Specification (`task.md`):** Условия задания из методички переносятся в Markdown-формат.
2. **Environment Isolation (`venv`):** Все скрипты запускаются в изолированном виртуальном окружении Python.
3. **Clean Code (`solution.py` / Cypher):**
   - Соблюдение стандартов оформления кода по **PEP 8** (имена переменных в `snake_case`, отступы в 4 пробела, форматирование строк).
   - Аннотация типов данных (`type hinting`) и декомпозиция задач на функции.
   - Использование относительных путей для чтения датасетов из подпапки `data/`.
4. **Report & Documentation (`README.md`):** Отчёт с результатами работы программы, таблицами, выводами и сохраненными графиками.
5. **Version Control:** Фиксация изменений по стандарту **Conventional Commits** (`feat:`, `docs:`, `fix:`, `refactor:`).

---

## Getting Started

### 1. Активация виртуального окружения
* **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 2. Запуск контейнера Neo4j (для практики 02)
```powershell
docker run -d --name tobd-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:community
```
* Интерфейс базы открывается в браузере: `http://localhost:7474`

### 3. Запуск практических работ
Запуск любого расчетного скрипта выполняется из корня проекта:
```powershell
python 01_recomendation_systems/solution.py
python 03_network_analysis/solution.py
python 04_distributed_dask/solution.py
```