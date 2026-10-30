# Theory of Big Data Processing (ТОБД)

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Distributed Engine](https://img.shields.io/badge/engine-Dask-orange.svg)
![HPC & GPU](https://img.shields.io/badge/acceleration-Numba%20%7C%20CUDA-green.svg)
![Graph Tech](https://img.shields.io/badge/graphs-Neo4j%20%7C%20NetworkX-emerald.svg)
![Code Style](https://img.shields.io/badge/code%20style-PEP%208-black.svg)

> Учебный и исследовательский репозиторий по дисциплине «Теория обработки больших данных». Практические работы по рекомендательным системам, графовым базам данных, сетевому анализу, распределённым вычислениям в Dask и аппаратному ускорению на CPU/GPU с помощью Numba.

---

## О дисциплине

В курсе изучаются ключевые методы, алгоритмы и архитектурные шаблоны обработки больших данных в условиях ограниченных вычислительных ресурсов:

* **Рекомендательные системы:** расчет мер сходства (Cosine Similarity), матричная факторизация (TruncatedSVD), латентные факторы и формирование подборок Top-N рекомендаций.
* **Графовые базы данных и сетевой анализ:** безиндексное связывание данных в Neo4j (Cypher), расчет топологических характеристик и метрик центральности (Betweenness, Closeness, PageRank) в NetworkX.
* **Распределённые вычисления (Dask):** ленивые вычисления (DAG), блочная обработка таблиц и массивов (Out-of-Core), профилирование вычислительной стоимости операций и управление кэшированием (`persist`).
* **Распределённое машинное обучение:** обучение моделей поблочно через Dask-ML, оптимизаторы распределенной оптимизации (ADMM/L-BFGS) и параллельный подбор гиперпараметров по сетке.
* **Высокопроизводительные вычисления (HPC):** динамическая JIT-компиляция в машинный код через LLVM, обход Global Interpreter Lock (GIL), многопоточный параллелизм (`prange`), векторизация NumPy ufunc и параллельные вычисления на видеокартах NVIDIA (Numba CUDA).

---

## Tech Stack

* **Language:** Python 3.10+
* **Distributed Computing:** Dask (DataFrame, Array, Delayed, Distributed, Dask-ML)
* **High Performance Computing:** Numba (JIT, parallel, CUDA)
* **Graph Technologies:** Neo4j (Cypher), NetworkX
* **Data Processing & Storage:** Pandas, NumPy, Apache Parquet, PyArrow, Pillow
* **Infrastructure & Tools:** Visual Studio Code, Git, Docker & Docker Compose

---

## Repository Structure

```text
Big_data/
├── materials/                              # Методические указания кафедры (PDF, DOCX)
│   ├── 01_PZ_8_Rekomendatelnye_sistemy.pdf
│   ├── 02_4_1_Grafovye_BD.pdf
│   ├── 03_4_2_Analiz_seti_sredstvami_NetworkX.docx
│   ├── 04_PZ_Skolzaschaya_Dask.docx
│   ├── 05_4_3_ML_Dask.pdf
│   ├── 06_PZ_4_1.docx
│   ├── 07_PZ_4_2.docx
│   ├── 08_PZ_5_1_Numba.docx
│   └── 09_PZ_5_2_GPU.pdf
├── PR/                                     # Практические модули курса
│   ├── 01_recomendation_systems/           # ПР 01: Рекомендательные алгоритмы (Cosine, SVD)
│   │   ├── data/
│   │   │   └── data.csv
│   │   ├── README.md
│   │   ├── solution.py
│   │   └── task.md
│   ├── 02_graph_databases/                 # ПР 02: Графовые базы данных (Neo4j, Cypher)
│   │   ├── docker-compose.yml
│   │   ├── queries.cypher
│   │   ├── README.md
│   │   └── task.md
│   ├── 03_network_analysis/                # ПР 03: Анализ графовых сетей (NetworkX)
│   │   ├── data/
│   │   │   └── variant_1.csv
│   │   ├── network_metrics.csv
│   │   ├── README.md
│   │   ├── solution.py
│   │   └── task.md
│   ├── 04_distributed_dask/                # ПР 04: Скользящая средняя урожайности и профилирование Dask
│   │   ├── data/
│   │   │   └── urozhaynost_zernovykh_30_regionov_100_let.xlsx
│   │   ├── images/
│   │   │   └── crop_yield_moving_average.png
│   │   ├── README.md
│   │   ├── solution.py
│   │   └── task.md
│   ├── 05_ml_dask/                         # ПР 05: Распределенный ML кредитного скоринга (Dask-ML)
│   │   ├── data/
│   │   │   └── logisticheskaya_regressia_dannye.xlsx
│   │   ├── images/
│   │   │   └── model_evaluation.png
│   │   ├── README.md
│   │   ├── solution.py
│   │   └── task.md
│   ├── 06_dask_partitions/                 # ПР 06: Секционирование, blocksize и стоимость операций
│   │   ├── data/
│   │   │   └── FY2016-STC-Category-Table.csv
│   │   ├── images/
│   │   │   └── property_tax_distribution.png
│   │   ├── README.md
│   │   ├── solution.py
│   │   └── task.md
│   ├── 07_dask_distributed/                # ПР 07: Архитектура dask.distributed, Parquet и фракталы
│   │   ├── data/
│   │   │   └── taxes2.parquet/
│   │   ├── images/
│   │   │   ├── mandelbrot.png
│   │   │   └── mandelbrot_benchmark.png
│   │   ├── README.md
│   │   ├── solution.py
│   │   └── task.md
│   ├── 08_numba_hpc/                       # ПР 08: Компилятор Numba, обход GIL, JIT и prange
│   │   ├── images/
│   │   │   ├── mandelbrot_numba.png
│   │   │   └── numba_benchmark.png
│   │   ├── README.md
│   │   ├── solution.py
│   │   └── task.md
│   └── 09_numba_cuda/                      # ПР 09: Ускорение на GPU NVIDIA (Numba CUDA Kernels)
│       ├── images/
│       │   ├── gpu_acceleration_benchmark.png
│       │   └── mandelbrot_gpu.png
│       ├── README.md
│       ├── solution.py
│       └── task.md
├── .gitignore                              # Исключение venv, кэша __pycache__, чекпоинтов и логов
├── LICENSE
└── README.md                               # Главная витрина репозитория
```
---

## Practical Modules Matrix

| Модуль | Технологии | Реализованная задача | Входные данные | Статус |
| :--- | :--- | :--- | :--- | :---: |
| **01. Рекомендации** | Pandas, Scikit-Learn | Сходство (Cosine), матричная факторизация (TruncatedSVD) | `data.csv` | 🟢 Done |
| **02. Графовые БД** | Neo4j, Cypher, Docker | DDL узлов/связей, многошаговый поиск кратчайших путей | Предметная область | 🟢 Done |
| **03. Анализ сетей** | NetworkX, Matplotlib | Метрики связности, центральности (Betweenness), экспорт метрик | `variant_1.csv` | 🟢 Done |
| **04. Dask Series** | Dask DataFrame | Параллельный расчет скользящего среднего, устранение OOM | `urozhaynost_...xlsx` | 🟢 Done |
| **05. Dask ML** | Dask-ML, Distributed | Распределенная логистическая регрессия (ADMM), подбор $C$ | `logisticheskaya_...xlsx` | 🟢 Done |
| **06. Секционирование** | Dask DataFrame | Анализ DAG, категориальные индексы, blocksize, `persist()` | `FY2016-STC-...csv` | 🟢 Done |
| **07. Distributed** | dask.distributed, Parquet | Планировщик, Dashboard, Parquet, Мандельброт на Dask Array | Сетка координат | 🟢 Done |
| **08. Numba HPC** | Numba JIT, LLVM | Ускорение в 100+ раз, отключение GIL (`nogil`), `prange`, ufunc | Комплексная сетка | 🟢 Done |
| **09. Numba CUDA (GPU)** | Numba CUDA, GPU Kernels | Сетки, блоки, потоки, device-функции, генерация фрактала на GPU | VRAM графической карты | 🟢 Done |

---

## Assignment Workflow & Standards

Каждая практическая работа организована в виде изолированного пакета по единому регламенту:

1. **Requirements Specification (`task.md`):** Условия задания, теоретические концепции и контрольные вопросы кафедры в формате Markdown.
2. **Environment Isolation (`venv`):** Все вычисления и зависимости изолированы внутри виртуального окружения Python.
3. **Clean Code (`solution.py` / Cypher):**
   - Соблюдение стандартов оформления кода по **PEP 8** (именование в `snake_case`, единые отступы, структурированные docstrings).
   - Строгая типизация параметров (`type hinting`).
   - Использование `pathlib.Path` для кроссплатформенной адресации входных наборов данных и каталогов артефактов (`images/`).
4. **Report & Documentation (`README.md`):** Структурированный отчёт, архитектурные схемы потоков данных, сводные таблицы замеров, графики и ответы на вопросы защиты.
5. **Version Control:** Фиксация изменений осмысленными коммитами по стандарту **Conventional Commits** (`feat:`, `docs:`, `fix:`, `refactor:`).

---

## Getting Started

### 1. Активация виртуального окружения
* **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
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
python PR/01_recomendation_systems/solution.py
python PR/03_network_analysis/solution.py
python PR/04_distributed_dask/solution.py
python PR/05_ml_dask/solution.py
python PR/06_dask_partitions/solution.py
python PR/07_dask_distributed/solution.py
python PR/08_numba_hpc/solution.py
python PR/09_numba_cuda/solution.py
```