import time
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Определение путей к данным
current_dir = Path(__file__).resolve().parent
excel_path = current_dir / "data" / "urozhaynost_zernovykh_30_regionov_100_let.xlsx"

# Если файл лежит в корне папки задания
if not excel_path.exists():
    excel_path = current_dir / "urozhaynost_zernovykh_30_regionov_100_let.xlsx"

print(f"Поиск файла данных: {excel_path}")


# 2. Задача 1.1: Расчет скользящей средней по урожайности регионов
def process_crop_yield_series(file_path: Path) -> None:
    """Загрузка урожайности, расчет скользящего среднего в Dask и сохранение графика."""
    print("\n" + "=" * 60)
    print("ЗАДАЧА 1.1: АНАЛИЗ УРОЖАЙНОСТИ И СКОЛЬЗЯЩЕЙ СРЕДНЕЙ")
    print("=" * 60)

    # Чтение Excel через Pandas и конвертация в Dask DataFrame
    pdf = pd.read_excel(file_path)
    print(f"Файл успешно прочитан. Размерность таблицы: {pdf.shape}")

    # Первая колонка — временной индекс/год, остальные — регионы
    year_col = pdf.columns[0]
    region_cols = [c for c in pdf.columns if c != year_col]
    target_region = region_cols[0]

    print(f"Анализ временного ряда для региона: {target_region}")

    # Превращаем в Dask DataFrame с 2 партициями
    ddf = dd.from_pandas(pdf[[year_col, target_region]], npartitions=2)

    # Базовые операции Dask: фильтрация и ленивое скользящее среднее (окно 5 лет)
    window_size = 5
    ddf["moving_avg"] = ddf[target_region].rolling(window=window_size).mean()

    # Единая материализация результата через .compute()
    result_df = ddf.compute()

    # Отрисовка графика
    plt.figure(figsize=(11, 5))
    plt.plot(
        result_df[year_col],
        result_df[target_region],
        label="Фактическая урожайность (ц/га)",
        alpha=0.6,
        color="tab:blue"
    )
    plt.plot(
        result_df[year_col],
        result_df["moving_avg"],
        label=f"Скользящая средняя (окно={window_size} лет)",
        linewidth=2.2,
        color="tab:red"
    )

    plt.title(f"Динамический ряд урожайности зерновых: {target_region}")
    plt.xlabel("Год")
    plt.ylabel("Урожайность, ц/га")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Сохранение в существующую папку images
    images_dir = current_dir / "images"
    images_dir.mkdir(exist_ok=True)
    plot_output = images_dir / "crop_yield_moving_average.png"

    plt.savefig(plot_output, dpi=150)
    plt.close()
    print(f"График динамического ряда успешно сохранен в: {plot_output}")


# 3. Задача 1.2: Профилирование и устранение антипаттернов Dask
def run_profiling_benchmark() -> None:
    """Сравнение времени работы антипаттерна (compute внутри) и оптимизированной версии."""
    print("\n" + "=" * 60)
    print("ЗАДАЧА 1.2: ПРОФИЛИРОВАНИЕ И СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ DASK")
    print("=" * 60)

    # Генерация синтетического набора данных
    n_rows = 150_000
    dates = pd.date_range("2010-01-01", periods=n_rows, freq="min")
    raw_data = pd.DataFrame({
        "timestamp": dates,
        "metric_val": np.random.normal(50, 10, n_rows) + np.sin(np.arange(n_rows) * 0.05) * 5
    })

    test_csv = current_dir / "benchmark_data.csv"
    raw_data.to_csv(test_csv, index=False)
    print(f"Сгенерирован тестовый набор данных: {n_rows:,} строк")

    # --- Подход 1: Неоптимальный код с антипаттерном ---
    start_time = time.time()
    dask_df_bad = dd.read_csv(test_csv)
    # Антипаттерн: вызов .compute() внутри цепочки преобразований
    dask_df_bad["rolling_val"] = dask_df_bad["metric_val"].rolling(window=30).mean().compute()
    _ = dask_df_bad["metric_val"].compute()
    time_bad = time.time() - start_time
    print(f"1. Неоптимальный код (compute внутри цепочки): {time_bad:.4f} сек")

    # --- Подход 2: Оптимизированный ленивый пайплайн ---
    start_time = time.time()
    dask_df_good = dd.read_csv(test_csv)
    # Ленивое формирование графа задач
    dask_df_good["rolling_val"] = dask_df_good["metric_val"].rolling(window=30).mean()
    # Единый .compute() в самом конце
    _ = dask_df_good.compute()
    time_good = time.time() - start_time
    print(f"2. Оптимизированный код (единый compute в конце):  {time_good:.4f} сек")

    # Расчет ускорения
    speedup = ((time_bad - time_good) / time_bad) * 100
    print("\n" + "-" * 40)
    print(f"Итоговое ускорение: {speedup:.1f}%")
    print(f"Экономия процессорного времени: {time_bad - time_good:.4f} сек")
    print("-" * 40)

    # Удаление временного тестового файла
    if test_csv.exists():
        test_csv.unlink()


def main() -> None:
    if excel_path.exists():
        process_crop_yield_series(excel_path)
    else:
        print(f"Файл {excel_path} не найден. Пропуск Задачи 1.1.")

    run_profiling_benchmark()


if __name__ == "__main__":
    main()