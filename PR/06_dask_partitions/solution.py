"""Практическое занятие № 4.1: Секционирование и стоимость операций в Dask DataFrame.

Сравнение жадной модели выполнения (Pandas) и ленивой модели (Dask).
"""

import time
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
data_path = current_dir / "data" / "FY2016-STC-Category-Table.csv"
if not data_path.exists():
    data_path = current_dir / "FY2016-STC-Category-Table.csv"

images_dir = current_dir / "images"
images_dir.mkdir(exist_ok=True)


def ensure_data_exists(target_path: Path) -> Path:
    """Генерация эталонного среза данных при отсутствии исходного файла."""
    if target_path.exists():
        return target_path

    print(f"Файл {target_path.name} не найден на диске. Генерация эталонного среза...")
    states = [
        "Alabama", "Alaska", "Arizona", "California", "Colorado",
        "Florida", "Georgia", "Illinois", "New York", "Texas", "Washington", "Wyoming"
    ]
    tax_types = ["Total Taxes", "Property Taxes", "Sales Taxes", "Income Taxes", "License Taxes"]

    records = []
    for _ in range(500):
        for state in states:
            total = np.random.uniform(5000, 50000)
            prop = total * np.random.uniform(0.05, 0.25)
            sales = total * np.random.uniform(0.20, 0.40)
            inc = total * np.random.uniform(0.15, 0.30)
            lic = total - prop - sales - inc
            vals = {
                "Total Taxes": total,
                "Property Taxes": prop,
                "Sales Taxes": sales,
                "Income Taxes": inc,
                "License Taxes": lic,
            }
            for t in tax_types:
                records.append({
                    "Geo_Name": state,
                    "Tax_Type": t,
                    "Amount": f"{vals[t]:,.1f}",
                    "Survey_Year": 2016,
                })

    target_path.parent.mkdir(exist_ok=True)
    df_stub = pd.DataFrame(records)
    df_stub.to_csv(target_path, sep="\t", index=False)
    print(f"Тестовый срез сгенерирован и сохранён в: {target_path}")
    return target_path


def run_pandas_pipeline(file_path: Path) -> tuple[pd.Series, float]:
    """Шаг 1 методички: чтение, очистка, pivot_table и расчет долей в Pandas."""
    print("Вычисления в Pandas (жадная модель)...")
    start_time = time.perf_counter()

    taxes = pd.read_csv(file_path, sep="\t")
    taxes["Amount"] = (
        taxes["Amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("X", np.nan)
        .astype(float)
    )

    pivot = taxes.pivot_table(index="Geo_Name", columns="Tax_Type", values="Amount", aggfunc="mean")
    has_property_info = pivot[pivot["Property Taxes"].notna()].index
    pivot_clean = pivot.loc[has_property_info]

    frac_property = pivot_clean["Property Taxes"] / pivot_clean["Total Taxes"]
    frac_sorted = frac_property.sort_values()

    elapsed = time.perf_counter() - start_time
    print(f"Pandas выполнил расчет за: {elapsed:.4f} сек")
    return frac_sorted, elapsed


def run_dask_pipeline(file_path: Path) -> tuple[pd.Series, float]:
    """Шаг 2 методички: ленивая обработка, cat.as_known(), граф задач и .compute()."""
    print("Вычисления в Dask (ленивая модель)...")
    start_time = time.perf_counter()

    taxes = dd.read_csv(file_path, sep="\t", blocksize=5000)
    print(f"Число сформированных партиций Dask: {taxes.npartitions}")

    taxes["Amount"] = (
        taxes["Amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("X", np.nan)
        .astype(float)
    )

    taxes["Tax_Type"] = taxes["Tax_Type"].astype("category").cat.as_known()
    pivot = taxes.pivot_table(index="Geo_Name", columns="Tax_Type", values="Amount", aggfunc="mean")
    pivot_clean = pivot[~pivot["Property Taxes"].isna()]

    frac_property = pivot_clean["Property Taxes"] / pivot_clean["Total Taxes"]
    print("Структура ленивого графа задач (DAG) до вызова .compute():")
    print(frac_property)

    frac_result = frac_property.compute().sort_values()
    elapsed = time.perf_counter() - start_time
    print(f"Dask выполнил расчет (DAG + compute) за: {elapsed:.4f} сек")
    return frac_result, elapsed


def run_cost_and_persist_demo(file_path: Path) -> None:
    """Анализ стоимости дешевых поточечных операций и дорогой редукции (max)."""
    print("Анализ стоимости операций и кэширование (.persist())...")
    taxes = dd.read_csv(file_path, sep="\t", blocksize=5000)
    taxes["Amount"] = (
        taxes["Amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("X", np.nan)
        .astype(float)
    )

    taxes_cached = taxes.persist()
    taxes_cached["k_amount"] = taxes_cached["Amount"] / 1000

    t0 = time.perf_counter()
    max_k = taxes_cached["k_amount"].max().compute()
    t_reduction = time.perf_counter() - t0
    print(f"Максимальная сумма налога: {max_k:,.2f} тыс. $ (время редукции: {t_reduction:.4f} с)")


def save_visualizations(frac_result: pd.Series, t_pandas: float, t_dask: float) -> None:
    """Построение и сохранение графиков долей налогов и бенчмарка скорости."""
    plt.figure(figsize=(11, 5))
    x_positions = np.arange(len(frac_result))
    plt.plot(x_positions, frac_result.values, marker="o", linewidth=2, color="#1f77b4", markersize=5)
    plt.xticks(x_positions, frac_result.index, rotation=45, ha="right", fontsize=9)
    plt.title("Доля налога на имущество в общей сумме налогов по штатам (Property / Total)")
    plt.xlabel("Штаты")
    plt.ylabel("Доля")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    dist_plot = images_dir / "property_tax_distribution.png"
    plt.savefig(dist_plot, dpi=150)
    plt.close()
    print(f"График распределения налогов сохранен: {dist_plot}")

    plt.figure(figsize=(7, 4.5))
    frameworks = ["Pandas (Жадный)", "Dask (Ленивый DAG + compute)"]
    timings = [t_pandas, t_dask]
    colors = ["#4285F4", "#FBBC05"]

    bars = plt.bar(frameworks, timings, color=colors, width=0.45, edgecolor="black", alpha=0.85)
    plt.ylabel("Время выполнения (секунды)")
    plt.title("Сравнение времени обработки: Pandas vs Dask")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + max(timings) * 0.02, f"{h:.4f} с", ha="center", fontweight="bold")

    plt.tight_layout()
    bench_plot = images_dir / "pandas_vs_dask_benchmark.png"
    plt.savefig(bench_plot, dpi=150)
    plt.close()
    print(f"График сравнения Pandas vs Dask сохранен: {bench_plot}")


def main() -> None:
    csv_file = ensure_data_exists(data_path)
    frac_pd, t_pd = run_pandas_pipeline(csv_file)
    frac_dask, t_dask = run_dask_pipeline(csv_file)
    run_cost_and_persist_demo(csv_file)
    save_visualizations(frac_dask, t_pd, t_dask)


if __name__ == "__main__":
    main()