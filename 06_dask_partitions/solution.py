from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Настройка путей
current_dir = Path(__file__).resolve().parent
data_path = current_dir / "data" / "FY2016-STC-Category-Table.csv"

images_dir = current_dir / "images"
images_dir.mkdir(exist_ok=True)

def run_tax_pipeline(file_path: Path) -> None:
    """Выполнение основного пайплайна: предобработка, расчет долей и визуализация."""
    print("ЭТАП 1: ЛЕНИВАЯ ПРЕДОБРАБОТКА И РАСЧЕТ ДОЛИ ИМУЩЕСТВЕННОГО НАЛОГА")

    # Чтение с явным секционированием по блокам (Out-of-Core)
    taxes = dd.read_csv(file_path, sep="\t", blocksize=5000)
    print(f"Количество созданных партиций (блоков): {taxes.npartitions}")

    # Очистка числовых значений
    taxes["Amount_clean"] = (
        taxes["Amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("X", np.nan)
        .astype(float)
    )

    # Обязательное явное задание категорий для Dask DataFrame
    taxes["Tax_Type"] = taxes["Tax_Type"].astype("category").cat.as_known()

    # Сводная таблица
    pivot = taxes.pivot_table(index="Geo_Name", columns="Tax_Type", values="Amount_clean")

    # Расчет доли налога на имущество
    frac_property = pivot["Property Taxes"] / pivot["Total Taxes"]

    print("Объект frac_property представляет собой ленивый граф задач:")
    print(frac_property)

    # Материализация результатов через .compute()
    frac_result = frac_property.compute().dropna().sort_values()
    print("\nИтоговые доли налога на имущество по штатам (Top-5):")
    print(frac_result.head())

    # Построение графика распределения
    plt.figure(figsize=(12, 5))
    x_pos = np.arange(len(frac_result))
    plt.plot(x_pos, frac_result.values, marker="o", color="#1a73e8", linewidth=2, markersize=5)
    plt.xticks(x_pos, frac_result.index, rotation=45, ha="right", fontsize=9)
    plt.title("Доля налога на имущество в общих налоговых сборах (Property Taxes / Total Taxes)")
    plt.xlabel("Штат / Территория")
    plt.ylabel("Доля от общей суммы")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_file = images_dir / "property_tax_distribution.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"График успешно сохранён: {plot_file}")


def run_cost_analysis(file_path: Path) -> None:
    """Анализ стоимости операций: поточечные вычисления против глобальной редукции."""
    print("ЭТАП 2: АНАЛИЗ СТОИМОСТИ ОПЕРАЦИЙ И КЭШИРОВАНИЕ (PERSIST)")

    taxes = dd.read_csv(file_path, sep="\t", blocksize=5000)
    taxes["Amount_clean"] = (
        taxes["Amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("X", np.nan)
        .astype(float)
    )

    # Кэширование промежуточного очищенного состояния в RAM воркеров
    taxes_cached = taxes.persist()

    # Дешевая поэлементная операция (выполняется параллельно и независимо внутри каждой секции)
    taxes_cached["k_amount"] = taxes_cached["Amount_clean"] / 1000

    # Дорогая операция редукции (требует межузлового сбора локальных максимумов в единый узел)
    max_k = taxes_cached["k_amount"].max().compute()
    print(f"Максимальная сумма налога (в тыс. долл.): {max_k:,.2f}")


def main() -> None:
    run_tax_pipeline(data_path)
    run_cost_analysis(data_path)


if __name__ == "__main__":
    main()