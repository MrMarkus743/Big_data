import time
from pathlib import Path

import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import Client
from PIL import Image

# 1. Настройка путей
current_dir = Path(__file__).resolve().parent
data_path = current_dir / "data" / "FY2016-STC-Category-Table.csv"
if not data_path.exists():
    data_path = current_dir / "FY2016-STC-Category-Table.csv"
if not data_path.exists():
    # Проверяем соседнюю папку 06_dask_partitions
    alt_path = current_dir.parent / "06_dask_partitions" / "data" / "FY2016-STC-Category-Table.csv"
    if alt_path.exists():
        data_path = alt_path

images_dir = current_dir / "images"
images_dir.mkdir(exist_ok=True)


# 2. Шаги 1 и 2: Repartitioning и экспорт в Parquet
def demo_repartition_and_storage(csv_file: Path) -> None:
    """Демонстрация repartition, divisions и сохранения в Parquet."""
    print("ЭТАП 1: РАССЕКЦИОНИРОВАНИЕ (REPARTITION) И СОХРАНЕНИЕ В PARQUET")

    ddf = dd.read_csv(csv_file, sep="\t", blocksize=4000)
    print(f"Исходное число секций: {ddf.npartitions}")

    # Установка индекса по штату для получения осмысленных divisions
    ddf_indexed = ddf.set_index("Geo_Name", shuffle="tasks")
    print(f"Границы секций после set_index (divisions): {ddf_indexed.divisions}")

    # Сокращение количества секций
    ddf_repart = ddf_indexed.repartition(npartitions=2)
    print(f"Число секций после repartition: {ddf_repart.npartitions}")

    # Сохранение в формате Parquet
    parquet_output = current_dir / "data" / "taxes2.parquet"
    ddf_repart.to_parquet(parquet_output, write_index=True, overwrite=True)
    print(f"Распределённый датасет сохранен в Parquet: {parquet_output}")


# 3. Шаг 4: Расчет множества Мандельброта на Dask Array
def compute_point(c: complex) -> int:
    """Итерационный расчет принадлежности точки множеству Мандельброта."""
    t = -1
    z = complex(0, 0)
    max_iter = 200
    while abs(z) < 2:
        t += 1
        if t == max_iter:
            break
        z = z**2 + c
    return 255 - (255 * t) // max_iter


def block_prepare_pos_array(size: int, start: tuple, end: tuple, block_coords: np.ndarray) -> np.ndarray:
    """Генерация комплексных координат для отдельного блока матрицы."""
    nrows, ncols = block_coords.shape
    res = np.empty(shape=(nrows, ncols), dtype=np.complex128)
    startx, starty = start
    endx, endy = end

    for r in range(nrows):
        x = (endx - startx) * ((block_coords[r, 0] // size) / size) + startx
        for c in range(ncols):
            y = (endy - starty) * ((block_coords[r, c] % size) / size) + starty
            res[r, c] = complex(x, y)
    return res


def generate_mandelbrot(size: int = 1000, n_chunks: int = 4) -> Path:
    """Распределенное построение фрактала Мандельброта и сохранение картинки."""
    print(f"ЭТАП 2: РАСЧЕТ МНОЖЕСТВА МАНДЕЛЬБРОТА ({size}x{size}) В DASK")

    start = (-2.0, -1.5)
    end = (1.0, 1.5)

    # 1. Формирование двумерной координатной сетки через Dask Array
    chunk_dim = size // int(np.sqrt(n_chunks))
    range_arr = da.arange(0, size * size).reshape((size, size)).rechunk((chunk_dim, chunk_dim))

    # 2. Блочная генерация комплексных позиций
    pos_arr = da.blockwise(
        lambda b: block_prepare_pos_array(size, start, end, b),
        "ij",
        range_arr,
        "ij",
        dtype=np.complex128
    )

    # 3. Векторизация расчетной функции
    u_compute_point = da.frompyfunc(compute_point, 1, 1)
    img_arr = u_compute_point(pos_arr)

    # 4. Материализация и сохранение
    start_t = time.time()
    img_np = img_arr.compute().astype(np.uint8)
    elapsed = time.time() - start_t
    print(f"Расчет выполнен за {elapsed:.3f} сек (размер: {size}x{size}, блоков: {n_chunks})")

    img = Image.fromarray(img_np, mode="L")
    output_png = images_dir / "mandelbrot.png"
    img.save(output_png)
    print(f"Изображение фрактала сохранено в: {output_png}")
    return output_png


# 4. Шаг 5: Профилирование производительности
def run_benchmarks() -> None:
    """Замеры времени генерации фрактала при различных размерах сетки."""
    print("ЭТАП 3: БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ DASK ARRAY")

    test_sizes = [500, 1000, 2000]
    timings = []

    for sz in test_sizes:
        t0 = time.time()
        chunk_dim = sz // 2
        r_arr = da.arange(0, sz * sz).reshape((sz, sz)).rechunk((chunk_dim, chunk_dim))
        p_arr = da.blockwise(
            lambda b: block_prepare_pos_array(sz, (-2.0, -1.5), (1.0, 1.5), b),
            "ij", r_arr, "ij", dtype=np.complex128
        )
        u_calc = da.frompyfunc(compute_point, 1, 1)
        _ = u_calc(p_arr).compute()
        elapsed = time.time() - t0
        timings.append(elapsed)
        print(f"Сетка {sz}x{sz} (4 блока): {elapsed:.3f} сек")

    # Построение графика бенчмарка
    plt.figure(figsize=(8, 4.5))
    plt.plot([f"{s}x{s}" for s in test_sizes], timings, marker="o", color="#e67e22", linewidth=2.2)
    for i, t in enumerate(timings):
        plt.text(i, t + max(timings) * 0.03, f"{t:.2f} с", ha="center", fontweight="bold")
    plt.title("Масштабируемость расчета множества Мандельброта на Dask")
    plt.xlabel("Размер матрицы (пиксели)")
    plt.ylabel("Время вычисления (секунды)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    bench_plot = images_dir / "mandelbrot_benchmark.png"
    plt.savefig(bench_plot, dpi=150)
    plt.close()
    print(f"График бенчмарка сохранен: {bench_plot}")


def main() -> None:
    csv_target = data_path

    # Инициализация локального распределенного кластера
    client = Client(n_workers=4, threads_per_worker=2, memory_limit="2GB")
    print(f"Dask Dashboard доступен: {client.dashboard_link}")

    try:
        demo_repartition_and_storage(csv_target)
        generate_mandelbrot(size=1000, n_chunks=4)
        run_benchmarks()
    finally:
        client.close()
        print("\nРаспределенный кластер Dask корректно остановлен.")


if __name__ == "__main__":
    main()