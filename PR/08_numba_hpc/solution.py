import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange, vectorize
from PIL import Image

# 1. Настройка путей
current_dir = Path(__file__).resolve().parent
images_dir = current_dir / "images"
images_dir.mkdir(exist_ok=True)


# 2. Базовая расчетная функция на чистом Python
def compute_point(c: complex, max_iter: int = 200) -> int:
    """Расчет итераций Мандельброта для отдельной точки комплексной плоскости."""
    num_iter = -1
    z = complex(0, 0)
    while abs(z) < 2:
        num_iter += 1
        if num_iter == max_iter:
            break
        z = z**2 + c
    return 255 - (255 * num_iter) // max_iter


# 3. Последовательный обход сетки (двойной цикл)
def do_all(size: int, start: tuple, end: tuple, img_array: np.ndarray, compute_fun) -> None:
    """Заполнение матрицы пикселей путем поточечного вызова compute_fun."""
    startx, starty = start
    endx, endy = end
    for xp in range(size):
        x = (endx - startx) * (xp / size) + startx
        for yp in range(size):
            y = (endy - starty) * (yp / size) + starty
            img_array[yp, xp] = compute_fun(complex(x, y))


# 4. Многопоточный параллельный обход GIL
@jit(nopython=True, parallel=True, nogil=True)
def pdo_all(size: int, start: tuple, end: tuple, img_array: np.ndarray, compute_fun) -> None:
    """Параллельное распределение внешнего цикла по потокам через prange."""
    startx, starty = start
    endx, endy = end
    for xp in prange(size):
        x = (endx - startx) * (xp / size) + startx
        for yp in range(size):
            y = (endy - starty) * (yp / size) + starty
            img_array[yp, xp] = compute_fun(complex(x, y))


# 5. Векторизация в универсальную функцию NumPy (ufunc)
compute_point_ufunc = vectorize(
    ["uint8(complex128, uint64)"],
    target="parallel"
)(compute_point)


def prepare_pos_array(start: tuple, end: tuple, pos_array: np.ndarray) -> None:
    """Подготовка матрицы комплексных координат для ufunc."""
    size = pos_array.shape[0]
    startx, starty = start
    endx, endy = end
    for xp in range(size):
        x = (endx - startx) * (xp / size) + startx
        for yp in range(size):
            y = (endy - starty) * (yp / size) + starty
            pos_array[yp, xp] = complex(x, y)


def run_benchmarks() -> None:
    """Сравнительное тестирование производительности всех режимов вычислений."""
    # Для демонстрации чистого Python и forceobj используем сетку 1000x1000,
    # чтобы замер выполнился за разумное время без зависания ПК
    size = 1000
    start = (-1.5, -1.3)
    end = (0.5, 1.3)

    print(f"БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ NUMBA (РАЗМЕР МАТРИЦЫ: {size}x{size})")

    results = {}

    # --- Режим 1: Чистый Python ---
    print("\n1. Запуск версии на чистом Python...")
    img_pure = np.empty((size, size), dtype=np.uint8)
    t0 = time.time()
    do_all(size, start, end, img_pure, compute_point)
    t_pure = time.time() - t0
    results["Pure Python"] = t_pure
    print(f"-> Время Pure Python: {t_pure:.4f} сек")

    # --- Режим 2: Numba JIT (нативный режим) ---
    print("\n2. Запуск Numba JIT (nopython=True)...")
    compute_point_numba = jit(nopython=True)(compute_point)
    # «Прогревочный» вызов для JIT-компиляции в LLVM
    _ = compute_point_numba(complex(4, 4))

    img_numba = np.empty((size, size), dtype=np.uint8)
    t0 = time.time()
    do_all(size, start, end, img_numba, compute_point_numba)
    t_numba = time.time() - t0
    results["Numba JIT"] = t_numba
    print(f"-> Время Numba JIT: {t_numba:.4f} сек (ускорение в {t_pure / t_numba:.1f}x)")

    # --- Режим 3: Numba с объектным режимом (forceobj=True) ---
    print("\n3. Запуск Numba Object Mode (forceobj=True)...")
    # Замер на уменьшенном срезе 300x300 с последующей экстраполяцией,
    # чтобы не ждать несколько минут объектного оверхеда
    sub_size = 300
    compute_point_obj = jit(forceobj=True)(compute_point)
    _ = compute_point_obj(complex(4, 4))
    img_obj = np.empty((sub_size, sub_size), dtype=np.uint8)
    t0 = time.time()
    do_all(sub_size, start, end, img_obj, compute_point_obj)
    t_sub = time.time() - t0
    t_obj = t_sub * ((size / sub_size) ** 2)
    results["Numba Object Mode"] = t_obj
    print(f"-> Время Numba forceobj (экстраполировано): ~{t_obj:.2f} сек (замедление из-за CPython)")

    # --- Режим 4: Numba Parallel (prange + nogil) ---
    print("\n4. Запуск многопоточной версии (parallel=True, nogil=True)...")
    # Прогревочный запуск параллельного цикла
    dummy_arr = np.empty((10, 10), dtype=np.uint8)
    pdo_all(10, start, end, dummy_arr, compute_point_numba)

    img_parallel = np.empty((size, size), dtype=np.uint8)
    t0 = time.time()
    pdo_all(size, start, end, img_parallel, compute_point_numba)
    t_parallel = time.time() - t0
    results["Numba Parallel"] = t_parallel
    print(f"-> Время Numba Parallel: {t_parallel:.4f} сек (ускорение в {t_pure / t_parallel:.1f}x)")

    # --- Режим 5: Векторизация NumPy ufunc ---
    print("\n5. Запуск векторизованной версии (@vectorize target='parallel')...")
    pos_array = np.empty((size, size), dtype=np.complex128)
    prepare_pos_array(start, end, pos_array)
    # Прогрев ufunc
    _ = compute_point_ufunc(pos_array[:2, :2], 200)

    t0 = time.time()
    img_ufunc = compute_point_ufunc(pos_array, 200)
    t_ufunc = time.time() - t0
    results["NumPy ufunc"] = t_ufunc
    print(f"-> Время NumPy ufunc: {t_ufunc:.4f} сек (ускорение в {t_pure / t_ufunc:.1f}x)")

    # 6. Сохранение итогового фрактального изображения
    mandelbrot_img = Image.fromarray(img_parallel, mode="L")
    img_out = images_dir / "mandelbrot_numba.png"
    mandelbrot_img.save(img_out)
    print(f"\nФрактальное изображение сохранено в: {img_out}")

    # 7. Построение графика сравнительной производительности
    plot_benchmarks(results)


def plot_benchmarks(results: dict) -> None:
    """Построение логарифмической диаграммы времени выполнения."""
    labels = list(results.keys())
    times = list(results.values())
    colors = ["#e74c3c", "#3498db", "#95a5a6", "#2ecc71", "#9b59b6"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, times, color=colors, edgecolor="black", alpha=0.85)
    plt.yscale("log")
    plt.ylabel("Время выполнения, сек (лог. шкала)")
    plt.title("Сравнение производительности режимов вычислений в Numba")
    plt.grid(axis="y", linestyle="--", alpha=0.5, which="both")

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h * 1.15, f"{h:.3f} с", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_out = images_dir / "numba_benchmark.png"
    plt.savefig(chart_out, dpi=150)
    plt.close()
    print(f"Диаграмма бенчмарка успешно сохранена в: {chart_out}")


def main() -> None:
    run_benchmarks()


if __name__ == "__main__":
    main()