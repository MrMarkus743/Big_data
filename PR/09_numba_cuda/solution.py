import time
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Настройка путей
current_dir = Path(__file__).resolve().parent
images_dir = current_dir / "images"
images_dir.mkdir(exist_ok=True)


# 2. Проверка аппаратного обеспечения и окружения CUDA
def inspect_cuda_environment() -> bool:
    """Диагностика доступности GPU NVIDIA, драйверов и runtime CUDA."""
    print("ЭТАП 1: ДИАГНОСТИКА СИСТЕМЫ И ДОСТУПНОСТИ CUDA")

    try:
        import numba
        from numba import cuda
        print(f"Версия Numba: {numba.__version__}")
        cuda_ok = cuda.is_available()
        print(f"CUDA доступна в системе: {cuda_ok}")

        if cuda_ok:
            dev = cuda.get_current_device()
            cc = dev.compute_capability
            print(f"Обнаружен GPU: {dev.name.decode('utf-8') if isinstance(dev.name, bytes) else dev.name}")
            print(f"Compute Capability: {cc[0]}.{cc[1]}")
            return True
        else:
            print("Предупреждение: Графический ускоритель NVIDIA не обнаружен или отсутствуют библиотеки CUDA toolkit.")
            print("Скрипт перейдет в режим симуляции и CPU-fallback для успешной демонстрации.")
            return False
    except Exception as e:
        print(f"Диагностика CUDA завершилась с ошибкой: {e}")
        return False


CUDA_AVAILABLE = inspect_cuda_environment()


# 3. Реализация CUDA-кернелов (при доступности GPU)
if CUDA_AVAILABLE:
    from numba import cuda

    @cuda.jit
    def double_safe_1d(my_array):
        """Безопасное одномерное удвоение элементов с проверкой границ."""
        pos = cuda.grid(1)
        if pos >= my_array.shape[0]:
            return
        my_array[pos] *= 2

    @cuda.jit
    def double_matrix_safe_2d(my_matrix):
        """Безопасное двумерное удвоение матрицы с явным контролем осей X и Y."""
        x, y = cuda.grid(2)
        if x >= my_matrix.shape[0] or y >= my_matrix.shape[1]:
            return
        my_matrix[y, x] *= 2

    @cuda.jit(device=True)
    def compute_point_device(c):
        """Device-функция расчета одной точки множества Мандельброта."""
        i = -1
        z = complex(0.0, 0.0)
        max_iter = 200
        while abs(z) < 2.0:
            i += 1
            if i == max_iter:
                break
            z = z**2 + c
        return 255 - (255 * i) // max_iter

    @cuda.jit
    def compute_all_points_cuda(startx, starty, endx, endy, size, img_array):
        """Кернел генерации полного множества Мандельброта на GPU."""
        x, y = cuda.grid(2)
        if x >= img_array.shape[0] or y >= img_array.shape[1]:
            return

        mandel_x = (endx - startx) * (x / size) + startx
        mandel_y = (endy - starty) * (y / size) + starty
        img_array[y, x] = compute_point_device(complex(mandel_x, mandel_y))


# 4. Fallback-реализация для систем без выделенного GPU NVIDIA
def run_cpu_fallback(size: int, start: tuple, end: tuple) -> np.ndarray:
    """Оптимизированный расчет для хостов без аппаратной поддержки CUDA."""
    from numba import njit, prange

    @njit(parallel=True, fastmath=True)
    def compute_mandelbrot_cpu(size, startx, starty, endx, endy):
        img = np.empty((size, size), dtype=np.uint8)
        for xp in prange(size):
            x = (endx - startx) * (xp / size) + startx
            for yp in range(size):
                y = (endy - starty) * (yp / size) + starty
                c = complex(x, y)
                z = complex(0.0, 0.0)
                it = -1
                while abs(z) < 2.0:
                    it += 1
                    if it == 200:
                        break
                    z = z**2 + c
                img[yp, xp] = 255 - (255 * it) // 200
        return img

    return compute_mandelbrot_cpu(size, start[0], start[1], end[0], end[1])


# 5. Демонстрация 1D и 2D операций удвоения массива
def test_array_operations() -> None:
    print("ЭТАП 2: ТЕСТИРОВАНИЕ СЕТКИ И ГРАНИЧНЫХ УСЛОВИЙ (1D И 2D)")

    n_elements = 1000
    arr = np.ones(n_elements, dtype=np.float32)

    if CUDA_AVAILABLE:
        threads_per_block = 16
        blocks_per_grid = ceil(n_elements / threads_per_block)
        print(f"1D Запуск: элементов={n_elements}, потоков/блок={threads_per_block}, блоков={blocks_per_grid}")
        print(f"Всего потоков: {threads_per_block * blocks_per_grid} (лишних потоков: {threads_per_block * blocks_per_grid - n_elements})")

        double_safe_1d[blocks_per_grid, threads_per_block](arr)
        assert (arr == 2).all(), "Ошибка: не все элементы 1D массива равны 2!"
        print("-> Проверка 1D массива пройдена: все элементы успешно удвоены.")

        # 2D Матрица 1000x1000
        mat = np.ones((1000, 1000), dtype=np.float32)
        threads_2d = (16, 16)
        blocks_2d = (ceil(1000 / 16), ceil(1000 / 16))
        double_matrix_safe_2d[blocks_2d, threads_2d](mat)
        assert (mat == 2).all(), "Ошибка: в матрице присутствуют некорректные значения!"
        print("-> Проверка 2D матрицы пройдена: все значения строго равны 2.0.")
    else:
        arr *= 2
        print("-> Режим CPU: массив 1000 элементов удвоен.")


# 6. Расчет множества Мандельброта и сохранение графиков
def run_mandelbrot_pipeline() -> None:
    print("ЭТАП 3: ГЕНЕРАЦИЯ МНОЖЕСТВА МАНДЕЛЬБРОТА")

    size = 2000
    start = (-1.5, -1.3)
    end = (0.5, 1.3)
    img_array = np.empty((size, size), dtype=np.uint8)

    t0 = time.time()
    if CUDA_AVAILABLE:
        threads_2d = (16, 16)
        blocks_2d = (ceil(size / 16), ceil(size / 16))
        print(f"Конфигурация CUDA ядра: Сетка={blocks_2d}, Блоки={threads_2d}")
        compute_all_points_cuda[blocks_2d, threads_2d](
            start[0], start[1], end[0], end[1], size, img_array
        )
        # Синхронизация хоста и девайса
        cuda.synchronize()
        exec_mode = "GPU (Numba CUDA)"
    else:
        img_array = run_cpu_fallback(size, start, end)
        exec_mode = "CPU Fallback (Numba Parallel)"

    elapsed = time.time() - t0
    print(f"Расчет матрицы {size}x{size} завершен за: {elapsed:.4f} сек ({exec_mode})")

    # Сохранение фрактала
    img = Image.fromarray(img_array, mode="L")
    output_png = images_dir / "mandelbrot_gpu.png"
    img.save(output_png)
    print(f"Изображение фрактала сохранено: {output_png}")

    # Построение инфографики
    plt.figure(figsize=(9, 4.5))
    modes = ["CPU Pure Python\n(ПЗ 5.1)", "CPU Numba Parallel\n(ПЗ 5.1)", f"{exec_mode}\n(ПЗ 5.2)"]
    timings = [64.09, 0.81, max(elapsed, 0.05)]
    bars = plt.bar(modes, timings, color=["#e74c3c", "#3498db", "#2ecc71"], width=0.55)
    plt.yscale("log")
    plt.ylabel("Время вычисления, сек (лог. шкала)")
    plt.title("Сравнение ускорения: CPU против GPU (Мандельброт 2000x2000)")
    plt.grid(axis="y", linestyle="--", alpha=0.5, which="both")

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h * 1.2, f"{h:.2f} с", ha="center", fontweight="bold")

    plt.tight_layout()
    chart_png = images_dir / "gpu_acceleration_benchmark.png"
    plt.savefig(chart_png, dpi=150)
    plt.close()
    print(f"Диаграмма производительности сохранена: {chart_png}")


def main() -> None:
    test_array_operations()
    run_mandelbrot_pipeline()


if __name__ == "__main__":
    main()