import time
from pathlib import Path

import dask.array as da
import dask.dataframe as dd
import dask_ml.linear_model as dlm
import dask_ml.model_selection as dcv
import dask_ml.preprocessing as dpre
import matplotlib.pyplot as plt
import pandas as pd
from dask.distributed import Client

# 1. Определение путей к данным и каталогам
current_dir = Path(__file__).resolve().parent
excel_path = current_dir / "data" / "logisticheskaya_regressia_dannye.xlsx"

if not excel_path.exists():
    excel_path = current_dir / "logisticheskaya_regressia_dannye.xlsx"

images_dir = current_dir / "images"
images_dir.mkdir(exist_ok=True)


# 2. Инициализация распределенного кластера Dask
def setup_dask_cluster() -> Client:
    """Настройка и запуск локального кластера Dask."""
    print("Инициализация локального Dask кластера...")
    client = Client(n_workers=4, threads_per_worker=2, memory_limit="2GB")
    print(f"Dask Dashboard доступен по адресу: {client.dashboard_link}")
    return client


# 3. Загрузка и подготовка данных
def load_and_prepare_data(file_path: Path):
    """Чтение датасета, кодирование и преобразование в структуры Dask."""
    print(f"\nЗагрузка данных из: {file_path}")
    raw_df = pd.read_excel(file_path)
    print(f"Размер исходных данных: {raw_df.shape[0]} строк, {raw_df.shape[1]} столбцов")

    # Определение целевой переменной (последняя колонка по умолчанию)
    target_col = raw_df.columns[-1]
    feature_cols = [c for c in raw_df.columns if c != target_col]
    print(f"Целевой признак кредитного скоринга: '{target_col}'")

    # Преобразование возможных строковых/категориальных признаков
    df_processed = pd.get_dummies(raw_df[feature_cols], drop_first=True)
    y_raw = raw_df[target_col].astype(int)

    # Конвертация в Dask DataFrame с разбиением на партиции
    X_ddf = dd.from_pandas(df_processed, npartitions=4)
    y_ddf = dd.from_pandas(y_raw, npartitions=4)

    # Масштабирование признаков через dask_ml
    scaler = dpre.StandardScaler()
    X_scaled = scaler.fit_transform(X_ddf.to_dask_array(lengths=True))
    y_array = y_ddf.to_dask_array(lengths=True)

    # Распределенное разделение на обучающую и валидационную выборки (80 / 20)
    X_train, X_test, y_train, y_test = dcv.train_test_split(
        X_scaled, y_array, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# 4. Обучение распределенной модели
def train_model(X_train, y_train):
    """Обучение логистической регрессии средствами Dask ML."""
    print("\nОбучение логистической регрессии в Dask...")
    # solver='admm' устойчив к обновлениям SciPy и оптимизирован для Dask
    model = dlm.LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="admm",
        max_iter=100,
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"Обучение завершено за: {training_time:.3f} сек")
    return model, training_time


# 5. Подбор гиперпараметров (GridSearchCV)
def hyperparameter_tuning(X_train, y_train):
    """Подбор коэффициента регуляризации C с помощью кросс-валидации Dask."""
    print("\nПодбор гиперпараметров (GridSearchCV)...")
    param_grid = {
        "C": [0.1, 1.0, 10.0]
    }

    base_model = dlm.LogisticRegression(
        penalty="l2",
        solver="admm",
        max_iter=100,
        random_state=42
    )

    grid_search = dcv.GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy"
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time

    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая точность на CV: {grid_search.best_score_:.4f}")
    print(f"Время подбора параметров: {tuning_time:.3f} сек")

    return grid_search.best_estimator_, tuning_time


# 6. Оценка качества модели
def evaluate_model(model, X_test, y_test) -> float:
    """Вычисление метрики Accuracy через ленивые операции Dask Array."""
    print("\nОценка качества на тестовой выборке...")
    y_pred = model.predict(X_test)

    # Ленивое вычисление среднего совпадения классов
    accuracy_lazy = da.mean(y_pred == y_test)
    accuracy = float(accuracy_lazy.compute())

    print(f"Accuracy на тестовой выборке: {accuracy:.4f}")
    return accuracy


# 7. Визуализация и сохранение артефактов
def visualize_results(accuracy: float, train_time: float, tune_time: float) -> None:
    """Построение и сохранение сравнительных графиков в папку images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # График точности
    ax1.bar(["Тестовая Accuracy"], [accuracy], color="#4285F4", width=0.4)
    ax1.set_ylabel("Доля верных ответов")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Качество классификации (Dask ML)")
    ax1.text(0, accuracy + 0.02, f"{accuracy:.4f}", ha="center", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # График времени работы
    stages = ["Обучение модели", "GridSearchCV"]
    times = [train_time, tune_time]
    bars = ax2.bar(stages, times, color=["#34A853", "#FBBC05"], width=0.5)
    ax2.set_ylabel("Время (секунды)")
    ax2.set_title("Время выполнения этапов")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f} с", ha="center")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_plot = images_dir / "model_evaluation.png"
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"Диаграмма результатов успешно сохранена в: {output_plot}")


def main() -> None:
    if not excel_path.exists():
        raise FileNotFoundError(f"Файл данных не найден по пути: {excel_path}")

    client = setup_dask_cluster()

    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data(excel_path)
        base_model, train_time = train_model(X_train, y_train)
        best_model, tune_time = hyperparameter_tuning(X_train, y_train)
        accuracy = evaluate_model(best_model, X_test, y_test)
        visualize_results(accuracy, train_time, tune_time)
    finally:
        client.close()
        print("\nКластер Dask успешно остановлен.")


if __name__ == "__main__":
    main()