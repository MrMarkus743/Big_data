from pathlib import Path

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

data_path = Path(__file__).resolve().parent / 'data' / 'data.csv'

# Загружаем датасет
df = pd.read_csv(data_path)

print("Исходные данные:")
print(df.head())

# Создаем матрицу "пользователь-фильм"
user_movie_matrix = df.pivot_table(
    index='user_id',
    columns='item_id',
    values='rating',
    fill_value=0
)

print("\nМатрица пользователь-фильм:")
print(user_movie_matrix.head())


# ==========================================
# Вариант 1. Сходство между фильмами (Cosine)
# ==========================================

# Вычисляем косинусное сходство между фильмами (транспонируем матрицу)
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

print("\nСходство между фильмами:")
print(movie_similarity_df.iloc[:5, :5])


def recommend_movies(movie_id, n_recommendations=3):
    """Рекомендует фильмы на основе косинусного сходства."""
    similar_movies = movie_similarity_df[movie_id].sort_values(ascending=False)
    # Исключаем сам фильм из рекомендаций
    similar_movies = similar_movies[similar_movies.index != movie_id]
    return similar_movies.head(n_recommendations)


# Тестируем рекомендации для первого фильма
first_movie = user_movie_matrix.columns[0]
print(f"\nРекомендации для фильма {first_movie}:")
print(recommend_movies(first_movie))


# ==========================================
# Вариант 2. SVD (Матричная факторизация)
# ==========================================

# Уменьшаем размерность до 2 компонент
svd = TruncatedSVD(n_components=2, random_state=42)
matrix_reduced = svd.fit_transform(user_movie_matrix)

# Восстанавливаем матрицу с предсказанными оценками
matrix_approx = svd.inverse_transform(matrix_reduced)
matrix_approx_df = pd.DataFrame(
    matrix_approx,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)

print("\nПриближенная матрица после SVD:")
print(matrix_approx_df.iloc[:5, :5].round(2))


def recommend_for_user(user_id, n_recommendations=2):
    """Рекомендует фильмы для конкретного пользователя."""
    user_ratings = matrix_approx_df.loc[user_id]
    # Ищем фильмы, которые пользователь еще не смотрел (где стоял 0)
    watched_movies = user_movie_matrix.loc[user_id]
    unwatched_movies = user_ratings[watched_movies == 0]
    return unwatched_movies.sort_values(ascending=False).head(n_recommendations)


# Тестируем рекомендации для первого пользователя
first_user = user_movie_matrix.index[0]
print(f"\nРекомендации для пользователя {first_user}:")
print(recommend_for_user(first_user))