from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# 1. Поиск и загрузка CSV-файла
current_dir = Path(__file__).resolve().parent
data_path = current_dir / "data" / "variant_1.csv"

# Проверяем наличие файла в директории исполняемого кода
if not data_path.exists():
    data_path = current_dir / "variant_1.csv"

if not data_path.exists():
    raise FileNotFoundError(f"Файл данных не найден: {data_path}")

print(f"Загрузка данных из: {data_path}")
df = pd.read_csv(data_path)

print("\nИсходные данные:")
print(df.head())


# 2. Проверка структуры колонок и весов
if not {"source", "target"}.issubset(df.columns):
    raise ValueError("CSV-файл должен содержать столбцы 'source' и 'target'")

if "weight" not in df.columns:
    df["weight"] = 1

df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1)


# 3. Создание неориентированного графа
G = nx.from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr="weight",
    create_using=nx.Graph()
)


# 4. Общие топологические характеристики графа
print("\n" + "=" * 50)
print("ОСНОВНЫЕ ПОКАЗАТЕЛИ ГРАФА")
print("=" * 50)
print(f"Количество вершин (узлов): {G.number_of_nodes()}")
print(f"Количество рёбер (связей): {G.number_of_edges()}")
print(f"Плотность графа: {round(nx.density(G), 4)}")
print(f"Количество компонент связности: {nx.number_connected_components(G)}")
print(f"Средний коэффициент кластеризации: {round(nx.average_clustering(G), 4)}")


# 5. Расчет степеней вершин
degree = dict(G.degree())
weighted_degree = dict(G.degree(weight="weight"))
avg_degree = sum(degree.values()) / G.number_of_nodes()
print(f"Средняя степень вершины: {round(avg_degree, 4)}")


# 6. Показатели расстояния (поиск пути)
if nx.is_connected(G):
    H = G
    print("Граф связный: Да")
else:
    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()
    print("Граф связный: Нет (расчет для наибольшей компоненты)")

if H.number_of_nodes() > 1:
    print(f"Средняя длина кратчайшего пути: {round(nx.average_shortest_path_length(H), 4)}")
    print(f"Диаметр графа: {nx.diameter(H)}")
    print(f"Радиус графа: {nx.radius(H)}")


# 7. Расчет показателей центральности
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=2000)
pagerank_val = nx.pagerank(G, weight="weight")
clustering_val = nx.clustering(G)


# 8. Формирование сводной таблицы метрик
result = pd.DataFrame({
    "node": list(G.nodes()),
    "degree": [degree[n] for n in G.nodes()],
    "weighted_degree": [weighted_degree[n] for n in G.nodes()],
    "degree_centrality": [degree_centrality[n] for n in G.nodes()],
    "betweenness_centrality": [betweenness_centrality[n] for n in G.nodes()],
    "closeness_centrality": [closeness_centrality[n] for n in G.nodes()],
    "eigenvector_centrality": [eigenvector_centrality[n] for n in G.nodes()],
    "pagerank": [pagerank_val[n] for n in G.nodes()],
    "clustering_coefficient": [clustering_val[n] for n in G.nodes()]
})

metric_cols = [
    "weighted_degree", "degree_centrality", "betweenness_centrality",
    "closeness_centrality", "eigenvector_centrality", "pagerank", "clustering_coefficient"
]
result[metric_cols] = result[metric_cols].round(4)
result = result.sort_values(by="betweenness_centrality", ascending=False)

print("\n" + "=" * 50)
print("ПОКАЗАТЕЛИ ПО ВЕРШИНАМ (сортировка по Betweenness)")
print("=" * 50)
print(result.to_string(index=False))


# 9. Поиск ключевых вершин
print("\n" + "=" * 50)
print("КЛЮЧЕВЫЕ ВЕРШИНЫ СЕТИ")
print("=" * 50)
print(f"Лидер по Degree Centrality: {max(degree_centrality, key=degree_centrality.get)}")
print(f"Лидер по Betweenness Centrality: {max(betweenness_centrality, key=betweenness_centrality.get)}")
print(f"Лидер по Closeness Centrality: {max(closeness_centrality, key=closeness_centrality.get)}")
print(f"Лидер по PageRank: {max(pagerank_val, key=pagerank_val.get)}")


# 10. Сохранение метрик в CSV
output_csv = current_dir / "network_metrics.csv"
result.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\nТаблица метрик успешно сохранена в: {output_csv}")


# 11. Визуализация сети
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42, weight="weight")

# Размер вершин пропорционален Degree Centrality
node_sizes = [1200 + 7000 * degree_centrality[node] for node in G.nodes()]
# Толщина ребра пропорциональна его весу
edge_widths = [0.7 + G[u][v]["weight"] * 0.45 for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#4CAF50", alpha=0.9)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")

# Подписи весов на ребрах
edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Граф взаимодействий агробиотехнологической системы (Вариант 1)")
plt.axis("off")
plt.tight_layout()
plt.show()