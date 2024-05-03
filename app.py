import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Загрузка данных
@st.cache
def load_data(file):
    return pd.read_csv(file)

# Функция для проведения EDA
def perform_eda(data):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write(data.head())  # Показать первые несколько строк данных
    st.write(data.describe())  # Статистика данных

    # Гистограммы числовых признаков
    st.subheader("Histograms of Numeric Columns")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols)
    if selected_cols:
        st.write(data[selected_cols].hist(bins=20, figsize=(12, 6)))
        st.pyplot()

    # Корреляционный анализ
    st.subheader("Correlation Analysis")
    corr_matrix = data.corr()
    st.write(corr_matrix)

    # Тепловая карта корреляций
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    st.pyplot()

# Функция для кластерного анализа
def perform_clustering(data):
    st.subheader("Clustering Analysis")

    # Выбор колонок для кластеризации
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect("Select Columns for Clustering", numeric_cols, default=numeric_cols)

    if selected_cols:
        k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_labels = kmeans.fit_predict(data[selected_cols])

        data['Cluster'] = cluster_labels
        st.write(data.head())

# Функция для построения регрессионной модели
def build_regression_model(data):
    st.subheader("Regression Model")

    # Выбор целевой переменной (Target)
    target_col = st.selectbox("Select Target Variable", data.columns)

    # Выбор признаков (Features)
    feature_cols = [col for col in data.columns if col != target_col]

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(data[feature_cols], data[target_col], test_size=0.2, random_state=0)

    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Оценка модели на тестовых данных
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    st.write("R-squared (R2) Score:", r2)

# Основная часть приложения
def main():
    st.title("Data Analysis App")

    # Загрузка данных
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        data = load_data(file)
        st.write("Data loaded successfully!")

        # Выполнение EDA
        perform_eda(data)

        # Выполнение кластерного анализа
        perform_clustering(data)

        # Построение регрессионной модели
        build_regression_model(data)

if __name__ == "__main__":
    main()
