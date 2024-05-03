import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
st.set_option('deprecation.showPyplotGlobalUse', False)

# Загрузка данных
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

    # Преобразование числовых столбцов
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return data

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

        # Выбираем только числовые столбцы для корреляционной матрицы
        numeric_data = data.select_dtypes(include=[np.number])

        if not numeric_data.empty:  # Проверяем, что есть числовые столбцы
            corr_matrix = numeric_data.corr()
            st.write(corr_matrix)

            # Тепловая карта корреляций
            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            st.pyplot()
        else:
            st.write("No numeric columns found for correlation analysis.")

# Функция для кластерного анализа
def perform_clustering(data):
    st.subheader("Clustering Analysis")

    # Выбор колонок для кластеризации
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect("Select Columns for Clustering", numeric_cols, default=numeric_cols)

    if selected_cols:
        if data[selected_cols].isnull().values.any():
            st.warning("Dataset contains missing values. Handling missing values before clustering...")
            # Заполнение пропущенных значений средними значениями
            imputer = SimpleImputer(strategy='mean')
            data = pd.DataFrame(imputer.fit_transform(data[selected_cols]), columns=selected_cols)

        k_values = list(range(1, 11))  # Определяем диапазон значений k для метода локтя
        distortions = []

        # Вычисляем distortions для различных значений k
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data[selected_cols])
            distortions.append(
                kmeans.inertia_)  # Сохраняем значение inertia_ (сумма квадратов расстояний до центроидов)

        # Строим график метода локтя
        st.subheader("Elbow Method for Optimal K")
        plt.plot(k_values, distortions, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        st.pyplot()

        # Выбираем оптимальное количество кластеров на основе метода локтя
        optimal_k = st.slider("Select optimal number of clusters (k)", min_value=2, max_value=10, value=3)

        # Производим кластеризацию с выбранным числом кластеров
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)
        cluster_labels = kmeans.fit_predict(data[selected_cols])

        data['Cluster'] = cluster_labels
        st.write(data.head())

        # Применение PCA для уменьшения размерности до 2 компонентов
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data[selected_cols])

        # Добавление результатов PCA в DataFrame
        data['PCA Component 1'] = pca_result[:, 0]
        data['PCA Component 2'] = pca_result[:, 1]

        # Построение scatter plot с цветовой разметкой кластеров
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="PCA Component 1", y="PCA Component 2",
            hue="Cluster",
            palette=sns.color_palette("hsv", k),  # Используем цвета для каждого кластера
            data=data,
            legend="full",
            alpha=0.7
        )
        plt.title("Clustered Data (PCA Components)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        st.pyplot()

        # Группируем данные по кластерам и вычисляем средние значения по категориям
        summary_tab = data.groupby('Cluster')[selected_cols].mean().reset_index()

        st.subheader("Cluster Summary (Mean Values)")
        st.write(summary_tab)

        # Построение boxplot на основе выбранных данных
        st.subheader("Boxplot of Selected Data by Cluster")
        selected_data = data[selected_cols + ['Cluster']]  # Выбор данных для boxplot
        # Преобразование данных для boxplot с использованием melt
        long_data = selected_data.melt(id_vars=['Cluster'], var_name='Feature', value_name='Values')

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Feature', y='Values', hue='Cluster', data=long_data)
        plt.xticks(rotation=45)
        plt.title("Boxplot of Selected Data by Cluster")
        plt.xlabel("Feature")
        plt.ylabel("Values")
        st.pyplot()
    else:
        st.write("No numeric columns selected for clustering.")


# Функция для построения регрессионной модели
def build_regression_model(data):
    st.subheader("Regression Model")

    # Выбор целевой переменной (Target)
    target_col = st.selectbox("Select Target Variable", data.select_dtypes(include=[np.number]).columns)

    # Выбор признаков (Features)
    feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col != target_col]

    if data.isnull().values.any():
        data = data.dropna(subset=feature_cols + [target_col])

    # Разделение данных на обучающий и тестовый наборы
    X = data[feature_cols]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Добавление константы к признакам
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Обучение модели OLS
    model = sm.OLS(y_train, X_train)
    results = model.fit()

    # Вывод статистической сводки
    st.write("Regression Results:")
    st.write(results.summary())

    # График остатков
    residuals = y_test - results.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals Plot")
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    st.pyplot()

    # График предсказанных значений
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, results.predict(X_test))
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    st.pyplot()

    # Оценка качества модели на тестовых данных
    r2 = r2_score(y_test, results.predict(X_test))
    st.write("R-squared (R2) Score on Test Data:", r2)

    # Вывод коэффициентов модели
    st.subheader("Model Coefficients:")
    for idx, coef in enumerate(results.params):
        if idx < len(X.columns):  # Проверяем, что idx не выходит за пределы списка X.columns
            st.write(f"{X.columns[idx]}: {coef}")

    # Вывод интерпретации результатов
    st.subheader("Interpretation:")
    st.write("This regression model explains {:.2f}% of the variance in the target variable.".format(r2 * 100))


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
