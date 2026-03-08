import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Настройка страницы
st.set_page_config(
    page_title="Классификатор астероидов",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Определяем базовую директорию
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Функция для загрузки моделей
@st.cache_resource
def load_models():
    """Загрузка всех сохраненных моделей"""
    models = {}
    model_files = {
        'Логистическая регрессия': os.path.join(BASE_DIR, 'models', 'model_logreg.pkl'),
        'Градиентный бустинг': os.path.join(BASE_DIR, 'models', 'model_gb.pkl'),
        'XGBoost': os.path.join(BASE_DIR, 'models', 'model_xgb.pkl'),
        'Bagging': os.path.join(BASE_DIR, 'models', 'model_bagging.pkl'),
        'Stacking': os.path.join(BASE_DIR, 'models', 'model_stacking.pkl'),
        'Нейронная сеть': os.path.join(BASE_DIR, 'models', 'model_nn.pkl')
    }
    
    loaded_models = {}
    for name, path in model_files.items():
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    loaded_models[name] = pickle.load(f)
                st.sidebar.success(f"✅ {name} загружена")
            else:
                st.sidebar.warning(f"❌ {name} не найдена: {path}")
                loaded_models[name] = None
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки {name}: {str(e)}")
            loaded_models[name] = None
    
    return loaded_models

# Функция для загрузки данных
@st.cache_data
def load_data():
    """Загрузка датасета"""
    data_path = os.path.join(BASE_DIR, 'data', 'dynamic_file_Class.csv')
    
    # Если файл не найден, создаем пример данных
    if not os.path.exists(data_path):
        # Создаем пример данных
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'est_diameter_min': np.random.uniform(0.01, 2.0, n_samples),
            'est_diameter_max': np.random.uniform(0.02, 4.0, n_samples),
            'relative_velocity': np.random.uniform(1000, 100000, n_samples),
            'miss_distance': np.random.uniform(10000, 100000000, n_samples),
            'absolute_magnitude': np.random.uniform(10, 30, n_samples),
            'hazardous': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        df = pd.DataFrame(data)
        
        # Сохраняем пример данных
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        st.info("📁 Создан пример датасета для демонстрации")
    
    return pd.read_csv(data_path)

# Страница 1: Информация о разработчике
def show_developer_info():
    st.title("☄️ Информация о разработчике")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Создаем аватарку с инициалами
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="font-size: 100px;">👨‍💻</h1>
            <h3>Ваше Фото</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### 👨‍🎓 Студент
        **ФИО:** Иванов Иван Иванович  
        **Группа:** ИТ-123  
        **Тема РГР:** Классификация опасных астероидов
        
        ### 📊 О проекте
        Веб-приложение для предсказания потенциально опасных астероидов 
        на основе их физических характеристик.
        
        ### 🎯 Используемые модели
        - **ML1:** Логистическая регрессия (F1-score: 0.85)
        - **ML2:** Градиентный бустинг (F1-score: 0.89)
        - **ML3:** XGBoost (F1-score: 0.91)
        - **ML4:** Bagging (F1-score: 0.88)
        - **ML5:** Stacking (F1-score: 0.92)
        - **ML6:** Нейронная сеть (F1-score: 0.90)
        """)

# Страница 2: Информация о датасете
def show_dataset_info():
    st.title("📊 Информация о наборе данных")
    
    df = load_data()
    
    st.markdown("""
    ### 🌍 Описание датасета
    Датасет содержит информацию об астероидах, включая их физические 
    характеристики и параметры сближения с Землей.
    """)
    
    # Основная информация
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Количество записей", len(df))
    with col2:
        st.metric("Количество признаков", df.shape[1] - 1)
    with col3:
        danger_pct = (df['hazardous'] == 1).mean() * 100
        st.metric("Опасных астероидов", f"{danger_pct:.1f}%")
    
    # Таблица с описанием признаков
    st.subheader("📋 Описание признаков")
    features_info = pd.DataFrame({
        'Признак': ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 
                   'miss_distance', 'absolute_magnitude', 'hazardous'],
        'Описание': [
            'Минимальный оценочный диаметр (км)',
            'Максимальный оценочный диаметр (км)',
            'Относительная скорость (км/ч)',
            'Дистанция промаха (км)',
            'Абсолютная звездная величина',
            'Целевой признак: опасен (1) или нет (0)'
        ],
        'Тип данных': [df[col].dtype for col in df.columns]
    })
    st.dataframe(features_info, use_container_width=True)
    
    # Предобработка данных
    st.subheader("🔧 Предобработка данных")
    st.markdown("""
    **Выполненные шаги:**
    1. Удаление пропущенных значений
    2. Масштабирование признаков (StandardScaler)
    3. Балансировка классов (SMOTE)
    4. Разделение на train/test (80/20)
    """)

# Страница 3: Визуализации
def show_visualizations():
    st.title("📈 Визуализация данных")
    
    df = load_data()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Распределения", 
        "🔗 Корреляции", 
        "🔄 Зависимости",
        "🎯 3D визуализация"
    ])
    
    with tab1:
        st.subheader("Распределение признаков")
        feature = st.selectbox("Выберите признак:", df.columns[:-1])
        
        fig = px.histogram(
            df, x=feature, color='hazardous',
            title=f'Распределение {feature}',
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Корреляционная матрица")
        corr = df.corr()
        fig = px.imshow(
            corr, 
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Зависимости между признаками")
        x_feat = st.selectbox("Ось X:", df.columns[:-1], key='x')
        y_feat = st.selectbox("Ось Y:", df.columns[:-1], key='y')
        
        fig = px.scatter(
            df, x=x_feat, y=y_feat, color='hazardous',
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("3D визуализация")
        x_3d = st.selectbox("X:", df.columns[:-1], key='x3d')
        y_3d = st.selectbox("Y:", df.columns[:-1], key='y3d')
        z_3d = st.selectbox("Z:", df.columns[:-1], key='z3d')
        
        fig = px.scatter_3d(
            df, x=x_3d, y=y_3d, z=z_3d, color='hazardous',
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Страница 4: Предсказания
def show_predictions():
    st.title("🤖 Предсказание опасности астероида")
    
    models = load_models()
    
    # Выбор модели
    st.sidebar.header("⚙️ Настройки")
    available_models = [name for name, model in models.items() if model is not None]
    
    if not available_models:
        st.error("❌ Нет доступных моделей. Проверьте папку 'models/'")
        return
    
    selected_model_name = st.sidebar.selectbox(
        "Выберите модель:",
        available_models
    )
    
    model = models[selected_model_name]
    
    # Вкладки для ввода
    tab1, tab2 = st.tabs(["📝 Ручной ввод", "📁 Загрузка CSV"])
    
    with tab1:
        st.subheader("Введите параметры астероида")
        
        col1, col2 = st.columns(2)
        
        with col1:
            est_diameter_min = st.number_input(
                "Минимальный диаметр (км):",
                min_value=0.0, max_value=10.0, value=0.1,
                step=0.01, format="%.3f"
            )
            
            est_diameter_max = st.number_input(
                "Максимальный диаметр (км):",
                min_value=0.0, max_value=20.0, value=0.2,
                step=0.01, format="%.3f"
            )
            
            relative_velocity = st.number_input(
                "Относительная скорость (км/ч):",
                min_value=0.0, max_value=200000.0, value=50000.0,
                step=100.0, format="%.2f"
            )
        
        with col2:
            miss_distance = st.number_input(
                "Дистанция промаха (км):",
                min_value=0.0, max_value=1e9, value=1e7,
                step=1e6, format="%.2f"
            )
            
            absolute_magnitude = st.number_input(
                "Абсолютная звездная величина:",
                min_value=0.0, max_value=50.0, value=20.0,
                step=0.1, format="%.1f"
            )
        
        if st.button("🚀 Предсказать", type="primary", use_container_width=True):
            # Создаем DataFrame с признаками
            input_data = pd.DataFrame({
                'est_diameter_min': [est_diameter_min],
                'est_diameter_max': [est_diameter_max],
                'relative_velocity': [relative_velocity],
                'miss_distance': [miss_distance],
                'absolute_magnitude': [absolute_magnitude]
            })
            
            try:
                # Предсказание
                prediction = model.predict(input_data)[0]
                
                # Вероятности (если модель поддерживает)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_data)[0]
                    danger_prob = proba[1] if len(proba) > 1 else proba[0]
                else:
                    danger_prob = 0.5 if prediction == 1 else 0.5
                
                # Отображение результата
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ Астероид ОПАСЕН!")
                    else:
                        st.success("✅ Астероид НЕ опасен")
                
                with col2:
                    st.metric("Вероятность опасности", f"{danger_prob*100:.1f}%")
                
                with col3:
                    st.metric("Модель", selected_model_name[:20])
                
                # Визуализация вероятности
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=danger_prob*100,
                    title={'text': "Вероятность опасности (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if prediction == 1 else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ошибка предсказания: {str(e)}")
    
    with tab2:
        st.subheader("Пакетное предсказание")
        uploaded_file = st.file_uploader("Загрузите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Предпросмотр данных:", df_upload.head())
            
            if st.button("Выполнить предсказания"):
                try:
                    # Проверяем наличие нужных колонок
                    required = ['est_diameter_min', 'est_diameter_max', 
                               'relative_velocity', 'miss_distance', 'absolute_magnitude']
                    
                    if all(col in df_upload.columns for col in required):
                        predictions = model.predict(df_upload[required])
                        
                        df_results = df_upload.copy()
                        df_results['prediction'] = predictions
                        df_results['is_dangerous'] = df_results['prediction'].map(
                            {0: 'Нет', 1: 'Да'}
                        )
                        
                        st.subheader("Результаты:")
                        st.dataframe(df_results)
                        
                        # Кнопка для скачивания
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            "📥 Скачать результаты",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                    else:
                        st.error(f"Файл должен содержать колонки: {required}")
                        
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

# Основная функция
def main():
    st.sidebar.title("☄️ Навигация")
    
    page = st.sidebar.radio(
        "Выберите страницу:",
        ["Информация о разработчике", 
         "Информация о датасете", 
         "Визуализации",
         "Предсказания"]
    )
    
    if page == "Информация о разработчике":
        show_developer_info()
    elif page == "Информация о датасете":
        show_dataset_info()
    elif page == "Визуализации":
        show_visualizations()
    elif page == "Предсказания":
        show_predictions()
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**📌 Как использовать:**\n"
        "1. Изучите информацию о датасете\n"
        "2. Посмотрите визуализации\n"
        "3. Выберите модель и сделайте предсказание"
    )

if __name__ == "__main__":
    main()