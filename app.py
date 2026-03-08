import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Классификатор астероидов",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Класс модели
class MyClassificationModel(nn.Module):
    # любая модель в PyTorch - это набор слоев
    # при этом, мы сами определяем порядок их выполнения
    # в конструкторе мы задаем набор слоев с указанием параметров
    def __init__(self):
        super(MyClassificationModel, self).__init__()

        self.first_linear = nn.Linear(6, 32)
        # определяем первый слой ReLU
        self.first_relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.second_linear = nn.Linear(32, 64)
        self.second_relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.third_linear = nn.Linear(64, 32)
        self.third_relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fourth_linear = nn.Linear(32, 6)
        self.fourth_relu = nn.ReLU()
        self.dropout4 = nn.Dropout(0.1)
        self.fifth_linear = nn.Linear(6, 1)

        #self.softmax = nn.Softmax(dim=1)

        self.sigmoid = nn.Sigmoid()

    # в методе forward мы определяем, как слои будут связаны друг с другом
    def forward(self, x):
        # y - результат выполнения первого слоя
        y = self.first_linear(x)
        # в теперь продолжаем накидывать оставшиеся слои
        y = self.first_relu(y)
        y = self.dropout1(y)
        y = self.second_linear(y)
        y = self.second_relu(y)
        y = self.dropout2(y)
        y = self.third_linear(y)
        y = self.third_relu(y)
        y = self.dropout3(y)
        y = self.fourth_linear(y)
        y = self.fourth_relu(y)
        y = self.dropout4(y)
        y = self.fifth_linear(y)

        y = self.sigmoid(y)
        #y = self.softmax(y)
        #self.sigmoid = nn.Sigmoid()
        return y

# Загрузка моделей
@st.cache_resource
def load_models():
    models = {}
    models_dir = os.path.join(BASE_DIR, 'models')
    
    model_files = [
        ('Логистическая регрессия', 'model_logreg.pkl'),
        ('Градиентный бустинг', 'model_gb.pkl'),
        ('XGBoost', 'model_xgb.pkl'),
        ('Bagging', 'model_bagging.pkl'),
        ('Stacking', 'model_stacking.pkl'),
        ('KNN', 'model_knn.pkl')
    ]
    
    for name, filename in model_files:
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
            except:
                try:
                    with open(path, 'rb') as f:
                        models[name] = pickle.load(f, fix_imports=True, encoding='latin1')
                except:
                    pass
    
    nn_path = os.path.join(models_dir, 'model_nn.pth')
    if os.path.exists(nn_path):
        try:
            torch.serialization.add_safe_globals([MyClassificationModel, nn.Linear, nn.ReLU, nn.Dropout, nn.Sigmoid, nn.Sequential])
            models['Нейронная сеть'] = torch.load(nn_path, map_location='cpu', weights_only=True).eval()
        except:
            try:
                models['Нейронная сеть'] = torch.load(nn_path, map_location='cpu', weights_only=False).eval()
            except:
                pass
    return models

# Загрузка данных
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'dynamic_file_Class.csv'))
        if 'avg_diameter' not in df.columns:
            df['avg_diameter'] = (df['est_diameter_min'] + df['est_diameter_max']) / 2
        return df
    except:
        return pd.DataFrame()

def predict(model, input_data, model_name):
    """
    Универсальная функция предсказания
    """
    try:
        # Для нейронной сети
        if 'Нейронная сеть' in model_name or isinstance(model, torch.nn.Module):
            # Убеждаемся, что данные в правильном порядке
            feature_order = ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 
                            'miss_distance', 'absolute_magnitude', 'avg_diameter']
            
            # Проверяем, что все признаки есть
            for feat in feature_order:
                if feat not in input_data.columns:
                    if feat == 'avg_diameter':
                        input_data[feat] = (input_data['est_diameter_min'] + input_data['est_diameter_max']) / 2
                    else:
                        st.error(f"Отсутствует признак: {feat}")
                        return None, None
            
            # Берем признаки в правильном порядке
            X = input_data[feature_order].values
            
            with torch.no_grad():
                # Преобразуем в тензор
                X_tensor = torch.tensor(X, dtype=torch.float32)
                
                # Получаем предсказание
                output = model(X_tensor)
                
                # Для одного примера
                if len(X) == 1:
                    prob = output.item() if output.numel() == 1 else output[0].item()
                    pred = 1 if prob > 0.5 else 0
                    return pred, prob
                else:
                    # Для нескольких примеров
                    probs = output.numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                    return preds, probs
        
        # Для sklearn моделей
        else:
            # Проверяем ожидаемые признаки
            if hasattr(model, 'feature_names_in_'):
                expected = list(model.feature_names_in_)
                
                # Проверяем наличие всех признаков
                for feat in expected:
                    if feat not in input_data.columns:
                        if feat == 'avg_diameter':
                            input_data[feat] = (input_data['est_diameter_min'] + input_data['est_diameter_max']) / 2
                        else:
                            st.error(f"Модель ожидает признак: {feat}")
                            return None, None
                
                # Выбираем признаки в правильном порядке
                X = input_data[expected]
            else:
                # Если нет информации о признаках, используем все
                X = input_data
            
            # Предсказание
            preds = model.predict(X)
            
            # Вероятности
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                if len(probs.shape) == 2 and probs.shape[1] == 2:
                    # Для бинарной классификации
                    danger_probs = probs[:, 1]
                else:
                    danger_probs = probs.flatten()
            else:
                danger_probs = np.array([0.5] * len(preds))
            
            # Для одного примера возвращаем скаляры
            if len(preds) == 1:
                return preds[0], danger_probs[0]
            else:
                return preds, danger_probs
            
    except Exception as e:
        st.error(f"Ошибка в предсказании: {str(e)}")
        return None, None

# Основной интерфейс
def main():
    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
    "Выберите страницу:", 
    ["Разработчик", "Данные", "Визуализация", "Предсказание"],
    label_visibility="collapsed"
)
    
    models = load_models() if page in ["Предсказание"] else {}
    df = load_data() if page in ["Данные", "Визуализация"] else pd.DataFrame()
    
    # СТРАНИЦА 1: РАЗРАБОТЧИК
    if page == "Разработчик":
        st.title("Информация о разработчике")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Kokorin.png", width=100)
        with col2:
            st.markdown("""
            **ФИО:** Кокорин Артём Владимирович 
            **Группа:** ФИТ-231  
            **Тема РГР:** Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных
            """)
    
    # СТРАНИЦА 2: ДАННЫЕ
    elif page == "Данные":
        st.title("Информация о наборе данных")
        
        if df.empty:
            st.error("Данные не загружены")
            return
        
        st.markdown("""
        ### Описание предметной области
        Датасет содержит информацию об астероидах. 
        Цель - классификация астероидов на опасные (hazardous=1) и неопасные (hazardous=0) 
        на основе их физических характеристик.
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего записей", len(df))
        with col2:
            st.metric("Признаков", df.shape[1]-1)
        with col3:
            st.metric("Опасных", f"{(df['hazardous']==1).sum()} ({(df['hazardous']==1).mean()*100:.1f}%)")
        with col4:
            st.metric("Безопасных", f"{(df['hazardous']==0).sum()} ({(df['hazardous']==0).mean()*100:.1f}%)")
        
        st.subheader("Описание признаков")
        desc_df = pd.DataFrame({
            'Признак': ['est_diameter_min', 'est_diameter_max', 'avg_diameter', 'relative_velocity', 
                       'miss_distance', 'absolute_magnitude', 'hazardous'],
            'Описание': ['Минимальный диаметр', 'Максимальный диаметр', 'Средний диаметр', 
                        'Относительная скорость', 'Дистанция промаха', 'Абсолютная магнитуда', 'Опасность'],
            'Ед. изм.': ['км', 'км', 'км', 'км/ч', 'км', 'магн.', '0/1']
        })
        st.dataframe(desc_df, use_container_width=True)
        
        st.subheader("Предобработка данных")
        st.markdown("""
        **Выполненные шаги:**
        1. Удаление пропущенных значений
        2. Добавление признака avg_diameter
        3. Балансировка классов (SMOTE)
        4. Разделение на train/test (80/20)
        """)
        
        st.subheader("EDA - Основные статистики")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Пример данных")
        st.dataframe(df.head(10), use_container_width=True)
    
    # СТРАНИЦА 3: ВИЗУАЛИЗАЦИЯ
    elif page == "Визуализация":
        st.title("Визуализация зависимостей")
        
        if df.empty:
            st.error("Данные не загружены")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs(["Распределения", "Корреляции", "Зависимости", "3D"])
        
        with tab1:
            feat = st.selectbox("Признак", df.columns[:-1], key='hist')
            fig = px.histogram(df, x=feat, color='hazardous', barmode='overlay', 
                              color_discrete_map={0: 'blue', 1: 'red'},
                              title=f'Распределение {feat}')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.imshow(df.corr(), text_auto=True, aspect='auto', 
                          color_continuous_scale='RdBu_r', title='Корреляционная матрица')
            st.plotly_chart(fig, use_container_width=True)
            
            corr_target = df.corr()['hazardous'].drop('hazardous').sort_values()
            fig2 = px.bar(x=corr_target.values, y=corr_target.index, orientation='h',
                         title='Корреляция с опасностью', color=corr_target.values,
                         color_continuous_scale='RdBu_r')
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            x_feat = st.selectbox("Ось X", df.columns[:-1], key='x')
            y_feat = st.selectbox("Ось Y", df.columns[:-1], key='y')
            fig = px.scatter(df, x=x_feat, y=y_feat, color='hazardous',
                           color_discrete_map={0: 'blue', 1: 'red'},
                           title=f'{y_feat} vs {x_feat}')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            x3d = st.selectbox("X", df.columns[:-1], key='x3d')
            y3d = st.selectbox("Y", df.columns[:-1], key='y3d')
            z3d = st.selectbox("Z", df.columns[:-1], key='z3d')
            fig = px.scatter_3d(df, x=x3d, y=y3d, z=z3d, color='hazardous',
                               color_discrete_map={0: 'blue', 1: 'red'},
                               title=f'3D проекция')
            st.plotly_chart(fig, use_container_width=True)
    
    # ПРЕДСКАЗАНИЕ
    elif page == "Предсказание":
        st.title("Предсказание опасности астероида")
        
        if not models:
            st.error("Модели не загружены. Проверьте папку models/")
            return
        
        model_name = st.sidebar.selectbox("Выберите модель", list(models.keys()))
        model = models[model_name]
        
        tab1, tab2 = st.tabs(["Ручной ввод", "Загрузка CSV"])
        
        #  Ручной ввод
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                d_min = st.number_input("Мин. диаметр (км)", 0.0, 100.0, 0.1, step=0.1)
                d_max = st.number_input("Макс. диаметр (км)", 0.0, 200.0, 0.2, step=0.1)
                velocity = st.number_input("Скорость (км/ч)", 0.0, 300000.0, 50000.0, step=1000.0)
            with col2:
                distance = st.number_input("Дистанция (км)", 0.0, 1e9, 1e7, step=1e6)
                magnitude = st.number_input("Магнитуда", 0.0, 50.0, 20.0, step=0.1)
            
            if st.button("Предсказать", use_container_width=True):
                avg_d = (d_min + d_max) / 2
                input_df = pd.DataFrame([[d_min, d_max, velocity, distance, magnitude, avg_d]], 
                                       columns=['est_diameter_min', 'est_diameter_max', 
                                               'relative_velocity', 'miss_distance', 
                                               'absolute_magnitude', 'avg_diameter'])
                
                pred, prob = predict(model, input_df, model_name)
                
                if pred is not None:
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if pred == 1:
                            st.error("ОПАСЕН")
                        else:
                            st.success("НЕ ОПАСЕН")
                    with col2:
                        st.metric("Вероятность опасности", f"{prob*100:.1f}%")
                    with col3:
                        st.metric("Модель", model_name[:15])
                    
                    #fig = go.Figure(go.Indicator(
                    #    mode="gauge+number",
                    #    value=prob*100,
                    #    title="Вероятность опасности (%)",
                    #    gauge={'axis': {'range': [0, 100]},
                    #          'bar': {'color': 'red' if pred == 1 else 'green'},
                    #          'steps': [{'range': [0, 50], 'color': 'lightgreen'},
                    #                   {'range': [50, 100], 'color': 'lightcoral'}]}
                    #))
                    #fig.update_layout(height=250)
                    #st.plotly_chart(fig, use_container_width=True)
        
        # Вкладка 2: Загрузка CSV
        with tab2:
            st.markdown("""
            **Формат CSV:**
            - est_diameter_min, est_diameter_max, relative_velocity, miss_distance, absolute_magnitude
            """)
            
            uploaded = st.file_uploader("Выберите CSV файл", type=['csv'])
            
            if uploaded:
                df_upload = pd.read_csv(uploaded)
                if 'avg_diameter' not in df_upload.columns:
                    df_upload['avg_diameter'] = (df_upload['est_diameter_min'] + df_upload['est_diameter_max']) / 2
                
                st.write("Предпросмотр:", df_upload.head())
                
                if st.button("Выполнить предсказание"):
                    results = []
                    for _, row in df_upload.iterrows():
                        input_df = pd.DataFrame([row])
                        pred, prob = predict(model, input_df, model_name)
                        results.append({'prediction': pred, 'probability': prob})
                    
                    res_df = df_upload.copy()
                    res_df['prediction'] = [r['prediction'] for r in results]
                    res_df['probability'] = [r['probability'] for r in results]
                    res_df['status'] = res_df['prediction'].map({0: 'Безопасен', 1: 'ОПАСЕН'})
                    
                    st.subheader("Результаты")
                    st.dataframe(res_df)
                    
                    # Статистика
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Опасных", (res_df['prediction']==1).sum())
                    col2.metric("Безопасных", (res_df['prediction']==0).sum())
                    col3.metric("Всего", len(res_df))
                    
                    # График
                    fig = px.pie(res_df, names='status', title='Распределение предсказаний')
                    st.plotly_chart(fig)
                    
                    # Скачивание
                    csv = res_df.to_csv(index=False)
                    st.download_button("Скачать результаты", csv, "predictions.csv", "text/csv")

if __name__ == "__main__":
    main()