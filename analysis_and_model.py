import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

def analysis_and_model_page():
    st.title("📊 Анализ данных и модель")

    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Предобработка
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Масштабирование
        numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                              'Torque [Nm]', 'Tool wear [min]']
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Разделение
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        feature_order = X.columns.tolist()  # сохраняем порядок признаков
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Оценка
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_rep = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Метрики
        st.header("✅ Результаты модели")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**ROC-AUC:** {roc_auc:.4f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(class_rep)

        # ROC-кривая
        st.subheader("ROC-кривая")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig2 = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(fig2)

        # Предсказания на новых данных
        st.header("🔍 Предсказание отказа")
        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
            with col2:
                air_temp = st.number_input("Air temperature [K]")
            with col3:
                process_temp = st.number_input("Process temperature [K]")

            col4, col5 = st.columns(2)
            with col4:
                speed = st.number_input("Rotational speed [rpm]")
            with col5:
                torque = st.number_input("Torque [Nm]")

            tool_wear = st.number_input("Tool wear [min]")

            submitted = st.form_submit_button("Предсказать")

            if submitted:
                type_encoded = {'L': 0, 'M': 1, 'H': 2}[product_type]
                # Собираем в том же порядке, как было при обучении
                input_data = {
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear],
                    'Type': [type_encoded]
                }
                input_df = pd.DataFrame(input_data)
                input_df[numerical_features] = scaler.transform(input_df[numerical_features])
                input_df = input_df[feature_order]  # фикс: тот же порядок, что и на обучении

                pred = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0][1]

                st.write(f"**Результат:** {'❌ Отказ' if pred == 1 else '✅ Всё в порядке'}")
                st.write(f"**Вероятность отказа:** {pred_proba:.4f}")
