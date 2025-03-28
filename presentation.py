import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("📽️ Презентация проекта")

    # Markdown-слайды
    presentation_md = """
# Бинарная классификация для предиктивного обслуживания оборудования
---

## 🔍 Цель проекта
- Построить модель для предсказания отказа оборудования.
- Оформить результат в виде Streamlit-приложения.

---

## 📊 Этапы работы
1. Загрузка и предобработка данных
2. Масштабирование и кодирование признаков
3. Обучение модели (Logistic Regression)
4. Оценка (Accuracy, ROC-AUC, Confusion Matrix)
5. Streamlit-интерфейс
6. Презентация проекта

---

## 📈 Использованные библиотеки
- pandas
- scikit-learn
- streamlit
- matplotlib
- seaborn
- xgboost
- streamlit-reveal-slides

---

## 🧠 Результаты
- Accuracy ~ 95%
- ROC-AUC ~ 0.98
- Простое и понятное приложение

---

## ✅ Возможные улучшения
- Добавить больше моделей
- Сделать автообновление результатов
- Использовать реальные данные с сенсоров

---

## 🔗 Спасибо за внимание!
    """

    # Панель с настройками
    with st.sidebar:
        st.header("Настройки слайдов")
        theme = st.selectbox("🎨 Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        transition = st.selectbox("🌀 Переход", ["slide", "convex", "concave", "zoom", "none"])
        height = st.slider("📐 Высота", 400, 1000, 600)

    # Отображение слайдов
    rs.slides(
        presentation_md,
        height=height,
        theme=theme,
        config={"transition": transition},
        markdown_props={"data-separator-vertical": "^--$"},
    )
