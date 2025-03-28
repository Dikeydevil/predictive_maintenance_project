# 📈 Проект: Предиктивное обслуживание оборудования (бинарная классификация)

## 🔍 Описание проекта
Разработано Streamlit-приложение, которое:
- Загружает и обрабатывает данные по оборудованию
- Обучает модель бинарной классификации (Logistic Regression)
- Предсказывает вероятность отказа оборудования
- Визуализирует метрики (Accuracy, ROC-AUC, Confusion Matrix)
- Включает презентацию проекта прямо в приложении

## 📊 Используемый датасет
**AI4I 2020 Predictive Maintenance Dataset**  
Источник: [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)

Содержит данные о температуре, износе инструмента, скорости и других параметрах оборудования.  
Целевая переменная: `Machine failure` (0 — нет отказа, 1 — произошёл отказ)

## 🧰 Установка и запуск

🔧 Шаг 1: Клонировать репозиторий
```bash
git clone https://github.com/Dikeydevil/predictive_maintenance_project.git
cd predictive_maintenance_project
```
📦 Шаг 3: Установить зависимости
```bash
pip install -r requirements.txt
```
🚀 Шаг 4: Запустить приложение
```bash
streamlit run app.py
