# 🔍 Antiplagiarism System for Engineering 2D Drawings

Данный проект представляет собой систему для автоматического выявления плагиата в инженерных чертежах (формат `.dxf`). Система строит графовое представление на основе геометрических примитивов, использует алгоритм RANSAC с двойной нормализацией и нейросетевые модели GCN и GIN для оценки степени совпадения. Предусмотрена визуализация совпадений, ручная разметка и подсистема сбора статистики. Имеется как веб-интерфейс, так и десктопное приложение.

## 📁 Структура проекта

```plaintext
ready_project/
│
├── desktop_interface.py         # Десктопный интерфейс (DearPyGUI)
├── requirements.txt             # Зависимости проекта
├── README.md                    # Документация
│
├── app/                         # Web-интерфейс на FastAPI
│   ├── main.py                  # Точка входа
│   ├── database.py              # Работа с базой данных (SQLite)
│   └── routers.py               # Роутинг (загрузка, сравнение, просмотр)
│
├── core/                        # Логика сравнения и пайплайн
│   ├── pipeline_runner.py       # Основной пайплайн сравнения
│   ├── comparer_graphs.py       # Сравнение графов (RANSAC, GCN, GIN)
│   ├── convert_dxf_graphs.py    # Парсинг DXF → графы
│   ├── statistics_utils.py      # Подсчёт и логирование метрик
│   └── models.py                # Вспомогательные структуры данных
│
├── data/                        # Датасет (DXF-файлы, графы, веса)
│   ├── comparisons.db           # SQLite база с результатами
│   └── [v1_0_x]/                # Папки версий: dxf, pkl, pt-файлы
│       ├── dxf/                 # Исходные чертежи
│       ├── pkl/                 # Графы в формате pickle
│       └── pt/                  # Графы в формате PyTorch для обучения
│
├── models/                      # Обученные модели
│   ├── best_model_gcn.pt
│   └── best_model_gin.pt
│
├── scripts/                     # Утилиты и тестовые пайплайны
│   ├── generate_png.py          # Генерация PNG-сравнений
│   ├── test_pipeline.py         # Быстрый запуск пайплайна
│   └── test/                    # Тестовые графы (pkl/pt)
│
├── static/                      # Статические файлы
│   ├── fonts/                   # Шрифты для UI
│   └── histograms/              # Гистограммы статистики
│
├── templates/                   # HTML-шаблоны Jinja2
│   ├── index.html
│   ├── stats.html
│   └── versions.html
│
└── test/                        # Тестовые скрипты и визуализация
    ├── check_db.py              # Проверка базы
    ├── delete_rows.py           # Очистка данных
    ├── interface_test.py        # Проверка интерфейса
    ├── test_ransac_visualization.py
    └── ransac_image/            # Картинки RANSAC и отладка
