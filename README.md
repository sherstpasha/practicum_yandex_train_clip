# CLIP Product Search

Практический проект Яндекс Практикума по разработке системы поиска товаров на основе мультимодальной модели CLIP.

## Описание

Интеллектуальная система поиска товаров по текстовому описанию с использованием fine-tuned модели CLIP. Система обеспечивает семантический поиск изображений товаров, понимая связь между текстовыми запросами и визуальным содержимым.

## Технологический стек

- **Vision-Language Model**: OpenAI CLIP (ViT-Base-Patch32)
- **Vector Search**: FAISS (IndexFlatIP)
- **Fine-tuning**: PyTorch, Transformers
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Data Processing**: Pandas, NumPy, PIL

## Метрики качества

- **Baseline CLIP score**: 30.37
- **Fine-tuned CLIP score**: 34.42 (+13%)
- **Среднее время поиска**: ~30 мс

## Структура проекта

```
.
├── demo/                          # Демонстрационное приложение
│   ├── gradio_app.py              # Gradio интерфейс
│   ├── server.py                  # FastAPI сервер
│   ├── retrieval_system.py        # Класс системы поиска
├── archive/                       # Данные
│   ├── data.csv                   # Метаданные товаров
│   └── data/                      # Изображения товаров
├── clip_best.pth                  # Fine-tuned модель
├── clip_image.index               # FAISS индекс
├── solution.ipynb                 # Jupyter notebook с полным анализом
├── gradio_app.py                  # Standalone Gradio приложение
├── requirements.txt               # Зависимости проекта
└── README.md                      # Этот файл
```

## Установка

```bash
pip install -r requirements.txt
```