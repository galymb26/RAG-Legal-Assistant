# RAG Legal Assistant

Это пет-проект, реализующий систему Retrieval-Augmented Generation (RAG) для анализа юридических вопросов и автоматической генерации ответов на русском языке. Система использует FAISS для поиска релевантных документов, NLTK для разбиения текста и модель LLaMA 3.2 для генерации ответов.

## Возможности
- Очистка и нормализация юридических текстов из CSV-файла.
- Адаптивное разбиение текста на чанки с сохранением целостности предложений.
- Поиск релевантных документов с помощью FAISS.
- Генерация ответов на основе контекста с использованием LLaMA 3.2.
