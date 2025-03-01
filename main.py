import pandas as pd
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document

# Установка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка punkt_tab для NLTK
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def load_data(file_path):
    """Загружает данные из CSV."""
    doc = pd.read_csv(file_path)
    loader = DataFrameLoader(doc, page_content_column="answer")
    return loader.load()

def split_documents(documents):
    """Адаптивное разбиение документов на чанки."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, length_function=len)
    split_docs = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc.page_content)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= 600:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    split_docs.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
                current_chunk = sentence + " "
        if current_chunk:
            split_docs.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
    return split_docs

def setup_rag(documents):
    """Настраивает RAG-систему: FAISS и модель."""
    # Векторизация
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/LaBSE-en-ru", model_kwargs={"device": device})
    split_docs = split_documents(documents)
    db = FAISS.from_documents(split_docs, embeddings)
    
    # Настройка модели
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512, truncation=True)
    local_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Настройка цепочки
    template = """Answer the question based only on the following context:\n{context}\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = db.as_retriever(search_type="similarity", k=3)
    
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | local_llm
        | StrOutputParser()
    )
    
    return chain

def safe_invoke(chain, query):
    """Безопасный запуск цепочки."""
    try:
        return chain.invoke(query)
    except Exception as e:
        return f"Ошибка: {str(e)}"

def main():
    """Основная функция для запуска."""
    # Загружаем данные
    file_path = "my_dataset.csv"
    print("Загрузка данных...")
    documents = load_data(file_path)
    
    # Настраиваем RAG
    print("Настройка RAG-системы...")
    chain = setup_rag(documents)
    
    # Простой консольный интерфейс
    print("RAG-система готова! Введите запрос (или 'exit' для выхода):")
    while True:
        query = input("> ")
        if query.lower() == "exit":
            break
        result = safe_invoke(chain, query)
        print("Ответ:", result)

if __name__ == "__main__":
    main()