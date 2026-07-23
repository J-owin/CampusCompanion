from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path


data_folder = Path("data")
pdf_files = data_folder.glob("*.pdf")
documents = []

for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    pages = loader.load()
    documents.extend(pages)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
print(f"Loaded {len(documents)} pages from all PDFs.")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
print("vector database created successfully!")