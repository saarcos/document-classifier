from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
import joblib
import os
import uuid
import shutil
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from fastapi.middleware.cors import CORSMiddleware


# --- SQLAlchemy Imports (Persistencia) ---
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from utils.pdf_reader import extract_text_from_pdf

load_dotenv()

# --- CONFIGURACIÓN DE BASE DE DATOS SQLITE ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./document_intelligence.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo de Tabla para Metadatos
class DocumentDB(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    label = Column(String)
    word_count = Column(Integer)
    file_url = Column(String) # Ruta para que Angular acceda
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Dependencia para obtener la sesión de DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="Document Intelligence API")

# Carpetas de Almacenamiento
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

origins = [
    "http://localhost:4200",   # El puerto por defecto de Angular
    "http://127.0.0.1:4200",  
]

# Montar carpeta estática: permite acceder a archivos vía http://localhost:8000/files/...
app.mount("/files", StaticFiles(directory=STORAGE_DIR), name="files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Permite Angular
    allow_credentials=True,
    allow_methods=["*"],               # Permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],               # Permite todos los headers (Content-Type, etc.)
)

MODEL_PATH = "artifacts/document_classifier.joblib"

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Loading ML model...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")

embeddings = OpenAIEmbeddings()

# Inicializar Base de Datos Vectorial (Se crea una carpeta 'db' en tu proyecto)
vector_db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

# ----- Request / Response Schemas -----

class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    label: str

class UploadResponse(BaseModel):
    id: str
    filename: str
    label: str
    word_count: int
    file_url: str
    class Config:
        from_attributes = True

# ----- Endpoints -----

@app.get("/")
def root():
    return {"status": "ok", "message": "Document Intelligence API running"}

@app.get("/documents", response_model=List[UploadResponse])
def list_documents(db: Session = Depends(get_db)):
    """Retorna todos los documentos guardados en la base de datos"""
    return db.query(DocumentDB).all()

@app.post("/classify", response_model=ClassificationResponse)
def classify_document(req: ClassificationRequest):
    text = req.text

    if not text or len(text.strip()) < 10:
        return {"label": "Unknown"}

    prediction = model.predict([text])[0]
    print("Prediction: ", prediction)

    return {"label": prediction}            
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    doc_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    try:
        # 1. Guardar archivo temporal
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extraer texto
        text = extract_text_from_pdf(temp_path)
        
        if not text or len(text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        prediction = model.predict([text])[0] # Ej: "Invoice"

        # 3. Organizar archivo físicamente por carpeta de categoría
        category_dir = os.path.join(STORAGE_DIR, prediction)
        os.makedirs(category_dir, exist_ok=True)
        
        final_file_name = f"{doc_id}_{file.filename}"
        final_path = os.path.join(category_dir, final_file_name)
        shutil.move(temp_path, final_path) # Movemos del temp al storage oficial

        # 4. RAG: Indexar en ChromaDB
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        docs = [Document(page_content=t, metadata={"doc_id": doc_id}) for t in chunks]
        vector_db.add_documents(docs)

        # 5. Persistencia en SQLite
        file_url = f"http://localhost:8000/files/{prediction}/{final_file_name}"
        db_doc = DocumentDB(
            id=doc_id,
            filename=file.filename,
            label=prediction,
            word_count=len(text.split()),
            file_url=file_url
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)

        return db_doc

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 6. Limpieza: Eliminar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
@app.get("/ask")
async def ask_question(question: str, document_id: str = None):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    #Filtro si envió un id
    search_kwargs = {}
    if document_id:
        search_kwargs["filter"] = {"doc_id": document_id}
    system_prompt = (
        "You are an expert business document analyst. "
        "Use the provided context to answer the user's question accurately. "
        "If the answer is not contained within the context, politely state that you do not have enough information. "
        "Keep answers concise and professional.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    retriever = vector_db.as_retriever(search_kwargs = search_kwargs)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 🔥 LCEL / Runnable RAG Chain moderna
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    response = rag_chain.invoke(question)

    return {
        "answer": response.content
    }