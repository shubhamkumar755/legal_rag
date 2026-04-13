import logging
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Query, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, constr
from jose import JWTError, jwt
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from nim_db import save_chat, init_db, close_db, get_db_pool, get_history, get_chat_history
import httpx
from contextlib import asynccontextmanager
from typing import Optional
import uuid
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

persist_dir = "./nim_chroma_store"
os.makedirs(persist_dir, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

app = FastAPI(lifespan=lifespan)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def extract_text_from_url(url: str):
    try:
        logger.info(f"Extracting text from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return "\n".join([line.get_text(strip=True) for line in soup.find_all(['h1', 'h2', 'h3', 'p'])])
    except Exception as e:
        logger.error(f"URL extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"URL processing failed: {str(e)}")

def extract_text_from_pdf(pdf_path: str):
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")

def extract_text_from_image(image_path: str):
    try:
        logger.info(f"Extracting text from image: {image_path}")
        pages = convert_from_path(image_path)
        return "\n".join([pytesseract.image_to_string(page) for page in pages])
    except Exception as e:
        logger.error(f"Image extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

def split_and_store(text: str, source: str, session_id: str):
    try:
        logger.info(f"Splitting and storing data for session {session_id}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        try:
            existing_store = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir,
                collection_name=session_id
            )
            existing_store.delete_collection()
        except Exception as e:
            logger.warning(f"No existing collection to delete or error in deletion: {str(e)}")

        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=session_id,
            metadatas=[{"source": source}] * len(chunks)
        )
    except Exception as e:
        logger.error(f"Vector storage failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vector storage failed: {str(e)}")

class QueryReq(BaseModel):
    session_id: str
    question: str

class UserInfo(BaseModel):
    email: EmailStr
    password: constr(min_length=8) # type: ignore

@app.post("/load")
async def load_data(
    session_id: str = Form(...),
    mode: int = Form(...),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    logger.info(f"Load requested: mode={mode}, session={session_id}, user={user_id}")
    
    try:
        text = ""
        source = ""
        
        if mode == 3:  # URL mode
            if not url:
                raise HTTPException(status_code=400, detail="URL is required for URL mode")
            text = extract_text_from_url(url)
            source = url
            
        else:  # File mode
            if not file:
                raise HTTPException(status_code=400, detail="File is required for file mode")

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            try:
                content = await file.read()
                temp_file.write(content)
                temp_file.close()

                if mode == 1:  # PDF
                    text = extract_text_from_pdf(temp_file.name)
                elif mode == 2:  # Image
                    text = extract_text_from_image(temp_file.name)
                else:
                    raise HTTPException(status_code=400, detail="Invalid mode. Use 1 for PDF, 2 for Image, 3 for URL")

                source = file.filename or "uploaded_file"
                
            finally:
                os.unlink(temp_file.name)

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the source")

        # Store the text in vector database
        split_and_store(text, source, session_id)
        
        # Log successful document addition to chat history
        await save_chat(session_id, "system", f"Document '{source}' has been uploaded and is now available for questions.", user_id)

        return {
            "message": "Document loaded successfully and is now available for questions",
            "source": source,
            "user_id": user_id,
            "text_length": len(text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/chat")
async def query_data(request: QueryReq, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        logger.info(f"Chat request from user={user_id}, session={request.session_id}")

        # Get chat history
        db_pool = get_db_pool()
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content FROM chat_history WHERE session_id = $1 ORDER BY timestamp ASC",
                request.session_id
            )

        # Try to get document context, but don't fail if it doesn't exist
        context = ""
        retrieved_docs = []
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir,
                collection_name=request.session_id
            )
            
            # Check if collection actually exists and has documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(request.question)
            
            if retrieved_docs and len(retrieved_docs) > 0:
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                logger.info(f"Found {len(retrieved_docs)} relevant document chunks")
            else:
                logger.info("No relevant document chunks found")
                
        except Exception as e:
            # Log the warning but continue without context
            logger.warning(f"Could not retrieve document context (this is normal if no documents uploaded): {str(e)}")
            context = ""

        # Build messages for the LLM
        messages = []
        
        # System message - adjust based on whether we have document context
        if context:
            system_message = "You are a helpful assistant. Use the provided document context to answer user questions when relevant, but you can also answer general questions using your knowledge. Be concise and accurate."
        else:
            system_message = "You are a helpful assistant. Answer user questions using your knowledge. Be concise and accurate."
            
        messages.append({"role": "system", "content": system_message})

        # Add chat history
        for row in rows:
            messages.append({"role": row["role"], "content": row["content"]})

        # Prepare user message
        user_content = request.question
        if context:
            user_content += f"\n\nRelevant document context:\n{context}"

        messages.append({
            "role": "user",
            "content": user_content
        })

        # Save user message to history
        await save_chat(request.session_id, "user", request.question, user_id)

        # Call NVIDIA API
        nvidia_api_key = os.getenv('NVIDIA_API_KEY')
        if not nvidia_api_key:
            raise HTTPException(status_code=500, detail="NVIDIA API key not configured")

        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Accept": "application/json"
        }

        payload = {
            "model": 'meta/llama-4-maverick-17b-128e-instruct',
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()

        assistant_response = response_data['choices'][0]['message']['content']
        
        # Save assistant response to history
        await save_chat(request.session_id, "assistant", assistant_response, user_id)

        logger.info(f"Chat response generated for session {request.session_id}")
        return {
            "answer": assistant_response,
            "has_document_context": bool(context),
            "context_chunks": len(retrieved_docs) if retrieved_docs else 0
        }
        
    except httpx.TimeoutException:
        logger.error(f"NVIDIA API timeout for session {request.session_id}")
        raise HTTPException(status_code=504, detail="AI service timeout - please try again")
    except httpx.HTTPStatusError as e:
        logger.error(f"NVIDIA API error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail="AI service error - please try again")
    except Exception as e:
        logger.error(f"Query failed for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error - please try again")

@app.get("/history")
async def get_chat_history(session_id: str = Query(...), current_user: dict = Depends(get_current_user)):
    try:
        logger.info(f"Fetching history for session {session_id}")
        db_pool = get_db_pool()
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content FROM chat_history WHERE session_id = $1 ORDER BY timestamp ASC",
                session_id
            )
        history = [{"role": row["role"], "content": row["content"]} for row in rows]
        return {"history": history}
    except Exception as e:
        logger.error(f"History fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_user_sessions(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        logger.info(f"Fetching sessions for user {user_id}")
        db_pool = get_db_pool()
        async with db_pool.acquire() as conn:
            query = """
            SELECT session_id, MAX(timestamp) AS timestamp
            FROM chat_history
            WHERE user_id = $1
            GROUP BY session_id
            ORDER BY timestamp DESC
            LIMIT 50
            """
            rows = await conn.fetch(query, user_id)

        sessions = [
            {"session_id": row["session_id"], "timestamp": row["timestamp"].isoformat()}
            for row in rows
        ]
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Session fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/create")
async def create_session(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"Creating new session for user {user_id}: {session_id}")
        db_pool = get_db_pool()
        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chat_history (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
                session_id, user_id, "system", "New session created"
            )
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Session creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/documents")
async def check_session_documents(session_id: str, current_user: dict = Depends(get_current_user)):
    """Check if a session has any uploaded documents"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_name=session_id
        )
        
        # Try to get a small sample to see if collection exists and has documents
        test_results = vectorstore.as_retriever(search_kwargs={"k": 1}).invoke("test")
        
        return {
            "has_documents": len(test_results) > 0,
            "document_count": len(test_results)
        }
    except Exception:
        return {
            "has_documents": False,
            "document_count": 0
        }

@app.post("/login")
async def login(request: UserInfo):
    logger.info(f"Login attempt: {request.email}")
    db_pool = get_db_pool()
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT password FROM users WHERE email = $1", request.email)
        if not user or not pwd_context.verify(request.password, user["password"]):
            logger.warning(f"Login failed for {request.email}")
            raise HTTPException(status_code=400, detail="Incorrect email or password")

    # Create proper JWT token
    access_token_expires = timedelta(hours=24)  # Token valid for 24 hours
    access_token = create_access_token(
        data={"sub": request.email}, 
        expires_delta=access_token_expires
    )
    
    logger.info(f"Login successful for {request.email}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(request: UserInfo):
    logger.info(f"Registering user: {request.email}")
    db_pool = get_db_pool()
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT 1 FROM users WHERE email = $1", request.email)
        if existing:
            logger.warning(f"User already exists: {request.email}")
            raise HTTPException(status_code=400, detail="User already exists")

        hashed_pw = pwd_context.hash(request.password)
        await conn.execute("INSERT INTO users (email, password) VALUES ($1, $2)", request.email, hashed_pw)

    return {"message": "User registered successfully"}