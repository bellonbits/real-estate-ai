import os
import asyncio
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import hashlib

import psycopg2
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
import redis

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Configuration
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-70b-8192"

# Database configuration
DB_CONFIG = {
    "database": "bcp",
    "user": "bcp",
    "password": "developer@123",
    "host": "139.59.57.88",
    "port": 5400
}

# Redis configuration for caching
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0
}

# Initialize FastAPI app
app = FastAPI(
    title="Cosmas Ngeno - Real Estate Assistant API",
    description="AI-powered real estate assistant for Nafuu Kenya",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

class ChatRequest(BaseModel):
    message: str
    session_id: str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    customer_profile: Optional[Dict[str, Any]] = None

class PropertyFilter(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    location: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    limit: Optional[int] = 20

# Global variables
vector_store = None
qa_chain = None
embeddings_model = None
redis_client = None
executor = ThreadPoolExecutor(max_workers=4)

# Initialize services
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

async def get_database_pool():
    """Create async database connection pool for better performance."""
    return await asyncpg.create_pool(
        **DB_CONFIG,
        min_size=5,
        max_size=20,
        command_timeout=60
    )

def get_redis_client():
    """Initialize Redis client for caching."""
    try:
        return redis.Redis(**REDIS_CONFIG, decode_responses=True)
    except:
        return None

async def fetch_property_data_async(pool):
    """Fetch property data asynchronously with connection pooling."""
    async with pool.acquire() as connection:
        query = "SELECT * FROM pro.properties"
        rows = await connection.fetch(query)
        
        # Convert to DataFrame-like structure
        columns = list(rows[0].keys()) if rows else []
        data = [dict(row) for row in rows]
        
        return pd.DataFrame(data, columns=columns)

def create_optimized_embeddings():
    """Initialize embeddings model with optimized settings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
        encode_kwargs={'batch_size': 32}  # Only include batch_size, let show_progress default
    )

def process_documents_batch(documents: List[str], batch_size: int = 50):
    """Process documents in batches for faster embedding creation."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced chunk size for faster processing
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    all_chunks = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        chunks = text_splitter.create_documents(batch)
        all_chunks.extend(chunks)
    
    return all_chunks

async def create_optimized_vector_store(df: pd.DataFrame):
    """Create vector store with optimized processing and caching."""
    global redis_client
    
    # Create cache key based on data hash
    data_hash = hashlib.sha256(str(df.values.tobytes()).encode()).hexdigest()[:16]
    cache_key = f"vector_store_{data_hash}"
    
    # Try to load from cache
    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                vector_store = pickle.loads(cached_data.encode('latin1'))
                return vector_store
        except:
            pass
    
    # Process documents efficiently
    documents = []
    for _, row in df.iterrows():
        # Create more structured text representation
        text_parts = []
        for col, val in row.items():
            if val is not None and str(val).strip():
                text_parts.append(f"{col}: {val}")
        
        if text_parts:
            documents.append(" | ".join(text_parts))
    
    # Process in batches for better performance
    chunks = await asyncio.get_event_loop().run_in_executor(
        executor, process_documents_batch, documents, 50
    )
    
    # Create embeddings and vector store
    vector_store = await asyncio.get_event_loop().run_in_executor(
        executor, FAISS.from_documents, chunks, embeddings_model
    )
    
    # Cache the vector store
    if redis_client:
        try:
            serialized_data = pickle.dumps(vector_store)
            redis_client.setex(cache_key, 3600, serialized_data.decode('latin1'))  # Cache for 1 hour
        except:
            pass
    
    return vector_store

def create_optimized_qa_chain(vector_store):
    """Create QA chain with optimized retrieval settings."""
    llm = ChatGroq(
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.7,
        max_tokens=512,  # Limit tokens for faster response
        streaming=False
    )
    
    template = """
You are Cosmas Ngeno, a warm, professional, and knowledgeable real estate assistant for Nafuu Kenya Real Estate. You help customers find their ideal property by understanding their needs and offering tailored recommendations.

Your personality:

Friendly, approachable, and professional

Attentive to customer preferences and details

Always introduce yourself as Cosmas Ngeno when meeting a new customer

Enthusiastic about helping clients find their dream property

Instructions:

Always address the customer by name (if provided)

Ask clear, non-repetitive follow-up questions only when necessary (e.g., budget, preferred location, property type)

Use information provided to give specific, personalized property recommendations

Keep responses concise, professional, and on point—avoid repetition

Be helpful and enthusiastic without overwhelming the customer

Input:

Context from property database: {context}

Current question from customer: {question}

Your task:
Respond as Cosmas Ngeno. Keep it brief, tailored, and focused on guiding the customer toward a property they'll love.

    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Reduced for faster retrieval
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False  # Skip source docs for speed
    )
    
    return qa_chain

def extract_customer_preferences(query: str, existing_profile: Dict) -> Dict:
    """Extract customer preferences from query."""
    query_lower = query.lower()
    profile = existing_profile.copy()
    
    # Extract budget information
    budget_keywords = ['budget', 'afford', 'price', 'cost', 'million', 'thousand', 'ksh', 'kes']
    if any(word in query_lower for word in budget_keywords):
        profile['budget_mentions'] = profile.get('budget_mentions', [])
        profile['budget_mentions'].append(query)
    
    # Extract location preferences
    locations = ['westlands', 'kilimani', 'karen', 'lavington', 'kileleshwa', 
                'parklands', 'ruaka', 'runda', 'riverside', 'spring valley']
    mentioned_locations = [loc.title() for loc in locations if loc in query_lower]
    if mentioned_locations:
        profile['preferred_locations'] = list(set(
            profile.get('preferred_locations', []) + mentioned_locations
        ))
    
    # Extract property type
    property_types = ['apartment', 'house', 'villa', 'townhouse', 'bungalow', 'maisonette']
    for prop_type in property_types:
        if prop_type in query_lower:
            profile['property_type'] = prop_type.title()
    
    # Extract bedroom requirements
    for i in range(1, 6):
        if f'{i} bedroom' in query_lower:
            profile['bedrooms'] = i
            break
    
    return profile

async def get_conversational_response(query: str, chat_history: List[ChatMessage]) -> str:
    """Get response with conversation context."""
    # Format recent chat history for context
    formatted_history = ""
    if chat_history:
        recent_messages = chat_history[-4:]  # Last 4 messages for context
        for msg in recent_messages:
            role = "Customer" if msg.role == "user" else "Cosmas"
            formatted_history += f"{role}: {msg.content}\n"
    
    # Enhanced query with context
    enhanced_query = query
    if formatted_history:
        enhanced_query = f"Previous conversation:\n{formatted_history}\n\nCurrent question: {query}"
    
    # Get response from QA chain
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            executor, qa_chain.invoke, {"query": enhanced_query}
        )
        return response["result"]
    except Exception as e:
        return f"I apologize, but I encountered an issue processing your request. Please try rephrasing your question."

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global vector_store, qa_chain, embeddings_model, redis_client, db_pool
    
    print("🚀 Starting Cosmas Ngeno Real Estate Assistant API...")
    
    # Initialize Redis
    redis_client = get_redis_client()
    if redis_client:
        print("✅ Redis cache connected")
    else:
        print("⚠️ Redis cache not available - running without cache")
    
    # Initialize database pool
    db_pool = await get_database_pool()
    print("✅ Database connection pool created")
    
    # Initialize embeddings model
    embeddings_model = create_optimized_embeddings()
    print("✅ Embeddings model loaded")
    
    # Load and process data
    try:
        print("📊 Loading property data...")
        df = await fetch_property_data_async(db_pool)
        print(f"✅ Loaded {len(df)} property records")
        
        print("🔄 Creating optimized vector store...")
        vector_store = await create_optimized_vector_store(df)
        print("✅ Vector store created and cached")
        
        qa_chain = create_optimized_qa_chain(vector_store)
        print("✅ QA chain initialized")
        
        print("🎉 API ready to serve requests!")
        
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        raise

# Session storage (in production, use Redis or database)
sessions = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_cosmas(request: ChatRequest):
    """Main chat endpoint."""
    try:
        # Get or create session
        if request.session_id not in sessions:
            sessions[request.session_id] = {
                "customer_profile": {},
                "chat_history": []
            }
        
        session = sessions[request.session_id]
        
        # Update chat history
        user_message = ChatMessage(
            role="user", 
            content=request.message, 
            timestamp=datetime.now()
        )
        session["chat_history"].append(user_message)
        
        # Extract customer preferences
        session["customer_profile"] = extract_customer_preferences(
            request.message, 
            session["customer_profile"]
        )
        
        # Get AI response
        response_text = await get_conversational_response(
            request.message, 
            session["chat_history"]
        )
        
        # Add AI response to history
        ai_message = ChatMessage(
            role="assistant", 
            content=response_text, 
            timestamp=datetime.now()
        )
        session["chat_history"].append(ai_message)
        
        # Keep only last 20 messages to prevent memory issues
        if len(session["chat_history"]) > 20:
            session["chat_history"] = session["chat_history"][-20:]
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            timestamp=datetime.now(),
            customer_profile=session["customer_profile"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/properties")
async def get_properties(
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    location: Optional[str] = None,
    property_type: Optional[str] = None,
    bedrooms: Optional[int] = None,
    limit: int = 20
):
    """Get filtered property listings."""
    try:
        # Build dynamic query
        query = "SELECT * FROM pro.properties WHERE 1=1"
        params = []
        param_count = 1
        
        if min_price:
            query += f" AND price >= ${param_count}"
            params.append(min_price)
            param_count += 1
            
        if max_price:
            query += f" AND price <= ${param_count}"
            params.append(max_price)
            param_count += 1
            
        if location:
            query += f" AND LOWER(location) LIKE LOWER(${param_count})"
            params.append(f"%{location}%")
            param_count += 1
            
        if property_type:
            query += f" AND LOWER(property_type) LIKE LOWER(${param_count})"
            params.append(f"%{property_type}%")
            param_count += 1
            
        if bedrooms:
            query += f" AND bedrooms = ${param_count}"
            params.append(bedrooms)
            param_count += 1
        
        query += f" LIMIT ${param_count}"
        params.append(limit)
        
        async with db_pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            properties = [dict(row) for row in rows]
            
        return {
            "properties": properties,
            "count": len(properties),
            "filters_applied": {
                "min_price": min_price,
                "max_price": max_price,
                "location": location,
                "property_type": property_type,
                "bedrooms": bedrooms
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session data including chat history and customer profile."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "customer_profile": sessions[session_id]["customer_profile"],
        "message_count": len(sessions[session_id]["chat_history"])
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session data."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "database": "connected" if db_pool else "disconnected",
            "redis": "connected" if redis_client else "disconnected",
            "vector_store": "loaded" if vector_store else "not_loaded",
            "qa_chain": "ready" if qa_chain else "not_ready"
        }
    }

@app.post("/refresh-data")
async def refresh_property_data(background_tasks: BackgroundTasks):
    """Refresh property data and rebuild vector store."""
    async def refresh_task():
        global vector_store, qa_chain
        try:
            print("🔄 Refreshing property data...")
            df = await fetch_property_data_async(db_pool)
            vector_store = await create_optimized_vector_store(df)
            qa_chain = create_optimized_qa_chain(vector_store)
            print("✅ Data refreshed successfully")
        except Exception as e:
            print(f"❌ Error refreshing data: {str(e)}")
    
    background_tasks.add_task(refresh_task)
    return {"message": "Data refresh initiated in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8222,
        reload=True,
        workers=1  # Use 1 worker to maintain in-memory state
    )