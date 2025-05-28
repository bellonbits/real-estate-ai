import os
import asyncio
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Any
import hashlib
import base64

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Configuration - Use environment variables for security
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key")
GROQ_MODEL = "llama3-70b-8192"

# Database configuration - Use environment variables
DB_CONFIG = {
    "database": os.getenv("DB_NAME", "bcp"),
    "user": os.getenv("DB_USER", "bcp"),
    "password": os.getenv("DB_PASSWORD", "developer@123"),
    "host": os.getenv("DB_HOST", "139.59.57.88"),
    "port": int(os.getenv("DB_PORT", "5400"))
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

# Cache for vector store (serverless functions are stateless)
vector_store_cache = {}

# Initialize services
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

async def get_database_connection():
    """Create single database connection for serverless function."""
    try:
        return await asyncpg.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

async def fetch_property_data():
    """Fetch property data with single connection."""
    connection = await get_database_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        query = "SELECT * FROM pro.properties LIMIT 1000"  # Limit for serverless
        rows = await connection.fetch(query)
        
        columns = list(rows[0].keys()) if rows else []
        data = [dict(row) for row in rows]
        
        return pd.DataFrame(data, columns=columns)
    finally:
        await connection.close()

def create_embeddings():
    """Initialize embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': 16}  # Smaller batch for serverless
    )

def process_documents(documents: List[str]):
    """Process documents for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for serverless
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = []
    for doc in documents[:100]:  # Limit documents for serverless
        doc_chunks = text_splitter.create_documents([doc])
        chunks.extend(doc_chunks)
    
    return chunks

async def get_or_create_vector_store():
    """Get or create vector store with caching."""
    cache_key = "main_vector_store"
    
    # Check if vector store exists in cache
    if cache_key in vector_store_cache:
        return vector_store_cache[cache_key]
    
    try:
        # Fetch property data
        df = await fetch_property_data()
        
        # Create embeddings model
        embeddings_model = create_embeddings()
        
        # Process documents
        documents = []
        for _, row in df.iterrows():
            text_parts = []
            for col, val in row.items():
                if val is not None and str(val).strip():
                    text_parts.append(f"{col}: {val}")
            
            if text_parts:
                documents.append(" | ".join(text_parts))
        
        # Create chunks
        chunks = process_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings_model)
        
        # Cache the vector store
        vector_store_cache[cache_key] = vector_store
        
        return vector_store
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise HTTPException(status_code=500, detail="Failed to create vector store")

def create_qa_chain(vector_store):
    """Create QA chain."""
    llm = ChatGroq(
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.7,
        max_tokens=400,
        streaming=False
    )
    
    template = """
You are Cosmas Ngeno, a warm, professional, and knowledgeable real estate assistant for Nafuu Kenya Real Estate. You help customers find their ideal property by understanding their needs and offering tailored recommendations.

Your personality:
- Friendly, approachable, and professional
- Attentive to customer preferences and details
- Always introduce yourself as Cosmas Ngeno when meeting a new customer
- Enthusiastic about helping clients find their dream property

Instructions:
- Always address the customer by name (if provided)
- Ask clear, non-repetitive follow-up questions only when necessary (e.g., budget, preferred location, property type)
- Use information provided to give specific, personalized property recommendations
- Keep responses concise, professional, and on point—avoid repetition
- Be helpful and enthusiastic without overwhelming the customer

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
            search_kwargs={"k": 2}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
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

# Simple in-memory session storage (for demo - use external storage in production)
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
        
        # Get vector store and create QA chain
        vector_store = await get_or_create_vector_store()
        qa_chain = create_qa_chain(vector_store)
        
        # Format recent chat history for context
        formatted_history = ""
        if session["chat_history"]:
            recent_messages = session["chat_history"][-3:]  # Last 3 messages
            for msg in recent_messages:
                role = "Customer" if msg.role == "user" else "Cosmas"
                formatted_history += f"{role}: {msg.content}\n"
        
        # Enhanced query with context
        enhanced_query = request.message
        if formatted_history:
            enhanced_query = f"Previous conversation:\n{formatted_history}\n\nCurrent question: {request.message}"
        
        # Get AI response
        try:
            response = qa_chain.invoke({"query": enhanced_query})
            response_text = response["result"]
        except Exception as e:
            response_text = "I apologize, but I encountered an issue processing your request. Please try rephrasing your question."
        
        # Add AI response to history
        ai_message = ChatMessage(
            role="assistant", 
            content=response_text, 
            timestamp=datetime.now()
        )
        session["chat_history"].append(ai_message)
        
        # Keep only last 10 messages to prevent memory issues
        if len(session["chat_history"]) > 10:
            session["chat_history"] = session["chat_history"][-10:]
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            timestamp=datetime.now(),
            customer_profile=session["customer_profile"]
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
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
    connection = await get_database_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
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
        params.append(min(limit, 50))  # Limit for serverless
        
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
    finally:
        await connection.close()

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
        "environment": "vercel"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cosmas Ngeno Real Estate Assistant API",
        "version": "1.0.0",
        "status": "running"
    }

# Vercel handler
handler = app
