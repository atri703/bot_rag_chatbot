from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import Optional, List
from datetime import datetime

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import json
import redis

import os
from dotenv import load_dotenv
load_dotenv()

# Load OpenAI Key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

# =============================
# INIT
# =============================

app = FastAPI(title="BOT GPT Backend", version="1.0")

# SQLite for conversations + messages
# engine = create_engine("sqlite:///botgpt.db")
PASSWORD = os.getenv("POSTGRES_PASSWORD")
DATABASE_URL = f"postgresql://postgres:{PASSWORD}@localhost:5432/botgpt"

# Ensure database exists
def ensure_database_exists():
    dbname = "botgpt"
    user = "postgres"
    password = PASSWORD
    host = "localhost"
    port = "5432"

    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            dbname="postgres",
            user=user,
            password=password,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if DB exists
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}';")
        exists = cur.fetchone()

        if not exists:
            print("Database 'botgpt' does not exist. Creating it...")
            cur.execute(f"CREATE DATABASE {dbname};")
            print("Database created successfully!")

        cur.close()
        conn.close()

    except Exception as e:
        print("Failed to check or create database:", e)

ensure_database_exists()

engine = create_engine(DATABASE_URL, echo=True)

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Redis cache
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# ChromaDB (already populated)
VECTOR_DB = Chroma(
    persist_directory="chroma_db2/unified",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),

    ("system", """
You are a helpful assistant.

Here is the factual context from the knowledge base:
{context}

You must use BOTH:
1. chat_history (previous messages)
2. the above context

keep answer under 50 words.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

retriever = VECTOR_DB.as_retriever()

# LLM
LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_KEY
)

retrieve_only_question = RunnableLambda(lambda x: retriever.invoke(x["question"]))

rag_chain = (
    {
        "context": retrieve_only_question,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | prompt
    | LLM
)


# =============================
# DATABASE MODELS
# =============================

class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    role: str  # user | assistant
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================
# SCHEMAS
# =============================

class CreateConversation(BaseModel):
    title: str
    first_message: str

class AddMessage(BaseModel):
    message: str


# =============================
# HELPERS
# =============================

def load_chat_history(session: Session, cid: int):
    msgs = session.exec(
        select(Message).where(Message.conversation_id == cid).order_by(Message.created_at)
    ).all()

    history = []
    for m in msgs:
        if m.role == "user":
            history.append(HumanMessage(content=m.content))
        else:
            history.append(AIMessage(content=m.content))

    return history

def run_rag_with_history(cid: int, question: str):
    with Session(engine) as session:
        # 1. Load chat history from DB
        chat_history = load_chat_history(session, cid)

        # === DEBUGGING: PRINT HISTORY GOING INTO LLM ===
        print("\n================ CHAT HISTORY DEBUG ================")
        print(f"Conversation ID: {cid}")
        print(f"User question: {question}")
        print("History being sent to LLM (in order):")
        for msg in chat_history:
            role = "User" if isinstance(msg, HumanMessage) else "AI Message"
            print(f" - {role}: {msg.content}")
        print("====================================================\n")

        # 2. Build chain input
        chain_input = {
            "question": question,
            "chat_history": chat_history
        }

        # 3. Check Redis cache
        cache_key = f"cid:{cid}:q:{question}"
        cached = redis_client.get(cache_key)
        if cached:
            return cached

        # 4. Run chain
        response = rag_chain.invoke(chain_input)
        answer = response.content

        # 5. Save to Redis cache
        redis_client.set(cache_key, answer)

        return answer

# =============================
# API ENDPOINTS
# =============================


# 1) CREATE CONVERSATION
@app.post("/conversations")
def create_conversation(payload: CreateConversation):
    with Session(engine) as session:
        convo = Conversation(title=payload.title)
        session.add(convo)
        session.commit()
        session.refresh(convo)

        # Save user message
        m1 = Message(
            conversation_id=convo.id,
            role="user",
            content=payload.first_message
        )
        session.add(m1)
        session.commit()

        # RAG answer
        answer = run_rag_with_history(convo.id, payload.first_message)

        m2 = Message(
            conversation_id=convo.id,
            role="assistant",
            content=answer
        )
        session.add(m2)
        session.commit()

        return {"conversation_id": convo.id, "answer": answer}


# 2) LIST CONVERSATIONS
@app.get("/conversations")
def list_conversations():
    with Session(engine) as session:
        return session.exec(select(Conversation)).all()


# 3) GET FULL HISTORY
@app.get("/conversations/{cid}")
def get_conversation(cid: int):
    with Session(engine) as session:
        msgs = session.exec(
            select(Message).where(Message.conversation_id == cid)
        ).all()
        if not msgs:
            raise HTTPException(404, "Conversation not found")
        return msgs


# 4) ADD MESSAGE TO EXISTING CONVERSATION
@app.post("/conversations/{cid}/messages")
def add_message(cid: int, payload: AddMessage):
    with Session(engine) as session:
        convo = session.get(Conversation, cid)
        if not convo:
            raise HTTPException(404, "Conversation not found")

        # Save user message
        m1 = Message(
            conversation_id=cid,
            role="user",
            content=payload.message
        )
        session.add(m1)
        session.commit()

        # RAG + LLM
        answer = run_rag_with_history(cid, payload.message)

        # Save assistant message
        m2 = Message(
            conversation_id=cid,
            role="assistant",
            content=answer
        )
        session.add(m2)
        session.commit()

        return {"answer": answer}