from typing import TypedDict, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from langsmith import Client
from langsmith import traceable
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq

import asyncio
from fastapi.responses import StreamingResponse

import os

# =====================================================
# Load ENV
# =====================================================

load_dotenv()

import os

# LangSmith Tracing Configuration

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "devryze_agent"

# =====================================================
# FastAPI
# =====================================================

app = FastAPI()

templates = Jinja2Templates(
    directory="templates"
)


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0.4,
    streaming=True
)


# =====================================================
# Embedding Model
# =====================================================

embedding_model = SentenceTransformer(

    "all-MiniLM-L6-v2"

)


# =====================================================
# Pinecone
# =====================================================

pc = Pinecone(

    api_key=os.getenv(
        "PINECONE_API_KEY"
    )

)

index = pc.Index(

    os.getenv(
        "PINECONE_INDEX"
    )

)


# =====================================================
# LangGraph State
# =====================================================

class ChatState(TypedDict):

    input: str

    retrieved_answer: str

    output: str

    messages: List


# =====================================================
# Retrieve Node
# =====================================================
@traceable(name="pinecone_retrieval")
def retrieve(state):

    query = state["input"]

    embedding = embedding_model.encode(

        query

    ).tolist()


    result = index.query(

        vector=embedding,

        top_k=1,

        include_metadata=True

    )


    matches = result["matches"]


    if matches:

        match = matches[0]

        question = match[
            "metadata"
        ][
            "question"
        ]

        answer = match[
            "metadata"
        ][
            "answer"
        ]


        context = f"""

        Topic:
        {question}

        Information:
        {answer}

        """

    else:

        context = """

        No relevant DEVRYZE information found.

        """


    return {

        "retrieved_answer":
        context

    }


# =====================================================
# Generate Node
# =====================================================
# @traceable(name="llm_generation")
# def generate(state):

#     history = state.get(

#         "messages",

#         []

#     )


#     history.append(

#         (

#             "human",

#             f"""

#             User Question:

#             {state["input"]}

#             DEVRYZE Knowledge:

#             {state["retrieved_answer"]}

#             """

#         )

#     )


#     response = llm.invoke(

#         [

#             (

#                 "system",

#                 """
#                 You are DEVRYZE Assistant.

#                 About DEVRYZE:

#                 DEVRYZE bridges the gap between
#                 complex AI research and
#                 production-ready software.

#                 Rules:

#                 - Answer professionally
#                 - Use retrieved context
#                 - Never invent facts
#                 - Keep responses concise
#                 - Remember previous conversation
#                 - If context is unavailable,
#                   say you could not find it
#                 """

#             )

#         ]

#         +

#         history

#     )


#     history.append(

#         (

#             "assistant",

#             response.content

#         )

#     )


#     return {

#         "output":
#         response.content,

#         "messages":
#         history

#     }

from langsmith import traceable

@traceable(name="llm_generation")
def build_generation_messages(state):

    history = state.get("messages", [])

    history.append((
        "human",
        f"""
        User Question:
        {state["input"]}

        DEVRYZE Knowledge:
        {state["retrieved_answer"]}
        """
    ))

    messages = [
        (
            "system",
            """
            You are DEVRYZE Assistant.

            About DEVRYZE:
            DEVRYZE bridges the gap between complex AI research and production-ready software.

            Rules:
            - Answer professionally
            - Use retrieved context
            - Never invent facts
            - Keep responses concise
            - Remember previous conversation
            - If context is unavailable, say you could not find it
            """
        )
    ] + history

    return messages, history


@traceable(name="llm_generation")
def generate(state):

    messages, history = build_generation_messages(state)

    response = llm.invoke(messages)
    full_response = response.content

    history.append(("assistant", full_response))

    return {
        "messages": history,
        "output": full_response
    }
# =====================================================
# Build LangGraph
# =====================================================

builder = StateGraph(
    ChatState
)


builder.add_node(
    "retrieve",
    retrieve
)

builder.add_node(
    "generate",
    generate
)


builder.set_entry_point(
    "retrieve"
)


builder.add_edge(
    "retrieve",
    "generate"
)

builder.add_edge(
    "generate",
    END
)


# =====================================================
# Memory
# =====================================================

memory = MemorySaver()


graph = builder.compile(

    checkpointer=memory

)


# =====================================================
# Request Model
# =====================================================

class ChatRequest(
    BaseModel
):

    message: str

    session_id: str = "default_user"


# =====================================================
# Routes
# =====================================================

@app.get("/")

async def home(
    request: Request
):

    return templates.TemplateResponse(

        "index.html",

        {

            "request":
            request

        }

    )


@app.post("/chat")

async def chat(
    data: ChatRequest
):

    result = graph.invoke(

        {

            "input":
            data.message

        },

        config={

            "configurable": {

                "thread_id":
                data.session_id
            },
            "metadata": {
                "user_id": data.session_id,
                "app": "devryze-chatbot"
            }

        }

    )


    return JSONResponse(

        {

            "response":
            result["output"]

        }

    )

@app.post("/chat-stream")
async def chat_stream(data: ChatRequest):

    async def event_generator():

        retrieved = retrieve({"input": data.message})
        messages, _ = build_generation_messages({
            "input": data.message,
            "retrieved_answer": retrieved["retrieved_answer"],
            "messages": []
        })

        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", "")

            if token:
                yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/plain; charset=utf-8"
    )