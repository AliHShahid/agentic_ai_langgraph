from typing import TypedDict
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph,END
from langchain_groq import ChatGroq

import os

load_dotenv()


# ===================
# FastAPI
# ===================

app=FastAPI()

templates=Jinja2Templates(
    directory="templates"
)


# ===================
# LLM
# ===================

llm=ChatGroq(

    model="llama-3.3-70b-versatile",
    temperature=0.4
)


# ===================
# Embedding Model
# ===================

embedding_model=SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ===================
# Pinecone
# ===================

pc=Pinecone(

    api_key=
    os.getenv(
        "PINECONE_API_KEY"
    )
)

index=pc.Index(

    os.getenv(
        "PINECONE_INDEX"
    )
)


# ===================
# State
# ===================

class ChatState(TypedDict):

    input:str
    retrieved_answer:str
    output:str


# ===================
# Retrieve node
# ===================

def retrieve(state):

    query=state["input"]

    embedding=embedding_model.encode(
        query
    ).tolist()


    result=index.query(

        vector=embedding,

        top_k=1,

        include_metadata=True
    )


    matches=result["matches"]


    if matches:

        match=matches[0]

        question=match[
            "metadata"
        ][
            "question"
        ]

        answer=match[
            "metadata"
        ][
            "answer"
        ]


        context=f"""

        Topic:
        {question}

        Information:
        {answer}

        """

    else:

        context=""


    return {

        "retrieved_answer":
        context
    }


# ===================
# Generate node
# ===================

def generate(state):

    response=llm.invoke(

        [

            (

                "system",

                """
                You are DEVRYZE Assistant.

                Rules:

                - Answer as DEVRYZE assistant
                - Use company context
                - Do not invent facts
                - Keep answers concise
                """

            ),

            (

                "human",

                f"""

                User Question:

                {state["input"]}

                Context:

                {state["retrieved_answer"]}

                """

            )

        ]

    )


    return {

        "output":
        response.content
    }


# ===================
# Build Graph
# ===================

builder=StateGraph(
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

graph=builder.compile()


# ===================
# API
# ===================

class ChatRequest(
    BaseModel
):

    message:str


@app.get("/")

async def home(
    request:Request
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
    data:ChatRequest
):

    result=graph.invoke(

        {

            "input":
            data.message

        }

    )

    return JSONResponse(

        {

            "response":
            result["output"]

        }

    )