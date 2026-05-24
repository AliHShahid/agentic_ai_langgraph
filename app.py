from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from groq import Groq
import pandas as pd
import os
import copy
from dataclasses import dataclass
from typing import Any, Callable


# =============================
# FastAPI
# =============================

app = FastAPI()

templates = Jinja2Templates(
    directory="templates"
)

# =============================
# Groq
# =============================

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

# =============================
# Dataset
# =============================

df = pd.read_csv(
    "faq_dataset.csv"
)

# =============================
# Types
# =============================

State = dict[str, Any]
Update = dict[str, Any]
NodeFn = Callable[[State], Update]

END = "__end__"

# =============================
# Graph
# =============================


@dataclass
class Edge:

    src:str
    dst:str


class StateGraph:

    def __init__(self):

        self.nodes={}
        self.edges={}
        self.entry=None

    def add_node(self,name,fn):

        self.nodes[name]=fn

    def set_entry(self,name):

        self.entry=name

    def add_edge(self,src,dst):

        self.edges.setdefault(
            src,
            []
        ).append(
            Edge(
                src=src,
                dst=dst
            )
        )

    def next_node(
        self,
        current
    ):

        edges=self.edges.get(
            current,
            []
        )

        if edges:

            return edges[0].dst

        return None


class Runner:

    def __init__(
        self,
        graph
    ):

        self.graph=graph


    def run(
        self,
        initial_state
    ):

        state=copy.deepcopy(
            initial_state
        )

        current=self.graph.entry

        while current!=END:

            fn=self.graph.nodes[current]

            update=fn(
                state
            )

            state={
                **state,
                **update
            }

            current=self.graph.next_node(
                current
            )

        return state


# =============================
# Nodes
# =============================

def classify(state):

    text=state["input"].lower()

    if "refund" in text:

        route="refund"

    elif (
        "crash" in text
        or
        "bug" in text
        or
        "error" in text
    ):

        route="bug"

    else:

        route="general"

    return {

        "route":route

    }


def retrieve(state):

    query=state[
        "input"
    ].lower()

    query_words=set(
        query.split()
    )

    best_answer=None
    max_score=0

    for _,row in df.iterrows():

        question=str(
            row["question"]
        ).lower()

        answer=str(
            row["answer"]
        ).lower()

        combined=(
            question
            +" "+
            answer
        )

        words=set(
            combined.split()
        )

        score=len(
            query_words.intersection(
                words
            )
        )

        if score>max_score:

            max_score=score
            best_answer=row[
                "answer"
            ]

    if best_answer is None:

        best_answer="No information found"

    return {

        "retrieved_answer":
        best_answer

    }


def generate_response(state):

    prompt=f"""
    User Question:
    {state["input"]}

    Knowledge:
    {state["retrieved_answer"]}

    Generate a friendly response.
    """

    response=client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    text=response.choices[
        0
    ].message.content

    return {

        "final_response":
        text
    }


def send(state):

    return {

        "output":
        state[
            "final_response"
        ]
    }


# =============================
# Build graph
# =============================

graph=StateGraph()

graph.add_node(
    "classify",
    classify
)

graph.add_node(
    "retrieve",
    retrieve
)

graph.add_node(
    "generate",
    generate_response
)

graph.add_node(
    "send",
    send
)

graph.set_entry(
    "classify"
)

graph.add_edge(
    "classify",
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "generate"
)

graph.add_edge(
    "generate",
    "send"
)

graph.add_edge(
    "send",
    END
)

runner=Runner(
    graph
)

# =============================
# API
# =============================

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
            "request":request
        }
    )


@app.post("/chat")
async def chat(
    data:ChatRequest
):

    result=runner.run(

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