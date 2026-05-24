from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import streamlit as st
from groq import Groq


# ======================================================
# LLM
# ======================================================

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


# ======================================================
# Dataset
# ======================================================

df = pd.read_csv("faq_dataset.csv")


# ======================================================
# Types
# ======================================================

State = dict[str, Any]
Update = dict[str, Any]
NodeFn = Callable[[State], Update]
Router = Callable[[State], str]

END = "__end__"


# ======================================================
# Graph Core
# ======================================================

@dataclass
class Edge:

    src: str
    dst: str
    router: Router | None = None


class StateGraph:

    def __init__(self):

        self.nodes = {}
        self.edges = {}
        self.entry = None

    def add_node(self, name, fn):

        self.nodes[name] = fn

    def set_entry(self, name):

        self.entry = name

    def add_edge(self, src, dst):

        self.edges.setdefault(
            src,
            []
        ).append(
            Edge(
                src=src,
                dst=dst
            )
        )

    def _next(self, current, state):

        for edge in self.edges.get(current, []):

            if edge.router is None:
                return edge.dst

            if edge.router(state):
                return edge.dst

        return None


# ======================================================
# Checkpointer
# ======================================================

class InMemoryCheckpointer:

    def __init__(self):

        self.store = {}

    def save(
        self,
        session,
        node,
        state
    ):

        self.store.setdefault(
            session,
            []
        ).append(
            (
                node,
                copy.deepcopy(state)
            )
        )


# ======================================================
# Runner
# ======================================================

class Runner:

    def __init__(
        self,
        graph,
        checkpointer
    ):

        self.graph = graph
        self.checkpointer = checkpointer

    def run(
        self,
        session_id,
        initial_state
    ):

        state = copy.deepcopy(
            initial_state
        )

        current = self.graph.entry

        while current != END:

            fn = self.graph.nodes[current]

            update = fn(state)

            state = {
                **state,
                **update
            }

            self.checkpointer.save(
                session_id,
                current,
                state
            )

            current = self.graph._next(
                current,
                state
            )

        return state


# ======================================================
# Nodes
# ======================================================

def classify(state):

    text = state["input"].lower()

    if "refund" in text:

        route = "refund"

    elif (
        "crash" in text
        or
        "bug" in text
        or
        "error" in text
    ):

        route = "bug"

    else:

        route = "general"

    return {
        "route": route
    }


def retrieve(state):

    query = state["input"].lower()

    query_words = set(
        query.split()
    )

    best_answer = None
    max_score = 0

    for _, row in df.iterrows():

        question = str(
            row["question"]
        ).lower()

        answer = str(
            row["answer"]
        ).lower()

        combined = (
            question
            +
            " "
            +
            answer
        )

        words = set(
            combined.split()
        )

        score = len(
            query_words.intersection(
                words
            )
        )

        if score > max_score:

            max_score = score
            best_answer = row["answer"]

    if best_answer is None:

        best_answer = (
            "No matching information found."
        )

    return {

        "retrieved_answer":
        best_answer
    }


def generate_response(state):

    question = state["input"]

    context = state[
        "retrieved_answer"
    ]

    prompt = f"""
    User Question:
    {question}

    Knowledge Base:
    {context}

    Generate a helpful and professional response.
    """

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],

        temperature=0.5
    )

    text = (
        response
        .choices[0]
        .message
        .content
    )

    return {
        "final_response": text
    }


def send(state):

    return {

        "output":
        state["final_response"]

    }


# ======================================================
# Build Graph
# ======================================================

def build_graph():

    graph = StateGraph()

    graph.add_node(
        "classify",
        classify
    )

    graph.add_node(
        "retrieve",
        retrieve
    )

    graph.add_node(
        "generate_response",
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
        "generate_response"
    )

    graph.add_edge(
        "generate_response",
        "send"
    )

    graph.add_edge(
        "send",
        END
    )

    return graph


# ======================================================
# Initialize
# ======================================================

graph = build_graph()

checkpoint = InMemoryCheckpointer()

runner = Runner(
    graph,
    checkpoint
)


# ======================================================
# Streamlit UI
# ======================================================

st.title(
    "LangGraph + Groq Chatbot"
)

if (
    "messages"
    not in st.session_state
):

    st.session_state.messages = []


for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )


query = st.chat_input(
    "Ask a question..."
)


if query:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":query
        }
    )

    with st.chat_message(
        "user"
    ):

        st.write(query)

    result = runner.run(

        "session001",

        {
            "input":query
        }
    )

    answer = result["output"]

    with st.chat_message(
        "assistant"
    ):

        st.write(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )