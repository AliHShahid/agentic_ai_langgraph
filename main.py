from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd


# ======================
# Types
# ======================

State = dict[str, Any]
Update = dict[str, Any]
NodeFn = Callable[[State], Update]
Router = Callable[[State], str]

START = "__start__"
END = "__end__"


# ======================
# Load Dataset
# ======================

df = pd.read_csv("faq_dataset.csv")


# ======================
# Graph Classes
# ======================

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
        self.edges.setdefault(src, []).append(
            Edge(src=src, dst=dst)
        )

    def add_conditional_edges(
        self,
        src,
        router,
        targets
    ):
        for value, dst in targets.items():
            self.edges.setdefault(src, []).append(
                Edge(
                    src=src,
                    dst=dst,
                    router=_make_router(router, value)
                )
            )

    def _next(self, current, state):

        for edge in self.edges.get(current, []):

            if edge.router is None:
                return edge.dst

            if edge.router(state):
                return edge.dst

        return None


def _make_router(router, expected):

    def fn(state):
        return router(state) == expected

    return fn


# ======================
# Checkpoint System
# ======================

class InMemoryCheckpointer:

    def __init__(self):
        self.store = {}

    def save(self, session, node, state):

        self.store.setdefault(
            session,
            []
        ).append(
            (
                node,
                copy.deepcopy(state)
            )
        )

    def load_latest(self, session):

        history = self.store.get(session, [])

        if not history:
            return None

        return history[-1]


class PausedAtNode(Exception):

    def __init__(self, node, state):

        super().__init__(node)

        self.node = node
        self.state = state


# ======================
# Runner
# ======================

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

        state = copy.deepcopy(initial_state)

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

            if state.get("_pause_reason"):

                raise PausedAtNode(
                    current,
                    state
                )

            current = self.graph._next(
                current,
                state
            )

        return state


# ======================
# Nodes
# ======================

def classify(state):

    text = state["input"].lower()

    if "refund" in text:
        route = "refund"

    elif "bug" in text or "crash" in text:
        route = "bug"

    else:
        route = "sales"

    return {
        "route": route
    }
def retrieve(state):

    query = state["input"].lower()

    best_answer = None
    max_score = 0

    query_words = set(query.split())

    for _, row in df.iterrows():

        question = str(row["question"]).lower()
        answer = str(row["answer"]).lower()

        # combine both fields
        combined_text = question + " " + answer

        text_words = set(combined_text.split())

        score = len(
            query_words.intersection(
                text_words
            )
        )

        if score > max_score:

            max_score = score
            best_answer = row["answer"]

    if best_answer is None or max_score == 0:

        best_answer = "No matching answer found."

    return {
        "retrieved_answer": best_answer,
        "match_score": max_score
    }
    
def human_gate(state):

    if not state.get(
        "human_approval"
    ):

        return {
            "_pause_reason":
            "waiting human approval"
        }

    return {}


def send(state):

    return {
        "output":
        f"""
Route: {state['route']}

Answer:
{state['retrieved_answer']}
"""
    }


# ======================
# Build Graph
# ======================

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
        "human_gate",
        human_gate
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
        "human_gate"
    )

    graph.add_edge(
        "human_gate",
        "send"
    )

    graph.add_edge(
        "send",
        END
    )

    return graph


# ======================
# Main
# ======================

def main():

    graph = build_graph()

    ckpt = InMemoryCheckpointer()

    runner = Runner(
        graph,
        ckpt
    )

    initial = {

        "input":
        "how to contact you?",

        "human_approval":
        True
    }

    result = runner.run(
        "session001",
        initial
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )


if __name__ == "__main__":
    main()