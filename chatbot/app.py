from typing import TypedDict, List, Optional
import json
import os
import re

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langsmith import traceable
from pinecone import Pinecone
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# =====================================================
# Load ENV
# =====================================================

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "devryze_agent"


# =====================================================
# FastAPI
# =====================================================

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# =====================================================
# LLM / Embeddings / Pinecone
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    streaming=True,
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


# =====================================================
# Lead Submission
# =====================================================

DJANGO_API_URL = os.getenv(
    "DJANGO_API_URL",
    "https://www.devryze.tech/api/submit-lead/",
)

lead_cache: dict[str, dict] = {}


@traceable(name="submit_lead_form")
def submit_lead(form_data: dict) -> dict:
    try:
        response = httpx.post(DJANGO_API_URL, json=form_data, timeout=10.0)

        if response.status_code in (200, 201):
            return {
                "status": "success",
                "message": "Lead submitted successfully",
                "data": response.json(),
            }

        return {
            "status": "error",
            "message": f"Django API error: {response.status_code}",
            "data": response.text,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to submit form: {exc}",
        }


SERVICE_ALIASES = {
    "chatbot": "chatbots",
    "chatbots": "chatbots",
    "automation": "automation",
    "mobile": "mobile",
    "consult": "consult",
    "consultation": "consult",
    "web": "web",
    "website": "web",
    "analytics": "analytics",
    "mldl": "mldl",
    "machine learning": "mldl",
    "custom": "custom",
}


def normalize_service(value: str) -> str:
    normalized = value.strip().lower()
    for key, mapped in SERVICE_ALIASES.items():
        if key in normalized:
            return mapped
    return normalized if normalized in SERVICE_ALIASES.values() else "custom"


def extract_form_data(text: str) -> Optional[dict]:
    cleaned_text = text.strip()

    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            form_data = json.loads(json_match.group(1))
            if isinstance(form_data, dict):
                required = ["name", "email", "company", "package", "phone", "message"]
                if all(key in form_data and form_data[key] for key in required):
                    form_data["package"] = normalize_service(str(form_data["package"]))
                    return form_data
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    parts = [part.strip() for part in cleaned_text.split(",") if part.strip()]
    if len(parts) >= 6 and "@" in parts[1]:
        return {
            "name": parts[0],
            "email": parts[1],
            "package": normalize_service(parts[2]),
            "phone": parts[3],
            "company": parts[4],
            "message": ", ".join(parts[5:]),
        }

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cleaned_text)
    phone_match = re.search(r"(?:\+?\d[\d\s\-()]{6,}\d)", cleaned_text)

    if not email_match:
        return None

    package = "custom"
    lower_text = cleaned_text.lower()
    for key, mapped in SERVICE_ALIASES.items():
        if key in lower_text:
            package = mapped
            break

    name_match = re.search(r"(?:my name is|name is|i am|i'm)\s+([A-Za-z][A-Za-z\s.'-]{1,60})", cleaned_text, re.IGNORECASE)
    company_match = re.search(r"(?:company is|company name is|from|for)\s+([A-Za-z0-9&.,'\-\s]{2,80})", cleaned_text, re.IGNORECASE)
    message_match = re.search(r"(?:message|project|need|want|interested in)\s*[:\-]?\s*(.+)", cleaned_text, re.IGNORECASE)

    name = name_match.group(1).strip().rstrip(".,") if name_match else "Unknown"
    company = company_match.group(1).strip().rstrip(".,") if company_match else "Unknown"
    message = message_match.group(1).strip() if message_match else cleaned_text

    if len(message) > 240:
        message = message[:240].rstrip()

    return {
        "name": name,
        "email": email_match.group(0),
        "company": company,
        "package": package,
        "phone": phone_match.group(0) if phone_match else "",
        "message": message,
    }


def lead_is_complete(form_data: dict) -> bool:
    required = ["name", "email", "company", "package", "phone", "message"]
    return all(form_data.get(key) for key in required)


def merge_lead_draft(existing: Optional[dict], new_data: Optional[dict]) -> Optional[dict]:
    if not new_data:
        return existing

    merged = dict(existing or {})
    for key, value in new_data.items():
        if value:
            merged[key] = value
    return merged


def is_submission_confirmation(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False

    confirmation_phrases = (
        "yes",
        "yes do it",
        "do it",
        "submit",
        "save it",
        "send it",
        "proceed",
        "go ahead",
        "confirm",
    )

    return any(phrase == normalized or phrase in normalized for phrase in confirmation_phrases)


def update_lead_cache(session_id: str, message: str) -> tuple[Optional[dict], Optional[dict]]:
    extracted = extract_form_data(message)
    if extracted:
        lead_cache[session_id] = merge_lead_draft(lead_cache.get(session_id), extracted) or {}
    return lead_cache.get(session_id), extracted


def maybe_submit_lead(session_id: str, message: str):
    current_draft, extracted = update_lead_cache(session_id, message)
    if not current_draft:
        return None

    if lead_is_complete(current_draft) and (is_submission_confirmation(message) or extracted):
        result = submit_lead(current_draft)
        if result["status"] == "success":
            lead_cache.pop(session_id, None)
            return JSONResponse(
                {
                    "response": "Your inquiry has been submitted successfully. Our team will contact you soon.",
                    "lead_submitted": True,
                    "lead_data": current_draft,
                }
            )

        return JSONResponse(
            {
                "response": f"I found your inquiry details, but submission failed: {result['message']}",
                "lead_submitted": False,
                "lead_data": current_draft,
            },
            status_code=502,
        )

    if lead_is_complete(current_draft):
        return JSONResponse(
            {
                "response": "I have the lead details. Reply with 'yes do it' to submit them to the form.",
                "lead_ready": True,
                "lead_data": current_draft,
            }
        )

    missing_fields = [field for field in ["name", "email", "company", "package", "phone", "message"] if not current_draft.get(field)]
    return JSONResponse(
        {
            "response": f"I still need: {', '.join(missing_fields)}.",
            "lead_ready": False,
            "lead_data": current_draft,
        }
    )


# =====================================================
# LangGraph State
# =====================================================

class ChatState(TypedDict):
    input: str
    retrieved_answer: str
    output: str
    messages: List
    tool_calls: Optional[dict]
    tool_result: Optional[str]
    should_use_tool: bool


# =====================================================
# Retrieval Node
# =====================================================

@traceable(name="pinecone_retrieval")
def retrieve(state):
    query = state["input"]
    embedding = embedding_model.encode(query).tolist()

    result = index.query(
        vector=embedding,
        top_k=1,
        include_metadata=True,
    )

    matches = result.get("matches", [])
    if matches:
        match = matches[0]
        question = match["metadata"].get("question", "")
        answer = match["metadata"].get("answer", "")
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

    return {"retrieved_answer": context}


# =====================================================
# Generation Node
# =====================================================

@traceable(name="llm_generation")
def build_generation_messages(state):
    history = state.get("messages", [])
    history.append(
        (
            "human",
            f"""
            User Question:
            {state["input"]}

            DEVRYZE Knowledge:
            {state["retrieved_answer"]}
            """,
        )
    )

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
            - If the user is trying to fill the contact form, collect missing fields naturally
            - Do not invent contact details
            """,
        )
    ] + history

    return messages, history


@traceable(name="llm_generation")
def generate(state):
    messages, history = build_generation_messages(state)
    response = llm.invoke(messages)
    full_response = response.content

    history.append(("assistant", full_response))
    form_data = extract_form_data(full_response)
    return {
        "messages": history,
        "output": full_response,
        "tool_calls": form_data,
        "should_use_tool": bool(form_data),
    }


# =====================================================
# Tools Node
# =====================================================

@traceable(name="tools_node")
def tools_node(state):
    tool_result = None
    output = state.get("output", "")

    if state.get("tool_calls"):
        result = submit_lead(state["tool_calls"])
        if result["status"] == "success":
            tool_result = "Your inquiry has been submitted successfully. Our team will contact you soon."
        else:
            tool_result = f"Error submitting form: {result['message']}"

        output = output + "\n\n" + tool_result

    return {"tool_result": tool_result, "output": output}


def should_use_tools(state) -> str:
    return "tools" if state.get("should_use_tool", False) else "end"


# =====================================================
# Graph
# =====================================================

builder = StateGraph(ChatState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("tools", tools_node)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_conditional_edges(
    "generate",
    should_use_tools,
    {"tools": "tools", "end": END},
)
builder.add_edge("tools", END)

graph = builder.compile(checkpointer=MemorySaver())


# =====================================================
# Request Model
# =====================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"


# =====================================================
# Routes
# =====================================================

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(data: ChatRequest):
    lead_response = maybe_submit_lead(data.session_id, data.message)
    if lead_response is not None:
        return lead_response

    result = graph.invoke(
        {"input": data.message},
        config={
            "configurable": {"thread_id": data.session_id},
            "metadata": {"user_id": data.session_id, "app": "devryze-chatbot"},
        },
    )

    return JSONResponse({"response": result["output"]})


@app.post("/chat-stream")
async def chat_stream(data: ChatRequest):
    lead_response = maybe_submit_lead(data.session_id, data.message)
    if lead_response is not None:
        async def submit_event_generator():
            payload = json.loads(lead_response.body.decode("utf-8"))
            yield payload.get("response", "")

        return StreamingResponse(submit_event_generator(), media_type="text/plain; charset=utf-8")

    async def event_generator():
        retrieved = retrieve({"input": data.message})
        messages, _ = build_generation_messages(
            {
                "input": data.message,
                "retrieved_answer": retrieved["retrieved_answer"],
                "messages": [],
            }
        )

        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", "")
            if token:
                yield token

    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")
