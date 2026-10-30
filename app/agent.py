import operator
from typing import Annotated, TypedDict

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.config import settings
from app.generation import NOT_FOUND, build_context, extract_citations, generate_answer
from app.retrieval import hybrid_search

MAX_REWRITES = 1
TOP_K = 4

REWRITE_SYSTEM = """You rewrite a user question into a short search query for a document search engine.
Use wording likely to appear in formal company documents such as policies and reports.
Output only the query on a single line, with no explanation."""


class AgentState(TypedDict):
    question: str
    query: str
    chunks: list[dict]
    answer: str
    sources: list[dict]
    grounded: bool
    attempts: int
    trace: Annotated[list[str], operator.add]


def _chat(system: str, user: str, schema: dict | None = None) -> str:
    payload = {
        "model": settings.llm_model,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if schema:
        payload["format"] = schema
    with httpx.Client(base_url=settings.ollama_url, timeout=600) as client:
        resp = client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def retrieve(state: AgentState, config: RunnableConfig) -> dict:
    db = config["configurable"]["db"]
    chunks = hybrid_search(
        db, state["query"], TOP_K, min_similarity=settings.min_similarity
    )
    return {
        "chunks": chunks,
        "trace": [f"retrieve: {len(chunks)} chunks for '{state['query']}'"],
    }


def rewrite(state: AgentState) -> dict:
    raw = _chat(
        REWRITE_SYSTEM,
        f"Question: {state['question']}\nPrevious query that failed: {state['query']}",
    )
    lines = [ln.strip().strip('"') for ln in raw.splitlines() if ln.strip()]
    new_query = lines[0] if lines else state["question"]
    return {
        "query": new_query,
        "attempts": state["attempts"] + 1,
        "trace": [f"rewrite: '{new_query}'"],
    }


def generate(state: AgentState) -> dict:
    answer = generate_answer(state["question"], state["chunks"])
    sources = extract_citations(answer, state["chunks"])
    refused = NOT_FOUND in answer
    grounded = bool(sources) or refused
    outcome = "refused" if refused else f"{len(sources)} sources cited"
    return {
        "answer": answer,
        "sources": sources,
        "grounded": grounded,
        "trace": [f"generate: {outcome}, grounded={grounded}"],
    }


def give_up(state: AgentState) -> dict:
    return {
        "answer": NOT_FOUND,
        "sources": [],
        "grounded": True,
        "trace": ["give_up: nothing relevant found"],
    }


def route_after_retrieve(state: AgentState) -> str:
    if state["chunks"]:
        return "generate"
    return "rewrite" if state["attempts"] < MAX_REWRITES else "give_up"


def route_after_generate(state: AgentState) -> str:
    refused = NOT_FOUND in state["answer"]
    if refused and state["attempts"] < MAX_REWRITES:
        return "rewrite"
    return "end"

builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)
builder.add_node("give_up", give_up)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges(
    "retrieve", route_after_retrieve, ["generate", "rewrite", "give_up"]
)
builder.add_conditional_edges(
    "generate", route_after_generate, {"rewrite": "rewrite", "end": END}
)
builder.add_edge("rewrite", "retrieve")
builder.add_edge("give_up", END)

graph = builder.compile()


def run_agent(db: Session, question: str) -> dict:
    final = graph.invoke(
        {
            "question": question,
            "query": question,
            "chunks": [],
            "answer": "",
            "sources": [],
            "grounded": False,
            "attempts": 0,
            "trace": [],
        },
        config={"configurable": {"db": db}},
    )
    return {
        "question": question,
        "answer": final["answer"],
        "sources": final["sources"],
        "grounded": final["grounded"],
        "trace": final["trace"],
    }