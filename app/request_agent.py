import operator
import uuid
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings
from app.generation import NOT_FOUND, extract_citations, generate_answer, has_content
from app.mcp_client import call_tool


class RequestState(TypedDict):
    requester_id: int
    request_text: str
    passages: list[dict]
    draft: str
    policy_reference: str
    decision: dict
    status: str
    request_id: int | None
    trace: Annotated[list[str], operator.add]


def lookup_policy(state: RequestState) -> dict:
    result = call_tool("search_documents", {"query": state["request_text"], "top_k": 4})
    passages = result.get("results", [])
    return {
        "passages": passages,
        "trace": [f"lookup_policy (MCP search_documents): {len(passages)} passages"],
    }


def write_draft(state: RequestState) -> dict:
    passages = state["passages"]
    if passages:
        question = (
            f"An employee wants to do this: {state['request_text']}\n"
            "In two sentences, state whether the policy allows it and any conditions that apply."
        )
        answer = generate_answer(question, passages)
    else:
        answer = NOT_FOUND
    sources = extract_citations(answer, passages)
    reference = "; ".join(f"{s['filename']} p.{s['page_number']}" for s in sources) or "none"
    return {
        "draft": answer,
        "policy_reference": reference,
        "trace": [f"write_draft: {len(sources)} policy sources cited"],
    }


def approval_gate(state: RequestState) -> dict:
    decision = interrupt(
        {
            "requester_id": state["requester_id"],
            "request": state["request_text"],
            "draft": state["draft"],
            "policy_reference": state["policy_reference"],
        }
    )
    return {
        "decision": decision,
        "trace": [f"approval_gate: approved={bool(decision.get('approved'))}"],
    }


def submit_request(state: RequestState) -> dict:
    summary = f"Request: {state['request_text']}\nPolicy check: {state['draft']}"
    result = call_tool(
        "create_approval_request",
        {
            "requester_id": state["requester_id"],
            "summary": summary,
            "policy_reference": state["policy_reference"],
            "reviewer_note": state["decision"].get("note", ""),
        },
    )
    return {
        "status": "submitted",
        "request_id": result["request_id"],
        "trace": [f"submit_request (MCP create_approval_request): id {result['request_id']}"],
    }


def no_policy(state: RequestState) -> dict:
    return {
        "status": "no_policy",
        "trace": ["no_policy: no applicable policy found, nothing submitted"],
    }


def rejected(state: RequestState) -> dict:
    return {"status": "rejected", "trace": ["rejected: nothing was written"]}


def route_after_draft(state: RequestState) -> str:
    text = state["draft"]
    if NOT_FOUND in text or not has_content(text):
        return "no_policy"
    return "approval_gate"


def route_after_gate(state: RequestState) -> str:
    return "submit_request" if state["decision"].get("approved") else "rejected"


builder = StateGraph(RequestState)
builder.add_node("lookup_policy", lookup_policy)
builder.add_node("write_draft", write_draft)
builder.add_node("approval_gate", approval_gate)
builder.add_node("submit_request", submit_request)
builder.add_node("no_policy", no_policy)
builder.add_node("rejected", rejected)

builder.add_edge(START, "lookup_policy")
builder.add_edge("lookup_policy", "write_draft")
builder.add_conditional_edges(
    "write_draft", route_after_draft, ["approval_gate", "no_policy"]
)
builder.add_conditional_edges(
    "approval_gate", route_after_gate, ["submit_request", "rejected"]
)
builder.add_edge("submit_request", END)
builder.add_edge("no_policy", END)
builder.add_edge("rejected", END)


@lru_cache
def get_graph():
    pool = ConnectionPool(
        settings.psycopg_url,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return builder.compile(checkpointer=checkpointer)


def _present(thread_id: str, result: dict) -> dict:
    interrupts = result.get("__interrupt__")
    if interrupts:
        return {
            "thread_id": thread_id,
            "status": "waiting_for_approval",
            "review": interrupts[0].value,
            "trace": result["trace"],
        }
    return {
        "thread_id": thread_id,
        "status": result["status"],
        "request_id": result.get("request_id"),
        "draft": result.get("draft"),
        "trace": result["trace"],
    }


def start_request(requester_id: int, request_text: str) -> dict:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = get_graph().invoke(
        {
            "requester_id": requester_id,
            "request_text": request_text,
            "passages": [],
            "draft": "",
            "policy_reference": "",
            "decision": {},
            "status": "started",
            "request_id": None,
            "trace": [],
        },
        config,
    )
    return _present(thread_id, result)


def get_request(thread_id: str) -> dict:
    snapshot = get_graph().get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise LookupError("Unknown request")
    return {
        "thread_id": thread_id,
        "status": "waiting_for_approval" if snapshot.next else snapshot.values["status"],
        "draft": snapshot.values.get("draft"),
        "policy_reference": snapshot.values.get("policy_reference"),
        "trace": snapshot.values.get("trace", []),
    }


def decide(thread_id: str, approved: bool, note: str = "") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    if not get_graph().get_state(config).next:
        raise LookupError("No pending approval for this request")
    result = get_graph().invoke(
        Command(resume={"approved": approved, "note": note}), config
    )
    return _present(thread_id, result)