from typing import Annotated, TypedDict
import operator
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools import TOOLS

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def _build_model_candidates() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    env_fallbacks = os.getenv("GEMINI_MODEL_FALLBACKS", "")
    default_fallbacks = ["gemini-1.5-flash", "gemini-1.5-pro"]

    candidates: list[str] = [primary]
    for model in [m.strip() for m in env_fallbacks.split(",") if m.strip()] + default_fallbacks:
        if model not in candidates:
            candidates.append(model)
    return candidates


MODEL_CANDIDATES = _build_model_candidates()
_LLM_CACHE: dict[str, ChatGoogleGenerativeAI] = {}


def _get_llm_with_tools(model_name: str):
    if model_name not in _LLM_CACHE:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        _LLM_CACHE[model_name] = llm.bind_tools(TOOLS)
    return _LLM_CACHE[model_name]


def _invoke_with_model_fallback(messages):
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            return _get_llm_with_tools(model_name).invoke(messages)
        except Exception as exc:
            message = str(exc)
            if "NOT_FOUND" in message or "is not found" in message or "404" in message:
                last_error = exc
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("No se pudo inicializar ningun modelo Gemini")

SYSTEM = """Eres un asistente analitico de conversaciones digitales.
Tienes acceso a microservicios especializados para analizar publicaciones
y comentarios de redes sociales.

Reglas:
- Ante cualquier pregunta analitica, usa SIEMPRE la herramienta adecuada.
- Nunca inventes datos. Toda respuesta debe basarse en el resultado de las herramientas.
- Tras recibir el resultado de una herramienta, genera una respuesta clara,
  conversacional y facil de entender para el usuario.
- Si el usuario hace una pregunta de seguimiento sobre un resultado anterior,
  puedes responder usando la informacion ya obtenida sin volver a llamar la herramienta,
  a menos que necesites datos nuevos."""


def agent_node(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM)] + state["messages"]
    response = _invoke_with_model_fallback(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(TOOLS)

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()