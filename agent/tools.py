import asyncio
import os

from fastmcp import Client
from langchain_core.tools import tool

MCP_EMOCIONES = os.getenv("MCP_EMOCIONES_URL", "http://localhost:8001")
MCP_METRICAS = os.getenv("MCP_METRICAS_URL", "http://localhost:8002")
MCP_PROPAGACION = os.getenv("MCP_PROPAGACION_URL", "http://localhost:8003")


def _call_mcp(base_url: str, tool_name: str, payload: dict):
    async def _run():
        async with Client(f"{base_url}/mcp") as client:
            result = await client.call_tool(tool_name, payload)
            response_payload = result.structured_content if result.structured_content is not None else result.data
            if isinstance(response_payload, dict) and set(response_payload.keys()) == {"result"}:
                return response_payload["result"]
            return response_payload

    return asyncio.run(_run())


@tool
def analizar_emociones(texto: str) -> dict:
    """Analiza las emociones presentes en un texto de red social."""
    return _call_mcp(MCP_EMOCIONES, "analizar_emociones", {"texto": texto})


@tool
def analizar_emociones_por_id(post_id: str) -> dict:
    """Analiza las emociones de un post especifico usando su ID."""
    return _call_mcp(MCP_EMOCIONES, "analizar_emociones_por_id", {"post_id": post_id})


@tool
def actores_influyentes(top_n: int = 10) -> list:
    """Identifica los usuarios mas influyentes por engagement total."""
    return _call_mcp(MCP_METRICAS, "actores_influyentes", {"top_n": top_n})


@tool
def post_mayor_impacto() -> dict:
    """Encuentra el post con mayor engagement o viralidad en el dataset."""
    return _call_mcp(MCP_METRICAS, "post_mayor_impacto", {})


@tool
def metricas_generales() -> dict:
    """Obtiene estadisticas globales de la conversacion."""
    return _call_mcp(MCP_METRICAS, "metricas_generales", {})


@tool
def analizar_propagacion(post_id: str) -> dict:
    """Analiza como se propago un mensaje especifico en la red."""
    return _call_mcp(MCP_PROPAGACION, "analizar_propagacion", {"post_id": post_id})


TOOLS = [
    analizar_emociones,
    analizar_emociones_por_id,
    actores_influyentes,
    post_mayor_impacto,
    metricas_generales,
    analizar_propagacion,
]
