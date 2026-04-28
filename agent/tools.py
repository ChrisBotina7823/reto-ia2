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

    try:
        return asyncio.run(_run())
    except Exception as exc:
        return {"error": f"Error al llamar {base_url}/{tool_name}: {str(exc)}"}


@tool
def analizar_emociones(texto: str) -> dict:
    """Analiza las emociones (alegria, tristeza, enojo, miedo, sorpresa, disgusto) de un texto de red social. Usar cuando el usuario proporcione texto directamente para analizar."""
    return _call_mcp(MCP_EMOCIONES, "analizar_emociones", {"texto": texto})


@tool
def analizar_emociones_por_id(post_id: str) -> dict:
    """Analiza las emociones de un post especifico del dataset usando su ID. Usar cuando el usuario mencione un ID de post y quiera saber el estado emocional de ese post."""
    return _call_mcp(MCP_EMOCIONES, "analizar_emociones_por_id", {"post_id": post_id})


@tool
def actores_influyentes(top_n: int = 10) -> list:
    """Identifica los usuarios con mayor engagement total. Usar cuando pregunten quienes son los mas influyentes, activos, o con mayor impacto en la conversacion."""
    return _call_mcp(MCP_METRICAS, "actores_influyentes", {"top_n": top_n})


@tool
def post_mayor_impacto() -> dict:
    """Retorna el post con mayor engagement o viralidad. Usar cuando pregunten por el post mas viral, con mas impacto, repercusion o el mas popular."""
    return _call_mcp(MCP_METRICAS, "post_mayor_impacto", {})


@tool
def metricas_generales() -> dict:
    """Devuelve estadisticas globales: total de posts, usuarios, engagement promedio y maximo. Usar para preguntas generales sobre el volumen o dimension de la conversacion."""
    return _call_mcp(MCP_METRICAS, "metricas_generales", {})


@tool
def analizar_propagacion(post_id: str) -> dict:
    """Analiza como se propago un mensaje especifico en la red: alcance, usuarios alcanzados y velocidad de la primera respuesta. Requiere un post_id exacto."""
    return _call_mcp(MCP_PROPAGACION, "analizar_propagacion", {"post_id": post_id})


TOOLS = [
    analizar_emociones,
    analizar_emociones_por_id,
    actores_influyentes,
    post_mayor_impacto,
    metricas_generales,
    analizar_propagacion,
]
