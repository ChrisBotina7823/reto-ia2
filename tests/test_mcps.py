import asyncio
import sys
from pathlib import Path

from fastmcp import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_EMOCIONES = "http://localhost:8001"
BASE_METRICAS = "http://localhost:8002"
BASE_PROPAGACION = "http://localhost:8003"


def call_tool(base_url: str, tool_name: str, payload: dict):
    async def _run():
        async with Client(f"{base_url}/mcp") as client:
            result = await client.call_tool(tool_name, payload)
            response_payload = result.structured_content if result.structured_content is not None else result.data
            if isinstance(response_payload, dict) and set(response_payload.keys()) == {"result"}:
                return response_payload["result"]
            return response_payload

    return asyncio.run(_run())


def test_emociones_texto_positivo():
    data = call_tool(BASE_EMOCIONES, "analizar_emociones", {"texto": "Estoy muy feliz con este resultado!"})
    assert "emocion_dominante" in data
    assert "emociones" in data
    assert "confianza" in data


def test_emociones_texto_negativo():
    data = call_tool(BASE_EMOCIONES, "analizar_emociones", {"texto": "Estoy furioso, esto es una injusticia."})
    assert "emocion_dominante" in data


def test_emociones_por_id_inexistente():
    data = call_tool(BASE_EMOCIONES, "analizar_emociones_por_id", {"post_id": "ID_QUE_NO_EXISTE_999999"})
    assert "error" in data


def test_metricas_generales():
    data = call_tool(BASE_METRICAS, "metricas_generales", {})
    assert "total_posts" in data
    assert data["total_posts"] > 0


def test_actores_influyentes():
    data = call_tool(BASE_METRICAS, "actores_influyentes", {"top_n": 5})
    assert isinstance(data, list)
    assert len(data) <= 5


def test_post_mayor_impacto():
    data = call_tool(BASE_METRICAS, "post_mayor_impacto", {})
    assert "post_id" in data
    assert "engagement_total" in data


def test_propagacion_id_inexistente():
    data = call_tool(BASE_PROPAGACION, "analizar_propagacion", {"post_id": "ID_QUE_NO_EXISTE_999999"})
    assert "error" in data


def test_propagacion_id_real():
    from shared.data_loader import load_dataset

    df = load_dataset()
    post_id = str(df["post_id"].iloc[0])

    data = call_tool(BASE_PROPAGACION, "analizar_propagacion", {"post_id": post_id})
    assert "alcance" in data
    assert "usuarios_alcanzados" in data
