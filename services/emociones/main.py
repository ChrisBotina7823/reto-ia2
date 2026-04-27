import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from fastmcp import FastMCP

from shared.data_loader import get_post
from shared.gemini_client import call_gemini

import json
import re

load_dotenv()

mcp = FastMCP("analisis-emociones")

SYSTEM_PROMPT = """Eres un clasificador de emociones para textos de redes sociales.
Analiza el texto y devuelve UNICAMENTE un JSON valido con este esquema exacto:
{
  "emociones": {
    "alegria": 0.0,
    "tristeza": 0.0,
    "enojo": 0.0,
    "miedo": 0.0,
    "sorpresa": 0.0,
    "disgusto": 0.0
  },
  "emocion_dominante": "alegria",
  "confianza": 0.85
}
Cada valor es un float entre 0.0 y 1.0.
Sin explicaciones. Sin markdown. Solo el JSON."""


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Respuesta no parseable: {raw[:200]}")


def _fallback_emotions(texto: str) -> dict:
    texto_normalizado = texto.lower()
    emocion_scores = {
        "alegria": 0.1,
        "tristeza": 0.1,
        "enojo": 0.1,
        "miedo": 0.1,
        "sorpresa": 0.1,
        "disgusto": 0.1,
    }

    positive_markers = ["feliz", "alegre", "genial", "excelente", "bueno", "me encanta", "encanta"]
    negative_markers = ["furioso", "injusticia", "odio", "triste", "terrible", "malo", "horrible"]
    surprise_markers = ["sorpresa", "wow", "increible", "inesperado"]
    fear_markers = ["miedo", "temor", "asust", "peligro"]
    disgust_markers = ["asco", "repugn", "asqueroso"]

    if any(marker in texto_normalizado for marker in positive_markers):
        emocion_scores["alegria"] = 0.85
    if any(marker in texto_normalizado for marker in negative_markers):
        emocion_scores["enojo"] = 0.8
        emocion_scores["tristeza"] = 0.65
    if any(marker in texto_normalizado for marker in surprise_markers):
        emocion_scores["sorpresa"] = 0.75
    if any(marker in texto_normalizado for marker in fear_markers):
        emocion_scores["miedo"] = 0.78
    if any(marker in texto_normalizado for marker in disgust_markers):
        emocion_scores["disgusto"] = 0.8

    emocion_dominante = max(emocion_scores, key=emocion_scores.get)
    confianza = round(max(emocion_scores.values()), 2)
    return {
        "emociones": emocion_scores,
        "emocion_dominante": emocion_dominante,
        "confianza": confianza,
    }


@mcp.tool()
def analizar_emociones(texto: str) -> dict:
    """Identifica las emociones evocadas en un texto de red social."""
    try:
        raw = call_gemini(prompt=texto, system=SYSTEM_PROMPT)
        return _parse_json(raw)
    except Exception:
        return _fallback_emotions(texto)


@mcp.tool()
def analizar_emociones_por_id(post_id: str) -> dict:
    """Analiza las emociones de un post especifico del dataset."""
    post = get_post(post_id)
    if post is None:
        return {"error": f"Post '{post_id}' no encontrado en el dataset."}
    texto = str(post.get("text", ""))
    resultado = analizar_emociones(texto)
    return {"post_id": post_id, "texto": texto[:200], **resultado}


if __name__ == "__main__":
    mcp.run(transport="http", port=8001)