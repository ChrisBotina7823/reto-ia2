import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import graph
from agent.tools import (
    actores_influyentes,
    analizar_emociones,
    analizar_emociones_por_id,
    analizar_propagacion,
    metricas_generales,
    post_mayor_impacto,
)

load_dotenv()

def _render_assistant_content(content) -> str:
    """
    Gemini/LangChain a veces devuelve `content` como lista de "parts"
    (dicts con keys como type/text). En consola queremos texto legible.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                continue
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
        if parts:
            return "\n".join(parts)
    return str(content)


def _has_gemini_key() -> bool:
    # LangChain acepta GOOGLE_API_KEY o GEMINI_API_KEY dependiendo del stack.
    import os

    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    return bool(key)


def _infer_post_id(text: str) -> str | None:
    """
    Heurística simple: busca un token parecido a un id (hex largo o prefijos tipo tikapi_).
    """
    import re

    m = re.search(r"\b(tikapi_[A-Za-z0-9_]+)\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b[a-f0-9]{16,64}\b", text.lower())
    if m:
        return m.group(0)
    return None


def _invoke_tool(tool_obj, payload: dict):
    """
    En este repo, las herramientas están declaradas con @tool (LangChain),
    así que no son funciones llamables directamente en modo degradado.
    Se invocan con .invoke(payload).
    """
    return tool_obj.invoke(payload)


def _fallback_answer(user_input: str, memory: dict) -> str:
    """
    Modo degradado SIN LLM: enruta por intención y llama directamente a los MCP.
    Mantiene un pequeño estado (último post y último análisis) para preguntas de seguimiento.
    """
    t = user_input.strip().lower()

    if any(k in t for k in ["metricas", "métricas", "cuántos", "cuantos", "posts", "usuarios", "estadisticas", "estadísticas"]):
        data = _invoke_tool(metricas_generales, {})
        memory["last_tool"] = "metricas_generales"
        memory["last_result"] = data
        return (
            f"En el dataset hay {data.get('total_posts')} posts y {data.get('total_usuarios')} usuarios. "
            f"Engagement promedio: {data.get('engagement_promedio')}, máximo: {data.get('engagement_maximo')}."
        )

    if any(k in t for k in ["influy", "influ", "actores", "top"]):
        top_n = 5 if "5" in t else 10
        data = _invoke_tool(actores_influyentes, {"top_n": top_n})
        memory["last_tool"] = "actores_influyentes"
        memory["last_result"] = data
        lines = [f"{i+1}. user_id={row.get('user_id')} (posts={row.get('posts')}, engagement_total={row.get('engagement_total')})" for i, row in enumerate(data)]
        return "Top de actores influyentes:\n" + "\n".join(lines)

    if ("post" in t and any(k in t for k in ["impact", "viral", "repercusion", "repercusión", "mayor"])) or "más impacto" in t:
        data = _invoke_tool(post_mayor_impacto, {})
        memory["last_tool"] = "post_mayor_impacto"
        memory["last_result"] = data
        memory["last_post_id"] = data.get("post_id")
        return (
            f"El post con mayor impacto es {data.get('post_id')} (autor {data.get('user_id')}). "
            f"Engagement total: {data.get('engagement_total')}. Texto (recorte): {data.get('texto')}"
        )

    if any(k in t for k in ["emocion", "emociones", "sentimiento"]):
        post_id = _infer_post_id(user_input) or memory.get("last_post_id")
        if post_id:
            data = _invoke_tool(analizar_emociones_por_id, {"post_id": post_id})
        else:
            data = _invoke_tool(analizar_emociones, {"texto": user_input})
        memory["last_tool"] = "emociones"
        memory["last_result"] = data
        dom = data.get("emocion_dominante")
        conf = data.get("confianza")
        return f"Emoción dominante: {dom}. Confianza: {conf}. Detalle: {data.get('emociones')}"

    if any(k in t for k in ["propag", "difusi", "difusión", "alcance", "velocidad"]):
        post_id = _infer_post_id(user_input) or memory.get("last_post_id")
        if not post_id:
            return "Necesito un `post_id` (o primero pregunta por el post de mayor impacto) para analizar propagación."
        data = _invoke_tool(analizar_propagacion, {"post_id": post_id})
        memory["last_tool"] = "propagacion"
        memory["last_result"] = data
        return (
            f"Propagación de {data.get('post_id')}: alcance={data.get('alcance')}, "
            f"usuarios_alcanzados={data.get('usuarios_alcanzados')}, "
            f"primera_respuesta={data.get('velocidad_primera_respuesta')}, "
            f"estrategia={data.get('estrategia_usada')}."
        )

    # Pregunta de seguimiento típica del demo: "¿La emoción dominante es positiva o negativa?"
    if "positiv" in t or "negativ" in t:
        last = memory.get("last_result") if memory.get("last_tool") == "emociones" else None
        if not isinstance(last, dict) or "emocion_dominante" not in last:
            return "No tengo un análisis de emociones previo. Pide primero: 'Analiza las emociones de ...'."
        dom = str(last.get("emocion_dominante"))
        positivos = {"alegria", "sorpresa"}
        negativos = {"tristeza", "enojo", "miedo", "disgusto"}
        if dom in positivos:
            pol = "más bien positiva"
        elif dom in negativos:
            pol = "más bien negativa"
        else:
            pol = "mixta/neutral"
        return f"La emoción dominante fue **{dom}**, así que el tono es {pol}."

    return (
        "No tengo LLM configurado (falta `GOOGLE_API_KEY`) y no pude inferir la intención.\n"
        "Prueba con: 'métricas generales', 'top 5 influyentes', 'post de mayor impacto', "
        "'emociones del post <id>', o 'propagación del post <id>'."
    )


def chat():
    print("=" * 50)
    print("Agente Conversacional - Reto ICESI")
    print("Escribe 'salir' para terminar.")
    print("=" * 50)

    history = []
    memory: dict = {}
    degraded = not _has_gemini_key()
    if degraded:
        print("\n[Modo degradado] No hay `GOOGLE_API_KEY`. Se usará ruteo local + MCP (sin LLM).")

    while True:
        user_input = input("\nTu: ").strip()
        if user_input.lower() in ("salir", "exit", "quit"):
            print("Sesion terminada.")
            break
        if not user_input:
            continue

        if degraded:
            answer = _fallback_answer(user_input, memory)
            print(f"\nAgente: {answer}")
            continue

        history.append(HumanMessage(content=user_input))
        try:
            result = graph.invoke({"messages": history})
            history = result["messages"]

            last = result["messages"][-1]
            print(f"\nAgente: {_render_assistant_content(last.content)}")
        except Exception as exc:
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                print(
                    "\n[Error Gemini 429] Cuota/rate-limit alcanzado. "
                    "Espera un momento y reintenta, o configura modo degradado borrando GOOGLE_API_KEY."
                )
                continue
            elif "API_KEY_INVALID" in msg or "API key not valid" in msg or ("INVALID_ARGUMENT" in msg and "400" in msg):
                print(
                    "\n[Error Gemini 400] GOOGLE_API_KEY invalida. "
                    "Verifica que pegaste la key exacta en .env."
                )
                break
            else:
                print(f"\n[Error inesperado] {msg}")
                raise


if __name__ == "__main__":
    chat()