import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

import pandas as pd
from fastmcp import FastMCP

from shared.column_map import PROPAGACION_STRATEGY
from shared.data_loader import get_post, load_dataset

load_dotenv()

mcp = FastMCP("analisis-propagacion")


def _respuestas_directas(df: pd.DataFrame, post_id: str) -> pd.DataFrame:
    """Encuentra posts que son respuesta directa al post_id dado."""
    if PROPAGACION_STRATEGY == "reply_col" and "reply_to" in df.columns:
        return df[df["reply_to"].astype(str) == str(post_id)]

    if PROPAGACION_STRATEGY == "text_mentions" and "text" in df.columns:
        return df[df["text"].astype(str).str.contains(str(post_id), na=False)]

    if PROPAGACION_STRATEGY == "time_window" and "timestamp" in df.columns:
        post = get_post(post_id)
        if post is None:
            return pd.DataFrame()
        t_orig = pd.to_datetime(post.get("timestamp"), unit="ms", errors="coerce")
        if pd.isna(t_orig):
            return pd.DataFrame()
        t_fin = t_orig + pd.Timedelta(minutes=30)
        timestamps = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        mask = (timestamps > t_orig) & (timestamps <= t_fin) & (df["post_id"].astype(str) != str(post_id))
        return df[mask]

    return pd.DataFrame()


@mcp.tool()
def analizar_propagacion(post_id: str) -> dict:
    """Analiza como se propago un mensaje especifico en la red."""
    df = load_dataset()
    post = get_post(post_id)
    if post is None:
        return {"error": f"Post '{post_id}' no encontrado en el dataset."}

    respuestas = _respuestas_directas(df, post_id)

    velocidad = "N/A"
    if not respuestas.empty and "timestamp" in df.columns:
        try:
            t_orig = pd.to_datetime(post.get("timestamp"), unit="ms", errors="coerce")
            t_resp = pd.to_datetime(respuestas["timestamp"], unit="ms", errors="coerce").min()
            if pd.notna(t_orig) and pd.notna(t_resp):
                delta_min = (t_resp - t_orig).total_seconds() / 60
                velocidad = f"{round(delta_min, 1)} min"
        except Exception:
            pass

    resultado = {
        "post_id": str(post_id),
        "texto_original": str(post.get("text", ""))[:200],
        "autor_original": str(post.get("user_id", "")),
        "timestamp_original": str(post.get("timestamp", "N/A")),
        "alcance": int(len(respuestas)),
        "usuarios_alcanzados": int(respuestas["user_id"].nunique()) if not respuestas.empty else 0,
        "velocidad_primera_respuesta": velocidad,
        "estrategia_usada": PROPAGACION_STRATEGY,
    }

    if "liked" in df.columns:
        likes_orig = len(post.get("liked", [])) if isinstance(post.get("liked", None), list) else 0
        likes_resp = int(respuestas["liked"].apply(lambda value: len(value) if isinstance(value, list) else 0).sum()) if not respuestas.empty else 0
        resultado["likes_acumulados"] = int(likes_orig + likes_resp)

    return resultado


if __name__ == "__main__":
    mcp.run(transport="http", port=8003)