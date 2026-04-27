import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

import ast
import pandas as pd
from fastmcp import FastMCP

from shared.data_loader import load_dataset

load_dotenv()

mcp = FastMCP("analisis-metricas")


def _parse_like_count(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, list):
        return float(len(value))
    text = str(value).strip()
    if not text or text == "[]":
        return 0.0
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return float(len(parsed))
    except Exception:
        pass
    return 0.0


def _engagement(df: pd.DataFrame) -> pd.Series:
    """Calcula engagement usando el mejor proxy disponible en el dataset."""
    score = pd.Series(0.0, index=df.index, dtype=float)

    if "influenceScore" in df.columns:
        influence = pd.to_numeric(df["influenceScore"], errors="coerce").fillna(0)
        if float(influence.abs().sum()) > 0:
            score += influence

    if "engagementRate" in df.columns:
        engagement_rate = pd.to_numeric(df["engagementRate"], errors="coerce").fillna(0)
        if float(engagement_rate.abs().sum()) > 0:
            score += engagement_rate

    if "liked" in df.columns:
        score += df["liked"].apply(_parse_like_count)

    return score


@mcp.tool()
def actores_influyentes(top_n: int = 10) -> list[dict]:
    """Identifica los usuarios con mayor engagement total en la conversacion."""
    df = load_dataset().copy()
    df["_engagement"] = _engagement(df)
    ranking = (
        df.groupby("user_id")
        .agg(
            posts=("post_id", "count"),
            engagement_total=("_engagement", "sum"),
        )
        .sort_values("engagement_total", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return ranking.to_dict(orient="records")


@mcp.tool()
def post_mayor_impacto() -> dict:
    """Retorna el post con mayor engagement en todo el dataset."""
    df = load_dataset().copy()
    df["_engagement"] = _engagement(df)
    top = df.loc[df["_engagement"].idxmax()]
    result = {
        "post_id": str(top.get("post_id", "")),
        "user_id": str(top.get("user_id", "")),
        "texto": str(top.get("text", ""))[:300],
        "engagement_total": float(top["_engagement"]),
    }
    if "liked" in df.columns:
        result["likes"] = int(_parse_like_count(top.get("liked", None)))
    if "timestamp" in df.columns:
        result["timestamp"] = str(top.get("timestamp", ""))
    if "influenceScore" in df.columns:
        influence_score = pd.to_numeric(pd.Series([top.get("influenceScore", 0)]), errors="coerce").fillna(0).iloc[0]
        result["influence_score"] = float(influence_score)
    return result


@mcp.tool()
def metricas_generales() -> dict:
    """Resumen estadistico global de la conversacion."""
    df = load_dataset().copy()
    df["_engagement"] = _engagement(df)
    result = {
        "total_posts": int(len(df)),
        "total_usuarios": int(df["user_id"].nunique()),
        "engagement_promedio": round(float(df["_engagement"].mean()), 2),
        "engagement_maximo": int(df["_engagement"].max()),
    }
    if "liked" in df.columns:
        result["total_likes"] = int(df["liked"].apply(_parse_like_count).sum())
    if "influenceScore" in df.columns:
        result["total_influence_score"] = float(pd.to_numeric(df["influenceScore"], errors="coerce").fillna(0).sum())
    return result


if __name__ == "__main__":
    mcp.run(transport="http", port=8002)