"""
eda.py — Exploración ciega del dataset
Sin supuestos sobre nombres de columnas.
Ejecutar: python eda/eda.py
Output:   eda/reporte_eda.txt
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


PARQUET_PATH = os.getenv("DATASET_PATH", "./data/Reto_data_20251023_122206.parquet")
OUTPUT_PATH = Path("eda/reporte_eda.txt")

lines: list[str] = []


def log(msg: str = "") -> None:
    lines.append(str(msg))
    print(msg)


def sep() -> None:
    log("=" * 60)


def sub(t: str) -> None:
    log(f"\n--- {t} ---")


sep()
log("1. CARGA INICIAL")
df = pd.read_parquet(PARQUET_PATH)
log(f"Filas:    {len(df):,}")
log(f"Columnas: {len(df.columns)}")

sep()
log("2. INVENTARIO DE COLUMNAS")
log(f"\n{'#':<4} {'nombre':<35} {'dtype':<15} {'nulos':>8} {'%nulos':>8} {'unicos':>10}")
log("-" * 82)
for i, col in enumerate(df.columns):
    n_nulos = int(df[col].isna().sum())
    pct = n_nulos / len(df) * 100
    try:
        n_uniq = int(df[col].nunique())
    except Exception:
        n_uniq = -1
    log(f"{i:<4} {col:<35} {str(df[col].dtype):<15} {n_nulos:>8,} {pct:>7.1f}% {n_uniq:>10,}")

sep()
log("3. MUESTRA DE VALORES (5 por columna)")
for col in df.columns:
    sub(col)
    sample = df[col].dropna().head(5).tolist()
    for v in sample:
        log(f"  {repr(v)[:120]}")
    if df[col].dtype == object:
        lengths = df[col].dropna().astype(str).str.len()
        if not lengths.empty:
            log(f"  [longitud min={lengths.min()} max={lengths.max()} media={lengths.mean():.0f}]")

sep()
log("4. ESTADISTICAS DE COLUMNAS NUMERICAS")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    log(f"\n{df[num_cols].describe().T.to_string()}")
else:
    log("No hay columnas numericas.")

sep()
log("5. ESTADISTICAS DE COLUMNAS TIPO OBJECT/STRING")
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
cols_relevantes = [
    c for c in [
        "id",
        "author",
        "authorId",
        "createdAt",
        "liked",
        "isReplied",
        "parentId",
        "threadId",
        "text",
        "title",
        "sentiment",
        "socialType",
        "sourceName",
        "type",
    ]
    if c in obj_cols
]
for col in cols_relevantes:
    sub(col)
    vc = df[col].value_counts()
    log(f"  Unicos: {df[col].nunique():,}")
    log("  Top 5 valores mas frecuentes:")
    for val, cnt in vc.head(5).items():
        log(f"    {repr(str(val))[:60]:<64} {cnt:>6,} veces")

sep()
log("6. DETECCION DE COLUMNAS TEMPORALES")
for col in df.columns:
    if df[col].dtype == object or "datetime" in str(df[col].dtype):
        parsed = pd.to_datetime(df[col], errors="coerce")
        n_ok = parsed.notna().sum()
        if n_ok > len(df) * 0.5:
            log(f"  '{col}': {n_ok:,} fechas validas")
            log(f"    min: {parsed.min()}")
            log(f"    max: {parsed.max()}")
            log(f"    rango: {(parsed.max() - parsed.min()).days} dias")

sep()
log("7. BUSQUEDA DE RELACIONES ENTRE FILAS (posibles claves foraneas)")
log("  parentId es la referencia de respuesta directa al post padre")
log("  threadId agrupa conversaciones/hilos y complementa el rastreo de propagacion")

sep()
log("8. DISTRIBUCION DE NULOS POR FILA")
nulos_por_fila = df.isna().sum(axis=1)
log(f"  Filas sin ningun nulo:     {(nulos_por_fila == 0).sum():,}")
log(f"  Filas con al menos 1 nulo: {(nulos_por_fila > 0).sum():,}")
log(f"  Max. columnas nulas en una fila: {nulos_por_fila.max()}")

sep()
log("9. CANDIDATOS PARA COLUMN_MAP")
log("  post_id   -> id")
log("  user_id   -> author")
log("  text      -> text")
log("  timestamp -> createdAt")
log("  likes     -> liked")
log("  replies   -> None")
log("  reply_to  -> parentId")
log("  propagacion -> reply_col")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
log(f"\nReporte guardado en: {OUTPUT_PATH}")