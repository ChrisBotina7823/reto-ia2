# Plan maestro — Reto ICESI: Agentes Conversacionales y Análisis de Conversaciones Digitales

**Stack:** Python 3.11+ · Google Gemini 1.5 Pro · LangGraph · FastMCP  
**Entorno:** Windows · Cursor / Windsurf / Copilot  
**Entregable:** 3 servicios MCP + agente conversacional con demo funcional

---

## Reglas de ejecución para el agente

- Ejecutar **en orden estricto**. No avanzar a la siguiente tarea sin confirmar que la anterior terminó sin errores.
- Usar **PowerShell** para todos los comandos de terminal.
- Ante cualquier error, leerlo completo antes de intentar corregirlo.
- **No asumir nombres de columnas** del dataset hasta haber leído `eda/reporte_eda.txt`.
- Cada archivo creado debe guardarse antes de ejecutarlo.
- El `COLUMN_MAP` es la única fuente de verdad sobre el dataset. Se define en la Fase 2 y se reutiliza en todas las fases siguientes.

---

## Fase 0 — Entorno y estructura del proyecto

### 0.1 Crear estructura de carpetas

```powershell
mkdir reto-icesi
cd reto-icesi
mkdir data, eda, shared, services\emociones, services\metricas, services\propagacion, agent, tests
```

### 0.2 Crear entorno virtual e instalar dependencias

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install fastmcp langgraph langchain-google-genai langchain-core `
    google-generativeai pandas pyarrow httpx python-dotenv pytest
```

### 0.3 Crear `.env` en la raíz del proyecto

```
GOOGLE_API_KEY=TU_CLAVE_AQUI
DATASET_PATH=./data/Reto_data_20251023_122206.parquet
MCP_EMOCIONES_URL=http://localhost:8001
MCP_METRICAS_URL=http://localhost:8002
MCP_PROPAGACION_URL=http://localhost:8003
```

> Nunca subir `.env` al repositorio. Agregar al `.gitignore`.

### 0.4 Crear `.gitignore`

```
.venv/
.env
__pycache__/
*.pyc
eda/reporte_eda.txt
```

### 0.5 Descargar el dataset

```powershell
python -c "
import urllib.request, os
url = 'https://github.com/armandoordonez/AI-Engineering/raw/main/data/Reto_data_20251023_122206.parquet'
urllib.request.urlretrieve(url, 'data/Reto_data_20251023_122206.parquet')
print('Dataset descargado.')
"
```

**Verificar:** el archivo existe en `data/` y pesa más de 0 bytes.

---

## Fase 1 — Exploración del dataset (EDA)

> Esta fase es obligatoria antes de escribir cualquier servicio.  
> El agente NO debe asumir nada sobre las columnas del dataset.

### 1.1 Crear `eda/eda.py`

```python
"""
eda.py — Exploración ciega del dataset
Sin supuestos sobre nombres de columnas.
Ejecutar: python eda/eda.py
Output:   eda/reporte_eda.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

PARQUET_PATH = os.getenv("DATASET_PATH", "./data/Reto_data_20251023_122206.parquet")
OUTPUT_PATH  = Path("eda/reporte_eda.txt")

lines = []
def log(msg=""):
    lines.append(str(msg))
    print(msg)

sep = lambda: log("=" * 60)
sub = lambda t: log(f"\n--- {t} ---")

# 1. CARGA
sep(); log("1. CARGA INICIAL")
df = pd.read_parquet(PARQUET_PATH)
log(f"Filas:    {len(df):,}")
log(f"Columnas: {len(df.columns)}")

# 2. INVENTARIO COMPLETO DE COLUMNAS
sep(); log("2. INVENTARIO DE COLUMNAS")
log(f"\n{'#':<4} {'nombre':<35} {'dtype':<15} {'nulos':>8} {'%nulos':>8} {'unicos':>10}")
log("-" * 82)
for i, col in enumerate(df.columns):
    n_nulos = int(df[col].isna().sum())
    pct     = n_nulos / len(df) * 100
    try:
        n_uniq = int(df[col].nunique())
    except Exception:
        n_uniq = -1
    log(f"{i:<4} {col:<35} {str(df[col].dtype):<15} {n_nulos:>8,} {pct:>7.1f}% {n_uniq:>10,}")

# 3. MUESTRA DE VALORES POR COLUMNA
sep(); log("3. MUESTRA DE VALORES (5 por columna)")
for col in df.columns:
    sub(col)
    sample = df[col].dropna().head(5).tolist()
    for v in sample:
        log(f"  {repr(v)[:120]}")
    if df[col].dtype == object:
        lengths = df[col].dropna().str.len()
        log(f"  [longitud min={lengths.min()} max={lengths.max()} media={lengths.mean():.0f}]")

# 4. ESTADISTICAS NUMERICAS
sep(); log("4. ESTADISTICAS DE COLUMNAS NUMERICAS")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    log(f"\n{df[num_cols].describe().T.to_string()}")
else:
    log("No hay columnas numericas.")

# 5. FRECUENCIAS DE COLUMNAS OBJETO
sep(); log("5. ESTADISTICAS DE COLUMNAS TIPO OBJECT/STRING")
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in obj_cols:
    sub(col)
    vc = df[col].value_counts()
    log(f"  Unicos: {df[col].nunique():,}")
    log(f"  Top 5 valores mas frecuentes:")
    for val, cnt in vc.head(5).items():
        log(f"    {repr(str(val))[:60]:<64} {cnt:>6,} veces")

# 6. DETECCION DE FECHAS
sep(); log("6. DETECCION DE COLUMNAS TEMPORALES")
for col in df.columns:
    if df[col].dtype == object or "datetime" in str(df[col].dtype):
        parsed = pd.to_datetime(df[col], errors="coerce")
        n_ok = parsed.notna().sum()
        if n_ok > len(df) * 0.5:
            log(f"  '{col}': {n_ok:,} fechas validas")
            log(f"    min: {parsed.min()}")
            log(f"    max: {parsed.max()}")
            log(f"    rango: {(parsed.max() - parsed.min()).days} dias")

# 7. DETECCION DE RELACIONES ENTRE FILAS
sep(); log("7. BUSQUEDA DE RELACIONES ENTRE FILAS (posibles claves foraneas)")
all_cols = df.columns.tolist()
for col in all_cols:
    for id_col in [c for c in all_cols if c != col]:
        try:
            overlap = df[col].dropna().isin(df[id_col].dropna())
            pct_overlap = overlap.sum() / len(df) * 100
            if pct_overlap > 5:
                log(f"  '{col}' solapa con '{id_col}': "
                    f"{overlap.sum():,} filas ({pct_overlap:.1f}%)")
        except Exception:
            pass

# 8. NULOS POR FILA
sep(); log("8. DISTRIBUCION DE NULOS POR FILA")
nulos_por_fila = df.isna().sum(axis=1)
log(f"  Filas sin ningun nulo:     {(nulos_por_fila == 0).sum():,}")
log(f"  Filas con al menos 1 nulo: {(nulos_por_fila > 0).sum():,}")
log(f"  Max. columnas nulas en una fila: {nulos_por_fila.max()}")

# GUARDAR
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
log(f"\nReporte guardado en: {OUTPUT_PATH}")
```

### 1.2 Ejecutar el EDA

```powershell
python eda/eda.py
```

### 1.3 Leer el reporte completo

```powershell
Get-Content eda\reporte_eda.txt
```

### 1.4 Construir el COLUMN_MAP

Después de leer el reporte, identificar el nombre real de cada concepto y crear `shared/column_map.py`:

```python
# shared/column_map.py
# Rellenar con los nombres REALES encontrados en eda/reporte_eda.txt
# No inventar ni asumir — solo lo que el reporte confirma que existe.

COLUMN_MAP = {
    "post_id":   "RELLENAR",   # columna que identifica cada post
    "user_id":   "RELLENAR",   # columna que identifica al autor
    "text":      "RELLENAR",   # columna con el contenido textual
    "timestamp": "RELLENAR",   # columna de fecha/hora (o None si no existe)
    "likes":     "RELLENAR",   # columna de likes/favoritos (o None)
    "replies":   "RELLENAR",   # columna de respuestas/comentarios (o None)
    "reply_to":  "RELLENAR",   # columna de referencia al post padre (o None)
}

# Para columnas que no existen en el dataset, dejar el valor como None:
# "reply_to": None

# ESTRATEGIA DE PROPAGACION (completar según lo hallado):
# Si reply_to existe    → grafo directo con esa columna
# Si no existe reply_to → indicar aqui la estrategia alternativa encontrada
PROPAGACION_STRATEGY = "RELLENAR"  # "reply_col" | "text_mentions" | "time_window"
```

**No avanzar a la Fase 2 hasta que todos los valores de `COLUMN_MAP` estén rellenados.**

---

## Fase 2 — Módulo compartido (`shared/`)

### 2.1 Crear `shared/__init__.py`

```python
# vacío
```

### 2.2 Crear `shared/data_loader.py`

```python
import pandas as pd
from functools import lru_cache
import os
from shared.column_map import COLUMN_MAP

@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    path = os.getenv("DATASET_PATH", "./data/Reto_data_20251023_122206.parquet")
    df = pd.read_parquet(path)
    # Renombrar columnas reales al esquema interno estándar
    rename = {v: k for k, v in COLUMN_MAP.items()
              if v is not None and v in df.columns}
    df = df.rename(columns=rename)
    return df

def get_post(post_id: str) -> dict | None:
    df = load_dataset()
    row = df[df["post_id"].astype(str) == str(post_id)]
    return row.iloc[0].to_dict() if not row.empty else None
```

### 2.3 Crear `shared/gemini_client.py`

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

_model = None

def get_model():
    global _model
    if _model is None:
        _model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
        )
    return _model

def call_gemini(prompt: str, system: str = "") -> str:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=system if system else None,
    )
    response = model.generate_content(prompt)
    return response.text.strip()
```

### 2.4 Verificar que el módulo shared funciona

```powershell
python -c "from shared.data_loader import load_dataset; df = load_dataset(); print(df.columns.tolist()); print(len(df))"
```

**Verificar:** imprime las columnas renombradas y el número de filas sin error.

---

## Fase 3 — Microservicio MCP #1: Análisis de emociones

### 3.1 Crear `services/emociones/__init__.py`

```python
# vacío
```

### 3.2 Crear `services/emociones/main.py`

```python
from fastmcp import FastMCP
from shared.data_loader import load_dataset, get_post
from shared.gemini_client import call_gemini
import json, re
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("analisis-emociones", port=8001)

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
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Respuesta no parseable: {raw[:200]}")

@mcp.tool()
def analizar_emociones(texto: str) -> dict:
    """Identifica las emociones evocadas en un texto de red social.

    Args:
        texto: El texto del post o comentario a analizar.

    Returns:
        Dict con scores por emocion, emocion dominante y confianza.
    """
    raw = call_gemini(prompt=texto, system=SYSTEM_PROMPT)
    return _parse_json(raw)

@mcp.tool()
def analizar_emociones_por_id(post_id: str) -> dict:
    """Analiza las emociones de un post especifico del dataset.

    Args:
        post_id: ID del post en el dataset.

    Returns:
        Dict con post_id, texto usado y resultado del analisis.
    """
    post = get_post(post_id)
    if post is None:
        return {"error": f"Post '{post_id}' no encontrado en el dataset."}
    texto = str(post.get("text", ""))
    resultado = analizar_emociones(texto)
    return {"post_id": post_id, "texto": texto[:200], **resultado}

if __name__ == "__main__":
    mcp.run()
```

### 3.3 Probar el servicio de emociones en terminal separada

```powershell
# Terminal A
cd reto-icesi
.venv\Scripts\Activate.ps1
python services/emociones/main.py
```

### 3.4 Verificar en otra terminal que responde

```powershell
# Terminal B
python -c "
import httpx
r = httpx.post('http://localhost:8001/api/v1/analizar_emociones',
               json={'texto': 'Estoy muy feliz con este resultado!'})
print(r.status_code)
print(r.json())
"
```

**Verificar:** `status_code == 200` y la respuesta contiene `emocion_dominante`.

---

## Fase 4 — Microservicio MCP #2: Análisis de métricas

### 4.1 Crear `services/metricas/__init__.py`

```python
# vacío
```

### 4.2 Crear `services/metricas/main.py`

> Antes de escribir este archivo, confirmar en `shared/column_map.py` qué columnas
> de engagement existen realmente (`likes`, `replies`). Si alguna es `None`,
> eliminar su uso en las funciones de abajo y ajustar el cálculo de engagement.

```python
from fastmcp import FastMCP
from shared.data_loader import load_dataset
from shared.column_map import COLUMN_MAP
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("analisis-metricas", port=8002)

def _engagement(df: pd.DataFrame) -> pd.Series:
    """Calcula engagement sumando las columnas disponibles."""
    total = pd.Series(0, index=df.index, dtype=float)
    if "likes" in df.columns:
        total += pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    if "replies" in df.columns:
        total += pd.to_numeric(df["replies"], errors="coerce").fillna(0)
    return total

@mcp.tool()
def actores_influyentes(top_n: int = 10) -> list[dict]:
    """Identifica los usuarios con mayor engagement total en la conversacion.

    Args:
        top_n: Cantidad de usuarios a retornar (por defecto 10).

    Returns:
        Lista de usuarios ordenados por engagement total.
    """
    df = load_dataset()
    df["_engagement"] = _engagement(df)
    agg = {"_posts": ("post_id", "count"), "_engagement_total": ("_engagement", "sum")}
    if "likes" in df.columns:
        agg["_likes"] = ("likes", "sum")
    if "replies" in df.columns:
        agg["_replies"] = ("replies", "sum")
    ranking = (
        df.groupby("user_id")
          .agg(**agg)
          .sort_values("_engagement_total", ascending=False)
          .head(top_n)
          .reset_index()
    )
    ranking.columns = [c.lstrip("_") for c in ranking.columns]
    return ranking.to_dict(orient="records")

@mcp.tool()
def post_mayor_impacto() -> dict:
    """Retorna el post con mayor engagement en todo el dataset.

    Returns:
        Dict con datos del post mas viral y sus metricas.
    """
    df = load_dataset()
    df["_engagement"] = _engagement(df)
    top = df.loc[df["_engagement"].idxmax()]
    result = {
        "post_id": str(top.get("post_id", "")),
        "user_id": str(top.get("user_id", "")),
        "texto":   str(top.get("text", ""))[:300],
        "engagement_total": float(top["_engagement"]),
    }
    if "likes" in df.columns:
        result["likes"] = int(top.get("likes", 0))
    if "replies" in df.columns:
        result["replies"] = int(top.get("replies", 0))
    if "timestamp" in df.columns:
        result["timestamp"] = str(top.get("timestamp", ""))
    return result

@mcp.tool()
def metricas_generales() -> dict:
    """Resumen estadistico global de la conversacion.

    Returns:
        Dict con totales y promedios de toda la conversacion.
    """
    df = load_dataset()
    df["_engagement"] = _engagement(df)
    result = {
        "total_posts":    int(len(df)),
        "total_usuarios": int(df["user_id"].nunique()),
        "engagement_promedio": round(float(df["_engagement"].mean()), 2),
        "engagement_maximo":   int(df["_engagement"].max()),
    }
    if "likes" in df.columns:
        result["total_likes"] = int(pd.to_numeric(df["likes"], errors="coerce").sum())
    if "replies" in df.columns:
        result["total_replies"] = int(pd.to_numeric(df["replies"], errors="coerce").sum())
    return result

if __name__ == "__main__":
    mcp.run()
```

### 4.3 Probar el servicio de métricas

```powershell
# Terminal A (detener emociones primero si está corriendo, o usar nueva terminal)
python services/metricas/main.py
```

```powershell
# Terminal B — verificar
python -c "
import httpx
r = httpx.post('http://localhost:8002/api/v1/metricas_generales', json={})
print(r.status_code, r.json())
"
```

**Verificar:** `status_code == 200` y el JSON contiene `total_posts`.

---

## Fase 5 — Microservicio MCP #3: Propagación de mensajes (obligatorio)

> Esta es la fase más dependiente del EDA. Leer `PROPAGACION_STRATEGY`
> en `shared/column_map.py` antes de elegir qué bloque implementar.

### 5.1 Crear `services/propagacion/__init__.py`

```python
# vacío
```

### 5.2 Crear `services/propagacion/main.py`

Implementar según la estrategia encontrada en el EDA:

```python
from fastmcp import FastMCP
from shared.data_loader import load_dataset, get_post
from shared.column_map import COLUMN_MAP, PROPAGACION_STRATEGY
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("analisis-propagacion", port=8003)

def _respuestas_directas(df: pd.DataFrame, post_id: str) -> pd.DataFrame:
    """Encuentra posts que son respuesta directa al post_id dado."""

    if PROPAGACION_STRATEGY == "reply_col" and "reply_to" in df.columns:
        # Estrategia A: existe columna explícita de relación
        return df[df["reply_to"].astype(str) == str(post_id)]

    elif PROPAGACION_STRATEGY == "text_mentions" and "text" in df.columns:
        # Estrategia B: buscar menciones del ID en el texto
        return df[df["text"].str.contains(str(post_id), na=False)]

    elif PROPAGACION_STRATEGY == "time_window" and "timestamp" in df.columns:
        # Estrategia C: posts del mismo hilo en ventana de 30 min tras el post
        post = get_post(post_id)
        if post is None:
            return pd.DataFrame()
        t_orig = pd.to_datetime(post.get("timestamp"))
        t_fin  = t_orig + pd.Timedelta(minutes=30)
        mask = (
            (pd.to_datetime(df["timestamp"], errors="coerce") > t_orig) &
            (pd.to_datetime(df["timestamp"], errors="coerce") <= t_fin) &
            (df["post_id"].astype(str) != str(post_id))
        )
        return df[mask]

    else:
        return pd.DataFrame()

@mcp.tool()
def analizar_propagacion(post_id: str) -> dict:
    """Analiza como se propago un mensaje especifico en la red.

    Mide el alcance, velocidad y usuarios impactados a partir del ID del post.

    Args:
        post_id: ID del mensaje original a analizar.

    Returns:
        Dict con alcance, velocidad de propagacion y metricas del mensaje.
    """
    df = load_dataset()

    post = get_post(post_id)
    if post is None:
        return {"error": f"Post '{post_id}' no encontrado en el dataset."}

    respuestas = _respuestas_directas(df, post_id)

    velocidad = "N/A"
    if not respuestas.empty and "timestamp" in df.columns:
        try:
            t_orig = pd.to_datetime(post.get("timestamp"))
            t_resp = pd.to_datetime(respuestas["timestamp"], errors="coerce").min()
            delta_min = (t_resp - t_orig).total_seconds() / 60
            velocidad = f"{round(delta_min, 1)} min"
        except Exception:
            pass

    resultado = {
        "post_id":                  str(post_id),
        "texto_original":           str(post.get("text", ""))[:200],
        "autor_original":           str(post.get("user_id", "")),
        "timestamp_original":       str(post.get("timestamp", "N/A")),
        "alcance":                  int(len(respuestas)),
        "usuarios_alcanzados":      int(respuestas["user_id"].nunique()) if not respuestas.empty else 0,
        "velocidad_primera_respuesta": velocidad,
        "estrategia_usada":         PROPAGACION_STRATEGY,
    }

    if "likes" in df.columns:
        likes_orig = pd.to_numeric(post.get("likes", 0), errors="coerce") or 0
        likes_resp = pd.to_numeric(respuestas.get("likes", pd.Series(dtype=float)), errors="coerce").sum() if not respuestas.empty else 0
        resultado["likes_acumulados"] = int(likes_orig + likes_resp)

    return resultado

if __name__ == "__main__":
    mcp.run()
```

### 5.3 Probar el servicio de propagación

```powershell
python services/propagacion/main.py
```

```powershell
# Obtener un post_id real del dataset para probar
python -c "
from shared.data_loader import load_dataset
df = load_dataset()
print(df['post_id'].head(3).tolist())
"
```

```powershell
python -c "
import httpx
r = httpx.post('http://localhost:8003/api/v1/analizar_propagacion',
               json={'post_id': 'PONER_ID_REAL_AQUI'})
print(r.status_code, r.json())
"
```

**Verificar:** respuesta con `alcance` numérico, sin errores de columna.

---

## Fase 6 — Agente conversacional con LangGraph

### 6.1 Crear `agent/__init__.py`

```python
# vacío
```

### 6.2 Crear `agent/tools.py`

```python
import httpx
import os
from langchain_core.tools import tool

MCP_EMOCIONES   = os.getenv("MCP_EMOCIONES_URL",   "http://localhost:8001")
MCP_METRICAS    = os.getenv("MCP_METRICAS_URL",    "http://localhost:8002")
MCP_PROPAGACION = os.getenv("MCP_PROPAGACION_URL", "http://localhost:8003")

def _call_mcp(base_url: str, endpoint: str, payload: dict) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{base_url}/api/v1/{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()

@tool
def analizar_emociones(texto: str) -> dict:
    """Analiza las emociones presentes en un texto de red social.
    Usar cuando el usuario pregunte por sentimientos, emociones, estado
    emocional, o el clima de la conversacion en un texto especifico."""
    return _call_mcp(MCP_EMOCIONES, "analizar_emociones", {"texto": texto})

@tool
def analizar_emociones_por_id(post_id: str) -> dict:
    """Analiza las emociones de un post especifico usando su ID.
    Usar cuando el usuario mencione un ID de post y quiera saber
    las emociones o sentimientos de ese post en particular."""
    return _call_mcp(MCP_EMOCIONES, "analizar_emociones_por_id", {"post_id": post_id})

@tool
def actores_influyentes(top_n: int = 10) -> list:
    """Identifica los usuarios mas influyentes por engagement total.
    Usar cuando pregunten quienes dominan la conversacion, quienes
    tienen mas impacto, repercusion, likes o son mas activos."""
    return _call_mcp(MCP_METRICAS, "actores_influyentes", {"top_n": top_n})

@tool
def post_mayor_impacto() -> dict:
    """Encuentra el post con mayor engagement o viralidad en el dataset.
    Usar cuando pregunten cual es el post mas viral, popular o con
    mayor repercusion."""
    return _call_mcp(MCP_METRICAS, "post_mayor_impacto", {})

@tool
def metricas_generales() -> dict:
    """Obtiene estadisticas globales de la conversacion: total de posts,
    usuarios, likes, engagement. Usar cuando pidan un resumen numerico,
    estadistico o general de toda la conversacion."""
    return _call_mcp(MCP_METRICAS, "metricas_generales", {})

@tool
def analizar_propagacion(post_id: str) -> dict:
    """Analiza como se propago un mensaje especifico en la red.
    Usar cuando pregunten por el alcance, velocidad, impacto o
    difusion de un post concreto. Requiere el ID exacto del post.

    Args:
        post_id: El ID exacto del post a analizar.
    """
    return _call_mcp(MCP_PROPAGACION, "analizar_propagacion", {"post_id": post_id})

TOOLS = [
    analizar_emociones,
    analizar_emociones_por_id,
    actores_influyentes,
    post_mayor_impacto,
    metricas_generales,
    analizar_propagacion,
]
```

### 6.3 Crear `agent/graph.py`

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
from agent.tools import TOOLS

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
llm_with_tools = llm.bind_tools(TOOLS)

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
    response = llm_with_tools.invoke(messages)
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
```

### 6.4 Crear `agent/chat.py`

```python
from agent.graph import graph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def chat():
    print("=" * 50)
    print("Agente Conversacional - Reto ICESI")
    print("Escribe 'salir' para terminar.")
    print("=" * 50)

    history = []

    while True:
        user_input = input("\nTu: ").strip()
        if user_input.lower() in ("salir", "exit", "quit"):
            print("Sesion terminada.")
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": history})
        history = result["messages"]

        last = result["messages"][-1]
        print(f"\nAgente: {last.content}")

if __name__ == "__main__":
    chat()
```

---

## Fase 7 — Tests de integración

### 7.1 Crear `tests/__init__.py`

```python
# vacío
```

### 7.2 Crear `tests/test_mcps.py`

```python
import pytest
import httpx

BASE_EMOCIONES   = "http://localhost:8001"
BASE_METRICAS    = "http://localhost:8002"
BASE_PROPAGACION = "http://localhost:8003"

# ── Emociones ────────────────────────────────────────────────────

def test_emociones_texto_positivo():
    r = httpx.post(f"{BASE_EMOCIONES}/api/v1/analizar_emociones",
                   json={"texto": "Estoy muy feliz con este resultado!"}, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert "emocion_dominante" in data
    assert "emociones" in data
    assert "confianza" in data

def test_emociones_texto_negativo():
    r = httpx.post(f"{BASE_EMOCIONES}/api/v1/analizar_emociones",
                   json={"texto": "Estoy furioso, esto es una injusticia."}, timeout=30)
    assert r.status_code == 200
    assert "emocion_dominante" in r.json()

def test_emociones_por_id_inexistente():
    r = httpx.post(f"{BASE_EMOCIONES}/api/v1/analizar_emociones_por_id",
                   json={"post_id": "ID_QUE_NO_EXISTE_999999"}, timeout=30)
    assert r.status_code == 200
    assert "error" in r.json()

# ── Metricas ─────────────────────────────────────────────────────

def test_metricas_generales():
    r = httpx.post(f"{BASE_METRICAS}/api/v1/metricas_generales",
                   json={}, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert "total_posts" in data
    assert data["total_posts"] > 0

def test_actores_influyentes():
    r = httpx.post(f"{BASE_METRICAS}/api/v1/actores_influyentes",
                   json={"top_n": 5}, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) <= 5

def test_post_mayor_impacto():
    r = httpx.post(f"{BASE_METRICAS}/api/v1/post_mayor_impacto",
                   json={}, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert "post_id" in data
    assert "engagement_total" in data

# ── Propagacion ───────────────────────────────────────────────────

def test_propagacion_id_inexistente():
    r = httpx.post(f"{BASE_PROPAGACION}/api/v1/analizar_propagacion",
                   json={"post_id": "ID_QUE_NO_EXISTE_999999"}, timeout=30)
    assert r.status_code == 200
    assert "error" in r.json()

def test_propagacion_id_real():
    # Obtener un ID real del dataset para esta prueba
    from shared.data_loader import load_dataset
    df = load_dataset()
    post_id = str(df["post_id"].iloc[0])

    r = httpx.post(f"{BASE_PROPAGACION}/api/v1/analizar_propagacion",
                   json={"post_id": post_id}, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert "alcance" in data
    assert "usuarios_alcanzados" in data
```

---

## Fase 8 — Ejecución completa y demo

### 8.1 Arrancar los tres servicios (terminales separadas)

```powershell
# Terminal 1
cd reto-icesi && .venv\Scripts\Activate.ps1
python services/emociones/main.py

# Terminal 2
cd reto-icesi && .venv\Scripts\Activate.ps1
python services/metricas/main.py

# Terminal 3
cd reto-icesi && .venv\Scripts\Activate.ps1
python services/propagacion/main.py
```

### 8.2 Correr los tests (con los tres servicios activos)

```powershell
# Terminal 4
cd reto-icesi && .venv\Scripts\Activate.ps1
pytest tests/ -v
```

**Verificar:** todos los tests pasan en verde antes de la demo.

### 8.3 Arrancar el agente

```powershell
# Terminal 4 (la misma de tests)
python -m agent.chat
```

### 8.4 Interacciones mínimas para la demo

Ejecutar estas preguntas en orden durante la demo. Cubren los tres MCPs y demuestran memoria de contexto:

```
¿Cuántos posts y usuarios tiene la conversación?
¿Cuáles son los 5 usuarios más influyentes?
¿Cuál es el post que más impacto ha tenido?
Analiza las emociones de ese post.
¿La emoción dominante es positiva o negativa?
¿Cómo se propagó ese mensaje en la red?
```

La última pregunta de seguimiento ("¿La emoción dominante es positiva o negativa?")
demuestra que el agente recuerda el contexto sin volver a llamar la herramienta.

---

## Árbol final del proyecto

```
reto-icesi/
├── .env
├── .gitignore
├── data/
│   └── Reto_data_20251023_122206.parquet
├── eda/
│   ├── eda.py
│   └── reporte_eda.txt          ← generado al ejecutar eda.py
├── shared/
│   ├── __init__.py
│   ├── column_map.py            ← rellenar tras leer el reporte EDA
│   ├── data_loader.py
│   └── gemini_client.py
├── services/
│   ├── emociones/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── metricas/
│   │   ├── __init__.py
│   │   └── main.py
│   └── propagacion/
│       ├── __init__.py
│       └── main.py
├── agent/
│   ├── __init__.py
│   ├── tools.py
│   ├── graph.py
│   └── chat.py
└── tests/
    ├── __init__.py
    └── test_mcps.py
```

---

## Checklist de entrega

- [ ] `eda/reporte_eda.txt` generado y leído
- [ ] `shared/column_map.py` con todos los valores reales del dataset
- [ ] Los 3 servicios MCP arrancan sin errores en sus puertos respectivos
- [ ] `pytest tests/ -v` pasa todos los tests en verde
- [ ] `python -m agent.chat` inicia sin errores
- [ ] El agente responde las 6 preguntas de demo usando las herramientas MCP correctas
- [ ] El agente demuestra memoria de contexto en preguntas de seguimiento
