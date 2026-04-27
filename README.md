# Reto ICESI: Agentes Conversacionales y Análisis de Conversaciones Digitales

Implementación en Python 3.11+ de un sistema compuesto por tres servicios MCP, un agente conversacional orquestado con LangGraph y un flujo de análisis sobre un dataset de conversaciones digitales.

El proyecto está diseñado para funcionar en Windows con PowerShell y usa FastMCP como capa de servicios, Google Gemini como modelo principal y un fallback local para mantener funcionalidad básica cuando no hay clave de API configurada.

## Qué incluye

- Un análisis exploratorio del dataset en [eda/reporte_eda.txt](eda/reporte_eda.txt).
- Un mapa canónico de columnas en [shared/column_map.py](shared/column_map.py).
- Tres servicios MCP independientes.
- Un agente conversacional que consume esos servicios.
- Pruebas de integración sobre los servicios en vivo.

## Arquitectura

El flujo general es:

1. Se carga el dataset parquet desde la ruta definida en `DATASET_PATH`.
2. El EDA identifica las columnas reales y permite construir el `COLUMN_MAP`.
3. El módulo compartido normaliza el dataset a un esquema interno estándar.
4. Cada servicio MCP expone herramientas especializadas.
5. El agente conversa con el usuario y llama herramientas del MCP correcto según la intención.

### Servicios

- `services/emociones`: clasifica emociones del texto o de un post del dataset.
- `services/metricas`: calcula métricas globales e identifica actores influyentes.
- `services/propagacion`: analiza la propagación de un mensaje usando la estrategia de respuesta directa.

### Agente

El agente vive en `agent/chat.py` y está respaldado por un grafo LangGraph definido en `agent/graph.py`. Sus herramientas usan el cliente FastMCP para conectarse a los servicios en `http://localhost:8001/mcp`, `http://localhost:8002/mcp` y `http://localhost:8003/mcp`.

## Estructura del repositorio

- `agent/`: grafo, herramientas y chat interactivo.
- `data/`: dataset parquet descargado.
- `eda/`: script y reporte exploratorio.
- `services/emociones/`: MCP de emociones.
- `services/metricas/`: MCP de métricas.
- `services/propagacion/`: MCP de propagación.
- `shared/`: cargador de datos, mapa de columnas y cliente Gemini.
- `tests/`: pruebas de integración contra los servicios activos.
- `plan_agente_reto_icesi.md`: plan original de ejecución.

## Requisitos

- Python 3.11 o superior.
- PowerShell en Windows.
- Conexión a internet para instalar dependencias y, si se desea usar Gemini, para consultar la API.

## Configuración inicial

1. Crea y activa el entorno virtual.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instala dependencias.

```powershell
pip install fastmcp langgraph langchain-google-genai langchain-core google-generativeai pandas pyarrow httpx python-dotenv pytest
```

3. Verifica el archivo `.env` en la raíz.

El proyecto usa estas variables:

- `GOOGLE_API_KEY`: clave de Gemini. Si no está presente, el servicio de emociones usa un fallback local.
- `GEMINI_MODEL`: modelo principal para llamadas a Gemini. Recomendado: `gemini-2.0-flash`.
- `GEMINI_MODEL_FALLBACKS`: modelos alternativos separados por coma si el principal no existe en tu API, por ejemplo: `gemini-1.5-flash,gemini-1.5-pro`.
- `DATASET_PATH`: ruta al parquet del dataset.
- `MCP_EMOCIONES_URL`: URL base del servicio de emociones.
- `MCP_METRICAS_URL`: URL base del servicio de métricas.
- `MCP_PROPAGACION_URL`: URL base del servicio de propagación.

## Dataset

El dataset esperado está en:

`data/Reto_data_20251023_122206.parquet`

Si todavía no existe, puedes descargarlo manualmente desde el enlace original del reto o reutilizar el archivo ya incluido en el entorno.

El EDA confirmó que el esquema canónico se apoya en estas columnas reales:

- `post_id` -> `id`
- `user_id` -> `author`
- `text` -> `text`
- `timestamp` -> `createdAt`
- `likes` -> `liked`
- `replies` -> no existe en el dataset
- `reply_to` -> `parentId`

La estrategia de propagación elegida es `reply_col`, porque el dataset sí contiene una columna de referencia al mensaje padre.

## Cómo ejecutar el EDA

El script de exploración está en [eda/eda.py](eda/eda.py).

```powershell
.\.venv\Scripts\python.exe eda\eda.py
```

El resultado se guarda en [eda/reporte_eda.txt](eda/reporte_eda.txt).

## Cómo ejecutar los servicios MCP

En tres terminales distintas, desde la raíz del repo:

```powershell
.\.venv\Scripts\python.exe services\emociones\main.py
```

```powershell
.\.venv\Scripts\python.exe services\metricas\main.py
```

```powershell
.\.venv\Scripts\python.exe services\propagacion\main.py
```

Cada servicio expone su transporte HTTP de FastMCP en:

- `http://127.0.0.1:8001/mcp`
- `http://127.0.0.1:8002/mcp`
- `http://127.0.0.1:8003/mcp`

## Herramientas disponibles

### Emociones

- `analizar_emociones(texto)`
- `analizar_emociones_por_id(post_id)`

### Métricas

- `actores_influyentes(top_n)`
- `post_mayor_impacto()`
- `metricas_generales()`

### Propagación

- `analizar_propagacion(post_id)`

## Cómo ejecutar el agente

Antes de abrir el chat, asegúrate de que los tres servicios estén corriendo.

```powershell
.\.venv\Scripts\python.exe agent\chat.py
```

El agente mantiene el historial de la conversación y decide qué herramienta llamar según la pregunta del usuario.

## Cómo correr las pruebas

Las pruebas de integración usan el cliente FastMCP contra los tres servicios en vivo.

1. Arranca los tres servicios MCP.
2. Ejecuta:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_mcps.py
```

## Qué hace cada módulo compartido

- [shared/data_loader.py](shared/data_loader.py): carga el parquet y renombra las columnas reales al esquema interno.
- [shared/column_map.py](shared/column_map.py): fuente de verdad del esquema canónico.
- [shared/gemini_client.py](shared/gemini_client.py): encapsula la llamada a Gemini con inicialización diferida.

## Notas de implementación

- El servicio de emociones intenta usar Gemini y, si no hay API key o la llamada falla, responde con un clasificador local determinista.
- El módulo de métricas usa `influenceScore` como proxy principal de engagement y complementa con `liked` cuando aporta información útil.
- El servicio de propagación usa `parentId` para reconstruir respuestas directas al post padre.
- Los entrypoints de los servicios ajustan `sys.path` para que los imports del paquete raíz funcionen al ejecutar archivos directamente.

## Verificación realizada

- El EDA se ejecutó correctamente y generó el reporte.
- El mapa de columnas fue construido a partir del parquet real.
- Los tres servicios MCP arrancan con FastMCP sobre HTTP.
- Las pruebas de integración pasan sobre los servicios activos.

## Solución de problemas

### No hay clave de Gemini

Si `GOOGLE_API_KEY` no está definida, el servicio de emociones seguirá funcionando con el fallback local.

### El agente no encuentra los servicios

Revisa que los tres procesos estén ejecutándose y que las variables `MCP_EMOCIONES_URL`, `MCP_METRICAS_URL` y `MCP_PROPAGACION_URL` apunten a los puertos correctos.

### No se puede leer el parquet

Verifica que `DATASET_PATH` apunte al archivo correcto y que el dataset exista en `data/`.

## Flujo recomendado de uso

1. Ejecuta el EDA y revisa el reporte.
2. Inicia los tres servicios MCP.
3. Lanza el chat del agente.
4. Ejecuta las pruebas de integración.

## Créditos

Este repositorio implementa el plan maestro del reto ICESI para análisis de conversaciones digitales con agentes conversacionales y servicios MCP.