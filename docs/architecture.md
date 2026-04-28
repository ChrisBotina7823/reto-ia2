# Arquitectura del sistema — Reto ICESI

El sistema está compuesto por tres microservicios MCP independientes y un agente conversacional orquestado con LangGraph. Cada servicio expone herramientas analíticas que el agente consume bajo demanda según la intención del usuario. El dataset fuente es un archivo Parquet que se carga una sola vez en memoria y se comparte entre los servicios a través del módulo `shared/`.

---

## Arquitectura general

```mermaid
flowchart LR
    U([Usuario\nCLI]) --> A

    subgraph Agente["Agente conversacional"]
        A[chat.py] --> G[LangGraph\nagent_node]
        G -->|tool_calls| T[ToolNode]
        T -->|FastMCP Client| G
    end

    subgraph MCP1["MCP Emociones :8001"]
        E1[analizar_emociones]
        E2[analizar_emociones_por_id]
    end

    subgraph MCP2["MCP Métricas :8002"]
        M1[actores_influyentes]
        M2[post_mayor_impacto]
        M3[metricas_generales]
    end

    subgraph MCP3["MCP Propagación :8003"]
        P1[analizar_propagacion]
    end

    T -->|HTTP /mcp| MCP1
    T -->|HTTP /mcp| MCP2
    T -->|HTTP /mcp| MCP3

    MCP1 -->|prompt| GEM[(Google\nGemini API)]
    MCP1 & MCP2 & MCP3 --> DS[(Parquet\ndataset)]
```

---

## Flujo del agente conversacional

```mermaid
flowchart TB
    Start([Mensaje del usuario]) --> H[Añadir HumanMessage\nal historial]
    H --> KEY{¿GOOGLE_API_KEY\nconfigurada?}

    KEY -- No --> FB[Modo degradado:\nenrutamiento por palabras clave]
    FB --> FTOOL[Invocación directa\nde herramienta MCP]
    FTOOL --> FOUT([Respuesta formateada\nen consola])

    KEY -- Sí --> AN[agent_node\nGemini LLM + 6 tools]
    AN --> TC{¿tool_calls\nen respuesta?}
    TC -- Sí --> TN[ToolNode ejecuta\nla herramienta]
    TN --> TMR[ToolMessage con\nJSON del MCP]
    TMR --> AN
    TC -- No --> OUT([Respuesta natural\nen consola])
```

---

## Flujo de datos

```mermaid
flowchart TB
    PQ[("Parquet\ndata/Reto_data_20251023_122206.parquet")] --> DL[data_loader.py\nlru_cache]
    DL --> CM[column_map.py\nrenombra columnas]
    CM --> DF[(DataFrame\nnormalizado)]

    DF --> SVC1[MCP Emociones\npandas → texto → Gemini]
    DF --> SVC2[MCP Métricas\npandas groupby/agg]
    DF --> SVC3[MCP Propagación\nfiltro por parentId]

    SVC1 --> R1["{emociones, emocion_dominante, confianza}"]
    SVC2 --> R2["{total_posts, usuarios, engagement_*}"]
    SVC3 --> R3["{alcance, usuarios_alcanzados, velocidad}"]
```

---

## Puertos y endpoints

| Servicio      | Puerto | Transporte     | Herramientas disponibles                                          |
|---------------|--------|----------------|-------------------------------------------------------------------|
| emociones     | 8001   | HTTP `/mcp`    | `analizar_emociones`, `analizar_emociones_por_id`                 |
| metricas      | 8002   | HTTP `/mcp`    | `actores_influyentes`, `post_mayor_impacto`, `metricas_generales` |
| propagacion   | 8003   | HTTP `/mcp`    | `analizar_propagacion`                                            |

Los tres servicios se inician con `python services/<nombre>/main.py` y exponen el transporte FastMCP HTTP en `http://127.0.0.1:<puerto>/mcp`. El agente los consume a través del cliente `fastmcp.Client` configurado con las variables de entorno `MCP_EMOCIONES_URL`, `MCP_METRICAS_URL` y `MCP_PROPAGACION_URL`.
