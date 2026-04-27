# Rellenado a partir de eda/reporte_eda.txt y la inspeccion del parquet.

COLUMN_MAP = {
    "post_id": "id",
    "user_id": "author",
    "text": "text",
    "timestamp": "createdAt",
    "likes": "liked",
    "replies": None,
    "reply_to": "parentId",
}

PROPAGACION_STRATEGY = "reply_col"