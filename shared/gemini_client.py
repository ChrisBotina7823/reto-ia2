import os

_model = None


def _model_candidates() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    env_fallbacks = os.getenv("GEMINI_MODEL_FALLBACKS", "")
    default_fallbacks = ["gemini-1.5-flash", "gemini-1.5-pro"]

    candidates: list[str] = [primary]
    for model in [m.strip() for m in env_fallbacks.split(",") if m.strip()] + default_fallbacks:
        if model not in candidates:
            candidates.append(model)
    return candidates


def get_model():
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY no esta configurada")

    genai.configure(api_key=api_key)

    global _model
    if _model is None:
        candidates = _model_candidates()
        last_error = None
        for model_name in candidates:
            try:
                _model = genai.GenerativeModel(model_name=model_name)
                break
            except Exception as exc:
                last_error = exc
        if _model is None and last_error is not None:
            raise last_error
    return _model


def call_gemini(prompt: str, system: str = "") -> str:
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY no esta configurada")

    genai.configure(api_key=api_key)

    last_error = None
    for model_name in _model_candidates():
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system if system else None,
            )
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            message = str(exc)
            if "NOT_FOUND" in message or "is not found" in message or "404" in message:
                last_error = exc
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("No se pudo generar contenido con ningun modelo Gemini")