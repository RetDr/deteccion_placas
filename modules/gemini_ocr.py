import os
import json
import base64
import urllib.request
import urllib.error
import cv2
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_FILE = os.path.join(BASE_DIR, ".gemini_key")

_CACHED_KEY = None

PATRON_CARRO = re.compile(r'^[A-Z]{3}[0-9]{3}$')
PATRON_MOTO = re.compile(r'^[A-Z]{3}[0-9]{2}[A-Z]$')


def obtener_gemini_key() -> str | None:
    global _CACHED_KEY
    if _CACHED_KEY:
        return _CACHED_KEY

    # 1. Variables de entorno
    for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        k = os.environ.get(var, "").strip()
        if k:
            _CACHED_KEY = k
            return _CACHED_KEY

    # 2. Archivo local .gemini_key
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r", encoding="utf-8") as f:
                k = f.read().strip()
                if k:
                    _CACHED_KEY = k
                    return _CACHED_KEY
        except Exception:
            pass

    return None


def guardar_gemini_key(api_key: str) -> bool:
    global _CACHED_KEY
    clean_key = (api_key or "").strip().rstrip('.')
    _CACHED_KEY = clean_key if clean_key else None
    if clean_key:
        os.environ["GEMINI_API_KEY"] = clean_key
    else:
        os.environ.pop("GEMINI_API_KEY", None)

    try:
        if clean_key:
            with open(KEY_FILE, "w", encoding="utf-8") as f:
                f.write(clean_key)
        else:
            if os.path.exists(KEY_FILE):
                os.remove(KEY_FILE)
    except Exception:
        pass

    return True


def gemini_disponible() -> bool:
    return bool(obtener_gemini_key())


def consultar_gemini_placa(crop_bgr: np.ndarray) -> dict | None:
    """
    Envía el recorte de la placa a la API de Gemini 1.5 Flash para un reconocimiento de máxima precisión.
    Utiliza HTTP POST directo (urllib) sin librerías externas adicionales.
    Retorna un diccionario con formato idéntico al pipeline de visión, o None si falla.
    """
    api_key = obtener_gemini_key()
    if not api_key:
        return None

    if crop_bgr is None or crop_bgr.size < 400:
        return None

    # Codificar imagen a JPEG en memoria
    ret, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        return None

    b64_img = base64.b64encode(buf).decode("utf-8")

    prompt = (
        "Analiza esta imagen que contiene una placa vehicular de Colombia. "
        "Devuelve estrictamente un objeto JSON con las siguientes claves:\n"
        "- 'placa': exactamente 6 caracteres alfanumericos en mayusculas sin espacios ni simbolos "
        "(ejemplo: 'ABC123' para automovil o 'ABC12D' para moto). Si la imagen no contiene una placa valida o legible, pon null.\n"
        "- 'tipo': 'CARRO' o 'MOTO'.\n"
        "- 'municipio': nombre oficial de la ciudad de transito que aparece en la parte inferior "
        "(ejemplo: 'BOGOTA D.C.', 'MEDELLIN', 'CALI', 'COLOMBIA'). Si no es legible, pon null.\n"
        "No incluyas explicaciones ni formato markdown, unicamente el objeto JSON."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_img
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            if resp.status != 200:
                return None
            res_json = json.loads(resp.read().decode("utf-8"))

            candidates = res_json.get("candidates", [])
            if not candidates:
                return None

            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts:
                return None

            raw_reply = content_parts[0].get("text", "").strip()
            # Limpiar posibles bloques markdown accidentales
            raw_reply = re.sub(r'^```json\s*', '', raw_reply)
            raw_reply = re.sub(r'^```\s*', '', raw_reply)
            raw_reply = re.sub(r'\s*```$', '', raw_reply).strip()

            parsed = json.loads(raw_reply)
            if not isinstance(parsed, dict):
                return None

            placa = (parsed.get("placa") or "").strip().upper()
            placa = re.sub(r'[^A-Z0-9]', '', placa)

            if len(placa) != 6:
                return None

            tipo = (parsed.get("tipo") or "CARRO").upper()
            if not PATRON_CARRO.match(placa) and not PATRON_MOTO.match(placa):
                return None

            if PATRON_MOTO.match(placa):
                tipo = "MOTO"
            elif PATRON_CARRO.match(placa):
                tipo = "CARRO"

            municipio = parsed.get("municipio")
            if municipio:
                municipio = municipio.strip().upper()
            else:
                municipio = "COLOMBIA" if tipo == "MOTO" else "COLOMBIA"

            print(f"[GEMINI OCR] Exito identificando: {placa} ({tipo} | {municipio})")

            return {
                "placa_raw": placa,
                "confianza": 0.985,
                "tipo": tipo,
                "municipio": municipio,
                "recorte_img": crop_bgr
            }

    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"[GEMINI OCR] Clave restringida por politica de Google Cloud (HTTP 403). Usando EasyOCR local de alta precision.")
        else:
            print(f"[GEMINI OCR] Error HTTP {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"[GEMINI OCR] Excepcion en llamada: {e}")
        return None
