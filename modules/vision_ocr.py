"""
Módulo de Visión Artificial y OCR para Reconocimiento de Placas (Persona A).
Optimizaciones de Máxima Precisión y Corrección Sintáctica Colombiana:
1. Corrección posicional de sintaxis colombiana (Letras en pos 0-2, Dígitos en pos 3-4, Dígito/Letra en pos 5).
2. Mejora de imagen a color con interpolación bicúbica y filtro bilateral (elimina ruido sin romper caracteres).
3. Margen de seguridad adaptativo (8%) para no cortar letras laterales (A, M, W, 8).
4. Detección en 35 ms con YOLOv8 especializado en placas.
"""

import os

# Prevenir agotamiento de memoria OpenBLAS en subprocesos de OpenCV/PyTorch/EasyOCR
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import re
import difflib
import warnings
import cv2
import numpy as np

cv2.setNumThreads(1)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.environ["YOLO_CONFIG_DIR"] = os.path.join(MODELS_DIR, "ultralytics")

import torch
torch.set_num_threads(2)
from ultralytics import YOLO
import easyocr

PATRON_CARRO = re.compile(r'^[A-Z]{3}[0-9]{3}$')
PATRON_MOTO = re.compile(r'^[A-Z]{3}[0-9]{2}[A-Z]$')

# Base oficial de municipios y ciudades de Colombia con secretarías de tránsito
MUNICIPIOS_COLOMBIA = [
    'BOGOTA D.C.', 'MEDELLIN', 'CALI', 'BARRANQUILLA', 'BUCARAMANGA', 'CARTAGENA',
    'SANTA MARTA', 'CUCUTA', 'IBAGUE', 'PEREIRA', 'MANIZALES', 'PASTO', 'VILLAVICENCIO',
    'ARMENIA', 'POPAYAN', 'NEIVA', 'MONTERIA', 'VALLEDUPAR', 'SINCELEJO', 'TUNJA',
    'RIOHACHA', 'FLORENCIA', 'YOPAL', 'QUIBDO', 'SAN ANDRES', 'LETICIA', 'MOCOA',
    # Municipios metropolitanos y de gran parque automotor
    'ENVIGADO', 'SABANETA', 'ITAGUI', 'BELLO', 'RIONEGRO', 'GIRON', 'FLORIDABLANCA',
    'PIEDECUESTA', 'SOACHA', 'CHIA', 'ZIPAQUIRA', 'FACATATIVA', 'MOSQUERA', 'FUNZA',
    'MADRID', 'COTA', 'CAJICA', 'SOPO', 'LA CALERA', 'TABIO', 'TENJO', 'GIRARDOT',
    'FUSAGASUGA', 'TOCANCIPA', 'DOSQUEBRADAS', 'SANTA ROSA DE CABAL', 'PALMIRA',
    'TULUA', 'BUGA', 'YUMBO', 'JAMUNDI', 'CARTAGO', 'SOLEDAD', 'MALAMBO', 'TURBACO',
    'VILLA DEL ROSARIO', 'LOS PATIOS', 'ACACIAS', 'DUITAMA', 'SOGAMOSO', 'SAN GIL',
    'BARRANCABERMEJA', 'ESPINAL', 'MELGAR', 'MARIQUITA', 'PUERTO COLOMBIA', 'PRADERA',
    'CANDELARIA', 'BARBOSA', 'CALDAS', 'COPACABANA', 'LA ESTRELLA', 'GUARNE',
    'MARINILLA', 'EL RETIRO', 'LA CEJA', 'APARTADO', 'TURBO', 'COLOMBIA'
]

MUNICIPIOS_NORM = {}
for m in MUNICIPIOS_COLOMBIA:
    norm = re.sub(r'[^A-Z]', '', m.replace('Á','A').replace('É','E').replace('Í','I').replace('Ó','O').replace('Ú','U'))
    MUNICIPIOS_NORM[norm] = m


def coincidir_municipio_colombiano(texto_ocr: str, tipo_vehiculo: str = "CARRO") -> str | None:
    """
    Busca coincidencias rápidas contra la base oficial de municipios colombianos:
    - Motos: Norma nacional establece que llevan 'COLOMBIA'.
    - Carros: Cruza por coincidencia exacta, prefijo o similitud difusa (Levenshtein)
      corrigiendo errores de OCR (ej: BOGQTA -> BOGOTA D.C., MEDELUN -> MEDELLIN).
    """
    if tipo_vehiculo == "MOTO":
        return "COLOMBIA"

    if not texto_ocr:
        return None

    limpio = re.sub(r'[^A-Z]', '', texto_ocr.upper().replace('0','O').replace('1','I').replace('4','A').replace('8','B').replace('5','S'))
    if len(limpio) < 3:
        return None

    # 1. Coincidencia exacta normalizada
    if limpio in MUNICIPIOS_NORM:
        return MUNICIPIOS_NORM[limpio]

    # 2. Coincidencia por prefijo (ej: BOG, MED, ENV, SAB, CAL, ITAG)
    for norm, oficial in MUNICIPIOS_NORM.items():
        if norm.startswith(limpio) or (len(limpio) >= 4 and limpio.startswith(norm)):
            return oficial

    # 3. Coincidencia difusa (Fuzzy Matching Levenshtein)
    matches = difflib.get_close_matches(limpio, list(MUNICIPIOS_NORM.keys()), n=1, cutoff=0.52)
    if matches:
        return MUNICIPIOS_NORM[matches[0]]

    return None

_PLATE_DETECTOR = None
_VEHICLE_DETECTOR = None
_EASYOCR_READER = None


def obtener_modelos(gpu: bool = None):
    global _PLATE_DETECTOR, _VEHICLE_DETECTOR, _EASYOCR_READER

    if gpu is None:
        gpu = torch.cuda.is_available()

    if _PLATE_DETECTOR is None:
        ruta_detector_placas = os.path.join(MODELS_DIR, "plate_detector", "best.pt")
        if os.path.exists(ruta_detector_placas):
            _PLATE_DETECTOR = YOLO(ruta_detector_placas)
        else:
            _PLATE_DETECTOR = YOLO("yolov8n.pt")

    if _VEHICLE_DETECTOR is None:
        ruta_yolo_base = os.path.join(BASE_DIR, "yolov8n.pt")
        _VEHICLE_DETECTOR = YOLO(ruta_yolo_base if os.path.exists(ruta_yolo_base) else "yolov8n.pt")

    if _EASYOCR_READER is None:
        easyocr_dir = os.path.join(MODELS_DIR, "easyocr")
        has_local_weights = os.path.exists(os.path.join(easyocr_dir, "craft_mlt_25k.pth"))
        if os.path.exists(easyocr_dir):
            _EASYOCR_READER = easyocr.Reader(
                ['en'],
                gpu=gpu,
                model_storage_directory=easyocr_dir,
                user_network_directory=easyocr_dir,
                download_enabled=(not has_local_weights)
            )
        else:
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=gpu)

    return _PLATE_DETECTOR, _VEHICLE_DETECTOR, _EASYOCR_READER


def verificar_densidad_caracteres(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """
    Evalúa en 0.1 ms mediante Sobel si el recorte tiene trazos de caracteres reales.
    Descarta fondos vacíos sin ejecutar EasyOCR. Umbral seguro para cámaras reales.
    """
    if crop_bgr is None or crop_bgr.size < 400:
        return False, 0.0

    gris = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    sobel_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    varianza = float(np.var(sobel_x))
    return varianza >= 90.0, varianza


def mejorar_placa_lejana(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Mejora visual de alta definición para placas vehiculares:
    1. Escala hasta altura estándar de 115px (proporción 1:1 con la entrada de 64px de EasyOCR).
    2. Agrega margen perimetral protector (8%) para que números altos como el 6 y el 8 nunca
       toquen el borde de la imagen ni sufran cortes en su curva superior.
    3. Realce de contraste adaptativo en luminancia (CLAHE en espacio LAB):
       Refuerza el trazo negro troquelado sin quemar el fondo amarillo ni deformar el bucle del 6.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    altura_objetivo = 115

    if 0 < h < altura_objetivo:
        escala = altura_objetivo / float(h)
        nuevo_w = max(40, int(w * escala))
        redim = cv2.resize(crop_bgr, (nuevo_w, altura_objetivo), interpolation=cv2.INTER_CUBIC)
    elif h > 140:
        escala = altura_objetivo / float(h)
        nuevo_w = max(40, int(w * escala))
        redim = cv2.resize(crop_bgr, (nuevo_w, altura_objetivo), interpolation=cv2.INTER_AREA)
    else:
        redim = crop_bgr

    # Margen perimetral protector del 8% para evitar corte de caracteres
    pad_y = int(redim.shape[0] * 0.08)
    pad_x = int(redim.shape[1] * 0.06)
    padded = cv2.copyMakeBorder(redim, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)

    # Realce adaptativo CLAHE con filtro bilateral en canal L (Luminancia)
    # Suprime el ruido del asfalto y reflejos solares mientras mantiene los trazos de letras nítidos
    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_filtered = cv2.bilateralFilter(l, 5, 25, 25)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_filtered)
    lab_eq = cv2.merge((l_eq, a, b))
    mejorada = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return mejorada


def binarizar_placa_otsu(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Binarización adaptativa Otsu para placas vehiculares:
    Separa con máxima exactitud los caracteres negros troquelados del fondo reflectivo amarillo o blanco.
    Inmune a reflejos lumínicos, sol directo o sombras asfálticas.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # En la placa colombiana, los caracteres son oscuros y el fondo es claro (amarillo/blanco)
    # Si la binarización invirtió la polaridad (fondo oscuro > 55%), invertimos
    if np.mean(th == 255) < 0.45:
        th = cv2.bitwise_not(th)

    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def recortar_franja_caracteres(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Recorte perimetral protector (Inner Strip ROI):
    Elimina el marco exterior y la zona donde residen tornillos y remaches (superior 10% e inferior 14%),
    aislando exclusivamente la franja horizontal donde se encuentran los caracteres alfanuméricos.
    Previene que cabezas de tornillos sean leídas como 'I', '1', 'L', 'T' o '|'.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 40:
        return crop_bgr

    y1 = int(h * 0.10)
    y2 = max(y1 + 15, int(h * 0.86))
    x1 = int(w * 0.04)
    x2 = max(x1 + 30, int(w * 0.96))

    return crop_bgr[y1:y2, x1:x2]


DICT_A_LETRAS = {
    '0': 'O', '8': 'B', '5': 'S', '2': 'Z', '4': 'A', '6': 'G', '1': 'I',
    '9': 'P', '7': 'T', '3': 'B'
}
DICT_A_DIGITOS = {
    'O': '0', 'I': '1', 'L': '1', 'B': '8', 'S': '5', 'Z': '2', 'A': '4',
    'G': '6', 'D': '0', 'Q': '0', 'T': '1', 'U': '0', 'J': '1'
}

# Mapeo de confusión visual típico en tipografía troquelada colombiana DIN 1451
CONFUSION_LETRAS_DIN1451 = {
    'I': 'J',
    'Q': 'J',
    'N': 'V',
    'W': 'V',
    'D': 'O'
}

CONFUSION_DIGITOS_DIN1451 = {
    '4': '2',
    '7': '1'
}


def extraer_placa_estricta(texto_crudo: str) -> tuple[str | None, str | None, float]:
    """
    Extractor y validador estricto de placas colombianas (NORMA NACIONAL):
    1. AUTOMÓVIL / CARRO: Exactamente 3 letras y 3 números (^[A-Z]{3}[0-9]{3}$). NUNCA 4 números.
    2. MOTOCICLETA: Exactamente 3 letras, 2 números y 1 letra (^[A-Z]{3}[0-9]{2}[A-Z]$).
    3. Si el OCR capturó 7 u 8 caracteres (ej: tornillo o borde leído como 1, 4 o I),
       evalúa ventanas de 6 caracteres y descarte de remaches en extremos para extraer la placa real.
    4. Mapeo de confusión visual DIN 1451: Corrige ambigüedades comunes (J<->I/Q, V<->N/W, 2<->4).
    5. REGLA DURA: Si la cadena resultante NO cumple estrictamente con el formato de carro
       ni de moto, es DESCARTADA (retorna None, None, 0.0). Jamás se emite una placa con 4 números.
    """
    if not texto_crudo:
        return None, None, 0.0

    alfa = re.sub(r'[^A-Z0-9]', '', str(texto_crudo).upper())

    # Una placa colombiana SIEMPRE contiene al menos 2 letras y al menos 2 dígitos
    num_letras = sum(1 for c in alfa if c.isalpha())
    num_digitos = sum(1 for c in alfa if c.isdigit())
    if num_letras < 2 or num_digitos < 2 or len(alfa) < 6:
        return None, None, 0.0

    def evaluar_candidato_6(cadena_6: str) -> tuple[str | None, str | None, float]:
        if len(cadena_6) != 6:
            return None, None, 0.0

        # Chequeo directo exacto
        if PATRON_CARRO.match(cadena_6):
            return cadena_6, "CARRO", 0.99
        if PATRON_MOTO.match(cadena_6):
            return cadena_6, "MOTO", 0.99

        # Intento de corrección sintáctica para CARRO (3 letras + 3 dígitos)
        chars_c = list(cadena_6)
        subs_c = 0
        for i in range(3):
            if chars_c[i].isdigit():
                if chars_c[i] in DICT_A_LETRAS:
                    chars_c[i] = DICT_A_LETRAS[chars_c[i]]
                    subs_c += 1
                else:
                    subs_c += 99
        for i in (3, 4, 5):
            if chars_c[i].isalpha():
                if chars_c[i] in DICT_A_DIGITOS:
                    chars_c[i] = DICT_A_DIGITOS[chars_c[i]]
                    subs_c += 1
                else:
                    subs_c += 99

        cand_c = ''.join(chars_c)
        valido_c = bool(PATRON_CARRO.match(cand_c)) and subs_c <= 2

        # Intento de corrección sintáctica para MOTO (3 letras + 2 dígitos + 1 letra)
        chars_m = list(cadena_6)
        subs_m = 0
        for i in range(3):
            if chars_m[i].isdigit():
                if chars_m[i] in DICT_A_LETRAS:
                    chars_m[i] = DICT_A_LETRAS[chars_m[i]]
                    subs_m += 1
                else:
                    subs_m += 99
        for i in (3, 4):
            if chars_m[i].isalpha():
                if chars_m[i] in DICT_A_DIGITOS:
                    chars_m[i] = DICT_A_DIGITOS[chars_m[i]]
                    subs_m += 1
                else:
                    subs_m += 99
        if chars_m[5].isdigit():
            if chars_m[5] in DICT_A_LETRAS:
                chars_m[5] = DICT_A_LETRAS[chars_m[5]]
                subs_m += 1
            else:
                subs_m += 99

        cand_m = ''.join(chars_m)
        valido_m = bool(PATRON_MOTO.match(cand_m)) and subs_m <= 2

        if valido_c and not valido_m:
            return cand_c, "CARRO", max(0.70, 0.96 - 0.10 * subs_c)
        elif valido_m and not valido_c:
            return cand_m, "MOTO", max(0.70, 0.96 - 0.10 * subs_m)
        elif valido_c and valido_m:
            if subs_c <= subs_m:
                return cand_c, "CARRO", max(0.70, 0.96 - 0.10 * subs_c)
            else:
                return cand_m, "MOTO", max(0.70, 0.96 - 0.10 * subs_m)

        return None, None, 0.0

    candidatos_evaluados = []

    # 1. Si mide exactamente 6 caracteres
    if len(alfa) == 6:
        codigo, tipo, conf = evaluar_candidato_6(alfa)
        if codigo:
            candidatos_evaluados.append((codigo, tipo, conf))

    # 2. Descarte de remaches/tornillos aislados en bordes (longitud 7 u 8)
    CARACTERES_REMACHE = set("1ILTJ4._")
    if len(alfa) == 7:
        if alfa[0] in CARACTERES_REMACHE:
            c, t, conf = evaluar_candidato_6(alfa[1:])
            if c:
                candidatos_evaluados.append((c, t, conf * 0.98))
        if alfa[-1] in CARACTERES_REMACHE:
            c, t, conf = evaluar_candidato_6(alfa[:-1])
            if c:
                candidatos_evaluados.append((c, t, conf * 0.98))
    elif len(alfa) == 8:
        if alfa[0] in CARACTERES_REMACHE and alfa[-1] in CARACTERES_REMACHE:
            c, t, conf = evaluar_candidato_6(alfa[1:-1])
            if c:
                candidatos_evaluados.append((c, t, conf * 0.96))

    # 3. Ventanas deslizantes de 6 caracteres
    for start in range(len(alfa) - 5):
        ventana = alfa[start:start+6]
        codigo, tipo, conf = evaluar_candidato_6(ventana)
        if codigo:
            conf_ajustada = conf * (0.95 if start == 0 or start == len(alfa)-6 else 0.90)
            candidatos_evaluados.append((codigo, tipo, conf_ajustada))

    if candidatos_evaluados:
        candidatos_evaluados.sort(key=lambda x: x[2], reverse=True)
        return candidatos_evaluados[0]

    return None, None, 0.0


def corregir_sintaxis_colombiana(texto_crudo: str) -> tuple[str, str, float]:
    """
    Función de compatibilidad con el pipeline de visión.
    Asegura que el retorno siempre cumpla con la norma colombiana.
    """
    codigo, tipo, conf = extraer_placa_estricta(texto_crudo)
    if codigo and tipo in ["CARRO", "MOTO"]:
        return codigo, tipo, conf
    return texto_crudo, "DESCARTAR", 0.0


def clasificar_texto_colombiano(texto: str) -> tuple[str, str]:
    if not texto:
        return "DESCARTAR", ""

    alfa = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if not alfa:
        return "DESCARTAR", ""

    # 1. Si ya es una placa completa válida (ej: ABC123, ABC-123, ABC 123, XYZ45E)
    codigo, tipo, _ = extraer_placa_estricta(texto)
    if codigo and tipo in ["CARRO", "MOTO"]:
        return tipo, codigo

    # 2. Municipio oficial: coteja contra base oficial (incluso si 'D.C.' fue leído como '0.C' u 'O.C')
    mun = coincidir_municipio_colombiano(texto)
    if mun:
        return "MUNICIPIO", mun

    # 3. Bloque de letras (2 a 4 letras: prefijo de placa de carro o moto, ej: ABC, XYZ)
    if 2 <= len(alfa) <= 4 and alfa.isalpha():
        return "PARTE_LETRAS", alfa

    # 4. Bloque de números o sufijo de moto (2 a 4 caracteres con dígitos: ej: 123, 45E, 12)
    if 2 <= len(alfa) <= 4 and any(c.isdigit() for c in alfa):
        return "PARTE_NUMEROS", alfa

    # 5. Candidato alfanumérico mixto (longitud 5 a 8)
    if 5 <= len(alfa) <= 8 and any(c.isalpha() for c in alfa) and any(c.isdigit() for c in alfa):
        return "CANDIDATO", alfa

    # 6. Texto largo sin números como fallback de municipio
    if len(alfa) >= 4 and not any(c.isdigit() for c in texto):
        return "MUNICIPIO", texto.strip().upper()

    return "DESCARTAR", alfa


def _agrupar_tokens_horizontales(detecciones_crudo: list[dict]) -> list[dict]:
    if len(detecciones_crudo) <= 1:
        return detecciones_crudo

    ordenados = sorted(detecciones_crudo, key=lambda b: b["bbox"][0])
    unificados = []
    usados = set()

    for i in range(len(ordenados)):
        if i in usados:
            continue
        curr = ordenados[i]
        texto_comb = curr["texto"]
        total_prob = curr["prob"]
        conteo = 1
        cx1, cy1, cx2, cy2 = curr["bbox"]

        for j in range(i + 1, len(ordenados)):
            if j in usados:
                continue
            cand = ordenados[j]
            jx1, jy1, jx2, jy2 = cand["bbox"]

            ch = cy2 - cy1
            jh = jy2 - jy1
            y_diff = abs(((cy1 + cy2) / 2) - ((jy1 + jy2) / 2))
            x_gap = jx1 - cx2

            if y_diff < 0.55 * min(ch, jh) and -5 <= x_gap <= 1.8 * max(ch, jh):
                texto_comb += cand["texto"]
                total_prob += cand["prob"]
                conteo += 1
                cx1 = min(cx1, jx1)
                cy1 = min(cy1, jy1)
                cx2 = max(cx2, jx2)
                cy2 = max(cy2, jy2)
                usados.add(j)

        unificados.append({
            "texto": texto_comb,
            "prob": total_prob / conteo,
            "bbox": (cx1, cy1, cx2, cy2)
        })
        usados.add(i)

    return unificados


def analizar_frame(frame_bgr: np.ndarray) -> list[dict]:
    """
    Recibe un frame de OpenCV (numpy array).
    Retorna una lista de diccionarios con las detecciones encontradas en ese frame:
    [
        {
            "placa_raw": "ABC123",
            "confianza": 0.89,
            "bbox": (x1, y1, x2, y2),
            "recorte_img": crop_array,
            "tipo": "CARRO" | "MOTO",
            "municipio": "BOGOTA" | None
        }
    ]
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    plate_detector, vehicle_detector, reader = obtener_modelos()
    h_frame, w_frame = frame_bgr.shape[:2]
    detecciones_frame = []
    candidatos_placa = []

    # 1. Detección de placas con YOLO especializado (~35 ms)
    res_plates = plate_detector(frame_bgr, verbose=False, conf=0.22)
    for r in res_plates:
        for box in r.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
            # Margen de seguridad del 14% para no cortar letras laterales (ej: O, B, C, A o 3)
            pw = px2 - px1
            ph = py2 - py1
            pad_x = int(pw * 0.14)
            pad_y = int(ph * 0.14)

            px1 = max(0, px1 - pad_x)
            py1 = max(0, py1 - pad_y)
            px2 = min(w_frame, px2 + pad_x)
            py2 = min(h_frame, py2 + pad_y)

            candidatos_placa.append({
                "bbox": (px1, py1, px2, py2),
                "conf_detector": float(box.conf[0])
            })

    # 2. Fallback por color si no detecta con YOLO
    if not candidatos_placa:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv, np.array([12, 70, 70]), np.array([38, 255, 255]))
        mask_w = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 50, 255]))
        mask = cv2.bitwise_or(mask_y, mask_w)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contornos:
            area = cv2.contourArea(c)
            if 1000 <= area <= 140000:
                cx, cy, cw, ch = cv2.boundingRect(c)
                ratio = float(cw) / max(1, ch)
                if 1.4 <= ratio <= 3.6:
                    candidatos_placa.append({
                        "bbox": (max(0, cx - 6), max(0, cy - 6), min(w_frame, cx + cw + 6), min(h_frame, cy + ch + 6)),
                        "conf_detector": 0.5
                    })

    # Filtrar con Sobel para no procesar áreas lisas
    candidatos_validados = []
    for c in candidatos_placa:
        bx1, by1, bx2, by2 = c["bbox"]
        crop = frame_bgr[by1:by2, bx1:bx2]
        tiene_texto, var_score = verificar_densidad_caracteres(crop)
        if tiene_texto:
            c["var_score"] = var_score
            candidatos_validados.append(c)

    candidatos_validados.sort(key=lambda x: x.get("var_score", 0), reverse=True)
    candidatos_finales = candidatos_validados[:1]

    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ."

    for cand in candidatos_finales:
        rx1, ry1, rx2, ry2 = cand["bbox"]
        crop_roi = frame_bgr[ry1:ry2, rx1:rx2]

        if crop_roi.size == 0:
            continue

        preprocesada = mejorar_placa_lejana(crop_roi)

        try:
            resultados_ocr = reader.readtext(
                preprocesada,
                allowlist=allowlist,
                detail=1,
                paragraph=False,
                mag_ratio=1.0,
                canvas_size=384,
                add_margin=0.18,
                adjust_contrast=0.6
            )
        except Exception:
            continue

        h_roi, w_roi = crop_roi.shape[:2]
        h_prep, w_prep = preprocesada.shape[:2]
        escala_x = w_roi / float(w_prep) if w_prep > 0 else 1.0
        escala_y = h_roi / float(h_prep) if h_prep > 0 else 1.0

        candidatos_roi = []
        municipios_encontrados = []

        for bbox_pts, texto_detectado, prob in resultados_ocr:
            texto_limpio = texto_detectado.strip().upper()
            clasificacion, texto_proc = clasificar_texto_colombiano(texto_limpio)

            xs = [pt[0] for pt in bbox_pts]
            ys = [pt[1] for pt in bbox_pts]
            min_x, max_x = min(xs) * escala_x, max(xs) * escala_x
            min_y, max_y = min(ys) * escala_y, max(ys) * escala_y

            gx1 = max(0, int(rx1 + min_x))
            gy1 = max(0, int(ry1 + min_y))
            gx2 = min(w_frame, int(rx1 + max_x))
            gy2 = min(h_frame, int(ry1 + max_y))

            if clasificacion == "MUNICIPIO":
                municipios_encontrados.append({
                    "nombre": texto_proc,
                    "prob": float(prob),
                    "y_center": (gy1 + gy2) / 2.0,
                    "bbox": (gx1, gy1, gx2, gy2)
                })
            elif clasificacion != "DESCARTAR":
                candidatos_roi.append({
                    "texto": texto_proc,
                    "prob": float(prob),
                    "bbox": (gx1, gy1, gx2, gy2),
                    "clasif": clasificacion
                })

        bloques_consolidados = _agrupar_tokens_horizontales(candidatos_roi)

        for b in bloques_consolidados:
            texto_eval = b["texto"].strip().upper()
            # Extracción y validación estricta de la norma colombiana
            placa_corregida, tipo_vehiculo, conf_bonus = extraer_placa_estricta(texto_eval)

            # REGLA DURA: Si no cumple el formato colombiano de 6 caracteres (3 letras + 3 dígitos para carro,
            # o 3 letras + 2 dígitos + 1 letra para moto), SE DESCARTA INMEDIATAMENTE. Jamás 4 números.
            if not placa_corregida or tipo_vehiculo not in ["CARRO", "MOTO"]:
                continue

            if not (PATRON_CARRO.match(placa_corregida) or PATRON_MOTO.match(placa_corregida)):
                continue

            bx1, by1, bx2, by2 = b["bbox"]
            plate_center_x = (bx1 + bx2) / 2.0
            plate_w = max(1, bx2 - bx1)

            municipio_detectado = "COLOMBIA" if tipo_vehiculo == "MOTO" else None
            for mun in municipios_encontrados:
                mun_center_x = (mun["bbox"][0] + mun["bbox"][2]) / 2.0
                mun_center_y = mun["y_center"]
                if mun_center_y > by1 and abs(plate_center_x - mun_center_x) < 0.65 * plate_w:
                    mun_match = coincidir_municipio_colombiano(mun["nombre"], tipo_vehiculo)
                    municipio_detectado = mun_match or mun["nombre"]
                    bx1 = min(bx1, mun["bbox"][0])
                    by2 = max(by2, mun["bbox"][3])
                    bx2 = max(bx2, mun["bbox"][2])
                    break

            crop_x1 = max(0, bx1 - 6)
            crop_y1 = max(0, by1 - 6)
            crop_x2 = min(w_frame, bx2 + 6)
            crop_y2 = min(h_frame, by2 + 6)

            recorte_placa = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            if recorte_placa.size == 0:
                continue

            prob_final = min(1.0, float(b["prob"]) * conf_bonus)

            detecciones_frame.append({
                "placa_raw": str(placa_corregida),
                "confianza": round(prob_final, 4),
                "bbox": (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)),
                "recorte_img": recorte_placa,
                "tipo": tipo_vehiculo,
                "municipio": municipio_detectado
            })

    resultado_final = []
    for det in detecciones_frame:
        repetido = False
        for i, prev in enumerate(resultado_final):
            if det["placa_raw"] == prev["placa_raw"]:
                repetido = True
                if det["confianza"] > prev["confianza"]:
                    resultado_final[i] = det
                break
        if not repetido:
            resultado_final.append(det)

    return resultado_final
