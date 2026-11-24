"""
Motor de captura y procesamiento en tiempo real multihilo (Producer-Consumer).
Integrado con el Rastreador IoU (Object Tracking):
1. Elimina duplicados: 1 vehículo en movimiento = 1 sola placa registrada en el historial.
2. Reduce la latencia a 35 ms: Cuando la placa ya está reconocida, el tracker la sigue
   usando detección geométrica rápida sin re-ejecutar OCR costoso en cada cuadro.
3. Transmisión continua fluida a 30 FPS en vivo.
"""

import os
import sys

# Prevenir agotamiento de memoria OpenBLAS en hilos de captura y tracking
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import base64
import queue
import threading
import cv2
import numpy as np

cv2.setNumThreads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.vision_ocr import (
    obtener_modelos,
    mejorar_placa_lejana,
    binarizar_placa_otsu,
    recortar_franja_caracteres,
    clasificar_texto_colombiano,
    extraer_placa_estricta,
    coincidir_municipio_colombiano,
    _agrupar_tokens_horizontales,
    verificar_densidad_caracteres
)
from modules.search_engine import consultar_placa
from modules.tracker import RealtimePlateTracker, calcular_iou, son_la_misma_placa
from modules.gemini_ocr import gemini_disponible, consultar_gemini_placa


def get_roi_coordinates(w_f: int, h_f: int) -> tuple[int, int, int, int]:
    """
    Calcula las coordenadas del Encuadre Óptico adaptativo.
    En horizontal (PC / webcam / celular apaisado): rectángulo centrado ~2.15:1.
    En vertical (celular en modo portrait / DroidCam vertical): rectángulo centrado ~1.95:1.
    Garantiza que la placa o el celular quepan de forma ergonómica y natural.
    """
    if w_f >= h_f:
        # Orientación horizontal
        roi_w = int(w_f * 0.62)
        roi_h = int(roi_w / 2.15)
        if roi_h > int(h_f * 0.72):
            roi_h = int(h_f * 0.72)
            roi_w = int(roi_h * 2.15)
    else:
        # Orientación vertical (celular en retrato 9:16)
        roi_w = int(w_f * 0.88)
        roi_h = int(roi_w / 1.95)
        if roi_h > int(h_f * 0.55):
            roi_h = int(h_f * 0.55)
            roi_w = int(roi_h * 1.95)

    rx1 = max(0, (w_f - roi_w) // 2)
    ry1 = max(0, (h_f - roi_h) // 2)
    rx2 = min(w_f, rx1 + roi_w)
    ry2 = min(h_f, ry1 + roi_h)
    return rx1, ry1, rx2, ry2


def detectar_placas_por_color(frame_bgr: np.ndarray) -> list[tuple]:
    """
    Descomposición cromática HSV para placas colombianas (Amarillo y Blanco).
    Calibrado para cámaras de celulares modernos con balance de blancos automático y HDR.
    Actúa como red de seguridad infalible cuando el modelo YOLO no detecta la placa.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    h_f, w_f = frame_bgr.shape[:2]
    cajas = []

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Rango extendido amarillo colombiano (5-42) y blanco reflectivo (0-180 con alta luminancia)
    mask_y = cv2.inRange(hsv, np.array([5, 30, 70]), np.array([42, 255, 255]))
    mask_w = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 55, 255]))
    # Celulares y pantallas LCD: blanco frío / cianico de retroiluminación
    mask_screen = cv2.inRange(hsv, np.array([80, 10, 160]), np.array([135, 75, 255]))
    mask = cv2.bitwise_or(mask_y, cv2.bitwise_or(mask_w, mask_screen))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        if 600 <= area <= int(h_f * w_f * 0.98):
            cx, cy, cw, ch = cv2.boundingRect(c)
            ratio = float(cw) / max(1, ch)
            if 0.75 <= ratio <= 6.5 and cw >= 35 and ch >= 14:
                crop = frame_bgr[cy:cy+ch, cx:cx+cw]
                tiene_texto, _ = verificar_densidad_caracteres(crop)
                # Si el área es grande (acercamiento de celular), aceptar incluso si la densidad varía
                if tiene_texto or area >= (h_f * w_f * 0.08):
                    pad_x = int(cw * 0.14)
                    pad_y = int(ch * 0.10)
                    gx1 = max(0, cx - pad_x)
                    gy1 = max(0, cy - pad_y)
                    gx2 = min(w_f, cx + cw + pad_x)
                    gy2 = min(h_f, cy + ch + pad_y)
                    cajas.append((gx1, gy1, gx2, gy2))

    return cajas


_CAMARAS_CACHE = None
_CAMARAS_CACHE_TIME = 0.0


def abrir_dispositivo_captura(source, quick_probe=False):
    """
    Abre una cámara física, virtual (DroidCam, Iriun, Camo) o stream de red.
    En Windows utiliza DirectShow (CAP_DSHOW) de forma prioritaria para evitar bloqueos
    y demoras de Media Foundation (MSMF).
    """
    is_num = str(source).isdigit() or isinstance(source, int)
    if is_num:
        idx = int(source)
        # En Windows DirectShow es instantáneo y estable; CAP_ANY es el fallback rápido
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith('win') else [cv2.CAP_ANY]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        if not quick_probe:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                        return cap
                    else:
                        cap.release()
            except Exception:
                pass
        return None
    else:
        src_str = str(source).strip()
        try:
            cap = cv2.VideoCapture(src_str)
            if cap and cap.isOpened():
                if src_str.startswith("http") or src_str.startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    return cap
                cap.release()
        except Exception:
            pass
        return None


def listar_camaras_disponibles(force_refresh=False) -> list[dict]:
    """
    Escanea índices de cámara de forma ultrarrápida (0 a 2) con DirectShow.
    Utiliza caché en memoria para responder en 0 ms en llamadas sucesivas.
    """
    global _CAMARAS_CACHE, _CAMARAS_CACHE_TIME
    now = time.time()
    if not force_refresh and _CAMARAS_CACHE is not None and (now - _CAMARAS_CACHE_TIME < 15.0):
        return _CAMARAS_CACHE

    camaras = []
    # Escanear índices 0, 1 y 2 (cubre webcam integrada, cámara USB/celular y secundaria)
    max_indices = 3
    for i in range(max_indices):
        backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
        cap = None
        try:
            cap = cv2.VideoCapture(i, backend)
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    label = f"Cámara {i} (Principal / Webcam - {w}x{h})" if i == 0 else f"Cámara {i} (USB / Celular - {w}x{h})"
                    camaras.append({
                        "id": str(i),
                        "label": label
                    })
        except Exception:
            pass
        finally:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass

    test_video = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "video_prueba.mp4")
    if os.path.exists(test_video):
        camaras.append({
            "id": "file",
            "label": "Video de Prueba Local (video_prueba.mp4)"
        })

    _CAMARAS_CACHE = camaras
    _CAMARAS_CACHE_TIME = now
    return camaras


def aplicar_nms_cajas(bboxes: list[tuple], iou_threshold: float = 0.20) -> list[tuple]:
    """
    Supresión de No Máximos (NMS): fusiona o elimina cajas duplicadas de la misma placa
    para que nunca aparezcan 2 o 3 recuadros verdes compitiendo en pantalla.
    """
    if len(bboxes) <= 1:
        return bboxes

    # Ordenar por área descendente
    boxes = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    seleccionadas = []

    for b in boxes:
        solapado = False
        for s in seleccionadas:
            if son_la_misma_placa(b, s) or calcular_iou(b, s) >= iou_threshold:
                solapado = True
                break
        if not solapado:
            seleccionadas.append(b)

    return seleccionadas


class LiveCameraEngine:
    def __init__(self, source=0, sampling_step=2, min_confidence=0.28):
        self.source = source
        self.sampling_step = sampling_step
        self.min_confidence = min_confidence

        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()

        # Encuadre Óptico (ROI): Aísla la zona activa para triplicar la densidad de píxeles
        self.use_roi = True

        # Rastreador en tiempo real (IoU + Centroides)
        self.tracker = RealtimePlateTracker(iou_threshold=0.20, max_invisible_frames=30, deduplication_window_sec=8.0)

        # Cola para inferencia OCR asíncrona desacoplada (mantiene tracking a 25-30 FPS)
        self.ocr_queue = queue.Queue(maxsize=1)
        self.pending_ocr_tracks = set()

        # Buffers
        self.latest_frame = None
        self.latest_detections = []

        # Telemetría en tiempo real
        self.camera_fps = 0.0
        self.ocr_fps = 0.0
        self.inference_latency_ms = 0.0
        self.total_frames_read = 0
        self.total_frames_analyzed = 0

        self.capture_thread = None
        self.tracking_thread = None
        self.ocr_worker_thread = None
        self.listeners = []

    def get_roi_coordinates(self, w_f: int, h_f: int) -> tuple[int, int, int, int]:
        return get_roi_coordinates(w_f, h_f)

    @property
    def history_plates(self):
        return self.tracker.confirmed_history

    def add_listener(self, callback):
        self.listeners.append(callback)

    def remove_listener(self, callback):
        if callback in self.listeners:
            self.listeners.remove(callback)

    def _notify_listeners(self, event_data):
        for listener in list(self.listeners):
            try:
                listener(event_data)
            except Exception:
                pass

    def start(self, source=None):
        if self.is_running:
            self.stop()

        if source is not None:
            self.source = source

        src = int(self.source) if str(self.source).isdigit() else self.source

        self.cap = abrir_dispositivo_captura(src)

        if not self.cap or not self.cap.isOpened():
            print(f"[ERROR] No se pudo inicializar la fuente de video: {self.source}")
            return False

        self.is_running = True

        # Limpiar cola de OCR y pistas pendientes de sesiones anteriores
        self.ocr_queue = queue.Queue(maxsize=1)
        self.pending_ocr_tracks = set()

        # 1. Hilo de captura de cámara a 30 FPS
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # 2. Hilo de tracking y detección YOLO a 25-30 FPS
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()

        # 3. Hilo trabajador de OCR asíncrono en segundo plano (no bloquea el video)
        self.ocr_worker_thread = threading.Thread(target=self._async_ocr_worker, daemon=True)
        self.ocr_worker_thread.start()

        return True

    def stop(self):
        self.is_running = False
        try:
            self.ocr_queue.put_nowait((None, None))
        except Exception:
            pass

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)
        if self.ocr_worker_thread and self.ocr_worker_thread.is_alive():
            self.ocr_worker_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def change_source(self, new_source):
        """
        Cambio dinámico en caliente sin reiniciar hilos de fondo ni romper la conexión MJPEG.
        Garantiza que la imagen nunca se congele o se dañe en el navegador.
        """
        src = int(new_source) if str(new_source).isdigit() else new_source
        prev_source = self.source

        with self.lock:
            old_cap = self.cap
            self.cap = None

        if old_cap:
            try:
                old_cap.release()
            except Exception:
                pass
        time.sleep(0.15)

        new_cap = abrir_dispositivo_captura(src)

        if new_cap and new_cap.isOpened():
            with self.lock:
                self.cap = new_cap
                self.source = new_source
                self.latest_frame = None
                self.tracker.active_tracks.clear()
                self.pending_ocr_tracks.clear()
            print(f"[CAMARA] Cambio exitoso a fuente: {new_source}")
            return True
        else:
            print(f"[ADVERTENCIA] No se pudo abrir la fuente solicitada: {new_source}. Restaurando respaldo...")
            # Intentar restaurar fuente previa o fallback a video local
            fallback_cap = abrir_dispositivo_captura(prev_source) if prev_source != new_source else None
            if not fallback_cap:
                test_vid = os.path.join(BASE_DIR, "data", "video_prueba.mp4")
                if os.path.exists(test_vid):
                    fallback_cap = abrir_dispositivo_captura(test_vid)
                    prev_source = "file"

            with self.lock:
                self.cap = fallback_cap
                self.source = prev_source
            return False

    def _capture_loop(self):
        last_time = time.time()
        fps_counter = 0

        while self.is_running:
            with self.lock:
                cap = self.cap
            if cap is None or not cap.isOpened():
                time.sleep(0.04)
                continue

            ret, frame = cap.read()
            if not ret:
                src_str = str(self.source)
                if not src_str.isdigit() and not src_str.startswith("http") and not src_str.startswith("rtsp") and cap:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.015)
                    continue

            with self.lock:
                self.latest_frame = frame
                self.total_frames_read += 1

            fps_counter += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.camera_fps = round(fps_counter / (now - last_time), 1)
                fps_counter = 0
                last_time = now

            time.sleep(0.003)

    def _ejecutar_ocr_en_recorte(self, crop_bgr: np.ndarray, reader, modo_profundo: bool = False) -> list[dict]:
        """
        Ejecuta OCR sobre el recorte con superresolución y coteja municipios de Colombia.
        En modo estándar (video en vivo): 2 pasadas rápidas (~250 ms).
        En modo profundo (al tomar foto): Batería completa de filtros de nitidez, contraste,
        Black-Hat morfológico de caracteres troquelados y umbrales adaptativos.
        """
        if crop_bgr is None or crop_bgr.size < 400:
            return []

        # 1. Modo Híbrido: Si el usuario configuró Gemini Flash, intentar usarlo
        if gemini_disponible():
            gemini_res = consultar_gemini_placa(crop_bgr)
            if gemini_res:
                return [gemini_res]

        allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- .@&"

        def ejecutar_easyocr(img_in, text_thresh=0.45, low_thresh=0.25):
            try:
                h_in, w_in = img_in.shape[:2]
                scale_x, scale_y = 1.0, 1.0
                # Si es un recorte ajustado de placa (h <= 220), normalizar a ~130 px de alto
                if 45 <= h_in <= 220:
                    if h_in > 150 or h_in < 95:
                        scale = 130.0 / float(h_in)
                        interp = cv2.INTER_AREA if h_in > 150 else cv2.INTER_CUBIC
                        new_w = max(40, int(w_in * scale))
                        img_to_feed = cv2.resize(img_in, (new_w, 130), interpolation=interp)
                        scale_x = new_w / float(w_in)
                        scale_y = 130.0 / float(h_in)
                    else:
                        img_to_feed = img_in
                elif h_in > 220:
                    # ES UN ENCUADRE ÓPTICO O PANTALLA DE CELULAR:
                    # No aplastar a 130 px; conservar resolución para que CRAFT y CRNN lean los caracteres nítidos
                    if max(h_in, w_in) > 720:
                        scale = 720.0 / float(max(h_in, w_in))
                        new_w = int(w_in * scale)
                        new_h = int(h_in * scale)
                        img_to_feed = cv2.resize(img_in, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        scale_x = new_w / float(w_in)
                        scale_y = new_h / float(h_in)
                    else:
                        img_to_feed = img_in
                else:
                    img_to_feed = img_in

                tokens = reader.readtext(
                    img_to_feed,
                    allowlist=allowlist,
                    detail=1,
                    paragraph=False,
                    mag_ratio=1.4,
                    text_threshold=text_thresh,
                    low_text=low_thresh,
                    add_margin=0.06
                )
                return tokens, scale_x, scale_y
            except Exception:
                return [], 1.0, 1.0

        def procesar_tokens_ocr(res_pack):
            if not res_pack:
                return None
            res_tokens, scale_x, scale_y = res_pack
            if not res_tokens:
                return None

            candidatos = []
            municipios = []
            todos_los_textos = []

            for bbox_pts, texto_detectado, prob in res_tokens:
                texto_limpio = texto_detectado.strip().upper()
                if not texto_limpio:
                    continue
                todos_los_textos.append((texto_limpio, float(prob)))
                clasificacion, texto_proc = clasificar_texto_colombiano(texto_limpio)

                # Mapear coordenadas de bbox_pts a crop_bgr
                xs = [pt[0] / scale_x for pt in bbox_pts]
                ys = [pt[1] / scale_y for pt in bbox_pts]
                token_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

                if clasificacion == "MUNICIPIO":
                    municipios.append(texto_proc)
                elif clasificacion != "DESCARTAR":
                    candidatos.append({
                        "texto": texto_proc,
                        "prob": float(prob),
                        "bbox": token_bbox,
                        "clasif": clasificacion
                    })

            bloques = _agrupar_tokens_horizontales(candidatos)
            cadenas_a_evaluar = []
            for b in bloques:
                cadenas_a_evaluar.append((b["texto"].strip().upper(), float(b["prob"])))
            for txt, p in todos_los_textos:
                cadenas_a_evaluar.append((txt, p))
            if len(candidatos) >= 2:
                cands_ordenados = sorted(candidatos, key=lambda c: c["bbox"][0])
                texto_unido = "".join(c["texto"] for c in cands_ordenados)
                prob_prom = sum(c["prob"] for c in cands_ordenados) / len(cands_ordenados)
                cadenas_a_evaluar.append((texto_unido, prob_prom))

            mejores_placas = []
            for texto_cand, prob_cand in cadenas_a_evaluar:
                placa_val, tipo_veh, conf_bonus = extraer_placa_estricta(texto_cand, crop_bgr=crop_bgr)
                if placa_val and tipo_veh in ["CARRO", "MOTO"]:
                    score_final = min(1.0, prob_cand * conf_bonus)
                    mejores_placas.append((placa_val, tipo_veh, score_final))

            if not mejores_placas:
                return None

            mejores_placas.sort(key=lambda x: x[2], reverse=True)
            mejor_placa, tipo_vehiculo, prob_final = mejores_placas[0]

            municipio_oficial = "COLOMBIA" if tipo_vehiculo == "MOTO" else None
            if municipios and tipo_vehiculo != "MOTO":
                municipio_oficial = coincidir_municipio_colombiano(municipios[0], tipo_vehiculo) or municipios[0]

            # Si el recorte original era amplio (pantalla de celular o encuadre), extraer sub_bbox ajustado
            plate_sub_bbox = None
            if candidatos:
                cands_en_placa = [c for c in candidatos if c["texto"] in mejor_placa]
                if not cands_en_placa:
                    cands_en_placa = candidatos
                if cands_en_placa:
                    min_x = max(0, min(c["bbox"][0] for c in cands_en_placa))
                    min_y = max(0, min(c["bbox"][1] for c in cands_en_placa))
                    max_x = min(w_orig, max(c["bbox"][2] for c in cands_en_placa))
                    max_y = min(h_orig, max(c["bbox"][3] for c in cands_en_placa))
                    pad_x = int((max_x - min_x) * 0.16)
                    pad_y = int((max_y - min_y) * 0.28)
                    plate_sub_bbox = (
                        max(0, min_x - pad_x),
                        max(0, min_y - pad_y),
                        min(w_orig, max_x + pad_x),
                        min(h_orig, max_y + pad_y)
                    )

            recorte_final = crop_bgr
            if plate_sub_bbox and (h_orig > 200 or w_orig > 380):
                bx1, by1, bx2, by2 = plate_sub_bbox
                if (bx2 - bx1) >= 40 and (by2 - by1) >= 18:
                    sub_crop = crop_bgr[by1:by2, bx1:bx2]
                    if sub_crop.size > 200:
                        recorte_final = sub_crop

            return {
                "placa_raw": str(mejor_placa),
                "confianza": round(prob_final, 4),
                "tipo": tipo_vehiculo,
                "municipio": municipio_oficial,
                "recorte_img": recorte_final,
                "sub_bbox": plate_sub_bbox
            }

        h_orig, w_orig = crop_bgr.shape[:2]
        # Solo recortar márgenes si ya es un recorte ajustado de placa (h <= 200 y proporción horizontal)
        if h_orig <= 200 and (w_orig / max(1, h_orig)) >= 1.2:
            inner_crop = crop_bgr[int(h_orig * 0.08):int(h_orig * 0.82), int(w_orig * 0.05):int(w_orig * 0.95)]
            prep_target = inner_crop if (inner_crop.size > 400 and inner_crop.shape[0] >= 20) else crop_bgr
        else:
            prep_target = crop_bgr

        # Atenuación de sobreexposición para pantallas de celular
        if prep_target.size > 0:
            gray_check = cv2.cvtColor(prep_target, cv2.COLOR_BGR2GRAY) if len(prep_target.shape) == 3 else prep_target
            if np.mean(gray_check > 235) > 0.20:
                prep_target = cv2.convertScaleAbs(prep_target, alpha=0.82, beta=-15)

        # 1. Filtro CLAHE en LAB (Excelente para luminancia general)
        mejorada = mejorar_placa_lejana(prep_target)
        dict_det = procesar_tokens_ocr(ejecutar_easyocr(mejorada))

        # 2. Filtro Afilado High-Pass Unsharp Masking (Aumenta contraste de bordes borrosos)
        if not dict_det:
            gauss = cv2.GaussianBlur(prep_target, (0, 0), 2.2)
            unsharp = cv2.addWeighted(prep_target, 2.4, gauss, -1.4, 0)
            dict_det = procesar_tokens_ocr(ejecutar_easyocr(unsharp))

        # Si estamos en modo profundo (al tomar foto), aplicar la batería de desblurring y superresolución:
        if modo_profundo and not dict_det:
            # 3. Realce Multiescala por Pirámide Laplaciana (Resuelve desenfoque gaussiano y falta de foco)
            try:
                g1 = cv2.GaussianBlur(prep_target, (0, 0), 0.9)
                g2 = cv2.GaussianBlur(prep_target, (0, 0), 2.2)
                sharp_multi = cv2.addWeighted(prep_target, 2.6, g1, -1.2, 0)
                sharp_multi = np.clip(cv2.addWeighted(sharp_multi, 1.25, g2, -0.35, 0), 0, 255).astype(np.uint8)
                dict_det = procesar_tokens_ocr(ejecutar_easyocr(sharp_multi, text_thresh=0.28, low_thresh=0.15))
            except Exception:
                pass

        if modo_profundo and not dict_det:
            # 4. Filtro Morfológico Black-Hat (Aísla trazos oscuros troquelados eliminando brillos amarillos/blancos)
            try:
                gray = cv2.cvtColor(prep_target, cv2.COLOR_BGR2GRAY)
                k_w = max(5, int(prep_target.shape[1] * 0.04))
                k_h = max(3, int(prep_target.shape[0] * 0.08))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
                blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
                _, th_bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                dict_det = procesar_tokens_ocr(ejecutar_easyocr(cv2.cvtColor(th_bh, cv2.COLOR_GRAY2BGR), text_thresh=0.30, low_thresh=0.18))
            except Exception:
                pass

        if modo_profundo and not dict_det:
            # 5. Binarización Adaptativa Local de Sauvola (Aísla caracteres borrosos o con sombras degradadas)
            try:
                gray_f = cv2.cvtColor(prep_target, cv2.COLOR_BGR2GRAY).astype(np.float32)
                media = cv2.blur(gray_f, (17, 17))
                media_sq = cv2.blur(gray_f**2, (17, 17))
                desv = np.sqrt(np.maximum(0, media_sq - media**2))
                umbral_sauvola = media * (1.0 + 0.28 * (desv / 128.0 - 1.0))
                sauvola_bin = np.where(gray_f > umbral_sauvola, 255, 0).astype(np.uint8)
                sauvola_bgr = cv2.cvtColor(sauvola_bin, cv2.COLOR_GRAY2BGR)
                dict_det = procesar_tokens_ocr(ejecutar_easyocr(sauvola_bgr, text_thresh=0.28, low_thresh=0.16))
            except Exception:
                pass

        if modo_profundo and not dict_det:
            # 6. Desconvolución de Wiener Espectral (Restaura barrido de movimiento horizontal)
            try:
                gray_w = cv2.cvtColor(prep_target, cv2.COLOR_BGR2GRAY).astype(np.float32)
                h_p, w_p = gray_w.shape
                psf = np.zeros((h_p, w_p), dtype=np.float32)
                largo_blur = max(5, int(w_p * 0.04))
                psf[h_p // 2, max(0, w_p // 2 - largo_blur // 2) : min(w_p, w_p // 2 + largo_blur // 2 + 1)] = 1.0 / largo_blur
                H = np.fft.fft2(np.fft.fftshift(psf))
                G = np.fft.fft2(gray_w)
                W = np.conj(H) / (np.abs(H)**2 + 0.02)
                wiener_res = np.clip(np.real(np.fft.ifft2(G * W)), 0, 255).astype(np.uint8)
                wiener_bgr = cv2.cvtColor(wiener_res, cv2.COLOR_GRAY2BGR)
                dict_det = procesar_tokens_ocr(ejecutar_easyocr(wiener_bgr, text_thresh=0.28, low_thresh=0.16))
            except Exception:
                pass

        if modo_profundo and not dict_det:
            # 7. Super-Resolución Lanczos 1.5x con detección ultra-sensible de bordes
            try:
                h_p, w_p = prep_target.shape[:2]
                upscaled = cv2.resize(prep_target, (int(w_p * 1.5), int(h_p * 1.5)), interpolation=cv2.INTER_LANCZOS4)
                # Filtro Laplaciano fino sobre imagen ampliada
                kernel_lap = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                sharpened_up = cv2.filter2D(upscaled, -1, kernel_lap)
                dict_det = procesar_tokens_ocr(ejecutar_easyocr(sharpened_up, text_thresh=0.22, low_thresh=0.12))
            except Exception:
                pass

        # 8. Fallback final sobre el recorte completo original si recortar bordes fue excesivo
        if not dict_det and crop_bgr.size > 400:
            dict_det = procesar_tokens_ocr(ejecutar_easyocr(crop_bgr, text_thresh=0.28 if modo_profundo else 0.45, low_thresh=0.16 if modo_profundo else 0.25))

        if dict_det:
            placa_cand = dict_det["placa_raw"]
            dict_det["placa_raw_ocr"] = placa_cand
            dict_det["auto_corregido"] = False
            return [dict_det]
        return []

    def _async_ocr_worker(self):
        """
        Trabajador en segundo plano: procesa solicitudes de OCR de manera asíncrona
        para que el hilo de captura y tracking corra libremente a 25-30 FPS.
        """
        _, _, reader = obtener_modelos()

        while self.is_running:
            try:
                item = self.ocr_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            track_id, crop = item
            if track_id is None:
                break

            ocr_res = self._ejecutar_ocr_en_recorte(crop, reader)
            if ocr_res:
                det_info = ocr_res[0]
                print(f"[OCR] Track #{track_id} IDENTIFICADO: {det_info['placa_raw']} ({det_info['tipo']} | {det_info.get('municipio', 'COLOMBIA')}) - Confianza: {det_info['confianza'] * 100:.1f}%")
                events = self.tracker.apply_async_ocr(track_id, det_info)
                for ev in events:
                    self._notify_listeners(ev)
            else:
                print(f"[OCR] Track #{track_id}: Sin placa valida en este cuadro.")

            self.pending_ocr_tracks.discard(track_id)
            self.ocr_queue.task_done()

    def _tracking_loop(self):
        """
        Bucle de Tracking ultrarrápido: corre YOLOv8 a 480p (~40 ms) y sigue la placa
        en tiempo real a 25+ FPS. Si una placa es nueva, delega el OCR al worker asíncrono.
        """
        fps_counter = 0
        last_calc_time = time.time()
        last_processed_frame_id = -1

        plate_detector, vehicle_detector, _ = obtener_modelos()

        while self.is_running:
            frame_to_process = None
            current_id = -1

            with self.lock:
                if self.latest_frame is not None and self.total_frames_read != last_processed_frame_id:
                    frame_to_process = self.latest_frame.copy()
                    current_id = self.total_frames_read

            if frame_to_process is None:
                time.sleep(0.005)
                continue

            last_processed_frame_id = current_id
            t0 = time.time()

            h_frame, w_frame = frame_to_process.shape[:2]
            detected_bboxes = []

            # 1. Detección YOLO especializada en placas vehiculares (conf=0.12 de alta sensibilidad)
            res_plates = plate_detector(frame_to_process, imgsz=640, verbose=False, conf=0.12)
            for r in res_plates:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    pw = bx2 - bx1
                    ph = by2 - by1
                    pad_x = int(pw * 0.14)
                    pad_y = int(ph * 0.10)
                    gx1 = max(0, bx1 - pad_x)
                    gy1 = max(0, by1 - pad_y)
                    gx2 = min(w_frame, bx2 + pad_x)
                    gy2 = min(h_frame, by2 + pad_y)
                    bw = gx2 - gx1
                    bh = gy2 - gy1
                    ratio = bw / max(1, bh)
                    # Filtro adaptativo para placas cercanas, lejanas o en perspectiva sin descartar acercamientos
                    if 0.8 <= ratio <= 8.5 and bw >= 35 and bh >= 15 and bw <= w_frame and bh <= h_frame:
                        detected_bboxes.append((gx1, gy1, gx2, gy2))

            # 2. Inferencia focalizada en el Encuadre Óptico (ROI)
            # Permite capturar placas en celulares y primeros planos aunque YOLO en plano completo falle
            rx1, ry1, rx2, ry2 = self.get_roi_coordinates(w_frame, h_frame)
            roi_crop = frame_to_process[ry1:ry2, rx1:rx2]

            if self.use_roi and roi_crop.size >= 1000:
                try:
                    roi_res = plate_detector(roi_crop, imgsz=640, verbose=False, conf=0.10)
                    for r in roi_res:
                        for box in r.boxes:
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                            gx1 = max(0, rx1 + bx1)
                            gy1 = max(0, ry1 + by1)
                            gx2 = min(w_frame, rx1 + bx2)
                            gy2 = min(h_frame, ry1 + by2)
                            detected_bboxes.append((gx1, gy1, gx2, gy2))
                except Exception:
                    pass

            # 3. Fallback por descomposición de color HSV si YOLO no detectó nada (fotos impresas, celulares, reflejos)
            if not detected_bboxes:
                if self.use_roi and roi_crop.size >= 1000:
                    roi_color = detectar_placas_por_color(roi_crop)
                    for (cx1, cy1, cx2, cy2) in roi_color:
                        detected_bboxes.append((rx1 + cx1, ry1 + cy1, rx1 + cx2, ry1 + cy2))
                if not detected_bboxes:
                    color_boxes = detectar_placas_por_color(frame_to_process)
                    if color_boxes:
                        detected_bboxes.extend(color_boxes)

            # 4. Fallback de encuadre óptico si YOLO y color no detectaron nada (pantallas de celular en primer plano)
            if not detected_bboxes and (self.use_roi or not self.tracker.active_tracks):
                tiene_txt, _ = verificar_densidad_caracteres(roi_crop)
                if tiene_txt or not self.tracker.active_tracks:
                    detected_bboxes.append((rx1, ry1, rx2, ry2))

            # Aplicar Supresión de No Máximos (NMS) para fusionar cajas solapadas de una misma placa
            detected_bboxes = aplicar_nms_cajas(detected_bboxes, iou_threshold=0.20)

            # 2. Seguimiento geométrico ultrarrápido (IoU + Centroides)
            active_visuals, events, tracks_needing_ocr = self.tracker.update(
                detected_bboxes,
                run_ocr_func=None,
                frame_bgr=frame_to_process
            )

            # 3. Enviar recortes al worker OCR asíncrono sin bloquear este bucle
            for track_id, crop in tracks_needing_ocr:
                if track_id not in self.pending_ocr_tracks or self.ocr_queue.empty():
                    if self.ocr_queue.full():
                        try:
                            old_tid, _ = self.ocr_queue.get_nowait()
                            self.pending_ocr_tracks.discard(old_tid)
                        except queue.Empty:
                            pass
                    try:
                        self.ocr_queue.put_nowait((track_id, crop))
                        self.pending_ocr_tracks.add(track_id)
                    except queue.Full:
                        pass

            latency = (time.time() - t0) * 1000.0
            self.inference_latency_ms = round(latency, 1)

            with self.lock:
                self.latest_detections = active_visuals
                self.total_frames_analyzed += 1

            for ev in events:
                self._notify_listeners(ev)

            fps_counter += 1
            now_t = time.time()
            if now_t - last_calc_time >= 1.0:
                self.ocr_fps = round(fps_counter / (now_t - last_calc_time), 1)
                fps_counter = 0
                last_calc_time = now_t

    @staticmethod
    def _draw_corner_rect(img, pt1, pt2, color, thickness=2, line_len=35):
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.line(img, (x1, y1), (x1 + line_len, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + line_len), color, thickness)
        cv2.line(img, (x2, y1), (x2 - line_len, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + line_len), color, thickness)
        cv2.line(img, (x1, y2), (x1 + line_len, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - line_len), color, thickness)
        cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)

    def get_annotated_frame(self):
        with self.lock:
            if self.latest_frame is None:
                # Standby visual para que la transmisión MJPEG nunca se corte o congele en el navegador
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                msg = f"CONECTANDO FUENTE: {self.source}..."
                cv2.putText(blank, msg, (100, 240), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 230, 118), 1, cv2.LINE_AA)
                return blank
            frame = self.latest_frame.copy()
            detections = list(self.latest_detections)

        h_f, w_f = frame.shape[:2]

        # Dibujar retícula de encuadre óptico si está activo
        if self.use_roi:
            rx1, ry1, rx2, ry2 = self.get_roi_coordinates(w_f, h_f)

            # Dibujar esquinas del encuadre en verde/cyan brillante
            self._draw_corner_rect(frame, (rx1, ry1), (rx2, ry2), (0, 230, 118), 2, 40)

            roi_label = "[ENCUADRE OPTICO: CENTRA LA PLACA O CELULAR]"
            cv2.putText(frame, roi_label, (rx1 + 10, ry1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, (18, 22, 28), 3, cv2.LINE_AA)
            cv2.putText(frame, roi_label, (rx1 + 10, ry1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 230, 118), 1, cv2.LINE_AA)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            placa = det["placa_raw"]
            tipo = det.get("tipo", "CARRO")
            mun = det.get("municipio")
            conf_val = det.get("confianza", 0.0)

            # Si la placa aún se está identificando en el hilo asíncrono
            es_pendiente = (placa == "DESCONOCIDO")

            color_bgr = (0, 220, 255) if es_pendiente else ((46, 204, 113) if tipo == "CARRO" else (241, 196, 15))

            # Dibujar recuadro de seguimiento
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            if es_pendiente:
                etiqueta = "[IDENTIFICANDO...]"
            else:
                conf_pct = f"{conf_val * 100:.0f}%"
                etiqueta = f"[{tipo}] {placa}  {conf_pct}"
                if mun and mun != "NO REGISTRADO":
                    etiqueta += f" | {mun}"

            (tw, th), baseline = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            ty = max(y1 - 8, th + 8)
            cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 10, ty + baseline - 2), (18, 22, 28), -1)
            cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 10, ty + baseline - 2), color_bgr, 1)

            cv2.putText(
                frame,
                etiqueta,
                (x1 + 5, ty - 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        hud_text = f"FPS CAM: {self.camera_fps} | TRACKING: {self.ocr_fps} FPS | LATENCIA: {self.inference_latency_ms} ms | ACTIVOS: {len(detections)}"
        cv2.putText(frame, hud_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, hud_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 118), 1, cv2.LINE_AA)

        return frame

    def generate_mjpeg(self):
        while self.is_running:
            frame = self.get_annotated_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.015)
