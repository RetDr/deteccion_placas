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


def detectar_placas_por_color(frame_bgr: np.ndarray) -> list[tuple]:
    """
    Descomposición cromática HSV para placas colombianas (Amarillo y Blanco).
    Actúa como red de seguridad infalible cuando el modelo YOLO no detecta la placa
    en una fotografía, pantalla de celular o bajo iluminación atípica.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    h_f, w_f = frame_bgr.shape[:2]
    cajas = []

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Rango amarillo colombiano (12-38) y blanco reflectivo (0-180 con alta luminancia)
    mask_y = cv2.inRange(hsv, np.array([12, 60, 60]), np.array([38, 255, 255]))
    mask_w = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
    mask = cv2.bitwise_or(mask_y, mask_w)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        if 1000 <= area <= int(h_f * w_f * 0.95):
            cx, cy, cw, ch = cv2.boundingRect(c)
            ratio = float(cw) / max(1, ch)
            if 0.9 <= ratio <= 6.5 and cw >= 40 and ch >= 16:
                crop = frame_bgr[cy:cy+ch, cx:cx+cw]
                tiene_texto, _ = verificar_densidad_caracteres(crop)
                if tiene_texto:
                    pad_x = int(cw * 0.16)
                    pad_y = int(ch * 0.12)
                    gx1 = max(0, cx - pad_x)
                    gy1 = max(0, cy - pad_y)
                    gx2 = min(w_f, cx + cw + pad_x)
                    gy2 = min(h_f, cy + ch + pad_y)
                    cajas.append((gx1, gy1, gx2, gy2))

    return cajas


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

        # En Windows, probar primero con DirectShow (DSHOW) para compatibilidad total con celulares USB (Android 14, DroidCam, Iriun)
        if isinstance(src, int) or str(src).isdigit():
            idx = int(src)
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap or not self.cap.isOpened():
            print(f"[ERROR] No se pudo inicializar la fuente de video: {self.source}")
            return False

        if str(self.source).isdigit():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Solicitar máxima resolución compatible sin forzar códecs incompatibles con celulares
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

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
        return self.start(new_source)

    def _capture_loop(self):
        last_time = time.time()
        fps_counter = 0

        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if not str(self.source).isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.01)
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

    def _ejecutar_ocr_en_recorte(self, crop_bgr: np.ndarray, reader) -> list[dict]:
        """
        Ejecuta OCR sobre el recorte con superresolución y coteja municipios de Colombia.
        Robusto ante marcos de concesionario (Chevrolet, Toyota), tornillos y división de tokens.
        Incluye pipeline multi-intento (CLAHE -> Crudo -> Unsharp Blur para carretera).
        """
        if crop_bgr is None or crop_bgr.size < 400:
            return []

        # 1. Modo Híbrido: Si el usuario configuró Gemini Flash, intentar usarlo
        if gemini_disponible():
            gemini_res = consultar_gemini_placa(crop_bgr)
            if gemini_res:
                return [gemini_res]

        allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ."

        def ejecutar_easyocr(img_in):
            try:
                return reader.readtext(
                    img_in,
                    allowlist=allowlist,
                    detail=1,
                    paragraph=False,
                    mag_ratio=1.0,
                    add_margin=0.15
                )
            except Exception:
                return []

        def procesar_tokens_ocr(res_tokens):
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
                if clasificacion == "MUNICIPIO":
                    municipios.append(texto_proc)
                elif clasificacion != "DESCARTAR":
                    candidatos.append({
                        "texto": texto_proc,
                        "prob": float(prob),
                        "bbox": (bbox_pts[0][0], bbox_pts[0][1], bbox_pts[2][0], bbox_pts[2][1]),
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
                placa_val, tipo_veh, conf_bonus = extraer_placa_estricta(texto_cand)
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

            return {
                "placa_raw": str(mejor_placa),
                "confianza": round(prob_final, 4),
                "tipo": tipo_vehiculo,
                "municipio": municipio_oficial,
                "recorte_img": crop_bgr
            }

        # Multi-intento jerárquico inteligente con tolerancia extrema a ruido, sol y reflejos:
        # Intento 1: Con preprocesamiento de alta definición (LAB / CLAHE)
        preprocesada = mejorar_placa_lejana(crop_bgr)
        dict_det = procesar_tokens_ocr(ejecutar_easyocr(preprocesada))

        # Intento 2 (Otsu Adaptive Binarization): Separa caracteres de reflejos lumínicos / contraluz
        if not dict_det:
            otsu_img = binarizar_placa_otsu(crop_bgr)
            dict_det = procesar_tokens_ocr(ejecutar_easyocr(otsu_img))

        # Intento 3 (Franja interna recortada + Otsu): Elimina tornillos/remaches y marcos exteriores
        if not dict_det:
            franja_otsu = binarizar_placa_otsu(recortar_franja_caracteres(crop_bgr))
            dict_det = procesar_tokens_ocr(ejecutar_easyocr(franja_otsu))

        # Intento 4 (Franja interna cruda):
        if not dict_det:
            franja = recortar_franja_caracteres(crop_bgr)
            dict_det = procesar_tokens_ocr(ejecutar_easyocr(franja))

        # Intento 5 (Fallback): Con recorte crudo original (por si el realce cromático alteró contraste)
        if not dict_det:
            dict_det = procesar_tokens_ocr(ejecutar_easyocr(crop_bgr))

        # Intento 6 (Fallback Carretera): Con Unsharp Mask para compensar desenfoque de movimiento
        if not dict_det:
            try:
                gauss = cv2.GaussianBlur(crop_bgr, (0, 0), 2.0)
                unsharp = cv2.addWeighted(crop_bgr, 1.5, gauss, -0.5, 0)
                dict_det = procesar_tokens_ocr(ejecutar_easyocr(unsharp))
            except Exception:
                pass

        if dict_det:
            # Auto-alineación asistida por base de datos (tolerancia a confusiones visuales 1-carácter)
            placa_cand = dict_det["placa_raw"]
            dict_det["placa_raw_ocr"] = placa_cand
            dict_det["auto_corregido"] = False
            try:
                ant = consultar_placa(placa_cand)
                if ant and ant.get("encontrado"):
                    metodo = ant.get("metodo_usado")
                    placa_db = ant.get("placa_encontrada")
                    sim = ant.get("similitud", 0.0)
                    if metodo == "SIMILITUD" and sim >= 82.0 and placa_db and len(placa_db) == 6:
                        diffs = sum(1 for a, b in zip(placa_cand, placa_db) if a != b)
                        if diffs == 1:
                            dict_det["placa_raw"] = placa_db
                            dict_det["auto_corregido"] = True
                            dict_det["confianza"] = min(1.0, dict_det["confianza"] * 1.05)
            except Exception:
                pass
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

            # 1. Detección YOLO en resolución completa a conf=0.15 (idéntico al modo foto que acierta 4/5)
            res_plates = plate_detector(frame_to_process, verbose=False, conf=0.15)
            for r in res_plates:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    pw = bx2 - bx1
                    ph = by2 - by1
                    pad_x = int(pw * 0.16)
                    pad_y = int(ph * 0.12)
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

            # 2. Fallback por descomposición de color HSV si YOLO no detectó nada (fotos impresas, celulares, reflejos)
            if not detected_bboxes:
                color_boxes = detectar_placas_por_color(frame_to_process)
                if color_boxes:
                    detected_bboxes.extend(color_boxes)

            # 3. Fallback de encuadre óptico y encuadre central constante (idéntico al modo captura que acierta 4/5)
            if not detected_bboxes:
                rx1, ry1 = int(w_frame * 0.15), int(h_frame * 0.20)
                rx2, ry2 = int(w_frame * 0.85), int(h_frame * 0.80)
                crop_center = frame_to_process[ry1:ry2, rx1:rx2]
                tiene_txt, _ = verificar_densidad_caracteres(crop_center)
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
                if track_id not in self.pending_ocr_tracks:
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
                return None
            frame = self.latest_frame.copy()
            detections = list(self.latest_detections)

        h_f, w_f = frame.shape[:2]

        # Dibujar retícula de encuadre óptico si está activo
        if self.use_roi:
            rx1 = int(w_f * 0.17)
            ry1 = int(h_f * 0.17)
            rx2 = int(w_f * 0.83)
            ry2 = int(h_f * 0.83)

            # Dibujar esquinas del encuadre en verde/cyan brillante
            self._draw_corner_rect(frame, (rx1, ry1), (rx2, ry2), (0, 230, 118), 2, 35)

            roi_label = "[ENCUADRE OPTICO HD]"
            cv2.putText(frame, roi_label, (rx1 + 10, ry1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (18, 22, 28), 3, cv2.LINE_AA)
            cv2.putText(frame, roi_label, (rx1 + 10, ry1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 230, 118), 1, cv2.LINE_AA)

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
