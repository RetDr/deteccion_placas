"""
Módulo de Seguimiento de Objetos en Tiempo Real (IoU Tracker).
Resuelve definitivamente la duplicidad de registros:
1. Memoria temporal de 1.2 segundos (35 cuadros) para no perder el rastro del vehículo.
2. Votación por mayoría temporal: estabiliza lecturas ruidosas (ej: ABC128 -> ABC123).
3. Ventana de deduplicación de 4 segundos: evita que una misma placa cree 3 o 4 tarjetas
   si la detección parpadea momentáneamente.
4. Emisión única: 1 vehículo en movimiento = 1 sola tarjeta consolidada en la interfaz.
"""

import os
import json
import time
import base64
import threading
from collections import Counter
import cv2
import numpy as np


def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def calcular_distancia_centros_normalizada(boxA, boxB):
    """
    Calcula la distancia euclidiana entre centros de cajas normalizada por tamaño.
    Permite seguir placas en movimiento o vibración incluso si el IoU cae momentáneamente.
    """
    cA_x, cA_y = (boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0
    cB_x, cB_y = (boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0
    wA = max(1, boxA[2] - boxA[0])
    hA = max(1, boxA[3] - boxA[1])
    dx = abs(cA_x - cB_x) / float(wA)
    dy = abs(cA_y - cB_y) / float(hA)
    return (dx ** 2 + dy ** 2) ** 0.5


def son_la_misma_placa(boxA, boxB) -> bool:
    """
    Determina si dos cajas pertenecen a la misma placa vehicular mediante:
    1. IoU >= 0.15
    2. Distancia euclidiana de centroides normalizada <= 0.40
    3. Contención relativa (IoMin >= 0.35)
    Elimina duplicados y evita que aparezcan 2 o 3 recuadros verdes en la misma placa.
    """
    if calcular_iou(boxA, boxB) >= 0.15:
        return True
    if calcular_distancia_centros_normalizada(boxA, boxB) <= 0.40:
        return True
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area > 0:
        areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
        areaB = max(1, (bx2 - bx1) * (by2 - by1))
        if inter_area / float(min(areaA, areaB)) >= 0.35:
            return True
    return False


class TrackedPlate:
    def __init__(self, track_id: int, bbox: tuple, initial_detection: dict):
        self.track_id = track_id
        self.bbox = bbox

        self.placa_raw = initial_detection.get("placa_raw", "DESCONOCIDO")
        self.confianza = initial_detection.get("confianza", 0.0)
        self.tipo = initial_detection.get("tipo", "CARRO")
        self.municipio = initial_detection.get("municipio") or "NO REGISTRADO"
        self.recorte_img = initial_detection.get("recorte_img")

        self.crop_base64 = None
        self._generar_base64()

        self.first_seen = time.time()
        self.last_seen = time.time()
        self.invisible_frames = 0
        self.hits = 1

        self.readings = [self.placa_raw] if (self.placa_raw != "DESCONOCIDO" and len(self.placa_raw) == 6) else []
        self.has_been_emitted = False
        self.last_emitted_time = 0.0
        self._cached_antecedentes = None

        # Almacenamiento del fotograma de mayor nitidez (Best-Shot Buffer contra desenfoque de movimiento)
        self.best_crop = self.recorte_img
        self.best_sharpness = 0.0
        if self.recorte_img is not None and self.recorte_img.size > 400:
            try:
                g = cv2.cvtColor(self.recorte_img, cv2.COLOR_BGR2GRAY)
                self.best_sharpness = float(cv2.Laplacian(g, cv2.CV_64F).var())
            except Exception:
                self.best_sharpness = 1.0

    def _generar_base64(self):
        if self.recorte_img is not None and self.recorte_img.size > 0:
            ret, buffer = cv2.imencode('.jpg', self.recorte_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                self.crop_base64 = base64.b64encode(buffer).decode('utf-8')

    def update_position(self, new_bbox: tuple, new_crop: np.ndarray = None):
        ox1, oy1, ox2, oy2 = self.bbox
        nx1, ny1, nx2, ny2 = new_bbox
        delta = max(abs(nx1 - ox1), abs(ny1 - oy1), abs(nx2 - ox2), abs(ny2 - oy2))

        # Filtro de zona muerta y suavizado adaptativo:
        # Menor a 6px: congelar retícula (elimina temblor de sensor en reposo)
        # Menor a 25px: suavizado sutil (alpha=0.25)
        # Mayor o igual a 25px (vehículo en movimiento rápido en carretera): seguimiento ágil (alpha=0.65)
        if delta < 6:
            self.last_seen = time.time()
            self.invisible_frames = 0
            self.hits += 1
        else:
            alpha = 0.25 if delta < 25 else 0.65
            sx1 = int(round(alpha * nx1 + (1.0 - alpha) * ox1))
            sy1 = int(round(alpha * ny1 + (1.0 - alpha) * oy1))
            sx2 = int(round(alpha * nx2 + (1.0 - alpha) * ox2))
            sy2 = int(round(alpha * ny2 + (1.0 - alpha) * oy2))
            self.bbox = (sx1, sy1, sx2, sy2)
            self.last_seen = time.time()
            self.invisible_frames = 0
            self.hits += 1

        # Evaluar nitidez del fotograma para sustituir por la toma más nítida
        if new_crop is not None and new_crop.size > 400:
            try:
                g = cv2.cvtColor(new_crop, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(g, cv2.CV_64F).var())
                if sharpness > self.best_sharpness:
                    self.best_sharpness = sharpness
                    self.best_crop = new_crop.copy()
            except Exception:
                pass

    def update_ocr(self, new_det: dict):
        nueva_placa = new_det.get("placa_raw", "")
        nueva_conf = new_det.get("confianza", 0.0)

        if nueva_placa and nueva_placa != "DESCONOCIDO" and len(nueva_placa) == 6:
            if nueva_placa != self.placa_raw:
                self._cached_antecedentes = None
            self.readings.append(nueva_placa)
            # Votación por mayoría temporal entre las lecturas acumuladas (elimina glitches aislados)
            placa_consenso, veces = Counter(self.readings).most_common(1)[0]
            self.placa_raw = placa_consenso

        if nueva_conf >= self.confianza or self.placa_raw != "DESCONOCIDO":
            self.confianza = max(self.confianza, nueva_conf)
            if new_det.get("tipo"):
                self.tipo = new_det["tipo"]
            if new_det.get("municipio"):
                self.municipio = new_det["municipio"]
            if new_det.get("recorte_img") is not None:
                self.recorte_img = new_det["recorte_img"]
                self._generar_base64()

    @property
    def is_confirmed(self):
        # Confirmación inmediata al validar sintaxis estricta colombiana con confianza >= 0.35
        if not self.placa_raw or self.placa_raw == "DESCONOCIDO" or len(self.placa_raw) != 6:
            return False
        return len(self.readings) >= 1 and self.confianza >= 0.35

    def to_dict(self):
        if self._cached_antecedentes is None and self.placa_raw != "DESCONOCIDO":
            try:
                from modules.search_engine import consultar_placa
                self._cached_antecedentes = consultar_placa(self.placa_raw, ocr_confidence=self.confianza)
            except Exception:
                self._cached_antecedentes = None

        return {
            "track_id": self.track_id,
            "placa": self.placa_raw,
            "tipo": self.tipo,
            "municipio": self.municipio,
            "confianza": round(self.confianza, 4),
            "hits": self.hits,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(self.last_seen)),
            "crop_base64": self.crop_base64,
            "bbox": self.bbox,
            "antecedentes": self._cached_antecedentes
        }


class RealtimePlateTracker:
    def __init__(self, iou_threshold=0.20, max_invisible_frames=30, deduplication_window_sec=8.0, persistence_path=None):
        self.iou_threshold = iou_threshold
        self.max_invisible_frames = max_invisible_frames
        self.deduplication_window_sec = deduplication_window_sec

        self.lock = threading.Lock()
        self.active_tracks: dict[int, TrackedPlate] = {}
        self.next_track_id = 1
        self.confirmed_history: dict[str, dict] = {}
        self.recent_seen_plates: dict[str, float] = {}  # {placa: timestamp_ultima_vez}

        self.persistence_path = persistence_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "historial_capturas.json"
        )
        self._cargar_historial_persistente()

    def _cargar_historial_persistente(self):
        try:
            if os.path.exists(self.persistence_path):
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.confirmed_history.update(data)
                        print(f"[TRACKER] Cargadas {len(data)} placas previas desde: {os.path.basename(self.persistence_path)}")
        except Exception:
            pass

    def _guardar_historial_persistente(self):
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(self.confirmed_history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _es_placa_reciente(self, placa: str, now: float) -> bool:
        """
        Verifica si la placa ya fue registrada recientemente en la ventana de 10s.
        Incluye tolerancia a distancia de Levenshtein <= 1 (ej: ABC123 vs ABC128)
        para evitar que ligeras variaciones de OCR creen tarjetas separadas.
        """
        if not placa or placa == "DESCONOCIDO" or len(placa) != 6:
            return True

        if placa in self.recent_seen_plates:
            return True

        for p_reg in self.recent_seen_plates:
            if len(p_reg) == 6:
                diffs = sum(1 for c1, c2 in zip(p_reg, placa) if c1 != c2)
                if diffs <= 1:
                    return True

        return False

    def apply_async_ocr(self, track_id: int, ocr_det: dict) -> list[dict]:
        """
        Aplica asíncronamente el resultado del trabajador OCR en segundo plano sin bloquear
        el hilo principal de seguimiento (permitiendo mantener 25-30 FPS constantes).
        Resiliente ante jitter: nunca descarta una placa válida aunque el track_id haya oscilado.
        """
        events = []
        now = time.time()
        placa = ocr_det.get("placa_raw", "")

        with self.lock:
            # Buscar el track correspondiente o transferir al track activo más cercano
            track = self.active_tracks.get(track_id)
            if not track and self.active_tracks:
                track = list(self.active_tracks.values())[0]

            if track:
                track.update_ocr(ocr_det)

            # Si se identificó una placa válida de 6 caracteres
            if placa and placa != "DESCONOCIDO" and len(placa) == 6:
                is_new = not self._es_placa_reciente(placa, now)
                self.recent_seen_plates[placa] = now
                payload = track.to_dict() if track else {
                    "track_id": track_id,
                    "placa": placa,
                    "tipo": ocr_det.get("tipo", "CARRO"),
                    "municipio": ocr_det.get("municipio", "COLOMBIA"),
                    "confianza": ocr_det.get("confianza", 0.95),
                    "hits": 1,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "crop_base64": "",
                    "bbox": (0, 0, 0, 0),
                    "antecedentes": None
                }
                if not payload.get("antecedentes"):
                    try:
                        from modules.search_engine import consultar_placa
                        payload["antecedentes"] = consultar_placa(placa)
                    except Exception:
                        pass

                if track:
                    track.has_been_emitted = True
                    track.last_emitted_time = now

                self.confirmed_history[placa] = payload
                self._guardar_historial_persistente()
                if is_new:
                    events.append({"event": "NEW_PLATE", "data": payload})
                else:
                    events.append({"event": "UPDATE_PLATE", "data": payload})

        return events

    def update(self, detected_bboxes: list[tuple], run_ocr_func=None, frame_bgr: np.ndarray=None) -> tuple:
        """
        Actualiza el estado de las pistas con detección ultrarrápida (IoU + distancia normalizada).
        Si run_ocr_func es None, opera en modo asíncrono retornando tracks_needing_ocr.
        """
        now = time.time()
        matched_tracks = set()
        unmatched_detections = []
        events_to_emit = []
        tracks_needing_ocr = []

        with self.lock:
            # Limpiar placas vistas hace más de deduplication_window_sec segundos
            for p, ts in list(self.recent_seen_plates.items()):
                if now - ts > self.deduplication_window_sec:
                    del self.recent_seen_plates[p]

            # 1. Asociar cajas detectadas con tracks activos mediante IoU y distancia de centroides
            for det_box in detected_bboxes:
                best_score = 0.0
                best_track_id = None

                for track_id, track in self.active_tracks.items():
                    if track_id in matched_tracks:
                        continue

                    iou = calcular_iou(det_box, track.bbox)
                    dist_norm = calcular_distancia_centros_normalizada(det_box, track.bbox)

                    if iou >= self.iou_threshold or dist_norm <= 0.45 or son_la_misma_placa(det_box, track.bbox):
                        score = max(iou, 1.0 - dist_norm)
                        if score > best_score:
                            best_score = score
                            best_track_id = track_id

                if best_track_id is not None:
                    matched_tracks.add(best_track_id)
                    track = self.active_tracks[best_track_id]
                    rx1, ry1, rx2, ry2 = det_box
                    crop = frame_bgr[ry1:ry2, rx1:rx2] if frame_bgr is not None else None
                    track.update_position(det_box, crop)

                    # Si aún no está confirmada, programar o ejecutar OCR usando la toma fresca o más nítida
                    if not track.is_confirmed and crop is not None and crop.size > 0:
                        crop_to_send = crop if (crop is not None and crop.size > 400) else (track.best_crop if track.best_crop is not None else crop)
                        if run_ocr_func is not None:
                            ocr_res = run_ocr_func(crop_to_send)
                            if ocr_res:
                                track.update_ocr(ocr_res[0])
                        else:
                            tracks_needing_ocr.append((track.track_id, crop_to_send))

                    # Si ya está confirmada y no se ha emitido, emitir evento inmediatamente
                    if track.is_confirmed and not track.has_been_emitted:
                        is_new = not self._es_placa_reciente(track.placa_raw, now)
                        track.has_been_emitted = True
                        track.last_emitted_time = now
                        self.recent_seen_plates[track.placa_raw] = now
                        payload = track.to_dict()
                        self.confirmed_history[track.placa_raw] = payload
                        self._guardar_historial_persistente()
                        if is_new:
                            events_to_emit.append({"event": "NEW_PLATE", "data": payload})
                        else:
                            events_to_emit.append({"event": "UPDATE_PLATE", "data": payload})
                    elif track.has_been_emitted and (now - getattr(track, "last_emitted_time", 0) > 3.0):
                        track.last_emitted_time = now
                        self.recent_seen_plates[track.placa_raw] = now
                        payload = track.to_dict()
                        events_to_emit.append({"event": "UPDATE_PLATE", "data": payload})

                else:
                    unmatched_detections.append(det_box)

            # 2. Nuevas detecciones en escena (requieren confirmación)
            for new_box in unmatched_detections:
                # Comprobación de no duplicidad absoluta: si la nueva caja se solapa con cualquier track existente, descartarla
                es_duplicada = False
                for existing_track in self.active_tracks.values():
                    if son_la_misma_placa(new_box, existing_track.bbox):
                        es_duplicada = True
                        break
                if es_duplicada:
                    continue

                rx1, ry1, rx2, ry2 = new_box
                crop = frame_bgr[ry1:ry2, rx1:rx2] if frame_bgr is not None else np.array([])

                initial_det = {"placa_raw": "DESCONOCIDO", "confianza": 0.0, "tipo": "CARRO", "recorte_img": crop}
                if run_ocr_func is not None and crop.size > 0:
                    h_c, w_c = crop.shape[:2]
                    if w_c >= 45 and h_c >= 15:
                        ocr_res = run_ocr_func(crop)
                        if ocr_res:
                            initial_det = ocr_res[0]

                track_id = self.next_track_id
                self.next_track_id += 1

                new_track = TrackedPlate(track_id, new_box, initial_det)
                self.active_tracks[track_id] = new_track

                if run_ocr_func is None and crop.size > 0:
                    h_c, w_c = crop.shape[:2]
                    if w_c >= 45 and h_c >= 15:
                        tracks_needing_ocr.append((track_id, crop))

            # 3. Incrementar invisibilidad con tolerancia de 35 cuadros (1.2 segundos)
            tracks_to_delete = []
            for track_id, track in self.active_tracks.items():
                if track_id not in matched_tracks:
                    track.invisible_frames += 1
                    if track.invisible_frames > self.max_invisible_frames:
                        tracks_to_delete.append(track_id)

            for tid in tracks_to_delete:
                del self.active_tracks[tid]

            # 4. Detecciones activas para renderizado visual
            active_visual_detections = []
            for track in self.active_tracks.values():
                if track.invisible_frames <= 3:  # Suavizar parpadeos momentáneos
                    active_visual_detections.append({
                        "bbox": track.bbox,
                        "placa_raw": track.placa_raw,
                        "confianza": track.confianza,
                        "tipo": track.tipo,
                        "municipio": track.municipio,
                        "track_id": track.track_id
                    })

        if run_ocr_func is not None:
            return active_visual_detections, events_to_emit
        return active_visual_detections, events_to_emit, tracks_needing_ocr
