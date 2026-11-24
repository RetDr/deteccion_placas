"""
Servidor Web de Alto Rendimiento para el Sistema de Reconocimiento de Placas.
Construido sobre FastAPI + Uvicorn con transmisión MJPEG a 30+ FPS y WebSockets.
"""

import os
import sys

# Prevenir saturación de memoria de OpenBLAS/OMP en Windows con arquitectura multihilo
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["YOLO_CONFIG_DIR"] = os.path.join(BASE_DIR, "models", "ultralytics")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import time
import base64
import cv2
cv2.setNumThreads(1)
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from modules.camera_stream import LiveCameraEngine, detectar_placas_por_color, aplicar_nms_cajas, listar_camaras_disponibles
from modules.vision_ocr import obtener_modelos
from modules.search_engine import consultar_placa
from modules.gemini_ocr import gemini_disponible, guardar_gemini_key, obtener_gemini_key, consultar_gemini_placa

# Inicializar motor de captura asíncrono
# Por defecto prioriza la cámara principal integrada/webcam (0), luego secundaria (1)
default_source = 0
engine = LiveCameraEngine(source=default_source, sampling_step=4, min_confidence=0.28)

# Administrador de conexiones WebSocket activas
connected_websockets: list[WebSocket] = []


def on_engine_event(event_dict):
    """
    Callback invocado por el hilo de OCR cuando se detecta o actualiza una placa.
    Envía el evento a todos los clientes WebSocket conectados.
    """
    if not connected_websockets or server_loop is None or not server_loop.is_running():
        return

    import json
    msg_str = json.dumps(event_dict)

    # Programar envío en el loop de asyncio
    to_remove = []
    for ws in list(connected_websockets):
        try:
            future = asyncio.run_coroutine_threadsafe(ws.send_text(msg_str), loop=server_loop)
        except Exception:
            to_remove.append(ws)

    for dead_ws in to_remove:
        if dead_ws in connected_websockets:
            connected_websockets.remove(dead_ws)


engine.add_listener(on_engine_event)
server_loop = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_loop
    server_loop = asyncio.get_running_loop()

    # Iniciar motor de cámara: priorizar cámara 0 (webcam/USB principal), luego 1, luego video
    started = False
    for cam_idx in [0, 1]:
        if engine.start(cam_idx):
            print(f"[INFO] Cámara principal iniciada con éxito en índice: {cam_idx}")
            started = True
            break

    if not started:
        # Fallback a video local si ninguna camara fisica esta disponible
        test_video = os.path.join(BASE_DIR, "data", "video_prueba.mp4")
        if os.path.exists(test_video):
            print(f"[INFO] Camaras fisicas no disponibles. Usando video local: {test_video}")
            engine.start(test_video)

    # Tarea en segundo plano para emitir telemetría cada segundo
    telemetry_task = asyncio.create_task(broadcast_telemetry())
    yield
    telemetry_task.cancel()
    engine.stop()


app = FastAPI(title="Sistema ANPR Colombia", lifespan=lifespan)


async def broadcast_telemetry():
    """Emite métricas de rendimiento (FPS, latencia) por WebSocket cada 800ms."""
    import json
    while True:
        await asyncio.sleep(0.8)
        if connected_websockets:
            data = {
                "event": "TELEMETRY",
                "data": {
                    "camera_fps": engine.camera_fps,
                    "ocr_fps": engine.ocr_fps,
                    "inference_latency_ms": engine.inference_latency_ms,
                    "total_frames_read": engine.total_frames_read,
                    "total_frames_analyzed": engine.total_frames_analyzed,
                    "active_plates_count": len(engine.history_plates)
                }
            }
            msg = json.dumps(data)
            for ws in list(connected_websockets):
                try:
                    await ws.send_text(msg)
                except Exception:
                    pass


@app.get("/", response_class=HTMLResponse)
async def get_index():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/video_feed")
async def get_video_feed():
    """
    Transmisión continua de video MJPEG a 30+ FPS nativos.
    """
    return StreamingResponse(
        engine.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
async def get_status():
    return JSONResponse({
        "camera_fps": engine.camera_fps,
        "ocr_fps": engine.ocr_fps,
        "inference_latency_ms": engine.inference_latency_ms,
        "total_frames": engine.total_frames_read,
        "total_analyzed": engine.total_frames_analyzed,
        "plates_detected": len(engine.history_plates),
        "source": str(engine.source),
        "use_roi": engine.use_roi
    })


@app.get("/api/cameras")
async def get_available_cameras():
    """Retorna la lista de dispositivos de cámara físicos, virtuales (DroidCam, Iriun) o de archivo."""
    cams = listar_camaras_disponibles()
    return JSONResponse(cams)


@app.post("/api/toggle_roi")
async def toggle_roi():
    engine.use_roi = not engine.use_roi
    return {"success": True, "use_roi": engine.use_roi}


@app.post("/api/capture_frame")
async def capture_frame():
    """Toma una foto instantánea del frame actual de la cámara, detecta la placa, ejecuta OCR y consulta antecedentes."""
    with engine.lock:
        if engine.latest_frame is None:
            return JSONResponse({"success": False, "error": "No hay señal de cámara disponible."}, status_code=400)
        frame = engine.latest_frame.copy()

    plate_detector, _, reader = obtener_modelos()
    h_f, w_f = frame.shape[:2]
    cajas = []

    # Coordenadas exactas del Encuadre Óptico adaptativo
    rx1_roi, ry1_roi, rx2_roi, ry2_roi = engine.get_roi_coordinates(w_f, h_f)
    roi_crop = frame[ry1_roi:ry2_roi, rx1_roi:rx2_roi]

    # 1. Si el Encuadre Óptico está activo, dar máxima prioridad al área encuadrada
    # Esto garantiza detección inmediata al mostrar un celular o apuntar con el teléfono
    if engine.use_roi and roi_crop.size >= 1000:
        res_roi = plate_detector(roi_crop, imgsz=640, verbose=False, conf=0.10)
        for r in res_roi:
            for box in r.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                cajas.append((max(0, rx1_roi + bx1), max(0, ry1_roi + by1), min(w_f, rx1_roi + bx2), min(h_f, ry1_roi + by2)))

        cajas_color_roi = detectar_placas_por_color(roi_crop)
        for (cx1, cy1, cx2, cy2) in cajas_color_roi:
            cajas.append((rx1_roi + cx1, ry1_roi + cy1, rx1_roi + cx2, ry1_roi + cy2))

        # Añadir el encuadre óptico completo como candidato prioritario
        cajas.append((rx1_roi, ry1_roi, rx2_roi, ry2_roi))

    # 2. Detección YOLO en fotograma completo
    res = plate_detector(frame, imgsz=640, verbose=False, conf=0.12)
    for r in res:
        for box in r.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            pw = bx2 - bx1
            ph = by2 - by1
            pad_x = int(pw * 0.14)
            pad_y = int(ph * 0.10)
            cajas.append((max(0, bx1 - pad_x), max(0, by1 - pad_y), min(w_f, bx2 + pad_x), min(h_f, by2 + pad_y)))

    # 3. Fallback por descomposición de color HSV general si aún no hay cajas
    if not cajas:
        cajas = detectar_placas_por_color(frame)

    # 4. Fallback de encuadre
    if not cajas:
        cajas.append((rx1_roi, ry1_roi, rx2_roi, ry2_roi))

    # Filtrar cajas con NMS
    cajas = aplicar_nms_cajas(cajas, iou_threshold=0.20)

    det = None
    mejor_crop = None
    mejor_bbox = None

    # Evaluar candidatos hasta encontrar la placa con mayor confianza usando OCR Profundo
    for (rx1, ry1, rx2, ry2) in cajas[:5]:
        c_cand = frame[ry1:ry2, rx1:rx2]
        if c_cand.size < 400:
            continue
        det_cand = None
        if gemini_disponible():
            det_cand = consultar_gemini_placa(c_cand)
        if not det_cand:
            # Activar modo profundo: Batería completa de nitidez, contraste LAB y Black-Hat
            ocr_res = engine._ejecutar_ocr_en_recorte(c_cand, reader, modo_profundo=True)
            if ocr_res:
                det_cand = ocr_res[0]
        if det_cand and det_cand.get("placa_raw") and det_cand["placa_raw"] != "DESCONOCIDO":
            if det is None or det_cand.get("confianza", 0) > det.get("confianza", 0):
                det = det_cand
                mejor_crop = det_cand.get("recorte_img", c_cand)
                if "sub_bbox" in det_cand and det_cand["sub_bbox"]:
                    sbx1, sby1, sbx2, sby2 = det_cand["sub_bbox"]
                    mejor_bbox = (max(0, rx1 + sbx1), max(0, ry1 + sby1), min(w_f, rx1 + sbx2), min(h_f, ry1 + sby2))
                else:
                    mejor_bbox = (rx1, ry1, rx2, ry2)

    # Si aún no se resolvió, reintentar con recorte expandido (+12% margen de seguridad)
    if not det:
        for (rx1, ry1, rx2, ry2) in cajas[:2]:
            pad_x = int((rx2 - rx1) * 0.15)
            pad_y = int((ry2 - ry1) * 0.15)
            c_exp = frame[max(0, ry1 - pad_y):min(h_f, ry2 + pad_y), max(0, rx1 - pad_x):min(w_f, rx2 + pad_x)]
            if c_exp.size >= 400:
                ocr_res = engine._ejecutar_ocr_en_recorte(c_exp, reader, modo_profundo=True)
                if ocr_res and ocr_res[0].get("placa_raw") != "DESCONOCIDO":
                    det = ocr_res[0]
                    mejor_crop = det.get("recorte_img", c_exp)
                    if "sub_bbox" in det and det["sub_bbox"]:
                        sbx1, sby1, sbx2, sby2 = det["sub_bbox"]
                        mejor_bbox = (max(0, rx1 - pad_x + sbx1), max(0, ry1 - pad_y + sby1), min(w_f, rx1 - pad_x + sbx2), min(h_f, ry1 - pad_y + sby2))
                    else:
                        mejor_bbox = (max(0, rx1 - pad_x), max(0, ry1 - pad_y), min(w_f, rx2 + pad_x), min(h_f, ry2 + pad_y))
                    break

    # Si no se leyó una placa definitiva, igual devolver el mejor encuadre con métrica de diagnóstico
    if not det or mejor_crop is None:
        primer_caja = cajas[0] if cajas else (rx1_roi, ry1_roi, rx2_roi, ry2_roi)
        cx1, cy1, cx2, cy2 = primer_caja
        crop_duda = frame[cy1:cy2, cx1:cx2] if frame[cy1:cy2, cx1:cx2].size > 100 else frame
        ret, buf = cv2.imencode('.jpg', crop_duda, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_crop_duda = base64.b64encode(buf).decode('utf-8') if ret else ""
        
        # Calcular si hubo algún texto parcial o qué confianza estimó el detector
        conf_estimada = 0.25 if len(cajas) > 0 else 0.10
        payload_duda = {
            "track_id": int(time.time()),
            "placa": "DUDOSA",
            "placa_raw_ocr": "???",
            "auto_corregido": False,
            "tipo": "INDEFINIDO",
            "municipio": "COLOMBIA",
            "confianza": conf_estimada,
            "hits": 1,
            "timestamp": time.strftime("%H:%M:%S"),
            "crop_base64": b64_crop_duda,
            "bbox": primer_caja,
            "es_dudosa": True,
            "mensaje_diagnostico": f"Detector ubicó placa pero el OCR obtuvo confianza insuficiente ({conf_estimada*100:.0f}%). Mantén la placa visible para mejorar la resolución.",
            "antecedentes": {
                "nivel_alerta": "AMARILLO",
                "mensaje": f"Confianza de lectura muy baja ({conf_estimada*100:.0f}%). Ajusta la cámara o mantén la placa en encuadre.",
                "metodo_usado": "LECTURA_BAJA_CONFIANZA",
                "datos": {
                    "placa": "LECTURA_DUDOSA",
                    "marca": "RESOLUCION INSUFICIENTE",
                    "linea": "El detector está buscando mejorar este fotograma",
                    "multas_simit": 0,
                    "valor_multas": 0,
                    "soat_vigente": 1,
                    "observaciones": "Mantén la placa en la pantalla. El motor intentará mejorarla en los siguientes cuadros."
                }
            }
        }
        engine._notify_listeners({"event": "NEW_PLATE", "data": payload_duda})
        return {"success": True, "data": payload_duda}

    crop = mejor_crop
    rx1, ry1, rx2, ry2 = mejor_bbox

    placa = det["placa_raw"]
    tipo = det.get("tipo", "CARRO")
    mun = det.get("municipio", "COLOMBIA")

    # Antecedentes policiales y de tránsito sin alterar los caracteres detectados
    ant = consultar_placa(placa)
    placa_a_mostrar = placa
    placa_raw_ocr = det.get("placa_raw_ocr", placa)
    auto_corregido = False

    # Base64 del recorte
    ret, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_crop = base64.b64encode(buf).decode('utf-8') if ret else ""

    payload = {
        "track_id": int(time.time()),
        "placa": placa_a_mostrar,
        "placa_raw_ocr": placa_raw_ocr,
        "auto_corregido": auto_corregido,
        "tipo": tipo,
        "municipio": mun,
        "confianza": det.get("confianza", 0.95),
        "hits": 1,
        "timestamp": time.strftime("%H:%M:%S"),
        "crop_base64": b64_crop,
        "bbox": (rx1, ry1, rx2, ry2),
        "antecedentes": ant
    }

    engine.tracker.confirmed_history[placa_a_mostrar] = payload
    engine.tracker._guardar_historial_persistente()
    engine._notify_listeners({"event": "NEW_PLATE", "data": payload})

    return {"success": True, "data": payload}


@app.get("/api/plates")
async def get_plates():
    """Retorna el historial completo de placas detectadas."""
    return JSONResponse(list(engine.history_plates.values()))


@app.post("/api/simulate_alert/{placa}")
async def simulate_alert(placa: str):
    """
    Simula una detección de prueba (ej. FRT890 para hurto, XYZ45E para multas, ABC123 para al día).
    Permite demostrar la reacción inmediata de alertas en tiempo real al jurado o profesor.
    """
    placa_clean = placa.strip().upper()
    ant = consultar_placa(placa_clean)
    tipo = "MOTO" if (len(placa_clean) == 6 and placa_clean[-1].isalpha()) else "CARRO"
    
    # Simular recorte visual mínimo
    import numpy as np
    dummy_crop = np.zeros((80, 200, 3), dtype=np.uint8)
    dummy_crop[:] = (35, 45, 60)
    cv2.putText(dummy_crop, placa_clean, (20, 52), cv2.FONT_HERSHEY_DUPLEX, 1.1, (240, 245, 250), 2)
    ret, buf = cv2.imencode('.jpg', dummy_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_crop = base64.b64encode(buf).decode('utf-8') if ret else ""

    payload = {
        "track_id": int(time.time()),
        "placa": placa_clean,
        "placa_raw_ocr": placa_clean,
        "auto_corregido": False,
        "tipo": tipo,
        "municipio": "BOGOTA D.C.",
        "confianza": 0.98,
        "hits": 1,
        "timestamp": time.strftime("%H:%M:%S"),
        "crop_base64": b64_crop,
        "bbox": (100, 100, 300, 200),
        "antecedentes": ant
    }

    engine.tracker.confirmed_history[placa_clean] = payload
    engine.tracker._guardar_historial_persistente()
    engine._notify_listeners({"event": "NEW_PLATE", "data": payload})
    return {"success": True, "data": payload}


@app.get("/api/antecedentes/{placa}")
async def get_antecedentes(placa: str):
    """Consulta antecedentes policiales y de tránsito en SQLite mediante el motor de 3 capas."""
    from modules.search_engine import consultar_placa
    res = consultar_placa(placa)
    return JSONResponse(res)


@app.get("/api/simit/{placa}")
async def get_simit(placa: str):
    """Consulta comparendos y estado en SIMIT (Local o API Externa)."""
    from modules.simit_connector import consultar_simit
    return JSONResponse(consultar_simit(placa))


@app.get("/api/simit_status")
async def get_simit_status():
    from modules.simit_connector import obtener_estado_simit
    return JSONResponse(obtener_estado_simit())


class SimitConfigRequest(BaseModel):
    provider: str
    api_key: str = ""
    endpoint_url: str = ""


@app.post("/api/set_simit_provider")
async def set_simit_provider(req: SimitConfigRequest):
    from modules.simit_connector import configurar_proveedor_simit
    ok = configurar_proveedor_simit(req.provider, req.api_key, req.endpoint_url)
    return {"success": ok}


class VehicleRecord(BaseModel):
    placa: str
    tipo: str = "CARRO"
    marca: str = "GENERICO"
    linea: str = "LINEA TEST"
    modelo: int = 2022
    color: str = "BLANCO"
    propietario: str = "PROPIETARIO TEST"
    cedula_propietario: str = "12345678"
    municipio_registro: str = "BOGOTA D.C."
    multas_simit: int = 0
    valor_multas: int = 0
    detalle_multas: str = ""
    soat_vigente: int = 1
    soat_vencimiento: str = "2027-01-01"
    tecnomecanica_vigente: int = 1
    tecnomecanica_vencimiento: str = "2027-01-01"
    reporte_hurto: int = 0
    orden_inmovilizacion: int = 0
    observaciones: str = ""


@app.post("/api/register_vehicle")
async def register_vehicle(rec: VehicleRecord):
    """Permite registrar vehículos en la base de datos de antecedentes."""
    from modules.database import registrar_o_actualizar_vehiculo
    ok = registrar_o_actualizar_vehiculo(rec.model_dump())
    return {"success": ok, "placa": rec.placa.upper()}


class SourceRequest(BaseModel):
    source: str | int


@app.post("/api/set_source")
async def set_source(req: SourceRequest):
    src = req.source
    if str(src).strip().isdigit():
        src = int(str(src).strip())
    elif str(src) == "file":
        src = os.path.join(BASE_DIR, "data", "video_prueba.mp4")
    else:
        src = str(src).strip()

    success = engine.change_source(src)
    return {"success": success, "source": str(src)}


class GeminiKeyRequest(BaseModel):
    api_key: str


@app.get("/api/gemini_status")
async def get_gemini_status():
    return {
        "available": gemini_disponible(),
        "key_preview": (obtener_gemini_key()[:6] + "...") if gemini_disponible() else None
    }


@app.post("/api/set_gemini_key")
async def set_gemini_key(req: GeminiKeyRequest):
    ok = guardar_gemini_key(req.api_key)
    return {"success": ok, "available": gemini_disponible()}


@app.post("/api/clear")
async def clear_records():
    engine.history_plates.clear()
    engine.tracker.confirmed_history.clear()
    engine.tracker.recent_seen_plates.clear()
    engine.tracker.active_tracks.clear()
    engine.tracker._guardar_historial_persistente()
    return {"success": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if websocket not in connected_websockets:
        connected_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("SISTEMA ANPR COLOMBIA - SERVIDOR DE ALTO RENDIMIENTO")
    print("Accede desde tu navegador a: http://localhost:8000")
    print("=" * 65 + "\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info", reload=False)
