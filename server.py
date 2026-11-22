"""
Servidor Web de Alto Rendimiento para el Sistema de Reconocimiento de Placas.
Construido sobre FastAPI + Uvicorn con transmisión MJPEG a 30+ FPS y WebSockets.
"""

import os
import sys
import time
import base64
import cv2
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.camera_stream import LiveCameraEngine, detectar_placas_por_color, aplicar_nms_cajas
from modules.vision_ocr import obtener_modelos
from modules.search_engine import consultar_placa
from modules.gemini_ocr import gemini_disponible, guardar_gemini_key, obtener_gemini_key, consultar_gemini_placa

# Inicializar motor de captura asíncrono
# Por defecto inicia con la cámara web 0 (o video de prueba si no hay cámara)
default_source = 0
engine = LiveCameraEngine(source=default_source, sampling_step=4, min_confidence=0.45)

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

    # Iniciar motor de cámara: probar cámara 0, 1 (celular USB / DroidCam / Iriun) o 2
    started = False
    for cam_idx in [0, 1, 2]:
        if engine.start(cam_idx):
            print(f"[INFO] Camara iniciada con exito en indice: {cam_idx}")
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

    # 1. Detección YOLO
    res = plate_detector(frame, verbose=False, conf=0.15)
    for r in res:
        for box in r.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            pw = bx2 - bx1
            ph = by2 - by1
            pad_x = int(pw * 0.16)
            pad_y = int(ph * 0.12)
            cajas.append((max(0, bx1 - pad_x), max(0, by1 - pad_y), min(w_f, bx2 + pad_x), min(h_f, by2 + pad_y)))

    # 2. Fallback por descomposición de color HSV si YOLO no detectó nada
    if not cajas:
        cajas = detectar_placas_por_color(frame)

    # 3. Fallback de encuadre central si todo lo anterior no encontró caja
    if not cajas:
        rx1, ry1 = int(w_f * 0.15), int(h_f * 0.20)
        rx2, ry2 = int(w_f * 0.85), int(h_f * 0.80)
        cajas.append((rx1, ry1, rx2, ry2))

    # Filtrar cajas con NMS
    cajas = aplicar_nms_cajas(cajas, iou_threshold=0.20)
    rx1, ry1, rx2, ry2 = cajas[0]
    crop = frame[ry1:ry2, rx1:rx2]

    # Ejecutar OCR (Gemini si está configurado, o EasyOCR local)
    det = None
    if gemini_disponible():
        det = consultar_gemini_placa(crop)

    if not det:
        ocr_res = engine._ejecutar_ocr_en_recorte(crop, reader)
        if ocr_res:
            det = ocr_res[0]

    if not det or not det.get("placa_raw") or det["placa_raw"] == "DESCONOCIDO":
        return JSONResponse({"success": False, "error": "No se pudo leer la placa con claridad en esta toma. Acerca la placa o ajusta la iluminación."})

    placa = det["placa_raw"]
    tipo = det.get("tipo", "CARRO")
    mun = det.get("municipio", "COLOMBIA")

    # Antecedentes policiales y de tránsito
    ant = consultar_placa(placa)

    # Base64 del recorte
    ret, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_crop = base64.b64encode(buf).decode('utf-8') if ret else ""

    payload = {
        "track_id": int(time.time()),
        "placa": placa,
        "tipo": tipo,
        "municipio": mun,
        "confianza": det.get("confianza", 0.95),
        "hits": 1,
        "timestamp": time.strftime("%H:%M:%S"),
        "crop_base64": b64_crop,
        "bbox": (rx1, ry1, rx2, ry2),
        "antecedentes": ant
    }

    engine.tracker.confirmed_history[placa] = payload
    engine.tracker._guardar_historial_persistente()
    engine._notify_listeners({"event": "NEW_PLATE", "data": payload})

    return {"success": True, "data": payload}


@app.get("/api/plates")
async def get_plates():
    """Retorna el historial completo de placas detectadas."""
    return JSONResponse(list(engine.history_plates.values()))


@app.get("/api/antecedentes/{placa}")
async def get_antecedentes(placa: str):
    """Consulta antecedentes policiales y de tránsito en SQLite mediante el motor de 3 capas."""
    from modules.search_engine import consultar_placa
    res = consultar_placa(placa)
    return JSONResponse(res)


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
