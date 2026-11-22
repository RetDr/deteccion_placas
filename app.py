"""
Aplicación Web Interactiva - Sistema Inteligente de Reconocimiento de Placas (MVP)
Permite procesar videos locales, videos subidos por el usuario o Cámara Web en tiempo real.
Integra directamente el módulo de Persona A (modules/vision_ocr.py).
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
import streamlit as st

# Asegurar que el directorio raíz esté en sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.vision_ocr import analizar_frame

st.set_page_config(
    page_title="Reconocimiento de Placas Colombianas",
    page_icon="🚗",
    layout="wide"
)

# Estilos visuales para resaltar alertas y etiquetas de placas
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222A;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 5px solid #00C853;
    }
    .plate-badge {
        background-color: #FFD600;
        color: #000000;
        font-family: monospace;
        font-weight: bold;
        font-size: 1.3rem;
        padding: 4px 12px;
        border-radius: 4px;
        border: 2px solid #000;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Sistema Inteligente de Reconocimiento de Placas")
st.caption("Reconocimiento de Placas Colombianas (Carros y Motos) en Video y Tiempo Real")

# Barra lateral de configuración
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")

    modo_origen = st.radio(
        "Fuente de Video:",
        [
            "📁 Video local por defecto (data/video_prueba.mp4)",
            "📤 Subir archivo de video propio",
            "📹 Cámara Web en Tiempo Real"
        ]
    )

    tasa_muestreo = st.slider(
        "Muestreo de frames (1 de cada N):",
        min_value=1,
        max_value=15,
        value=5,
        help="Procesar 1 de cada 5 o 10 frames optimiza la velocidad y fluidez en PC sin GPU"
    )

    umbral_conf = st.slider(
        "Umbral mínimo de confianza (%):",
        min_value=30,
        max_value=99,
        value=50,
        step=5
    ) / 100.0

    st.markdown("---")
    st.markdown("""
    **Formato Oficial Colombia:**
    * **Carros:** 3 letras + 3 números + Municipio abajo.
    * **Motos:** 3 letras + 2 números + 1 letra + 'COLOMBIA' abajo.
    """)

# Determinar origen del video
ruta_video = None
es_camara = False

if modo_origen == "📁 Video local por defecto (data/video_prueba.mp4)":
    default_path = os.path.join(BASE_DIR, "data", "video_prueba.mp4")
    if os.path.exists(default_path):
        ruta_video = default_path
    else:
        st.warning(f"No se encontró el video en `{default_path}`. Puedes generarlo o subir uno propio.")
elif modo_origen == "📤 Subir archivo de video propio":
    video_file = st.file_uploader("Selecciona un video (MP4, AVI, MOV):", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        ruta_video = tfile.name
elif modo_origen == "📹 Cámara Web en Tiempo Real":
    es_camara = True
    idx_cam = st.sidebar.number_input("Índice de la Cámara:", min_value=0, max_value=5, value=0)
    ruta_video = idx_cam

# Interfaz Principal
col_video, col_resultados = st.columns([3, 2])

with col_video:
    st.subheader("📺 Reproducción y Detección en Tiempo Real")
    placeholder_video = st.empty()

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        iniciar_btn = st.button("▶️ Iniciar Reconocimiento", type="primary", use_container_width=True)
    with col_btn2:
        detener_btn = st.button("⏹️ Detener", use_container_width=True)

    # Métricas en vivo
    col_m1, col_m2, col_m3 = st.columns(3)
    metric_fps = col_m1.empty()
    metric_frames = col_m2.empty()
    metric_placas = col_m3.empty()

with col_resultados:
    st.subheader("📋 Placas Detectadas")
    placeholder_alertas = st.empty()

# Control de ejecución
if iniciar_btn and (ruta_video is not None):
    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        st.error("No se pudo iniciar la captura de video.")
    else:
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        placas_detectadas_historial = {}
        frame_count = 0
        tiempo_inicio = time.time()
        ultimas_detecciones = []

        while cap.isOpened():
            if detener_btn:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Procesar OCR cada N frames
            if frame_count % tasa_muestreo == 0 or frame_count == 1:
                # LLAMADA AL CONTRATO DE PERSONA A
                detecciones = analizar_frame(frame)
                ultimas_detecciones = [d for d in detecciones if d["confianza"] >= umbral_conf]

                # Registrar detecciones en el historial
                for det in ultimas_detecciones:
                    placa = det["placa_raw"]
                    if placa not in placas_detectadas_historial:
                        placas_detectadas_historial[placa] = {
                            "datos": det,
                            "conteo": 1,
                            "timestamp": time.strftime("%H:%M:%S")
                        }
                    else:
                        placas_detectadas_historial[placa]["conteo"] += 1
                        if det["confianza"] > placas_detectadas_historial[placa]["datos"]["confianza"]:
                            placas_detectadas_historial[placa]["datos"] = det

            # Dibujar recuadros y etiquetas sobre el fotograma
            frame_draw = frame.copy()
            for det in ultimas_detecciones:
                x1, y1, x2, y2 = det["bbox"]
                tipo = det.get("tipo", "CARRO")
                placa = det["placa_raw"]
                mun = det.get("municipio")
                conf_pct = f"{det['confianza'] * 100:.0f}%"

                # Color: Verde para Carro, Naranja para Moto
                color = (0, 220, 0) if tipo == "CARRO" else (255, 140, 0)
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 3)

                texto = f"[{tipo}] {placa} ({conf_pct})"
                if mun:
                    texto += f" - {mun}"

                cv2.putText(frame_draw, texto, (max(0, x1), max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame_draw, texto, (max(0, x1), max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            # Convertir a RGB para Streamlit
            frame_rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
            placeholder_video.image(frame_rgb, channels="RGB", use_container_width=True)

            # Actualizar métricas
            elapsed = max(0.001, time.time() - tiempo_inicio)
            fps_actual = frame_count / elapsed
            metric_fps.metric("FPS Proceso", f"{fps_actual:.1f}")
            metric_frames.metric("Cuadro Actual", f"#{frame_count}")
            metric_placas.metric("Placas Únicas", len(placas_detectadas_historial))

            # Actualizar panel lateral con tarjetas de placas detectadas
            with placeholder_alertas.container():
                if placas_detectadas_historial:
                    for p_codigo, p_info in list(placas_detectadas_historial.items())[-5:]:
                        det_info = p_info["datos"]
                        col_card_img, col_card_txt = st.columns([1, 2])
                        with col_card_img:
                            recorte = det_info.get("recorte_img")
                            if recorte is not None and recorte.size > 0:
                                st.image(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with col_card_txt:
                            st.markdown(f"<div class='plate-badge'>{p_codigo}</div>", unsafe_allow_html=True)
                            tipo_ico = "🚗" if det_info.get("tipo") == "CARRO" else "🏍️"
                            st.write(f"{tipo_ico} **Tipo:** {det_info.get('tipo', 'CARRO')}")
                            if det_info.get("municipio"):
                                st.write(f"📍 **Ubicación:** {det_info['municipio']}")
                            st.write(f"📊 **Confianza:** {det_info['confianza'] * 100:.1f}% ({p_info['conteo']} lecturas)")
                        st.divider()
                else:
                    st.info("Buscando vehículos y placas en la transmisión...")

        cap.release()
        st.success(f"Procesamiento finalizado. Se detectaron {len(placas_detectadas_historial)} placas únicas.")
