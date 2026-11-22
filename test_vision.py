"""
Script de prueba y demostración para el Módulo de Visión y OCR (Persona A).
Soporta:
1. Procesamiento de archivos de video locales (.mp4, .avi, etc.).
2. Captura en TIEMPO REAL desde Webcam / Cámara / Stream RTSP.
3. Visualización interactiva en tiempo real con recuadros, etiquetas de placa,
   tipo de vehículo (Carro / Moto) y municipio inferior.
4. Muestra por consola las placas detectadas con su porcentaje de confianza.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Asegurar que el directorio raíz del proyecto esté en el sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.vision_ocr import analizar_frame


def generar_video_sintetico(ruta_destino: str, fps: int = 30, duracion_seg: int = 5):
    """
    Genera un video de prueba sintético con placas colombianas de Carro y Moto
    con municipio en la parte inferior para pruebas locales inmediatas.
    """
    print(f"\n[INFO] Generando video sintético con placas colombianas (Carro y Moto) en: {ruta_destino}...")
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)

    ancho, alto = 640, 480
    total_frames = fps * duracion_seg

    try:
        import imageio
        writer = imageio.get_writer(ruta_destino, fps=fps)
        usa_imageio = True
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(ruta_destino, fourcc, fps, (ancho, alto))
        usa_imageio = False

    vehiculos_demo = [
        # Carro colombiano: 3 letras + 3 números + Bogotá D.C.
        {
            "placa": "ABC 123",
            "municipio": "BOGOTA D.C.",
            "tipo": "CARRO",
            "color_vehiculo": (30, 30, 180),
            "inicio": 0,
            "fin": total_frames // 2
        },
        # Moto colombiana: 3 letras + 2 números + 1 letra + Medellín
        {
            "placa": "XYZ 45E",
            "municipio": "MEDELLIN",
            "tipo": "MOTO",
            "color_vehiculo": (20, 120, 20),
            "inicio": total_frames // 2,
            "fin": total_frames
        }
    ]

    for frame_idx in range(total_frames):
        # Asfalto de la vía
        frame = np.full((alto, ancho, 3), (55, 55, 55), dtype=np.uint8)
        cv2.line(frame, (0, 420), (ancho, 420), (255, 255, 255), 2)
        cv2.line(frame, (0, 80), (ancho, 80), (255, 255, 255), 2)

        for v in vehiculos_demo:
            if v["inicio"] <= frame_idx < v["fin"]:
                rel = (frame_idx - v["inicio"]) / float(v["fin"] - v["inicio"])
                x_pos = int(40 + rel * (ancho - 320))
                y_pos = 170

                if v["tipo"] == "CARRO":
                    # Silueta de carro
                    cv2.rectangle(frame, (x_pos - 20, y_pos - 70), (x_pos + 260, y_pos + 130), v["color_vehiculo"], -1)
                    cv2.rectangle(frame, (x_pos - 20, y_pos - 70), (x_pos + 260, y_pos + 130), (10, 10, 10), 2)
                    # Placa amarilla con municipio abajo
                    px1, py1 = x_pos + 20, y_pos + 10
                    px2, py2 = px1 + 200, py1 + 80
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 204, 255), -1)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 0), 2)
                    cv2.putText(frame, v["placa"], (px1 + 15, py1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, v["municipio"], (px1 + 35, py1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    # Silueta de moto
                    cv2.rectangle(frame, (x_pos + 20, y_pos - 60), (x_pos + 180, y_pos + 120), v["color_vehiculo"], -1)
                    cv2.rectangle(frame, (x_pos + 20, y_pos - 60), (x_pos + 180, y_pos + 120), (10, 10, 10), 2)
                    # Placa amarilla de moto (más compacta)
                    px1, py1 = x_pos + 25, y_pos + 20
                    px2, py2 = px1 + 150, py1 + 80
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 204, 255), -1)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 0), 2)
                    cv2.putText(frame, v["placa"], (px1 + 10, py1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, v["municipio"], (px1 + 22, py1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)

        if usa_imageio:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            writer.write(frame)

    if usa_imageio:
        writer.close()
    else:
        writer.release()
    print("[INFO] Video de demostración generado exitosamente.\n")


def ejecutar_reconocimiento(origen_video, es_camara: bool = False, cada_n_frames: int = 5, mostrar_gui: bool = False):
    """
    Ejecuta el reconocimiento de placas en video local o en tiempo real con cámara web.
    """
    if not es_camara and not os.path.exists(origen_video):
        print(f"[AVISO] No se encontró el video en '{origen_video}'.")
        generar_video_sintetico(origen_video)

    cap = cv2.VideoCapture(int(origen_video) if es_camara and str(origen_video).isdigit() else origen_video)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente de video: {origen_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not es_camara else -1
    modo_str = "TIEMPO REAL (Cámara Web)" if es_camara else "VIDEO LOCAL PREGRABADO"

    print("=" * 75)
    print("SISTEMA INTELIGENTE DE RECONOCIMIENTO DE PLACAS - PERSONA A")
    print(f"Modo de Operación   : {modo_str}")
    print(f"Origen              : {origen_video}")
    if not es_camara and total_frames > 0:
        duracion_seg = total_frames / fps
        print(f"Total de cuadros    : {total_frames} frames ({duracion_seg:.1f} seg @ {fps:.1f} FPS)")
    print(f"Tasa de muestreo    : Analizando 1 de cada {cada_n_frames} frames para fluidez")
    if mostrar_gui:
        print("Ventana visual      : ACTIVA (Presiona 'q' o 'ESC' en la ventana para salir)")
    print("=" * 75)
    print(f"{'FRAME':<8} | {'TIEMPO':<8} | {'TIPO':<7} | {'PLACA':<10} | {'CONFIANZA':<10} | {'MUNICIPIO':<12} | {'BBOX'}")
    print("-" * 75)

    detecciones_consolidadas = {}
    frames_totales = 0
    frames_analizados = 0
    ultimas_detecciones = []
    tiempo_inicio = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_totales += 1

            # Calcular timestamp
            if es_camara:
                segundos_transcurridos = time.time() - tiempo_inicio
            else:
                segundos_transcurridos = frames_totales / fps

            minutos = int(segundos_transcurridos // 60)
            segundos = segundos_transcurridos % 60
            timestamp_str = f"{minutos:02d}:{segundos:05.2f}"

            # Analizar frame cada N cuadros para mantener alta tasa de FPS
            if frames_totales % cada_n_frames == 0 or frames_totales == 1:
                frames_analizados += 1
                # Llamada al contrato estricto de Persona A
                ultimas_detecciones = analizar_frame(frame)

                for det in ultimas_detecciones:
                    placa = det["placa_raw"]
                    conf = det["confianza"]
                    tipo = det.get("tipo", "CARRO")
                    mun = det.get("municipio") or "N/D"
                    bbox = det["bbox"]
                    conf_pct = f"{conf * 100:.1f}%"

                    print(f"#{frames_totales:<7} | {timestamp_str:<8} | {tipo:<7} | {placa:<10} | {conf_pct:<10} | {mun:<12} | {bbox}")

                    # Consolidar estadísticas por placa única
                    if placa not in detecciones_consolidadas:
                        detecciones_consolidadas[placa] = {
                            "max_conf": conf,
                            "tipo": tipo,
                            "municipio": mun if mun != "N/D" else None,
                            "conteo": 1,
                            "primer_frame": frames_totales,
                            "ultimo_frame": frames_totales
                        }
                    else:
                        detecciones_consolidadas[placa]["conteo"] += 1
                        detecciones_consolidadas[placa]["ultimo_frame"] = frames_totales
                        if conf > detecciones_consolidadas[placa]["max_conf"]:
                            detecciones_consolidadas[placa]["max_conf"] = conf
                        if mun != "N/D" and not detecciones_consolidadas[placa]["municipio"]:
                            detecciones_consolidadas[placa]["municipio"] = mun

            # Visualización interactiva en tiempo real si está habilitada la GUI
            if mostrar_gui:
                frame_display = frame.copy()

                # Dibujar las detecciones sobre el fotograma actual
                for det in ultimas_detecciones:
                    x1, y1, x2, y2 = det["bbox"]
                    placa = det["placa_raw"]
                    conf_pct = f"{det['confianza'] * 100:.0f}%"
                    tipo = det.get("tipo", "CARRO")
                    mun = det.get("municipio")

                    color_box = (0, 220, 0) if tipo == "CARRO" else (255, 140, 0)
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color_box, 3)

                    # Etiqueta superior
                    texto_etiqueta = f"[{tipo}] {placa} ({conf_pct})"
                    if mun and mun != "N/D":
                        texto_etiqueta += f" - {mun}"

                    cv2.putText(
                        frame_display,
                        texto_etiqueta,
                        (max(0, x1), max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        frame_display,
                        texto_etiqueta,
                        (max(0, x1), max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                # Información de FPS y cuadros en pantalla
                fps_en_vivo = frames_totales / max(0.001, time.time() - tiempo_inicio)
                cv2.putText(
                    frame_display,
                    f"Modo: {modo_str} | FPS: {fps_en_vivo:.1f} | Frame: {frames_totales}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow("Sistema de Reconocimiento de Placas - Persona A", frame_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' o ESC
                    print("\n[INFO] Ejecución detenida por el usuario.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Detenido por combinación Ctrl+C.")
    finally:
        cap.release()
        if mostrar_gui:
            cv2.destroyAllWindows()

    tiempo_total = max(0.001, time.time() - tiempo_inicio)
    fps_promedio = frames_analizados / tiempo_total

    print("=" * 75)
    print("RESUMEN FINAL DE DETECCIONES")
    print("=" * 75)
    print(f"Frames leídos       : {frames_totales}")
    print(f"Frames analizados   : {frames_analizados}")
    print(f"Tiempo de cómputo   : {tiempo_total:.2f} s ({fps_promedio:.1f} frames analizados/segundo)")
    print(f"Placas únicas       : {len(detecciones_consolidadas)}")
    print("-" * 75)

    if detecciones_consolidadas:
        print(f"{'PLACA':<10} | {'TIPO':<7} | {'MÁX. CONF':<11} | {'MUNICIPIO':<14} | {'APARICIONES':<12} | {'RANGO'}")
        print("-" * 75)
        for placa, d in sorted(detecciones_consolidadas.items(), key=lambda x: x[1]["max_conf"], reverse=True):
            pct = f"{d['max_conf'] * 100:.1f}%"
            mun = d["municipio"] or "N/D"
            rango = f"#{d['primer_frame']} -> #{d['ultimo_frame']}"
            print(f"{placa:<10} | {d['tipo']:<7} | {pct:<11} | {mun:<14} | {d['conteo']:<12} | {rango}")
    else:
        print("[!] No se detectaron placas en la sesión de video.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Reconocimiento de Placas Colombianas (Persona A) - Video y Tiempo Real"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=os.path.join(BASE_DIR, "data", "video_prueba.mp4"),
        help="Ruta al video local (por defecto: data/video_prueba.mp4)"
    )
    parser.add_argument(
        "--camera",
        "--cam",
        type=str,
        default=None,
        help="Índice de cámara web para tiempo real (ej: 0, 1) o URL de stream RTSP"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Tasa de muestreo: analizar 1 de cada N frames (por defecto: 5)"
    )
    parser.add_argument(
        "--mostrar",
        "--live",
        action="store_true",
        help="Abrir ventana visual interactiva de OpenCV en tiempo real"
    )

    args = parser.parse_args()

    if args.camera is not None:
        ejecutar_reconocimiento(
            args.camera,
            es_camara=True,
            cada_n_frames=args.step,
            mostrar_gui=args.mostrar
        )
    else:
        ejecutar_reconocimiento(
            args.video,
            es_camara=False,
            cada_n_frames=args.step,
            mostrar_gui=args.mostrar
        )


if __name__ == "__main__":
    main()
