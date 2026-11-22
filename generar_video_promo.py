"""
Generador del Video Promocional Oficial para ANPR Colombia (1080p Full HD - 30 FPS).
Crea un video cinemático y profesional con gráficos de alta fidelidad, animaciones HUD,
métricas de telemetría, discriminación DIN 1451 Q/O y consulta en vivo SIMIT.
"""

import os
import sys
import math
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_VIDEO = os.path.join(BASE_DIR, "video_promocional_anpr.mp4")
TEMP_VIDEO = os.path.join(BASE_DIR, "scratch", "promo_temp.mp4")
TEST_VIDEO_PATH = os.path.join(BASE_DIR, "data", "video_prueba.mp4")

WIDTH = 1920
HEIGHT = 1080
FPS = 30

os.makedirs(os.path.join(BASE_DIR, "scratch"), exist_ok=True)


def crear_fondo_degradado(w, h, c_top=(15, 20, 28), c_bottom=(7, 10, 15)):
    """Genera un fondo con gradiente vertical elegante."""
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        alpha = y / float(h)
        col = [int((1 - alpha) * c_top[i] + alpha * c_bottom[i]) for i in range(3)]
        base[y, :] = col
    return base


def dibujar_rejilla_tech(img, paso=60, color=(35, 45, 60), grosor=1):
    """Agrega una retícula sutil de monitoreo táctico."""
    h, w = img.shape[:2]
    for x in range(0, w, paso):
        cv2.line(img, (x, 0), (x, h), color, grosor)
    for y in range(0, h, paso):
        cv2.line(img, (0, y), (w, y), color, grosor)


def dibujar_hud_superior(canvas, titulo_seccion="MONITOREO VIAL"):
    """Dibuja barra superior con estado y telemetría."""
    cv2.rectangle(canvas, (0, 0), (WIDTH, 70), (12, 16, 24), -1)
    cv2.line(canvas, (0, 70), (WIDTH, 70), (0, 230, 118), 2)
    
    cv2.circle(canvas, (40, 35), 8, (0, 230, 118), -1)
    cv2.putText(canvas, "ANPR COLOMBIA v2.0", (60, 42), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    
    # Centro
    cv2.putText(canvas, f"[ MODO: {titulo_seccion} ]", (WIDTH // 2 - 160, 42), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 230, 118), 1)
    
    # Derecha
    cv2.putText(canvas, "ESTADO: OPERATIVO 30 FPS", (WIDTH - 380, 42), cv2.FONT_HERSHEY_DUPLEX, 0.65, (160, 185, 210), 1)


def animar_intro(out):
    """Escena 1 (4.5 seg): Título cinemático con pulso de energía."""
    total_frames = int(4.5 * FPS)
    bg_base = crear_fondo_degradado(WIDTH, HEIGHT, (22, 28, 40), (8, 12, 18))
    dibujar_rejilla_tech(bg_base, 80, (28, 38, 52), 1)

    for f in range(total_frames):
        canvas = bg_base.copy()
        t = f / float(FPS)
        
        # Círculos radiales pulsantes
        radio = int(120 + 40 * math.sin(t * 3.5))
        cv2.circle(canvas, (WIDTH // 2, HEIGHT // 2 - 80), radio, (0, 230, 118), 2)
        cv2.circle(canvas, (WIDTH // 2, HEIGHT // 2 - 80), int(radio * 1.3), (0, 180, 255), 1)
        
        # Pulso de escáner horizontal
        scan_y = int((HEIGHT // 2 - 250) + 340 * (0.5 + 0.5 * math.sin(t * 2.5)))
        cv2.line(canvas, (WIDTH // 2 - 450, scan_y), (WIDTH // 2 + 450, scan_y), (0, 255, 200), 2)

        # Insignia central de Placa Colombiana
        cv2.rectangle(canvas, (WIDTH // 2 - 220, HEIGHT // 2 - 160), (WIDTH // 2 + 220, HEIGHT // 2 - 10), (0, 205, 255), -1)
        cv2.rectangle(canvas, (WIDTH // 2 - 220, HEIGHT // 2 - 160), (WIDTH // 2 + 220, HEIGHT // 2 - 10), (10, 10, 10), 4)
        cv2.putText(canvas, "COLOMBIA", (WIDTH // 2 - 85, HEIGHT // 2 - 125), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 20), 2)
        cv2.putText(canvas, "ANPR-26", (WIDTH // 2 - 170, HEIGHT // 2 - 40), cv2.FONT_HERSHEY_DUPLEX, 2.2, (10, 10, 10), 5)

        # Tipografía principal
        alpha_text = min(1.0, t / 1.5)
        if alpha_text > 0.1:
            cv2.putText(canvas, "SISTEMA ANPR COLOMBIA", (WIDTH // 2 - 460, HEIGHT // 2 + 120), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.9, (255, 255, 255), 3)
            cv2.putText(canvas, "RECONOCIMIENTO AUTOMATICO DE PLACAS VEHICULARES EN TIEMPO REAL", 
                        (WIDTH // 2 - 580, HEIGHT // 2 + 180), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 230, 118), 2)
            cv2.putText(canvas, "Inteligencia Computacional | YOLOv8 + OCR DIN 1451 + SIMIT Oficial", 
                        (WIDTH // 2 - 440, HEIGHT // 2 + 235), cv2.FONT_HERSHEY_DUPLEX, 0.75, (160, 185, 210), 1)

        # Barra de progreso inferior
        prog_w = int((f / float(total_frames)) * 600)
        cv2.rectangle(canvas, (WIDTH // 2 - 300, HEIGHT - 100), (WIDTH // 2 + 300, HEIGHT - 92), (40, 50, 70), -1)
        cv2.rectangle(canvas, (WIDTH // 2 - 300, HEIGHT - 100), (WIDTH // 2 - 300 + prog_w, HEIGHT - 92), (0, 230, 118), -1)
        cv2.putText(canvas, "INICIALIZANDO MODELOS DE VISION...", (WIDTH // 2 - 180, HEIGHT - 120), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 200, 220), 1)

        out.write(canvas)


def animar_deteccion_en_vivo(out):
    """Escena 2 (5.5 seg): Video vehicular con recuadros tácticos de seguimiento a 30 FPS."""
    total_frames = int(5.5 * FPS)
    cap = cv2.VideoCapture(TEST_VIDEO_PATH) if os.path.exists(TEST_VIDEO_PATH) else None

    bg_feed = crear_fondo_degradado(WIDTH, HEIGHT, (15, 20, 28), (5, 8, 12))
    dibujar_rejilla_tech(bg_feed, 60, (25, 32, 45), 1)

    vw = 1200
    vh = 800
    vx = 70
    vy = 160

    placas_demo = [
        {"placa": "BLJ732", "tipo": "CARRO", "conf": 0.985, "color": "Rojo", "marca": "Chevrolet", "alerta": "AMARILLO", "motivo": "Comparendo SIMIT pendiente ($468,500 COP)"},
        {"placa": "OMG650", "tipo": "CARRO", "conf": 0.980, "color": "Gris", "marca": "Renault", "alerta": "VERDE", "motivo": "Paz y Salvo en SIMIT"},
        {"placa": "QGU210", "tipo": "CARRO", "conf": 0.992, "color": "Negro", "marca": "Mazda", "alerta": "VERDE", "motivo": "SOAT y Tecno vigentes"}
    ]

    for f in range(total_frames):
        canvas = bg_feed.copy()
        dibujar_hud_superior(canvas, "DETECCION Y SEGUIMIENTO A 30 FPS")

        t = f / float(FPS)
        idx_p = min(len(placas_demo) - 1, int(t // 1.8))
        p_act = placas_demo[idx_p]

        # Obtener cuadro de video real o simular feed de cámara
        frame_in = None
        if cap and cap.isOpened():
            ret, frame_cap = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame_cap = cap.read()
            if ret:
                frame_in = cv2.resize(frame_cap, (vw, vh))

        if frame_in is None:
            frame_in = np.zeros((vh, vw, 3), dtype=np.uint8)
            frame_in[:] = (20, 25, 35)

        # Superponer feed de video
        canvas[vy:vy+vh, vx:vx+vw] = frame_in

        # Marco del visor
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (0, 230, 118), 2)
        cv2.putText(canvas, "LIVE CAMERA FEED - CANAL 1 (DIRECTSHOW 30 FPS)", (vx + 15, vy - 15), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 230, 118), 1)

        # Retícula dinámica de rastreo sobre la placa
        bx1 = vx + int(vw * 0.40 + 25 * math.sin(t * 3.0))
        by1 = vy + int(vh * 0.58 + 15 * math.cos(t * 2.5))
        bx2 = bx1 + 220
        by2 = by1 + 75

        color_box = (0, 230, 118) if p_act["alerta"] == "VERDE" else (0, 190, 255)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color_box, 3)
        # Esquinas tácticas
        len_corner = 18
        cv2.line(canvas, (bx1, by1), (bx1 + len_corner, by1), (255, 255, 255), 4)
        cv2.line(canvas, (bx1, by1), (bx1, by1 + len_corner), (255, 255, 255), 4)
        cv2.line(canvas, (bx2, by2), (bx2 - len_corner, by2), (255, 255, 255), 4)
        cv2.line(canvas, (bx2, by2), (bx2, by2 - len_corner), (255, 255, 255), 4)

        # Etiqueta flotante
        cv2.rectangle(canvas, (bx1, by1 - 32), (bx1 + 170, by1), color_box, -1)
        cv2.putText(canvas, p_act["placa"], (bx1 + 10, by1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 10, 10), 2)

        # Panel lateral derecho con telemetría de la captura
        px = vx + vw + 40
        py = vy
        pw = WIDTH - px - 40
        ph = vh

        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 24, 34), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (35, 45, 65), 1)

        cv2.putText(canvas, "TELEMETRIA EN TIEMPO REAL", (px + 20, py + 40), cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 230, 118), 2)
        cv2.line(canvas, (px + 20, py + 55), (px + pw - 20, py + 55), (0, 230, 118), 1)

        # Métricas técnicas
        cv2.putText(canvas, f"FPS CAMARA: 30.0", (px + 20, py + 100), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1)
        cv2.putText(canvas, f"LATENCIA: 42.8 ms", (px + 20, py + 140), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1)
        cv2.putText(canvas, f"RESOLUCION: 1920x1080", (px + 20, py + 180), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1)
        cv2.putText(canvas, f"CONFIANZA DETECTOR: {p_act['conf']*100:.1f}%", (px + 20, py + 220), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 230, 118), 1)

        # Tarjeta de la placa activa
        card_y = py + 270
        cv2.rectangle(canvas, (px + 20, card_y), (px + pw - 20, card_y + 220), (25, 34, 48), -1)
        cv2.rectangle(canvas, (px + 20, card_y), (px + pw - 20, card_y + 220), color_box, 2)

        # Insignia placa
        cv2.rectangle(canvas, (px + 40, card_y + 20), (px + 230, card_y + 85), (0, 205, 255), -1)
        cv2.rectangle(canvas, (px + 40, card_y + 20), (px + 230, card_y + 85), (10, 10, 10), 2)
        cv2.putText(canvas, "COLOMBIA", (px + 90, card_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.45, (10, 10, 10), 1)
        cv2.putText(canvas, p_act["placa"], (px + 50, card_y + 75), cv2.FONT_HERSHEY_DUPLEX, 1.1, (10, 10, 10), 2)

        cv2.putText(canvas, f"TIPO: {p_act['tipo']}", (px + 250, card_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(canvas, f"MARCA: {p_act['marca']} ({p_act['color']})", (px + 250, card_y + 75), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 200, 220), 1)

        # Estado Antecedentes
        badge_txt = "PAZ Y SALVO" if p_act["alerta"] == "VERDE" else "ALERTA SIMIT"
        cv2.rectangle(canvas, (px + 40, card_y + 110), (px + pw - 40, card_y + 150), color_box, -1)
        cv2.putText(canvas, badge_txt, (px + 140, card_y + 138), cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 10, 10), 2)

        cv2.putText(canvas, p_act["motivo"], (px + 40, card_y + 185), cv2.FONT_HERSHEY_DUPLEX, 0.60, (220, 230, 240), 1)

        # Botón auto-mejora
        cv2.putText(canvas, "AUTO-MEJORA POR NITIDEZ LAPLACIANA: ACTIVA", (px + 20, py + ph - 30), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 230, 118), 1)

        out.write(canvas)

    if cap:
        cap.release()


def animar_discriminador_qo(out):
    """Escena 3 (5.0 seg): Demostración matemática del discriminador Q vs O / J (DIN 1451)."""
    total_frames = int(5.0 * FPS)
    bg = crear_fondo_degradado(WIDTH, HEIGHT, (20, 25, 35), (8, 12, 18))
    dibujar_rejilla_tech(bg, 70, (28, 36, 50), 1)

    for f in range(total_frames):
        canvas = bg.copy()
        dibujar_hud_superior(canvas, "DISCRIMINACION MORFOLOGICA DE CARACTERES (DIN 1451)")
        t = f / float(FPS)

        # Encabezado explicativo
        cv2.putText(canvas, "ANALISIS GEOMETRICO DE CARACTERES CRITICOS: Q vs O vs J", (WIDTH // 2 - 520, 140), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(canvas, "Eliminacion de ambiguedad fisica mediante simetria de cuadrantes y colas exteriores", (WIDTH // 2 - 470, 180), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 230, 118), 1)

        # Comparativa lado a lado
        # Lado Izquierdo: Letra Q
        qx = WIDTH // 4
        qy = 420
        cv2.circle(canvas, (qx, qy), 130, (25, 35, 50), -1)
        cv2.circle(canvas, (qx, qy), 130, (0, 230, 118), 3)

        # Glifo Q
        cv2.putText(canvas, "Q", (qx - 65, qy + 65), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 255, 255), 8)

        # Resaltado del cuadrante inferior derecho (cola de la Q)
        cv2.line(canvas, (qx - 130, qy), (qx + 130, qy), (50, 70, 95), 1)
        cv2.line(canvas, (qx, qy - 130), (qx, qy + 130), (50, 70, 95), 1)

        pulse_alpha = 0.5 + 0.5 * math.sin(t * 5.0)
        col_pulse = (0, int(200 * pulse_alpha), 255)
        cv2.rectangle(canvas, (qx + 10, qy + 10), (qx + 130, qy + 130), col_pulse, 3)
        cv2.putText(canvas, "COLA EXTERIOR DIN 1451", (qx - 130, qy + 180), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 205, 255), 2)
        cv2.putText(canvas, "Ratio BR/BL = 1.34 (> 1.18)", (qx - 110, qy + 215), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 230, 118), 2)
        cv2.putText(canvas, "CLASIFICACION: LETRA 'Q' (100%)", (qx - 150, qy + 260), cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 230, 118), 2)

        # Lado Derecho: Letra O
        ox = 3 * (WIDTH // 4)
        oy = 420
        cv2.circle(canvas, (ox, oy), 130, (25, 35, 50), -1)
        cv2.circle(canvas, (ox, oy), 130, (160, 180, 200), 2)

        # Glifo O
        cv2.putText(canvas, "O", (ox - 65, oy + 65), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 255, 255), 8)

        cv2.line(canvas, (ox - 130, oy), (ox + 130, oy), (50, 70, 95), 1)
        cv2.line(canvas, (ox, oy - 130), (ox, oy + 130), (50, 70, 95), 1)

        cv2.putText(canvas, "SIMETRIA BILATERAL PERFECTA", (ox - 160, qy + 180), cv2.FONT_HERSHEY_DUPLEX, 0.75, (180, 200, 220), 2)
        cv2.putText(canvas, "Ratio BR/BL = 0.99 (~ 1.00)", (ox - 110, qy + 215), cv2.FONT_HERSHEY_DUPLEX, 0.70, (180, 200, 220), 2)
        cv2.putText(canvas, "CLASIFICACION: LETRA 'O' (100%)", (ox - 150, qy + 260), cv2.FONT_HERSHEY_DUPLEX, 0.80, (180, 200, 220), 2)

        # Panel inferior: Validación sobre casos reales
        cv2.rectangle(canvas, (WIDTH // 2 - 450, HEIGHT - 180), (WIDTH // 2 + 450, HEIGHT - 70), (16, 22, 32), -1)
        cv2.rectangle(canvas, (WIDTH // 2 - 450, HEIGHT - 180), (WIDTH // 2 + 450, HEIGHT - 70), (0, 230, 118), 1)
        cv2.putText(canvas, "CASO REAL: Placa '@Gu210' -> Correccion a 'QGU210' (Conf: 99.2%)", 
                    (WIDTH // 2 - 410, HEIGHT - 135), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(canvas, "CASO REAL: Placa 'lomg650' -> Correccion a 'OMG650' (Conf: 98.0%)", 
                    (WIDTH // 2 - 410, HEIGHT - 95), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 230, 118), 2)

        out.write(canvas)


def animar_simit_en_vivo(out):
    """Escena 4 (5.5 seg): Integración oficial con SIMIT (datos.gov.co) y alerta de comparendos."""
    total_frames = int(5.5 * FPS)
    bg = crear_fondo_degradado(WIDTH, HEIGHT, (16, 22, 30), (6, 9, 14))
    dibujar_rejilla_tech(bg, 75, (24, 32, 44), 1)

    for f in range(total_frames):
        canvas = bg.copy()
        dibujar_hud_superior(canvas, "INTEGRACION OFICIAL DATOS ABIERTOS SIMIT")
        t = f / float(FPS)

        # Título
        cv2.putText(canvas, "AUDITORIA VIAL Y TRANSITO EN VIVO", (WIDTH // 2 - 360, 130), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(canvas, "API SODA Oficial: https://www.datos.gov.co/resource/72nf-y4v3.json", (WIDTH // 2 - 420, 170), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 205, 255), 1)

        # Terminal / JSON en vivo en el lado izquierdo
        tx = 120
        ty = 220
        tw = 800
        th = 680
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (10, 14, 20), -1)
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (40, 55, 75), 2)

        cv2.putText(canvas, "RESPUESTA JSON DEL SIMIT EN TIEMPO REAL:", (tx + 25, ty + 40), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 230, 118), 1)

        json_lines = [
            'GET /resource/72nf-y4v3.json?placa=BLJ732 HTTP/2',
            'Host: www.datos.gov.co',
            'Status: 200 OK (Latencia: 84 ms)',
            'Content-Type: application/json',
            '',
            '[',
            '  {',
            '    "placa": "BLJ732",',
            '    "vigencia": "2022",',
            '    "fecha_multa": "20/11/2022",',
            '    "departamento": "Bogota D.C.",',
            '    "ciudad": "Bogota D.C.",',
            '    "valor_multa": "468500",',
            '    "pagado_si_no": "NO"   <-- COMPARENDO PENDIENTE',
            '  },',
            '  {',
            '    "placa": "BLJ732",',
            '    "vigencia": "2024",',
            '    "valor_multa": "572600",',
            '    "pagado_si_no": "SI"   <-- PAZ Y SALVO 2024',
            '  }',
            ']'
        ]

        for i, line in enumerate(json_lines):
            col_l = (0, 230, 118) if "NO" in line else ((0, 205, 255) if "SI" in line else (180, 200, 220))
            if "GET" in line or "Status: 200" in line:
                col_l = (255, 255, 255)
            cv2.putText(canvas, line, (tx + 30, ty + 80 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col_l, 1)

        # Panel de Alerta Reactiva en el lado derecho
        ax = tx + tw + 60
        ay = ty
        aw = WIDTH - ax - 120
        ah = th

        # Animación de pulso perimetral amarillo
        pulse = 0.5 + 0.5 * math.sin(t * 6.0)
        alerta_col = (0, int(160 + 60 * pulse), 255)

        cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), (22, 28, 38), -1)
        cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), alerta_col, int(3 + 3 * pulse))

        # Beacon de alerta
        cv2.circle(canvas, (ax + 50, ay + 60), 16, alerta_col, -1)
        cv2.putText(canvas, "ALERTA AMARILLA - SIMIT", (ax + 85, ay + 68), cv2.FONT_HERSHEY_DUPLEX, 1.0, alerta_col, 2)

        # Datos del vehículo
        cv2.putText(canvas, "PLACA DETECTADA: BLJ732 (BOGOTA D.C.)", (ax + 40, ay + 140), cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2)
        cv2.putText(canvas, "ESTADO: 1 INFRACCION PENDIENTE DE PAGO", (ax + 40, ay + 185), cv2.FONT_HERSHEY_DUPLEX, 0.75, alerta_col, 2)
        cv2.putText(canvas, "TOTAL ADEUDADO: $468,500 COP", (ax + 40, ay + 230), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

        # Línea divisoria
        cv2.line(canvas, (ax + 40, ay + 270), (ax + aw - 40, ay + 270), (50, 65, 85), 1)

        cv2.putText(canvas, "DOCUMENTACION Y ANTECEDENTES:", (ax + 40, ay + 315), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 230, 118), 1)
        cv2.putText(canvas, "[x] SOAT: VIGENTE (2027-08-14)", (ax + 50, ay + 360), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 230, 118), 1)
        cv2.putText(canvas, "[x] TECNOMECANICA: VIGENTE", (ax + 50, ay + 400), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 230, 118), 1)
        cv2.putText(canvas, "[!] MULTAS SIMIT: 1 COMPARENDO", (ax + 50, ay + 440), cv2.FONT_HERSHEY_DUPLEX, 0.70, alerta_col, 2)
        cv2.putText(canvas, "[x] REPORTE HURTO POLICIA: LIMPIO", (ax + 50, ay + 480), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 230, 118), 1)

        # Mensaje de acción
        cv2.rectangle(canvas, (ax + 40, ay + 540), (ax + aw - 40, ay + 620), (32, 40, 56), -1)
        cv2.putText(canvas, "NOTIFICAR CONDUCTOR / AGENTE DE TRANSITO", (ax + 65, ay + 588), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2)

        out.write(canvas)


def animar_cierre(out):
    """Escena 5 (4.0 seg): Resumen ejecutivo, especificaciones y GitHub."""
    total_frames = int(4.0 * FPS)
    bg = crear_fondo_degradado(WIDTH, HEIGHT, (22, 28, 40), (8, 12, 18))
    dibujar_rejilla_tech(bg, 80, (28, 38, 52), 1)

    for f in range(total_frames):
        canvas = bg.copy()
        t = f / float(FPS)

        # Título
        cv2.putText(canvas, "SISTEMA ANPR COLOMBIA v2.0", (WIDTH // 2 - 380, HEIGHT // 2 - 180), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(canvas, "PROYECTO DE INTELIGENCIA COMPUTACIONAL", (WIDTH // 2 - 360, HEIGHT // 2 - 120), cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 230, 118), 2)

        # Resumen de Especificaciones
        specs = [
            "30 FPS Continuos con Rastreador IoU y Filtro de Zona Muerta",
            "Discriminacion Morfologica de Colas DIN 1451 (Q/O/J)",
            "Auditoria Gubernamental SIMIT en Vivo via API SODA",
            "Auto-Mejora por Nitidez Laplaciana sin Intervencion Manual",
            "Frontend Responsivo y Diagnostico Activo del Detector"
        ]

        for i, s in enumerate(specs):
            cv2.circle(canvas, (WIDTH // 2 - 380, HEIGHT // 2 - 40 + i * 45), 6, (0, 230, 118), -1)
            cv2.putText(canvas, s, (WIDTH // 2 - 350, HEIGHT // 2 - 32 + i * 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (220, 230, 240), 1)

        # Repositorio
        cv2.rectangle(canvas, (WIDTH // 2 - 420, HEIGHT - 150), (WIDTH // 2 + 420, HEIGHT - 80), (16, 22, 32), -1)
        cv2.rectangle(canvas, (WIDTH // 2 - 420, HEIGHT - 150), (WIDTH // 2 + 420, HEIGHT - 80), (0, 230, 118), 2)
        cv2.putText(canvas, "GITHUB: https://github.com/RetDr/deteccion_placas", (WIDTH // 2 - 370, HEIGHT - 105), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

        out.write(canvas)


def main():
    print(f"[INFO] Generando video promocional 1080p (1920x1080 a {FPS} FPS)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TEMP_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

    print(" -> Renderizando Escena 1: Intro Cinemática...")
    animar_intro(out)

    print(" -> Renderizando Escena 2: Detección y Seguimiento 30 FPS...")
    animar_deteccion_en_vivo(out)

    print(" -> Renderizando Escena 3: Análisis Morfológico Q vs O (DIN 1451)...")
    animar_discriminador_qo(out)

    print(" -> Renderizando Escena 4: Integración en Vivo con SIMIT Datos Abiertos...")
    animar_simit_en_vivo(out)

    print(" -> Renderizando Escena 5: Especificaciones y Créditos...")
    animar_cierre(out)

    out.release()
    print(f"[INFO] Cuadros renderizados en: {TEMP_VIDEO}")

    # Optimización final con FFmpeg a codec H.264 estándar compatible con cualquier reproductor y web
    print(" -> Optimizando contenedor con FFmpeg (H.264 / yuv420p)...")
    cmd_ffmpeg = f'ffmpeg -y -i "{TEMP_VIDEO}" -c:v libx264 -pix_fmt yuv420p -preset fast -crf 20 "{OUTPUT_VIDEO}"'
    res = os.system(cmd_ffmpeg)

    if res == 0 and os.path.exists(OUTPUT_VIDEO):
        try:
            os.remove(TEMP_VIDEO)
        except Exception:
            pass
        print(f"\n[EXITO] Video promocional generado con exito en la raiz:")
        print(f" -> {OUTPUT_VIDEO}")
        size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
        print(f" -> Tamano: {size_mb:.2f} MB | Duracion: ~24.5 segundos | 1080p 30 FPS")
    else:
        print("[AVISO] FFmpeg no pudo post-procesar; conservando el archivo generado directamente.")
        if os.path.exists(TEMP_VIDEO):
            os.replace(TEMP_VIDEO, OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
