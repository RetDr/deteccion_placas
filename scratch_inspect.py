import json
import base64
import cv2
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.vision_ocr import obtener_modelos, extraer_placa_estricta

_, _, reader = obtener_modelos()

json_path = os.path.join(BASE_DIR, "data", "historial_capturas.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for p in ['CZR263', 'LLR265', 'JLJ72W', 'JML66L']:
    if p in data:
        buf = base64.b64decode(data[p]['crop_base64'])
        nparr = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res = reader.readtext(img)
        print(f"=== Placa registrada: {p} (Crop: {img.shape}) ===")
        for r in res:
            p_ext, tipo, c = extraer_placa_estricta(r[1])
            print(f"   Texto: '{r[1]}' (Conf: {r[2]:.3f}) -> Extrae: '{p_ext}' ({tipo})")
