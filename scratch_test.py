import cv2
import numpy as np
from modules.vision_ocr import analizar_frame

frame = np.full((600, 800, 3), (70, 70, 70), dtype=np.uint8)

# 1. Car plate: 'ABC 123' and 'BOGOTA D.C.'
car_x, car_y = 100, 80
cv2.rectangle(frame, (car_x, car_y), (car_x + 280, car_y + 120), (0, 204, 255), -1)
cv2.rectangle(frame, (car_x, car_y), (car_x + 280, car_y + 120), (0, 0, 0), 3)
cv2.putText(frame, 'ABC 123', (car_x + 25, car_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4, cv2.LINE_AA)
cv2.putText(frame, 'BOGOTA D.C.', (car_x + 60, car_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

# 2. Moto plate: 'XYZ 45E' and 'MEDELLIN'
moto_x, moto_y = 450, 80
cv2.rectangle(frame, (moto_x, moto_y), (moto_x + 280, moto_y + 120), (0, 204, 255), -1)
cv2.rectangle(frame, (moto_x, moto_y), (moto_x + 280, moto_y + 120), (0, 0, 0), 3)
cv2.putText(frame, 'XYZ 45E', (moto_x + 25, moto_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4, cv2.LINE_AA)
cv2.putText(frame, 'MEDELLIN', (moto_x + 75, moto_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

results = analizar_frame(frame)
print(f"Total detecciones: {len(results)}")
for r in results:
    pct = r['confianza'] * 100
    print(f"Placa: {r['placa_raw']} -- Tipo: {r['tipo']} -- Conf: {pct:.1f}% -- Municipio: {r.get('municipio')} -- Bbox: {r['bbox']}")
