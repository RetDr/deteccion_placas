# Sistema de Reconocimiento Automático de Placas (ANPR) y Evaluación de Riesgo Vial con Inteligencia Computacional

Sistema integral para el reconocimiento automático de placas vehiculares en Colombia en tiempo real (30 FPS) y toma de decisiones viales mediante **Inteligencia Computacional (Soft Computing)**. 

A diferencia de los sistemas tradicionales que solo hacen consultas determinísticas a bases de datos, este proyecto modela la incertidumbre sensorial, optimiza de forma bio-inspirada los parámetros de visión y clasifica conductores reincidentes mediante aprendizaje supervisado.

---

## 🧠 Núcleo de Inteligencia Computacional

1. **Lógica Difusa (Fuzzy Inference System - Mamdani):**
   - **Entradas:** Confianza visual del OCR $[0, 1]$, Severidad de Tránsito (multas SIMIT + SOAT/Tecno vencidos) $[0, 100]$ y Severidad Judicial (hurto/inmovilización) $[0, 100]$.
   - **Inferencia:** Base de reglas lingüísticas IF-THEN con T-norma de Gödel (Mínimo) y S-norma (Máximo).
   - **Salida:** Nivel de Riesgo Operativo $[0, 100]$ obtenido mediante **Defusificación por Centro de Gravedad (Centroid / COG)**:
     - `0 - 24`: Paso Permitido (Vehículo al día).
     - `25 - 49`: Alerta Preventiva (Multas pendientes leves).
     - `50 - 74`: Inspección Requerida (Documentación vencida).
     - `75 - 100`: Inmovilización Inmediata (Orden de captura o hurto).

2. **Optimización Bio-inspirada (Algoritmos Genéticos y PSO):**
   - **Algoritmo Genético (GA):** Calibra los hiperparámetros de preprocesamiento de imagen (`clip_limit`, `tile_grid_size`, máscara de enfoque y kernel de suavizado) maximizando una función de aptitud (*Fitness*) multiobjetivo basada en nitidez Laplaciana y Entropía de Shannon.
   - **Enjambre de Partículas (PSO):** Optimiza los coeficientes de asociación espacio-temporal (IoU, distancias euclidianas de centroide y suavizado EMA) del rastreador vehicular.

3. **Aprendizaje Supervisado (Red Neuronal MLP):**
   - Perceptrón Multicapa ($6 \to 16 \to 8 \to 1$) entrenado desde primeros principios con **Backpropagation** y optimización SGD Mini-Batch con regularización $L_2$.
   - Predice la probabilidad de reincidencia o fuga de conductores según perfil histórico.
   - Valida formalmente con **Matriz de Confusión ($TP, TN, FP, FN$)**, **Precision**, **Recall** y **F1-Score**.

---

## 📋 Requisitos del Sistema

- **Sistema Operativo:** Windows 10/11, Linux o macOS.
- **Python:** Versión 3.10 o superior (recomendado 3.11 a 3.14).
- **Cámara:** Cámara web USB, cámara integrada de laptop o cámara de celular (vía DroidCam, Iriun Webcam o IP Webcam). Si no hay cámara conectada, el sistema incluye un video de prueba de respaldo automático.

### Dependencias de Python

Instálalas con:
```bash
pip install -r requirements.txt
```

O directamente:
```bash
pip install opencv-python ultralytics easyocr fastapi uvicorn pydantic rapidfuzz torch torchvision
```

---

## 🚀 Cómo Ejecutar el Proyecto

Tienes dos modos de ejecución según lo que desees probar o sustentar:

### Modo 1: Pruebas de Inteligencia Computacional (Consola)
Ejecuta la validación de Lógica Difusa Mamdani, Algoritmos Genéticos, PSO y el entrenamiento de la Red Neuronal con sus matrices de confusión:

- **Con doble clic en Windows:**  
  Abre el archivo `probar_ic.bat`.

- **Desde la terminal:**  
  ```bash
  python test_computational_intelligence.py
  ```
  *(O ejecuta `python demo_ic.py` para una demostración guiada paso a paso ante el jurado).*

---

### Modo 2: Servidor Web y Detección en Tiempo Real

Inicia el motor de captura de video a 30 FPS, el detector YOLOv8, el lector OCR y el panel web reactivo:

- **Con doble clic en Windows:**  
  Abre el archivo `iniciar.bat`.

- **Desde la terminal:**  
  ```bash
  python server.py
  ```

Una vez que la consola indique que el servidor inició:
1. Abre tu navegador web (Chrome, Edge, Firefox).
2. Ingresa a la URL: **`http://localhost:8000`**
3. Verás la transmisión de video en vivo, las placas detectadas y las tarjetas de antecedentes con su **Riesgo Difuso** y el diagnóstico de la **Red Neuronal**.

---

## 📁 Estructura del Repositorio

```
deteccion_placas/
├── data/
│   ├── antecedentes.db          # Base de datos SQLite (marca, multas, SOAT, hurto)
│   ├── historial_capturas.json  # Registro de capturas realizadas
│   └── video_prueba.mp4         # Video de validación sin cámara física
├── models/
│   └── plate_detector/best.pt   # Pesos YOLOv8 especializados en placas
├── modules/
│   ├── camera_stream.py         # Captura multihilo y buffer DirectShow (30 FPS)
│   ├── database.py              # Consultas SQLite indexadas con Árbol B (< 0.2 ms)
│   ├── fuzzy_inference.py       # Sistema de Inferencia Difuso Mamdani + COG
│   ├── heuristic_optimizer.py   # Algoritmos Genéticos y PSO
│   ├── ml_classifier.py         # Red Neuronal MLP + Backpropagation + Métricas
│   ├── search_engine.py         # Motor de búsqueda en 3 capas + RapidFuzz
│   ├── tracker.py               # Rastreo IoU, EMA y buffer Best-Shot
│   └── vision_ocr.py            # Detección y normalización sintáctica colombiana
├── templates/
│   └── index.html               # Panel web responsivo con WebSockets y telemetría
├── iniciar.bat                  # Lanzador del servidor web en Windows
├── probar_ic.bat                # Ejecutable directo de pruebas de IC
├── server.py                    # Servidor FastAPI + WebSocket
├── test_computational_intelligence.py  # Suite de verificación científica
└── requirements.txt             # Dependencias del proyecto
```

---

## 📡 Endpoints de la API REST

- `GET /api/plates`: Retorna el historial de placas detectadas en memoria.
- `GET /api/antecedentes/{placa}`: Consulta antecedentes, cálculo difuso de riesgo y perfil de reincidencia.
- `GET /api/ic/metrics`: Retorna métricas formales del modelo (Accuracy, F1-Score, Matriz de Confusión, Estado Difuso).
- `POST /api/ic/run_genetic_optimization`: Dispara una calibración estocástica sobre el fotograma actual.
- `POST /api/register_vehicle`: Permite registrar nuevos vehículos en la base de datos de antecedentes.
