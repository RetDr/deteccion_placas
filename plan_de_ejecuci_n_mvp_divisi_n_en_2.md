# Plan de Ejecución MVP (1 Día): Sistema Inteligente de Reconocimiento de Placas

**Equipo:** 2 Personas (Persona A y Persona B)  
**Objetivo:** Tener hoy un prototipo funcional (MVP) que procese un video pregrabado, detecte placas, valide el texto, consulte antecedentes (con tolerancia a fallos) y muestre alertas en tiempo real.  
**Costo:** $0 (100% Open-Source / Gratuito).

---

## 1. Stack Tecnológico Recomendado (Rápido y Gratuito)
Todo el proyecto se construirá en **Python 3.10+** para evitar problemas de integración entre distintos lenguajes:
* **Visión Artificial y Video:** `opencv-python` + `ultralytics` (YOLOv8 preentrenado).
* **Lectura de Texto (OCR):** `easyocr` (fácil de instalar y funciona bien con placas).
* **Base de Datos y Búsqueda:** `sqlite3` (incluido en Python, soporta índices rápidos) + `rapidfuzz` (para búsqueda por similitud ante errores de lectura como `8`/`B` u `O`/`0`).
* **Interfaz de Usuario (Panel):** `streamlit` (crea un dashboard web interactivo con soporte de video y alertas en menos de 100 líneas de código).

---

## 2. Nota Crítica sobre la API del SIMIT
> **Realidad técnica:** El SIMIT (Colombia) **no cuenta con una API pública y gratuita** de libre acceso; su portal web utiliza protecciones anti-bots (reCAPTCHA y tokens dinámicos) que bloquean consultas automatizadas. Intentar hacer *web scraping* hoy puede consumirles todo el día y hacer fallar la demostración.

**Solución inteligente para el MVP de hoy:**
1. **Base de datos local indexada (SQLite) con formato SIMIT:** Crearemos una base de datos propia que replica la estructura real de respuestas del SIMIT (multas, comparendos, estado de SOAT/Tecnomecánica, reporte de hurto). Además, esto permite demostrar los **3 métodos de búsqueda** descritos en su documento (Indexada, Similitud y Reglas).
2. **Enlace directo de verificación:** En el panel, junto al resultado, incluiremos un botón de *"Verificar en Portal SIMIT"* para mostrar la proyección de integración real a futuro.

---

## 3. Estructura de Carpetas del Proyecto
Ambos deben crear exactamente esta estructura (pueden usar un repositorio en GitHub para sincronizar):

```text
proyecto_placas_mvp/
├── data/
│   ├── video_prueba.mp4       # Video corto (15-30 seg) de carros pasando
│   └── antecedentes.db        # Base de datos SQLite (Generada por Persona B)
├── modules/
│   ├── __init__.py
│   ├── vision_ocr.py          # <--- RESPONSABILIDAD PERSONA A
│   ├── search_engine.py       # <--- RESPONSABILIDAD PERSONA B
│   └── database.py            # <--- RESPONSABILIDAD PERSONA B
├── app.py                     # <--- RESPONSABILIDAD PERSONA B (Interfaz Streamlit)
└── requirements.txt
```

---

## 4. División del Trabajo

### 👤 PERSONA A: Módulo de Visión Artificial y OCR (El "Ojo")
**Misión:** Tomar un video (o fotograma), detectar dónde está el vehículo/placa, recortar la zona y extraer el texto alfanumérico.

**Tareas:**
1. Conseguir/descargar 1 o 2 videos cortos (15-30 seg) de vehículos con placas colombianas visibles (ej. desde YouTube o grabados con celular en una calle tranquila) y guardarlos en `data/video_prueba.mp4`.
2. Configurar OpenCV para leer el video frame por frame (procesando, por ejemplo, 1 de cada 5 o 10 frames para que corra fluido en un PC normal sin GPU).
3. Implementar la detección de placas usando YOLOv8 (`ultralytics`) con un modelo preentrenado de placas (o detección de vehículo + recorte de la parte inferior) y pasar el recorte por `EasyOCR`.
4. Aplicar preprocesamiento básico de imagen con OpenCV (escala de grises y contraste) para mejorar la lectura si hay sombras.
5. **Entregable:** Archivo `modules/vision_ocr.py`.

---

### 👤 PERSONA B: Motor de Búsqueda, Base de Datos y Panel UI (El "Cerebro" y la "Cara")
**Misión:** Crear la base de datos de antecedentes, implementar los algoritmos de búsqueda del documento y construir el dashboard visual con alertas.

**Tareas:**
1. Crear `modules/database.py`: Script que crea `data/antecedentes.db` en SQLite con una tabla `vehiculos` (**con índice `CREATE INDEX` en la columna `placa`** para cumplir con la *Búsqueda Indexada*). Poblar con 20-30 placas ficticias **+ las placas reales que aparezcan en el video de prueba de Persona A** (algunas limpias, otras con multas SIMIT, SOAT vencido o reporte de robo).
2. Crear `modules/search_engine.py` implementando las 3 capas del documento:
   * **Filtrado por Reglas (Regex):** Validar formato de placa colombiana (Carros: `^[A-Z]{3}[0-9]{3}$`, Motos: `^[A-Z]{3}[0-9]{2}[A-Z]$`) y limpiar caracteres basura (guiones, espacios).
   * **Búsqueda Indexada:** Consulta exacta `SELECT * FROM vehiculos WHERE placa = ?`.
   * **Búsqueda por Similitud (Tolerancia a fallos):** Si la búsqueda exacta falla, usar `rapidfuzz` (y reemplazos comunes como `0<->O`, `8<->B`, `1<->I`) para encontrar la placa más cercana con un umbral de similitud > 80%.
3. Crear `app.py` con **Streamlit**:
   * Selector para subir el video de prueba.
   * Reproductor visual que muestre los fotogramas analizados con el recuadro sobre la placa.
   * Tarjeta de resultados en tiempo real (Marca, Modelo, Propietario, Estado SIMIT).
   * **Alertas visuales** (verde: sin novedades / amarillo: multas o documentos vencidos / rojo: reporte de robo o inmovilización) y alerta sonora opcional.

---

## 5. Contrato de Integración (¡NO CAMBIAR!)
Para que ambos trabajen sin bloquearse, **Persona B** puede programar toda la interfaz usando datos falsos (mock) mientras **Persona A** termina el OCR. Al final, solo conectan esta función exacta en `modules/vision_ocr.py`:

```python
# Función que DEBE entregar Persona A en modules/vision_ocr.py
def analizar_frame(frame_bgr) -> list[dict]:
    """
    Recibe un frame de OpenCV (numpy array).
    Retorna una lista de diccionarios con las detecciones encontradas en ese frame:
    [
        {
            "placa_raw": "ABC123",       # Texto crudo leído por el OCR
            "confianza": 0.89,           # Float entre 0 y 1
            "bbox": (x1, y1, x2, y2),    # Coordenadas del recuadro en la imagen
            "recorte_img": crop_array    # Imagen recortada de la placa (para mostrar en UI)
        }
    ]
    """
```

Y esta función es la que expone **Persona B** en `modules/search_engine.py`:

```python
# Función que DEBE entregar Persona B en modules/search_engine.py
def consultar_placa(placa_raw: str) -> dict:
    """
    Limpia el texto, aplica Regex, busca por índice y si falla usa similitud.
    Retorna:
    {
        "placa_consultada": "ABC123",
        "metodo_usado": "INDEXADA" | "SIMILITUD" | "DESCARTADA_REGLA",
        "encontrado": True,
        "nivel_alerta": "VERDE" | "AMARILLO" | "ROJO",
        "datos": {
            "marca": "Mazda 3",
            "propietario": "Juan Pérez",
            "multas_simit": 2,
            "valor_multas": 1160000,
            "soat_vigente": False,
            "reporte_hurto": False
        }
    }
    """
```

---

## 6. Cronograma Express para Terminar HOY

| Bloque | Duración | Persona A (Visión/OCR) | Persona B (BD/Búsqueda/UI) |
| :--- | :--- | :--- | :--- |
| **1. Setup** | 30 min | Crear entorno virtual, instalar librerías y conseguir el video de prueba. **Pasar la lista de placas del video a Persona B.** | Crear entorno virtual, instalar librerías y crear el repositorio Git / estructura de carpetas. |
| **2. Desarrollo Core** | 2.5 hrs | Programar detección de placa + OCR en `vision_ocr.py` y probarlo en un script local con el video. | Programar `database.py` (con las placas del video), `search_engine.py` (Regex + Índice + Fuzzy) y probar consultas en consola. |
| **3. Interfaz y Ajustes** | 1.5 hrs | Optimizar velocidad (saltar frames) y precisión del OCR (filtros de imagen). | Construir el dashboard en `app.py` (Streamlit) simulando llamadas a `analizar_frame`. |
| **4. Integración** | 1 hr | **Juntos:** Conectar `vision_ocr.py` dentro de `app.py`. Corregir errores de formatos de imagen (BGR vs RGB). | **Juntos:** Verificar que al pasar el video en vivo por Streamlit salten las alertas correctas. |
| **5. Ensayo Demo** | 30 min | Grabar un video de respaldo de la pantalla funcionando (por si falla el computador en la presentación). | Preparar el guion de explicación usando los puntos de su propuesta original. |

---

## 7. Instalación Rápida (`requirements.txt`)
Guarden esto en `requirements.txt` y ejecuten `pip install -r requirements.txt`:

```text
opencv-python-headless
ultralytics
easyocr
streamlit
rapidfuzz
pandas
numpy
```