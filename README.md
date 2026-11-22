# Sistema de Reconocimiento Automático de Placas Vehiculares de Colombia (ANPR / ALPR)

Sistema integral de visión artificial e inteligencia computacional para el reconocimiento automático de placas vehiculares de Colombia en tiempo real (30 FPS), con rastreo geométrico de objetos (Object Tracking), supresión de duplicados, corrección sintáctica reglamentaria y consulta automatizada de antecedentes de tránsito (SIMIT, RUNT) y judiciales en base de datos local SQLite y nube.

---

## Tabla de Contenidos

- [Descripcion General](#descripcion-general)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Caracteristicas Principales](#caracteristicas-principales)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalacion y Configuracion](#instalacion-y-configuracion)
- [Guia de Uso y Ejecucion](#guia-de-uso-y-ejecucion)
- [Conexion de Camaras](#conexion-de-camaras)
- [Base de Datos y Antecedentes](#base-de-datos-y-antecedentes)
- [Documentacion de API y WebSockets](#documentacion-de-api-y-websockets)
- [Benchmark y Rendimiento](#benchmark-y-rendimiento)
- [Licencia](#licencia)

---

## Descripcion General

Este proyecto implementa una solucion de grado vial y carretera para la deteccion e identificacion instantanea de matriculas automotrices y de motocicletas bajo la normativa colombiana de transporte.

El sistema esta disenado para operar de forma continua bajo condiciones opticas desafiantes (movimiento a alta velocidad, iluminacion solar directa, sombras intensas, reflejos sobre superficies metalicas y ruido de sensor), proporcionando visualizacion continua mediante streaming MJPEG, notificaciones reactivas via WebSocket y un panel de control web responsivo de alta telemetria.

---

## Arquitectura del Sistema

El pipeline procesa cada cuadro mediante una arquitectura desacoplada Productor-Consumidor multihilo para garantizar 30 FPS constantes sin congelamientos:

```
[ Camara USB / Celular / RTSP / Video ]
                  │
                  ▼
         [ Hilo de Captura ] ──▶ Buffer DirectShow (YUY2/NV12/MJPG a 30 FPS)
                  │
                  ▼
         [ Hilo de Tracking ]
                  │
                  ├── 1. Deteccion Primaria: YOLOv8 especializado en placas (conf >= 0.15)
                  ├── 2. Red de Seguridad: Descomposicion cromatica HSV (Amarillo/Blanco)
                  ├── 3. Enfoque Dinamico: Encuadre Optico ROI adaptativo
                  ├── 4. Fusion Espacial: Supresion de No Maximos (NMS multi-criterio)
                  └── 5. Rastreador IoU: Filtro de zona muerta y suavizado EMA
                  │
                  ▼
       [ Cola Asincrona de OCR ]
                  │
                  ├── Evaluacion Laplaciana de Nitidez (Best-Shot Buffer)
                  ├── Superresolucion LAB + Realce de Contraste CLAHE
                  ├── Compensacion de desenfoque vial (Filtro Unsharp Mask)
                  ├── Motor OCR Local: EasyOCR optimizado para CPU/GPU
                  └── Modo Hibrido Nube: Google Gemini 1.5 Flash Vision (Opcional)
                  │
                  ▼
    [ Motor de Sintaxis Colombiana ]
                  │
                  ├── Validacion Estricta: Carro (ABC123) / Moto (ABC12D)
                  ├── Correccion Posicional (Letras en pos 0-2, Digitos en pos 3-5)
                  └── Cruce Fonetico de Municipios (Base oficial de 54 secretarias)
                  │
                  ▼
    [ Motor de Busqueda de Antecedentes ]
                  │
                  ├── Capa 1: Coincidencia Exacta Indexada SQLite (0.5 ms)
                  ├── Capa 2: Similitud Difusa Levenshtein (Tolerancia OCR)
                  └── Capa 3: Normalizacion Fonetica Soundex
                  │
                  ▼
[ Panel Web en Tiempo Real: FastAPI + WebSocket + Telemetria HUD ]
```

---

## Caracteristicas Principales

- **Deteccion Hibrida Multi-Escala:** Modelo YOLOv8 ultraligero complementado con segmentacion cromatica en espacio HSV para placas colombianas amarillas (servicio particular/publico) y blancas (servicio publico/oficial).
- **Rastreo con Eliminacion de Duplicados:** Un solo vehiculo en transito genera un unico registro consolidado en el historial, eliminando alertas redundantes o falsos positivos recurrentes.
- **Compensacion de Movimiento Vial y Auto-Mejora en Vivo:** Buffer Best-Shot que evalua la varianza del Laplaciano en tiempo real. Si una placa sigue en pantalla con baja confianza, el sistema reintenta el OCR automaticamente tan pronto la nitidez mejora (>25%).
- **Discriminacion Geometrica Avanzada (Q vs O / J):** Modulo morfologico basado en la norma DIN 1451 colombiana que analiza la simetria de cuadrantes exteriores, impidiendo confusiones entre la letra `Q`, la `O` o la `J`.
- **Integracion Oficial SIMIT en Vivo (Datos Abiertos Colombia):** Consulta automatica a la API SODA (`datos.gov.co/resource/72nf-y4v3.json`) para recuperar comparendos oficiales, valor adeudado ($ COP) y estado de pago (`pagado_si_no = NO/SI`) sin resolver CAPTCHAs.
- **Diagnostico Activo del Detector:** Si una imagen se toma movida o con baja luz, el detector no descarta en silencio; presenta la tarjeta con su porcentaje de confianza y advertencia diagnostica para guiar al operador.
- **Interfaz Web Responsiva con Alertas Reactivas:** Diseno adaptable para movil, tableta y escritorio (sin emojis), con animaciones de borde perimetral (Alerta Roja para hurto policial, Alerta Amarilla para infracciones de transito y SOAT).
- **Optimizacion para Camaras de Celular:** Compatibilidad nativa con Android 14 Webcam, DroidCam, Iriun e IP Webcam mediante backend DirectShow (`cv2.CAP_DSHOW`) sin bloqueos de negociacion de formato de color.
- **Modo Dual Local / Gemini:** Funcionamiento 100% offline con EasyOCR local y opcion de conmutacion dinamica a Google Gemini 1.5 Flash Vision mediante clave de API.

---

## Estructura del Repositorio

```
deteccion_placas/
│
├── data/
│   ├── antecedentes.db          # Base de datos SQLite con historial vehicular y antecedentes
│   ├── historial_capturas.json  # Registro persistente de capturas realizadas
│   └── video_prueba.mp4         # Secuencia de video para validacion sin camara fisica
│
├── models/
│   ├── plate_detector/
│   │   └── best.pt              # Pesos entrenados del detector YOLOv8 para placas
│   └── easyocr/                 # Pesos preentrenados del motor OCR local
│
├── modules/
│   ├── __init__.py
│   ├── camera_stream.py         # Motor de captura, hilos de tracking y streaming MJPEG
│   ├── database.py              # Administracion de base de datos SQLite y consultas
│   ├── gemini_ocr.py            # Integracion cliente HTTP para API Google Gemini Flash
│   ├── search_engine.py         # Motor de busqueda en cascada (SQLite + SIMIT en vivo)
│   ├── simit_connector.py       # Conector SODA a API Datos Abiertos Colombia (SIMIT)
│   ├── tracker.py               # Rastreador IoU, filtro de temblor y gestion de pistas
│   └── vision_ocr.py            # Segmentacion, discriminador Q/O y sintaxis estricta
│
├── templates/
│   └── index.html               # Panel web responsivo, alertas animadas e inspeccion
│
├── tests/                       # Suite de pruebas unitarias y validaciones de sintaxis
│   ├── scratch_test.py
│   ├── scratch_test_crops.py
│   ├── scratch_test_ocr.py
│   ├── scratch_test_qo.py
│   ├── scratch_test_rules.py
│   └── test_pipeline.py
│
├── app.py                       # Interfaz alternativa para pruebas en Streamlit
├── iniciar.bat                  # Lanzador rapido para Windows (CMD)
├── iniciar.ps1                  # Lanzador interactivo para Windows PowerShell
├── registrar_placa.py           # Herramienta CLI para registrar vehiculos en la base de datos
├── requirements.txt             # Dependencias del proyecto
├── server.py                    # Servidor web principal FastAPI con WebSockets
├── test_vision.py               # Validacion del pipeline de vision y recorte
└── yolov8n.pt                   # Pesos base de deteccion YOLOv8
```

---

## Requisitos del Sistema

- **Sistema Operativo:** Windows 10/11, Linux (Ubuntu 20.04+) o macOS.
- **Python:** Version 3.10, 3.11 o 3.12 (64-bit).
- **Hardware Recomendado:**
  - Procesador Intel Core i5 / AMD Ryzen 5 o superior.
  - 8 GB de memoria RAM minima (16 GB recomendado).
  - GPU NVIDIA con soporte CUDA (opcional, acelera inferencia pero no es mandatorio).
- **Camara:** Camara web USB integrada, camara externa o celular conectado mediante cable USB / Wi-Fi.

---

## Instalacion y Configuracion

### 1. Clonar el Repositorio

```bash
git clone https://github.com/RetDr/deteccion_placas.git
cd deteccion_placas
```

### 2. Crear y Activar un Entorno Virtual (Recomendado)

En Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

En Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Nota para aceleracion por GPU (NVIDIA CUDA):* Si dispones de tarjeta grafica NVIDIA, instala la version de PyTorch con soporte CUDA correspondiente desde [pytorch.org](https://pytorch.org).

---

## Guia de Uso y Ejecucion

### Inicio Rapido del Servidor

Puedes iniciar la aplicacion de tres formas:

**Opcion A - Usando PowerShell:**
```powershell
.\iniciar.ps1
```

**Opcion B - Usando archivo Batch:**
Doble clic sobre `iniciar.bat` o desde CMD:
```cmd
iniciar.bat
```

**Opcion C - Comando Python directo:**
```bash
python server.py
```

Una vez iniciado, abre tu navegador web en:
```
http://localhost:8000
```

---

## Conexion de Camaras

El sistema permite seleccionar la fuente de video directamente desde la interfaz web:

1. **Camara 0:** Camara web integrada de la laptop o primera camara USB detectada.
2. **Camara 1 / Camara 2:** Camara de telefono inteligente conectada via USB.
   - *Android 14:* Conecta el celular por cable USB, selecciona en la notificacion del telefono "Usar USB para: Camara web".
   - *DroidCam / Iriun:* Inicia la aplicacion en el celular y la aplicacion cliente en la PC; el sistema la reconocera como indice 1 o 2.
3. **Camara IP / Stream Personalizado:** Selecciona "TRANSMISION PERSONALIZADA (URL)" e introduce la direccion RTSP o HTTP (ej. `http://192.168.1.15:8080/video` con IP Webcam).
4. **Archivo de Video:** Selecciona "VIDEO DE PRUEBA LOCAL" para verificar el funcionamiento con el video incluido en `data/video_prueba.mp4`.

---

## Controles de la Interfaz Web

- **Captura Manual Instantanea:** Presiona el boton "TOMAR FOTO Y RECONOCER" o la **Barra Espaciadora** del teclado para procesar inmediatamente el cuadro actual y consultar antecedentes.
- **Encuadre Optico (ROI):** Activa o desactiva la zona de interes en pantalla para enfocar la atencion del algoritmo en el area de transito vehicular.
- **Motor Gemini Flash:** Permite ingresar tu API Key de Google Gemini para habilitar el reconocimiento hibrido en la nube, con conmutacion automatica a EasyOCR local en caso de caida o restriccion.
- **Limpiar Registros:** Reinicia el panel visual de detecciones activas.

---

## Base de Datos y Antecedentes

El archivo `data/antecedentes.db` contiene la estructura SQLite para la gestion vehicular:

### Esquema de la Tabla `vehiculos`

| Campo | Tipo | Descripcion |
|---|---|---|
| `placa` | TEXT (PK) | Codigo de placa normalizado (6 caracteres) |
| `tipo` | TEXT | CARRO o MOTO |
| `marca` | TEXT | Fabricante del vehiculo (Chevrolet, Mazda, etc.) |
| `linea` | TEXT | Linea o modelo comercial |
| `modelo` | INTEGER | Ano de fabricacion |
| `color` | TEXT | Color predominante |
| `propietario` | TEXT | Nombre completo del titular |
| `cedula_propietario` | TEXT | Documento de identidad |
| `municipio_registro` | TEXT | Secretaria de transito de matricula |
| `multas_simit` | INTEGER | Cantidad de infracciones registradas |
| `valor_multas` | REAL | Total adeudado en COP |
| `detalle_multas` | TEXT | Descripcion de comparendos vigentes |
| `soat_vigente` | INTEGER | 1 = Vigente, 0 = Vencido |
| `soat_vencimiento` | TEXT | Fecha limite de cobertura |
| `tecnomecanica_vigente`| INTEGER | 1 = Vigente, 0 = Vencido |
| `reporte_hurto` | INTEGER | 1 = Requerido por la Policia Nacional |
| `orden_inmovilizacion` | INTEGER | 1 = Orden de captura o inmovilizacion |
| `observaciones` | TEXT | Notas de transito o historial policial |

### Registrar Nuevas Placas

Para anadir vehiculos de prueba a la base de datos, ejecuta:
```bash
python registrar_placa.py
```
O realiza una peticion POST a la API REST:
```bash
curl -X POST "http://localhost:8000/api/register_vehicle" \
     -H "Content-Type: application/json" \
     -d "{\"placa\": \"ABC123\", \"tipo\": \"CARRO\", \"marca\": \"TOYOTA\", \"multas_simit\": 0, \"soat_vigente\": 1}"
```

---

## Documentacion de API y WebSockets

El servidor expone endpoints estandarizados para integracion externa:

| Metodo | Endpoint | Descripcion |
|---|---|---|
| `GET` | `/` | Retorna la interfaz web principal del panel de control |
| `GET` | `/video_feed` | Stream continuo de video MJPEG con reticulas y telemetria HUD |
| `POST` | `/api/capture_frame` | Captura fotograma actual, procesa OCR y consulta antecedentes |
| `GET` | `/api/plates` | Lista el historial completo de placas registradas en la sesion |
| `GET` | `/api/antecedentes/{placa}` | Consulta antecedentes policiales y de transito para una placa |
| `POST` | `/api/set_source` | Cambia la fuente activa de video (indice de camara o URL de stream) |
| `POST` | `/api/toggle_roi` | Conmuta el estado del encuadre optico |
| `GET` | `/api/status` | Retorna metricas de rendimiento (FPS camara, FPS OCR, latencia) |
| `POST` | `/api/clear` | Limpia el historial visual y el buffer en memoria |
| `WS` | `/ws` | Canal WebSocket para eventos `NEW_PLATE`, `UPDATE_PLATE` y `TELEMETRY` |

---

## Benchmark y Rendimiento

Mediciones realizadas en procesador estandar de consumo (Intel Core i7 12a Gen, sin GPU dedicada):

- **Inferencia de Deteccion YOLOv8:** 35 ms a 85 ms por cuadro.
- **Rastreo IoU + Suavizado:** < 2 ms por cuadro.
- **Inferencia OCR Local (EasyOCR):** 280 ms a 450 ms por recorte vehicular.
- **Consulta de Antecedentes SQLite:** < 1 ms por placa.
- **Tasa de Cuadros en Pantalla:** 28 a 30 FPS constantes.
- **Precision Sintactica:** 100% de cumplimiento con el estandar del Ministerio de Transporte de Colombia.

---

## Licencia

Proyecto desarrollado con fines academicos y de investigacion para la asignatura de **Inteligencia Computacional**.
Distribuido bajo la Licencia MIT.
