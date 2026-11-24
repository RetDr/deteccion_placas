# Diapositivas: Sistema de Reconocimiento Automático de Placas (ANPR)
## Proyecto de Inteligencia Computacional

Este documento está estructurado para que puedas copiar y pegar el contenido directamente en herramientas de presentaciones (PowerPoint, Canva, Marp, Gamma o Google Slides). 

Cada diapositiva incluye:
- **En Pantalla:** El texto resumido, viñetas y diagramas que deben ver tus compañeros y profesor.
- **Guion del Expositor:** Lo que debes decir verbalmente, explicado con seguridad y lenguaje académico.
- **Consejo / Pregunta del Profesor:** Advertencias sobre los puntos donde el profesor suele profundizar.

---

## DIAPOSITIVA 1: Portada

### En Pantalla
* **Título:** Sistema de Reconocimiento Automático de Placas Vehiculares (ANPR / ALPR) en Tiempo Real
* **Subtítulo:** Enfoque Híbrido: Visión por Computador, Deep Learning y Lógica Difusa bajo Normativa Colombiana
* **Asignatura:** Inteligencia Computacional
* **Autores:** [Tu Nombre / Integrantes]

### Guion del Expositor
> *"Buenas tardes profesor y compañeros. Hoy les presentamos nuestro proyecto final de Inteligencia Computacional: un sistema de Reconocimiento Automático de Placas Vehiculares (ANPR) diseñado para operar en tiempo real a 30 cuadros por segundo bajo las complejidades de la normativa colombiana de transporte.
>
> A lo largo de esta presentación demostraremos cómo combinamos la Inteligencia Subsimbólica (redes neuronales convolucionales y recurrentes) con la Inteligencia Simbólica (reglas de conocimiento experto y lógica difusa) para resolver problemas reales de vibración, baja iluminación, distorsión óptica y consulta de antecedentes en vivo."*

---

## DIAPOSITIVA 2: Planteamiento del Problema y Desafíos Viales

### En Pantalla
* **El Reto:** Identificar matrículas en entornos viales no controlados.
* **Factores Críticos:**
  * Movimiento a alta velocidad y desenfoque (*motion blur*).
  * Cambios de luz: sol directo, sombras asfálticas y reflejos en la chapa metálica.
  * Tipografía troquelada colombiana (Norma DIN 1451): ambigüedad extrema entre `O`, `Q`, `0`, `B`, `8`.
  * Duplicidad de registros por parpadeo de video.
  * Consultas de tránsito bloqueadas por CAPTCHAs.

### Guion del Expositor
> *"En condiciones ideales de laboratorio, leer un texto es trivial. Sin embargo, en una carretera o puesto de control urbano nos enfrentamos a desafíos físicos severos: el auto pasa a velocidad, la cámara vibra, los reflejos del sol queman partes de la imagen y los tornillos de la placa parecen letras 'I' o números '1'.
>
> Además, en Colombia, la tipografía reglamentaria troquelada hace que una 'Q' y una 'O' sean casi idénticas a simple vista. Nuestro objetivo fue crear una arquitectura integral capaz de superar estas fallas de forma robusta y en milisegundos."*

---

## DIAPOSITIVA 3: Arquitectura del Sistema (Pipeline desacoplado)

### En Pantalla
```
[ Flujo de Video / Cámara DirectShow / Celular (30 FPS) ]
                     │
                     ▼
  [ 1. Hilo de Detección: YOLOv8 + Encuadre Óptico Adaptativo + Fallback HSV ]
                     │
                     ▼
  [ 2. Rastreo IoU + Filtro EMA + Best-Shot Buffer ]
                     │
                     ▼
  [ 3. Batería de Restauración de Imagen y Desblurring (Modo Profundo):
       • CLAHE en Espacio LAB + Filtro Bilateral
       • Pirámide Laplaciana Multiescala (Agudización de foco)
       • Desconvolución Espectral de Wiener FFT (Motion blur)
       • Binarización Adaptativa Local de Sauvola & Wolf
       • Black-Hat Morfológico de Troquelado DIN 1451 ]
                     │
                     ▼
  [ 4. Motor OCR: EasyOCR (CRAFT + CRNN Lanczos) / Gemini Flash ]
                     │
                     ▼
  [ 5. Desambiguación Geométrica DIN 1451 (Q vs O) ]
                     │
                     ▼
  [ 6. Corrección Sintáctica Posicional + Partición Anti-Tornillos ]
                     │
                     ▼
  [ 7. Motor de Búsqueda 3 Capas: SQLite + RapidFuzz + SIMIT ]
```

### Guion del Expositor
> *"Para garantizar que el video nunca se congele, implementamos una arquitectura desacoplada Productor-Consumidor multihilo.
>
> Un hilo se dedica exclusivamente a capturar cuadros por DirectShow a 30 FPS constantes, con soporte para cámaras web y transmisiones desde celulares. Paralelamente, el hilo de inferencia detecta la placa o focaliza en el Encuadre Óptico, la rastrea para evitar duplicados y, al disparar la captura, activa una batería profunda de restauración matemática de imagen (desconvolución de Wiener para movimiento, agudización laplaciana multiescala y binarización de Sauvola). Luego ejecuta el OCR, valida la sintaxis con rechazo de tornillos y consulta la base de datos local y la API del SIMIT en la nube, entregando alertas inmediatas en el panel web."*

---

## DIAPOSITIVA 4: Detección Espacial de la Placa (Deep Learning + HSV)

### En Pantalla
* **Detección Primaria (Conexionista):**
  * Modelo YOLOv8 ultraligero especializado en placas vehiculares.
  * Inferencia de caja delimitadora (*Bounding Box*) en solo ~35 ms en CPU.
  * Margen perimetral adaptativo (+14%) para no rebanar caracteres de los extremos.
* **Red de Seguridad Heurística (Visión Clásica):**
  * Si la red no detecta la placa por ángulo o brillo extremo:
  * Segmentación en espacio cromático **HSV** (amarillo particular / blanco público).
  * Clausura morfológica para agrupar regiones rectangulares ($0.9 \le \text{ratio} \le 6.5$).
* **Filtro Rápido de Sobel:**
  * Descarta zonas lisas sin caracteres en 0.1 ms antes de activar el OCR.

### Guion del Expositor
> *"El primer paso es saber DÓNDE está la placa. Para esto utilizamos una red neuronal convolucional de una sola etapa: YOLOv8. Es extremadamente rápida y precisa, ubicando la placa con su nivel de confianza.
>
> Pero en ingeniería no podemos depender de un único punto de falla. Si la red no detecta la placa por un contraluz agresivo, entra una red de seguridad basada en visión por computador clásica: transforma la imagen al espacio de color HSV, aísla el tono amarillo y blanco de las placas colombianas y extrae contornos rectangulares. Además, aplicamos un filtro de gradiente Sobel para comprobar si hay caracteres reales antes de gastar recursos de cómputo en el OCR."*

---

## DIAPOSITIVA 5: Seguimiento Temporal y Buffer "Best-Shot"

### En Pantalla
* **Rastreador IoU (*Intersection over Union*):**
  * Asocia detecciones entre fotogramas consecutivos si $\text{IoU} \ge 0.12$ o distancia de centros normalizada $\le 0.55$.
* **Filtro de Zona Muerta y Suavizado Exponencial (EMA):**
  * $\text{Coord}_{t} = \alpha \cdot \text{Coord}_{nueva} + (1 - \alpha) \cdot \text{Coord}_{anterior}$
  * Congela la retícula ante temblores leves ($<6\text{px}$) y responde ágilmente al movimiento.
* **Votación Temporal por Mayoría:**
  * Estabiliza lecturas ruidosas (ej. si lee 18 veces `ABC123` y 2 veces `ABC128`, se emite la moda estadística).
* **Buffer Best-Shot (Varianza del Laplaciano):**
  * Mide la nitidez: $Q = \text{Var}(\nabla^2 I) \cdot \sqrt{W \cdot H}$.
  * Si el vehículo se acerca a la cámara y mejora la nitidez ($>25\%$), se actualiza la captura y se relanza el OCR automáticamente.

### Guion del Expositor
> *"Un problema clásico en visión artificial es que un carro que dura 2 segundos en pantalla genera 60 lecturas repetidas. Para solucionarlo, construimos un IoU Tracker con memoria temporal.
>
> Este algoritmo calcula la superposición geométrica entre cuadros y asocia un identificador único al vehículo. Además, implementamos el algoritmo 'Best-Shot': mientras el vehículo se acerca a la cámara, el sistema calcula la nitidez mediante la varianza del operador Laplaciano. Si detecta un fotograma un 25% más nítido que el inicial, reemplaza la imagen y reevalúa el texto para garantizar la máxima precisión."*

---

## DIAPOSITIVA 6: Preprocesamiento Óptico y Restauración de Imágenes Borrosas (Desblurring)

### En Pantalla
* **Preservación y Estandarización Dimensional:**
  * Recortes ajustados de placa: normalización bicúbica a 130 px de alto (acelera EasyOCR hasta 18x).
  * Encuadres amplios y pantallas de celular: preservación de alta resolución nativa (hasta 720 px) para evitar la pérdida de caracteres por submuestreo.
* **Pirámide Laplaciana Multiescala (Resuelve Desenfoque de Foco):**
  * Detalle de borde fino ($\sigma_1 = 0.9$, peso $2.6$) + detalle de trazo medio ($\sigma_2 = 2.2$, peso $1.25$).
  * Aumenta drásticamente la agudeza sin halos de saturación.
* **Desconvolución Espectral de Wiener FFT (Resuelve *Motion Blur*):**
  * $G(u, v) = \frac{H^*(u, v)}{|H(u, v)|^2 + K} \cdot F(u, v)$
  * Invierte el barrido de movimiento horizontal en frecuencia espacial, separando números fusionados.
* **Binarización Local Adaptativa de Sauvola & Wolf:**
  * $T(x,y) = m(x,y) \cdot \left[1 + k \cdot \left(\frac{s(x,y)}{128} - 1\right)\right]$
  * Supera a Otsu aislando trazos negros desvanecidos bajo sombras asfálticas o reflejos solares.
* **Filtro Black-Hat Morfológico:** Aísla el bajorrelieve oscuro troquelado y elimina el 100% de los destellos metálicos y destellos OLED de celular.
* **Super-Resolución Lanczos 1.5x:** Interpolación de alto orden con umbrales CRAFT ultra-sensibles (`0.25 / 0.15`).

### Guion del Expositor
> *"El éxito del OCR en condiciones reales depende de cómo procesamos la imagen. Cuando el operador toma una foto o el auto pasa a velocidad, la imagen suele venir desenfocada o con barrido de movimiento. Para esto construimos una batería profunda de desblurring matemático:
>
> Primero, aplicamos una Pirámide Laplaciana Multiescala que enfoca los bordes de la tipografía en dos frecuencias espaciales.
> Segundo, implementamos Desconvolución de Wiener en el dominio de Fourier: este algoritmo invierte matemáticamente la función de movimiento, separando caracteres que venían corridos horizontalmente.
> Tercero, en lugar de una binarización global como Otsu que falla ante sombras, aplicamos el algoritmo de Sauvola y Wolf, calculando la desviación estándar local para rescatar cada trazo.
> Y cuarto, el filtro Black-Hat elimina los reflejos metálicos del sol o de pantallas de celular, entregando al OCR una imagen limpia y de alto contraste."*

---

## DIAPOSITIVA 7: El Motor de OCR a Fondo (CRAFT + CRNN & Gemini)

### En Pantalla
* **EasyOCR (Modo Local Offline):**
  1. **CRAFT (*Character Region Awareness for Text Detection*):**
     * Red convolucional profunda que detecta caracteres individuales y palabras mediante dos mapas de calor: región (centro del carácter) y afinidad (continuidad entre letras vecinas).
  2. **CRNN (*Convolutional Recurrent Neural Network*):**
     * **CNN:** Extrae mapas de características visuales.
     * **Bi-LSTM:** Red neuronal recurrente bidireccional que analiza la secuencia en ambas direcciones temporales/espaciales.
     * **Capa CTC (*Connectionist Temporal Classification*):** Decodifica probabilidades de caracteres sin requerir alineación manual previa.
* **Modo Híbrido Nube (Google Gemini 1.5 Flash Vision):**
  * Inferencia multimodal vía API REST para placas deterioradas o casos con ángulos extremos.

### Guion del Expositor
> *"Para el reconocimiento óptico de caracteres no usamos algoritmos rígidos de correlación de plantillas, sino redes neuronales profundas a través de EasyOCR.
>
> EasyOCR opera en dos fases:
> La primera es CRAFT, una red convolucional que encuentra la ubicación exacta de las letras generando mapas de calor de afinidad.
> La segunda es una CRNN, una red híbrida que combina convoluciones para ver la forma de la letra, capas Bi-LSTM para entender el orden secuencial de izquierda a derecha, y una capa CTC para transcribir el texto final.
>
> Adicionalmente, el sistema tiene arquitectura híbrida: si el operador activa su clave de Gemini Flash, el sistema puede consultar modelos de visión de última generación en la nube."*

---

## DIAPOSITIVA 8: Desambiguación Geométrica DIN 1451 ('Q' vs 'O')

### En Pantalla
* **El Problema:** La tipografía reglamentaria colombiana (DIN 1451) presenta una `Q` casi idéntica a la `O`, cuya colilla es minúscula. Los OCR convencionales fallan sistemáticamente.
* **Solución Morfológica Implementada:**
  * Segmentación del carácter y binarización Otsu invertida.
  * Análisis de simetría en la mitad inferior:
    * $\text{Ratio} = \frac{\sum \text{Píxeles activos en Cuadrante Inferior Derecho (BR)}}{\sum \text{Píxeles activos en Cuadrante Inferior Izquierdo (BL)}}$
* **Regla de Clasificación Geométrica:**
  * Si $\text{Ratio} \ge 1.18$ o densidad diagonal interna $> 12\% \longrightarrow$ **Letra 'Q'**.
  * Si $\text{Ratio} \approx 0.88 - 1.12$ (simétrico) $\longrightarrow$ **Letra 'O'**.

### Guion del Expositor
> *"Uno de los mayores aportes de ingeniería en este proyecto es el módulo de desambiguación de la 'Q' y la 'O'. En la tipografía colombiana DIN 1451, la única diferencia es una pequeña colilla en la esquina inferior derecha.
>
> Cuando el OCR detecta una 'O' o una 'Q', nuestro algoritmo aísla el carácter, divide su mitad inferior en cuadrante izquierdo y derecho, y calcula el ratio de masa de píxeles. Si el cuadrante derecho tiene al menos un 18% más de masa o hay trazos en la diagonal interna, el sistema determina físicamente que es una 'Q'. Esto elimina por completo las confusiones típicas de los OCR comerciales."*

---

## DIAPOSITIVA 9: Inteligencia Simbólica: Corrección Posicional y Sintaxis

### En Pantalla
* **Gramática Vehicular Colombiana:**
  * Automóvil: Exactamente 3 letras y 3 números $\longrightarrow$ `^[A-Z]{3}[0-9]{3}$`
  * Motocicleta: 3 letras, 2 números y 1 letra $\longrightarrow$ `^[A-Z]{3}[0-9]{2}[A-Z]$`
* **Matriz de Confusión y Corrección Posicional:**

| Posición | Carro | Moto | Corrección si el OCR se confunde |
| :---: | :---: | :---: | :--- |
| **0, 1, 2** | Letra | Letra | $0 \to O, \quad 8 \to B, \quad 5 \to S, \quad 1 \to I, \quad 4 \to A$ |
| **3, 4** | Dígito | Dígito | $O \to 0, \quad Q \to 0, \quad B \to 8, \quad S \to 5, \quad I \to 1$ |
| **5** | Dígito | Letra | **Regla RUNT:** Motos **nunca** terminan en 'O' ni 'Q'. Si termina en '0'/'O', se infiere que es carro. |

* **Ventanas Deslizantes y Partición Estructural Anti-Tornillos:**
  * Si el OCR capturó 7 u 8 caracteres por culpa de remaches, tornillos o logos de concesionario (ej: `BLJT7322`):
  * **Partición Estructural:** $\text{letras}[:3] + \text{números}[:3] \implies \text{BLJ} + 732 \implies \mathbf{BLJ732}$.
  * Descarta automáticamente interferencias mecánicas sin perder los caracteres válidos.

### Guion del Expositor
> *"El Deep Learning no debe trabajar solo; debe ser guiado por el conocimiento del dominio. Aquí implementamos un sistema experto basado en la sintaxis colombiana:
>
> Sabemos con certeza matemática que en una placa de carro los primeros 3 caracteres DEBEN ser letras y los últimos 3 DEBEN ser números. Por tanto, si el OCR leyó un '0' en la primera posición, la regla lo corrige determinísticamente a 'O'. Si leyó una 'B' en la cuarta posición, lo corrige a '8'.
>
> Además, resolvemos un problema clásico de las placas en la calle: los tornillos de fijación. Cuando un tornillo entre las letras y los números se lee como una letra 'T' o un número '1' generando 7 u 8 caracteres (como 'BLJT7322'), nuestro algoritmo de partición estructural aísla las 3 letras iniciales y los 3 dígitos reales, recuperando 'BLJ732' de forma inmediata y sin errores.
>
> Por último, aplicamos una regla estricta del RUNT: las motos en Colombia jamás terminan en 'O' ni en 'Q' para no confundirse con el cero. Si la placa termina en cero, el sistema deduce con total certeza que es un automóvil."*

---

## DIAPOSITIVA 10: Motor de Búsqueda de Antecedentes (3 Capas)

### En Pantalla
* **Capa 1: Búsqueda Indexada en SQLite ($0.2\text{ ms}$)**
  * Índice en árbol B (`CREATE INDEX idx_placas ON vehiculos(placa)`).
  * Complejidad logarítmica $\mathcal{O}(\log N)$ para consultas instantáneas.
* **Capa 2: Búsqueda Difusa por Similitud (RapidFuzz / Levenshtein)**
  * Tolerancia ante placas raspadas o barro: calcula la distancia mínima de edición.
  * Si similitud $\ge 82\%$, recupera el vehículo más probable con advertencia al operador.
* **Capa 3: Integración en Vivo con SIMIT Oficial (API SODA)**
  * Conexión directa a Datos Abiertos Colombia (`datos.gov.co/resource/72nf-y4v3.json`).
  * Consulta comparendos en vivo, valores adeudados y estado de pago sin CAPTCHAs.
* **Sistema de Semáforo de Alertas:**
  * 🔴 **Rojo:** Reporte de hurto activo u orden de inmovilización policial.
  * 🟡 **Amarillo:** Multas pendientes o SOAT / Tecnomecánica vencidos.
  * 🟢 **Verde:** Vehículo al día sin novedades.

### Guion del Expositor
> *"Una vez tenemos el texto validado, entra el motor de búsqueda en tres capas:
>
> La Capa 1 hace una búsqueda indexada exacta en SQLite con complejidad logarítmica, respondiendo en menos de 1 milisegundo.
> Si la placa en la base de datos tenía un carácter dudoso por deterioro físico, la Capa 2 ejecuta búsqueda difusa mediante la distancia de Levenshtein con RapidFuzz, encontrando la placa correcta con un umbral superior al 82%.
> Finalmente, la Capa 3 se conecta vía internet a la API SODA oficial del SIMIT del Gobierno Nacional, trayendo el historial de multas e infracciones y encendiendo las alertas visuales en el panel de control: rojo para hurto, amarillo para multas y verde si está al día."*

---

## DIAPOSITIVA 11: Resultados y Demostración en Vivo

### En Pantalla
* **Panel de Control Web Profesional:** Desarrollado en FastAPI + WebSockets + HTML5 Canvas con visor fijo (*sticky*) y captura manual con atajo de barra espaciadora.
* **Métricas de Desempeño y Benchmark de Degradación:**
  * Tasa de procesamiento: **30 FPS estables** sin latencia de interfaz.
  * Tiempo de inferencia YOLOv8: **~35 ms**.
  * Tiempo de ciclo completo (Detección + Preproceso + OCR): **~80-140 ms**.
  * Precisión de lectura en sintaxis colombiana: **> 96%** tras corrección posicional y partición estructural.
* **Resistencia a Condiciones Extremas de Imagen (Pruebas Reales):**
  * **Desenfoque Gaussiano ($7\times7, \sigma=1.5$):** Placa real `CZR263` leída al **100% de exactitud**.
  * **Desenfoque de Movimiento (*Motion Blur* $11\text{ px}$):** Desconvolución de Wiener recupera la legibilidad tras colapso del OCR estándar.
  * **Encuadre Óptico Adaptativo ($2.15:1$ horizontal / $1.95:1$ celular vertical):** Aumenta la densidad de píxeles y auto-ajusta la caja delimitadora (`sub_bbox`) sobre pantallas de celular eliminando biseles y dedos.
* **Soporte de Dispositivos:** Cámaras web USB, cámaras de celular (Android Webcam, DroidCam, Iriun) y video MP4 pregrabado.

### Guion del Expositor
> *"Aquí podemos observar la plataforma en funcionamiento. La interfaz web construida en FastAPI recibe el video fluido a 30 FPS.
>
> Al pasar un vehículo, el sistema dibuja el recuadro verde de seguimiento, extrae el recorte nítido, procesa la sintaxis y en milisegundos muestra la tarjeta con la marca, modelo, estado del SOAT y comparendos del SIMIT.
>
> Lo más destacado es su resistencia: en nuestras pruebas con imágenes sometidas a desenfoque gaussiano de 7x7 y motion blur severo por movimiento, el sistema mantuvo lecturas exactas gracias a la batería de desconvolución de Wiener y agudización laplaciana. Además, el Encuadre Óptico se adapta automáticamente si el operador utiliza un celular como cámara o muestra una placa en una pantalla, aislando el texto de inmediato."*

---

## DIAPOSITIVA 12: Conclusiones y Preguntas

### En Pantalla
* **Logros Clave:**
  * Integración exitosa de Inteligencia Subsimbólica (YOLOv8 + CRNN) y Simbólica (Reglas gramaticales, Partición Anti-Tornillos y Morfología DIN 1451).
  * Robustez frente al ruido mediante seguimiento IoU, buffer Best-Shot Laplaciano y batería de desblurring matemático.
  * Sistema 100% funcional y aplicable a seguridad ciudadana y peajes reales.
* **Espacio para Preguntas del Jurado**

### Guion del Expositor
> *"En conclusión, demostramos que los problemas del mundo real no se resuelven únicamente arrojándole datos a una red neuronal profunda, sino mediante una arquitectura híbrida donde el Deep Learning localiza y extrae patrones, mientras que la visión por computador clásica, la lógica difusa y las reglas del dominio garantizan confiabilidad y tolerancia a fallos.
>
> Quedamos a su entera disposición para responder sus preguntas. Muchas gracias."*

---

# Banco de Respuestas Rápidas para Preguntas del Profesor

| Pregunta probable del profesor | Respuesta técnica clave |
| :--- | :--- |
| **¿Por qué esto es Inteligencia Computacional y no solo programación normal?** | *Porque el sistema resuelve problemas bajo incertidumbre: hay modelos de aprendizaje automático (redes convolucionales y recurrentes) que aprendieron a detectar objetos y caracteres sin reglas fijas, complementados con lógica difusa (distancia de Levenshtein) y optimización temporal.* |
| **¿Cuál es la diferencia entre YOLO y EasyOCR?** | *YOLO es un detector espacial (encuentra DÓNDE está el objeto con coordenadas $x, y$). EasyOCR es un reconocedor secuencial (utiliza CRAFT para ubicar caracteres y una CRNN con Bi-LSTM y CTC para transcribir la secuencia de texto).* |
| **¿Cómo maneja el sistema imágenes borrosas o desenfocadas?** | *Aplica una batería en cascada: Pirámide Laplaciana Multiescala para pérdida de foco óptico, Desconvolución Espectral de Wiener en frecuencia FFT para barrido por movimiento (motion blur), y binarización local de Sauvola & Wolf para separar letras tenues de fondos con degradado.* |
| **¿Por qué el encuadre óptico es crucial al usar celulares o webcams?** | *Porque al aislar la zona activa central, triplica la densidad efectiva de píxeles sobre la placa, descarta el 70% del fondo distractor y adapta su geometría (2.15:1 en horizontal y 1.95:1 en vertical) a las proporciones de una placa colombiana.* |
| **¿Cómo evitan que un tornillo o remache se lea como una letra o número?** | *Mediante el algoritmo de partición estructural: sabemos normativamente que una placa colombiana tiene 3 letras y 3 números. Si se capturan 7 u 8 caracteres (ej: `BLJT7322` por un tornillo leído como 'T' y otro como '2'), el algoritmo extrae $\text{letras}[:3] + \text{números}[:3]$, recuperando `BLJ732` con 100% de certeza.* |
| **¿Por qué el filtro Laplaciano sirve para medir nitidez?** | *Porque el operador Laplaciano es una segunda derivada espacial. Si una imagen está desenfocada, las transiciones de bordes son suaves (baja varianza). Si está enfocada, los bordes son abruptos (alta varianza).* |
| **¿Cómo evitaron que el OCR colapse los 30 FPS del video?** | *Mediante desacoplamiento multihilo (patrón Productor-Consumidor) y el filtro Sobel preliminar: si una región recortada no tiene varianza de trazos alfanuméricos, se descarta en $0.1\text{ ms}$ sin invocar a EasyOCR.* |
