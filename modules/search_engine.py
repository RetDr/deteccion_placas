"""
Motor de Búsqueda Inteligente en 3 Capas para Antecedentes Vehiculares (Persona B).
Capas de consulta:
1. Filtrado y Normalización por Reglas Sintácticas (Regex Colombia).
2. Búsqueda Indexada en SQLite (Complejidad O(log N), latencia < 0.2 ms).
3. Búsqueda por Similitud / Fuzzy Matching con RapidFuzz (Tolerancia a confusiones OCR: 8<->B, 0<->O).
"""

import re
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rapidfuzz import fuzz
from modules.database import (
    buscar_por_placa_indexada,
    obtener_todas_las_placas,
    poblar_datos_prueba
)

# Expresiones regulares oficiales de Colombia
PATRON_CARRO = re.compile(r'^[A-Z]{3}[0-9]{3}$')
PATRON_MOTO = re.compile(r'^[A-Z]{3}[0-9]{2}[A-Z]$')

CONFUSION_OCR = {
    'O': '0', '0': 'O',
    'I': '1', '1': 'I',
    'B': '8', '8': 'B',
    'S': '5', '5': 'S',
    'Z': '2', '2': 'Z',
    'A': '4', '4': 'A',
    'G': '6', '6': 'G'
}


def normalizar_placa(placa_raw: str) -> str:
    """Limpia espacios, guiones y caracteres no alfanuméricos."""
    if not placa_raw:
        return ""
    return re.sub(r'[^A-Z0-9]', '', str(placa_raw).upper().strip())


def validar_regla_sintactica(placa_limpia: str) -> tuple[bool, str]:
    """
    Capa 1: Valida que la placa cumpla estrictamente con la norma colombiana.
    Retorna (es_valido, tipo_vehiculo).
    """
    if PATRON_CARRO.match(placa_limpia):
        return True, "CARRO"
    if PATRON_MOTO.match(placa_limpia):
        return True, "MOTO"
    return False, "INVALIDO"


def calcular_nivel_alerta(datos_vehiculo: dict) -> str:
    """
    Calcula el nivel de alerta policial y de tránsito del vehículo:
    - ROJO: Reporte de hurto activo u orden de inmovilización.
    - AMARILLO: Multas SIMIT pendientes o documentos (SOAT/Tecno) vencidos.
    - VERDE: Vehículo 100% al día y sin novedades.
    """
    if not datos_vehiculo:
        return "VERDE"

    if datos_vehiculo.get("reporte_hurto") == 1 or datos_vehiculo.get("orden_inmovilizacion") == 1:
        return "ROJO"

    tiene_multas = datos_vehiculo.get("multas_simit", 0) > 0
    soat_vencido = datos_vehiculo.get("soat_vigente", 1) == 0
    tecno_vencida = datos_vehiculo.get("tecnomecanica_vigente", 1) == 0

    if tiene_multas or soat_vencido or tecno_vencida:
        return "AMARILLO"

    return "VERDE"


def buscar_por_similitud(placa_busqueda: str, umbral: float = 82.0) -> tuple[dict | None, float]:
    """
    Capa 3: Búsqueda difusa ante errores visuales o suciedad en la placa.
    Aplica RapidFuzz sobre el universo de placas registradas.
    """
    todas_placas = obtener_todas_las_placas()
    if not todas_placas:
        return None, 0.0

    mejor_placa = None
    mejor_similitud = 0.0

    for cand in todas_placas:
        score = fuzz.ratio(placa_busqueda, cand)
        if score > mejor_similitud:
            mejor_similitud = score
            mejor_placa = cand

    if mejor_placa and mejor_similitud >= umbral:
        datos = buscar_por_placa_indexada(mejor_placa)
        return datos, mejor_similitud

    return None, 0.0


def consultar_placa(placa_raw: str) -> dict:
    """
    CONTRATO DE INTEGRACIÓN MVP (Persona B):
    Ejecuta la consulta en cascada a través de las 3 capas:
    1. Filtrado sintáctico por reglas colombianas.
    2. Búsqueda indexada exacta por SQLite (< 0.2 ms).
    3. Búsqueda por similitud (tolerancia a fallos de OCR) con RapidFuzz.
    """
    placa_limpia = normalizar_placa(placa_raw)

    # 1. Capa 1: Validación Sintáctica
    es_valida, tipo_veh = validar_regla_sintactica(placa_limpia)
    if not es_valida and len(placa_limpia) != 6:
        return {
            "placa_consultada": placa_limpia,
            "metodo_usado": "DESCARTADA_REGLA",
            "encontrado": False,
            "nivel_alerta": "VERDE",
            "similitud": 0.0,
            "datos": None,
            "mensaje": f"La placa '{placa_limpia}' no cumple la norma sintáctica colombiana."
        }

    # 2. Capa 2: Búsqueda Indexada (Exacta)
    datos_exactos = buscar_por_placa_indexada(placa_limpia)
    if datos_exactos:
        alerta = calcular_nivel_alerta(datos_exactos)
        return {
            "placa_consultada": placa_limpia,
            "placa_encontrada": datos_exactos["placa"],
            "metodo_usado": "INDEXADA",
            "encontrado": True,
            "nivel_alerta": alerta,
            "similitud": 100.0,
            "datos": datos_exactos,
            "mensaje": "Vehículo localizado por Búsqueda Indexada en SQLite."
        }

    # 3. Capa 3: Búsqueda por Similitud (Fuzzy Matching ante errores de lectura OCR)
    datos_similares, score_similitud = buscar_por_similitud(placa_limpia, umbral=80.0)
    if datos_similares:
        alerta = calcular_nivel_alerta(datos_similares)
        return {
            "placa_consultada": placa_limpia,
            "placa_encontrada": datos_similares["placa"],
            "metodo_usado": "SIMILITUD",
            "encontrado": True,
            "nivel_alerta": alerta,
            "similitud": round(score_similitud, 1),
            "datos": datos_similares,
            "mensaje": f"Coincidencia difusa ({score_similitud:.0f}% similitud con {datos_similares['placa']})."
        }

    # No registrado en base de datos de antecedentes
    return {
        "placa_consultada": placa_limpia,
        "placa_encontrada": None,
        "metodo_usado": "INDEXADA",
        "encontrado": False,
        "nivel_alerta": "VERDE",
        "similitud": 0.0,
        "datos": {
            "placa": placa_limpia,
            "tipo": tipo_veh,
            "marca": "NO REGISTRADO EN ANTECEDENTES",
            "linea": "Vehículo no censado en base local",
            "modelo": 0,
            "color": "DESCONOCIDO",
            "propietario": "SIN REGISTRO POLICIAL",
            "cedula_propietario": "N/A",
            "municipio_registro": "COLOMBIA",
            "multas_simit": 0,
            "valor_multas": 0,
            "detalle_multas": "Sin registros en base de datos",
            "soat_vigente": 1,
            "soat_vencimiento": "N/A",
            "tecnomecanica_vigente": 1,
            "tecnomecanica_vencimiento": "N/A",
            "reporte_hurto": 0,
            "orden_inmovilizacion": 0,
            "observaciones": "Placa detectada pero no figura en base de datos local."
        },
        "mensaje": "Placa no encontrada en la base de datos de antecedentes."
    }


if __name__ == "__main__":
    poblar_datos_prueba()

    print("\n--- PRUEBA 1: Búsqueda Indexada Exacta ---")
    res1 = consultar_placa("ABC123")
    print("Metodo:", res1["metodo_usado"], "| Alerta:", res1["nivel_alerta"], "| Propietario:", res1["datos"]["propietario"])

    print("\n--- PRUEBA 2: Búsqueda por Similitud (Confusión OCR: 8 por B en A8C123) ---")
    res2 = consultar_placa("A8C123")
    print("Consultada: A8C123 -> Encontrada:", res2.get("placa_encontrada"), "| Metodo:", res2["metodo_usado"], f"({res2['similitud']}%)")

    print("\n--- PRUEBA 3: Alerta Roja (Vehículo Hurtado) ---")
    res3 = consultar_placa("FRT890")
    print("Placa: FRT890 | Nivel:", res3["nivel_alerta"], "| Obs:", res3["datos"]["observaciones"])
