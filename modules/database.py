"""
Módulo de Base de Datos SQLite para Antecedentes Vehiculares (SIMIT, RUNT, Policía).
Implementa la Búsqueda Indexada (Capa 2 del Sistema) con índice B-Tree en la columna 'placa'.
Cumple con la especificación del MVP (Persona B).
"""

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "antecedentes.db")


def obtener_conexion() -> sqlite3.Connection:
    """Retorna una conexión a la base de datos SQLite con soporte para diccionarios (Row)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def inicializar_bd(forzar_recreacion: bool = False):
    """
    Crea la tabla 'vehiculos' con índice B-Tree en la columna 'placa'
    para permitir búsquedas en menos de 0.2 milisegundos.
    """
    conn = obtener_conexion()
    cursor = conn.cursor()

    if forzar_recreacion:
        cursor.execute("DROP TABLE IF EXISTS vehiculos;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehiculos (
        placa TEXT PRIMARY KEY NOT NULL,
        tipo TEXT NOT NULL DEFAULT 'CARRO',
        marca TEXT NOT NULL,
        linea TEXT NOT NULL,
        modelo INTEGER NOT NULL,
        color TEXT NOT NULL,
        propietario TEXT NOT NULL,
        cedula_propietario TEXT NOT NULL,
        municipio_registro TEXT NOT NULL,
        multas_simit INTEGER NOT NULL DEFAULT 0,
        valor_multas INTEGER NOT NULL DEFAULT 0,
        detalle_multas TEXT DEFAULT '',
        soat_vigente INTEGER NOT NULL DEFAULT 1,
        soat_vencimiento TEXT DEFAULT '',
        tecnomecanica_vigente INTEGER NOT NULL DEFAULT 1,
        tecnomecanica_vencimiento TEXT DEFAULT '',
        reporte_hurto INTEGER NOT NULL DEFAULT 0,
        orden_inmovilizacion INTEGER NOT NULL DEFAULT 0,
        observaciones TEXT DEFAULT ''
    );
    """)

    # ÍNDICE B-TREE: Cumplimiento formal del requisito de Búsqueda Indexada ultra rápida
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_vehiculos_placa ON vehiculos (placa);
    """)

    conn.commit()
    conn.close()


def poblar_datos_prueba(sobrescribir: bool = False):
    """
    Puebla la base de datos con un conjunto representativo de vehículos colombianos:
    - Casos VERDES: Vehículos limpios al día (SOAT, Tecno, 0 multas).
    - Casos AMARILLOS: Multas SIMIT pendientes o documentos vencidos.
    - Casos ROJOS: Reporte de hurto activo u orden de inmovilización.
    Incluye las placas de prueba del sistema (ABC123, XYZ45E, etc.).
    """
    inicializar_bd(forzar_recreacion=sobrescribir)
    conn = obtener_conexion()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM vehiculos;")
    if cursor.fetchone()[0] > 0 and not sobrescribir:
        conn.close()
        return

    vehiculos_seed = [
        # 1. Caso Verde: Carro al día sin multas
        {
            "placa": "ABC123",
            "tipo": "CARRO",
            "marca": "Mazda",
            "linea": "Mazda 3 Grand Touring",
            "modelo": 2022,
            "color": "Rojo Diamante",
            "propietario": "Carlos Mario Restrepo",
            "cedula_propietario": "71.234.567",
            "municipio_registro": "BOGOTA D.C.",
            "multas_simit": 0,
            "valor_multas": 0,
            "detalle_multas": "Sin comparendos pendientes en SIMIT",
            "soat_vigente": 1,
            "soat_vencimiento": "2027-08-14",
            "tecnomecanica_vigente": 1,
            "tecnomecanica_vencimiento": "2027-02-20",
            "reporte_hurto": 0,
            "orden_inmovilizacion": 0,
            "observaciones": "Vehículo al día en obligaciones tributarias y de tránsito."
        },
        # 2. Caso Amarillo: Moto con multas SIMIT y SOAT vencido
        {
            "placa": "XYZ45E",
            "tipo": "MOTO",
            "marca": "Yamaha",
            "linea": "MT-03",
            "modelo": 2023,
            "color": "Azul Metálico",
            "propietario": "Valentina Gómez Londoño",
            "cedula_propietario": "1.037.654.321",
            "municipio_registro": "MEDELLIN",
            "multas_simit": 2,
            "valor_multas": 1160000,
            "detalle_multas": "Comparendo C02 (Estacionar en sitio prohibido) + C29 (Exceso de velocidad)",
            "soat_vigente": 0,
            "soat_vencimiento": "2026-10-01",
            "tecnomecanica_vigente": 1,
            "tecnomecanica_vencimiento": "2027-01-15",
            "reporte_hurto": 0,
            "orden_inmovilizacion": 0,
            "observaciones": "SOAT vencido hace 51 días. Requiere atención inmediata."
        },
        # 3. Caso Rojo: Carro con Reporte de Hurto activo y orden judicial
        {
            "placa": "FRT890",
            "tipo": "CARRO",
            "marca": "Toyota",
            "linea": "Hilux 4x4 Diésel",
            "modelo": 2021,
            "color": "Blanco Perlado",
            "propietario": "Distribuciones del Norte S.A.S.",
            "cedula_propietario": "900.876.543-1",
            "municipio_registro": "CALI",
            "multas_simit": 1,
            "valor_multas": 580000,
            "detalle_multas": "Comparendo C14 (Transitar por sitios restringidos)",
            "soat_vigente": 1,
            "soat_vencimiento": "2027-03-30",
            "tecnomecanica_vigente": 1,
            "tecnomecanica_vencimiento": "2027-04-10",
            "reporte_hurto": 1,
            "orden_inmovilizacion": 1,
            "observaciones": "ALERTA POLICIA NACIONAL: Hurto denunciado el 15/11/2026 en Cali. Notificar a cuadrante."
        },
        # 4. Caso Amarillo: Carro con Tecnomecánica vencida
        {
            "placa": "KLR650",
            "tipo": "CARRO",
            "marca": "Chevrolet",
            "linea": "Tracker Turbo",
            "modelo": 2020,
            "color": "Gris Ocaso",
            "propietario": "Andrés Felipe Jaramillo",
            "cedula_propietario": "98.765.432",
            "municipio_registro": "ENVIGADO",
            "multas_simit": 0,
            "valor_multas": 0,
            "detalle_multas": "Sin comparendos pendientes",
            "soat_vigente": 1,
            "soat_vencimiento": "2027-06-12",
            "tecnomecanica_vigente": 0,
            "tecnomecanica_vencimiento": "2026-09-30",
            "reporte_hurto": 0,
            "orden_inmovilizacion": 0,
            "observaciones": "Revisión Técnico-Mecánica vencida."
        },
        # 5. Caso Verde: Carro particular al día
        {
            "placa": "HYU789",
            "tipo": "CARRO",
            "marca": "Renault",
            "linea": "Duster Zen",
            "modelo": 2022,
            "color": "Negro Nácar",
            "propietario": "Martha Lucía Caicedo",
            "cedula_propietario": "43.567.890",
            "municipio_registro": "BARRANQUILLA",
            "multas_simit": 0,
            "valor_multas": 0,
            "detalle_multas": "Sin comparendos pendientes",
            "soat_vigente": 1,
            "soat_vencimiento": "2027-11-05",
            "tecnomecanica_vigente": 1,
            "tecnomecanica_vencimiento": "2027-11-05",
            "reporte_hurto": 0,
            "orden_inmovilizacion": 0,
            "observaciones": "Sin novedades."
        },
        # 6. Caso Rojo: Moto hurtada
        {
            "placa": "DKQ98D",
            "tipo": "MOTO",
            "marca": "Bajaj",
            "linea": "Pulsar NS 200",
            "modelo": 2021,
            "color": "Negro / Rojo",
            "propietario": "Santiago Echeverri",
            "cedula_propietario": "1.152.345.678",
            "municipio_registro": "ITAGUI",
            "multas_simit": 3,
            "valor_multas": 1740000,
            "detalle_multas": "Múltiples comparendos por no acatar señales de tránsito",
            "soat_vigente": 0,
            "soat_vencimiento": "2026-05-10",
            "tecnomecanica_vigente": 0,
            "tecnomecanica_vencimiento": "2026-05-10",
            "reporte_hurto": 1,
            "orden_inmovilizacion": 1,
            "observaciones": "ALERTA: Reportada como hurtada en sector El Poblado."
        }
    ]

    for v in vehiculos_seed:
        cursor.execute("""
        INSERT OR REPLACE INTO vehiculos (
            placa, tipo, marca, linea, modelo, color, propietario, cedula_propietario,
            municipio_registro, multas_simit, valor_multas, detalle_multas,
            soat_vigente, soat_vencimiento, tecnomecanica_vigente, tecnomecanica_vencimiento,
            reporte_hurto, orden_inmovilizacion, observaciones
        ) VALUES (
            :placa, :tipo, :marca, :linea, :modelo, :color, :propietario, :cedula_propietario,
            :municipio_registro, :multas_simit, :valor_multas, :detalle_multas,
            :soat_vigente, :soat_vencimiento, :tecnomecanica_vigente, :tecnomecanica_vencimiento,
            :reporte_hurto, :orden_inmovilizacion, :observaciones
        );
        """, v)

    conn.commit()
    conn.close()
    print(f"[DATABASE] Base de datos 'antecedentes.db' inicializada y poblada con {len(vehiculos_seed)} vehiculos de prueba.")


def buscar_por_placa_indexada(placa_limpia: str) -> dict | None:
    """
    Búsqueda indexada exacta por SQLite (< 0.2 ms).
    Retorna el diccionario con la ficha técnica y antecedentes o None.
    """
    if not placa_limpia:
        return None

    conn = obtener_conexion()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehiculos WHERE placa = ?;", (placa_limpia.upper().strip(),))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def obtener_todas_las_placas() -> list[str]:
    """Retorna la lista de todas las placas registradas en la base de datos."""
    conn = obtener_conexion()
    cursor = conn.cursor()
    cursor.execute("SELECT placa FROM vehiculos;")
    filas = cursor.fetchall()
    conn.close()
    return [r[0] for r in filas]


def registrar_o_actualizar_vehiculo(datos: dict) -> bool:
    """Inserta o actualiza un registro vehicular en la base de datos SQLite."""
    conn = obtener_conexion()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT OR REPLACE INTO vehiculos (
            placa, tipo, marca, linea, modelo, color, propietario, cedula_propietario,
            municipio_registro, multas_simit, valor_multas, detalle_multas,
            soat_vigente, soat_vencimiento, tecnomecanica_vigente, tecnomecanica_vencimiento,
            reporte_hurto, orden_inmovilizacion, observaciones
        ) VALUES (
            :placa, :tipo, :marca, :linea, :modelo, :color, :propietario, :cedula_propietario,
            :municipio_registro, :multas_simit, :valor_multas, :detalle_multas,
            :soat_vigente, :soat_vencimiento, :tecnomecanica_vigente, :tecnomecanica_vencimiento,
            :reporte_hurto, :orden_inmovilizacion, :observaciones
        );
        """, datos)
        conn.commit()
        return True
    except Exception as e:
        print(f"[ERROR] Error al registrar vehiculo en SQLite: {e}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    poblar_datos_prueba(sobrescribir=True)
    res = buscar_por_placa_indexada("ABC123")
    print("Prueba de consulta indexada:", res["placa"], res["marca"], res["propietario"])
