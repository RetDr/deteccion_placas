"""
Utilidad CLI interactiva para registrar o actualizar vehículos en antecedentes.db.
Permite registrar placas rápidamente sin errores de indentación.
Uso:
  python registrar_placa.py
  python registrar_placa.py --placa ABC123 --marca Mazda --linea "Mazda 3" --modelo 2022
"""

import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.database import registrar_o_actualizar_vehiculo
from modules.search_engine import consultar_placa


def main():
    parser = argparse.ArgumentParser(description="Registrar vehiculo en la base de datos de antecedentes.")
    parser.add_argument("--placa", type=str, help="Placa (ej: OMG650)")
    parser.add_argument("--tipo", type=str, default="CARRO", choices=["CARRO", "MOTO"], help="Tipo de vehiculo")
    parser.add_argument("--marca", type=str, default="Renault", help="Marca")
    parser.add_argument("--linea", type=str, default="Sandero Stepway", help="Linea / Modelo comercial")
    parser.add_argument("--modelo", type=int, default=2023, help="Ano del modelo")
    parser.add_argument("--color", type=str, default="Gris", help="Color")
    parser.add_argument("--propietario", type=str, default="Propietario de Prueba", help="Nombre del propietario")
    parser.add_argument("--cedula", type=str, default="1.020.304.050", help="Documento de identidad")
    parser.add_argument("--municipio", type=str, default="MEDELLIN", help="Municipio de matricula")
    parser.add_argument("--multas", type=int, default=0, help="Cantidad de multas SIMIT")
    parser.add_argument("--valor-multas", type=int, default=0, help="Valor total de multas en COP")
    parser.add_argument("--detalle-multas", type=str, default="", help="Detalle de las infracciones")
    parser.add_argument("--soat-vigente", type=int, default=1, choices=[0, 1], help="1=Vigente, 0=Vencido")
    parser.add_argument("--tecno-vigente", type=int, default=1, choices=[0, 1], help="1=Vigente, 0=Vencida")
    parser.add_argument("--hurto", type=int, default=0, choices=[0, 1], help="1=Reporte de hurto activo, 0=Limpio")
    parser.add_argument("--inmovilizacion", type=int, default=0, choices=[0, 1], help="1=Orden judicial activa, 0=Sin orden")
    parser.add_argument("--observaciones", type=str, default="", help="Notas u observaciones de policia/transito")

    args = parser.parse_args()

    placa = args.placa
    if not placa:
        print("\n" + "=" * 60)
        print("REGISTRO RAPIDO DE VEHICULO EN BASE DE DATOS DE ANTECEDENTES")
        print("=" * 60)
        placa = input("Placa (ej: OMG650): ").strip().upper()
        if not placa:
            print("[ERROR] La placa es obligatoria.")
            return

        tipo = input("Tipo [CARRO / MOTO] (default: CARRO): ").strip().upper() or "CARRO"
        marca = input("Marca (ej: Renault): ").strip() or "Renault"
        linea = input("Linea (ej: Sandero Stepway): ").strip() or "Sandero Stepway"
        modelo_str = input("Ano modelo (ej: 2023): ").strip() or "2023"
        color = input("Color (ej: Gris): ").strip() or "Gris"
        propietario = input("Propietario (ej: Carlos Restrepo): ").strip() or "Propietario de Prueba"
        cedula = input("Cedula propietario (ej: 1.020.304.050): ").strip() or "1.020.304.050"
        municipio = input("Municipio registro (ej: MEDELLIN): ").strip() or "MEDELLIN"
        multas_str = input("Cantidad multas SIMIT [0..N] (default: 0): ").strip() or "0"
        valor_str = input("Valor total multas COP (default: 0): ").strip() or "0"
        soat_str = input("SOAT vigente [1=Si / 0=No] (default: 1): ").strip() or "1"
        hurto_str = input("Reporte de hurto [1=Si / 0=No] (default: 0): ").strip() or "0"
        obs = input("Observaciones (opcional): ").strip() or "Registro agregado manualmente."

        datos = {
            "placa": placa,
            "tipo": tipo,
            "marca": marca,
            "linea": linea,
            "modelo": int(modelo_str),
            "color": color,
            "propietario": propietario,
            "cedula_propietario": cedula,
            "municipio_registro": municipio,
            "multas_simit": int(multas_str),
            "valor_multas": int(valor_str),
            "detalle_multas": "Comparendo registrado" if int(multas_str) > 0 else "Sin infracciones",
            "soat_vigente": int(soat_str),
            "soat_vencimiento": "2027-04-10" if int(soat_str) == 1 else "2026-01-01",
            "tecnomecanica_vigente": 1,
            "tecnomecanica_vencimiento": "2027-04-10",
            "reporte_hurto": int(hurto_str),
            "orden_inmovilizacion": int(hurto_str),
            "observaciones": obs
        }
    else:
        datos = {
            "placa": placa.upper(),
            "tipo": args.tipo,
            "marca": args.marca,
            "linea": args.linea,
            "modelo": args.modelo,
            "color": args.color,
            "propietario": args.propietario,
            "cedula_propietario": args.cedula,
            "municipio_registro": args.municipio,
            "multas_simit": args.multas,
            "valor_multas": args.valor_multas,
            "detalle_multas": args.detalle_multas or ("Comparendos activos" if args.multas > 0 else "Sin multas"),
            "soat_vigente": args.soat_vigente,
            "soat_vencimiento": "2027-04-10" if args.soat_vigente == 1 else "2026-01-01",
            "tecnomecanica_vigente": args.tecno_vigente,
            "tecnomecanica_vencimiento": "2027-04-10" if args.tecno_vigente == 1 else "2026-01-01",
            "reporte_hurto": args.hurto,
            "orden_inmovilizacion": args.inmovilizacion,
            "observaciones": args.observaciones
        }

    ok = registrar_o_actualizar_vehiculo(datos)
    if ok:
        print(f"\n[EXITO] Placa '{datos['placa']}' registrada correctamente en data/antecedentes.db.")
        consulta = consultar_placa(datos['placa'])
        print(f"Estado de alerta asignado: [{consulta['nivel_alerta']}]")
        print(f"Vehiculo: {datos['marca']} {datos['linea']} ({datos['modelo']})")
        print(f"Propietario: {datos['propietario']} | Multas: {datos['multas_simit']} (${datos['valor_multas']:,} COP)")
        print(f"SOAT: {'VIGENTE' if datos['soat_vigente'] else 'VENCIDO'} | Hurto: {'SI' if datos['reporte_hurto'] else 'NO'}\n")
    else:
        print(f"\n[ERROR] No se pudo guardar la placa '{datos['placa']}'.\n")


if __name__ == "__main__":
    main()
