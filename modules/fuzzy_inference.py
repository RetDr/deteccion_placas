"""
Módulo de Inteligencia Computacional: Sistema de Inferencia Difusa (Fuzzy Inference System - Mamdani)
Evalúa el Nivel de Riesgo y Prioridad de Intervención Vehicular modelando la incertidumbre
asociada a la confianza del OCR, gravedad/cuantía de multas, antigüedad y antecedentes.

Cumple con los fundamentos teóricos de Soft Computing / Inteligencia Computacional:
- Funciones de pertenencia (Triangulares y Trapezoidales).
- Operadores T-norma (Mínimo de Gödel / Zadeh) y S-norma (Máximo).
- Base de reglas difusas lingüísticas (IF-THEN).
- Agregación e Inferencia tipo Mamdani.
- Defusificación por Centro de Gravedad (Centroid / COG).
"""

import numpy as np


class MembershipFunction:
    """Clase base para funciones de pertenencia difusa."""
    def evaluate(self, x: float) -> float:
        raise NotImplementedError


class TriangularMF(MembershipFunction):
    def __init__(self, a: float, b: float, c: float):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def evaluate(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 1.0
        else:
            return (self.c - x) / (self.c - self.b) if self.c != self.b else 1.0


class TrapezoidalMF(MembershipFunction):
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def evaluate(self, x: float) -> float:
        if x < self.a or x > self.d:
            return 0.0
        elif self.a <= x <= self.b:
            return 1.0 if self.b == self.a else (x - self.a) / (self.b - self.a)
        elif self.b <= x <= self.c:
            return 1.0
        else:
            return 1.0 if self.d == self.c else (self.d - x) / (self.d - self.c)


class FuzzyVehicleRiskEngine:
    """
    Sistema de Inferencia Difuso Mamdani para Evaluación de Riesgo de Tránsito.
    Entradas:
      1. ocr_confidence: [0.0 - 1.0] Fiabilidad de la lectura del sensor/cámara.
      2. fine_severity: [0.0 - 100.0] Valor escalar compuesto por número de multas, valor económico y vigencia de SOAT/Tecno.
      3. judicial_severity: [0.0 - 100.0] Severidad de orden de inmovilización o reporte de hurto.
    Salida:
      - risk_score: [0.0 - 100.0] Nivel de Riesgo Operativo / Prioridad de Inmovilización.
      - risk_category: "BAJO", "MEDIO", "ALTO", "CRITICO_INMOVILIZACION"
    """

    def __init__(self):
        # Universo de discurso para la salida (0 a 100 con resolución 0.5)
        self.output_universe = np.linspace(0, 100, 201)

        # Variables Difusas de Entrada 1: Confianza OCR (Incertidumbre de percepción)
        self.mf_conf_baja = TrapezoidalMF(0.0, 0.0, 0.35, 0.55)
        self.mf_conf_media = TriangularMF(0.40, 0.65, 0.85)
        self.mf_conf_alta = TrapezoidalMF(0.70, 0.85, 1.0, 1.0)

        # Variables Difusas de Entrada 2: Severidad de Tránsito (Multas + Documentos)
        self.mf_transito_leve = TrapezoidalMF(0.0, 0.0, 15.0, 35.0)
        self.mf_transito_moderado = TriangularMF(25.0, 50.0, 75.0)
        self.mf_transito_grave = TrapezoidalMF(60.0, 80.0, 100.0, 100.0)

        # Variables Difusas de Entrada 3: Severidad Judicial / Hurto
        self.mf_judicial_nula = TrapezoidalMF(0.0, 0.0, 10.0, 25.0)
        self.mf_judicial_critica = TrapezoidalMF(40.0, 70.0, 100.0, 100.0)

        # Variable Difusa de Salida: Nivel de Riesgo Operativo (Mamdani)
        self.mf_riesgo_bajo = TrapezoidalMF(0.0, 0.0, 15.0, 35.0)
        self.mf_riesgo_medio = TriangularMF(25.0, 50.0, 70.0)
        self.mf_riesgo_alto = TriangularMF(60.0, 75.0, 90.0)
        self.mf_riesgo_critico = TrapezoidalMF(75.0, 90.0, 100.0, 100.0)

    def calcular_severidad_transito(self, multas_count: int, valor_multas: int, soat_vigente: int, tecno_vigente: int) -> float:
        """Convierte las infracciones de tránsito en una escala normalizada [0, 100]."""
        score = 0.0
        # Multas (hasta 40 pts)
        score += min(35.0, multas_count * 10.0)
        # Monto económico (hasta 25 pts)
        if valor_multas > 0:
            score += min(25.0, (valor_multas / 2_500_000.0) * 25.0)
        # Documentos vencidos (hasta 40 pts)
        if soat_vigente == 0:
            score += 20.0
        if tecno_vigente == 0:
            score += 20.0
        return float(min(100.0, max(0.0, score)))

    def calcular_severidad_judicial(self, reporte_hurto: int, orden_inmovilizacion: int) -> float:
        """Determina la severidad legal inmediata."""
        if reporte_hurto == 1:
            return 100.0
        if orden_inmovilizacion == 1:
            return 85.0
        return 0.0

    def inferir_riesgo(self, ocr_conf: float, multas_count: int, valor_multas: int,
                       soat_vigente: int, tecno_vigente: int,
                       reporte_hurto: int, orden_inmovilizacion: int) -> dict:
        """
        Ejecuta el motor de inferencia difusa completo:
        Fuzzificación -> Evaluación de Reglas -> Agregación -> Defusificación Centroid.
        """
        # Normalizar entradas
        conf = float(np.clip(ocr_conf, 0.0, 1.0))
        sev_transito = self.calcular_severidad_transito(multas_count, valor_multas, soat_vigente, tecno_vigente)
        sev_judicial = self.calcular_severidad_judicial(reporte_hurto, orden_inmovilizacion)

        # 1. Fuzzificación
        u_conf_baja = self.mf_conf_baja.evaluate(conf)
        u_conf_media = self.mf_conf_media.evaluate(conf)
        u_conf_alta = self.mf_conf_alta.evaluate(conf)

        u_tran_leve = self.mf_transito_leve.evaluate(sev_transito)
        u_tran_mod = self.mf_transito_moderado.evaluate(sev_transito)
        u_tran_gra = self.mf_transito_grave.evaluate(sev_transito)

        u_jud_nula = self.mf_judicial_nula.evaluate(sev_judicial)
        u_jud_crit = self.mf_judicial_critica.evaluate(sev_judicial)

        # 2. Base de Reglas Lingüísticas (Mamdani T-norma: min)
        # Regla 1: Si Alerta Judicial es Crítica y Confianza es Media/Alta -> Riesgo Crítico Inmediato
        r1 = min(u_jud_crit, max(u_conf_media, u_conf_alta))

        # Regla 2: Si Alerta Judicial es Crítica pero Confianza es Baja -> Riesgo Alto (Incertidumbre de sensor)
        r2 = min(u_jud_crit, u_conf_baja)

        # Regla 3: Si Tránsito es Grave y Confianza es Alta -> Riesgo Alto
        r3 = min(u_tran_gra, u_conf_alta, u_jud_nula)

        # Regla 4: Si Tránsito es Grave y Confianza es Media -> Riesgo Medio-Alto
        r4 = min(u_tran_gra, u_conf_media, u_jud_nula)

        # Regla 5: Si Tránsito es Moderado y Confianza Alta/Media -> Riesgo Medio
        r5 = min(u_tran_mod, max(u_conf_alta, u_conf_media), u_jud_nula)

        # Regla 6: Si Tránsito es Leve y Judicial es Nula -> Riesgo Bajo
        r6 = min(u_tran_leve, u_jud_nula)

        # Regla 7: Si Confianza es Baja y Tránsito Moderado -> Riesgo Bajo (no alarmar sin certeza)
        r7 = min(u_conf_baja, u_tran_mod, u_jud_nula)

        # Agregación de activaciones por consecuente (S-norma: max)
        act_critico = r1
        act_alto = max(r2, r3)
        act_medio = max(r4, r5)
        act_bajo = max(r6, r7)

        # 3. Composición difusa sobre el universo de discurso
        curva_agregada = np.zeros_like(self.output_universe)
        for idx, x in enumerate(self.output_universe):
            u_b = min(act_bajo, self.mf_riesgo_bajo.evaluate(x))
            u_m = min(act_medio, self.mf_riesgo_medio.evaluate(x))
            u_a = min(act_alto, self.mf_riesgo_alto.evaluate(x))
            u_c = min(act_critico, self.mf_riesgo_critico.evaluate(x))
            curva_agregada[idx] = max(u_b, u_m, u_a, u_c)

        # 4. Defusificación: Centroide de Gravedad (COG)
        sum_curva = np.sum(curva_agregada)
        if sum_curva == 0:
            risk_score = 10.0  # Base segura por defecto
        else:
            risk_score = float(np.sum(self.output_universe * curva_agregada) / sum_curva)

        # Interpretación categórica
        if risk_score >= 75.0:
            category = "CRITICO_INMOVILIZACION"
            color_ui = "#ef4444"  # Rojo
            decision = "Detención Inmediata e Inmovilización"
        elif risk_score >= 50.0:
            category = "ALTO"
            color_ui = "#f97316"  # Naranja
            decision = "Inspección Policial Requerida"
        elif risk_score >= 25.0:
            category = "MEDIO"
            color_ui = "#eab308"  # Amarillo
            decision = "Alerta de Tránsito Preventiva"
        else:
            category = "BAJO"
            color_ui = "#22c55e"  # Verde
            decision = "Vehículo en Regla / Paso Permitido"

        return {
            "score": round(risk_score, 2),
            "categoria": category,
            "decision": decision,
            "color_ui": color_ui,
            "detalles_fuzzificacion": {
                "confianza_ocr": round(conf, 3),
                "severidad_transito": round(sev_transito, 1),
                "severidad_judicial": round(sev_judicial, 1),
                "activacion_reglas": {
                    "bajo": round(float(act_bajo), 3),
                    "medio": round(float(act_medio), 3),
                    "alto": round(float(act_alto), 3),
                    "critico": round(float(act_critico), 3)
                }
            }
        }


# Instancia singleton para uso rápido en el sistema
sistema_difuso = FuzzyVehicleRiskEngine()
