"""
Suite de Pruebas y Validación Académica de Inteligencia Computacional (CI Test Suite).
Verifica:
1. Inferencia Difusa Mamdani con Defusificación Centroid ante diferentes grados de incertidumbre.
2. Optimización Heurística Bio-inspirada (Algoritmos Genéticos y PSO).
3. Red Neuronal MLP entrenada con Backpropagation y cálculo de matriz de confusión / F1-Score.
4. Integración completa con el motor de búsqueda en 3 capas.
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.fuzzy_inference import sistema_difuso
from modules.heuristic_optimizer import GeneticImageOptimizer, ParticleSwarmOptimizer
from modules.ml_classifier import clasificador_riesgo
from modules.search_engine import consultar_placa


def test_fuzzy_inference():
    print("\n[1/4] Evaluando Sistema de Inferencia Difusa (Fuzzy Logic - Mamdani)...")
    casos = [
        ("Vehículo al día", 0.95, 0, 0, 1, 1, 0, 0),
        ("Multas moderadas", 0.85, 2, 850_000, 1, 1, 0, 0),
        ("Infracciones graves sin SOAT ni Tecno", 0.90, 5, 2_800_000, 0, 0, 0, 0),
        ("Reporte judicial de hurto", 0.98, 0, 0, 1, 1, 1, 0)
    ]
    for desc, conf, multas, val, soat, tecno, hurto, inmov in casos:
        res = sistema_difuso.inferir_riesgo(conf, multas, val, soat, tecno, hurto, inmov)
        print(f"  - {desc:38} -> Riesgo: {res['score']:5.2f}/100 | Categoría: {res['categoria']:22} | Decisión: {res['decision']}")
        assert 0.0 <= res['score'] <= 100.0, "Score fuera de rango difuso"
    print("  -> Logica Difusa: VALIDADA CON EXITO (OK)")


def test_heuristic_optimization():
    print("\n[2/4] Evaluando Optimización Bio-inspirada (Algoritmos Genéticos y PSO)...")
    sample_img = np.random.randint(30, 220, (64, 180, 3), dtype=np.uint8)
    
    # Algoritmo Genético
    ga = GeneticImageOptimizer(population_size=12, generations=6, mutation_rate=0.2)
    ga_params, ga_history = ga.optimize(sample_img)
    print(f"  - GA Parámetros Óptimos de Imagen : {ga_params}")
    print(f"  - GA Evolución de Fitness Inicial -> Final: {ga_history[0]:.2f} -> {ga_history[-1]:.2f}")
    assert ga_history[-1] >= ga_history[0], "El Algoritmo Genético debe mantener o mejorar la aptitud"

    # Enjambre de Partículas (PSO)
    pso = ParticleSwarmOptimizer(num_particles=10, max_iter=8)
    pso_params = pso.optimize_tracking([])
    print(f"  - PSO Parámetros Óptimos Tracking: {pso_params}")
    assert 0.1 <= pso_params["iou_threshold"] <= 0.6, "Umbral IoU fuera de rango óptimo"
    print("  -> Optimización Heurística: VALIDADA CON EXITO (OK)")


def test_neural_network_and_metrics():
    print("\n[3/4] Evaluando Red Neuronal MLP y Métricas de Clasificación...")
    metrics = clasificador_riesgo.train_and_evaluate()
    print(f"  - Muestras Entrenamiento : {metrics['train_samples']} | Muestras Prueba: {metrics['test_samples']}")
    print(f"  - Pérdida BCE Final      : {metrics['final_bce_loss']:.4f}")
    print(f"  - Matriz de Confusión    : TP={metrics['confusion_matrix']['TP']}, TN={metrics['confusion_matrix']['TN']}, FP={metrics['confusion_matrix']['FP']}, FN={metrics['confusion_matrix']['FN']}")
    print(f"  - Accuracy               : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  - Precision              : {metrics['precision'] * 100:.2f}%")
    print(f"  - Recall                 : {metrics['recall'] * 100:.2f}%")
    print(f"  - F1-Score               : {metrics['f1_score'] * 100:.2f}%")
    assert metrics['test_accuracy'] >= 0.85, "Accuracy mínimo no alcanzado"
    print("  -> Red Neuronal y Métricas Formales: VALIDADAS CON EXITO (OK)")


def test_end_to_end_search_engine():
    print("\n[4/4] Evaluando Pipeline Integrado de Inteligencia Computacional...")
    placas = ["ABC123", "FRT890", "XYZ45E", "A8C123"]
    for p in placas:
        res = consultar_placa(p, ocr_confidence=0.92)
        print(f"  - Placa: {p:6} | Método: {res['metodo_usado']:9} | Alerta: {res['nivel_alerta']:8} | Riesgo Difuso: {res['evaluacion_difusa']['score']:5.2f} | ML: {res['prediccion_ml']['perfil_riesgo']}")
        assert "evaluacion_difusa" in res and "prediccion_ml" in res, "Faltan contratos de IC en la respuesta"
    print("  -> Pipeline End-to-End: VALIDADO CON EXITO (OK)")


if __name__ == "__main__":
    print("=" * 75)
    print("SUITE DE VERIFICACION CIENTIFICA DE INTELIGENCIA COMPUTACIONAL (ANPR COLOMBIA)")
    print("=" * 75)
    test_fuzzy_inference()
    test_heuristic_optimization()
    test_neural_network_and_metrics()
    test_end_to_end_search_engine()
    print("\n" + "=" * 75)
    print("TODAS LAS PRUEBAS DE INTELIGENCIA COMPUTACIONAL HAN SIDO SUPERADAS (100% OK)")
    print("=" * 75)
