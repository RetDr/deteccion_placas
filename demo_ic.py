"""
Script de Demostración Interactiva de Inteligencia Computacional.
Ejecuta y muestra en consola paso a paso:
1. Inferencia Difusa (Mamdani) evaluando casos reales.
2. Optimización Bio-inspirada con Algoritmo Genético (evolución de aptitud).
3. Red Neuronal MLP: Entrenamiento con Backpropagation y matriz de confusión.
4. Consulta Inteligente integrada con el motor de placas.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from test_computational_intelligence import (
    test_fuzzy_inference,
    test_heuristic_optimization,
    test_neural_network_and_metrics,
    test_end_to_end_search_engine
)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEMOSTRACION INTERACTIVA: SISTEMA DE INTELIGENCIA COMPUTACIONAL (ANPR)")
    print("=" * 80)
    
    test_fuzzy_inference()
    input("\nPresiona ENTER para continuar a la Optimizacion Bio-inspirada (GA / PSO)...")
    
    test_heuristic_optimization()
    input("\nPresiona ENTER para continuar al Entrenamiento de la Red Neuronal (MLP)...")
    
    test_neural_network_and_metrics()
    input("\nPresiona ENTER para probar la Consulta Integrada de Placas...")
    
    test_end_to_end_search_engine()
    
    print("\n" + "=" * 80)
    print("DEMOSTRACION FINALIZADA CON EXITO")
    print("=" * 80 + "\n")
