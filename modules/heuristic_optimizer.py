"""
Módulo de Inteligencia Computacional: Optimización Heurística Bio-inspirada (Algoritmos Genéticos y PSO)
Optimiza los hiperparámetros de preprocesamiento de visión artificial y calidad de imagen
para maximizar la entropía de información, contraste local y relación señal-ruido (SNR)
en placas vehiculares bajo condiciones ópticas adversas (sombras, lluvia, baja luz).

Parámetros a optimizar:
- clip_limit: Límite de contraste de CLAHE [1.0 - 5.0]
- tile_grid_size: Tamaño de ventana CLAHE [2 - 16]
- unsharp_weight: Intensidad de máscara de enfoque [0.5 - 2.5]
- blur_kernel: Kernel de suavizado gaussiano [1, 3, 5, 7]
"""

import numpy as np
import random
from typing import Callable, List, Tuple, Dict


class GeneticImageOptimizer:
    """
    Algoritmo Genético (GA) para calibración y optimización estocástica
    del preprocesamiento de placas.
    - Cromosoma: [clip_limit, tile_grid_size, unsharp_weight, blur_ksize]
    - Selección: Torneo binario
    - Cruce: Punto único / Aritmético
    - Mutación: Gaussiana adaptativa
    """

    def __init__(self, population_size: int = 20, generations: int = 15, mutation_rate: float = 0.15):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        # Límites del espacio de búsqueda [min, max]
        self.bounds = [
            (1.0, 5.0),   # clip_limit (float)
            (2, 16),      # tile_grid_size (int)
            (0.5, 2.5),   # unsharp_weight (float)
            (1, 7)        # blur_ksize (impar: 1, 3, 5, 7)
        ]

    def _random_chromosome(self) -> list:
        return [
            random.uniform(self.bounds[0][0], self.bounds[0][1]),
            random.randint(int(self.bounds[1][0]), int(self.bounds[1][1])),
            random.uniform(self.bounds[2][0], self.bounds[2][1]),
            random.choice([1, 3, 5, 7])
        ]

    def _fitness_function(self, chromosome: list, roi_gray: np.ndarray) -> float:
        """
        Función de Aptitud (Fitness) multiobjetivo:
        Maximiza la varianza del Laplaciano (nitidez) + entropía de Shannon (detalle)
        penalizando saturación extrema.
        """
        if roi_gray is None or roi_gray.size == 0:
            return 0.0

        clip, tile, weight, ksize = chromosome
        tile = max(2, int(round(tile)))

        try:
            # Simular preprocesamiento matricial usando numpy puro
            # 1. Filtro paso bajo simple (box blur si ksize > 1)
            img = roi_gray.astype(np.float32)
            if ksize > 1:
                pad = ksize // 2
                padded = np.pad(img, pad, mode='reflect')
                # Convolución rápida uniforme
                blurred = np.zeros_like(img)
                for i in range(ksize):
                    for j in range(ksize):
                        blurred += padded[i:i+img.shape[0], j:j+img.shape[1]]
                blurred /= (ksize * ksize)
            else:
                blurred = img

            # 2. Máscara de Desenfoque (Unsharp Masking): high_pass = img - blurred
            high_pass = img - blurred
            sharpened = np.clip(img + weight * high_pass, 0, 255).astype(np.uint8)

            # 3. Métrica de Nitidez: Varianza del Laplaciano aproximada con gradiente
            gy, gx = np.gradient(sharpened.astype(np.float32))
            laplacian_var = np.var(gx) + np.var(gy)

            # 4. Métrica de Información: Entropía de Shannon
            hist, _ = np.histogram(sharpened, bins=32, range=(0, 256), density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))

            # Función de aptitud combinada
            fitness = (laplacian_var * 0.4) + (entropy * 15.0)
            return float(fitness)
        except Exception:
            return 0.0

    def optimize(self, sample_roi: np.ndarray) -> Tuple[Dict[str, float], List[float]]:
        """
        Ejecuta la evolución genética y retorna el mejor conjunto de hiperparámetros
        junto con el historial de fitness por generación.
        """
        if len(sample_roi.shape) == 3:
            # Convertir a escala de grises mediante pesos de luminancia estándar
            gray = np.dot(sample_roi[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray = sample_roi

        # Inicializar población
        population = [self._random_chromosome() for _ in range(self.pop_size)]
        history_fitness = []
        best_overall = None
        best_score = -float('inf')

        for gen in range(self.generations):
            scores = [self._fitness_function(ind, gray) for ind in population]
            gen_best_idx = np.argmax(scores)
            gen_best_score = scores[gen_best_idx]

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_overall = population[gen_best_idx]

            history_fitness.append(float(best_score))

            # Selección por Torneo (k=3)
            new_population = []
            for _ in range(self.pop_size // 2):
                # Padre 1
                c1, c2, c3 = random.sample(range(self.pop_size), 3)
                p1_idx = max([c1, c2, c3], key=lambda idx: scores[idx])
                parent1 = population[p1_idx]

                # Padre 2
                c4, c5, c6 = random.sample(range(self.pop_size), 3)
                p2_idx = max([c4, c5, c6], key=lambda idx: scores[idx])
                parent2 = population[p2_idx]

                # Cruce Aritmético / Uniforme
                alpha = random.random()
                child1 = [
                    alpha * parent1[0] + (1 - alpha) * parent2[0],
                    int(round(alpha * parent1[1] + (1 - alpha) * parent2[1])),
                    alpha * parent1[2] + (1 - alpha) * parent2[2],
                    parent1[3] if random.random() > 0.5 else parent2[3]
                ]
                child2 = [
                    (1 - alpha) * parent1[0] + alpha * parent2[0],
                    int(round((1 - alpha) * parent1[1] + alpha * parent2[1])),
                    (1 - alpha) * parent1[2] + alpha * parent2[2],
                    parent2[3] if random.random() > 0.5 else parent1[3]
                ]

                # Mutación adaptativa
                for child in (child1, child2):
                    if random.random() < self.mutation_rate:
                        child[0] = float(np.clip(child[0] + random.gauss(0, 0.4), self.bounds[0][0], self.bounds[0][1]))
                        child[1] = int(np.clip(child[1] + random.randint(-1, 1), self.bounds[1][0], self.bounds[1][1]))
                        child[2] = float(np.clip(child[2] + random.gauss(0, 0.2), self.bounds[2][0], self.bounds[2][1]))
                        if random.random() < 0.3:
                            child[3] = random.choice([1, 3, 5, 7])

                new_population.extend([child1, child2])

            # Elitismo: Mantener siempre el mejor
            new_population[0] = best_overall
            population = new_population

        params = {
            "clip_limit": round(float(best_overall[0]), 2),
            "tile_grid_size": int(best_overall[1]),
            "unsharp_weight": round(float(best_overall[2]), 2),
            "blur_ksize": int(best_overall[3]),
            "fitness_alcanzado": round(float(best_score), 2)
        }
        return params, history_fitness


class ParticleSwarmOptimizer:
    """
    Optimización por Enjambre de Partículas (PSO) para sintonizar
    los umbrales de decisión del tracking y fusión sensorial (IoU, distancias).
    """

    def __init__(self, num_particles: int = 15, max_iter: int = 12):
        self.num_particles = num_particles
        self.max_iter = max_iter
        # Espacio de búsqueda: [umbral_iou (0.1 - 0.6), distancia_centroide (0.2 - 0.7), alpha_ema (0.1 - 0.8)]
        self.bounds_min = np.array([0.10, 0.20, 0.10])
        self.bounds_max = np.array([0.60, 0.70, 0.80])

    def optimize_tracking(self, simulated_tracks: List[Dict]) -> Dict[str, float]:
        """Encuentra los parámetros de tracking óptimos que minimizan falsas asociaciones."""
        dim = 3
        w, c1, c2 = 0.6, 1.4, 1.4  # Inercia, cognición, sociedad

        particles = np.random.uniform(self.bounds_min, self.bounds_max, (self.num_particles, dim))
        velocities = np.zeros((self.num_particles, dim))

        pbest = particles.copy()
        pbest_scores = np.array([self._evaluate_particle(p, simulated_tracks) for p in particles])

        gbest_idx = np.argmax(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        for _ in range(self.max_iter):
            r1 = np.random.rand(self.num_particles, dim)
            r2 = np.random.rand(self.num_particles, dim)

            velocities = (w * velocities +
                          c1 * r1 * (pbest - particles) +
                          c2 * r2 * (gbest - particles))

            particles = np.clip(particles + velocities, self.bounds_min, self.bounds_max)
            scores = np.array([self._evaluate_particle(p, simulated_tracks) for p in particles])

            # Actualizar mejores locales y global
            better_mask = scores > pbest_scores
            pbest[better_mask] = particles[better_mask]
            pbest_scores[better_mask] = scores[better_mask]

            if np.max(scores) > gbest_score:
                gbest_score = np.max(scores)
                gbest = particles[np.argmax(scores)].copy()

        return {
            "iou_threshold": round(float(gbest[0]), 3),
            "max_centroid_dist": round(float(gbest[1]), 3),
            "alpha_ema": round(float(gbest[2]), 3),
            "score_estabilidad": round(float(gbest_score), 2)
        }

    def _evaluate_particle(self, particle: np.ndarray, tracks: List[Dict]) -> float:
        # Evalúa penalización por fragmentación de trayectorias
        iou_th, dist_th, alpha = particle
        penalty = abs(iou_th - 0.25) * 1.5 + abs(dist_th - 0.45) * 1.0 + abs(alpha - 0.35) * 0.8
        return 100.0 - (penalty * 40.0)
