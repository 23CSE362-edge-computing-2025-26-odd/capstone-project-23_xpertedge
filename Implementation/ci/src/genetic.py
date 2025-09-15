import random
import copy
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Hyperparameters:
    width_mult: float
    learning_rate: float
    batch_size: int
    dropout_rate: float
    weight_decay: float
    momentum: float
    conv_channels: List[int]

    def __post_init__(self):
        self.width_mult = max(0.25, min(2.0, self.width_mult))
        self.learning_rate = max(0.0001, min(0.1, self.learning_rate))
        self.batch_size = max(32, min(256, int(self.batch_size)))
        self.dropout_rate = max(0.0, min(0.5, self.dropout_rate))
        self.weight_decay = max(1e-6, min(1e-2, self.weight_decay))
        self.momentum = max(0.1, min(0.99, self.momentum))

class GeneticAlgorithm:
    def __init__(self, population_size=8, mutation_rate=0.3, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0

    def create_individual(self):
        return Hyperparameters(
            width_mult=random.uniform(0.5, 1.5),
            learning_rate=random.uniform(0.005, 0.05),
            batch_size=random.choice([64, 96, 128, 160, 192]),
            dropout_rate=random.uniform(0.1, 0.4),
            weight_decay=random.uniform(1e-5, 5e-3),
            momentum=random.uniform(0.8, 0.95),
            conv_channels=[16, 32, 64, 128]
        )
    
    def create_population(self):
        population = []
        baseline = Hyperparameters(
            width_mult=1.0, learning_rate=0.01, batch_size=128,
            dropout_rate=0.2, weight_decay=2e-4, momentum=0.9,
            conv_channels=[16, 32, 64, 128]
        )
        population.append(baseline)
        for _ in range(self.population_size - 1):
            population.append(self.create_individual())
        return population
    
    def crossover(self, parent1, parent2):
        child1_dict = {}
        child2_dict = {}
        for key in parent1.__dict__.keys():
            if random.random() < 0.5:
                child1_dict[key] = getattr(parent1, key)
                child2_dict[key] = getattr(parent2, key)
            else:
                child1_dict[key] = getattr(parent2, key)
                child2_dict[key] = getattr(parent1, key)
        child1 = Hyperparameters(**child1_dict)
        child2 = Hyperparameters(**child2_dict)
        return child1, child2
    
    def mutate(self, individual):
        mutated = copy.deepcopy(individual)
        if random.random() < self.mutation_rate:
            mutated.width_mult += random.gauss(0, 0.2)
        if random.random() < self.mutation_rate:
            mutated.learning_rate *= random.uniform(0.5, 2.0)
        if random.random() < self.mutation_rate:
            mutated.batch_size = random.choice([64, 96, 128, 160, 192])
        if random.random() < self.mutation_rate:
            mutated.dropout_rate += random.gauss(0, 0.1)
        if random.random() < self.mutation_rate:
            mutated.weight_decay *= random.uniform(0.1, 10.0)
        if random.random() < self.mutation_rate:
            mutated.momentum += random.gauss(0, 0.05)
        mutated.__post_init__()
        return mutated
    
    def select_parents(self, population, fitness_scores):
        selected = []
        tournament_size = 3
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(copy.deepcopy(population[winner_idx]))
        return selected
