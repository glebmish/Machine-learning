from LinearRegression.methods.base import Base

import numpy as np


class Genetic(Base):

    def __init__(self, population_size=100,
                 nsteps=300, e=0.0000000001,
                 weight_low=-500, weight_high=500, random_seed=1,
                 mutation_rate=0.015, tournament_size=5):
        super().__init__()
        self.population_size = population_size
        self.nsteps = nsteps
        self.e = e
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.random_seed = random_seed
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def fit(self, X, y):
        m = X.shape[0]
        # add 1s column to also find w0 (absolute term of regression)
        X = np.hstack((np.ones(m).reshape(m, 1), X))

        self.X = X
        self.y = y

        np.random.seed(self.random_seed)

        population = self.init_population()
        generations = 0
        fittest = self.fittest(population)

        while generations < self.nsteps or fittest[1] < self.e:
            if generations % 100 == 0:
                print(generations)
            generations += 1
            population = self.evolve(population)
            fittest = self.fittest(population)

        self.W = fittest[0]
        return self.W

    def init_population(self):
        return np.random.randint(low=self.weight_low, high=self.weight_high, size=(self.population_size, 3))

    def fittest(self, population):
        fittest = population[0], self.mean_deviation(self.y, np.dot(self.X, population[0]))

        for each in population[1:]:
            error1 = self.mean_deviation(self.y, np.dot(self.X, each))
            if error1 < fittest[1]:
                fittest = each, error1

        return fittest

    def evolve(self, population):
        new_population = []

        for _ in range(len(population)):
            ind1 = self.tournament_selection(population)[0]
            ind2 = self.tournament_selection(population)[0]
            new_ind = self.crossover(ind1, ind2)
            new_population.append(new_ind)

        for i in range(len(new_population)):
            new_population[i] = self.mutate(new_population[i])

        return np.array(new_population)

    def tournament_selection(self, population):
        tournamenters = population[np.random.choice(population.shape[0], self.tournament_size), :]
        return self.fittest(tournamenters)

    def crossover(self, ind1, ind2):
        genes = np.random.randint(0, 2, size=3)
        new_ind = []

        new_ind.append(ind1[0] if genes[0] == 0 else ind2[0])
        new_ind.append(ind1[1] if genes[1] == 0 else ind2[1])
        new_ind.append(ind1[2] if genes[2] == 0 else ind2[2])

        return np.array(new_ind)

    def mutate(self, ind):
        to_mutate = np.random.uniform(0, 1)
        if to_mutate < self.mutation_rate:
            mutated_gene = np.random.randint(0, 3, size=1)[0]
            mutated_bit = np.random.randint(0, ind.itemsize, 1)
            ind[mutated_gene] = np.bitwise_xor(ind[mutated_gene], np.left_shift(1, mutated_bit))

        return ind