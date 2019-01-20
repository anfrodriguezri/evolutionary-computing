class Individual():
    def __init__(self):
        pass


class Population():
    def __init__(self, N=0, individuals=[], individual_factory=Individual):
        self.individuals = individuals
        length = len(individuals)

        if length != N:
            for i in range(N):
                self.individuals.append(individual_factory())

        self.N = len(individuals)

    def __getitem__(self, key):
        return self.individuals[key]

    def __setitem__(self, key, value):
        if type(value) is not Individual:
            raise TypeError('Only Individual class can be added to Population')

        self.individuals[key] = value

    def __len__(self):
        return self.N

    def initialize_random(self, N):
        pass


class Problem():
    def __init__(self, N, crossover_prob=.8, mutation_prob=.08):
        self.N = N
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def initialize_population(self):
        pass

    def fitness_function(self, individual):
        pass

    def evaluate(self, population):
        pass

    def empty_population(self):
        pass

    def pick_parents(self, population):
        pass

    def termination_reached(self, generation):
        pass
