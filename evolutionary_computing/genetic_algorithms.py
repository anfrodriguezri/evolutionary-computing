class GeneticAlgorithm():
    def __init__(self, problem, selection, crossover, mutation):
        self.problem = problem
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def initialize_population(self):
        pass

    def run(self):
        # get best solution
        current_population = self.problem.initialize_population()
        # fitness fi / avg f
        generation = 1
        while not self.problem.termination_reached(generation):
            self.problem.evaluate(current_population)

            # select a mating pool
            intermediate_population = self.selection.operate(
                current_population)

            new_population = self.problem.empty_population()

            while len(new_population) != len(current_population):
                parents = self.problem.pick_parents(intermediate_population)
                children = self.crossover.operate(parents)
                children = self.mutation.operate(children)

                new_population.add(children)

            current_population = new_population
            generation += 1
