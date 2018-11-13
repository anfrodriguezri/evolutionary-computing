import numpy as np

class RoulleteSelection:
	def get_proportional_prob(ind_fitness, population, optimal_sence='min'):
		if optimal_sence == 'max':
			return ind_fitness / population.fitness
		elif optimal_sence == 'min':
			return (1 - (ind_fitness / population.fitness)) / (len(population) - 1)


	def select(n, population):
		selection_probabilities = []

		population_size = len(population)

		for i in range(population_size):
			proportional_prob = RoulleteSelection.get_proportional_prob(population[i].fitness, population)
			selection_probabilities.append(proportional_prob)

		mating_pool = []
		randoms = np.random.random(population_size)


		for k in range(population_size):
			accum_probability = 0

			for i, individual in enumerate(population):
				accum_probability += selection_probabilities[i]

				if randoms[k] < accum_probability:
					mating_pool.append(individual)
					break

		return np.random.choice(mating_pool, n)
