import sys
import operator
import copy

import numpy as np
from abc import ABCMeta, abstractmethod

from evolutionary_computing import utilities

class Individual:
	def __init__(self, value, idx = None):
		self.value = value
		self.idx = idx
		self.fitness = None
		self.bin_repr = None

	def __repr__(self):
		return str(self.value)

	def __str__(self):
		return 'Value: {}\nBin Repr: {}\n'.format(str(self.value), self.bin_repr)


class RealValuedIndividual(Individual):
	def __init__(self, function, idx = None, dtype=np.uint32):
		if not np.issubdtype(dtype, np.unsignedinteger):
			raise Exception('dtype has to be subtype of np.unsignedinteger')
		
		self.idx = idx
		self.fitness = None;

		self.function = function
		self.dtype = dtype
		self.num_bits = self.dtype().itemsize*8

		self.lower_bound = function.domain['lower']
		self.upper_bound = function.domain['upper']

		max_value = np.iinfo(self.dtype).max / 4
		X_encoded = np.random.randint(0, max_value, self.function.dim, dtype=self.dtype)

		self.bin_repr = ''
		for x in X_encoded:
			self.bin_repr += bin(x)[2:].zfill(self.num_bits)

		self.value = []

		self.decode_value()

	def __repr__(self):
		return '\n{} {}'.format(str(self.idx) or '', self.fitness)

	def bin_to_real(self, binary):
		x_min = self.lower_bound
		x_max = self.upper_bound

		binary = binary[::-1]
		suma = sum([int(b)*(2**i) for i, b in enumerate(binary)])

		return x_min + (x_max - x_min)/(2**self.num_bits-1) * suma

	def decode_value(self):
		self.value = [self.bin_to_real(split_bin_repr) for split_bin_repr in self.get_splitted_bin_repr()]

	def get_splitted_bin_repr(self):
		bin_repr = self.bin_repr
		num_bits = self.num_bits

		return [bin_repr[i:i+num_bits] for i in range(0, len(bin_repr), num_bits)]

class Population:
	def __init__(self, n, ind_class, function):
		self.individuals = []
		self.fitness = 0
		self.avg_fitness = 0
		self.function = function

		thismodule = sys.modules[__name__]

		for i in range(n):
			self.individuals.append( getattr(thismodule, ind_class)(function, i) )

	def __getitem__(self, i):
		return self.individuals[i]

	def __len__(self):
		return len(self.individuals)

	def __str__(self):
		return str(self.individuals)

	def append(self, individual):
		self.individuals.append(individual)

	def get_best(self, fitness_func, inequality_op = operator.lt):
		self.best = self.individuals[0]
		self.fitness = 0
		
		for individual in self.individuals:
			if individual.fitness is None:
				individual.fitness = fitness_func.eval(individual.value)

			self.fitness += individual.fitness

			if inequality_op(individual.fitness, self.best.fitness):
				self.best = individual

		self.avg_fitness = self.fitness / len(self)

		return self.best

class GeneticAlgorithm:
	def __init__(self, fitness_func, ind_class, selection_meth, population_size = 100,
					 max_iter = 100, inequality_op = operator.lt, epsilon = 1e-6):
		self.fitness_func = fitness_func
		self.ind_class = ind_class
		self.selection_meth = selection_meth
		self.population_size = population_size
		self.max_iter = max_iter
		self.inequality_op = inequality_op
		self.epsilon = epsilon

	def generateOffspring(self, parents):
		num_children = len(parents)
		chromo_size = parents[0].num_bits * parents[0].function.dim

		crossover_points = [0]
		cross_point = 0

		for i in range(num_children-1):
			cross_point = np.random.randint(cross_point, chromo_size)
			crossover_points.append( cross_point )
		
		crossover_points.append(chromo_size)

		children = [copy.copy(parent) for parent in parents]
		new_chromosomes = ['' for i in range(num_children)]

		for i in range(num_children):
			for j in range(num_children):
				new_chromosomes[i] += children[(i+j) % num_children].bin_repr[crossover_points[j]:crossover_points[j+1]]

		for i in range(num_children):
			children[i].bin_repr = new_chromosomes[i]

		return children


	def run(self):
		population = Population(self.population_size, self.ind_class, self.fitness_func)
		
		for t in range(self.max_iter):
			best = population.get_best(self.fitness_func, self.inequality_op)
			print( population )

			if best.fitness - self.fitness_func.optimal < self.epsilon:
				return t, best


			parents = self.selection_meth.select(2, population)
			print( parents )
			children = self.generateOffspring(parents)


