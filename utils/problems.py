from evolutionary_computing.encoding import *
from bitarray import bitarray


class OneMaxIndividual(Individual):
    def __init__(self, length=0, bitarray=None):
        if not bitarray:
            self.bitarray = np.random.randint(2, size=length).tolist()
            self.bitarray = bitarray(self.bitarray)
        else:
            self.bitarray = bitarray

        self.length = bitarray.length()


class OneMaxPopulation(Population):
    def __init__(self, N=0, individuals=[], individual_factory=OneMaxIndividual):
        super(OneMaxPopulation, self).__init__()


class OneMaxProblem(Problem):
    def __init__(self):
        self.one_max = tf.OneMaxFunction()

    def initialize_population(self):
        return

    def fitness_function(self, individual):
        return one_max.eval(individual)
