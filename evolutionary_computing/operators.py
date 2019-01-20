import random


class Selection():
    def operate(self, population):
        pass


class RandomSelection(Selection):
    def operate(self, population):
        return random.choice(population)


class Crossover():
    def operate(self, parents):
        pass


class Mutation():
    def operate(self, individual):
        pass
