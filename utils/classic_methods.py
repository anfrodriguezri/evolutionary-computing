import numpy as np


def newton_raphson(f, X, max_iter):
    fitness = np.zeros(max_iter)

    for it in range(max_iter):
        grad = f.gradient(X)
        hess_det = f.hessian_det(X)

        Y = np.copy(X)

        for i in range(f.dim):
            Y[i] += (-grad[i]) / hess_det

        X = Y

        fitness[it] = f.eval(X)

    return fitness


def gradient_descent(f, X, max_iter):
    fitness = np.zeros(max_iter)
    gamma = .3

    for it in range(max_iter):
        grad = f.gradient(X)

        Y = np.copy(X)

        for i in range(f.dim):
            Y[i] += (-grad[i]) * gamma

        X = Y

        fitness[it] = f.eval(X)

    return fitness


def gradient_descent_with_momentum(f, X, max_iter):
    fitness = np.zeros(max_iter)
    alpha = .1
    miu = .8

    previous_delta = np.zeros(f.dim)
    delta = np.copy(previous_delta)

    for it in range(max_iter):
        grad = f.gradient(X)

        Y = np.copy(X)

        for i in range(f.dim):
            delta[i] = miu * previous_delta[i] - grad[i] * alpha
            Y[i] += delta[i]

            previous_delta = delta

        X = Y

        fitness[it] = f.eval(X)

    return fitness


def hill_climbing(f, X, max_iter, rand_f):
    fitness = np.zeros(max_iter)
    sigma = 5

    for it in range(max_iter):
        Y = np.copy(X)

        for i in range(f.dim):
            Y[i] += rand_f() * sigma

        if f.eval(Y) < f.eval(X):
            X = Y

        fitness[it] = f.eval(X)

    return fitness


def simulated_annealing(f, X, max_iter, rand_f):
    fitness = np.zeros(max_iter)
    sigma = 1.5
    t_max = 50
    t = 1

    for it in range(max_iter):
        if t == t_max:
            t = 1

        T = t / t_max

        Y = np.copy(X)

        for i in range(f.dim):
            Y[i] += rand_f() * sigma

        delta_D = f.eval(Y) - f.eval(X)

        r = min(1, np.exp(-delta_D * T))

        if delta_D < 0 or np.random.randn() < r:
            X = Y

        t += 1

        if abs(0 - f.eval(X)) < 1e-3:
            return it

    return max_iter
