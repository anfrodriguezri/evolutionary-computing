import math
import numpy as np

from abc import ABCMeta, abstractmethod

class Function:
    __metaclass__ = ABCMeta

    @abstractmethod
    def eval(self, X): pass
    
    def __init__(self, dim):
        self.dim = dim
        self.domain = {'lower': -math.inf, 'upper': math.inf}
    
    def feasible(self, X):
        for i in range(self.dim):
            if X[i] > self.domain['upper'] or X[i] < self.domain['lower']:
                return False
        return True

class DiffFunction:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def gradient(self, X): pass

class Diff2Function:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def hessian(self, X): pass
    
    def hessian_det(self, X):
        return np.linalg.det(self.hessian(X))

class QuadraticFunction(Function, DiffFunction, Diff2Function):
    dim = 1
    
    def eval(self, X):
        e = 0
        
        for i in range(self.dim):
            e += (X[i]**2)
        
        return e
        
    def gradient(self, X):
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            grad[i] = 2 * X[i]
        
        return grad

    def hessian(self, X):
        hess = np.zeros([self.dim, self.dim])
        
        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    hess[i][j] = 2
                                                                                  
        return hess

class RastriginFunction(Function, DiffFunction, Diff2Function):
    dim = 1
    domain = {'lower': -5.12, 'upper': 5.12}

    def eval(self, X):
        e = 10 * self.dim
        
        for i in range(self.dim):
            e += (X[i]**2) - 10 * math.cos(2 * PI * X[i])
        
        return e
        
    def gradient(self, X):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            grad[i] = 2 * ( X[i] + 10 * PI * math.sin(2*PI*X[i]) )
        
        return grad

    def hessian(self, X):
        hess = np.zeros([self.dim, self.dim])
        
        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    hess[i][j] = 2 * ( 1 + 20 * (PI**2) * math.cos(2*PI*X[i]) )
                                                                                  
        return hess

class GriewankFunction(Function, DiffFunction, Diff2Function):
    dim = 1
    domain = {'lower': -600, 'upper': 600}

    def eval(self, X):
        suma = 0
        prod = 1
        
        for i in range(self.dim):
            suma += (X[i]**2) / 4000
            prod *= math.cos(X[i]/math.sqrt(i+1))
            
        return suma - prod + 1
    
    def gradient(self, X):
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            prod = 1
            for k in range(self.dim):
                if k != i:
                    prod *= math.cos(X[i] / math.sqrt(i+1))
                    
            grad[i] = (X[i] / 2000) + math.sin(X[i]/math.sqrt(i+1)) * (1/math.sqrt(i+1)) * prod
        
        return grad

    def hessian(self, X):
        hess = np.zeros([self.dim, self.dim])
        
        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    prod = 1
                    
                    for k in range(self.dim):
                        if k != i:
                            prod *= math.cos(X[k]/math.sqrt(k+1))
                    
                    hess[i][j] = (1/2000) + math.cos(X[i]) * (1/(i+1)) * prod
                else:
                    prod = 1
                    
                    for k in range(self.dim):
                        if k != i and k != j:
                            prod *= math.cos(X[k]/math.sqrt(k+1))
                            
                    hess[i][j] = math.sin(X[i]/math.sqrt(i+1)) * (1/math.sqrt(i+1)) * prod
                    hess[i][j] *= -1 * math.sin(X[j]/math.sqrt(j+1)) * (1/math.sqrt(j+1))
                                                                                  
        return hess

class RosenbrockFunction(Function, DiffFunction, Diff2Function):
    dim = 1
    domain = {'lower': -2048, 'upper': 2048}

    def eval(self, X):
        suma = 0

        for i in range(self.dim):
            suma += (100 * (X[i+1] - x[i]**2)**2) + (1 - X[i])**2

        return suma

    def gradient(self, X):
        grad = np.zeros(self.dim)
        
        for i in range(self.dim):
            if( i != self.dim - 1 ):
                grad[i] += 200 * (X[i+1] - (X[i]**2)) * (-2 * X[i])

            if( i != 0 ):
                grad[i] += 200 * (X[i] - (X[i-1]**2))

            grad[i] += 2 * (X[i] - 1)
        
        return grad

    def hessian(self, X):
        hess = np.zeros([self.dim, self.dim])
        
        for i in range(self.dim):
            for j in range(self.dim):

                if (i < j - 1) or (i > j + 1):
                    continue

                if (i == j - 1):
                    if i != 0:
                        hess[i][j] += -400 * X[j-1]
                elif i == j:
                    if (i != self.dim - 1):
                        hess[i][j] += -400 * ( x[j+1] - (3*X[j]**2) )

                    if i != 0:
                        hess[i][j] += 200

                    hess[i][j] += 2
                
                elif (i == j + 1):
                    if (i != self.dim - 1):
                        hess[i][j] += -400 * X[j]

                                                                                  
        return hess

class SchwefelFunction(Function):
    dim = 1
    lower = 500
    def eval(self, X):
        suma = 0
        
        for i in range(self.dim):
            suma += X[i] * math.sin(math.sqrt(math.abs(X[i])))
        
        return 418.9829 * self.dim - suma
    
    def feasible(self, X):
        for i in range(self.dim):
            if X[i] > 500 or X[i] < -500:
                return False
        return True


