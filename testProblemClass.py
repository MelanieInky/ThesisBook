import numpy as np
import matplotlib.pyplot as plt

class testProblem:
## Define it as 
    def __init__(self,b,n) -> None:
        self.n = n
        self.b = b
        self.deltaX = 1 / (n+1)
        self.M = self.buildM()
        self.e = self.buildE()


    def buildM(self):
        """
        we go from u0 to u(n+1).
        """
        deltaX = self.deltaX
        n = self.n
        b = self.b
        A = 1/deltaX *(np.eye(n) -1 * np.eye(n,k = -1))
        B = 1/(deltaX ** 2) * b* (-2*np.eye(n) + np.eye(n, k = -1) + np.eye(n,k=1))
        #We scale it back
        gamma = 1/deltaX + 2 /(deltaX ** 2) * b
        return (A-B)/gamma
    
    def buildE(self):
        deltaX = self.deltaX
        b = self.b
        gamma = 1/deltaX + 2/(deltaX ** 2) * b
        return np.ones(self.n) / gamma
    
    def f(self,y):
        return self.e - self.M@y
    
    def oneStepSmoother(self,y,t,deltaT,alpha):
        """
        Perform one pseudo time step deltaT of the solver for the diff eq
        y' = e - My = f(y). .
        """
        k1 = self.f(y)
        k2 = self.f(y + alpha*deltaT*k1)
        yNext = y + deltaT*k2
        return yNext
    
    def findOptimalParameters(self):
        #This is where the reinforcement learning algorithm 
        #take place in
        return 0 , 0
    

    def mainSolver(self,n_iter = 10):
        """ Main solver for the problem, calculate the approximated solution
        after n_iter pseudo time steps. """
        resNormList = np.zeros(n_iter+1)
        t = 0
        #Initial guess y = e
        y = np.ones(e)
        resNormList[0] = np.linalg.norm(self.M@y-self.e)
        ##Finding the optimal params
        alpha, deltaT = self.findOptimalParameters()
        ##Will need to be removed, just for debugging
        alpha = 0.5
        deltaT = 0.00006
        #For now, we use our best guess
        for i in range(n_iter):
            y = self.oneStepSmoother(y,t,deltaT,alpha)
            t += deltaT
            resNorm = np.linalg.norm(self.M@y - self.e)
            resNormList[i+1] = resNorm
        return y , resNormList

    def mainSolver2(self,alpha, deltaT, n_iter = 10):
        """ Like the main solver, except we give 
        the parameters explicitely """
        resNormList = np.zeros(n_iter+1)
        t = 0
        #Initial guess y = e
        y = np.copy(self.e)

        resNormList[0] = np.linalg.norm(self.M@y-self.e,2)
        #For now, we use our best guess
        for i in range(n_iter):
            y = self.oneStepSmoother(y,t,deltaT,alpha)
            t += deltaT
            resNorm = np.linalg.norm(self.M@y - self.e,2)
            resNormList[i+1] = resNorm
        return y , resNormList
        