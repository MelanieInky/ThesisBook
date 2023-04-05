import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os


class testProblem:
## Define it as 
    def __init__(self,b,n) -> None:
        self.n = n
        self.b = b
        self.deltaX = 1 / (n+1)
        self.M = self.buildM()
        self.e = self.buildE()
        self.pol = policy()
    
    
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
    
    def update(self, b,n):
        self.b = b
        self.n = n
        self.M = self.buildM()
        self.e = self.buildE()
        self.deltaX = 1 / (n+1)

    def mainSolver(self,n_iter = 10):
        """ Main solver for the problem, calculate the approximated solution
        after n_iter pseudo time steps. """
        t = 0
        #Initial guess y = e
        y = np.copy(self.e)
        resNorm = np.linalg.norm(self.M@y-self.e)
        lastResNorm = resNorm
        ##Finding the parameters according to current policy
        alpha, deltaT = self.pol(self.b, self.n)
        for i in range(n_iter):
            y = self.oneStepSmoother(y,t,deltaT,alpha)
            t += deltaT
            newResNorm = np.linalg.norm(self.M@y - self.e)
            #Check for overflow!
            if np.isinf(newResNorm) or np.isnan(newResNorm):
                print("Overflow while calculating residual!")
                return y , resNorm, lastResNorm
            
            lastResNorm = resNorm
            resNorm = newResNorm
        return y , resNorm, lastResNorm
    
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

    def getActionAndReward(self,n_iter = 10):
        """Take an action, given the current problem parameters and policy,
        then compute the reward. Return the action taken, and the associated reward"""
        t = 0
        #Initial guess y = e
        y = np.copy(self.e)
        resNorm = np.linalg.norm(self.M@y-self.e)
        lastResNorm = resNorm
        ##Finding the parameters according to current policy
        alpha, deltaT = self.pol(self.b, self.n)
        for i in range(n_iter):
            y = self.oneStepSmoother(y,t,deltaT,alpha)
            t += deltaT
            newResNorm = np.linalg.norm(self.M@y - self.e)
            #Check for overflow!
            if np.isinf(newResNorm) or np.isnan(newResNorm) or newResNorm>100000:
                #print("Overflow while calculating residual!")
                return alpha, deltaT , -3
            lastResNorm = resNorm
            resNorm = newResNorm
        ratio =  resNorm / lastResNorm
        if(ratio<=1):
            reward = 1000*(1-ratio)
        else:
            reward = 1 - ratio
        return alpha, deltaT, reward
    


    def generateEpisode(self, length = 5):
        """Given the currert policy, generate an episode of fixed length"""
        bHistory = np.zeros(length)
        nHistory = np.zeros(length)
        alphaHistory = np.zeros(length)
        deltaTHistory = np.zeros(length)
        rewardHistory = np.zeros(length)
        for i in range(length):
            bHistory[i] = self.b
            nHistory[i] = self.n
            #Compute action and reward
            alpha, deltaT , reward = self.getActionAndReward()
            alphaHistory[i] = alpha
            deltaTHistory[i] = deltaT
            rewardHistory[i] = reward
            #gradient = self.pol.logLikGradient(self.b,self.n,alpha,deltaT)
            #Then jump to a new state, at random
            newb = np.random.uniform(0,1)
            newn = np.random.randint(5,200)
            self.update(newb,newn)

        myEpisode = episode(bHistory,nHistory,alphaHistory, deltaTHistory,rewardHistory)
        return myEpisode

    def learnOneEpisode(self,epLength = 100):
        gamma = 0.1
        myEpisode = self.generateEpisode(epLength)
        thetaArray = np.zeros((epLength,6))
        gradThetaArray = np.zeros((epLength,6))
        for t in range(epLength):
            #Calculate the sum of rewards
            Gt = 0
            for i in range(t,epLength):
                #Return on the trajectory
                Gt += myEpisode.rewardHist[i] * (gamma**(i-t))

            b = myEpisode.bHist[t]
            n = myEpisode.nHist[t]
            alpha = myEpisode.alphaHist[t]
            deltaT = myEpisode.deltaTHist[t]
            #Calculate the log likelihood gradient, for the current policy
            logLikGrad = self.pol.logLikGradient(b,n,alpha,deltaT)
            grad = logLikGrad*Gt
            thetaArray[t,:] = self.pol.theta
            gradThetaArray[t,:] = logLikGrad
            #Then applies the gradient ascent directly on the sample
            self.pol.theta = self.pol.theta + 0.00000008*grad
        return myEpisode , thetaArray , gradThetaArray
        
    def learn(self,length = 100, log = True,offset = 0, fileName = 'Log/log.csv'):
        #The offset is only for logging purposes.
        #So we can write the correct Episode number 
        #When making multiple files
        epLength = 100
        if(log):
            f = open(fileName , 'a')
            firstLine = 'ep_number , ep_state_nb , b , n , alpha , deltaT , reward , theta_0, theta_1 , theta_2 , theta_3 , theta_4 , theta_5 , grad_theta_0 , grad_theta_1 , grad_theta_2 , grad_theta_3 , grad_theta_4 , grad_theta_5 \n' 
            f.write(firstLine)
            print('Logging activated')
        #Loop for a certain number of episodes.
        meanRewardList = np.zeros(length)
        for i in range(length):
            ep , theta , grad = self.learnOneEpisode(epLength)
            meanRewardList[i] = np.mean(ep.rewardHist)
            ####Logging
            if(log):
                for k in range(epLength):
                    thetaStr = np.array2string(theta[k,:],separator=',').lstrip('[').rstrip(']').replace('\n', '') + ','
                    gradStr = np.array2string(grad[k,:], separator=',').lstrip('[').rstrip(']').replace('\n', '')
                    epStr = ep.logStr(k) + ','
                    epNumber = str(i + offset) + ','
                    totalString = epNumber + epStr + thetaStr + gradStr + '\n'
                    f.write(totalString)
        if(log):
            f.close()
        print('Mean reward for the batch' , np.mean(meanRewardList))
        return 0
        

            
        



class episode:
    def __init__(self,b,n,alpha,deltaT,reward):
        self.bHist = b
        self.nHist = n
        self.alphaHist = alpha
        self.deltaTHist = deltaT
        self.rewardHist = reward
    
    def length(self):
        return len(self.bHist)
    
    def logStr(self,k):
        #Log the state-action-reward-etc at time k.
        logStr = "%s , %s , %s , %s , %s , %s" % (k, self.bHist[k], self.nHist[k], self.alphaHist[k] , self.deltaTHist[k], self.rewardHist[k])
        return logStr



class policy:
    """ Policy class. Will be used to compute all sorts of things"""
    def __init__(self) -> None:
        """ Policy, first with random values"""
        self.theta = np.random.rand(6)/200
        self.stdDev = 0.1 #Will be learned later


    def set(self,theta):
        self.theta = theta
    

    def __call__(self, b,n):
        """ Evaluate the policy and return the alpha, deltaT parameters"""
        theta = self.theta
        alphaMean = theta[0]*b + theta[1]*n + theta[4]
        deltaTMean = theta[2]*b + theta[3]*n + theta[5]
        alpha = np.random.normal(alphaMean,self.stdDev)
        deltaT = np.random.normal(deltaTMean,self.stdDev)
        return alpha, deltaT
    
    def __str__(self) -> str:
        """Overload for the print"""
        D = np.reshape(self.theta[:4],(2,2))
        part1 = "Policy function, of the form [alpha , deltaT] = D*[b,n]' + bias, with D = \n"
        part2 =  D.__str__()
        part3 = "\nand b = " + self.theta[4:].__str__()
        part4 = "\nwith common std dev of " + str(self.stdDev) + "."
        return part1 + part2 + part3 + part4
    
    def logLikGradient(self,b,n,alpha,deltaT):
        """ Get the gradient of log likelihood, for each parameters"""
        theta = self.theta
        xi1 = (alpha - theta[0] * b - theta[1] * n - theta[4])
        xi2 = (deltaT - theta[2] * b - theta[3] * n - theta[5])
        gradTheta = np.zeros(6)
        gradTheta[0] = b*xi1
        gradTheta[1] = n*xi1
        gradTheta[4] = xi1
        gradTheta[2] = b*xi2
        gradTheta[3] = n*xi2
        gradTheta[5] = xi2
        self.gradient = gradTheta
        return gradTheta




problem = testProblem(0.05,100)



#problem.pol.set(theta= np.array([0,0,0,0,0.5,2]))
problem.pol.set(theta = np.array([-0.00959971, -0.00817385, -0.0023457 , -0.01549442,  0.47988811,1.99517977]))
date= datetime.now().strftime('%Hh%Mm%Ss')
os.mkdir('Log/' + date)

learningBatchSize = 1000
for i in range(100):
    fileName = 'Log/' + date + '/%s.csv' % i
    problem.learn(learningBatchSize,offset = learningBatchSize*i, fileName= fileName)





##Letting that here that 

#[-0.23202347,  0.00146731,  0.12682147, -0.00300641,  0.21202325, 1.36603787]


#problem.pol.set(theta = np.array([-0.23202347,  0.00146731,  0.12682147, -0.00300641,  0.21202325, 1.36603787]))
#Is a good policy apparently
