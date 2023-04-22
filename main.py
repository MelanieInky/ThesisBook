from testProblemClass import TestProblem
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os



#########HYPERPARAMETERS
gamma = 0.1
exploration_rate = 0.5
episode_length = 50
learning_rate = 1e-6
learning_mult = np.array([1,1/20,1,1/20,1,1])
initial_theta = np.array([0., 0., 0., 0., 0., 0.])
n_episodes = 40000 #Number of episodes to train for
#############


# Create new test problem
problem = TestProblem(gamma=gamma,
                      learning_rate=learning_rate,
                      exploration_rate=exploration_rate,
                      episode_length=episode_length,
                      learning_mult=learning_mult)

# And a set theta parameters to start with
problem.pol.set_theta(theta=initial_theta)

# Prepare everything needed to log our training
date = datetime.now().strftime('%d%b-%Hh%Mm%Ss')
os.mkdir('Log/' + date)
config_log_file = 'Log/' + date + '/config_log.txt'


#Writing the parameters used here 
with open(config_log_file, 'a') as f:
    f.write("Training parameters\n")
    f.write('gamma: %s\n' % gamma)
    f.write('Learning Rate: %s\n' % learning_rate)
    f.write('Exploration rate/Std_dev: %s\n' % exploration_rate)
    f.write('Episode length: %s\n' % episode_length)
    f.write('Initial theta: ' + np.array2string(initial_theta, separator=',') + '\n')
    f.write('Learning multiplier: ' + np.array2string(learning_mult, separator=','))


fileName = 'Log/' + date + '/log.csv'
problem.learn(length=n_episodes, fileName=fileName)


df = pl.read_csv(fileName)

plt.plot(df.select(pl.col('theta_5')))
plt.show()
plt.plot(df.select(pl.col('avg_ep_reward')))
plt.show()