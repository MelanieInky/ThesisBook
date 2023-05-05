from testProblemClass import TestProblem

import polars as pl
import numpy as np
from datetime import datetime
import os
from plotnine import *

#########HYPERPARAMETERS
gamma = 0
exploration_rate = 0.1
episode_length = 20
learning_rate = 2e-4
initial_theta = np.array([0,0,0.,0.,0.,0.])
n_episodes = 20000 #Number of episodes to train for
min_n = 5
max_n = 200
min_b = 0
max_b = 1
policy_version = 'scaled'
decay = False

#############

# Create new test problem
problem = TestProblem(gamma=gamma,
                    learning_rate=learning_rate,
                    exploration_rate=exploration_rate,
                    episode_length=episode_length,
                    min_b= min_b,
                    max_b = max_b,
                    min_n = min_n,
                    max_n = max_n,
                    decay = decay)

# And a set theta parameters to start with
problem.policy.set_theta(theta=initial_theta)
problem.policy.version = policy_version


######### Everything is set
### Run from here to continue training

# Prepare everything needed to log our training
date = datetime.now().strftime('%d%b-%Hh%Mm%Ss')
os.mkdir('Log/' + date)
config_log_file = 'Log/' + date + '/config_log.txt'


#Writing the parameters used here 
with open(config_log_file, 'a') as f:
    log_str = "Experiment training parameters: \n"
    log_str += 'gamma: %s\n' % gamma
    log_str += 'Learning Rate: %s\n' % learning_rate
    log_str += 'Exploration rate/Std_dev: %s\n' % exploration_rate
    log_str += 'Episode length: %s\n' % episode_length
    log_str += 'Initial theta: ' \
        + np.array2string(initial_theta, separator=',') + '\n'
    #log_str += 'Learning multiplier: ' \
    #    + np.array2string(learning_mult, separator=',') + '\n'
    log_str += 'Number of episodes: '+  str(n_episodes) + '\n'
    log_str += 'b can vary between [%s,%s)\n' % (min_b, max_b)
    log_str += 'n can vary between [%s, %s]\n' % (min_n,max_n)
    log_str += 'policy version: ' + policy_version + '\n'
    log_str += 'Decay' + str(decay)
    f.write(log_str)
    print(log_str)


fileName = 'Log/' + date + '/log.csv'
problem.learn(length=n_episodes, fileName=fileName)


df = pl.read_csv(fileName)



df2 = df.melt( 
    id_vars=['ep_number','avg_ep_reward'], 
    value_vars=['theta_0', 'theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5']
    ).with_columns(
        (pl.col('ep_number')/1000).alias('episode_no_1000')
    )

df3 = df.select('ep_number', 'avg_ep_reward').with_columns(pl.col('avg_ep_reward').rolling_mean(window_size=1000).alias('rolling reward'))



p = (ggplot(df2, aes('episode_no_1000','value'))
+ geom_line()
+ facet_wrap('~variable', scales='free_y')
+ theme_seaborn()
+ theme(panel_spacing_x= 0.7)
+ xlab('Episode no. / 1000')
+ ylab('Theta')
)



p2 = (ggplot(df3, aes('ep_number','rolling reward'))
+ geom_line()
+ theme_seaborn()
+ xlab('Episode number'))

p
p2


###### Some values that seems okay
#[-9.24809711e-03,  3.23655991e-04,  1.24970436e-01, -3.84203665e-03,
#        2.26767152e-01,  1.49379107e+00])

######

