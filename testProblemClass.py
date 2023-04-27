import numpy as np

# This is where we define the class. TestProblem as most of everything


class TestProblem:
    # Define it as
    def __init__(self, gamma=0.1, learning_rate=0.0000005, episode_length=100, exploration_rate=0.1, learning_mult = np.ones(6),b=0.05, n=100) -> None:
        """Generate a test problem, with some initial parameters"""
        self.n = n
        self.b = b
        self.delta_x = 1 / (n+1)
        self.M = self._build_M()
        self.e = self._build_e()
        # Get a Policy at random, or set it later
        self.exploration_rate = exploration_rate
        self.pol = Policy(exploration_rate)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_mult = learning_mult #Multiplier for directions
        self.episode_length = episode_length

    def _build_M(self):
        """
        we go from u0 to u(n+1).
        """
        delta_x = self.delta_x
        n = self.n
        b = self.b
        A = 1/delta_x * (np.eye(n) - 1 * np.eye(n, k=-1))
        B = 1/(delta_x ** 2) * b * (-2*np.eye(n) +
                                   np.eye(n, k=-1) + np.eye(n, k=1))
        # We scale it back
        gamma = 1/delta_x + 2 / (delta_x ** 2) * b
        return (A-B)/gamma

    def _build_e(self):
        delta_x = self.delta_x
        b = self.b
        gamma = 1/delta_x + 2/(delta_x ** 2) * b
        return np.ones(self.n) / gamma

    def f(self, y):
        """
        Return the residual r = e-My
        """
        return self.e - self.M@y

    def _one_step_smoother(self, y, t, delta_t, alpha):
        """
        Perform one pseudo time step delta_t of the solver for the diff eq
        y' = e - My = f(y). .
        """
        k1 = self.f(y)
        k2 = self.f(y + alpha*delta_t*k1)
        yNext = y + delta_t*k2
        return yNext

    def update(self, b, n):
        """ Change the problem parameters to the given one and automatically builds the mat-vec to solve for."""
        self.b = b
        self.n = n
        self.M = self._build_M()
        self.e = self._build_e()
        self.delta_x = 1 / (n+1)

    def main_solver(self, n_iter=10):
        """ Main solver for the problem, 
        use the current Policy to calculate the approximated solution
        after n_iter pseudo time steps. """
        t = 0
        # Initial guess y = e
        y = np.copy(self.e)
        resNorm = np.linalg.norm(self.M@y-self.e)
        lastResNorm = resNorm
        # Finding the parameters according to current Policy
        alpha, delta_t = self.pol(self.b, self.n)
        for i in range(n_iter):
            y = self._one_step_smoother(y, t, delta_t, alpha)
            t += delta_t
            newResNorm = np.linalg.norm(self.M@y - self.e)
            # Check for overflow!
            if np.isinf(newResNorm) or np.isnan(newResNorm):
                print("Overflow while calculating residual!")
                return y, resNorm, lastResNorm

            lastResNorm = resNorm
            resNorm = newResNorm
        return y, resNorm, lastResNorm

    def main_solver2(self, alpha, delta_t, n_iter=10):
        """ Like the main solver, except we give 
        the parameters explicitely """
        res_norm_list = np.zeros(n_iter+1)
        t = 0
        # Initial guess y = e
        y = np.copy(self.e)
        res_norm_list[0] = np.linalg.norm(self.M@y-self.e, 2)
        # For now, we use our best guess
        for i in range(n_iter):
            y = self._one_step_smoother(y, t, delta_t, alpha)
            t += delta_t
            resNorm = np.linalg.norm(self.M@y - self.e, 2)
            res_norm_list[i+1] = resNorm
        return y, res_norm_list

    def _get_action_and_reward(self, n_iter=10):
        """Take an action, given the current problem parameters and Policy,
        then compute the reward. 
        (This is similar to the step function in gym environments.)
        Returns the action taken, and the associated reward.
        """
        t = 0
        # Initial guess y = e
        y = np.copy(self.e)
        resNorm = np.linalg.norm(self.M@y-self.e)
        lastResNorm = resNorm
        # Finding the parameters according to current Policy
        alpha, delta_t = self.pol(self.b, self.n)
        for i in range(n_iter):
            y = self._one_step_smoother(y, t, delta_t, alpha)
            t += delta_t
            newResNorm = np.linalg.norm(self.M@y - self.e)
            # Check for overflow!
            if np.isinf(newResNorm) or np.isnan(newResNorm) or newResNorm > 1e20:
                print("Divering greatly! Capping the reward at -10")
                ratio = resNorm / lastResNorm
                return alpha, delta_t, -3
            lastResNorm = resNorm
            resNorm = newResNorm
        ratio = resNorm / lastResNorm
        # Compute reward
        if (ratio <= 1):
            reward = 500*(1-ratio)
        else:
            reward = max(-3, 1-ratio)
        return alpha, delta_t, reward

    def _generate_episode(self, length=5):
        """Given the current Policy, generate an episode of fixed length"""
        b_history = np.zeros(length)
        n_history = np.zeros(length)
        alpha_history = np.zeros(length)
        delta_t_history = np.zeros(length)
        reward_history = np.zeros(length)
        for i in range(length):
            b_history[i] = self.b
            n_history[i] = self.n
            # Compute action and reward
            alpha, delta_t, reward = self._get_action_and_reward()
            alpha_history[i] = alpha
            delta_t_history[i] = delta_t
            reward_history[i] = reward
            # gradient = self.pol.log_pdf_gradient(self.b,self.n,alpha,delta_t)

            # Then jump to a new state, at random
            newb = np.random.uniform(0, 1)
            newn = np.random.randint(5, 200)
            self.update(newb, newn)

        myEpisode = Episode(b_history, n_history, alpha_history,
                            delta_t_history, reward_history)
        return myEpisode

    def _learn_one_episode(self, ep_length=100):
        """
        Generate an episode, then update the Policy parameters accordingly. 
        Return the episode history for logging purposes.
        """
        gamma = self.gamma
        learning_rate = self.learning_rate
        myEpisode = self._generate_episode(ep_length)
        for t in range(ep_length):
            # Calculate the sum of rewards
            Gt = 0
            for i in range(t, ep_length):
                # Return on the trajectory
                Gt += myEpisode.reward_hist[i] * (gamma**(i-t))

            b = myEpisode.b_hist[t]
            n = myEpisode.n_hist[t]
            alpha = myEpisode.alpha_hist[t]
            delta_t = myEpisode.delta_t_hist[t]
            # Calculate the log likelihood gradient, for the current Policy
            log_pdf_grad = self.pol.log_pdf_gradient(b, n, alpha, delta_t)
            #Update the Policy parameters
            self.pol.theta += learning_rate*self.learning_mult*log_pdf_grad*Gt
        return myEpisode

    def learn(self, length=100, log=True, fileName='Log/log.csv'):
        """
        Repeat the training process for a certain amount of episode.  
        """
        # The offset is only for logging purposes.
        # So we can write the correct Episode number
        # When making multiple files
        if (log):
            f = open(fileName, 'a')
            firstLine = 'ep_number,avg_ep_reward,theta_0,theta_1,theta_2,theta_3,theta_4,theta_5\n'
            f.write(firstLine)
            print('Logging activated')
        # Loop for a certain number of episodes.
        mean_episode_rewards = np.zeros(length)
        for i in range(length):
            ###This is really the only thing it is doing!!
            ep = self._learn_one_episode(self.episode_length)
            ###########################################################
            ### Section for logging and printing stuff.
            #Get the average reward in the episode.
            mean_episode_rewards[i] = np.mean(ep.reward_hist)
            # Printing progress
            if (i % 200 == 0 and i != 0):
                print('episode number ', i, ' done.')
                mean_rew = np.mean(mean_episode_rewards[i-200:i])
                print('mean reward of the last 200 episodes : ', mean_rew)
            # Logging
            if (log):
                to_write1 = '%d,%f,' % (i, mean_episode_rewards[i])
                theta_to_str = np.array2string(self.pol.theta, separator=',').lstrip(
                    '[').rstrip(']').replace('\n', '')
                total_string = to_write1 + theta_to_str + '\n'
                total_string = total_string.replace(" ", "")
                f.write(total_string)
            ###########################################################
        if (log):
            f.close()
        print('Training done!')
        return 0


class Episode:
    """
    An episode is defined as an history of the state, action and reward
    So for example b[t] is the b parameter at instant t in the episode.
    Then alpha[t] is the action taken at instant t
    reward[t] is the reward we get AFTER taking action A_t at state S_t!
    """
    def __init__(self, b, n, alpha, delta_t, reward):
        self.b_hist = b
        self.n_hist = n
        self.alpha_hist = alpha
        self.delta_t_hist = delta_t
        self.reward_hist = reward

    def length(self):
        return len(self.b_hist)

    def log_str(self, k):
        # Log the state-action-reward-etc at time k.
        log_str = "%s , %s , %s , %s , %s , %s" % (
            k, self.b_hist[k], self.n_hist[k], self.alpha_hist[k], self.delta_t_hist[k], self.reward_hist[k])
        return log_str


class Policy:
    """ Policy class. Will be used to compute all sorts of things"""

    def __init__(self, std_dev=0.1) -> None:
        """ Policy, first with random values"""
        self.theta = np.random.rand(6)/200
        self.stdDev = std_dev  # Will be learned later

    def set_theta(self, theta):
        self.theta = theta

    def __call__(self, b, n, deterministic = False):
        """ Evaluate the Policy and return the alpha, delta_t parameters,
        if deterministic = True, return the mean parameters instead of adding noise.  
        """
        theta = self.theta
        alpha_mean = theta[0]*b + theta[1] * n + theta[4]
        delta_t_mean = theta[2]*b + theta[3]*n + theta[5]
        alpha = np.random.normal(alpha_mean, self.stdDev)
        delta_t = np.random.normal(delta_t_mean, self.stdDev)
        if(deterministic):
            return alpha_mean , delta_t_mean
        return alpha, delta_t

    def __str__(self) -> str:
        """Overload for the print"""
        D = np.reshape(self.theta[:4], (2, 2))
        part1 = "Policy function, of the form [alpha , delta_t] = D*[b,n]' + bias, with D = \n"
        part2 = D.__str__()
        part3 = "\nand b = " + self.theta[4:].__str__()
        part4 = "\nwith common std dev of " + str(self.stdDev) + "."
        return part1 + part2 + part3 + part4

    def log_pdf_gradient(self, b, n, alpha, delta_t):
        """ Get the gradient of log pdf, for each parameters"""
        theta = self.theta
        xi1 = (alpha - theta[0] * b - theta[1] *
               n - theta[4])/(self.stdDev**2)
        xi2 = (delta_t - theta[2] * b - theta[3]
               * n - theta[5])/(self.stdDev**2)
        grad_theta = np.zeros(6)
        grad_theta[0] = b*xi1
        grad_theta[1] = n*xi1
        grad_theta[4] = xi1
        grad_theta[2] = b*xi2
        grad_theta[3] = n*xi2
        grad_theta[5] = xi2
        self.gradient = grad_theta
        return grad_theta
    
    




# Letting that here that

# [-0.23202347,  0.00146731,  0.12682147, -0.00300641,  0.21202325, 1.36603787]


# problem.pol.set(theta = np.array([-0.23202347,  0.00146731,  0.12682147, -0.00300641,  0.21202325, 1.36603787]))
# Is a good Policy apparently


# First training used 0.00000008*grad. std deviation 0.1, gmma = 0.1 and episode lenght 100
