
import numpy as np
from ssm import util

# parameters of the gLDS
def test_implementation(user_function):
    d_latent = 2
    d_observation = 3
    A = .99* util.random_rotation(d_latent) # dynamics matrix, a slowly decaying rotation
    C = np.random.rand(d_observation,d_latent) # observation matrix, random

    Q = np.diag(np.random.rand(d_latent,)) # state noise covariance
    R = np.diag(np.random.rand(d_observation,)) # observation noise covariance


    pi_0 = np.zeros((d_latent,)) # initial state mean
    V_0 = Q # initial state covariance
    
    num_timesteps = 200
    seed = 2
    X, Y = mystery_function(A, C, Q, R, pi_0, V_0, num_timesteps, seed)
    X_user, Y_user = user_function(A, C, Q, R, pi_0, V_0, num_timesteps,seed)
    
    
    if not X_user.shape == X.shape:
        if not Y_user.shape == Y.shape:
            print('Try again! The shape of both your X and Y look wrong -- they should be {} and {}'.format(X.shape, Y.shape))
            return
        else:
            print('Try again! The shape of your X looks wrong -- it should be {}'.format(X.shape))
            return
    else: 
        if not Y_user.shape == Y.shape:
            print('Try again! The shape of your Y looks wrong -- it should be {}'.format(Y.shape))
            return
    

    if not np.array_equal(X, X_user):
        if not np.array_equal(Y, Y_user):
            print('Try again! Neither X nor Y matches our solution.')
    else:
        if not np.array_equal(Y, Y_user):
            print('Try again! X matches our solution, but Y does not')
        else:
            print('Good job! Your implementation matches our solution')
        

    
# don't look here u cheat    
def mystery_function(A, C, Q, R, pi_0, V_0, num_timesteps, seed=0):
    """ Generates a sequence of states and observations from a gaussian linear dynamical system.

    Args:
        A (np.matrix): Dynamics matrix (d_latent, d_latent)
        Q (np.matrix): State noise covariance (d_latent, d_latent)
        C (np.matrix): Observation matrix (d_observation, d_latent)
        R (np.matrix): Observation noise covariance (d_observation, d_observation)
        pi_0 (np.array): Initial state mean (d_latent, )
        V_0 (np.matrix): Initial state covariance (d_latent, d_latent)
        num_timesteps (int, optional): number of iterations for EM

    Returns:
        X (np.ndarray): (num_timesteps, d_latent) time-series of states
        Y (np.ndarray): (num_timesteps, d_observations) time-series of observations
    """
    np.random.seed(seed) # for consistency
    assert A.shape[0] == A.shape[1], "Dynamics matrix must be square"
    assert Q.shape[0] == Q.shape[1], "State noise covariance must be square"
    assert C.shape[1] == A.shape[1], "Number of columns in observation matrix must match d_latent "
    assert R.shape[0] == R.shape[1], "Observation noise covariance must be square"
    
    d_latent = A.shape[0]
    d_observation = C.shape[0]

    X = [] # list of states 
    Y = [] # list of observations
    
    state_noise_mean = np.zeros((d_latent,))
    observation_noise_mean = np.zeros((d_observation,))
    
    # generate initial state and observation
    x = np.random.multivariate_normal(pi_0, V_0).T
    y = C.dot(x) + np.random.multivariate_normal(observation_noise_mean, R)
    
    # add x and y to their respective lists
    X.append(x)
    Y.append(y)
    
    for _ in range(1, num_timesteps):
        """TODO: your code goes here! Fill in the formulas for x and y."""
        
        x = A.dot(x) + np.random.multivariate_normal(state_noise_mean,Q) #replace with function for x
        y = C.dot(x) + np.random.multivariate_normal(observation_noise_mean,R) #replace with function for y
        
        """End your code"""
        X.append(x)
        Y.append(y)
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y