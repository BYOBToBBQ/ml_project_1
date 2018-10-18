#implemented functions

import numpy as np


def least_squares(y,tx):
    #max likelihood estimator for w
    w = np.linalg.inv(tx.T@tx)@tx.T@y
    e=(y-tx@w)
    loss = 1/2*np.mean(e**2)
    return loss,w
 
    
def ridge_regression(y, tx, lambda_):
    lambda_prime = lambda_*2*len(y)
    w = np.linalg.inv(tx.T@tx + lambda_prime*np.eye(tx.shape[1]))@tx.T@y
    e = (y-tx@w)
    loss = 1/2*np.mean(e**2)+lambda_*np.dot(w,w)
    return loss,w


def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    
    #initialize w
    w = initial_w
    
    #loop for the iterations of the gradient
    for loop in range(max_iters):
        
        #computes the errors and the gradient
        e = y - tx.dot(w)
        grad = -tx.T.dot(e) / len(e)
             
        #gradient descent
        w = w - grad * gamma
        
    #calcultate the loss through the mean square method
    loss = 1/2*np.mean(e**2)
    
    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    
    #initialize w
    w = initial_w
    
    #loop for the iterations of the gradient
    for loop in range(max_iters):
        
	#generate the random line where the gradient is computed
        n = np.random.randint(len(y))
        #computes the errors and one value of the gradient 
        e = y[n] - tx[n].dot(w)
        grad = (-tx[n] * e)  * np.ones(len(w))
              
        #gradient descent
        w = w - grad * gamma
        
    #calcultate the loss through the mean square method
    err = y - tx.dot(w)
    loss = 1/2*np.mean(err**2)
    
    return loss, w