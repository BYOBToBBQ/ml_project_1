#implemented functions

import numpy as np


def least_squares(y,tx):
    w = np.linalg.inv(tx.T@tx)@tx.T@y
    e=(y-tx@w)
    loss = 1/2*np.mean(e**2)
    return w, loss
 
    
def ridge_regression(y, tx, lambda_):
    lambda_prime = lambda_*2*len(y)
    w = np.linalg.inv(tx.T@tx + lambda_prime*np.eye(tx.shape[1]))@tx.T@y
    e = (y-tx@w)
    loss = 1/2*np.mean(e**2)+lambda_*np.dot(w.T,w)
    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    
    w = initial_w
    
    for i in range(max_iters):
        
        #compute errors and gradient
        e = y - tx.dot(w)
        grad = -tx.T.dot(e) / len(e)
             
        #update weight vector
        w = w - grad * gamma
        
    #compute loss
    loss = 1/2*np.mean(e**2)
    
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    
    w = initial_w
    
    for i in range(max_iters):
        
        #pick one observation for gradient computation
        n = np.random.randint(len(y))
        #compute errors and gradient based on that observation 
        e = y[n] - tx[n].dot(w)
        grad = (-tx[n] * e)
              
        #update weight vector
        w = w - grad * gamma
        
    #compute loss
    err = y - tx.dot(w)
    loss = 1/2*np.mean(err**2)
    
    return w, loss

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def loss_f_lr(h, y):
    return -y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
   
    w = initial_w
    h = 0
    
    for i in range(max_iters):
    
        #Compute x_t*w
        z = np.dot(tx, w)
        #Compute sigmoid of z
        h = sigmoid(z)

        #Compute gradient
        gradient = tx.T.dot(h-y)
        
        #Update weight vector
        update = gamma*gradient
        w = w - update
    
    #Note that we return the total sample loss and not the mean loss
    loss = loss_f_lr(h, y)
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    w = initial_w
    h = 0
    
    for i in range(max_iters):
    
        z = np.dot(tx, w)
        h = sigmoid(z)

        #The regularization constraint is factored in the loss and gradient computation
        gradient = tx.T.dot(h-y) + lambda_*w
        w = w - gamma*gradient
    
    loss = loss_f_lr(h, y) + (1/2)*lambda_*np.dot(w.T,w)
    
    return w, loss