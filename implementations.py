#implemented functions

import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def split_data(x, y, ratio=0.8, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def AIC_forward(y, x):
    """Model building by forward selection based on AIC criterion"""
    left = set(range(1, x.shape[1]))
    picked = [0]
    
    current, new = 1000000.0, 1000000.0
    
    while left and current == new:
        
        aics_cov = []
        
        for covariate in left:
            columns = picked + [covariate]
            loss = least_squares(y, x[:,columns])[1]
            aic = 2*loss*y.shape[0] + 2*len(columns)
            aics_cov.append((aic, covariate))
        
        aics_cov.sort()
        new, best_cov = aics_cov[0]
        
        if current > new:
            left.remove(best_cov)
            picked.append(best_cov)
            current = new
            
    return picked

def least_squares(y,tx):
    """Least squares regression with normal equations"""
    w = np.linalg.inv(tx.T@tx)@tx.T@y
    e=(y-tx@w)
    loss = 1/2*np.mean(e**2)
    return w, loss
    
def ridge_regression(y, tx, lambda_):
    """Ridge regression"""
    lambda_prime = (lambda_*2*y.shape[0]) * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + lambda_prime
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    e = y - (tx.dot(w))
    loss = (1/2)*np.mean(e**2)+lambda_*np.dot(w.T,w)
    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    """Least squares regression with gradient descent"""
    w = initial_w
    
    for i in range(max_iters):
        
        #compute errors and gradient
        e = y - tx.dot(w)
        grad = -tx.T.dot(e) / len(e)
             
        #update weight vector
        w = w - grad * gamma
        
    #compute loss
    e = y - tx.dot(w)
    loss = 1/2*np.mean(e**2)
    
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    """Least squares regression with stochastic gradient descent"""
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
    """Sigmoid function"""
    return 1 / (1+np.exp(-z))

def loss_f_lr(h, y):
    """Logistic regression likelihood function loss"""
    return -y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    h = 0
    
    for i in range(max_iters):
    
        #Compute tx*w
        z = np.dot(tx, w)
        #Compute sigmoid of z
        h = sigmoid(z)

        #Compute gradient
        gradient = tx.T.dot(h-y)
        
        #Update weight vector
        update = gamma*gradient
        w = w - update
    
    #Note that we return the total sample loss and not the mean loss
    h = sigmoid(tx.dot(w))
    loss = loss_f_lr(h, y)
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression"""
    w = initial_w
    h = 0
    
    for i in range(max_iters):
    
        z = np.dot(tx, w)
        h = sigmoid(z)

        #The regularization constraint is factored in the loss and gradient computation
        gradient = tx.T.dot(h-y) + lambda_*w
        w = w - gamma*gradient
    
    h = sigmoid(tx.dot(w))
    loss = loss_f_lr(h, y) + (1/2)*lambda_*np.dot(w.T,w)
    
    return w, loss

def model_pick_ridge(x, y, ratio=0.8, seed=1, degrees=range(1,2), lambdas=range(1)):
    """Pick best polynomial basis expansion and ridge regression lambda based on cross
    validation mse scores"""

    # split data
    x_tr, x_te, y_tr, y_te = split_data(x, y, ratio, seed)

    # ridge regression with different basis degrees and lambdas
    rmse_tr = []
    rmse_te = []
    
    for degree in degrees:
        tx_tr = build_poly(x_tr, degree)
        tx_te = build_poly(x_te, degree)
        for lambda_ in lambdas:
            weight = ridge_regression(y_tr, tx_tr, lambda_)
            rmse_tr.append((np.sqrt(2 * compute_mse(y_tr, tx_tr, weight)), degree, lambda_))
            rmse_te.append((np.sqrt(2 * compute_mse(y_te, tx_te, weight)), degree, lambda_))
            
    rmse_tr.sort()
    rmse_te.sort()
    return rmse_tr, rmse_te