def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    
    #initialize w
    w = initial_w
    
    #loop for the iterations of the gradient
    for loop in range(max_iters):
        
        #computes the errors and the gradient
        e = y - tx.dot(w)
        grad = -tx.T.dot(e) / len(e)
        
        #calcultate the loss through the mean square method
        loss = 1/2*np.mean(e**2)
        
        #gradient descent
        w = w - grad * gamma
    
    return loss, w
