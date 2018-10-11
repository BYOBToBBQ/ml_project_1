def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    
    #initialize w
    w = initial_w
    
    #loop for the iterations of the gradient
    for loop in range(max_iters):
        
	#generate the random line where the gradient is computed
        n = np.random.randint(len(y))
        #computes the errors and one value of the gradient 
        e = y - tx.dot(w)
        grad = (-tx[n] * e[n])  * np.ones(len(w))
        
        #calcultate the loss through the mean square method
        loss = 1/2*np.mean(e**2)
        
        #gradient descent
        w = w - grad * gamma
    
    return loss, w
