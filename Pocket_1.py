# -*- coding: utf-8 -*-

# TODO: SAVE YOUR SOLUTION AS Pocket_1.py

import numpy as np

SEED = 424242
rng = np.random.default_rng(SEED)

def Pocket_1(X, y, T):
    '''
    Implement the Pocket algorithm using a PLA implementation that
    randomly selects each misclassified example used for weight update.

    Parameters
    ----------
    X : N x d NumPy array
        N points in d-dimensional space.
    y : N-element NumPy vector
        +1/-1 labels for each of the X points.
    T : positive int
        Number of times to iterate the main loop.

    Returns
    -------
    w_hat : (d+1)-element NumPy vector
        Weight vector defining a hyperplane separating the data.
        w[0] is the bias value.

    '''
    
    w_hat = None
    
    #TODO: YOUR CODE GOES BELOW THIS LINE

    X = np.hstack((np.ones((X.shape[0], 1)), X))  # adding bias feature to X
    #np.insert(X, obj=0, values= 1, axis=1) #axis 0 -> rows 1 -> col & obj -> index


    w = np.zeros(X.shape[1])  # initialize weight vector w
    w_hat = np.copy(w)  # initialize pocket weight vector w_hat
    Ein_min = np.sum(np.sign(X.dot(w)) != y)  # initialize minimum Ein


    for t in range(T):
        # Generate array of indices of all misclassified examples.
        # Note that np.nonzero() returns a tuple containing the
        # array we want as its first (and only) element.
        predicted_y = np.sign(X @ w)

        indices_of_misclassified = np.nonzero(predicted_y != y)[0]

        # Use a random misclassified example, if any, to perform a PLA
        # weight vector update. Otherwise, we are done updating.
        if indices_of_misclassified.size > 0:
            index_of_misclassified = rng.choice(indices_of_misclassified)  #----Random Choice
            w = w + X[index_of_misclassified] * y[index_of_misclassified]
            Ein = np.sum(np.sign(np.dot(X, w)) != y)  # calculate Ein for updated weight vector
            #X.dot(w)
            if Ein < Ein_min:  # update pocket weight vector w_hat if Ein improves
                w_hat = np.copy(w)
                Ein_min = Ein



    return w_hat

