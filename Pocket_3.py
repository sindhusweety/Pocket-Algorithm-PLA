# -*- coding: utf-8 -*-
# TODO: SAVE YOUR SOLUTION AS Pocket_3.py

import numpy as np

SEED = 424242
rng = np.random.default_rng(SEED)

def Pocket_3(Xs, ys, T):
    '''
    Compare the Ein performance of the Pocket based on PLA and 
    based on Adaline.

    Parameters
    ----------
    Xs : N x d x 20 NumPy array
        20 X arrays suitable for passing to Pocket_1.
    ys : N x 20 NumPy array
        20 y vectors of +1/-1 labels for each of the X points.
    T : positive int
        Number of times to iterate the main loop of the Pocket algorithm.

    Returns
    -------
    avg_pocket_PLA_Ein : Value in real interval (0,1]
        In-sample error for pocket algorithm based on PLA
    avg_pocket_Adaline_Ein : Value in real interval (0,1]
        In-sample error for pocket algorithm based on Adaline

    '''
    avg_pocket_PLA_Ein = 0
    avg_pocket_Adaline_Ein = 0
    
    #TODO: YOUR CODE GOES BELOW THIS LINE
    for r in range(Xs.shape[2]):
        X = Xs[:,:,r]
        y = ys[:, r]
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # adding bias feature to X
        w_hat=Ein_Pocket_PLA(X, y, T)
        w_hat_a = Ein_pocket_adaline(X, y, T)
        PLA_Ein = np.mean(np.sign(X.dot(w_hat)) != y)
        pocket_Ein = np.mean(np.sign(X.dot(w_hat_a)) != y)
        avg_pocket_PLA_Ein += PLA_Ein
        avg_pocket_Adaline_Ein += pocket_Ein


    avg_pocket_PLA_Ein /= Xs.shape[2]

    avg_pocket_Adaline_Ein /= Xs.shape[2]

    
    print(f"Average pocket PLA Ein:     {avg_pocket_PLA_Ein}")
    print(f"Average pocket Adaline Ein: {avg_pocket_Adaline_Ein}")
 
    return (avg_pocket_PLA_Ein, avg_pocket_Adaline_Ein)
    
#TODO: ADDITIONAL CODE CAN OPTIONALLY GO BELOW THIS LINE

def Ein_Pocket_PLA(X, y, T):

    # np.insert(X, obj=0, values= 1, axis=1) #axis 0 -> rows 1 -> col & obj -> index

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
            index_of_misclassified = rng.choice(indices_of_misclassified)  # ----Random Choice

            w = w + X[index_of_misclassified] * y[index_of_misclassified]
            Ein = np.sum(np.sign(np.dot(X, w)) != y)  # calculate Ein for updated weight vector
            if Ein < Ein_min:  # update pocket weight vector w_hat if Ein improves
                w_hat = np.copy(w)
                Ein_min = Ein



    return  w_hat

def Ein_pocket_adaline(X, y, T):
    w_a = np.zeros(X.shape[1])  # initialize weight vector w
    w_hat_a = np.copy(w_a)  # initialize pocket weight vector w_hat
    Ein_adaline = np.sum(np.sign(X.dot(w_a)) != y)
    for t in range(T):
        # Generate array of indices of all misclassified examples.
        # Note that np.nonzero() returns a tuple containing the
        # array we want as its first (and only) element.
        predicted_y_ = np.sign(X @ w_a)

        indices_of_misclassified_ = np.nonzero(predicted_y_ != y)[0]
        if indices_of_misclassified_.size > 0:
            index_of_misclassified_ = rng.choice(indices_of_misclassified_)  # ----Random Choice

            w_a = w_a + 0.001 * (y[index_of_misclassified_] - (w_a.T @ X[index_of_misclassified_])) * X[
                index_of_misclassified_]
            Ein_a = np.sum(np.sign(np.dot(X, w_a)) != y)
            if Ein_a < Ein_adaline:  # update pocket weight vector w_hat if Ein improves
                w_hat_a = np.copy(w_a)
                Ein_adaline = Ein_a
    return  w_hat_a