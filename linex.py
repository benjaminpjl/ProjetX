

alpha = 0.1
import numpy as np

def linex(y_true, y_pred):
    
    grad = (-alpha*np.exp(alpha*(y_true-y_pred)) + alpha)
    hess = alpha*alpha*np.exp(alpha*(y_true-y_pred))
    
    return grad, hess

def loss_linex(y_true, y_pred):
    
    return np.exp(alpha*(y_true-y_pred)) - alpha*(y_true-y_pred) - 1


