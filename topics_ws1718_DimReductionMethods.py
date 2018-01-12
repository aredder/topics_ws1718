import numpy as np
import logging
from numpy import linalg as LA
logger = logging.getLogger(__name__)

def PCA(X,n):
    X = X - X.mean(axis=0)
    U, S, V = LA.svd(X,full_matrices=False)
    T = np.matmul(U,np.diag(S))
    return T

