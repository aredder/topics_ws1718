import numpy as np
import logging
from numpy import linalg as LA
logger = logging.getLogger(__name__)

def PCA(X,n):
    X = X - X.mean(axis=0)
    U, S, V = LA.svd(X,full_matrices=False)
    T = np.matmul(U,np.diag(S))
    return T[:,0:n]

def LLE(X,k,n_comp):
    ### 1. compute k nearest neighbors
    # compute distance matrix
    m, n = X.shape
    P = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            P[i,j] = np.sqrt(np.sum(np.square(X[i,:]-X[j,:])))
    # find k nearest neighbors
    KNN = np.zeros((m,k))
    for i in range(m):
        KNN[i,:] = np.argsort(P[i, :])[0:k]
    ### 2. compute reconstruction weight matrix
    # when the number of nearest neighbors exceeds the number of features use regularization
    if k>n:
        alpha = 0.001
    else:
        alpha = 0
    w = np.zeros((m, m))    
    for i in range(m): # run over all samples
        # run over all NN and compute Gram matrix
        z = np.zeros((k,n))
        ind = 0
        for j in KNN[i,:]:  
            z[ind,:] = X[int(j),:] - X[i,:]
            ind = ind+1
        G = np.matmul(z,z.T) 
        # minimize the Lagrangian and normalize to get probabilities
        w_temp = np.matmul(LA.inv(G+np.eye(k)*alpha),np.ones((k,1)))/2
        w_temp = w_temp/np.sum(w_temp)
        for l in range(k):
            w[i,int(KNN[i,l])] = w_temp[l]
    ### 3. find coordinates
    M = np.matmul(np.transpose(np.eye(m)-w),np.eye(m)-w)
    W, V = LA.eig(M)
    idx = W.argsort()[::1]   
    W = W[idx]
    V = V[:,idx]
    return V[:,0:n_comp]

