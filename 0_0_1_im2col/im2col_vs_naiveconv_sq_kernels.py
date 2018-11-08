
import numpy as np
from utils import timeit
def image2D(N): return np.random.randint(-10,10,size=(N,N))
def filter2D(K): return np.random.randint(-3,3,size=(K,K))

X = image2D(3)
F = filter2D(2)
s = 1
p = 0

# Working in 2D Space
# -------------------

def output_dim(X,F,s,p):
    ''' Returns the output dimension of a convolution'''
    return int((X.shape[0] + 2*p - F.shape[0])/s + 1)

@timeit
def convolve_region(X,F):
    out = 0
    K = F.shape[0]
    for i in range(K):
        for j in range(K):
            out += X[i,j] * F[i,j]
    return out
        
convolve_region(X[:2,:2], F)
# 'convolve_region'  0.02 ms


@timeit
def naiveconv1(X, F, s=1, p=0):
    ''' Convolve 1 channel with 1 filter 
        Assume square images and kernels '''
    # Compute output dimensions
    O = int(output_dim(X,F,s,p))
    N = 1 ## 1 will be number of filters
    K = F.shape[0]
    FM = np.zeros((N,O,O)) 
    
    # Sliding filter horizontally
    for i in range(O):
        # Sliding filter vertically
        for j in range(O):
             # Convolve the filter on that region of the image
             FM[0,i,j] = convolve_region(X[i:K+i, j:K+j], F)
    return FM

out = naiveconv1(X,F)
#'output_dim'  0.01 ms
#'convolve_region'  0.01 ms
#'convolve_region'  0.01 ms
#'convolve_region'  0.00 ms
#'convolve_region'  0.01 ms
#'naiveconv1'  0.31 ms


@timeit
def im2col(X, F, s=1, p=0):
    ''' im2col '''
    K = F.shape[0]
    O = output_dim(X,F,s,p)
    O1, O2 = K**2, output_dim(X,F,s,p)**2 # Review
    X_ = np.zeros((O1,O2))
    
    col = 0
    # Sliding horizontally
    for i in range(O):
        # Sliding vertically
        for j in range(O):
            region = X[i:K+i, j:K+j]
            # Retrieve that region of the image as a columns
            X_[:,col] = region.ravel()
            col += 1
    return X_
  
    
@timeit
def im2col_inverse(x_, F, s=1, p=0):
    N = 1
    O = output_dim(X,F,s,p)
    out = np.zeros((N,O,O))
    
    row = 0
    # Populate horizontally
    for i in range(O):
        # Populate vertically
        for j in range(O):
            out[0,i,j] = x_[row]
            row += 1
    return out

x_ = im2col(X,F)
f_ = F.ravel()
out_ = x_.T @ f_
out_ = im2col_inverse(out_, F)
(out == out_).all()
#'im2col'  0.16 ms
#'im2col_inverse'  0.05 ms

     
# Time comparison
@timeit
def regular_convolution(X,F,s=1,p=0):
    return naiveconv1(X,F,s,p)

@timeit
def im2col_convolution(X,F,s=1,p=0):
    x_ = im2col(X,F).T @ F.ravel()
    return im2col_inverse(x_,F,s,p)

out = regular_convolution(X,F)
out_ = im2col_convolution(X,F)
#'regular_convolution'  0.63 ms

#'im2col'  0.53 ms
#'im2col_inverse'  0.05 ms
#'im2col_convolution'  0.82 ms

# mmm interesting, at this level the price of im2col is very expensive!

    

# Bigger Images
X = image2D(100)
F = filter2D(3)
out = regular_convolution(X,F)
out_ = im2col_convolution(X,F)

#'naiveconv1'  1256.98 ms
#'regular_convolution'  1256.99 ms
#'im2col'  67.09 ms
#'im2col_inverse'  4.83 ms
#'im2col_convolution'  183.01 ms




# Working in 3D Space
# -------------------
    
image3D = np.array([image(), image(), image()])
filter3D = np.array([filt(), filt(), filt()])

    