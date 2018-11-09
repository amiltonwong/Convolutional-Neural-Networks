
import numpy as np
from utils import timeit
def image2D(H,W): return np.random.randint(-10,10,size=(H,W))
def filter2D(K): return np.random.randint(-3,3,size=(K,K))

#X = image2D(3,3)
#F = filter2D(2)
#s = 1
#p = 0
#
#
## Working in 2D Space
## -------------------
#
#def output_dim(X,F,s,p):
#    ''' Returns the output dimension of a convolution'''
#    return int((X.shape[0] + 2*p - F.shape[0])/s + 1)
#
#@timeit
#def convolve_region(X,F):
#    out = 0
#    K = F.shape[0]
#    for i in range(K):
#        for j in range(K):
#            out += X[i,j] * F[i,j]
#    return out
#        
#convolve_region(X[:2,:2], F)
## 'convolve_region'  0.02 ms
#
#
#@timeit
#def naiveconv1(X, F, s=1, p=0):
#    ''' Convolve 1 channel with 1 filter 
#        Assume square images and kernels '''
#    # Compute output dimensions
#    O = int(output_dim(X,F,s,p))
#    N = 1 ## 1 will be number of filters
#    K = F.shape[0]
#    FM = np.zeros((N,O,O)) 
#    
#    # Sliding filter horizontally
#    for i in range(O):
#        # Sliding filter vertically
#        for j in range(O):
#             # Convolve the filter on that region of the image
#             FM[0,i,j] = convolve_region(X[i:K+i, j:K+j], F)
#    return FM
#
#out = naiveconv1(X,F)
##'output_dim'  0.01 ms
##'convolve_region'  0.01 ms
##'convolve_region'  0.01 ms
##'convolve_region'  0.00 ms
##'convolve_region'  0.01 ms
##'naiveconv1'  0.31 ms
#
#
#@timeit
#def im2col(X, F, s=1, p=0):
#    ''' im2col '''
#    K = F.shape[0]
#    O = output_dim(X,F,s,p)
#    O1, O2 = K**2, output_dim(X,F,s,p)**2 # Review
#    X_ = np.zeros((O1,O2))
#    
#    col = 0
#    # Sliding horizontally
#    for i in range(O):
#        # Sliding vertically
#        for j in range(O):
#            region = X[i:K+i, j:K+j]
#            # Retrieve that region of the image as a columns
#            X_[:,col] = region.ravel()
#            col += 1
#    return X_
#  
#    
#@timeit
#def im2col_inverse(x_, F, s=1, p=0):
#    N = 1
#    O = output_dim(X,F,s,p)
#    out = np.zeros((N,O,O))
#    
#    row = 0
#    # Populate horizontally
#    for i in range(O):
#        # Populate vertically
#        for j in range(O):
#            out[0,i,j] = x_[row]
#            row += 1
#    return out
#
#x_ = im2col(X,F)
#f_ = F.ravel()
#out_ = x_.T @ f_
#out_ = im2col_inverse(out_, F)
#(out == out_).all()
##'im2col'  0.16 ms
##'im2col_inverse'  0.05 ms
#
#     
## Time comparison
#@timeit
#def regular_convolution(X,F,s=1,p=0):
#    return naiveconv1(X,F,s,p)
#
#@timeit
#def im2col_convolution(X,F,s=1,p=0):
#    x_ = im2col(X,F).T @ F.ravel()
#    return im2col_inverse(x_,F,s,p)
#
#out = regular_convolution(X,F)
#out_ = im2col_convolution(X,F)
##'regular_convolution'  0.63 ms
#
##'im2col'  0.53 ms
##'im2col_inverse'  0.05 ms
##'im2col_convolution'  0.82 ms
#
## mmm interesting, at this level the price of im2col is very expensive!
#
#    
#
## Bigger Images
#X = image2D(3,3)
#F = filter2D(2)
#out = regular_convolution(X,F)
#out_ = im2col_convolution(X,F)
#
##'naiveconv1'  1256.98 ms
##'regular_convolution'  1256.99 ms
##'im2col'  67.09 ms
##'im2col_inverse'  4.83 ms
##'im2col_convolution'  183.01 ms
#



############################################################
############################################################
############################################################




# Working in 3D Space I - Mutiple Cin - 1 Cout
# --------------------------------------------
''' Still 2D convolutions! Cin channels == Channels of each of the filters '''

def image3D(Cin,H,W): return np.random.randint(-10,10,size=(Cin,H,W))
def filter3D(Cout,Cin,K): return np.random.randint(-3,3,size=(Cout,Cin,K,K))

X = image3D(2,3,3)            ## Images    2x(3x3)
F = filter3D(2,X.shape[0],2)  ## 2xFilters 2x(2x2)
s = 1
p = 0


def num_windows(X,F):
    ''' Return the number of possible sliding windows
    of a filte over an 2D input '''
    H, W, K = X.shape[1], X.shape[2], F.shape[2]
    return (H-K+1)*(W-K+1)
    
num_windows(image3D(3,5,4), filter3D(2,3,3))

def output_dim3d(X,F,s=1,p=0):
    ''' Returns the output dimension of a convolution'''
    return int((X.shape[1] + 2*p - F.shape[2])/s + 1)


def conv2d(X,F):
    '''
    Convolves with input channels == kernel channels
    :Returns a scalar
    '''
#    mess = 'Conv2D requires input channels = kernel channels'
#    assert X.shape[0] == F.shape[0] can't assert because it i n
    K = F.shape[-1]
    Cin = F.shape[0]

    out = 0
    # From left to right in the kernel width
    for i in range(K):
        # From top to bottom in the kernel heigh
        for j in range(K):
            # From front to back in the kernel channels
            for k in range(Cin):
                out += X[k,i,j] * F[k,i,j]
    return out
        

#@timeit
def naiveconv3d(X, F, s=1, p=0):
    ''' Convolve 1 channel with 1 filter 
        Assume square images and kernels '''
    # Compute output dimensions
    O = int(output_dim3d(X,F,s,p))
    N = F.shape[0]
    K = F.shape[-1]
    FM = np.zeros((N,O,O)) 
    
    # For each of the filters
    for n in range(N):
        # Sliding filter horizontally
        for i in range(O):
            # Sliding filter vertically
            for j in range(O):
                 # Convolve the filter on that region of the image
                 FM[n,i,j] = conv2d(X[:, i:K+i, j:K+j], F[n])
    return FM


#@timeit
def im2col3d(X, F, s=1, p=0):
    ''' im2col '''
    I1, I2 = X.shape[1], X.shape[2]
    Cout, Cin, K = F.shape[0], F.shape[1], F.shape[2]
    O1 = K * K * Cout         # Window size * Filters
    O2 = num_windows(X,F)     # Num of possible windows
    Xcol = np.zeros((O1, O2))
  
    # For each input channel
    for n in range(Cin):
        col = 0  
        # Sliding left to right over the region
        for i in range(I1-K+1):
            # Sliding up to down over that region
            for j in range(I2-K+1):
                # Get the region where the kernel overlaps the input
                region = X[n, i:i+K, j:j+K]
                # Retrieve that region of the image as a columns
                Xcol[(n*K**2):(n+1)*K**2, col] = region.ravel()
                col += 1               
    return Xcol


def fil2col3d(F):
    ''' vectorize multple filters '''
    N, C, K = F.shape[0], F.shape[1], F.shape[2]
    O1, O2 = K**2*C, N
    Fvec = np.zeros((O1, O2))
    for n in range(N):
        for c in range(C):
            Fvec[(K**2*c):(K**2*(c+1)),n] = F[n,c].ravel()
    return Fvec


#@timeit
def im2col_inverse3d(x_, F, s=1, p=0):
    N, _, _ = F.shape[0], F.shape[1], F.shape[2]
    O = output_dim3d(X,F,s,p)
    out = np.zeros((N,O,O))
    
    #Populate per-channel
    for n in range(N):
        row = 0
        # Populate horizontally
        for i in range(O):
            # Populate vertically
            for j in range(O):
                out[n,i,j] = x_[row, n]
                row += 1
    return out


#out = naiveconv3d(X,F)
#Xcol = im2col3d(X,F)      
#Fvec = fil2col3d(F)
#out_ = Xcol.T @ fil2col3d(F)
#out_ = im2col_inverse3d(out_, F)
#(out == out_).all()
#'im2col'  0.16 ms
#'im2col_inverse'  0.05 ms

     
# Time comparison
#@timeit
def regular_convolution3d(X,F,s=1,p=0):
    return naiveconv3d(X,F,s,p)

#@timeit
def im2col_convolution3d(X,F,s=1,p=0):
    x_ = im2col3d(X,F).T @ fil2col3d(F)
    return im2col_inverse3d(x_,F,s,p)

#out = regular_convolution3d(X,F)
#out_ = im2col_convolution3d(X,F)

#'naiveconv3d'  0.66 ms
#'regular_convolution3d'  0.69 ms
#'im2col3d'  0.44 ms
#'im2col_inverse3d'  0.02 ms
#'im2col_convolution3d'  0.77 ms


# mmm interesting, at this level the price of im2col is very expensive!

    

# Bigger Images
X = image3D(2,50,50)              ## Images     3x(100x100)
F = filter3D(8,X.shape[0],5)       ## 2xFilters 16x3x(5x5)

Xcol = im2col3d(X,F)      
Fvec = fil2col3d(F)
out_ = Xcol.T @ fil2col3d(F)
out_ = im2col_inverse3d(out_, F)

out = regular_convolution3d(X,F)
out_ = im2col_convolution3d(X,F)

#'naiveconv1'  1256.98 ms
#'regular_convolution'  1256.99 ms
#'im2col'  67.09 ms
#'im2col_inverse'  4.83 ms
#'im2col_convolution'  183.01 ms




    