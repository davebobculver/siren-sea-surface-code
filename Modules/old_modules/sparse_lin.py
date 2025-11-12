import scipy as sp
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, eye, csr_matrix, vstack
from scipy.sparse.linalg import lsqr



class sparse:
    """This is a sparse linear algebra method for solving a least squares problem
    and generatign a surface apporximation.
    x and y are the grid. 
    xs and ys are the points where the surface is evaluated. 
    s is the standard deviation. 
    zs is the z values at the points.
    """

    def __init__(self, x, y , xs, ys, zs, TargetRMS, mu, s, Bias ):
        
        X,Y = torch.meshgrid(x,y, indexing='ij')
        vecX  = X.reshape(-1,1)
        vecY  = Y.reshape(-1,1)
        
        points = torch.vstack((xs, ys)).T.numpy()
        grid = torch.hstack((vecX, vecY)).numpy()       
        ds = ((zs/s).reshape(-1)).numpy()
        
        nx = len(x)
        ny = len(y)
        n = nx * ny
        nGrid = len(vecX)

        tree = sp.spatial.cKDTree(grid)
        k= tree.query(points, k=1)[1]
        
        nGrid = len(vecX)
        nPts = len(k)

        rows = np.arange(nPts)
        cols = k
        data = np.full(nPts, 1/s)

        A = csr_matrix((data, (rows, cols)), shape=(nPts, nGrid))

        """Finite Difference Matrix in X and Y direction"""

        # Dimensions
        n = nx * ny

        # Dx equivalent
        e = np.ones(n - 1)
        Dx_lower = spdiags(e, -1, n, n).transpose()  # equivalent to spdiags(...,-1,n,n)' in MATLAB
        Dx = Dx_lower - eye(n, format='csr')
        Dx[n - 1, n - 1] = 0  

        # Dy construction
        neg_diag = eye(n, format='csr')  # Identity matrix
        rows = np.arange(n - nx)
        cols = rows + nx
        data = np.ones(n - nx)
        pos_diag = csr_matrix((data, (rows, cols)), shape=(n, n))

        # Extract only the first nx*(ny-1) rows
        keep_rows = nx * (ny - 1)
        Dy = -neg_diag[:keep_rows, :] + pos_diag[:keep_rows, :]

        # Pad Dy with zeros to match size (append nx rows of zeros)
        zero_pad = csr_matrix((nx, n))
        Dy = vstack([Dy, zero_pad])

        ### Now we solve the least squares problem
        bigd = np.concatenate([ds, np.zeros(n), np.zeros(n)])
        RMS = np.zeros(len(mu))
        for kk in range(len(mu)):
            CurrentMu = mu[-(kk + 1)]
            
            # Build bigA
            sqrt_mu = np.sqrt(CurrentMu)
            
            bigA = vstack([
                A,
                Dx.multiply(sqrt_mu),
                Dy.multiply(Bias*sqrt_mu)
            ])
            
            # Solve least squares problem
            result = lsqr(bigA, bigd)
            zvec = result[0]
            
            # Predict and compute RMS error
            pred_ds = A @ zvec
            RMS[kk] = np.sqrt(np.mean((pred_ds - ds)**2))
            
            if RMS[kk] < TargetRMS:
                break
            elif kk > 0 and abs(RMS[kk] - RMS[kk-1]) / RMS[kk] < 0.01:
                break
        
        self.s = s
        self.vecX = vecX
        self.vecY = vecY
        self.zvec = zvec #normalized
        self.Z = pred_ds*s  # non-normalized
        self.RMS = np.sqrt(np.mean((pred_ds - ds)**2)) #normalized

    def verify(self, xv, yv, zv): 
        
        points = torch.vstack((xv, yv)).T.numpy()
        grid = torch.hstack((self.vecX, self.vecY)).numpy()       
        #ds = ((zv / self.s).reshape(-1)).numpy()

        tree = sp.spatial.cKDTree(grid)
        k= tree.query(points, k=1)[1]
        nGrid = len(self.vecX)
        nPts = len(k)
        
        rows = np.arange(nPts)
        cols = k
        data = np.full(nPts, 1/self.s)
        A = csr_matrix((data, (rows, cols)), shape=(nPts, nGrid))

        err = (A @ self.zvec)*self.s - zv.numpy()
        RMSE = np.sqrt(np.mean((err)**2)/self.s**2)
        return RMSE
    
''' this is trouble, because it looks good. I cannot see the diff.'''
