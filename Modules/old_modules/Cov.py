"""Here we are constructing an object that 
forms covariance matricies of a workable size 
using a low rank approximation"""

import torch

def GetEffDim(E , tol):
    go = 1
    kk = 0
    while go == 1:
        kk = kk+1
        if (torch.sum(torch.abs(E[:kk])**2))/(torch.sum(torch.abs(E**2)))> (1-tol):
            go = 0
    return kk

class covariance:
    def __init__(self, x, y , lx, ly, tol):
        
        """Defining x covariance"""
        n =  len(x)
        Cx = torch.zeros((n,n))
        for i in range(Cx.shape[0]):
            for j in range(Cx.shape[1]):
                Cx[i,j] = torch.exp(-((x[i]-x[j])**2)/lx)

        Cx = Cx + Cx.T -torch.diag(torch.diag(Cx))

        Lx, Ux = torch.linalg.eigh(Cx)  # Safer for symmetric matrices
        i = torch.argsort(Lx, descending=True)  # Sort indices in descending order


        Lx = Lx[i]
        Ux = Ux[:, i]

        r =GetEffDim(Lx, tol )

        Lxa = Lx[:r]
        Uxa = Ux[:,:r]


        """This is for the Y"""

        Cy = torch.zeros((n,n))

        for i in range(Cy.shape[0]):
            for j in range(Cy.shape[1]):
                Cy[i,j] = torch.exp(-((y[i]-y[j])**2)/ly)

        Cy = Cy + Cy.T -torch.diag(torch.diag(Cy))

        Ly, Uy = torch.linalg.eigh(Cy)
        i = torch.argsort(Ly, descending= True)
        Ly = Ly[i]
        Uy = Uy[:, i]

        q =GetEffDim(Ly, tol )

        Lya = Ly[:q]
        Uya = Uy[:,:q]

        """Now we Combine"""
        

        L = torch.kron(Lxa, Lya)
        U = torch.kron(Uxa, Uya)

        i = torch.argsort(L, descending= True)
        L = L[i]                
        U = U[:, i]

        self.K = GetEffDim(L, tol)
        self.L = L[:self.K]
        self.L = self.L.view(-1,1)
        self.U = U[:,:self.K]
        
        
    def Covariant_Loss(self, y_approx):
        """Define the loss Matrix"""


        C_loss = (torch.tensor((self.U.T), dtype = torch.float32) @ y_approx)*torch.tensor((torch.sqrt(1/self.L)).reshape(-1,1), dtype = torch.float32)
        C_loss = torch.norm(C_loss, p =2)
        return C_loss



    


    

