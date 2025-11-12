import torch

class gradient_inputs:
    def __init__(self, central_point):
        self.cp = central_point

    def space(self,r, density):
        def ul(p,radius):
            l = p-radius
            u = p+radius
            return l,u
        
        (xl, xu) = ul(self.cp[0],r)
        (yl, yu) = ul(self.cp[1],r)

        x = torch.linspace(xl, xu, density)
        y = torch.linspace(yl, yu, density)

        X, Y = torch.meshgrid((x,y), indexing= 'ij')
        mask = (torch.sqrt((X-self.cp[0])**2+(Y-self.cp[1])**2) < r)

        X = X[mask]
        Y = Y[mask]

        x = X.flatten()
        y = Y.flatten()
        t = torch.full_like(x, self.cp[2])
        xyt = torch.vstack((x, y, t)).T

        return xyt
    def space_time(self, r, density):
        def ul(p,radius):
            l = p-radius
            u = p+radius
            return l,u
        
        (xl, xu) = ul(self.cp[0],r)
        (yl, yu) = ul(self.cp[1],r)
        (tl, tu) = ul(self.cp[2],r)


        x = torch.linspace(xl, xu, density)
        y = torch.linspace(yl, yu, density)
        t = torch.linspace(tl, tu, density)


        X, Y, T = torch.meshgrid((x,y,t), indexing= 'ij')
        space_mask = (torch.sqrt((X-self.cp[0])**2+(Y-self.cp[1])**2) < r)
        time_mask = (torch.sqrt((T-self.cp[2])**2) < r)
        mask = space_mask & time_mask

        X = X[mask]
        Y = Y[mask]
        T = T[mask]

        x = X.flatten()
        y = Y.flatten()
        t = T.flatten()
        xyt = torch.vstack((x, y, t)).T
        return xyt
    
    def time(self, radius, density):
        def ul(p,radius):
            l = p-radius
            u = p+radius
            return l,u
        
        tl,tu = ul(self.cp[2], radius)
        t = torch.linspace(tl, tu, density)
        x = torch.full_like(t, self.cp[0])
        y = torch.full_like(t, self.cp[1])

        xyt = torch.vstack((x,y,t)).T
        return xyt


class circle_meth(gradient_inputs): 
    def __init__(self, central_point, net):
        super().__init__(central_point)
        self.net = net
        self.device = net.device()

    def space_grad(self,r, density):
        inputs = self.space(r, density)
        inputs = inputs.to(self.device)
        inputs.requires_grad_(True)
        outputs = self.net(inputs)

        grad_outputs = torch.ones_like(outputs)
        grads = torch.autograd.grad(outputs=outputs, inputs=inputs,
                                     grad_outputs=grad_outputs, create_graph=True)[0]
        return grads
    
    def time_grad(self, r, density):
        inputs = self.space(r, density)
        inputs = inputs.to(self.device)
        inputs.requires_grad_(True)
        outputs = self.net(inputs)

        grad_outputs = torch.ones_like(outputs)
        grads = torch.autograd.grad(outputs=outputs, inputs=inputs,
                                     grad_outputs=grad_outputs, create_graph=True)[0]
        return grads
    def spt_grad(self, r, density):
        inputs = self.space(r, density)
        inputs = inputs.to(self.device)
        inputs.requires_grad_(True)
        outputs = self.net(inputs)

        grad_outputs = torch.ones_like(outputs)
        grads = torch.autograd.grad(outputs=outputs, inputs=inputs,
                                     grad_outputs=grad_outputs, create_graph=True)[0]
        return grads        
    


class take_grad:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def grad(self):
        grad_outs = torch.ones_like(self.outputs)
        grad = torch.autograd.grad(outputs=self.outputs, inputs=self.inputs,
                                    grad_outputs=grad_outs, create_graph=True)[0]
        return(grad)
    
    def second_grad(self):

        grad_outs = torch.ones_like(self.outputs)
        first_grad = torch.autograd.grad(outputs=self.outputs, inputs=self.inputs,
                                         grad_outputs=grad_outs, create_graph=True)[0]
        

        du_dx = first_grad[:, 0:1]  
        du_dy = first_grad[:, 1:2]  
        

        d2u_dx2 = torch.autograd.grad(outputs=du_dx, inputs=self.inputs,
                                       grad_outputs=torch.ones_like(du_dx),
                                       create_graph=True, retain_graph=True)[0][:, 0:1]
        

        d2u_dy2 = torch.autograd.grad(outputs=du_dy, inputs=self.inputs,
                                       grad_outputs=torch.ones_like(du_dy),
                                       create_graph=True)[0][:, 1:2]
        

        laplacian = d2u_dx2 + d2u_dy2
        
        return laplacian