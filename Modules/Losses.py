import torch
import numpy as np

def create_grid(device, x_range=(-253, -132), y_range=(-178, -146), step=.5):
    """Create a denser grid for better GPU utilization."""
    x = torch.arange(x_range[0], x_range[1], step, device=device)
    y = torch.arange(y_range[0], y_range[1], step, device=device)
    X, Y = torch.meshgrid([x, y], indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    XY_grid = torch.vstack((X_flat, Y_flat)).T.requires_grad_(True).to(device)
    return XY_grid

xy_grid = create_grid('cpu')
def spt_reg(batch_x, net, lx = 0, ly = 3.4 , lt = 0):

    # x = batch_x[:, 0:1].clone().detach().requires_grad_(True)  # [n_points, 1]
    # y = batch_x[:, 1:2].clone().detach().requires_grad_(True)  # [n_points, 1]
    # t = batch_x[:, 2:3].clone().detach().requires_grad_(True)  # [n_points, 1]
    
    # coords = torch.cat([x, y, t], dim=1)  # [n_points, 3]
    # pred = net(coords)

    time = batch_x[0, 0, 2].item()
    XY_grid = xy_grid.to(batch_x.device)

    # Create individual components that require gradients
    x = XY_grid[:, 0:1].clone().requires_grad_(True)  # Remove .detach()
    y = XY_grid[:, 1:2].clone().requires_grad_(True)  # Remove .detach()
    t = torch.full((XY_grid.shape[0], 1), time, device=batch_x.device).requires_grad_(True)

    # Concatenate into grid - this grid maintains connections to x, y, t
    XYT_grid = torch.cat((x, y, t), dim=1)

    # Forward pass
    pred = net(XYT_grid)

    scalar_output = pred.sum()
    
    # grad_x = torch.autograd.grad(scalar_output, x, create_graph=False, allow_unused=True, retain_graph=True)[0]
    grad_y = torch.autograd.grad(scalar_output, y, create_graph=False, allow_unused=True, retain_graph=True)[0]
    # grad_t = torch.autograd.grad(scalar_output, t, create_graph=False, allow_unused=True, retain_graph=True)[0]


    # x_contrib = lx * torch.mean(grad_x**2)
    y_contrib = ly * torch.mean(grad_y**2)
    # t_contrib = lt * torch.mean(grad_t**2)
    
    total_spatial_reg = y_contrib

    # del x_contrib
    del y_contrib
    # del t_contrib

    return total_spatial_reg


def crit( MSE = False, dloss = False, sp_t = False,no_dx =False, params = None, net =None,
         XY_grid= None, time = None):
    mse = torch.nn.MSELoss()
    if MSE:
        def cust(output, true_y, *args):
            return mse(output,true_y)
        return cust
    elif dloss:
        u, h = params
        def close(output, true_y, fg, fg_y):
            return d_loss(output, true_y, fg, fg_y,u,h)
        return close
    elif sp_t:
        if not params == None:
            (lx,ly, lt) = params
            
            def closed(batch_x, batch_y):
                pred = net(batch_x)
                return spt_reg(batch_x,net,lx, ly, lt) + mse(pred, batch_y)
        else:
            def closed(batch_x, batch_y):
                return sp_t_redo(net) + mse(net(batch_x), batch_y)
        return closed
    elif no_dx:
        u, h = params
        def func(out,true_y, fg, fg_y):
            return No_dx(out, true_y, fg, fg_y, u, h)
        return func









'------------------------------------------------------------------------------------------------------------------------'
'Old code below this line. Lots of ideas that you can pull from'
"""
Here is some naming
output = net approximation where data is
true_y = collected data points

fg = where we evaluate the net outside of the data
fg_y = neural net where there is no data
"""

def d_loss(output, true_y, fg, fg_y, u, h):

    mse_loss = torch.nn.MSELoss()(output, true_y)

    grad_outputs = torch.ones_like(fg_y)
    fg_y = fg_y.requires_grad_(True)
    gradients = torch.autograd.grad(outputs = fg_y,
                               inputs = fg,
                               grad_outputs = grad_outputs,
                               create_graph = True)[0]
 
    derivative_x1 = gradients[:, 0]
    derivative_x2 = gradients[:, 1]

    derivative_term =derivative_x1.pow(2).mean() + h*derivative_x2.pow(2).mean()
    
    return mse_loss + u*derivative_term


def sp_t_redo(net, ly = 3.4, print_info = False, XY_grid = None, time = None):

    XYT_grid = torch.cat((XY_grid, torch.full((XY_grid.shape[0], 1), time).to('cuda')), dim=1)
    XYT_grid.requires_grad_(True)
    pred = net(XYT_grid).sum()

    grad_outputs = torch.ones_like(pred)
    grad = torch.autograd.grad(inputs= XYT_grid, outputs= pred, 
                                    grad_outputs= grad_outputs,
                                    create_graph= False,
                                    allow_unused=True)[0]
    if grad is None:
        if print_info:
            print("Warning: No gradients computed")
        return torch.tensor(0.0, requires_grad=True)

    grad_y = grad[:,2]

    out = ly*grad_y.pow(2).mean()
    return out


def No_dx(output,true_y, fg, fg_y, u, h):
    mse_loss = torch.nn.MSELoss()(output, true_y)

    grad_outputs = torch.ones_like(fg_y)
    fg_y = fg_y.requires_grad_(True)
    gradients = torch.autograd.grad(outputs = fg_y,
                               inputs = fg,
                               grad_outputs = grad_outputs,
                               create_graph = True)[0]

    derivative_x1 = gradients[:, 0]
    derivative_x2 = gradients[:, 1]

    derivative_term =u*derivative_x1.pow(2).mean() + h*derivative_x2.pow(2).mean()
    
    return mse_loss + derivative_term



"""Takes the objects made by covariance in 
Cov."""

# def c_loss(U,L, predictions_on_grid ,predicted_y, true_y, u):
#     mse = torch.nn.MSELoss()(predicted_y, true_y)

#     cov = L**(-1/2) (U.T @ predictions_on_grid)  # we should be left with a small vector
#     cov_loss = torch.norm(cov, p=2)
#     return mse + u*cov_loss

"""This is the RMS used for comparason"""

def rms(data,y_approx,std):
    return torch.sqrt(torch.sum(((data-y_approx)/std)**2)/len(data))

def continuity_loss(prior,current):
    diff = current.reshape(-1,1)- prior.detach().numpy().reshape(-1, 1)  # Reshape prior to match current's shape
    # Ensure diff is a numpy array for mean calculation
    if not isinstance(diff, np.ndarray):
        diff = np.array(diff)
    return torch.tensor(np.sqrt(np.mean(diff**2)), dtype=torch.float32, requires_grad= True)  # Mean squared error between prior and current prediction


# def sp_t_reg(batch_x, net, lx, ly, lt):
#     """
#     Fixed version with proper tensor handling
#     """
#     print(f"Input batch_x shape: {batch_x.shape}")
    
#     # Ensure we're working with the right dimensions
#     if batch_x.dim() == 3:  # If batch_x is [1, n_points, 3]
#         batch_x = batch_x.squeeze(0)  # Make it [n_points, 3]
    
#     n_points = batch_x.shape[0]
#     print(f"Processing {n_points} spatial points")
    
#     # Create input coordinates that require gradients - keep as 2D
#     x = batch_x[:, 0:1].clone().detach().requires_grad_(True)  # [n_points, 1]
#     y = batch_x[:, 1:2].clone().detach().requires_grad_(True)  # [n_points, 1]
#     t = batch_x[:, 2:3].clone().detach().requires_grad_(True)  # [n_points, 1]
    
#     # Combine into input tensor
#     coords = torch.cat([x, y, t], dim=1)  # [n_points, 3]
#     print(f"Coords shape: {coords.shape}")
    
#     # Get network predictions
#     predictions = net(coords)
#     print(f"Predictions shape: {predictions.shape}")
    
#     # For gradient computation, we need a scalar output
#     scalar_output = predictions.sum()
    
#     # Compute gradients
#     grad_x = torch.autograd.grad(scalar_output, x, create_graph=True, retain_graph=True)[0]
#     grad_y = torch.autograd.grad(scalar_output, y, create_graph=True, retain_graph=True)[0]
#     grad_t = torch.autograd.grad(scalar_output, t, create_graph=True, retain_graph=True)[0]
    
#     print(f"grad_x shape: {grad_x.shape}, mean_squared: {torch.mean(grad_x**2):.6f}")
#     print(f"grad_y shape: {grad_y.shape}, mean_squared: {torch.mean(grad_y**2):.6f}")
#     print(f"grad_t shape: {grad_t.shape}, mean_squared: {torch.mean(grad_t**2):.6f}")
    
#     # Compute spatial regularization
#     x_contrib = lx * torch.mean(grad_x**2)
#     y_contrib = ly * torch.mean(grad_y**2)
    
#     total_spatial_reg = x_contrib + y_contrib
#     print(f"Spatial regularization: {total_spatial_reg:.6f}")

#     return total_spatial_reg

