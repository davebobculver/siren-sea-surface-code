import torch
import numpy as np
import math


'BIG ISSUES IN HERE!!!!!!!'
'NEED TO TRIPTLE CHECK INTIALIZATION'

class Net(torch.nn.Module):
    def __init__(self, act, input_size,  *args, w_0 =1):
        super(Net, self).__init__()
        self.act = act 
        self.w_0 = w_0
        self.hidden_layers = torch.nn.ModuleList()
        for layer_size in args:
            self.hidden_layers.append(torch.nn.Linear(input_size, layer_size))
            input_size = layer_size

        # Output layer
        self.output_layer = torch.nn.Linear(input_size, 1)

        for i, m in enumerate(self.hidden_layers):
            self.initialize_weights(m, i)
        print(self.hidden_layers)
    
    def initialize_weights(self, m, i):
        if isinstance(m, torch.nn.Linear):
            if self.act == torch.relu:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif self.act == torch.sin or self.act == torch.cos:
                bound = math.sqrt(6 / m.in_features)
                if i ==0:
                    torch.nn.init.uniform_(m.weight, -bound, bound)
                else:
                    torch.nn.init.uniform_(m.weight, bound/(-self.w_0), bound/(self.w_0))
                m.bias.data.fill_(0.0)
            else:
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.1)
        
        if self.act == torch.relu:
            torch.nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        else:
            torch.nn.init.xavier_normal_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.01)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.act(self.w_0*layer(x))
        x = self.output_layer(x)
        return x

    def device(self):
        return next(self.parameters()).device

        
# w_0 =1
# # model_config = model['model config']
# model_config = (3, 280, 550,1000, 100)
# net = Net(torch.sin, *model_config, w_0 = w_0).to('cuda')
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


"""Below is a function used to evaluate an optimized nueral network
and make the outputs ready to be plotted."""

def net_eval(net, inputs):
    out = net(inputs)
    return out.detach()

class NetWrapper:
    """
    Wrapper class to make a 3D temporal neural network compatible with 2D plotting functions.
    Takes 2D (x,y) input, adds time dimension, and passes through 3D network.
    """
    def __init__(self, temporal_net, frame =302):
        self.temporal_net = temporal_net
        self.fixed_time = frame/10
    
    def __call__(self, xy):
        """
        Make the wrapper callable like the original network.
        
        Args:
            xy: 2D tensor of shape (N, 2) with x,y coordinates
            
        Returns:
            Predictions from the temporal network at fixed time
        """
        with torch.no_grad():
            # Add time dimension
            time_tensor = torch.full((xy.shape[0], 1), self.fixed_time, device=xy.device)
            xyt = torch.cat([xy, time_tensor], dim=1)
            return self.temporal_net(xyt)
    
    def parameters(self):
        """Return parameters for compatibility with training functions."""
        return self.temporal_net.parameters()
    
    def to(self, device):
        """Move network to device."""
        self.temporal_net = self.temporal_net.to(device)
        return self
    
    def eval(self):
        """Set network to evaluation mode."""
        self.temporal_net.eval()
        return self