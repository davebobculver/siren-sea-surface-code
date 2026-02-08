'''Make a function which acepts in a real space sytem
- turns coords to NN coords using standard scaler
- evalutes coords through network
- handels backscaling of output
- returns output in real space

Considerations:
- All should be done on computation graph to incure ease of differentiablity
- Speed should be prioritized (no excess dataloader overhead)
- 
'''

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class Evaluator:
    def __init__(self, net, scalers, inputs):
        self.net = net.eval()
        self.in_scaler = scalers[0]
        self.out_scaler = scalers[1]
        self.inputs = inputs
        self.pi_half = torch.tensor(np.pi / 2, dtype=torch.float32).to(self.net.device())
    
    def scale_inputs(self):
        """Scale inputs using the input scaler"""
        if self.in_scaler is not None:
            if isinstance(self.in_scaler, StandardScaler):
                'Need to keep everything as a tensor so we can take gradients'
                'This is the old package which i am routing in. The else statement is way better'
                mean = torch.from_numpy(self.in_scaler.mean_).float().to(self.inputs.device)
                scale = torch.from_numpy(self.in_scaler.scale_).float().to(self.inputs.device)
                return ((self.inputs - mean) / scale) * self.pi_half
            else:
                return self.in_scaler.transform(self.inputs)
        return self.inputs
    
    def unscale_outputs(self, outputs):
        """Inverse transform outputs back to original scale"""
        if self.out_scaler is not None:
            if isinstance(self.out_scaler, StandardScaler):
                mean = torch.from_numpy(self.out_scaler.mean_).float().to(outputs.device)
                scale = torch.from_numpy(self.out_scaler.scale_).float().to(outputs.device)
                return (outputs/ self.pi_half) * scale + mean
            else:
                return self.out_scaler.inverse_transform(outputs)
        return outputs
    
    def net_eval(self):
        """Evaluate network with automatic scaling"""
        with torch.no_grad():
            scaled_inputs = self.scale_inputs()
            scaled_outputs = self.net(scaled_inputs)
            print(f"Raw network output range: {scaled_outputs.min():.3f} to {scaled_outputs.max():.3f}")
            
            outputs = self.unscale_outputs(scaled_outputs)
        return outputs
    
    def net_eval_with_grad(self):
        """Evaluate network while keeping gradients"""
        scaled_inputs = self.scale_inputs()
        scaled_outputs = self.net(scaled_inputs)

        outputs = self.unscale_outputs(scaled_outputs)
        return outputs
    


"_____ class to easily pick out data _____"

from Modules.beams import beam_trim

def prepare_tensors(data):
    """Convert beam data to coordinate tensors and move to device."""
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    z = data[:, 2].reshape(-1, 1)
    xy = torch.hstack((x, y))
    return xy, z

def temp_data(all_data, frame, beams, time_step = .1):
    'saves tensors on CPU. Need to move to GPU'
    space = beam_trim(all_data[frame], beams)

    xy, z = prepare_tensors(space)
    n = len(xy)
    t = torch.full((n,1), time_step*frame)

    xy_t = torch.hstack((xy,t))

    return xy_t, z

def in_out(all_data, beams, frames = range(10)):
    ins, outs = zip(*(temp_data(all_data, f, beams) for f in frames))
    return torch.cat(ins), torch.cat(outs)

def split(input, output, split=.8, mask =None):
    if mask != None:
        m = mask
    else:
        n = input.shape[0]
        m = torch.rand(n) < split
    ver_m = ~m

    train_input, train_output = [input[m], output[m]]
    ver_input, ver_output = [input[ver_m], output[ver_m]]

    return train_input, train_output , ver_input, ver_output, m


class pick_data:
    def __init__(self, data, beams, frames, masker = None, bin_width = .02, time_col =2):
        self.data = data
        self.beams = beams
        self.frames = frames
        self.inputs, self.outputs =  in_out(
            data, beams, frames=frames)
        
    def data_out(self, t):
        with torch.no_grad():
            delta = 1e-3
            mask = (self.inputs[:,2] < t+delta) & (self.inputs[:,2] > t-delta)
            raw_in = self.inputs[mask]
            raw_out = self.outputs[mask]
            return raw_in, raw_out

        
    def data_out_numpy(self, t):
        raw_in,   raw_out = self.data_out(t)
        return raw_in.numpy(), raw_out.numpy()