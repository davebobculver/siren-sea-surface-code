'Goal is to set up a much more streamlined training process'

import torch
from Modules.beams import beam_trim


def prepare_tensors(data):
    """Convert beam data to coordinate tensors and move to device."""
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    z = data[:, 2].reshape(-1, 1)
    xy = torch.hstack((x, y)).requires_grad_(True)
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

class data_gen:
    'This class will be the the set up into making the data loader from the data set'
    'The inputs are coming from beams'
    def __init__(self, data, beams, frames, bin_width = .02, time_col =2):
        self.data = data
        self.beams = beams
        self.frames = frames
        self.inputs, self.outputs =  in_out(
            data, beams, frames=frames)

        self.dataset = FrameDataset(self.inputs, self.outputs,
                                    bin_width= bin_width,
                                    time_col=time_col) 

def get_dataset(self):
    return self.dataset      
        




'-----------------------------------------'
'frame dataset below this line'

class FrameDataset(torch.utils.data.Dataset):
    """
    Groups time-stamped data into bins and returns
    all points belonging to a given time bin.
    """

    def __init__(self, inputs, outputs, bin_width=0.01, time_col=2):
        self.inputs = inputs
        self.outputs = outputs
        self.bin_width = bin_width

        # Extract timestamps
        t = inputs[:, time_col]

        # Build bin indices
        t0 = t.min()
        self.bin_indices = torch.floor((t - t0) / bin_width).long()

        # Unique bins in sorted order
        self.unique_bins = torch.unique(self.bin_indices)

        # Group every index by its bin
        self.groups = [
            torch.where(self.bin_indices == b)[0]
            for b in self.unique_bins
        ]

        # Store bin_start times for reference
        self.bin_starts = self.unique_bins * bin_width + t0

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        batch_idx = self.groups[idx]
        return self.inputs[batch_idx], self.outputs[batch_idx]

    def get_bin_info(self, idx):
        """Metadata for one bin."""
        start = self.bin_starts[idx].item()
        return {
            "bin_start": start,
            "bin_end": start + self.bin_width,
            "num_points": len(self.groups[idx]),
        }

    def print_bin_summary(self, n=10):
        n = min(n, len(self))
        print(f"\nShowing {n} bins:")
        for i in range(n):
            info = self.get_bin_info(i)
            print(f"  Bin {i}: [{info['bin_start']:.3f}, {info['bin_end']:.3f}) "
                  f"- {info['num_points']} points")


