'Goal is to set up a much more streamlined training process'
'Break apart pre_pro into more professional code'

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler as MM
from Modules.beams import beam_trim ## Need to put back to Modules.beams after finished with work
from torch.utils.data import DataLoader
import numpy as np



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

class SS():
    "PyTorch version of standard scaler for SIREN networks. Scales data to trainable range."
    "Give it a new std"
    def __init__(self, mean=None, std=None, new_std = 1.0):
        self.mean = mean
        self.std = std
        self.new_std = new_std

    def fit_transform(self, data):
        """Fit and transform data to zero mean, unit variance"""
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        if self.mean is None:
            self.mean = data.mean(dim=0)
        if self.std is None:
            self.std = data.std(dim=0)
        return ((data - self.mean) / (self.std + 1e-8))*self.new_std

    def transform(self, data):
        """Transform data using fitted mean and std"""
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        return ((data - self.mean) / (self.std + 1e-8))*self.new_std

    def inverse_transform(self, data):
        """Reverse the scaling transformation"""
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        return (data/self.new_std) * self.std + self.mean


class data_gen:
    'This class will be the the set up into making the data loader from the data set'
    'The inputs are coming from beams'
    def __init__(self, data, beams, frames, splits = .8,masker = None, bin_width = .02, time_col =2, normalize = False, w_0 =1):
        self.data = data
        self.beams = beams
        self.frames = frames
        self.w_0 = w_0
        self.inputs, self.outputs =  in_out(
            data, beams, frames=frames)
        
        """If we are smart here we can do all our work just 
        making the training and ver data loader here"""

        'Take an 80/20 split of inputs and outputs.'

        self.scalers = None
        if normalize == 'SS':
            sc_in = SS(new_std=torch.pi/(2*self.w_0))
            self.inputs = sc_in.fit_transform(self.inputs)
            sc_out = SS(new_std=torch.pi/(2*self.w_0))
            self.outputs = sc_out.fit_transform(self.outputs)
            self.scalers = [sc_in, sc_out]
    

        if normalize == 'minmax':
            sc_in = MM(feature_range=(-np.pi/2,np.pi/2))
            self.inputs = torch.tensor(sc_in.fit_transform(self.inputs), dtype=torch.float32)

            sc_out = MM(feature_range=(-np.pi/2,np.pi/2))
            self.outputs = torch.tensor(sc_out.fit_transform(self.outputs), dtype=torch.float32)
            
            self.scalers = [sc_in, sc_out]

        if isinstance(normalize, list):
            'only do this for verfication data'
            sc_in = normalize[0]
            sc_out = normalize[1]
    
            if isinstance(sc_in, StandardScaler):
                print('we are in the Standard Scaler')
                self.inputs =  torch.tensor(sc_in.transform(self.inputs), dtype=torch.float32)*(torch.pi/2)
                self.outputs = torch.tensor(sc_out.transform(self.outputs), dtype=torch.float32)*(torch.pi/2)

            if isinstance(sc_in, SS):
                print('we are in the siren section')
                self.inputs = sc_in.transform(self.inputs)
                self.outputs = sc_out.transform(self.outputs)
            
            self.scalers = [sc_in, sc_out]

        train_input, train_output, ver_input, ver_output,m = split(self.inputs,
                                                                self.outputs,
                                                                split = splits,
                                                                mask =masker)
        self.m = m

        self.train_dataset = FrameDataset(train_input, train_output,
                                    bin_width= bin_width,
                                    time_col=time_col)
        
        self.ver_dataset = FrameDataset(ver_input, ver_output,
                            bin_width= bin_width,
                            time_col=time_col)

    def get_mask(self):
        "return mask for repeatability"
        return self.m
    def get_scalers(self):
        'this function can return the scalers used in normalization'
        'Useful for inverse transforming later'
        return self.scalers

    def get_datasets(self):
        'this function can calls the dataloader forward'
        return self.train_dataset, self.ver_dataset

    def get_dataloaders(self,
                        pin_memory=True,
                        shuffle = True,
                        num_workers=1,
                        persistent_workers= True,
                        batch_size=1): 
        
        'this does the final step'
        with torch.no_grad():
            train_dataloader = DataLoader(self.train_dataset, pin_memory=pin_memory,
                            shuffle=shuffle,
                            num_workers=num_workers, 
                            persistent_workers=persistent_workers, 
                            batch_size=batch_size)
            
            ver_dataloader = DataLoader(self.ver_dataset, pin_memory=pin_memory,
                            shuffle=shuffle,
                            num_workers=num_workers, 
                            persistent_workers=persistent_workers, 
                            batch_size=batch_size) 

            return train_dataloader, ver_dataloader    
        




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







    # def SPT_debugging(self,max_num_epochs,
    #                     prints = False, saving = False, save_path =None, lx = 0, ly = 0 , lt = 0,
    #                     ver_dataloader = None, stopping_crit=None):
    #     mse = torch.nn.MSELoss()
    #     reg_losses = []
    #     mse_losses = []
    #     total_losses = []
    #     mse_ver = []
    #     stopping = torch.zeros(max_num_epochs)
    #     for epoch in range(max_num_epochs):
    #         num_batches = 0
    #         reg_epoch, mse_epoch, total_epoch = 0.0, 0.0, 0.0
    #         for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):   # DataLoader gives you batches
    #             # Move to GPU
    #             batch_x = batch_x.to(self.device)
    #             batch_y = batch_y.to(self.device)


    #             self.optimizer.zero_grad(set_to_none = True)

    #             loss1 = Losses.spt_reg(batch_x, net=self.net, lx = lx,ly =ly, lt =lt)
    #             loss2 = mse(self.net(batch_x), batch_y)
    #             loss = loss1 + loss2

    #             loss.backward()
    #             self.optimizer.step()
                
    #             if prints:
    #                 if batch_idx % 10== 0:
    #                     print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

    #             torch.cuda.empty_cache()
    #             gc.collect()
                
    #             num_batches += 1
        
    #         reg_epoch, mse_epoch, total_epoch = [0.0,0.0,0.0]
    #         for (batch_x, batch_y) in self.dataloader:
    #             batch_x = batch_x.to(self.device)
    #             batch_y = batch_y.to(self.device)

    #             reg_batch = Losses.spt_reg(batch_x, net=self.net, lx=lx, ly=ly, lt=lt).item()
    #             mse_batch = mse(self.net(batch_x), batch_y).item()

    #             reg_epoch += reg_batch
    #             mse_epoch += mse_batch
    #             total_epoch += (reg_batch + mse_batch)
        

    #         reg_losses.append(reg_epoch/num_batches)
    #         mse_losses.append(mse_epoch/num_batches)
    #         total_losses.append(total_epoch/num_batches)


    #         if ver_dataloader is not None:
    #             mse_ver_epoch = 0.0
    #             for (batch_x, batch_y) in ver_dataloader:
    #                 batch_x = batch_x.to(self.device)
    #                 batch_y = batch_y.to(self.device)
    #                 mse_ver_epoch += mse(self.net(batch_x), batch_y).item()
    #             mse_ver_epoch /= len(ver_dataloader)
    #             mse_ver.append(mse_ver_epoch)
                
    #         torch.cuda.empty_cache()
    #         if prints:
    #             print(f'Epoch {epoch} Complete - Avg Loss: {total_epoch:.6f}')
    #         if saving:
    #             torch.save({'model_state_dict': self.net.state_dict(),
    #             'optimizer_state_dict': self.optimizer.state_dict(),
    #             'epoch': epoch,
    #             'model config': self.model_config,
    #         }, save_path)
                
    #         stopping[epoch] = mse_ver_epoch
    #         'this method can totally be changed. When I rea make this class outsource this to another function for debugging.'
    #         if epoch> 40:
    #             moving_window = stopping[epoch-20:epoch]
    #             moving_window_avg = moving_window.mean()
    #             if stopping[epoch]- moving_window_avg > stopping_crit:
    #                 print('stopped_training at epoch: {epoch}')
    #                 break
            
                
    #     losses_dict = {
    #         'reg_losses': reg_losses,
    #         'mse_losses': mse_losses,
    #         'total_losses': total_losses}
        
    #     if ver_dataloader is not None:
    #         losses_dict['mse_ver'] = mse_ver

        
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     return losses_dict

