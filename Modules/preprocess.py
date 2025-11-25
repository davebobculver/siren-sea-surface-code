'Goal is to set up a much more streamlined training process'
'Break apart pre_pro into more professional code'

import torch
from Modules.beams import beam_trim
from torch.utils.data import DataLoader


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
        'this function can calls the dataloader forward'
        return self.dataset

    def get_dataloader(self,
                        pin_memory=True,
                        shuffle = True,
                        num_workers=1,
                        persistent_workers= True,
                        batch_size=1): 
        
        'this does the final step'
        with torch.no_grad():
            return DataLoader(self.dataset, pin_memory=pin_memory,
                            shuffle=shuffle,
                            num_workers=num_workers, 
                            persistent_workers=persistent_workers, 
                            batch_size=batch_size)     
        




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

