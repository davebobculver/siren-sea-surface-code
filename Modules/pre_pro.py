import torch
from Modules.beams import beam_trim
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import numpy as np
import os
import pickle
from scipy.stats import binned_statistic
from torch.utils.data import TensorDataset, DataLoader
from Modules.Neural_Net import Net
from Modules import Losses
import gc




def rmse(z1, z2, normalization=0.06):
    """Calculate Root Mean Square Error with normalization."""
    return torch.sqrt(torch.mean(((z1 - z2) / normalization)**2)).item()

def setup_data(all_data, frame=302):
    """Organize data into verification, training, and testing sets."""
    # Verification data (beam 3)
    beams_verification = [30]
    ver_data = beam_trim(all_data[frame], beams_verification)
    
    # Training data (beams 4-28, 1, 2, 31)
    beams_training = list(range(4, 29)) + [1, 2, 31]
    train_data = beam_trim(all_data[frame], beams_training)
    
    # Testing data (beams 3, 29, 30)
    beams_testing = [3, 29, 30]
    test_data = beam_trim(all_data[frame], beams_testing)
    
    # All data for visualization
    beams_all = list(range(32))
    all_beam_data = beam_trim(all_data[frame], beams_all)

    cen__beam_train = [11]
    cen_data_train = beam_trim(all_data[frame], cen__beam_train)
    
    cen__beam_ver = [10]
    cen_data_ver = beam_trim(all_data[frame], cen__beam_train)

    cen_data = [cen_data_train,cen_data_ver ]    
    return ver_data, train_data, test_data, all_beam_data, cen_data

def prepare_tensors(data, device):
    """Convert beam data to coordinate tensors and move to device."""
    x = data[:, 0].reshape(-1, 1).to(device)
    y = data[:, 1].reshape(-1, 1).to(device)
    z = data[:, 2].reshape(-1, 1).to(device)
    xy = torch.hstack((x, y)).requires_grad_(True)
    return xy, z

def create_grid(device, x_range=(-253, -132), y_range=(-178, -146), step=.5):
    """Create a denser grid for better GPU utilization."""
    x = torch.arange(x_range[0], x_range[1], step, device=device)
    y = torch.arange(y_range[0], y_range[1], step, device=device)
    X, Y = torch.meshgrid([x, y], indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    XY_grid = torch.vstack((X_flat, Y_flat)).T.requires_grad_(True).to(device)
    return XY_grid, X, Y

def plots(ver_xy, ver_z, grid_xy, X, Y, all_beam_data, net, 
                                train_xy, train_z, test_xy, test_z, save_path, cen_xy, cen_z,
                                frame, show = True):

    # Check device locations
    # net_device = next(net.parameters()).device

    # Move everything to network device
    device = 'cpu'
    ver_xy = ver_xy.to(device)
    ver_z = ver_z.to(device)
    grid_xy = grid_xy.to(device)
    X = X.to(device)
    Y = Y.to(device)

    ver_pred = net(ver_xy).detach().cpu().numpy().reshape(-1)
    grid_pred = net(grid_xy).detach().cpu().numpy().reshape(X.shape[0], X.shape[1])
    
    # Convert data to numpy
    ver_x = ver_xy[:, 0].detach().cpu().numpy()
    ver_z_np = ver_z.detach().cpu().numpy().reshape(-1)
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    
    all_x = all_beam_data[:, 0].detach().cpu().numpy()
    all_y = all_beam_data[:, 1].detach().cpu().numpy()
    all_z = all_beam_data[:, 2].detach().cpu().numpy()
    
    # Calculate metrics
    train_rmse = rmse(train_z, net(train_xy))
    test_rmse = rmse(test_z, net(test_xy))
    mean_wave_height = np.mean(grid_pred)

    
    # Create visualization
    plt.rcParams.update({
        'font.weight': 'bold',
        'font.size': 11,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8))
    
    # Top plot: Verification comparison
    sort_idx = np.argsort(ver_x)
    ax1.grid(True, alpha=0.3, zorder=0)
    scatter1 = ax1.scatter(ver_x[sort_idx], ver_z_np[sort_idx], 
                          c=ver_z_np[sort_idx], cmap=cmc.davos, 
                          s=40, alpha=0.8, label='LiDAR Data', 
                          zorder=2, vmin=0, vmax=6)
    ax1.plot(ver_x[sort_idx], ver_pred[sort_idx], 
            color=cmc.davos(0.6), linewidth=3, 
            label='NN Prediction', zorder=3)
    ax1.set_ylabel(r'$\eta$  (m)', labelpad= 28)
    ax1.set_title(f'Outer')
    ax1.set_xlim(-253, -132)
    ax1.set_ylim(0, 6)
    # ax1.legend(loc='upper right')
    
    # Middle plot: Neural network prediction grid
    im2 = ax2.pcolormesh(X_np, Y_np, grid_pred, cmap=cmc.davos, 
                        shading='nearest', vmin=0, vmax=6)
    ax2.set_ylabel('Along-Shore (m)', labelpad = 7)
    ax2.set_title(f'Neural Network')
    ax2.set_xlim(-253, -132)
    ax2.set_ylim(-178, -146)
    
    # Bottom plot: All LiDAR data
    scatter3 = ax3.scatter(all_x, all_y, c=all_z, cmap=cmc.davos, 
                          s=20, alpha=0.7, vmin=0, vmax=6)
    ax3.set_xlabel('Cross-Shore (m)')
    ax3.set_ylabel('Along-Shore (m)', labelpad = 7)
    ax3.set_title('LiDAR')
    ax3.set_xlim(-253, -132)
    ax3.set_ylim(-178, -146)
    
    # Add shared colorbar
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.85, top=0.95, hspace=0.4)
    cbar_ax = fig.add_axes([0.87, 0.08, 0.03, 0.87])
    cbar = plt.colorbar(scatter3, cax=cbar_ax)
    cbar.set_label(r'Sea Surface Elevation $\eta$  (m)', rotation=270, labelpad=20)
    
    save = os.path.join(save_path, f"main/frame_{frame}.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)

    plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    'central vs verification plots. '

    fig , axs = plt.subplots(2,1, figsize = (12,6), sharex=True, sharey=True)

    # Plot the verification data
    axs[0].scatter(ver_x, ver_z_np, 
                c=ver_z_np, s=50, cmap=cmc.davos, 
                label='LiDAR', vmin=0, vmax=6, edgecolor='white', alpha=.7)
    axs[0].set_title('Outer Beam', fontsize=14)
    axs[0].set_ylabel('Surface Elevation (m)', fontsize=12)
    axs[0].set_ylim(0, 6)
    axs[0].set_xlim(-253, -132)
    axs[0].set_xlabel('Cross-Shore (m)', fontsize=12)
    # axs[0].legend(loc='upper right', fontsize=12)
    axs[0].grid(True)

    axs[0].plot(ver_x, ver_pred, 
                color=cmc.davos(.8), linewidth=3, label='NN Prediction')


    # Plot the neural network predictions
    scatter = axs[1].scatter(cen_xy[:,0].cpu().detach().numpy(), cen_z.cpu().detach().numpy(), 
                c=net(cen_xy).cpu().detach().numpy(), s=50, cmap=cmc.davos, 
                label='NN Prediction', vmin=0, vmax=6, edgecolor='white', alpha=.7)
    axs[1].set_title('Central Beam', fontsize=14)
    axs[1].set_xlabel('Cross-Shore (m)', fontsize=12)
    axs[1].set_ylabel('Surface Elevation (m)', fontsize=12)
    axs[1].set_ylim(0, 6)
    axs[1].set_xlabel('Cross-Shore (m)', fontsize=12)
    # axs[1].legend(loc='upper right', fontsize=12)
    axs[1].grid(True)
    axs[1].set_xlim(-253, -132)

    axs[1].plot(cen_xy[:,0].cpu().detach().numpy(), net(cen_xy).cpu().detach().numpy(), 
                color=cmc.davos(.8), linewidth=3, label='NN Prediction')
    # axs[1].legend(loc='upper right', fontsize=12)
    # Add colorbar

    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.85, top=0.95, hspace=0.4)
    cbar_ax = fig.add_axes([0.87, 0.08, 0.03, 0.87])
    cbar = plt.colorbar(scatter, cax=cbar_ax)

    save = os.path.join(save_path, f"cen_ver/frame_{frame}.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()



    return train_rmse, test_rmse, mean_wave_height


def train_neural_network(net, train_xy, train_z, grid_xy, optimizer, criterion_func, frame, 
                        device, max_iterations=1000, initial_criterion=0.016, printing = False):
    """Train the neural network with adaptive convergence criteria and GPU monitoring."""
    
    # Ensure all inputs are on the correct device
    device = next(net.parameters()).device  # Get device from model
    train_xy = train_xy.to(device)
    train_z = train_z.to(device)
    grid_xy = grid_xy.to(device)
    
    loss = 1
    iteration = 0
    criterion = initial_criterion
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    while loss > criterion and iteration < max_iterations:
        iteration += 1
        optimizer.zero_grad()
        
        # Network outputs are automatically on same device as network
        train_pred = net(train_xy)
        grid_pred = net(grid_xy)
        
        loss_value = criterion_func(train_pred, train_z, grid_xy, grid_pred)
        loss_value.backward()
        optimizer.step()
        
        loss = loss_value.item()

        if iteration > 300:
           criterion +=.0001

    if printing:
        print(f'Training finished for frame: {frame} on iteration: {iteration} with crit:{criterion}')
    return iteration

def preformance(grid_xy,all_data_xy, all_data_z, net, 
                                train_xy, train_z, test_xy, test_z,bins = 240, save_path=None):
    test_rms = rmse(test_z, net(test_xy))
    train_rms = rmse(train_z, net(train_xy))
    grid_pred = net(grid_xy)
    mean_z = torch.mean(grid_pred).item()

    bin_result = binned_statistic(
        grid_xy[:, 0].detach().cpu().numpy().flatten(),
        grid_pred.detach().cpu().numpy().flatten(),
        bins=bins
    )
    nn_bin = torch.tensor(bin_result[0].astype(np.float32)).half()

    x = all_data_xy[:,0].detach().cpu().numpy().flatten()
    z = all_data_z.detach().cpu().numpy().flatten()

    binned_res = binned_statistic(x,z,bins = bins)
    data_bin = torch.tensor(binned_res[0]).half()

    return test_rms, train_rms, mean_z, nn_bin, data_bin

def saving(path, test_rms, train_rms, mean_z, nn_bin, data_bin, frame):
    if path == 'MSE':
        with open(f'../data/mse/test_rms.pkl', 'wb') as f:
            pickle.dump(test_rms, f)
        with open(f'../data/mse/train_rms.pkl', 'wb') as f:
            pickle.dump(train_rms, f)
        with open(f'../data/mse/mean_z.pkl', 'wb') as f:
            pickle.dump(mean_z, f)
        with open(f'../data/mse/nn_bin.pkl', 'wb') as f:
            pickle.dump(nn_bin, f)
        with open(f'../data/mse/data_bin.pkl', 'wb') as f:
            pickle.dump(data_bin, f)
        with open(f'../data/mse/frame.pkl', 'wb') as f:
            pickle.dump(frame, f)            
    elif path == 'dloss':
        with open(f'../data/dloss/test_rms.pkl', 'wb') as f:
            pickle.dump(test_rms, f)
        with open(f'../data/dloss/train_rms.pkl', 'wb') as f:
            pickle.dump(train_rms, f)
        with open(f'../data/dloss/mean_z.pkl', 'wb') as f:
            pickle.dump(mean_z, f)
        with open(f'../data/dloss/nn_bin.pkl', 'wb') as f:
            pickle.dump(nn_bin, f)
        with open(f'../data/dloss/data_bin.pkl', 'wb') as f:
            pickle.dump(data_bin, f)
        with open(f'../data/mse/frame.pkl', 'wb') as f:
            pickle.dump(frame, f)            
    elif path == 'dy':
        with open(f'../data/dy/test_rms.pkl', 'wb') as f:
            pickle.dump(test_rms, f)
        with open(f'../data/dy/train_rms.pkl', 'wb') as f:
            pickle.dump(train_rms, f)
        with open(f'../data/dy/mean_z.pkl', 'wb') as f:
            pickle.dump(mean_z, f)
        with open(f'../data/dy/nn_bin.pkl', 'wb') as f:
            pickle.dump(nn_bin, f)
        with open(f'../data/dy/data_bin.pkl', 'wb') as f:
            pickle.dump(data_bin, f)
        with open(f'../data/mse/frame.pkl', 'wb') as f:
            pickle.dump(frame, f)            

def sparse():
    pass



'''  --- After here we begin use this module for the temporal structure and batching ---'''

def temp_data(all_data, frame, beams, time_step = .1):
    'saves tensors on CPU. Need to move to GPU'

    space = beam_trim(all_data[frame], beams)

    xy, z = prepare_tensors(space, 'cpu')
    n = len(xy)
    t = (torch.ones(n)*time_step*frame).reshape(-1,1)
    xy_t = torch.hstack((xy,t))

    return xy_t, z

def in_out(all_data, beams, frames = range(10)):
    input = []
    output = []
    for frame in frames:
        ins, out = temp_data(all_data, frame, beams, time_step = .1)
        input.append(ins)
        output.append(out)

    input = torch.cat(input)
    output = torch.cat(output)
    return input, output

def batching(inputs, outputs, batch_size = 16000, workers = 0):
    dataset = TensorDataset(inputs.cpu(),outputs.cpu())
    dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True,            
    num_workers=workers,     
    pin_memory=False,     
    drop_last=True)
    return dataloader


class FrameDataset(torch.utils.data.Dataset):
    '''
    Custom PyTorch dataset that groups data by time bins.
    
    Key Features:
    - Bins data points into fixed time intervals (e.g., 0.1s bins)
    - Each sample contains ALL data points from a single time bin
    - Proper bin assignment: [bin_start, bin_start + bin_width)
    - Easy to adjust bin width
    '''
    def __init__(self, inputs_sorted, outputs_sorted, bin_width=0.01, time_col=2):
        self.inputs = inputs_sorted
        self.outputs = outputs_sorted
        self.bin_width = bin_width
        self.time_col = time_col
        
        # Extract time values
        t_values = inputs_sorted[:, time_col]
        

        min_time = t_values.min().item()
        max_time = t_values.max().item()
        
        bin_indices = torch.floor((t_values - min_time) / bin_width).long()
        
        # Calculate actual bin start times
        self.bin_starts = torch.unique(bin_indices) * bin_width + min_time
        
        # Group data indices by bin
        self.bin_groups = []
        for bin_idx in torch.unique(bin_indices):
            mask = bin_indices == bin_idx
            indices = torch.where(mask)[0]
            self.bin_groups.append(indices)
        
    def __len__(self):
        return len(self.bin_groups)
    
    def __getitem__(self, idx):
        indices = self.bin_groups[idx]
        return self.inputs[indices], self.outputs[indices]
    
    def get_bin_info(self, idx):
        """Get information about a specific bin"""
        bin_start = self.bin_starts[idx].item()
        bin_end = bin_start + self.bin_width
        num_points = len(self.bin_groups[idx])
        return {
            'bin_start': bin_start,
            'bin_end': bin_end,
            'num_points': num_points,
            'time_range': f"[{bin_start:.3f}, {bin_end:.3f})"
        }
    
    def print_bin_summary(self, max_bins=10):
        """Print summary of first few bins"""
        print(f"\nBin Summary (showing first {min(max_bins, len(self))} bins):")
        for i in range(min(max_bins, len(self))):
            info = self.get_bin_info(i)
            print(f"  Bin {i}: {info['time_range']} - {info['num_points']} points")


class GPUDataLoader:
    '''Wrapper to torch.utils.data.DataLoader to move tensors to GPU'''
    def __init__(self, dataloader, device):
        self.device = device    
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield self._move_to_device(batch)
    
    def __len__(self):
        return len(self.dataloader)
    
    def _move_to_device(self, batch):
        """Recursively move tensors in batch to the specified device"""
        if hasattr(batch, 'to'):  # torch.Tensor
            return batch.to(self.device, non_blocking = True)
        elif isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(item) for item in batch)
        else:
            return batch  # Return as-is if not a tensor or container
    
def temp_nn_setup(data, frames, device, pin_memory = True, num_workers = 4, SP_T = False, MSE = False, bin_width = .1):
    if SP_T:
        'I want to just save the data into'
        inputs, outputs = in_out(data,frames = frames)
        inputs = inputs.detach().clone().cpu()
        outputs = outputs.detach().clone().cpu()
        dataset = FrameDataset(inputs.cpu(),outputs.cpu(), bin_width=bin_width)
        dataloader = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, num_workers=num_workers,
                                                 persistent_workers= True)
        # dataloader = GPUDataLoader(dataloader, device=device)
        return dataloader
    if MSE:
        inputs, outputs = in_out(data,frames = frames)
        inputs = inputs.detach().clone().cpu()
        outputs = outputs.detach().clone().cpu()
        dataset = torch.utils.data.TensorDataset(inputs, outputs )
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size= 10000,
                                                num_workers=num_workers,
                                                pin_memory = pin_memory,
                                                persistent_workers= True,
                                                shuffle=True)
        # dataloader = GPUDataLoader(dataloader, device=device)
        return dataloader


def net_loading(net, net_params_path):

    if not os.path.exists(net_params_path):
        raise FileNotFoundError(f"Model file not found: {net_params_path}")
    
    net.load_state_dict(torch.load(net_params_path, weights_only=True))
    return net

class temp_nn_wrap:
    '''This acts as a function that we can call to train our temp nn
        -- We give it a net and the function trains it
        -- specify the num_epochs
        -- specify the training data
        -- what our loss function is
        -- what optimizer we use
        -- the device we are on'''
    
    def __init__(self, device, dataloader, model_config, optimizer, net):
        self.dataloader = dataloader
        self.net = net 
        self.optimizer = optimizer
        self.model_config = model_config
        self.device = device


    def set_params(self, net_params, optim_params):
        self.net.load_state_dict_state_dict(net_params)
        self.optim.load_state_dict(optim_params)

    # def train(self, num_epochs):
    #     batched_training(self.dataloader, num_epochs, self.device, self.net, self.crit, self.optim)

    def MSE_train(self,num_epochs,
                        prints = False, saving = False, save_path =None):
        crit = Losses.crit(MSE = True)
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):   # DataLoader gives you batches
                # Move to GPU
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.net(batch_x)

                loss = crit(predictions, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track progress
                epoch_loss += loss.item()
                num_batches += 1
                del batch_x, batch_y
                torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / num_batches
            if prints:
                print(f'Epoch {epoch} Complete - Avg Loss: {avg_loss:.6f}')
            if saving:
                torch.save({'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'model config': self.model_config,
                'loss': loss.item(),
                }, save_path)
                
        return avg_loss, epoch
    
    def SP_T_train(self,num_epochs,
                        prints = False, saving = False, save_path =None, lx = 0, ly = 0 , lt = 0):
        mse = torch.nn.MSELoss()
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):   # DataLoader gives you batches
                # Move to GPU
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)


                self.optimizer.zero_grad(set_to_none = True)


                loss = Losses.spt_reg(batch_x, net=self.net, lx = lx,ly =ly, lt =lt)+mse(self.net(batch_x), batch_y)

                loss.backward()
                self.optimizer.step()
                
                # Track progress
                epoch_loss += loss.item()
                num_batches += 1

        

                torch.cuda.empty_cache()
                gc.collect()
            
            avg_loss = epoch_loss / num_batches

            torch.cuda.empty_cache()
            if prints:
                print(f'Epoch {epoch} Complete - Avg Loss: {avg_loss:.6f}')
            if saving:
                torch.save({'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'model config': self.model_config,
            }, save_path)
                

        gc.collect()
        torch.cuda.empty_cache()
        return avg_loss, epoch
    
    def SPT_debugging(self,max_num_epochs,
                        prints = False, saving = False, save_path =None, lx = 0, ly = 0 , lt = 0,
                        ver_dataloader = None, stopping_crit=None):
        mse = torch.nn.MSELoss()
        reg_losses = []
        mse_losses = []
        total_losses = []
        mse_ver = []
        stopping = torch.zeros(max_num_epochs)
        for epoch in range(max_num_epochs):
            num_batches = 0
            reg_epoch, mse_epoch, total_epoch = 0.0, 0.0, 0.0
            for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):   # DataLoader gives you batches
                # Move to GPU
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)


                self.optimizer.zero_grad(set_to_none = True)

                loss1 = Losses.spt_reg(batch_x, net=self.net, lx = lx,ly =ly, lt =lt)
                loss2 = mse(self.net(batch_x), batch_y)
                loss = loss1 + loss2

                loss.backward()
                self.optimizer.step()
                
                if prints:
                    if batch_idx % 10== 0:
                        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

                torch.cuda.empty_cache()
                gc.collect()
                
                num_batches += 1
        
            reg_epoch, mse_epoch, total_epoch = [0.0,0.0,0.0]
            for (batch_x, batch_y) in self.dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                reg_batch = Losses.spt_reg(batch_x, net=self.net, lx=lx, ly=ly, lt=lt).item()
                mse_batch = mse(self.net(batch_x), batch_y).item()

                reg_epoch += reg_batch
                mse_epoch += mse_batch
                total_epoch += (reg_batch + mse_batch)
        

            reg_losses.append(reg_epoch/num_batches)
            mse_losses.append(mse_epoch/num_batches)
            total_losses.append(total_epoch/num_batches)


            if ver_dataloader is not None:
                mse_ver_epoch = 0.0
                for (batch_x, batch_y) in ver_dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    mse_ver_epoch += mse(self.net(batch_x), batch_y).item()
                mse_ver_epoch /= len(ver_dataloader)
                mse_ver.append(mse_ver_epoch)
                
            torch.cuda.empty_cache()
            if prints:
                print(f'Epoch {epoch} Complete - Avg Loss: {total_epoch:.6f}')
            if saving:
                torch.save({'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'model config': self.model_config,
            }, save_path)
                
            stopping[epoch] = mse_ver_epoch
            'this method can totally be changed. When I rea make this class outsource this to another function for debugging.'
            if epoch> 40:
                moving_window = stopping[epoch-20:epoch]
                moving_window_avg = moving_window.mean()
                if stopping[epoch]- moving_window_avg > stopping_crit:
                    print('stopped_training at epoch: {epoch}')
                    break
            
                
        losses_dict = {
            'reg_losses': reg_losses,
            'mse_losses': mse_losses,
            'total_losses': total_losses}
        
        if ver_dataloader is not None:
            losses_dict['mse_ver'] = mse_ver

        
        gc.collect()
        torch.cuda.empty_cache()
        return losses_dict