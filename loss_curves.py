
'modules set up'
import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from Modules import pre_pro
import torch
from torch.utils.data import TensorDataset, DataLoader
from Modules.Neural_Net import Net
from Modules.Losses import crit
import numpy as np
from Modules.beams import beam_trim
import matplotlib.pyplot as plt

torch.set_num_threads(6)
torch.set_num_interop_threads(6)


'Training Dataloader'
data = torch.load('../../data/data.pt', weights_only=True)
frames = np.arange(301,302, 1)
device = 'cuda'

beams = list(range(0, 32))
indices_to_remove = [4, 30]
beams = [b for i, b in enumerate(beams) if i not in indices_to_remove]

with torch.no_grad():
    input, output = pre_pro.in_out(data, beams=beams, frames=frames)


dataset = pre_pro.FrameDataset(inputs_sorted=input, outputs_sorted= output)
spt_dataloader = DataLoader(dataset, pin_memory=True,shuffle = True, num_workers=1, persistent_workers= True, batch_size=1)
print(len(spt_dataloader))


'Testing Dataloader'
data = torch.load('../../data/data.pt', weights_only=True)
frames = np.arange(301,302, 1)
device = 'cuda'

beams = [4,30]

with torch.no_grad():
    input, output = pre_pro.in_out(data, beams=beams, frames=frames)


dataset = pre_pro.FrameDataset(inputs_sorted=input, outputs_sorted= output)
ver_dataloader = DataLoader(dataset, pin_memory=True,shuffle = True, num_workers=1, persistent_workers= True, batch_size=1)
print(len(spt_dataloader))


epochs, alpha = [10000, 5]

loss_curve = 'data/e_10000_a_5.pt'

network_info = torch.load('nets/MSE/MSE_one_frame.pt', weights_only=True)
retrain_net = Net(torch.cos, *network_info['model config'])
retrain_net.to('cuda')
spt_optimizer = torch.optim.Adam(retrain_net.parameters(), lr = 1e-3 )



if __name__ == '__main__':

    retrainer = pre_pro.temp_nn_wrap('cuda', spt_dataloader, model_config=network_info['model config'],
                                    optimizer=spt_optimizer, net = retrain_net)

    losses = retrainer.SPT_debugging(epochs, prints=True, saving= True,
                                    save_path='nets/SPT/spt_loss_curves.pt', ly =alpha, ver_dataloader = ver_dataloader )

torch.save(losses, loss_curve)