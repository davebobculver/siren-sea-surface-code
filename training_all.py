import os

# Set all major underlying math libraries to use exactly 6 threads
os.environ["OMP_NUM_THREADS"] = "6"        # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "6"   # OpenBLAS (Default for most NumPy installs)
os.environ["MKL_NUM_THREADS"] = "6"        # Intel MKL (Default for Conda installs)
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # Apple Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "6"    # NumExpr (Used by Pandas)

import torch
import Modules.preprocess as pp
import numpy as np 
import Modules.trainers as tr
import torch
from Modules.Neural_Net import Net, Sine
from multiprocessing import freeze_support
import gc



data  = torch.load('../../data/data.pt', weights_only=True)

w_0 =1
beams = list(range(0,32))
act = Sine(w_0=w_0)
num_epochs = 200
model_config = (3, 300, 500, 500, 300)


data_index = [0,200, 400, 600, 750, 940, 1080, 1300, 1500, 1680, 1850, 2027,2250, 2450, 2619, 2780, 2945, 3150, 3300, 3514 ]


if __name__ == '__main__':
    freeze_support()
    for i in range(1, len(data_index)):

        frames = np.arange(data_index[i-1], data_index[i], 2)
        print(frames)

        datasets = pp.data_gen(data =data, beams = beams, frames = frames, normalize='SS',splits = .8, w_0=1, new_std = 1)
        train_dataloader, ver_dataloader = datasets.get_dataloaders(num_workers =4, batch_size = 1)


        net = Net(act, *model_config, w_0 = w_0, siren= False).to('cuda')
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = num_epochs)


        'Start training'


        path = f'nets/full_fit/{data_index[i-1]:04d}_{data_index[i]:04d}.pt'


        trainer =  tr.training(device='cuda', dataloader= train_dataloader, model_config=model_config,
                                optimizer=optimizer, net=net, scalers=datasets.get_scalers(), masker=datasets.get_mask(), w_0 = w_0, act =act )

        train = trainer.MSE_training(num_epochs, prints=True, saving=True, 
                                        save_path=path, count = 100)
        
        # 2. EXPLICIT CLEANUP
        # Delete dataloaders and run garbage collection to kill worker processes
        # before the next iteration of the loop starts.
        del train_dataloader
        del ver_dataloader
        del datasets
        del trainer
        del net
        
        gc.collect()
        torch.cuda.empty_cache() # Frees up unreferenced VRAM between fits



