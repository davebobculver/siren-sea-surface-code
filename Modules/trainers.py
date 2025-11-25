import torch
import numpy as np
import Losses


def stop(losses, idx, crit = .5, window =20 ,start_idx = 50):
    'losses in a numpy array of shapoe (n)'
    if idx < start_idx :
        return False
    else:
        lb = idx - window
        if lb < 0 :
            raise ValueError('set start_idx higher or shrink window')
        ub = idx-1
        errors = losses[lb:ub]
        numbers = np.array(list(range(window-1)))
        coeffs = np.polyfit(numbers, errors, 2)

        'Now subtract polyfit off the errors'
        pred_errors = np.polyval(coeffs, numbers)
        err = errors - pred_errors  
        std = np.std(err)

        'Now we check the prediction of losses[idx]'
        y = losses[idx]
        pred_y = np.polyval(coeffs, y)
        err_y = y - pred_y

        "Check if y is outised of 2 std's"
        return y > 2*std

            




class training:
    'This class is a more polisher version of what I have done below'
    
    def __init__(self, device, dataloader, model_config, optimizer, net):
        self.dataloader = dataloader
        self.net = net 
        self.optimizer = optimizer
        self.model_config = model_config
        self.device = device


    def set_params(self, net_params, optim_params):
        self.net.load_state_dict(net_params)
        self.optimizer.load_state_dict(optim_params)


    def MSE_training(self,max_num_epochs,
                        prints = False, saving = False, save_path =None,
                        ver_dataloader = None, stopping_crit=None):
        
        mse = torch.nn.MSELoss()
        mse_losses, mse_ver = [[], []]

        stopping = np.zeros(max_num_epochs)
        num_epochs = 0
        for epoch in range(max_num_epochs):
            num_batch = 0 
            for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):   # DataLoader gives you batches
                # Move to GPU
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad(set_to_none = True)

                loss = mse(self.net(batch_x), batch_y)
                loss.backward()
                self.optimizer.step()            
                if prints:
                    if batch_idx % 10== 0:
                        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
                num_batch+= 1



            mse_epoch = 0.0
            for (batch_x, batch_y) in self.dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                mse_batch = mse(self.net(batch_x), batch_y).item()
                mse_epoch += mse_batch
            mse_losses.append(mse_epoch/num_batch)
                
            mse_ver_epoch = 0.0
            for (batch_x, batch_y) in ver_dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                mse_ver_epoch += mse(self.net(batch_x), batch_y).item()
            mse_ver_epoch /= num_batch
            mse_ver.append(mse_ver_epoch)

            'Stopping Crits'
            stopping[epoch] = mse_ver_epoch

            try:
                should_stop = stop(stopping, epoch, crit=stopping_crit)
                if not isinstance(should_stop, bool):
                    raise TypeError(f"stop() returned a non-bool: {type(should_stop)}")
                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch}. Validation loss spike detected.")
                    break
            except ValueError as e:
                print(f"Skipping early stopping check: {e}")

            if mse_epoch/(.06**2)<1:
                print(f'Low training error: {mse_epoch}<1')
                print(f"Finished training at epoch {epoch}")
                break


            if prints:
                print(f'Epoch {epoch} Complete - Avg Loss: {mse_epoch:.6f}')
            if saving:
                torch.save({'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'model config': self.model_config,
            }, save_path)
                    
        losses_dict = {
        'mse_losses': mse_losses,
        'mse_ver':mse_ver}

        return losses_dict

        
                





            










'old trainer is below'
'-------------------------------------------------------------'

# class temp_nn_wrap:
#     '''This acts as a function that we can call to train our temp nn
#         -- We give it a net and the function trains it
#         -- specify the num_epochs
#         -- specify the training data
#         -- what our loss function is
#         -- what optimizer we use
#         -- the device we are on'''
    
#     def __init__(self, device, dataloader, model_config, optimizer, net):
#         self.dataloader = dataloader
#         self.net = net 
#         self.optimizer = optimizer
#         self.model_config = model_config
#         self.device = device


#     def set_params(self, net_params, optim_params):
#         self.net.load_state_dict_state_dict(net_params)
#         self.optim.load_state_dict(optim_params)

#     # def train(self, num_epochs):
#     #     batched_training(self.dataloader, num_epochs, self.device, self.net, self.crit, self.optim)

#     def MSE_train(self,num_epochs,
#                         prints = False, saving = False, save_path =None):
#         crit = Losses.crit(MSE = True)
#         for epoch in range(num_epochs):
#             epoch_loss = 0
#             num_batches = 0
#             for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):   # DataLoader gives you batches
#                 # Move to GPU
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
#                 # Forward pass
#                 self.optimizer.zero_grad()
#                 predictions = self.net(batch_x)

#                 loss = crit(predictions, batch_y)

#                 # Backward pass
#                 loss.backward()
#                 self.optimizer.step()
                
#                 # Track progress
#                 epoch_loss += loss.item()
#                 num_batches += 1
#                 del batch_x, batch_y
#                 torch.cuda.empty_cache()
            
#             avg_loss = epoch_loss / num_batches
#             if prints:
#                 print(f'Epoch {epoch} Complete - Avg Loss: {avg_loss:.6f}')
#             if saving:
#                 torch.save({'model_state_dict': self.net.state_dict(),
#                 'optimizer_state_dict': self.optimizer.state_dict(),
#                 'epoch': epoch,
#                 'model config': self.model_config,
#                 'loss': loss.item(),
#                 }, save_path)
                
#         return avg_loss, epoch
    
#     def SP_T_train(self,num_epochs,
#                         prints = False, saving = False, save_path =None, lx = 0, ly = 0 , lt = 0):
#         mse = torch.nn.MSELoss()
#         for epoch in range(num_epochs):
#             epoch_loss = 0
#             num_batches = 0
#             for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):   # DataLoader gives you batches
#                 # Move to GPU
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)


#                 self.optimizer.zero_grad(set_to_none = True)


#                 loss = Losses.spt_reg(batch_x, net=self.net, lx = lx,ly =ly, lt =lt)+mse(self.net(batch_x), batch_y)

#                 loss.backward()
#                 self.optimizer.step()
                
#                 # Track progress
#                 epoch_loss += loss.item()
#                 num_batches += 1

        

#                 torch.cuda.empty_cache()
#                 gc.collect()
            
#             avg_loss = epoch_loss / num_batches

#             torch.cuda.empty_cache()
#             if prints:
#                 print(f'Epoch {epoch} Complete - Avg Loss: {avg_loss:.6f}')
#             if saving:
#                 torch.save({'model_state_dict': self.net.state_dict(),
#                 'optimizer_state_dict': self.optimizer.state_dict(),
#                 'epoch': epoch,
#                 'model config': self.model_config,
#             }, save_path)
                

#         gc.collect()
#         torch.cuda.empty_cache()
#         return avg_loss, epoch
    
#     def SPT_debugging(self,max_num_epochs,
#                         prints = False, saving = False, save_path =None, lx = 0, ly = 0 , lt = 0,
#                         ver_dataloader = None, stopping_crit=None):
#         mse = torch.nn.MSELoss()
#         reg_losses = []
#         mse_losses = []
#         total_losses = []
#         mse_ver = []
#         stopping = torch.zeros(max_num_epochs)
#         for epoch in range(max_num_epochs):
#             num_batches = 0
#             reg_epoch, mse_epoch, total_epoch = 0.0, 0.0, 0.0
#             for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):   # DataLoader gives you batches
#                 # Move to GPU
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)


#                 self.optimizer.zero_grad(set_to_none = True)

#                 loss1 = Losses.spt_reg(batch_x, net=self.net, lx = lx,ly =ly, lt =lt)
#                 loss2 = mse(self.net(batch_x), batch_y)
#                 loss = loss1 + loss2

#                 loss.backward()
#                 self.optimizer.step()
                
#                 if prints:
#                     if batch_idx % 10== 0:
#                         print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

#                 torch.cuda.empty_cache()
#                 gc.collect()
                
#                 num_batches += 1
        
#             reg_epoch, mse_epoch, total_epoch = [0.0,0.0,0.0]
#             for (batch_x, batch_y) in self.dataloader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)

#                 reg_batch = Losses.spt_reg(batch_x, net=self.net, lx=lx, ly=ly, lt=lt).item()
#                 mse_batch = mse(self.net(batch_x), batch_y).item()

#                 reg_epoch += reg_batch
#                 mse_epoch += mse_batch
#                 total_epoch += (reg_batch + mse_batch)
        

#             reg_losses.append(reg_epoch/num_batches)
#             mse_losses.append(mse_epoch/num_batches)
#             total_losses.append(total_epoch/num_batches)


#             if ver_dataloader is not None:
#                 mse_ver_epoch = 0.0
#                 for (batch_x, batch_y) in ver_dataloader:
#                     batch_x = batch_x.to(self.device)
#                     batch_y = batch_y.to(self.device)
#                     mse_ver_epoch += mse(self.net(batch_x), batch_y).item()
#                 mse_ver_epoch /= len(ver_dataloader)
#                 mse_ver.append(mse_ver_epoch)
                
#             torch.cuda.empty_cache()
#             if prints:
#                 print(f'Epoch {epoch} Complete - Avg Loss: {total_epoch:.6f}')
#             if saving:
#                 torch.save({'model_state_dict': self.net.state_dict(),
#                 'optimizer_state_dict': self.optimizer.state_dict(),
#                 'epoch': epoch,
#                 'model config': self.model_config,
#             }, save_path)
                
#             stopping[epoch] = mse_ver_epoch
#             'this method can totally be changed. When I rea make this class outsource this to another function for debugging.'
#             if epoch> 40:
#                 moving_window = stopping[epoch-20:epoch]
#                 moving_window_avg = moving_window.mean()
#                 if stopping[epoch]- moving_window_avg > stopping_crit:
#                     print('stopped_training at epoch: {epoch}')
#                     break
            
                
#         losses_dict = {
#             'reg_losses': reg_losses,
#             'mse_losses': mse_losses,
#             'total_losses': total_losses}
        
#         if ver_dataloader is not None:
#             losses_dict['mse_ver'] = mse_ver

        
#         gc.collect()
#         torch.cuda.empty_cache()
#         return losses_dict