import torch
from beams import beam_trim 
from Neural_Net import Net
def add_t(tensors,time):   
    shape = torch.ones_like(tensors[:,1:2])
    return torch.hstack((tensors,shape*time ))

class data:
    def __init__(self, dataset, time, beams):
        self.frame = dataset[round(time*10)]
        self.train_in = add_t(beam_trim(self.frame[:,[0,1,3]], beams, beam_col=2), time)

        self.ver_in = add_t( beam_trim(self.frame[:,[0,1,3]], beams, beam_col=2), time)
        self.ver_z = beam_trim(self.frame[:,[2,3]],beams, beam_col=1)
        self.train_z = beam_trim(self.frame[:,[2,3]],beams, beam_col =1)
    
    def training_data(self):
        return [self.train_in, self.train_z]
    
    def verification_data(self):
        return [self.ver_in, self.ver_z]




class stats:
    def __init__(self, training_data, verification_data,model):
        self.train = training_data
        self.ver = verification_data
        self.model = model

    def NRMS(self, data, pred):
        return torch.sqrt(((data-pred)/.06).pow(2).mean())

    def train_rms(self):
        model_pred = self.model(self.train[0]).reshape(-1)
        obs = self.train[1].reshape(-1)
        return self.NRMS(model_pred,obs)

    def ver_rms(self):
        return self.NRMS(self.ver[0],self.ver[1])
    
point_cloud = torch.load('../../data/data.pt', weights_only=True)

network_info = torch.load('nets/MSE/MSE_0_50.pt', weights_only=True)
model_config = network_info['model config']
net = Net(torch.cos, *model_config)
net.load_state_dict(network_info['model_state_dict'])

device = 'cpu'
net = net.to(device)
cp = [-180,-160, 2.5]

datas = data(point_cloud, 25, [4,30])
eval = stats(datas.training_data(), datas.verification_data(), net)
print(eval.train_rms())