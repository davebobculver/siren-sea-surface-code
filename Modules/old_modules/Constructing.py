# from Modules.data_gen import loader
# # from Modules import Losses
# # from Modules.Neural_Net import Net
# import torch

# class Constructing:
#     def __init__(self, data):
#         self.data =data
#         # # Create evaluation grid
#         # self.F = gridmaking(-250, -146, -178, -130, 245,65)
#         # self.fg = self.F.grid
#         # self.x = self.F.X
#         # self.y = self.F.Y
#         # self.net = Net(torch.cos, 2, 100, 100, 100, 100)
#         # self.optim = torch.optim.Adam(self.net.parameters(), lr=0.01)
#         # self.rms = Losses.rms
#         # self.crit = Losses.d_loss

    
#     def withold(self, beams, frame):
#         """Now we want to withhold the beams, [0,3,29,30]"""
#         self.loader = loader(self.data, frame, -253, -132)
#         self.loader.withold(self.data, frame, -253, -132, beams)
#         self.x_w = self.loader.x_w
#         self.y_w = self.loader.y_w
#         # self.train_input = self.loader.xy_w
#         self.train_height = self.loader.z_w

