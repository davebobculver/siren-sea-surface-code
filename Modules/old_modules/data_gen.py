# import numpy as np
# import torch
# from Modules.DataTrim_3d import Trimmer

# class loader:
#     def __init__(self, data, k, lb, ub):
#         pass
#     #     x_AB = data[k]['X']
#     #     y_AB = data[k]['Y']
#     #     z_AB = data[k]['Z']
#     #     BN = data[k]['Beamnum'][0]

#     #     x_min = -253
#     #     y_min = -178
#     #     x_max = -132
#     #     y_max = -146

#     #     x = []
#     #     y = []
#     #     z = []

#     #     for j in range(32):
#     #         load = Trimmer(x_AB, y_AB, z_AB, BN)
#     #         load.trim_cut(j, lb, ub)
#     #         x.append(load.x)
#     #         y.append(load.y)
#     #         z.append(load.z)

#     #     x = np.concatenate(x)
#     #     y = np.concatenate(y)
#     #     z = np.concatenate(z)

#     #     # Vectorized filter
#     #     mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
#     #     x_new = x[mask]
#     #     y_new = y[mask]
#     #     z_new = z[mask]

#     #     self.x = torch.tensor(x_new, dtype=torch.float32)
#     #     self.y = torch.tensor(y_new, dtype=torch.float32)
#     #     self.z = torch.tensor(z_new, dtype=torch.float32).view(-1, 1)
#     #     self.xy = torch.vstack((self.x, self.y)).T

#     def withold(self, data, k, lb, ub, beams):
#         x_AB = data[k]['X']
#         y_AB = data[k]['Y']
#         z_AB = data[k]['Z']
#         BN = data[k]['Beamnum'][0]

#         x_min = -253
#         y_min = -178
#         x_max = -132
#         y_max = -146

#         x = []
#         y = []
#         z = []

#         for j in beams:
#             load = Trimmer(x_AB, y_AB, z_AB, BN)
#             load.trim_cut(j, lb, ub)
#             x.append(load.x)
#             y.append(load.y)
#             z.append(load.z)

#         x = np.concatenate(x)
#         y = np.concatenate(y)
#         z = np.concatenate(z)

#         # Vectorized filter
#         mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
#         x_new = x[mask]
#         y_new = y[mask]
#         z_new = z[mask]

#         self.x_w = torch.tensor(x_new, dtype=torch.float32)
#         self.y_w = torch.tensor(y_new, dtype=torch.float32)
#         self.z_w = torch.tensor(z_new, dtype=torch.float32).view(-1, 1)
#         self.xy_w = torch.vstack((self.x_w, self.y_w)).T


# class gridmaking:
#     def __init__(self, x_lb, x_ub, y_lb, y_ub, x_num, y_num):
#         # x = torch.linspace(x_lb, x_ub, x_num)
#         y = torch.linspace(y_lb, y_ub, y_num)
#         self.X, self.Y = torch.meshgrid(x, y, indexing='ij')

#         x_flat = self.X.flatten()
#         y_flat = self.Y.flatten()

#         grid_points = np.vstack((x_flat.numpy(), y_flat.numpy())).T
#         self.grid = torch.tensor(grid_points, dtype=torch.float32).requires_grad_(True)