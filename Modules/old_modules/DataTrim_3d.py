# import numpy as np

# class Trimmer:
#     def __init__(self, x_data, y_data, z_data, beams):
#         self.x_data = x_data
#         self.y_data = y_data
#         self.z_data = z_data
#         self.beams = beams

#     def trim_cut(self, beamnum, x_min, x_max):
#         self.x_min = x_min
#         self.x_max = x_max

#         # Vectorized mask for beam number
#         beam_mask = self.beams == beamnum

#         x = self.x_data[beam_mask]
#         y = self.y_data[beam_mask]
#         z = self.z_data[beam_mask]

#         # Vectorized x-bound mask
#         cut_mask = (x > self.x_min) & (x < self.x_max)

#         self.x = x[cut_mask]
#         self.y = y[cut_mask]
#         self.z = z[cut_mask]
