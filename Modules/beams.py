import torch

def beam_trim(data_frame, beams, beam_col = 3):
    beam_col_idx = beam_col
    beam_numbers = data_frame[:, beam_col_idx] 
    beams_tensor = torch.tensor(beams, device=data_frame.device)
    mask = (beam_numbers.unsqueeze(1) == beams_tensor).any(dim=1)
    removed = data_frame[mask]
    columns_to_keep = [i for i in range(removed.shape[1]) if i != beam_col_idx]
    removed = removed[:, columns_to_keep]


    return removed