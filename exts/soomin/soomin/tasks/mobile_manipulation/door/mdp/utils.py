import torch

def multiply_quat(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # q and r are tensors of shape (N, 4) representing quaternions
    w1, x1, y1, z1 = q.unbind(dim=-1)
    w2, x2, y2, z2 = r.unbind(dim=-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return torch.stack([w, x, y, z], dim=-1)

def inverse_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)