import torch
from typing import Any

class NormaizerPathPlanning():
    def __init__(self, xmin_max: Any) -> None:
        self.xmin = xmin_max[0]
        self.xmax = xmin_max[1]
    
    def normalize(self, x: Any) -> Any:
        return (x - self.xmin) / (self.xmax - self.xmin) * 2 - 1

    def unnormalize(self, y: Any) -> Any:
        return 0.5 * (y + 1.0) * (self.xmax - self.xmin) + self.xmin

class NormalizerPoseMotion():
    def __init__(self, xmin_max: Any) -> None:
        self.xmin = xmin_max[0]
        self.xmax = xmin_max[1]

    def normalize(self, x: Any) -> Any:
        if torch.is_tensor(x):
            xmin = torch.tensor(self.xmin, device=x.device)
            xmax = torch.tensor(self.xmax, device=x.device)
            return (x - xmin) / (xmax - xmin) * 2 - 1

        return (x - self.xmin) / (self.xmax - self.xmin) * 2 - 1

    def unnormalize(self, y: Any) -> Any:
        if torch.is_tensor(y):
            xmin = torch.tensor(self.xmin, device=y.device)
            xmax = torch.tensor(self.xmax, device=y.device)
            return 0.5 * (y + 1.0) * (xmax - xmin) + xmin

        return 0.5 * (y + 1.0) * (self.xmax - self.xmin) + self.xmin