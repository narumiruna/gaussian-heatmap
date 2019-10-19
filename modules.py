import torch
from torch import nn


class HeatmapToPoint(nn.Module):
    def __init__(self, size, beta=0, gamma=1):
        super(HeatmapToPoint, self).__init__()
        self.size = size

        grid_y, grid_x = self._make_grids(size)
        self.register_buffer('grid_y', grid_y)
        self.register_buffer('grid_x', grid_x)

        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))

    def _make_grids(self, size):
        h, w = size

        y = torch.arange(h).float().add(0.5).div(h)
        x = torch.arange(w).float().add(0.5).div(w)

        grid_y, grid_x = torch.meshgrid(y, x)

        return grid_y, grid_x

    def forward(self, heatmap):
        grid_y = self.grid_y[None, None, :, :].expand_as(heatmap)
        grid_x = self.grid_x[None, None, :, :].expand_as(heatmap)

        prob = heatmap.add(self.beta).mul(self.gamma)
        prob = prob.flatten(2).softmax(dim=2).view_as(heatmap)

        y = prob.mul(grid_y).flatten(2).sum(2)
        x = prob.mul(grid_x).flatten(2).sum(2)

        return torch.stack([y, x], dim=2)
