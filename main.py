import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch import nn
from torch.nn.functional import pairwise_distance
from torch.nn.modules.utils import _pair
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image


def gaussian(x, mu=0.0, sigma=1.0, a=1.0):
    r"""Gaussian function

    f(x) = a e^{ - \frac{(x - \mu)^2}{2\sigma^2}}
    """
    return x.sub(mu).div(sigma).pow(2).div(2).neg().exp().mul(a)


def get_meshgrid(size, **kwargs):
    _, _, h, w = size

    y = torch.arange(h, **kwargs).float().add(0.5).div(h)
    x = torch.arange(w, **kwargs).float().add(0.5).div(w)

    grid_y, grid_x = torch.meshgrid(y, x)

    grid_y = grid_y[None, None, :, :].expand(*size)
    grid_x = grid_x[None, None, :, :].expand(*size)

    return grid_y, grid_x


def gaussian_heatmap(size, centers, sigma=1.0, a=1.0, device=None):
    """
    Example:
        size = (32, 68, 64, 64)
        centers = torch.rand(32, 68, 2)
        heatmap = gaussian_heatmap(size, centers, sigma=0.1)
        print(heatmap.size())
        
        >>> torch.Size([32, 68, 64, 64])
    """
    grid_y, grid_x = get_meshgrid(size, device=device)

    c_y = centers[:, :, 0][:, :, None, None]
    c_x = centers[:, :, 1][:, :, None, None]

    return a * gaussian(grid_y, mu=c_y, sigma=sigma) * gaussian(
        grid_x, mu=c_x, sigma=sigma)


def heatmap_to_point(heatmap, scale=1.0):
    grid_y, grid_x = get_meshgrid(heatmap.size(), device=heatmap.device)

    prob = heatmap.sub(0.5).mul(scale).flatten(2).softmax(
        dim=2).view_as(heatmap)

    y = prob.mul(grid_y).flatten(2).sum(2)
    x = prob.mul(grid_x).flatten(2).sum(2)

    return torch.stack([y, x], dim=2)


def heatmap_to_point_by_argmax(heatmap):
    h, w = heatmap.size(2), heatmap.size(3)
    indices = heatmap.flatten(2).argmax(dim=2)

    y_indices = indices.div(w).float().div(h)
    x_indices = indices.fmod(w).float().div(w)

    return torch.stack([y_indices, x_indices], dim=-1)


def draw_point(img, center, radius=1, fill=None):
    x, y = center
    xy = [x - radius, y - radius, x + radius, y + radius]

    draw = ImageDraw.Draw(img)
    draw.ellipse(xy=xy, fill=fill)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    size = (64, 3, 480, 640)
    _, _, h, w = size
    mu = torch.rand(size[0], size[1], 2).to(device)
    heatmap = gaussian_heatmap(size, mu, sigma=0.05, a=100, device=device)

    save_image(heatmap, 'image.jpg', normalize=True)

    new_mu = heatmap_to_point(heatmap)
    err = pairwise_distance(new_mu.view(-1, 2), mu.view(-1, 2), p=2).mean()
    print(f'reconstruction error: {err}')


if __name__ == '__main__':
    main()
