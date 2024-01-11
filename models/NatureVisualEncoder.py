import torch
from torch import nn
# from utils.encoder_utils import conv_output_shape

from typing import Tuple, Optional, Union
from torch.nn import init

class NatureVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int,config = None, device = "cuda:0"
    ):
        super().__init__()
        self.config = config
        self.is_pretraining = False
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)
        self.final_flat = conv_3_hw[0] * conv_3_hw[1] * 64

        self.height = height
        self.width = width
        self.initial_channels = initial_channels
        self.device = device


        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.initial_channels, 32, [8, 8], [4, 4]),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], [2, 2]),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], [1, 1]),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Flatten(),
        )
        self.conv_layers.to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

            

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # if not exporting_to_onnx.is_exporting():
        # Not sure why they are permuting the visual observations. They basically change it from
        # (B, 84, 84, 3) to (B, 4, 84, 84)
        # Convert numpy to tensor
        # visual_obs.reshape(-1, self.config["obs_shape"][2],self.config["obs_shape"][0],self.config["obs_shape"][1]).cuda()
        visual_obs.squeeze_().to(self.device)

        if visual_obs.dim() == 4:
            # this means it is during forward pass, i.e batch size is one and not training
            visual_obs = visual_obs.permute(0,3,1,2).to(self.device)
        if visual_obs.dim() == 5:
            # this means we are training. Therefore visual obs will be passed in the shape
            # (B,N,84,84,3)
            # Therefore we reshape the tensor to be of shape (B*N,84,84,3) before permuting it
            visual_obs = visual_obs.reshape(-1,self.height,self.width,3).permute(0,3,1,2).to(self.device)


        hidden = self.conv_layers(visual_obs)
        return hidden
        # return self.dense(hidden)


    
        
    def z_score_norm(self, input):
         # z = (x-mu)/sigma
        # x is the original value of the data point
        # mu is the mean of the variable being normalized
        # sigma is the standard deviation of the variable being normalized
        # z is the normalized value of the data point

        # mu = obs.mean(dim=-1, keepdim=True)
        # sigma = obs.var(dim=-1, keepdim=True)

        # z = (obs-mu)/sigma
        epsilon = 1e-6

        mu = input.mean(dim=-1, keepdim=True)
        sigma = input.var(dim=-1, keepdim=True)+epsilon

        z = (input-mu)/sigma

        return z
    
    def min_max_norm(self,input):
        self.minval = torch.min(input, dim=-1, keepdim=True).values
        self.maxval = torch.max(input, dim=-1, keepdim=True).values

        shifted = input - self.minval
        scaled = shifted/self.maxval
        # normed = (scaled*2)-1

        return scaled
    
    # 4 1 2 10
    def set_is_pretraining(self):
        self.is_pretraining = True

def conv_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    ) -> Tuple[int, int]:
        """
        Calculates the output shape (height and width) of the output of a convolution layer.
        kernel_size, stride, padding and dilation correspond to the inputs of the
        torch.nn.Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
        :param h_w: The height and width of the input.
        :param kernel_size: The size of the kernel of the convolution (can be an int or a
        tuple [width, height])
        :param stride: The stride of the convolution
        :param padding: The padding of the convolution
        :param dilation: The dilation of the convolution
        """
        from math import floor

        if not isinstance(kernel_size, tuple):
            kernel_size = (int(kernel_size), int(kernel_size))
        h = floor(
            ((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
        )
        w = floor(
            ((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
        )
        return h, w


def pool_out_shape(h_w: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a max pooling layer.
    kernel_size corresponds to the inputs of the
    torch.nn.MaxPool2d layer (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    :param kernel_size: The size of the kernel of the convolution
    """
    height = (h_w[0] - kernel_size) // 2 + 1
    width = (h_w[1] - kernel_size) // 2 + 1
    return height, width