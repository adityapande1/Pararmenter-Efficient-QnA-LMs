"""Implements an Adapter, Low-rank adapters and Hyper-adapter Layers."""
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from .adapter_utils import Activations

from .hypercomplex.layers import PHMLinear
from .low_rank_layer import LowRankLinear

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels,x, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x_windows = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x_windows = x_windows.contiguous().view(*x_windows.size()[:-2], -1)     
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, *x_windows.size()[2:])
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x_windows = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x_windows = x_windows.contiguous().view(*x_windows.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x_windows.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
class LinearCombinationModel(nn.Module):
    def __init__(self, c, d):
        super(LinearCombinationModel, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(d, 1) for _ in range(c)])

    def forward(self, x):
        # Apply linear layers and concatenate outputs along the third dimension
        linear_outputs = [linear_layer(x[:,:,i,:]) for i,linear_layer in enumerate(self.linear_layers)]
        return torch.cat(linear_outputs, dim=2)


class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size,
                                          w_init=config.low_rank_w_init,
                                          rank=config.low_rank_rank)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim,
                                        w_init=config.low_rank_w_init,
                                        rank=config.low_rank_rank)

        self.track_z = config.track_z

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        if self.track_z:
            self.z = z
        output = self.up_sampler(z)
        return output
"""      
class Adapter(nn.Module):
    Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized.

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = nn.Tanh()
        self.linear_co=LinearCombinationModel(191,8) #kernel (1,8) stride (1,4)
        self.linear=nn.Linear(191,self.input_dim)
        self.track_z = config.track_z

    def forward(self, x):
        x = self.linear_co(x.unfold(2,8,4))
        x = self.activation(x)
        x = self.linear(x)
        if self.track_z:
            self.z = x
        return x
"""
"""
class Adapter(nn.Module):
    Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized.

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = nn.Tanh()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1, 4), stride=(1, 1))  # Adjust stride here
        self.conv_up1 = nn.ConvTranspose2d(128, 1, kernel_size=(1, 4), stride=(1, 1))  # Adjust stride here
        self.track_z = config.track_z

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x.unsqueeze(1))
        x = self.activation(x)
        x = self.conv_up1(x)
        x = x.squeeze(1)
        if self.track_z:
            self.z = x
        return x
        
"""
class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler1 = nn.Linear(self.input_dim, self.down_sample_size//2)
        #self.down_sampler2 = nn.Linear(self.down_sample_size//2, self.down_sample_size)  
        #self.up_sampler1 = nn.Linear(self.down_sample_size, self.down_sample_size*2) 
        self.up_sampler2 = nn.Linear(self.down_sample_size//2, self.input_dim) 

        self.track_z = config.track_z

    def forward(self, x):
        z = self.down_sampler1(x)
        z = self.activation(z)
        #z = self.down_sampler2(z)
        #z = self.activation(z)
        if self.track_z:
            self.z = z
        #z=self.up_sampler1(z)
        output = self.up_sampler2(z)
        return output 

class OutputAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config, output_dim):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = 16
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, output_dim) 

    def forward(self, x):
        print("ivvdrnfciebn")
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output 

    def resize_up_sampler(self, resized_size):
        self.up_sampler = nn.Linear(self.down_sample_size, resized_size)


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLinear(in_features=self.input_dim,
                                      out_features=self.down_sample_size,
                                      bias=True,
                                      c_init=config.phm_c_init,
                                      phm_dim=config.hypercomplex_division,
                                      learn_phm=config.learn_phm,
                                      w_init=config.hypercomplex_nonlinearity,
                                      shared_phm_rule=config.shared_phm_rule,
                                      factorized_phm=config.factorized_phm,
                                      shared_W_phm=config.shared_W_phm,
                                      factorized_phm_rule=config.factorized_phm_rule,
                                      phm_rank=config.phm_rank,
                                      phm_init_range=config.phm_init_range,
                                      kronecker_prod=config.kronecker_prod)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.input_dim, 
                                    bias=True,
                                    c_init=config.phm_c_init,
                                    phm_dim=config.hypercomplex_division,
                                    learn_phm=config.learn_phm,
                                    w_init=config.hypercomplex_nonlinearity,
                                    shared_phm_rule=config.shared_phm_rule,
                                    factorized_phm=config.factorized_phm,
                                    shared_W_phm=config.shared_W_phm,
                                    factorized_phm_rule=config.factorized_phm_rule,
                                    phm_rank=config.phm_rank,
                                    phm_init_range=config.phm_init_range,
                                    kronecker_prod=config.kronecker_prod)

        self.track_z = config.track_z

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        if self.track_z:
            self.z = z
        return self.up_sampler(z)