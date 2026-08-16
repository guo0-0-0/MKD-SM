# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from math import sqrt
import torch
from torch import Tensor
from torch import nn
import math

class SRCNN(nn.Module):
    r""" Construct SRCNN super-resolution model.
    
    Args:
        mode (optional, str): Because the SRCNN model is inconsistent in the training and testing mode.
                              If set to `train`, the convolutional layer does not need to fill the edge of 
                              the image, otherwise it is filled. (Default: `train`)
    """

    def __init__(self, mode: str = "train", init_weights: bool = True) -> None:
        super(SRCNN, self).__init__()
        # The model does not need to fill the edges during the training process, 
        # and needs to fill the edges during the testing mode.
        if mode == "train":
            padding = False
        elif mode == "eval":
            padding = True
        else:
            padding = True

        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 9, 1, 0 if not padding else 4),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.ReLU(True)
        )

        # Reconstruction the layer.
        self.reconstruction = nn.Conv2d(32, 1, 5, 1, 0 if not padding else 2)

        # Initialize model weights.
        if init_weights:
            self._initialize_weights()

    # The filter weights of each layer are initialized by random sampling and have 
    # a Gaussian distribution with zero mean and standard deviation of 0.001 (with a deviation of 0).
    def _initialize_weights(self) -> None:
        for m in self.features or self.map:
            if isinstance(m, nn.Conv2d):
                mean = 0.0
                std = sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

class SRCNN_2channels(nn.Module):
    r""" Construct SRCNN super-resolution model.
    
    Args:
        mode (optional, str): Because the SRCNN model is inconsistent in the training and testing mode.
                              If set to `train`, the convolutional layer does not need to fill the edge of 
                              the image, otherwise it is filled. (Default: `train`)
    """

    def __init__(self, init_weights: bool = True) -> None:
        super(SRCNN_2channels, self).__init__()

        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, 9, 1, padding=9 // 2),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, 5, 1, padding=5 // 2),
            nn.ReLU(True)
        )

        # Reconstruction the layer.
        self.reconstruction = nn.Conv2d(32, 2, 5, 1, padding=5 // 2)

        # Initialize model weights.
        if init_weights:
            self._initialize_weights()

    # The filter weights of each layer are initialized by random sampling and have 
    # a Gaussian distribution with zero mean and standard deviation of 0.001 (with a deviation of 0).
    def _initialize_weights(self) -> None:
        for m in self.features or self.map:
            if isinstance(m, nn.Conv2d):
                mean = 0.0
                std = sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out


class VDSR_2channels(nn.Module):
    def __init__(self):
        super(VDSR_2channels, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))


#######  SMRnet_2d  #######

class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='leakyrelu', mode='CNA', dim=3):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode, dim=dim)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode, dim=dim)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode, dim=dim)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode, dim=dim)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=last_act, mode=mode, dim=dim)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='leakyrelu', mode='CNA', dim=3):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode, dim=dim)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode, dim=dim)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode, dim=dim)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def pad(pad_type, padding, dim=3): #Dim = 2
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    # dim = 2 or 3
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect' and not dim == 3:
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding) if not dim == 3 else nn.ReplicationPad3d(padding)
    else:
        raise NotImplementedError('padding layer [{:s} {:d}] is not implemented'.format(pad_type, dim))
    return layer


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA', dim = 3): #Dim = 2
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''

    #  dim = 2 or 3

    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type=pad_type, padding=padding, dim=dim) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    conv_func = nn.Conv2d if not dim == 3 else nn.Conv3d
    c = conv_func(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
                  dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None

    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)

    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                       pad_type='zero', norm_type=None, act_type='relu', dim=3):# Dim = 2
    # dim = 2 or 3
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** dim), kernel_size, stride, bias=bias, \
                      pad_type=pad_type, norm_type=None, act_type=None, dim=dim)

    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                 pad_type='zero', norm_type=None, act_type='relu', mode='nearest', dim=3):
    # dim = 2 or 3
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    if dim == 2:
        upsample = nn.Upsample(upscale_factor, mode=mode)
    else:
        upsample = nn.Upsample(scale_factor=(1, upscale_factor, upscale_factor), mode=mode)

    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, dim=dim)
    return sequential(upsample, conv)


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
                 act_type='leakyrelu', mode='CNA', upsample_mode='upconv', dim=3):
        super(RRDBNet, self).__init__()

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode, dim=dim)

        rb_blocks = [RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                          norm_type=norm_type, act_type=act_type, mode='CNA', dim=dim) for _ in range(nb)]

        LR_conv = conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode, dim=dim)

        if upsample_mode == 'upconv':
            upsample_block = upconv_block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type, dim=dim)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type, dim=dim) for _ in range(n_upscale)]

        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode, dim=dim)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None, mode=mode, dim=dim)

        self.model = sequential(fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)),
                                *upsampler, HR_conv0, HR_conv1)
    def forward(self, x):
        x = self.model(x)
        return x
