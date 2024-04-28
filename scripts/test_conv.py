import torch
import torch.nn as nn

residual_channels = 32
dilation_channels = 64
kernel_size = 2
new_dilation = 1
conv = nn.Conv1d(in_channels=residual_channels,
                out_channels=dilation_channels,
                kernel_size=(1, kernel_size), dilation=new_dilation)

input_tensor = torch.randn(32, 32, 207, 13)

output_tensor = conv(input_tensor)

print(output_tensor.shape)