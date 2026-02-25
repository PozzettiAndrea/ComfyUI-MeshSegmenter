import contextlib

import torch
import torch.nn as nn
import comfy.ops
ops = comfy.ops.disable_weight_init

class VanillaMLP(nn.Module):
    def __init__(self, input_dim, output_dim, out_activation, n_hidden_layers=4, n_neurons=64, activation="ReLU",
                 device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        self._operations = operations
        self._device = device
        self._dtype = dtype
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.out_activation = out_activation
        layers = [
            self.make_linear(input_dim, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, output_dim, is_first=False, is_last=True)
        ]
        if self.out_activation == "sigmoid":
            layers += [nn.Sigmoid()]
        elif self.out_activation == "tanh":
            layers += [nn.Tanh()]
        elif self.out_activation == "hardtanh":
            layers += [nn.Hardtanh()]
        elif self.out_activation == "GELU":
            layers += [nn.GELU()]
        elif self.out_activation == "RELU":
            layers += [nn.ReLU()]
        else:
            raise NotImplementedError
        self.layers = nn.Sequential(*layers)

    def forward(self, x, split_size=100000):
        with contextlib.nullcontext():
            out = self.layers(x)
        return out

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = self._operations.Linear(dim_in, dim_out, bias=False, device=self._device, dtype=self._dtype)
        return layer

    def make_activation(self):
        if self.activation == "ReLU":
            return nn.ReLU(inplace=True)
        elif self.activation == "GELU":
            return nn.GELU()
        else:
            raise NotImplementedError