import torch.nn as nn

__all__ = ['SharedMLP']


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1, device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            operations = nn
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            if dim == 1:
                layers.append(operations.Conv1d(in_channels, oc, 1, device=device, dtype=dtype))
            elif dim == 2:
                layers.append(operations.Conv2d(in_channels, oc, 1, device=device, dtype=dtype))
            else:
                raise ValueError(f"Unsupported dim={dim}")
            # InstanceNorm equivalent: GroupNorm with num_groups == num_channels
            layers.append(operations.GroupNorm(oc, oc, affine=False, device=device, dtype=dtype))
            layers.append(nn.ReLU(True))
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)
