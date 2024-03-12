# --------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------

from typing import Type
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 act: Type[nn.Module],
                 ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(n, k), act())
            for n, k in zip([input_dim] + h, [hidden_dim] * num_layers)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)
