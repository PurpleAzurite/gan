#  MIT License
#
#  Copyright (c) 2022 Misagh Asgari
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

import torch.nn as nn

# A Gan consists of two networks, a discriminator that determines whether an image is fake or real
# and a generator that generates new data by factoring in some noise

# gan() -> discriminator, generator


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(          # parameters:
            nn.Linear(img_dim, 128),        # dimensions, no. nodes
            nn.LeakyReLU(0.1),              # slope
            nn.Linear(128, 1),              # ~, fake/real
            nn.Sigmoid(),                   # Ensures last layer output is between 0 and 1
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):     # dimensions of the noise
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),                      # makes sure the output is -1 to 1
        )

    def forward(self, x):
        return self.gen(x)


def gan(image_dim, z_dim) -> (Discriminator, Generator):
    return Discriminator(image_dim), Generator(z_dim, image_dim)
