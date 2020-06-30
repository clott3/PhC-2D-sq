#!/usr/bin/env python3

import torch.nn as nn
import torch
import math

class PhCNet(nn.Module):
    ''' This creates n separate branches that are initialized similarly
    for each of the n bands. We use a moduleDict to accept variable number of bands'''
    def __init__(self, fv, ks, num_bands, output = 529):
        super(PhCNet, self).__init__()

        self.output_dim = output # output size is 23*23 by default
        self.num_bands = num_bands
        self.fv = fv
        
        self.enc_block = nn.Sequential(
            nn.Conv2d(1, fv[0], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(fv[0]),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(fv[0], fv[1], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(fv[1]),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(fv[1], fv[2], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(fv[2]),
            nn.ReLU(),
            nn.MaxPool2d(4,4)
            )
        self.enc_linear = nn.Sequential(
            nn.Linear(fv[2] * 1 * 1, fv[3]),
            nn.ReLU(),
            nn.Linear(fv[3], fv[4]),
            nn.ReLU()
            )

        self.branches = nn.ModuleDict()
        for i in range(self.num_bands):
            branch = 'band'+str(i)
            self.branches.update({branch: self.fc_block()})

    def fc_block(self):
        return nn.Sequential(
            nn.Linear(self.fv[4], self.fv[5]),
            nn.ReLU(),
            nn.Linear(self.fv[5], self.fv[6]),
            nn.ReLU(),
            nn.Linear(self.fv[6], self.fv[7]),
            nn.ReLU(),
            nn.Linear(self.fv[8], self.output_dim)
            )

    def forward(self, x):
        x = self.enc_block(x)
        x = x.view(-1, self.fv[2] * 1 * 1) # flatten
        x = self.enc_linear(x)

        out = self.branches['band0'](x)
        if self.num_bands > 1:
            out = out.unsqueeze_(2) # forward pass for first band
            for i in range(1,self.num_bands):
                branch = 'band'+str(i)
                outband = self.branches[branch](x).unsqueeze_(2) # forward pass for each of the remaining bands
                out = torch.cat((out,outband),dim=2) # concatenate all bands
        return out ## This has dim (batchsize,output_dim, n) if n > 1 or (batchsize,output_dim) if n == 1
