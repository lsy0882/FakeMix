import pandas as pd 
import numpy as np 
import os 
import torch
import sklearn
from tqdm.auto import tqdm
import datetime
import argparse
import random
import collections
import json
import torchvision.models as models
import torch.nn.functional as F



class SmdaNet_video(torch.nn.Module):
    # input size = torch.Size([batch, 2, 25, 3, 224, 224])
    # input 2개의 이미지를 받아서 crossattention을 적용한 후 fc layer를 거쳐 class를 분류하는 모델
    def __init__(self, input_shape, num_classes, backbone="resnet50", dropout=0.5):
        # print("input_shape: ", input_shape)

        super(SmdaNet_video, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.dropout = dropout
        self.conv1 = torch.nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # test backbone: resnet50
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = torch.nn.Linear(2048, 512)

        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.5)
        self.fc = torch.nn.Linear(512, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # x: torch.Size([batch, 2, 3, 224, 224])
        batch_size = x.shape[0]
        num_clip = x.shape[1]
        num_channel = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]
        
        # pre_x: query, next_x: key, value
        pre_x = x[:, 0, :, :, :]
        next_x = x[:, 1, :, :, :]

        # x: torch.Size([batch,3, 224, 224]) -> torch.Size([batch,512])
        pre_x = self.backbone(pre_x)
        next_x = self.backbone(next_x)

        # 1d conv

        # x: torch.Size([batch,512]) -> torch.Size([batch, 1, 512])
        pre_x = pre_x.unsqueeze(1)
        next_x = next_x.unsqueeze(1)

        # pre_x, _ = self.cross_attention(pre_x, next_x, next_x)
        next_x, _ = self.cross_attention(next_x, pre_x, pre_x)

        # x: torch.Size([batch, 1, 512]) -> torch.Size([batch, 512])
        x_rep = next_x.squeeze(1)

        # x: torch.Size([batch, 512]) -> torch.Size([batch, num_classes])
        x = self.fc(x_rep)
        # x = self.softmax(x)

        return x