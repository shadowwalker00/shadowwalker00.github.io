---
title: GAN论文阅读笔记
date: 2019-01-03 16:03:16
author: 陈光皓
img: https://cdn-images-1.medium.com/max/1600/1*FpiLozEcc6-8RyiSTHjjIw.png
mathjax: false
tags: [cGAN]
categories: Deep Learning
---

# cGan

原文：《Conditional Generative Adversarial Nets》

## 原理

## Pytorch程序

+ 遍历batch

  ```python
  for i, (imgs, labels) in enumerate(dataloader)
  ```

+ 网络结构

+ ```python
  class Generator(nn.Module):
      # initializers
      def __init__(self):
          super(Generator, self).__init__()
          self.fc1_1 = nn.Linear(100, 256)
          self.fc1_1_bn = nn.BatchNorm1d(256)
          self.fc1_2 = nn.Linear(10, 256)
          self.fc1_2_bn = nn.BatchNorm1d(256)
          self.fc2 = nn.Linear(512, 512)
          self.fc2_bn = nn.BatchNorm1d(512)
          self.fc3 = nn.Linear(512, 1024)
          self.fc3_bn = nn.BatchNorm1d(1024)
          self.fc4 = nn.Linear(1024, H*W)
      # forward method
      def forward(self, input, label):
          x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
          y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
          x = torch.cat([x, y], 1)
          x = F.relu(self.fc2_bn(self.fc2(x)))
          x = F.relu(self.fc3_bn(self.fc3(x)))
          x = F.tanh(self.fc4(x))
          return x
  
  class Discriminator(nn.Module):
      # initializers
      def __init__(self):
          super(Discriminator, self).__init__()
          self.fc1_1 = nn.Linear(H*W, 1024)
          self.fc1_2 = nn.Linear(10, 1024)
          self.fc2 = nn.Linear(2048, 512)
          self.fc2_bn = nn.BatchNorm1d(512)
          self.fc3 = nn.Linear(512, 256)
          self.fc3_bn = nn.BatchNorm1d(256)
          self.fc4 = nn.Linear(256, 1)
      # forward method
      def forward(self, input, label):
          x = F.leaky_relu(self.fc1_1(input.view(input.size(0),-1)), 0.2)
          y = F.leaky_relu(self.fc1_2(label), 0.2)
          x = torch.cat([x, y], 1)
          x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
          x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
          x = F.sigmoid(self.fc4(x))
          return x
  ```
