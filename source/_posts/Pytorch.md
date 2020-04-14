---
title: Pytorch 用法
date: 2019-02-25 16:30:46
tags:
---

# 保存模型
```python
# 保存和加载整个模型
torch.save(model_object, 'model.pkl')
model = torch.load('model.pkl')

# 仅保存和加载模型参数(推荐使用)
torch.save(model_object.state_dict(), 'params.pkl')
model_object.load_state_dict(torch.load('params.pkl'))

```

<!--more-->

# 初始化

```python
print("1.使用另一个Conv层的权值")
q=torch.nn.Conv2d(2,2,3,padding=1) # 假设q代表一个训练好的卷积层
print(q.weight) # 可以看到q的权重和w是不同的
w.weight=q.weight # 把一个Conv层的权重赋值给另一个Conv层
print(w.weight)

# 第二种方法
print("2.使用来自Tensor的权值")
ones=torch.Tensor(np.ones([2,2,3,3])) # 先创建一个自定义权值的Tensor，这里为了方便将所有权值设为1
w.weight=torch.nn.Parameter(ones) # 把Tensor的值作为权值赋值给Conv层，这里需要先转为torch.nn.Parameter类型，否则将报错
print(w.weight)
```

# 自定义Dataloader

dataloader.py

```python
import torch
from torch.utils import data
import os
from PIL import Image
from torchvision import transforms

class Dataset(data.Dataset):
  # Dataloader for UTKFace

  def __init__(self, list_IDs, labels, transform=transforms.ToTensor()):             
        self.data_path = "/home/ghao/datasets/Face/UTKFace"
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
  def __len__(self):
        return len(self.list_IDs)
  def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        X = Image.open(os.path.join(self.data_path, ID))
        X = self.transform(X)
        y = self.labels[ID]
        return X, y
```

train.py

```python
transform = transforms.Compose([
			transforms.Resize(img_size),			
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
	with open(os.path.join(data_dir,"partition.pickle"), "rb") as f:
		partition = pickle.load(f)
	with open(os.path.join(data_dir,"ageLabel.pickle"), "rb") as f:
		agelabels = pickle.load(f)

	training_set = Dataset(partition['train'], agelabels, transform = transform)
	training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
```

# 随机数

## Numpy产生多维随机小数

```python
np.random.random((3,3))
```

## python产生一个随机数

```python
import random
print(random.randint(0,9))
```



