# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Study/Pytorch/pytorch_study'))
	print(os.getcwd())
except:
	pass

# %%
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pylab as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle


# %%
# データセットをダウンロード
mnist_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
data_loader = DataLoader(mnist_data,
                         batch_size=4,
                         shuffle=False)


# %%
data_iter = iter(data_loader)
images, labels = data_iter.next()

# matplotlibで1つ目のデータを可視化してみる
npimg = images[0].numpy()
npimg = npimg.reshape((28, 28))
plt.imshow(npimg, cmap='gray')
print('Label:', labels[0])


# %%
def img_show(img : np.ndarray, cmap = 'gray', vmin = 0, vmax = 255, interpolation = 'none') -> None:
    '''np.arrayを引数とし、画像を表示する。'''

    #dtypeをuint8にする
    img = np.clip(img,vmin,vmax).astype(np.uint8)

    #画像を表示
    plt.imshow(img, cmap = cmap, vmin = vmin, vmax = vmax, interpolation = interpolation)
    plt.show()
    plt.close()


# %%
test = mnist_data.train_data[0].numpy()


# %%
#参照する座標が四捨五入で計算されるため、100,450にするとインデックスエラーになる
height, width =  12, 40

x,y = np.mgrid[:height,:width]
xy_after = np.dstack((x,y,np.ones((height, width))))

#アフィン変換の行列を用意
#縦に2倍、横に3倍
affin = np.matrix('0.45,0,0;0,1.45,0;0,0,1')
inv_affin = np.linalg.inv(affin)

#参照する座標を計算
ref_xy = np.einsum('ijk,lk->ijl',xy_after,inv_affin)[...,:2]
ref_nearmost_xy = (ref_xy + 0.5).astype(int)
img_nearmost = test[ref_nearmost_xy[...,0],ref_nearmost_xy[...,1]]

img_show(img_nearmost)


# %%
def nearmost_convert(image, height_size, width_size, height_ratio, width_ratio):
    """参照する座標が四捨五入で計算されるため、height_sizeとratioを調整して、
    heightの値があるように
    image : numpy
    height_size : 縦のサイズ
    width_size : 横のサイズ
    height_ratio : 縦の圧縮率
    width_ratio : 横の圧縮率
    """
    x,y = np.mgrid[:height,:width]
    xy_after = np.dstack((x,y,np.ones((height_size, width_size))))
    
    #アフィン変換の行列を用意
    #縦に2倍、横に3倍
    affin = np.matrix('{},0,0;0,{},0;0,0,1'.format(height_ratio, width_ratio))
    inv_affin = np.linalg.inv(affin)
    
    #参照する座標を計算
    ref_xy = np.einsum('ijk,lk->ijl',xy_after,inv_affin)[...,:2]
    ref_nearmost_xy = (ref_xy + 0.5).astype(int)
    img_nearmost = image[ref_nearmost_xy[...,0],ref_nearmost_xy[...,1]]
    
    return  img_nearmost


# %%
img_show(nearmost_convert(test, 12, 40, 0.45, 1.45) )

# %% [markdown]
# # 訓練データとテストデータを用意
# # 超強引ver
# train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
# train_data.train_data = train_data.train_data[:,:20,:24].resize(60000,12,40)
# train_loader = DataLoader(train_data,
#                          batch_size=4,
#                          shuffle=True)
# test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
# test_data.test_data = test_data.test_data[:,:20,:24].resize(10000,12,40)
# test_loader = DataLoader(test_data,
#                          batch_size=4,
#                          shuffle=False)

# %%
# 訓練データとテストデータを用意
# アフィン変換ver
train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
with open('mnist_train_convert', 'rb') as f:
    train = pickle.load(f)

train_ = torch.tensor(train, dtype=torch.uint8)
train_data.train_data = train_

train_loader = DataLoader(train_data,
                         batch_size=4,
                         shuffle=True)


test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
with open('mnist_test_convert', 'rb') as f:
    test = pickle.load(f)
    
test_ = torch.tensor(test, dtype=torch.uint8)
test_data.test_data = test_
test_loader = DataLoader(test_data,
                         batch_size=4,
                         shuffle=False)


# %%
data_iter = iter(train_loader)
images, labels = data_iter.next()

# matplotlibで1つ目のデータを可視化してみる
npimg = images[0].numpy()
npimg = npimg.reshape((12, 40))
plt.imshow(npimg, cmap='gray')
print('Label:', labels[0])

# %% [markdown]
# # 訓練データとテストデータを用意
# # 正規ver
# train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
# train_loader = DataLoader(train_data,
#                          batch_size=4,
#                          shuffle=True)
# test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
# test_loader = DataLoader(test_data,
#                          batch_size=4,
#                          shuffle=False)

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,stride=1) # 12x40x32 -> 10x38x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,stride=1) # 10x38x64 -> 8x36x64 
        self.pool = nn.MaxPool2d(2, 2) # 8x36x64 -> 4x18x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(4 * 18 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 4 * 18 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
net = Net()

# %% [markdown]
# # 元ver
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(2,5),stride=1) # 12x40x32 -> 11x35x32
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(2,5),stride=1) # 11x39x64 -> 10x30x64 
#         self.pool = nn.MaxPool2d(2, 2) # 10x30x64 -> 5x15x64
#         self.dropout1 = nn.Dropout2d()
#         self.fc1 = nn.Linear(5 * 15 * 64, 128)
#         self.dropout2 = nn.Dropout2d()
#         self.fc2 = nn.Linear(128, 10)
# 
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.dropout1(x)
#         x = x.view(-1, 5 * 15 * 64)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x
#     
# net = Net()
# %% [markdown]
# class Net_normal(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
#         self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
#         self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
#         self.dropout1 = nn.Dropout2d()
#         self.fc1 = nn.Linear(12 * 12 * 64, 128)
#         self.dropout2 = nn.Dropout2d()
#         self.fc2 = nn.Linear(128, 10)
# 
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.dropout1(x)
#         x = x.view(-1, 12 * 12 * 64)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x
#     
# net = Net_normal()
# %% [markdown]
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Linear(12 * 40, 50) # 入力層から隠れ層へ
#         self.l2 = nn.Linear(50, 10) # 隠れ層から出力層へ
#         
#     def forward(self, x):
#         x = x.view(-1, 12 * 40) # テンソルのリサイズ: (N, 1, 28, 28) --> (N, 784)
#         x = self.l1(x)
#         x = self.l2(x)
#         return x
#     
# mlp = MLP()
# %% [markdown]
# class MLP_normal(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Linear(28 * 28, 50) # 入力層から隠れ層へ
#         self.l2 = nn.Linear(50, 10) # 隠れ層から出力層へ
#         
#     def forward(self, x):
#         x = x.view(-1, 28 * 28) # テンソルのリサイズ: (N, 1, 28, 28) --> (N, 784)
#         x = self.l1(x)
#         x = self.l2(x)
#         return x
#     
# mlp = MLP_normal()

# %%

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        # Variableに変換
        inputs, labels = Variable(inputs), Variable(labels)
        
        # 勾配情報をリセット
        optimizer.zero_grad()
        
        # 順伝播
        outputs = net(inputs)
        
        # コスト関数を使ってロスを計算する
        loss = criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        
        # パラメータの更新
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 5000 == 4999:
            print('%d %d loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
            
print('Finished Training')


# %%
inputs.shape


# %%
correct = 0
total = 0
for data in test_loader:
    inputs, labels = data
    outputs = net(Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
print('Accuracy %d / %d = %f' % (correct, total, correct.numpy() / total))


# %%
test_iter = iter(test_loader)
inputs, labels = test_iter.next()
outputs = net(Variable(inputs))
_, predicted = torch.max(outputs.data, 1)

plt.imshow(inputs[0].numpy().reshape(12, 40), cmap='gray')
print('Label:', predicted[0])


# %%


