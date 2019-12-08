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
image = mnist_data.train_data[0].numpy()

# %% [markdown]
# アフィン変換をする  
# 参考 : https://qiita.com/secang0/items/7531e47305a199f02aae  
# https://qiita.com/secang0/items/1229212a37d8c9922901

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
#高さ3、幅4の画像を作る
height, width = 3,4

x_len=height
y_len=width
#mgridでx座標の行列、y座標の行列を作成
x, y = np.mgrid[:x_len,:y_len]

#dstackでx座標、y座標、1を組み合わせる
xy_after = np.dstack((x,y,np.ones((x_len, y_len))))
xy_after


# %%
#縦横に2倍拡大するアフィン変換
affin = np.matrix('2,0,0;0,2,0;0,0,1')

#逆行列
inv_affin = np.linalg.inv(affin)

#行列の掛け算をアインシュタイン和で計算
ref_xy = np.einsum('ijk,lk->ijl',xy_after,inv_affin)[...,:2]
ref_xy


# %%
img = plt.imread('tiger.jpg')[1390:1440,375:425]
img.shape


# %%
#参照する座標が四捨五入で計算されるため、100,450にするとインデックスエラーになる
height, width = 99, 149

x,y = np.mgrid[:height,:width]
xy_after = np.dstack((x,y,np.ones((height, width))))

#アフィン変換の行列を用意
#縦に2倍、横に3倍
affin = np.matrix('2,0,0;0,3,0;0,0,1')
inv_affin = np.linalg.inv(affin)

#参照する座標を計算
ref_xy = np.einsum('ijk,lk->ijl',xy_after,inv_affin)[...,:2]
ref_nearmost_xy = (ref_xy + 0.5).astype(int)
img_nearmost = img[ref_nearmost_xy[...,0],ref_nearmost_xy[...,1]]

img_show(img_nearmost)


# %%
# グレースケール
img_mid_v = np.max(img, axis = 2)/2 +np.min(img, axis = 2)/2
img_show(img_mid_v)

# %% [markdown]
# グレー用にする

# %%
#参照する座標が四捨五入で計算されるため、100,450にするとインデックスエラーになる
height, width = 99, 149

x,y = np.mgrid[:height,:width]
xy_after = np.dstack((x,y,np.ones((height, width))))

#アフィン変換の行列を用意
#縦に2倍、横に3倍
affin = np.matrix('2,0,0;0,3,0;0,0,1')
inv_affin = np.linalg.inv(affin)

#参照する座標を計算
ref_xy = np.einsum('ijk,lk->ijl',xy_after,inv_affin)[...,:2]
ref_nearmost_xy = (ref_xy + 0.5).astype(int)
img_nearmost = img_mid_v[ref_nearmost_xy[...,0],ref_nearmost_xy[...,1]]

img_show(img_nearmost)


# %%
#参照する座標が四捨五入で計算されるため、100,450にするとインデックスエラーになる
height, width =  55, 83

x,y = np.mgrid[:height,:width]
xy_after = np.dstack((x,y,np.ones((height, width))))

#アフィン変換の行列を用意
#縦に2倍、横に3倍
affin = np.matrix('2,0,0;0,3,0;0,0,1')
inv_affin = np.linalg.inv(affin)

#参照する座標を計算
ref_xy = np.einsum('ijk,lk->ijl',xy_after,inv_affin)[...,:2]
ref_nearmost_xy = (ref_xy + 0.5).astype(int)
img_nearmost = image[ref_nearmost_xy[...,0],ref_nearmost_xy[...,1]]

img_show(img_nearmost)


# %%
12/28


# %%
40/28


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
img_nearmost = image[ref_nearmost_xy[...,0],ref_nearmost_xy[...,1]]

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
# 訓練データとテストデータを用意
train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())

test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())


# %%
train = train_data.train_data.numpy()
height=12
width=40

train_new = np.empty((0,height,width), float)
for i in range(2):
    data_convert = nearmost_convert(train[i,:,:],height,width,0.45,1.45)
    data_convert = data_convert.reshape(1,height, width)
    train_new = np.concatenate([train_new, data_convert], axis=0)


# %%
test = test_data.test_data.numpy()
height=12
width=40

test_new = np.empty((0,height,width), float)
for i in range(test.shape[0]):
    data_convert = nearmost_convert(test[i,:,:],height,width,0.45,1.45)
    data_convert = data_convert.reshape(1,height, width)
    test_new = np.concatenate([test_new, data_convert], axis=0)


# %%
import pickle
with open("mnist_train_convert", "wb") as f:
    pickle.dump(train_new, f)


# %%
import pickle
with open("mnist_test_convert", "wb") as f:
    pickle.dump(test_new, f)


# %%



# %%



# %%



# %%



# %%



# %%



# %%
# 訓練データとテストデータを用意
train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_data,
                         batch_size=4,
                         shuffle=True)
test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data,
                         batch_size=4,
                         shuffle=False)


# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
net = Net()


# %%
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 50) # 入力層から隠れ層へ
        self.l2 = nn.Linear(50, 10) # 隠れ層から出力層へ
        
    def forward(self, x):
        x = x.view(-1, 28 * 28) # テンソルのリサイズ: (N, 1, 28, 28) --> (N, 784)
        x = self.l1(x)
        x = self.l2(x)
        return x
    
mlp = MLP()


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
correct = 0
total = 0
for data in test_loader:
    inputs, labels = data
    outputs = net(Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
print('Accuracy %d / %d = %f' % (correct, total, correct / total))


# %%
test_iter = iter(test_loader)
inputs, labels = test_iter.next()
outputs = net(Variable(inputs))
_, predicted = torch.max(outputs.data, 1)

plt.imshow(inputs[0].numpy().reshape(28, 28), cmap='gray')
print('Label:', predicted[0])


# %%


