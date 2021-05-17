import glob
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

def make_dataset():
    x = []
    y = []

    not_trop_files = []
    for file in glob.glob("/media/matthew/Bigspace/ADS/not_troponin_train/*.png"):
        not_trop_files.append(file)
    trop_files = []
    for file in glob.glob("/media/matthew/Bigspace/ADS/troponin_train/*.png"):
        trop_files.append(file)

    for not_trop_file in tqdm(not_trop_files):#[:10000]):
        img = cv2.imread(not_trop_file)
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #flattened_image = image.flatten()
        #flattened_image = flattened_image.tolist()
        # print(len(flattened_image))
        #if len(flattened_image) != 67500:
        #    continue

        #x.append(flattened_image)
        image = np.array(image)
        image = np.mean(image, axis=2, dtype=int)
        image = np.expand_dims(image, axis=2)
        x.append(image)
        y.append(0)

    for trop_file in tqdm(trop_files):
        img = cv2.imread(trop_file)
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #flattened_image = image.flatten()
        #flattened_image = flattened_image.tolist()
        #if len(flattened_image) != 67500:
        #    continue

        #x.append(flattened_image)
        image = np.array(image)
        image = np.mean(image, axis=2, dtype=int)
        image = np.expand_dims(image, axis=2)
        x.append(image)
        y.append(1)

    x = np.array(x)
    y = np.array(y)
    print("x shape", x.shape)
    print("y shape", y.shape)

    return x, y

x, y = make_dataset()
x = x.transpose(0,3,1,2)
x = x/255
x = torch.from_numpy(x).double()
y = torch.from_numpy(y)#.double()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

train_set = []
test_set = []
for i in range(len(x_train)):
    train_set.append([x_train[i], y_train[i]])
for i in range(len(x_test)):
    test_set.append([x_test[i], y_test[i]])

trainloader = DataLoader(train_set, batch_size=10,
                        shuffle=True)

testloader = DataLoader(test_set, batch_size=100)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        self.fc1 = nn.Linear(in_features=64*15*15, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=2)

    def forward(self, t):
        # input layer
        t = t

        # hidden conv layer
        t = self.conv1(t)
        t = F.relu(t) #rectifier activation function
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        # hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # hidden conv layer
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # hidden dense layer
        t = t.reshape(-1,64*15*15)
        t = self.fc1(t)
        t = F.relu(t)

        # hidden dense layer
        t = self.fc2(t)
        t = F.relu(t)

        # output layer (dense)
        t = self.out(t)
        #t = F.sigmoid(t)

        return t

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

torch.set_grad_enabled(True) # start training

network = Network().double()
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(2):  # all batches in training set in one epoch, multiple epochs

    total_loss = 0
    total_correct = 0

    for idx, batch in tqdm(enumerate(trainloader)):                # Get batch
        images, labels = batch

        preds = network(images)               # Pass batch to network
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()                 # need to reset gradients to zero after each batch
        loss.backward()                       # Calculate Gradients
        optimizer.step()                      # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(
        "epoch", epoch,
        "total_correct:", total_correct,
        "loss:", total_loss
    )

torch.set_grad_enabled(False) # start training
total_correct = 0
for idx, batch in tqdm(enumerate(testloader)):
        images, labels = batch

        preds = network(images)               # Pass batch to network

        total_correct += get_num_correct(preds, labels)

print(total_correct, total_correct/708)
