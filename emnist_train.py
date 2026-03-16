import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model_emnist import ModelEmnist

import cv2
import numpy as np
import random
from skimage.morphology import skeletonize

def transform_to_dots(image : torch.FloatTensor, min_dots=10, max_dots=20):
    image = image.numpy()[0]

    binary = (image > 0.5).astype(np.uint8)
    if binary.sum() == 0: return image
    
    skeleton = skeletonize(binary).astype(np.uint8)
    points = np.column_stack(np.where(skeleton > 0))

    num_dots = min(len(points), random.randint(min_dots, max_dots))
    if num_dots < 1: return np.zeros((28, 28))
    
    selected_indices = np.random.choice(len(points), num_dots, replace=False)
    selected_points = points[selected_indices]

    dot_image = np.zeros((28, 28), dtype=np.float32)
    for y, x in selected_points:
        jx = x + random.uniform(-0.5, 0.5)
        jy = y + random.uniform(-0.5, 0.5)

        cv2.circle(dot_image, (int(jx), int(jy)), 1, 1.0, -1, lineType=cv2.LINE_AA)

    if random.random() > 0.7:
        for _ in range(random.randint(1, 2)):
            rx, ry = random.randint(0, 27), random.randint(0, 27)
            cv2.circle(dot_image, (rx, ry), 1, 0.8, -1)

    kernel = np.ones((2,2), np.float32)
    dot_image = cv2.dilate(dot_image, kernel, iterations=1)
    dot_image = cv2.GaussianBlur(dot_image, (3, 3), 0)

 #   cv2.imwrite('emnist_train.png', dot_image.T * 255)
#    assert num_dots == 19

    return torch.FloatTensor(dot_image, device = device).unsqueeze(0)

data_dir = 'emnist/data'
model_path = 'emnist/model2.pt'

# Model and Training
batch_size=128 # input batch size for training (default: 64)
test_batch_size=1000 #input batch size for testing (default: 1000)
num_workers=0 # parallel data loading to speed things up
lr=1.0 #learning rate (default: 1.0)
gamma=0.7 #Learning rate step gamma (default: 0.7)
no_cuda=True #disables CUDA training (default: False)
seed=21 #random seed (default: 355)
log_interval=10 #how many batches to wait before logging training status (default: 10)
save_model=True #save the trained model (default: False)

use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")

print("Device:", device)

data_train = datasets.EMNIST(data_dir, split='balanced', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transform_to_dots,
                        transforms.Normalize((0.5,), (0.5,))
                    ]))

data_test = datasets.EMNIST(data_dir, split='balanced', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transform_to_dots,
                        transforms.Normalize((0.5,), (0.5,))
                    ]))

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(data_test, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

image_gen = iter(test_loader)
test_img, test_trg = next(image_gen)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('\r\tTrain epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='')
            
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\rTest epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def train_and_test(model, dl_train, dl_test, save_name=model_path, lr=lr, gamma=gamma, epochs=5):
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, dl_train, optimizer, epoch)
        test(model, device, dl_test, epoch)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), save_name)

print(data_train.classes)

model = ModelEmnist()

train_and_test(model, train_loader, test_loader)
