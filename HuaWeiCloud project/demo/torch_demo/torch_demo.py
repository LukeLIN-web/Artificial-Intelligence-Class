# #################################################################
# Title: Demo for PyTorch Usage
# Author: Jiachen Li
# Time: 2021.5.12
# 
# Tips:
# - Pytorch version: 1.6.0
###################################################################

import torch
import numpy as np
import argparse, os, tqdm

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='', type=str, help='GPU IDs. E.g. 0,1,2. Do not add this argument if you are using CPU mode.')
args = parser.parse_args()

# Assign GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Simple demo network
class SimpleNN(torch.nn.Module):
    '''A simple fully-connected neural network demo.'''
    def __init__(self, input_size, emb_size, num_class):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.num_class = num_class

        # FC layers
        self.fc1 = torch.nn.Linear(self.input_size, self.emb_size)
        self.fc2 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.fc3 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.fc4 = torch.nn.Linear(self.emb_size, self.num_class)

        # Activation layers
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        x = self.activate(x)
        x = self.fc3(x)
        x = self.activate(x)
        x = self.fc4(x)
        return x

# Pytorch dataset definition
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file, mode='train'):
        assert mode == 'train' or mode == 'test' or mode == 'val', ('Mode should be [train|test|val]!')
        self.mode = mode
        data = np.load(npz_file)
        if mode == 'train':
            self.images = data['x_train'][:len(data['x_train'])//2]
            self.labels = data['y_train'][:len(data['y_train'])//2]
        elif mode == 'test':
            self.images = data['x_test']
            self.labels = data['y_test']
        elif mode == 'val':
            self.images = data['x_train'][len(data['x_train'])//2:]
            self.labels = data['y_train'][len(data['y_train'])//2:]

    def __getitem__(self, index):
        # Get current data
        img = self.images[index,:]
        label = self.labels[index]
        
        # Preprocessing
        img = torch.from_numpy(img).float()
        label = torch.as_tensor(label, dtype=torch.long).item()
        h, w = img.size(0), img.size(1)
        img = img / 255.0                  # [0,255] -> [0,1]
        img = img.view(h*w)              # Squeeze into a 1-dim vector
        img = img - torch.mean(img) # [0,1] -> [-1,1]
            
        return img, label

    def __len__(self):
        return self.labels.shape[0]

if __name__ == '__main__':
    # Prepare data
    data_path = '../mnist.npz'
    train_dataset = SimpleDataset(data_path, mode='train')
    test_dataset = SimpleDataset(data_path, mode='test')
    val_dataset = SimpleDataset(data_path, mode='val')

    # Create dataloader
    bsize = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=8, drop_last=True)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsize, shuffle=False, num_workers=8, drop_last=True)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsize, shuffle=False, num_workers=8, drop_last=True)

    # Create model
    model = SimpleNN(input_size=28*28, emb_size=128, num_class=10)
    if torch.cuda.is_available():
        model = model.cuda()

    # Set optimizer
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training
    epochs = 10
    for epoch in range(epochs):
        avg_loss = 0
        cnt = 0
        for i, (img, lbl) in enumerate(tqdm.tqdm(train_loader)):
            # Clear optimizer history info.
            opt.zero_grad()

            if torch.cuda.is_available():
                img = img.cuda()
                lbl = lbl.cuda()
            pred = model(img)

            # Loss function
            loss = torch.nn.functional.cross_entropy(pred, lbl)
            avg_loss += loss.item()
            cnt += 1

            # Get gradients
            loss.backward()
            
            # Update model
            opt.step()
        avg_loss /= cnt
        
        # Validation
        # torch.no_grad() does not record gradients, designed for inference.
        avg_acc = 0
        cnt = 0
        with torch.no_grad():
            model = model.eval() # Set model in evaluation mode.
            for i, (img, lbl) in enumerate(val_loader):
                if torch.cuda.is_available():
                    img = img.cuda()
                    lbl = lbl.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)
                nxor = ~torch.logical_xor(pred, lbl)
                n_correct = len(list(filter(lambda x: x == True, nxor)))
                avg_acc += n_correct / len(nxor)
                cnt += 1
            model = model.train() # Return to training mode.
        avg_acc /= cnt
        print('-----------------------------------------------------------------')
        print('Epoch:{}, avg_loss:{:.4f}, avg_val_acc:{:.2f}%'.format(epoch, avg_loss, avg_acc*100))
        print('-----------------------------------------------------------------')
    print('>>> Training finished. Start evaluation...')

    # Evaluation
    avg_acc = 0
    cnt = 0
    with torch.no_grad():
        model = model.eval() # Set model in evaluation mode.
        for i, (img, lbl) in enumerate(test_loader):
            if torch.cuda.is_available():
                img = img.cuda()
                lbl = lbl.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1)
            nxor = ~torch.logical_xor(pred, lbl)
            n_correct = len(list(filter(lambda x: x == True, nxor)))
            avg_acc += n_correct / len(nxor)
            cnt += 1
        model = model.train() # Return to training mode.
    avg_acc /= cnt
    print('Evaluation result:')
    print('Acc.:{:.2f}'.format(avg_acc*100))
