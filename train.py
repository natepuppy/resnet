from dataloader.nturgbd_dataset import *
import model.resnet as models
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import argparse
from signal import signal, SIGINT

PROJECT_DIRECTORY = os.getcwd() + '/'

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler=None, num_epochs=25):
    """Helper function for training and testing a model.
    args:
        model: A model that implements nn.Module
        train_loader: a pytorch dataloader containing the data to test the model
        test_loader: a pytorch dataloader containing the data to test the model
        criterion: criterion to determine training/testing loss
            (e.g. nn.CrossEntropyLoss)
        optimizer: Optimizer used to adjust model weights (e.g. optim.Adam)
        scheduler (optional): a pytorch learning rate scheduler.
        num_epochs: number of epochs to train/test the model.
    """
    global train_losses, train_accuracy, test_losses, test_accuracy, name #used for early exiting
    if name : print('Running model:',name)
    train_plot = 100 #plot every 100 training instances

    train_losses = []
    test_losses = []
    batch_losses = []

    train_accuracy = []
    test_accuracy = []
    batch_accuracy = []

    start = time.time()
    for epoch in range(num_epochs):
        loop = tqdm(total=len(train_loader), position=0, leave=True)
        model.train()  # Set model to training mode

        for batch, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            batch_losses.append(loss.item())

            #get average number of correct classifications
            accuracy = (preds == labels).float().mean()
            batch_accuracy.append(accuracy.item())

            optimizer.step()
            if scheduler : scheduler.step()

            if batch % train_plot == 0:
                train_losses.append(np.mean(batch_losses))
                batch_losses = []

                train_accuracy.append(np.mean(batch_accuracy))
                batch_accuracy = []

                loop.set_description("Epoch:{}/{}, loss:{:.4f}, accuracy:{:.3f}"
                    .format(epoch + 1, num_epochs, train_losses[-1], train_accuracy[-1]))
            loop.update()

        #Test
        if not test_loader : continue
        print('Testing...')
        model.eval()
        losses = []
        acc_list = []
        for x, y in test_loader:
            output = model(x.cuda())
            loss = criterion(output, y.cuda()).item()
            losses.append(loss)
            accuracy = (output.argmax(1) == y.cuda()).float().mean().item()
            acc_list.append(accuracy)

        test_losses.append(np.mean(losses))
        test_accuracy.append(np.mean(acc_list))
        print('TEST SET | loss: {:.4f}, accuracy:{:.3f}'.format(test_losses[-1], test_accuracy[-1]))
        print()

    end = time.time()
    loop.close()

    plot_loss_accuracy(train_losses, train_accuracy, scale=train_plot, title='Training', dest='results/' + name + '/')
    plot_loss_accuracy(test_losses, test_accuracy, title='Testing', x_label='Time (epochs)', dest='results/' + name + '/')
    print('Time elapsed:', end - start)

    return model

def plot_loss_accuracy(losses, accuracies, scale=1, title='', x_label='Time (iterations)', dest='results/'):
    """Plots and saves the lists of losses and accuracies on a single graph.
    *Note the time steps for losses and accuracies should be the same.

    args:
        losses: a list of all the losses
        accuracies: a list of accuracies (98% should be represented as .98)
        title: Title of the resuls graph
        x_label: A string representing the label for the x-axis.
        scale: How often each point was sampled (example: 10 means sampled
            every 10 instances).
    """
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    x = np.arange(len(losses)) * scale

    ax1.plot(x, losses, label='Loss', color='orange')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(x, accuracies, label='Accuracy')
    ax2.set_ylabel('Accuracy')

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    fig.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels)

    if not os.path.exists(dest):
        os.makedirs(dest)
    plt.savefig(dest + title + '.png')

def gpu_info():
    """Displays the following for each gpu: 'NAME (DEVICE_NUMBER) : CAPABILITY'
    """
    #Display any gpu warnings (happens automatically when calling torch.cuda)
    devices = [(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]

    #Display gpu information
    print()
    print('GPU INFO:{')
    visible = []
    for id, name in devices:
        capability = torch.cuda.get_device_capability(id)
        print('\t' + name + ' (' + str(id) + ') :', capability)
        if capability[0] >=3 and capability[1] >= 5:
            visible.append(id)
    print('}\n')
    print('Currently using:', devices[torch.cuda.current_device()][1])
    print('Cuda available:', torch.cuda.is_available())
    print("To hide gpu warning on this machine: 'export CUDA_VISIBLE_DEVICES=0'") #0

def main():
    """Trains and tests the model on the NTURGB+D dataset with the cross-subject
        protocol.
    """
    torch.cuda.set_device(0)
    train_dataset = NTU_RGB_D(PROJECT_DIRECTORY + 'datasets/raw/cross_subject/train/',
        filetype='pt', preprocess=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,pin_memory=True)

    test_dataset = NTU_RGB_D(PROJECT_DIRECTORY + 'datasets/raw/cross_subject/test/',
        filetype='pt', preprocess=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,pin_memory=True)

    model = models.resnet18(num_classes=train_dataset.num_classes)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5000, gamma=0.5) # Decay LR by a factor of 0.5 every 5000 iteration
    model = train_model(model, train_loader, test_loader, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)

def test():
    """Function used to test that everything is set up correctly.
        Possible Issues:
            * You need to generate the data (see dataloader/nturgbd_dataset for
              helper functions)
            * Your custom model has an error (like a shape mismatch)

        I also tend to run this function on the cpu to test different neural
        net architectures while my model is training on the gpu.
    """
    train_dataset = NTU_RGB_D(PROJECT_DIRECTORY + 'datasets/raw/cross_subject/train/', filetype='pt',
        preprocess=False)
    #train_dataset = NTU_RGB_D(PROJECT_DIRECTORY + '/datasets/raw/cross_subject/train/',
        #filetype='pt', preprocess=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1)

    model = models.resnet18(num_classes=train_dataset.num_classes, device='cpu')
    #model = model.cuda() #Uncomment to run on gpu
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)
    accuracy_list = []

    for batch, (inputs, labels) in enumerate(train_loader):
        image = inputs[0]
        #inputs, labels = inputs.cuda(), labels.cuda() #Uncomment to run on gpu

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)
        loss.backward()

        accuracy = (preds == labels).float().mean()
        accuracy_list.append(accuracy)

        if batch % 5 == 0:
            print('WORKS')
            #visualize(inputs[0], zoom=.5, media='video')
            exit()


def sig_handler(sig_num, frame):
    """Signal handler used for saving the results of train_model() after an early exit (ctrl-c)
    """
    plot_loss_accuracy(train_losses, train_accuracy, scale=100, title='Training', dest='results/' + name + '/')
    plot_loss_accuracy(test_losses, test_accuracy, title='Testing', x_label='Time (epochs)', dest='results/' + name + '/')
    exit()

if __name__ == '__main__':
    global train_losses, train_accuracy, test_losses, test_accuracy, name
    signal(SIGINT, sig_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=False, default='resnet18',
        help='Name of directory to store experiment results under')
    args = parser.parse_args()
    name = args.name
    train_losses = train_accuracy = test_losses = test_accuracy = []
    main()
