# This script was copied mnist/main.py in https://github.com/pytorch/examples.
# It was modified to record timing information to a separate file, and to use
# a different file to hold downloaded data.
#
# Additionally, it uses cuda:1 rather than cuda:0 (if GPU use is enabled)
# because my system has two GPUs and GPU 0 has a display hooked up.
from __future__ import print_function
import argparse
import json
import os
import time

# We'll do the following stuff before pytorch is even imported, to make sure
# that pytorch's queues use the default CU mask.
import rocm_control
cu_mask_string = os.environ.get("HSA_DEFAULT_CU_MASK")
if cu_mask_string is not None:
    cu_mask = int(cu_mask_string, 16)
    rocm_control.set_default_cu_mask(cu_mask)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def get_vector_of_tensor_size(n, entry_size):
    """ Returns the torch.Size for a tensor containing n entries, and each
    entry has a size given by entry_size. (There may be a better way to do
    this, but I don't know pytorch well enough yet.) """
    tmp = [n]
    for s in entry_size:
        tmp.append(s)
    return torch.Size(tmp)

class PreloadDataset(torch.utils.data.Dataset):
    """ A dataset class that loads another dataset and buffers it all in
    memory. """

    def __init__(self, data, device):
        """ Requires an existing dataset and a device on which to create the
        buffer. """
        self.data = None
        self.data_count = len(data)
        self.results = torch.LongTensor(len(data)).to(device)
        i = 0
        skipped_size_mismatch = 0
        for input_data, result in data:
            # Once we know the shape of the input we can allocate the data
            # buffer.
            if self.data is None:
                data_shape = get_vector_of_tensor_size(len(data),
                    input_data.size())
                self.data = torch.zeros(data_shape, device=device)
            self.data[i] = input_data.to(device)
            self.results[i] = result
            i += 1

    def __getitem__(self, index):
        return (self.data[index], self.results[index])

    def __len__(self):
        return self.data_count

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, times):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        iteration_start = time.time()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        iteration_duration = time.time() - iteration_start
        # Ignore epoch 0, it contains "warm-up" times at both the first and
        # last iteration (likely due to the last iteration's different batch
        # size.
        if (epoch > 1) and (len(times) < args.max_time_samples):
            times.append(iteration_duration)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--time-output', default=None,
                        help='The name of a JSON file that will hold times ' +
                            'for individual training iterations.')
    parser.add_argument('--max-time-samples', default=100000,
                        help='If --time-output is set, this will be the ' +
                            'limit on the number of data points to output.')
    args = parser.parse_args()

    if not args.no_cuda and not torch.cuda.is_available():
        print("Can't use CUDA without CUDA being available. Pass --no-cuda.")
        os.exit(1)

    torch.manual_seed(args.seed)

    device = torch.device("cuda:1" if not args.no_cuda else "cpu")

    stream = None
    if not args.no_cuda:
        stream = torch.cuda.Stream(device=device)

    # Buffer all data into device memory to avoid noise in measurements later.
    print("Loading datasets onto the device...")
    train_dataset = PreloadDataset(datasets.MNIST('./temp_data', train=True,
        download=True, transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])), device)
    test_dataset = PreloadDataset(datasets.MNIST('./temp_data', train=False,
        transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])), device)
    print("Done loading datasets onto the device.")

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=True)
    model = Net().to(device)
    if not args.no_cuda:
        tmp = float(torch.cuda.max_memory_allocated(device)) / (1024.0 *
            1024.0)
        print("Device memory usage: %.02f MB" % (tmp,))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    times = []
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        # Use a user-specified stream if we're using the GPU.
        if not args.no_cuda:
            with torch.cuda.stream(stream):
                train(args, model, device, train_loader, optimizer, epoch, times)
        else:
            train(args, model, device, train_loader, optimizer, epoch, times)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.time_output is not None:
        with open(args.time_output, "w") as f:
            f.write(json.dumps(times))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
