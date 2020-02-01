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
import threading
import time
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
        # I've made the following stuff use self.data_count rather than
        # len(data) to make it easier to exclude some of the input data, if
        # desired, by modifying self.data_count.
        self.data = None
        self.data_count = len(data)
        self.results = torch.LongTensor(self.data_count).to(device)
        i = 0
        skipped_size_mismatch = 0
        print("")
        for input_data, result in data:
            if i >= self.data_count:
                break
            msg = "Loading datasets onto the device: %.02f%% (%d/%d)\r" % (
                (float(i) / float(self.data_count)) * 100.0, i,
                self.data_count)
            print(msg, end="")
            # Once we know the shape of the input we can allocate the data
            # buffer.
            if self.data is None:
                data_shape = get_vector_of_tensor_size(self.data_count,
                    input_data.size())
                self.data = torch.zeros(data_shape, device=device)
            self.data[i] = input_data.to(device)
            self.results[i] = result
            i += 1
        print("")

    def __getitem__(self, index):
        return (self.data[index], self.results[index])

    def __len__(self):
        return self.data_count

class RandomDataset(torch.utils.data.Dataset):
    """ A dataset that holds a bunch of random junk in device memory.
    Compatible with the SimpleLoader object. """

    def __init__(self, data, device):
        """ Requires a PreloadDataset object, from which the structure of this
        data will be copied. """
        self.data = torch.rand(data.data.size(), device=device)
        self.results = torch.LongTensor(data.results.size()).to(device)
        self.results = torch.zeros(data.results.size(), dtype=torch.long).to(device)
        self.data_count = len(data)

    def __getitem__(self, index):
        return (self.data[index], self.results[index])

    def __len__(self):
        return self.data_count

class SimpleLoader(object):
    """ This class replaces PyTorch's loader, and simply wraps our
    PreloadDataset to return subsequent slices. """

    def __init__(self, dataset, batch_size):
        """ Expects a PreloadDataset. """
        self.dataset = dataset
        self.current_index = 0
        self.batch_size = batch_size

    def __next__(self):
        if (self.current_index + self.batch_size) >= len(self.dataset):
            raise StopIteration
        i = self.current_index
        data_slice = self.dataset.data[i : (i + self.batch_size)]
        result_slice = self.dataset.results[i : (i + self.batch_size)]
        self.current_index += self.batch_size
        return (data_slice, result_slice)

    def __iter__(self):
        return self

    def __len__(self):
        return int(int(len(self.dataset)) / int(self.batch_size))

class CompetitorNet(nn.Module):
    """ This class is just an arbitrary neural network, left untrained, and
    used to generate competing work. It also takes 32x32 single-channel inputs,
    but we just feed it random data during execution. """
    def __init__(self):
        super(CompetitorNet, self).__init__()
        # This network is similar to the MNIST one, but has a larger number of
        # larger fully connected layers to generate more work.
        extra_fc_layer_count = 4
        extra_fc_layer_size = 1024
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, extra_fc_layer_size)
        self.extra_fc_layers = torch.nn.ModuleList()
        for i in range(extra_fc_layer_count):
            self.extra_fc_layers.append(torch.nn.Linear(extra_fc_layer_size,
                extra_fc_layer_size))
        self.fc3 = nn.Linear(extra_fc_layer_size, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        for i in range(len(self.extra_fc_layers)):
            x = self.extra_fc_layers[i](x)
            x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

competitor_should_quit = False
def run_competitor(args, model, device, dataset, barrier):
    """ This function is intended to be run in its own thread, and will run the
    competing neural network in order to generate contention for the GPU. """
    global competitor_should_quit
    stream = torch.cuda.Stream(device)
    cu_mask_string = args.competitor_cu_mask
    if cu_mask_string is not None:
        if "set_cu_mask" not in dir(stream):
            print("Setting a stream's CU mask is not supported in this " +
                "build of PyTorch.")
            exit(1)
        stream.set_cu_mask(int(cu_mask_string, 16))
    iterations = 0
    # We'll use done_initializing to cause the competitor to warm up once, and
    # then wait for the main thread's to be ready to continue running.
    wait_once = True
    while not competitor_should_quit:
        # If necessary, the batch size could probably be increased further to
        # generate more work.
        loader = SimpleLoader(dataset, 128)
        with torch.no_grad():
            with torch.cuda.stream(stream):
                for (data, target) in loader:
                    if competitor_should_quit:
                        break
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    stream.synchronize()
                    if wait_once:
                        wait_once = False
                        barrier.wait()
        iterations += 1
    print("Competitor done. Ran %d iterations." % (iterations,))

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

def test(args, model, device, loader, epoch, times, stream, barrier):
    batch_index = 0
    # We'll wait for the competitor to have run a single iteration before
    # recording times and continuing our own iterations. This should hopefully
    # ensure both threads are "warmed up" and able to run when we're trying to
    # measure interference. wait_once shouldn't do anything if barrier is None,
    # implying there's no competitor.
    wait_once = True
    for (data, target) in loader:
        iteration_start = time.time()
        output = model(data)
        loss = F.nll_loss(output, target)
        if stream is not None:
            stream.synchronize()
        iteration_duration = time.time() - iteration_start
        # Ignore epoch 0, it may contain "warm-up" times at both the first and
        # last iteration (likely due to the last iteration's different batch
        # size.
        if (not wait_once) and (epoch > 1) and (len(times) < args.max_time_samples):
            times.append(iteration_duration)
        if batch_index % args.log_interval == 0:
            print("Epoch %d, batch %d/%d." % (epoch, batch_index, len(loader)))
        batch_index += 1
        if wait_once:
            # Wait for the competitor thread to be finished, so both threads
            # will be running at the same time.
            wait_once = False
            if (barrier is not None) and not barrier.broken:
                barrier.wait()
                barrier.abort()

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to run (default: 14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--time-output', default=None,
                        help='The name of a JSON file that will hold times ' +
                            'for individual training iterations.')
    parser.add_argument('--max-time-samples', default=100000,
                        help='If --time-output is set, this will be the ' +
                            'limit on the number of data points to output.')
    parser.add_argument('--stream-cu-mask', default=None,
        help="A hex string specifying the CU mask for the stream, if GPU " +
            "usage is enabled.")
    parser.add_argument("--enable-competitor", action='store_true',
            default=False, help="If set, run a competing thread.")
    parser.add_argument("--competitor-cu-mask", default=None,
        help="A hex string specifying the CU mask for the competitor's " +
            "stream. Only used if --enable-competitor is set.")
    args = parser.parse_args()

    if not args.no_cuda and not torch.cuda.is_available():
        print("Can't use CUDA without CUDA being available. Pass --no-cuda.")
        os.exit(1)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:1" if not args.no_cuda else "cpu")
    if not args.no_cuda:
        print("Running on device: " + torch.cuda.get_device_name(device))
    stream = None
    if not args.no_cuda:
        stream = torch.cuda.Stream(device)
        cu_mask_string = args.stream_cu_mask
        if (cu_mask_string is not None):
            if "set_cu_mask" not in dir(stream):
                print("Setting a stream's CU mask isn't supported in this " +
                    "build of PyTorch.")
                exit(1)
            stream.set_cu_mask(int(cu_mask_string, 16))

    # Buffer all data into device memory to avoid noise in measurements later.
    print("Loading datasets onto the device...")
    dataset = PreloadDataset(datasets.MNIST('./temp_data', train=True,
        download=True, transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])), device)
    competitor_dataset = RandomDataset(dataset, device)
    model = Net()
    model.load_state_dict(torch.load("./mnist_cnn.pt", map_location=device))
    model.to(device)
    model.eval()

    # Set up the competitor thread (but don't quite start it yet) if we're
    # running a competitor.
    competitor_thread = None
    competitor_model = None
    barrier = None
    if args.enable_competitor:
        if args.no_cuda:
            print("Running a competitor without CUDA isn't supported.")
            exit()
        print("Starting a competitor.")
        barrier = threading.Barrier(2)
        competitor_model = CompetitorNet().to(device)
        competitor_model.eval()
        competitor_thread = threading.Thread(target=run_competitor, args=(
            args, competitor_model, device, competitor_dataset, barrier))
        competitor_thread.start()

    if not args.no_cuda:
        tmp = float(torch.cuda.max_memory_allocated(device)) / (1024.0 *
            1024.0)
        print("Device memory usage: %.02f MB" % (tmp,))
    print("Done preparing.")

    times = []
    for epoch in range(1, args.epochs + 1):
        with torch.no_grad():
            if stream is not None:
                with torch.cuda.stream(stream):
                    loader = SimpleLoader(dataset, args.batch_size)
                    test(args, model, device, loader, epoch, times, stream,
                        barrier)
            else:
                loader = SimpleLoader(dataset, args.batch_size)
                test(args, model, device, loader, epoch, times, stream,
                    barrier)

    global competitor_should_quit
    competitor_should_quit = True
    if args.time_output is not None:
        with open(args.time_output, "w") as f:
            f.write(json.dumps(times))
    if competitor_thread is not None:
        competitor_thread.join()

if __name__ == '__main__':
    main()

