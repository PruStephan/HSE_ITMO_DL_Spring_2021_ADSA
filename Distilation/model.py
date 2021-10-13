from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch
import time


def download_dataset(train, save_path: str):
    if train:
        transformers = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(32, 32), padding=4)]
    else:
        transformers = []
    transformers.append(transforms.ToTensor()),
    transformers.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return torchvision.datasets.CIFAR10(
        root=save_path, train=train, download=True,
        transform=transforms.Compose(transformers)
    )


def add_conv_layer(channels):
    return nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)


class SimpleBlock(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)

        self.conv1 = add_conv_layer(channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = add_conv_layer(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.bn2(self.conv2(c1))
        return F.relu(x + c2)


class DownBlock(nn.Module):
    def __init__(self, in_chan):
        super(DownBlock, self).__init__()
        self.out_chan = in_chan * 2
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(in_chan, self.out_chan, kernel_size=3, stride=2, padding=1)
        self.conv2 = add_conv_layer(self.out_chan)
        self.bn2 = nn.BatchNorm2d(self.out_chan)
        # noinspection PyTypeChecker
        self.conv_down = nn.Conv2d(in_chan, self.out_chan, kernel_size=1, stride=2, bias=False)
        self.bn_down = nn.BatchNorm2d(self.out_chan)

        self.bn_down.weight.data.fill_(1)
        self.bn_down.bias.data.fill_(0)

        self.bn1 = nn.BatchNorm2d(self.out_chan)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.bn2(self.conv2(c1))
        down = self.bn_down(self.conv_down(x))
        return F.relu(down + c2)


class ResNet(nn.Module):
    def __init__(self, n):
        nn.Module.__init__(self)

        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.seq32_32 = nn.Sequential(
            *[SimpleBlock(16) for _ in range(n)]
        )
        self.seq16_16 = nn.Sequential(
            *[DownBlock(16) if i == 0 else SimpleBlock(32) for i in range(n)]
        )
        self.seq8_8 = nn.Sequential(
            *[DownBlock(32) if i == 0 else SimpleBlock(64) for i in range(n)]
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        s32_32 = self.seq32_32(x)
        s16_16 = self.seq16_16(s32_32)
        s8_8 = self.seq8_8(s16_16)

        features = F.avg_pool2d(s8_8, (8, 8))
        flat = features.view(features.size()[0], -1)

        return self.fc(flat)


def accuracy(model, test_dataset, batch_size):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X).argmax(dim=1)
            if y == y_hat:
                total_correct += 1

    return total_correct / len(test_dataset)


def train_model(model, epochs,
                train_dataset, test_dataset,
                teacher=None, alpha=0.5,
                train_batch_size=128, test_batch_size=128,
                epochs_passed=0):
    if teacher is not None:
        teacher.eval()

    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    train_start_time = time.time()
    accs = []
    acc = accuracy(model, test_dataset, batch_size=test_batch_size)
    if epochs_passed == 0:
        accs.append(acc)
    print("Initial acc = {0}".format(acc))

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        for X, y in train_loader:
            opt.zero_grad()
            y_hat = model(X)

            target_loss = F.cross_entropy(y_hat, y)
            if teacher is None:
                total_loss = target_loss
            else:
                y_hat_teacher = teacher(X)
                dist_loss = F.mse_loss(y_hat, y_hat_teacher)
                total_loss = alpha * target_loss + (1 - alpha) * dist_loss

            total_loss.backward()
            opt.step()

        acc = accuracy(model, test_dataset, batch_size=test_batch_size)
        accs.append(acc)

        cur_time = time.time()
        epoch_time_spent = int(cur_time - epoch_start_time)
        total_time_spent = int(cur_time - train_start_time)
        print(
            "Epochs passed = {0}, acc = {1}, seconds per epoch = {2}, total seconds elapsed = {3}".format(
                epochs_passed + epoch + 1, acc, epoch_time_spent, total_time_spent
            )
        )

    return accs
