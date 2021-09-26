import attack_model as am
import mnist_model as mm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]

def run_model(model, device, test_loader, epsilon, attack=False):
    correct = 0
    adv_examples = []

    for x in test_loader:
        data, target = x[0], x[1]

        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)

        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        if attack:
            perturbed_data = am.fgsm_attack(data, epsilon, data_grad)
        else:
            perturbed_data = data
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples

def initialize_model():

    seed = 5
    save_model = True
    gamma = 0.7
    epochs = 14
    lr = 1.
    device = torch.device("cpu")
    torch.manual_seed(seed)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1)

    model = mm.MNISTModel().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        mm.train(model, device, train_loader, optimizer, epoch)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "saved_model/mnist_cnn.pt")
    return model, device, test_loader

def test_attack(pretrain):
    accuracies = []
    examples = []
    if pretrain:
        model, device, test_loader = mm.generate_mnist_model()
    else:
        model, device, test_loader = initialize_model()
    model.eval()

    for eps in epsilons:
        acc, ex = run_model(model, device, test_loader, eps, True)
        accuracies.append(acc)
        examples.append(ex)

    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("Real: {}\nPredicted: {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_attack(True)
