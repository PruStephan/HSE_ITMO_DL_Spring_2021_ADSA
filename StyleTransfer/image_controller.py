from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


def load_image(path, image_size):
    loader = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor()])
    device = torch.device("cpu")
    image = Image.open(path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def show_image(tensor, title):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    plt.title(title)
    plt.pause(0.001)
