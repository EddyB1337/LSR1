from torchvision import datasets as dts
from torchvision import transforms


def get_data_mnist():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    traindt = dts.MNIST(
        root='data',
        train=True,
        transform=data_transform,
        download=True,
    )
    testdt = dts.MNIST(
        root='data',
        train=False,
        transform=data_transform
    )
    return traindt, testdt
