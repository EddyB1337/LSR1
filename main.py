import torch
import torch.nn as nn

from getData.getDataMnist import get_data_mnist
from models.cnn import CNN
from optimizer.lsr1 import LSR1
from train.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traindt, testdt = get_data_mnist()

loss_func = nn.CrossEntropyLoss()
model = CNN()
model.eval()
optimizer_LSR1 = LSR1(model.parameters())
train(100, model, 256, optimizer_LSR1, traindt, testdt, loss_func)
