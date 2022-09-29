import torch
import torch.nn as nn

from loadings.getDataMnist import get_data_mnist
from models.cnn import CNN
from optimizer.lsr1 import LSR1
from loadings.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First get our data
traindt, testdt = get_data_mnist()

# Create a model, choose two numbers, its the size of our model
size = [3, 3] #small modell
model = CNN(size)

# Define loss function
loss_func = nn.CrossEntropyLoss()

# Set Epochs
epochs = 100

# Set batch size
batch_size = 256

# Set hyperparametrs
lr = 1
max_iter = 20
tolerance_grad = 1e-15
tolerance_change = 1e-15
tr_radius = 0.00075
history_size = 3
mu = 0.75
nu = 0.75
alpha_S = 0
newton_maxit = None
cg_iter = None
line_search_fn = "strong_wolfe"
trust_solver = "Steihaug_cg"

#Define the optimizer
optimizer_LSR1 = LSR1(model.parameters(), lr, max_iter, tolerance_grad, tolerance_change, tr_radius, history_size, mu,
                      nu, alpha_S, newton_maxit, cg_iter, line_search_fn, trust_solver)

#Train
train(epochs, model, batch_size, optimizer_LSR1, traindt, testdt, loss_func)
