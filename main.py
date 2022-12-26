import torch
import torch.nn as nn
from loadings.getDataMnist import get_data_mnist
from models.cnn import CNN
from optimizer.lsr1 import LSR1
from loadings.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# First get our data
traindt, testdt = get_data_mnist()

# Create a model, choose two numbers, the size of our model
size = [18, 16]
model = CNN(size).to(device)

#Define hyperparameters
tr_radius = 0.00075
memory_size = 3
mu = 0.75
nu = 0.75
line_search_fn = "strong_wolfe"         #either 'strong_wolfe' or None
lr = 1                                  #set line_search_fn = None to use learning rate
trust_solver = "OBS"                    #either 'OBS' or 'Cauchy_Point_Calculation' or 'Steihaug_cg'
epochs = 10
batch_size = 256

# Define loss function
loss_func = nn.CrossEntropyLoss()

#Define optimizer
optimizer_LSR1 = LSR1(model.parameters(), trust_solver=trust_solver, nu=nu, mu=mu, memory_size=memory_size,
                      tr_radius=tr_radius, line_search_fn=line_search_fn, lr=lr)

# Train
train(epochs, model, batch_size, optimizer_LSR1, traindt, testdt, loss_func)


