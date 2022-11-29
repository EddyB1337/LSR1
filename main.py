import torch
import torch.nn as nn
#import wandb
from loadings.getDataMnist import get_data_mnist
from models.cnn import CNN
from optimizer.lsr1 import LSR1
from loadings.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################
#           Now try mnist
###################################

# First get our data
traindt, testdt = get_data_mnist()

# Create a model, choose two numbers, its the size of our model
size = [3, 3]  # small modell
model = CNN(size).to(device)

# Define loss function
loss_func = nn.CrossEntropyLoss()
optimizer_LSR1 = LSR1(model.parameters(), trust_solver="OBS")

    # Train
    # print(sweep_id)
    # wandb.watch(model)
train(10, model, 256, optimizer_LSR1, traindt, testdt, loss_func)
'''sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'test_acc'},
    'parameters':
        {
            'batch_size': {'values': [128, 256, 512]},
            'tr_radius': {'max': 0.001, 'min': 0.0005},
            'history_size': {'values': [3, 5, 7, 9, 11, 13, 15, 17, 23, 29, 31, 37, 47, 64, 100, 128]},
            'trust_solver': {'values': ["OBS"]}
        }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="LSR1")


#

def test():
    run = wandb.init()
    # Set epochs
    epochs = 10

    # Set batch size
    batch_size = wandb.config.batch_size

    # Set hyperparametrs
    lr = 1
    max_iter = 20
    tolerance_grad = 1e-15
    tolerance_change = 1e-15
    tr_radius = wandb.config.tr_radius
    history_size = wandb.config.history_size
    mu = 0.75
    nu = 0.75
    alpha_S = 0
    newton_maxit = None
    cg_iter = None
    line_search_fn = "strong_wolfe"
    trust_solver = wandb.config.trust_solver

    # Define the optimizer
    optimizer_LSR1 = LSR1(model.parameters(), lr, max_iter, tolerance_grad, tolerance_change, tr_radius, history_size,
                          mu,
                          nu, alpha_S, newton_maxit, cg_iter, line_search_fn, trust_solver)

    # Train
    # print(sweep_id)
    wandb.watch(model)
    train(epochs, model, batch_size, optimizer_LSR1, traindt, testdt, loss_func)


wandb.agent(sweep_id, function=test)'''
