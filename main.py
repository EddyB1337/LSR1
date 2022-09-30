import torch
import torch.nn as nn
import wandb
from loadings.getDataMnist import get_data_mnist
from models.cnn import CNN
from optimizer.lsr1 import LSR1
from loadings.train import train, train_f, print_path, print_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rosenbrock function
rosenbrock = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + 3

# start point
n_iter = 100
xy_init = torch.zeros(2).to(device)
xy_init = torch.tensor([-2.0, 3.5]).to(device)
xy_t = torch.tensor(xy_init, requires_grad=True).to(device)

# optimizer
optimizer_LSR1 = LSR1([xy_t], history_size=2, max_iter=100, tr_radius=5, mu=0, nu=0, trust_solver="Steihaug_cg")

# train rosenbrock minimum
path = train_f(xy_t, n_iter, optimizer_LSR1, rosenbrock)
print(path[-1])
print(rosenbrock(path[-1]))
minimum = 3

# minimum
print_path(path)
print_loss(rosenbrock(path.T), minimum)
# print(A)

###################################
#           Now try mnist
###################################

# First get our data
traindt, testdt = get_data_mnist()

# Create a model, choose two numbers, its the size of our model
size = [3, 3]  # small modell
model = CNN(size)

# Define loss function
loss_func = nn.CrossEntropyLoss()

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'test_acc'},
    'parameters':
        {
            'batch_size': {'values': [16, 32, 64, 128, 256, 512]},
            'tr_radius': {'max': 0.1, 'min': 0.000001},
            'history_size': {'values': [3, 5, 7, 9, 11, 13, 15, 17, 23, 29, 31, 37, 47, 64, 100, 128]},
            'trust_solver': {'values': ["OBS", "Steihaug_cg", "Cauchy_Point_Calculation"]}
        }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="LSR1")

'''hyp = dict(
    batch_size=16,
    tr_radius=0.001,
    history_size=3,
    trust_solver="OBS",
)'''


#

def test():
    run = wandb.init()
    # Set epochs
    epochs = 100

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
    # wandb.watch(model)
    train(epochs, model, batch_size, optimizer_LSR1, traindt, testdt, loss_func)


wandb.agent(sweep_id, function=test)
