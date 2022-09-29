from torch.autograd import Variable
import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses, test_losses = [], []
steps = 0
running_loss = 0
print_every = 10


def train(num_epochs, cnn, batch_size, optimizer, train_data, test_data, loss_func):
    loaders = {
        'train': torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True),

        'test': torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            shuffle=True),
    }
    running_loss = 0
    n = len(loaders['train'])
    m = len(loaders['test'])
    cnn.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(tqdm(loaders['train'])):
            images, labels = images.to(device), labels.to(device)

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward(retain_graph=True)

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                output = cnn(b_x)[0]
                loss = loss_func(output, b_y)
                if loss.requires_grad:
                    loss.backward()
                return loss

            optimizer.step(closure=closure)
            running_loss += loss.item()
            optimizer.zero_grad()

            if (i + 1) % 10 == 0:
                test_loss = 0
                accuracy = 0
                cnn.eval()

                with torch.no_grad():
                    for inputs, labelss in loaders['test']:
                        inputs, labelss = inputs.to(device), labelss.to(device)
                        b_xx = Variable(inputs)  # batch x
                        b_yy = Variable(labelss)  # batch y
                        logps = cnn(b_xx)[0]
                        batch_loss = loss_func(logps, b_yy)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labelss.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / n)
                test_losses.append(test_loss / m)
                print(f"Epoch, Steps [{epoch + 1}/{num_epochs}, {i+1}/{len(loaders['train'])}], .. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / m:.3f}.. "
                      f"Test accuracy: {accuracy / m:.3f}")
                running_loss = 0


def train_f(xy_init, n_iter, optimizer, loss_func):
    """Run optimization finding the minimum of the Rosenbrock function.
      From https://github.com/jankrepl/mildlyoverfitted/blob/master/mini_tutorials/custom_optimizer_in_pytorch/src.py
    Parameters
    ----------
    xy_init : tuple
        Two floats representing the x resp. y coordinates.
    optimizer_class : object
        Optimizer class.
    n_iter : int
        Number of iterations to run the optimization for.
    optimizer : An torch optimizer.
    Returns
    -------
    path : np.ndarray
        2D array of shape `(n_iter + 1, 2)`. Where the rows represent the
        iteration and the columns represent the x resp. y coordinates.
    """

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init.cpu().detach().numpy()
    xy_t = xy_init
    for i in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()
        loss = loss_func(xy_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_t, 1.0)

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            loss = loss_func(xy_t)
            if loss.requires_grad:
                loss.backward()
            return loss

        optimizer.step(closure)

        path[i, :] = xy_t.cpu().detach().numpy()

    return path
