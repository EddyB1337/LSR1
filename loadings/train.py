#import wandb
from torch.autograd import Variable
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses, test_losses = [], []
steps = 0
running_loss = 0
print_every = 10


def train(num_epochs, cnn, batch_size, optimizer, train_data, test_data, loss_func):
    num_work = 4
    if not torch.cuda.is_available():
        num_work = 0
    #wandb.init()
    loaders = {
        'train': torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=num_work),

        'test': torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=num_work),
    }
    #wandb.watch(cnn, loss_func, log="all", log_freq=10)
    running_loss = 0
    n = len(loaders['train'])
    m = len(loaders['test'])
    cnn.train()
    acc_list = []

    # Train the model
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
            #wandb.log({
            #    'epoch': epoch,
            #    'train_loss': running_loss / (i + 1),
            #})
            '''if (i + 1) % 10 == 0:
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
                print(f"Epoch, Steps [{epoch + 1}/{num_epochs}, {i + 1}/{n}], .. "
                      f"Train loss: {running_loss / n:.3f}.. "
                      f"Test loss: {test_loss / m:.3f}.. "
                      f"Test accuracy: {accuracy / m:.3f}")
                wandb.log({
                    'epoch': epoch,
                    'test_loss': test_loss/m,
                    'test_acc': accuracy/m
                })
        running_loss = 0'''
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
        acc_list.append(accuracy/m)
        print(f"Epoch, Steps [{epoch + 1}/{num_epochs}, {i + 1}/{n}], .. "
              f"Train loss: {running_loss / n:.3f}.. "
              f"Test loss: {test_loss / m:.3f}.. "
              f"Test accuracy: {accuracy / m:.3f}")
        #wandb.log({
        #    'epoch': epoch,
        #    'test_loss': test_loss / m,
        #    'test_acc': accuracy / m
        #})
        running_loss = 0
    return acc_list
