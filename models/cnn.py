import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_input_sizes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_input_sizes[0],
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input_sizes[0], num_input_sizes[1], 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(3)
        )  # fully connected layer, output 10 classes
        self.out = nn.Linear(num_input_sizes[1] * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization
