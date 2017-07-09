import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import ipdb

batch_size = 100
learning_rate = 0.001
num_epochs = 24

in_layer_size = 784
out_layer_size = 10
hidden_layer_sizes = [500]

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/mnist/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=download_ds)

test_dataset = dsets.MNIST(root='./data/mnist/',
                           train=False,
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def get_loss_weights(target, num_classes=2):
    """
    Gets loss weights by counting target classes examples

    To help with skewed datasets we need to weight the gradient.
    Each class need to have the same influence in gradient.
    So, each example will have weight equal to 1/num_classes * 1/class_count
    """

    weights = torch.zeros(target.data.size())

    # TODO: check if torch.unique is implemented
    for cls in range(num_classes):
        weights += 1 / (torch.sum(y == cls).data[0]) * (y == cls).type(torch.FloatTensor)

    weights /= num_classes
    return weights


# Feed-Forward Neural Network Model (N hidden layer)
class FFN(nn.Module):
    def __init__(self, in_layer_size, hidden_layer_sizes, out_layer_size):
        super(FFN, self).__init__()

        layer_sizes = [in_layer_size] + hidden_layer_sizes + [out_layer_size]
        self.layers = []

        for i in range(1, len(layer_sizes)):

            # Dont add activation to last layer for the sake of generality
            if i == (len(layer_sizes) - 1):
                layer = nn.Linear(layer_sizes[i-1], layer_sizes[i])

            else:
                layer = nn.Sequential(
                    nn.Linear(layer_sizes[i-1], layer_sizes[i]),
                    nn.Sigmoid())

            layer_name = "layer{}".format(i)
            setattr(self, layer_name, layer)
            self.layers.append(layer_name)
        return

    def forward(self, x):
        for layer in self.layers:
            x = getattr(self, layer)(x)
        return x


model = FFN(in_layer_size, hidden_layer_sizes, out_layer_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_func = nn.functional.binary_cross_entropy_with_logits

# Train the Model
for epoch in range(num_epochs):
    for i, (X, y) in enumerate(train_loader):
        # Convert torch tensor to Variable
        X = Variable(X)
        y = Variable(y)

        batch_weights = get_loss_weights(y)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = model(X)
        loss = loss_func(outputs, y, weights=batch_weights, size_avarage=False)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
