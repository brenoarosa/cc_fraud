import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from tensorboard import TBLogger

batch_size = 2**10
learning_rate = 0.001
num_epochs = 10

in_layer_size = 30
out_layer_size = 1
hidden_layer_sizes = [50, 50, 50]

X_train = torch.load("./data/X_train.p")
y_train = torch.load("./data/y_train.p")
X_test = torch.load("./data/X_test.p")
y_test = torch.load("./data/y_test.p")

log_path = "./log/ff_50_50_50_adam_1e-3"
logger = TBLogger(log_path)

# MNIST Dataset
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def get_loss_weights(target, num_classes=2, eps=1e-08):
    """
    Gets loss weights by counting target classes examples

    To help with skewed datasets we need to weight the gradient.
    Each class need to have the same influence in gradient.
    So, each example will have weight equal to 1/num_classes * 1/class_count

    Usage:
    >>> tensor = Variable(torch.FloatTensor([1, 0, 0, 0]))
    >>> get_loss_weights(tensor)
        0.5000
        0.1667
        0.1667
        0.1667
        [torch.FloatTensor of size 4]
    """

    target = target.data.long()

    target_onehot = torch.FloatTensor(target.size(0), num_classes)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)

    cls_weight = (target_onehot.sum(dim=0) + eps).pow(-1)
    weights = torch.mm(target_onehot, cls_weight.t())

    # count classes present in target
    num_classes = (cls_weight <= 1).float().sum()
    weights /= num_classes
    return weights


# Feed-Forward Neural Network Model (N hidden layer)
class FFN(nn.Module):
    def __init__(self, in_layer_size, hidden_layer_sizes, out_layer_size):
        super(FFN, self).__init__()

        self.bn = nn.BatchNorm1d(in_layer_size)

        self.layers = []
        layer_sizes = [in_layer_size] + hidden_layer_sizes + [out_layer_size]

        # An proper implementation shoulden't add activation to the last
        # layer as it could be computed in loss func for classifications
        # and it isn't used in regressions
        for i in range(1, len(layer_sizes)):
            layer = nn.Sequential(
                nn.Linear(layer_sizes[i-1], layer_sizes[i]),
                nn.Sigmoid())

            layer_name = "layer{}".format(i)
            setattr(self, layer_name, layer)
            self.layers.append(layer_name)
        return

    def forward(self, x):
        x = self.bn(x)
        for layer in self.layers:
            x = getattr(self, layer)(x)
        return x


model = FFN(in_layer_size, hidden_layer_sizes, out_layer_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.functional.binary_cross_entropy

# Train the Model
step = 0
for epoch in range(num_epochs):
    for i, (X, y) in enumerate(train_loader):
        step += 1
        y = y.float()
        # Convert torch tensor to Variable
        X = Variable(X)
        y = Variable(y)

        batch_weights = get_loss_weights(y)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = model(X)
        loss = loss_func(outputs, y, weight=batch_weights, size_average=False)

        loss.backward()
        optimizer.step()

        info = {
            'loss': loss.data[0]
        }

        for key, value in info.items():
            logger.scalar_summary(key, value, step)


    confusion = torch.zeros([2, 2]).long()
    for i, (X, y) in enumerate(test_loader):
        y = y.long()
        X = Variable(X)
        y = Variable(y)

        output = model(X)
        pred = (output.data > .5).long()

        pred_correct = (pred == y.data)
        pred_wrong = (pred != y.data)

        fraud = (y.data == 1)
        nonfraud = (y.data == 0)

        tp = torch.sum(fraud * pred_correct)
        tn = torch.sum(nonfraud * pred_correct)
        fp = torch.sum(nonfraud * pred_wrong)
        fn = torch.sum(fraud * pred_wrong)

        confusion += torch.LongTensor([[tp, fn], [fp, tn]])

    print("Epoch [{:d}/{:d}], loss: {:.4f}".format(epoch+1, num_epochs, loss.data[0]))

print(confusion)

import ipdb
ipdb.set_trace()
