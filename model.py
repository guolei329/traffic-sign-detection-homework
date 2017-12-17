import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        # relu is a typical way to introduce non-linearity
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # drop some random activations in the conv2 layer
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x)

# VGG11 with batch normalization:
# ref: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# there are also VGG16 and VGG19 which are really popular today
# ref: https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
# special help to Yihui Wu for some help on the config

# since VGG11 on torch vision is too complicated for our project,
# we write our own simple version:

def make_layers():
    # this is config B in reference
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                       nn.BatchNorm2d(v),
                       nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.features = make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, nclasses),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x)
