from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from helper import conv2d_output_size


class MLPBase(nn.Module):
    def __init__(self, input_shape, dropout, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.dropout = dropout
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(self.input_shape[0] * self.input_shape[1], 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


class MLP1(nn.Module):

    def __init__(self, input_shape, dropout, num_classes):
        super().__init__()
        self.dropout = dropout
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(

            nn.Linear(self.input_shape[0] * self.input_shape[1], 512),  # be careful of the input shape!
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            # nn.Linear(64, ),
            # nn.ReLU(),
            # nn.Dropout(self.dropout),

            nn.Linear(128, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


class CNN3(nn.Module):
    def __init__(self, input_shape, num_cats=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.dense1 = nn.Linear(256*(((input_shape[0]//2)//2)//2)*(((input_shape[1]//2)//2)//2), 500)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(500, num_cats)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class CNNScratch1(nn.Module):

    def __init__(self, dropout, flattened_size):
        super().__init__()

        self.dropout = dropout
        self.flattened_size = flattened_size
        # 4 conv blocks / flatten / linear / softmax
        # input -> 64, 87
        # 13
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        # output1 -> 33, 44
        # 7
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # output2 -> 17, 23

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(32 * 1 * 2, 64)
        self.linear1 = nn.Linear(self.flattened_size, 64)
        self.dropout = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(64, 10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        predictions = self.softmax(logits)
        return predictions


class CNNScratch(nn.Module):

    def __init__(self):
        super().__init__()

        # 4 conv blocks / flatten / linear / softmax
        # input -> 64, 87
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # output1 -> 33, 44
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # output2 -> 17, 23

        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # output3 -> 9, 12

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.dropout4 = nn.Dropout(0.5)
        # output 4 -> 5, 7

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.dropout5 = nn.Dropout(0.5)
        # output 3 -> 4

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 4 * 3, 10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout4(x)
        x = self.conv5(x)
        x = self.dropout5(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        num_classes = 10
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class Inception(nn.Module):
    def __init__(self, pretrained=True):
        super(Inception, self).__init__()
        num_classes = 10
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        num_classes = 10
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

