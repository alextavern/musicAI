from torch import nn
import torchvision.models as models
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


class CNNScratch-ESC(nn.Module):
    def __init__(self, input_shape):
        super().__init__():
        self.input_shape = input_shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        )
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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        # self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size / 2), int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), num_classes)

    def forward(self, x):
        x = x.float()
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out