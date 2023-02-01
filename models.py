from torch import nn


class MLP1(nn.Module):

    def __init__(self, dropout):
        self.dropout = dropout
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(

            nn.Linear(64 * 87, 512),  # be careful of the input shape!
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

            nn.Linear(128, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

class CNN1(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
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

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 3 * 3, 10)
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
