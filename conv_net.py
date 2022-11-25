
# region packages
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# endregion


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        n_layer = 5
        n_chan1 = 32
        n_chan2 = 64
        n_chan3 = 128
        n_chan4 = 256
        n_chan5 = 512
        self.activation = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.BatchNorm2d(n_chan2),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.BatchNorm2d(n_chan3),
            nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.BatchNorm2d(n_chan4),
            nn.MaxPool2d(2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan5, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )

        n_pix_final = int(96 / 2**n_layer)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_chan5*n_pix_final**2, n_chan4),
            nn.BatchNorm1d(n_chan4),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(n_chan4, n_chan3),
            nn.BatchNorm1d(n_chan3),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(n_chan3, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        y = y.view(x.shape[0], -1)
        y = self.fc(y)
        return y


class AlexNet(nn.Module):
    def __init__(self, dim_image):
        super(AlexNet, self).__init__()
        n_layer = 5
        n_chan1 = 96
        n_chan2 = 256
        n_chan3 = 384
        n_chan4 = 384
        n_chan5 = 256
        n_neurons1 = 4096

        self.activation = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(0.5)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan5),
            self.activation,
            nn.MaxPool2d(2)
        )

        n_pix_final = int(dim_image / 2**n_layer)
        self.fc1 = nn.Sequential(
            nn.Linear(n_chan5*n_pix_final**2, n_neurons1),
            self.activation
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_neurons1, n_neurons1),
            self.activation
        )
        self.fc3 = nn.Sequential(
            nn.Linear(n_neurons1, 2)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        y = y.view(x.shape[0], -1)
        y = self.drop_out(y)
        y = self.fc1(y)
        y = self.drop_out(y)
        y = self.fc2(y)
        y = self.drop_out(y)
        y = self.fc3(y)
        return y


class VGG16(nn.Module):
    def __init__(self, dim_image):
        super(VGG16, self).__init__()
        n_maxpool2 = 5
        n_chan1 = 64
        n_chan2 = 128
        n_chan3 = 256
        n_chan4 = 512

        self.activation = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.layer1_3_chan1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
        )  # dim_image

        self.layer2_chan1_chan1_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
            self.maxpool
        )  # dim_image / 2

        self.layer3_chan1_chan2_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
        )  # dim_image / 2

        self.layer4_chan2_chan2_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
            self.maxpool
        )  # dim_image / 2^2

        self.layer5_chan2_chan3_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
        )  # dim_image / 2^2

        self.layer6_chan3_chan3_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
        )  # dim_image / 2^2

        self.layer7_chan3_chan3_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
            self.maxpool
        )  # dim_image / 2^3

        self.layer8_chan3_chan4_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^3

        self.layer9_chan4_chan4_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^3

        self.layer10_chan4_chan4_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
            self.maxpool
        )  # dim_image / 2^4

        self.layer11_chan4_chan4_4 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^4

        self.layer12_chan4_chan4_5 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^4

        self.layer13_chan4_chan4_6 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
            self.maxpool
        )  # dim_image / 2^5

        # layer_AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(None)

        n_pix_final = int(dim_image / 2 ** n_maxpool2)
        n_chan_final = n_chan4

        self.fc1 = nn.Sequential(
            nn.Linear(n_chan_final * n_pix_final ** 2, n_chan4),
            # nn.BatchNorm1d(n_chan4),
            self.activation
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_chan4, 10),
            self.activation
        )
        self.fc3 = nn.Sequential(
            nn.Linear(10, 2)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layer1_3_chan1_1(x)
        y = self.layer2_chan1_chan1_2(y)
        y = self.layer3_chan1_chan2_1(y)
        y = self.layer4_chan2_chan2_2(y)
        y = self.layer5_chan2_chan3_1(y)
        y = self.layer6_chan3_chan3_2(y)
        y = self.layer7_chan3_chan3_3(y)
        y = self.layer8_chan3_chan4_1(y)
        y = self.layer9_chan4_chan4_2(y)
        y = self.layer10_chan4_chan4_3(y)
        y = self.layer11_chan4_chan4_4(y)
        y = self.layer12_chan4_chan4_5(y)
        y = self.layer13_chan4_chan4_6(y)
        y = y.view(x.shape[0], -1)
        y = self.drop_out(y)
        y = self.fc1(y)
        y = self.drop_out(y)
        y = self.fc2(y)
        y = self.drop_out(y)
        y = self.fc3(y)
        return y


class VGG16p(nn.Module):
    def __init__(self, dim_image):
        super(VGG16p, self).__init__()
        n_maxpool2 = 4
        n_chan1 = 32
        n_chan2 = 64
        n_chan3 = 128
        n_chan4 = 256

        self.activation = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.layer1_3_chan1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
        )  # dim_image

        self.layer2_chan1_chan1_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
        )  # dim_image

        self.layer3_chan1_chan1_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
            self.maxpool
        )  # dim_image / 2

        self.layer4_chan1_chan2_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
        )  # dim_image / 2

        self.layer5_chan2_chan2_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
        )  # dim_image / 2

        self.layer6_chan2_chan2_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
            self.maxpool
        )  # dim_image / 2^2

        self.layer7_chan2_chan3_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
        )  # dim_image / 2^2

        self.layer8_chan3_chan3_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
        )  # dim_image / 2^2

        self.layer9_chan3_chan3_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
            self.maxpool
        )  # dim_image / 2^3

        self.layer10_chan3_chan4_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^3

        self.layer11_chan4_chan4_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^3

        self.layer12_chan4_chan4_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
            self.maxpool
        )  # dim_image / 2^4

        n_pix_final = int(dim_image / 2 ** n_maxpool2)
        n_chan_final = n_chan4

        self.fc1 = nn.Sequential(
            nn.Linear(n_chan_final * n_pix_final ** 2, 512),
            # nn.BatchNorm1d(n_chan4),
            self.activation
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 10),
            self.activation
        )
        self.fc3 = nn.Sequential(
            nn.Linear(10, 2)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layer1_3_chan1_1(x)
        y = self.layer2_chan1_chan1_2(y)
        y = self.layer3_chan1_chan1_3(y)
        y = self.layer4_chan1_chan2_1(y)
        y = self.layer5_chan2_chan2_2(y)
        y = self.layer6_chan2_chan2_3(y)
        y = self.layer7_chan2_chan3_1(y)
        y = self.layer8_chan3_chan3_2(y)
        y = self.layer9_chan3_chan3_3(y)
        y = self.layer10_chan3_chan4_1(y)
        y = self.layer11_chan4_chan4_2(y)
        y = self.layer12_chan4_chan4_3(y)
        y = y.view(x.shape[0], -1)
        # y = self.drop_out(y)
        y = self.fc1(y)
        # y = self.drop_out(y)
        y = self.fc2(y)
        # y = self.drop_out(y)
        y = self.fc3(y)
        return y


class VGG16p2(nn.Module):
    def __init__(self, dim_image):
        super(VGG16p2, self).__init__()
        n_maxpool2 = 4
        n_chan1 = 32
        n_chan2 = 64
        n_chan3 = 128
        n_chan4 = 256

        self.activation = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.layer1_3_chan1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
        )  # dim_image

        self.layer2_chan1_chan1_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
        )  # dim_image

        self.layer3_chan1_chan1_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan1),
            self.activation,
            self.maxpool
        )  # dim_image / 2

        self.layer4_chan1_chan2_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
        )  # dim_image / 2

        self.layer5_chan2_chan2_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
        )  # dim_image / 2

        self.layer6_chan2_chan2_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan2),
            self.activation,
            self.maxpool
        )  # dim_image / 2^2

        self.layer7_chan2_chan3_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
        )  # dim_image / 2^2

        self.layer8_chan3_chan3_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
        )  # dim_image / 2^2

        self.layer9_chan3_chan3_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan3),
            self.activation,
            self.maxpool
        )  # dim_image / 2^3

        self.layer10_chan3_chan4_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan3, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^3

        self.layer11_chan4_chan4_2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
        )  # dim_image / 2^3

        self.layer12_chan4_chan4_3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan4, out_channels=n_chan4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_chan4),
            self.activation,
            self.maxpool
        )  # dim_image / 2^4

        n_pix_final = int(dim_image / 2 ** n_maxpool2)
        n_chan_final = n_chan4

        self.fc1 = nn.Sequential(
            nn.Linear(n_chan_final * n_pix_final ** 2, 512),
            # nn.BatchNorm1d(n_chan4),
            self.activation
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 10),
            self.activation
        )
        self.fc3 = nn.Sequential(
            nn.Linear(10, 2)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.layer1_3_chan1_1(x)
        y = self.layer2_chan1_chan1_2(y)
        y = self.layer3_chan1_chan1_3(y)
        y = self.layer4_chan1_chan2_1(y)
        y = self.layer5_chan2_chan2_2(y)
        y = self.layer6_chan2_chan2_3(y)
        y = self.layer7_chan2_chan3_1(y)
        y = self.layer8_chan3_chan3_2(y)
        y = self.layer9_chan3_chan3_3(y)
        y = self.layer10_chan3_chan4_1(y)
        y = self.layer11_chan4_chan4_2(y)
        y = self.layer12_chan4_chan4_3(y)
        y = y.view(x.shape[0], -1)
        y = self.drop_out(y)
        y = self.fc1(y)
        y = self.drop_out(y)
        y = self.fc2(y)
        y = self.drop_out(y)
        y = self.fc3(y)
        return y


# def train_model(model, loss_function, optimizer):
#     pass
