import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class InputTransformNet(nn.Module):
    def __init__(self):
        super(InputTransformNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 9)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        """
        x: [B, 3, N]
        """
        B, N = x.shape[0], x.shape[2]
        x = self.relu(self.bn1(self.conv1(x))) #[B, 64, N]
        x = self.relu(self.bn2(self.conv2(x))) #[B, 128, N]
        x = self.relu(self.bn3(self.conv3(x))) #[B, 1024, N]
        x = nn.MaxPool1d(N)(x) #[B, 1024, 1]
        x = x.view(B, 1024) #[B, 1024]
        x = self.relu(self.bn4(self.fc1(x))) #[B, 512]
        x = self.relu(self.bn5(self.fc2(x))) #[B, 256]
        x = self.transform(x) #[B, 9]
        x = x.view(B, 3, 3) #[B, 3, 3]
        return x

class FeatureTransformNet(nn.Module):
    def __init__(self):
        super(FeatureTransformNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 64 * 64)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(64, 64))

    def forward(self, x):
        """
        x: [B, 64, N]
        """
        B, N = x.shape[0], x.shape[2]
        x = self.relu(self.bn1(self.conv1(x))) #[B, 64, N]
        x = self.relu(self.bn2(self.conv2(x))) #[B, 128, N]
        x = self.relu(self.bn3(self.conv3(x))) #[B, 1024, N]
        x = nn.MaxPool1d(N)(x) #[B, 1024, 1]
        x = x.view(B, 1024) #[B, 1024]
        x = self.relu(self.bn4(self.fc1(x))) #[B, 512]
        x = self.relu(self.bn5(self.fc2(x))) #[B, 256]
        x = self.transform(x) #[B, 64]
        x = x.view(B, 64, 64) #[B, 64, 64]
        return x

if __name__ == '__main__':
    a = torch.rand(8, 3, 1000)
    t = InputTransformNet()
    x = t(a)
    print(x.shape)