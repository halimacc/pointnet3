import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transform_nets import InputTransformNet, FeatureTransformNet

class PointNet(nn.Module):
    def __init__(self, global_feature=True):
        super(PointNet, self).__init__()
        self.global_feature = global_feature
        self.input_transform = InputTransformNet()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.feature_transform = FeatureTransformNet()
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        """
        x: [B, 3, N]
        """
        B, N = x.shape[0], x.shape[2]
        input_transform = self.input_transform(x) #[B, 3, 3]
        x = torch.matmul(x.permute(0, 2, 1), input_transform.permute(0, 2, 1)).permute(0, 2, 1) #[B, 3, N]
        x = F.relu(self.bn1(self.conv1(x))) #[B, 64, N]
        x = F.relu(self.bn2(self.conv2(x))) #[B, 64, N]
        feature_transform = self.feature_transform(x) #[B, 64, 64]
        x = torch.matmul(x.permute(0, 2, 1), feature_transform.permute(0, 2, 1)).permute(0, 2, 1) #[B, 64, N]
        point_feature = x
        x = F.relu(self.bn3(self.conv3(x))) #[B, 64, N]
        x = F.relu(self.bn4(self.conv4(x))) #[B, 128, N]
        x = F.relu(self.bn5(self.conv5(x))) #[B, 1024, N]
        x = nn.MaxPool1d(N)(x) #[B, 1024, 1]
        if not self.global_feature:
            x = x.repeat([1, 1, N]) #[B, 1024, N]
            x = torch.cat([point_feature, x], 1) #[B, 1088, N]
        return x

class PointNetCls(nn.Module):
    def __init__(self):
        super(PointNetCls, self).__init__()
        self.pointnet = PointNet()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(256, 40)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.pointnet(x) # [B, 1024, 1]
        x = x.view(B, 1024) #[B, 1024]
        x = self.drop1(F.relu(self.bn1(self.fc1(x)))) #[B, 512]
        x = self.drop2(F.relu(self.bn2(self.fc2(x)))) #[B, 256]
        x = self.fc3(x) #[B, 40]
        x = F.log_softmax(x, dim=-1) #[B, 40]
        return x

class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2048):
        super(PointNetSeg, self).__init__()
        self.num_classes = num_classes
        self.pointnet = PointNet(global_feature=False)
        self.conv1 = nn.Conv1d(1088, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 1024, 1)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.7)
        self.conv3 = nn.Conv1d(1024, self.num_classes, 1)
        
    def forward(self, x):
        x = self.pointnet(x) #[B, 1088, N]
        x = F.relu(self.bn1(self.conv1(x))) #[B, 512, N]
        x = self.drop1(F.relu(self.bn2(self.conv2(x)))) #[B, 256, N]
        x = self.conv3(x) #[B, num_classes, N]
        x = F.log_softmax(x, dim=1) #[B, num_classes, N]
        return x
        

if __name__ == '__main__':
    net = PointNet()
    x = torch.rand(8, 3, 1024)
    x = net(x)
    print(x.shape)
        