import argparse
from pointnet import PointNetCls, PointNetSeg
from pointnet2 import PointNet2SemSeg, PointNet2PartSeg, PointNet2Seg
from datasets import *
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
import os

def train(num_epochs, batch_size, ckpt_dir):
    num_classes = 1000
    num_points = 2048

    #train_dataset = TensorBodyDataset('data/seg1024rand', train=True)
    train_dataset = SMPLDataset('D:\\Data\\CMUPointclouds')
    train_examples = len(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    train_batches = len(train_dataloader)

    #test_dataset = TensorBodyDataset('data/seg1024rand', train=False)
    test_dataset = SMPLDataset('D:\\Data\\CMUPointclouds', train=False)
    test_examples = len(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    #classifier = PointNetSeg(num_classes=num_classes)
    classifier = PointNet2Seg(num_classes=num_classes)

    # load params
    print("Load parameters...")
    state_dict = torch.load('smpl1000pretrain/3.pth')
    own_state = classifier.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:# or not name.startswith('sa'):
            print(name)
            continue
        own_state[name].copy_(param)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3*0.9*0.9*0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    print("Train examples: {}".format(train_examples))
    print("Evaluation examples: {}".format(test_examples))
    print("Start training...")
    cudnn.benchmark = True
    classifier.cuda()
    for epoch in range(4, num_epochs):
        print("--------Epoch {}--------".format(epoch))

        # train one epoch
        classifier.train()
        scheduler.step()
        total_train_loss = 0
        correct_examples = 0
        for batch_idx, data in enumerate(train_dataloader, 0):
            pointcloud, label = data
            pointcloud = pointcloud.permute(0, 2, 1)
            pointcloud, label = pointcloud.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = classifier(pointcloud)

            loss = F.nll_loss(pred, label)
            pred_choice = pred.max(1)[1]

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_examples += pred_choice.eq(label).sum().item()
            
        print("Train loss: {:.4f}, train accuracy: {:.2f}%".format(total_train_loss / train_batches, correct_examples / train_examples / num_points * 100.0))
        torch.save(classifier.state_dict(), os.path.join(ckpt_dir, '{}.pth'.format(epoch)))

        # eval one epoch
        classifier.eval()
        correct_examples = 0
        for batch_idx, data in enumerate(test_dataloader, 0):
            pointcloud, label = data
            pointcloud = pointcloud.permute(0, 2, 1)
            pointcloud, label = pointcloud.cuda(), label.cuda()

            pred = classifier(pointcloud)
            pred_choice = pred.max(1)[1]
            correct = pred_choice.eq(label).sum()
            correct_examples += correct.item()

        print("Eval accuracy: {:.2f}%".format(correct_examples / test_examples / num_points * 100.0))

def test():
    num_classes = 1024
    num_points = 2048

    test_dataset = TensorBodyDataset('data/seg1024rand', train=True)
    test_examples = len(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    print(test_examples)

    classifier = PointNet2Seg(num_classes=num_classes)
    classifier.load_state_dict(torch.load('ckpt/10.pth'))

    print("Start testing...")
    classifier.cuda()

    # eval one epoch
    classifier.eval()
    correct_examples = 0
    for batch_idx, data in enumerate(test_dataloader, 0):
        pointcloud, label = data
        pointcloud = pointcloud.permute(0, 2, 1)
        pointcloud, label = pointcloud.cuda(), label.cuda()

        pred = classifier(pointcloud)
        pred_choice = pred.max(1)[1]
        correct = pred_choice.eq(label).sum()
        correct_examples += correct.item()
        print(correct.item())

    print("Eval accuracy: {:.2f}%".format(correct_examples / test_examples / num_points * 100.0))

if __name__ == '__main__':
    batch_size = 8
    num_epochs = 10
    ckpt_dir = 'smpl1000pretrain'
    train(num_epochs, batch_size, ckpt_dir)
    #test()