import torch.utils.data as data
import h5py
import numpy as np
import os
from glob import glob

class ModelNetDataset(data.Dataset):
    def __init__(self, train=True):
        if train:
            data_file = 'data/modelnet40_ply_hdf5_2048/train_files.txt'
        else:
            data_file = 'data/modelnet40_ply_hdf5_2048/test_files.txt'
        file_list = [line.rstrip() for line in open(data_file, 'r')]
        
        all_data = np.zeros([0, 2048, 3], np.float32)
        all_label = np.zeros([0, 1], np.int64)
        for filename in file_list:
            f = h5py.File(filename)
            data = f['data'][:]
            label = f['label'][:]

            all_data = np.concatenate([all_data, data], 0)
            all_label = np.concatenate([all_label, label], 0)

        self.pointcloud = all_data
        self.label = all_label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        return self.pointcloud[index], self.label[index]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class TensorBodyDataset():
    def __init__(self, data_dir, normalize=True, train=True):
        self.normalize = normalize
        self.pointcloud_files = []
        self.label_files = []
        file_list = os.path.join(data_dir, 'data_list.txt')
        with open(file_list, 'r') as file:
            for line in file:
                if line:
                    pointcloud_file, label_file = line.rstrip().split(' ')
                    self.pointcloud_files.append(os.path.join(data_dir, pointcloud_file))
                    self.label_files.append(os.path.join(data_dir, label_file))
        if train:
            self.idxs = np.arange(len(self.pointcloud_files))[:30000]
        else:
            self.idxs = np.arange(len(self.pointcloud_files))[30000:]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        pointcloud = np.load(self.pointcloud_files[self.idxs[index]]).astype(np.float32)
        label = np.load(self.label_files[self.idxs[index]]).astype(np.int64)

        if self.normalize:
            pointcloud = pc_normalize(pointcloud)

        return pointcloud, label

class SMPLDataset():
    def __init__(self, data_dir, normalize=True, train=True):
        self.normalize = normalize
        self.pointcloud_files = glob(os.path.join(data_dir, 'pointclouds', '*/*.npy'))
        self.label_files = glob(os.path.join(data_dir, 'labels', '*/*.npy'))
        N = len(self.pointcloud_files)  
        indices = np.random.choice(N, N, replace=False)
        part = int(N * 0.8)
        if train:
            self.idxs = indices[:part]
        else:
            self.idxs = indices[part:]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        pointcloud = np.load(self.pointcloud_files[self.idxs[index]]).astype(np.float32)
        label = np.load(self.label_files[self.idxs[index]]).astype(np.int64)

        if self.normalize:
            pointcloud = pc_normalize(pointcloud)

        return pointcloud, label


if __name__ == '__main__':
    #dataset = ModelNetDataset()
    #dataset = TensorBodyDataset('data/seg1024')
    dataset = SMPLDataset('D:\\Data\\CMUPointclouds')
    print(len(dataset))
