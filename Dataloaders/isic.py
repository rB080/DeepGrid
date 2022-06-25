import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader

def load_data(img_path, mask_path, resize_dims=(128,128)):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # , cv2.IMREAD_GRAYSCALE

    img = cv2.resize(img, resize_dims, cv2.INTER_AREA)
    mask = cv2.resize(mask, resize_dims, cv2.INTER_AREA)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0

    mask = mask[np.newaxis, :, :] / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0

    return img, mask

def load_list(root_path, split="train"):
    if split=="train":
        label_path = os.path.join(root_path, 'ISIC-2017_Training_Part1_GroundTruth')
        label_path = os.path.join(label_path, 'ISIC-2017_Training_Part1_GroundTruth')
        data_path = os.path.join(root_path, 'ISIC-2017_Training_Data')
        data_path = os.path.join(data_path, 'ISIC-2017_Training_Data')
    elif split=="val":
        label_path = os.path.join(root_path, 'ISIC-2017_Validation_Part1_GroundTruth')
        label_path = os.path.join(label_path, 'ISIC-2017_Validation_Part1_GroundTruth')
        data_path = os.path.join(root_path, 'ISIC-2017_Validation_Data')
        data_path = os.path.join(data_path, 'ISIC-2017_Validation_Data')
    else:
        label_path = os.path.join(root_path, 'ISIC-2017_Test_v2_Part1_GroundTruth')
        label_path = os.path.join(label_path, 'ISIC-2017_Test_v2_Part1_GroundTruth')
        data_path = os.path.join(root_path, 'ISIC-2017_Test_v2_Data')
        data_path = os.path.join(data_path, 'ISIC-2017_Test_v2_Data')

    labels = sorted(os.listdir(label_path))
    img_paths = []
    gt_paths = []
    for i, name in enumerate(labels):
        gt_paths.append(os.path.join(label_path, name))
        img_paths.append(os.path.join(data_path, name[:12]+'.jpg'))
    if split == "trainval":
        i, g = load_list(root_path, split="val")
        gt_paths += g
        img_paths += i
    return img_paths, gt_paths


class Dataset(Dataset):

    def __init__(self, root_path, dims=(128,128), split="train"):
        assert split in {"train", "val", "trainval", "test"}, 'No such splits available'
        super().__init__()
        self.root = root_path
        self.split = split
        self.dims = dims
        self.images, self.labels = load_list(self.root, self.split)

    def __getitem__(self, index):
        img, mask = load_data(self.images[index], self.labels[index], self.dims)
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

def getTrainLoader(root, batch_size, dims=(256,256)):
    train_dataset = Dataset(root, split="train", dims=dims)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, len(train_dataset)

def getValLoader(root, batch_size, dims=(256,256)):
    val_dataset = Dataset(root, split="val", dims=dims)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return val_loader, len(val_dataset)

def getTrainvalLoader(root, batch_size, dims=(256,256)):
    trainval_dataset = Dataset(root, split="trainval", dims=dims)
    trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)
    return trainval_loader, len(trainval_dataset)

def getTestLoader(root, batch_size, dims=(256,256)):
    test_dataset = Dataset(root, split="test", dims=dims)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, len(test_dataset)
