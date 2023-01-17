import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import pandas as pd


class openimg_dataset(Dataset):
    def __init__(self, root_dir, csv_file, label_idx,  transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file, header=0)
        self.img_ids = self.data["ImageID"].unique()
        self.label_idx = label_idx
    

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]
        img_path = os.path.join(self.root_dir, img_name+".jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.rollaxis(img, 2, 0)
        bb_list = self.data[self.data["ImageID"] == img_name]
        boxes = bb_list[["XMin", "YMin", "XMax", "YMax"]].values
        truncated = bb_list["IsTruncated"].values
        
        labels = bb_list["LabelName"].values
        label = []
        box = []
        trunc = []
        for i in range(len(labels)):
            if(labels[i] in self.label_idx):
                label.append(self.label_idx[labels[i]])
                box.append(boxes[i])
                trunc.append(truncated[i])
        boxes = torch.as_tensor(box.copy(), dtype=torch.float32)
        # labels = [self.label_idx[label] for label in labels]
        labels = torch.as_tensor(label.copy(), dtype=torch.int64)
        truncted = torch.as_tensor(trunc.copy(), dtype=torch.int64)

        if self.transform:
            sample = self.transform(sample)
        
        # result = {"Image": img,"Boxes": boxes, "Labels": labels, "Truncated": truncted}
        return img, boxes.numpy(), labels.numpy(), np.abs(truncted.numpy())

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        scale = list()
        istruncated = list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            scale.append(b[3])
            istruncated.append(b[4])
        images = np.stack(images, axis=0)
        return torch.as_tensor(images, dtype=torch.float32), torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.int64), torch.as_tensor(scale, dtype=torch.float32), torch.as_tensor(istruncated, dtype=torch.int64)
