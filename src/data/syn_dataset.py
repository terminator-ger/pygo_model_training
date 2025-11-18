import os
import pandas as pd
import torch as th
from torchvision.io import decode_image
import torchvision
import cv2
import numpy as np
import cmath
from math import atan2
import torch.nn.functional as F
from random import random

def convexHull(pts):    #Graham's scan.
    xleftmost, yleftmost = min(pts)
    by_theta = [(atan2(x-xleftmost, y-yleftmost), x, y) for x, y in pts]
    by_theta.sort()
    as_complex = [complex(x, y) for _, x, y in by_theta]
    chull = as_complex[:2]
    for pt in as_complex[2:]:
        #Perp product.
        while ((pt - chull[-1]).conjugate() * (chull[-1] - chull[-2])).imag < 0:
            chull.pop()
        chull.append(pt)
    return [(pt.real, pt.imag) for pt in chull]


def dft(xs):
    return [sum(x * cmath.exp(2j*np.pi*i*k/len(xs)) 
                for i, x in enumerate(xs))
            for k in range(len(xs))]

def interpolateSmoothly(xs, N):
    """For each point, add N points."""
    fs = dft(xs)
    half = (len(xs) + 1) // 2
    fs2 = fs[:half] + [0]*(len(fs)*N) + fs[half:]
    return [x.real / len(xs) for x in dft(fs2)[::-1]]


class GOSYNImageDataset(th.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, train=True):
        self.img_labels = pd.read_parquet(os.path.join(img_dir, annotations_file))
        if train:
            # reserve 80% for training
            self.img_labels = self.img_labels.iloc[:int(0.8*len(self.img_labels))]
        else:
            # reserve 20% for testing
            self.img_labels = self.img_labels.iloc[int(0.8*len(self.img_labels)):]
            
        self.img_dir = img_dir
        self.transform = self.image_transformer
        self.occlude_factor = 0.4

    def image_transformer(self, image, label):
        # scale down image size
        image = torchvision.transforms.Resize((256, 192))(image)
        h,w = image.shape[1], image.shape[2]
        is_occluded = th.tensor([0], dtype=th.int32)

        background_path = "D:" + os.sep + "unlabeled2017"
        background_images = os.listdir(background_path)
        bg = decode_image(os.path.join(background_path, background_images[th.randint(0, len(background_images), (1,)).item()]))
        bg = torchvision.transforms.Resize((h,w))(bg)
        image_fg = image[3].numpy()  # Convert to grayscale
        ret, thresh = cv2.threshold(image_fg, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = image[3] != 0  # Alpha channel mask
        image = th.where(mask, image[:3], bg)
        
        if False:# th.rand(1).item() < self.occlude_factor:
            is_occluded = th.tensor([1], dtype=th.int32)
            label = th.zeros_like(label, dtype=th.int32)
            # Apply occlusion
            # sample a random point within the image bounds
            pts = [(random() / 2 + 0.5) * cmath.exp(2j*np.pi*i/7) for i in range(7)]
            pts = convexHull([(pt.real, pt.imag ) for pt in pts])
            xs, ys = [interpolateSmoothly(zs, 30) for zs in zip(*pts)]
            scale_x = int(np.random.randint(w//16, w//4, 1))
            center_x = int(np.random.randint(0, w, 1))
            center_y = int(np.random.randint(0, h, 1))
            
            while cv2.pointPolygonTest(contours[0], (center_x, center_y), False) < 0:
                center_x = int(np.random.randint(0, w, 1))
                center_y = int(np.random.randint(0, h, 1))
            
            xs = [int((x * scale_x) + center_x) for x in xs]
            ys = [int((y * scale_x) + center_y) for y in ys]
            
            cnt = np.stack((xs,ys))[:,:,None].transpose(1,2,0)
            color = np.random.randint(0,255,size=(3,)).tolist()
            img_np = image.permute(1,2,0).detach().numpy().copy()
            image = cv2.drawContours(img_np, (cnt,), 0, color=color, thickness=cv2.FILLED)
            image = th.tensor(image).permute(2,0,1)
            label = th.ones_like(label, dtype=th.int32) * 3
            
        image = image / 255.0  # Normalize to [0, 1]
        return image, label#, is_occluded

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['filename'])
        image = decode_image(img_path)
        positions = th.from_numpy(np.stack(self.img_labels.iloc[idx]['labels'])).to(th.int32)
        #bs = self.img_labels.iloc[idx]['board_size']
        #if bs == 19:
        #    board_size = th.tensor([0], dtype=th.long)
        #elif bs == 13:
        #    board_size = th.tensor([1], dtype=th.long)
        #elif bs == 9:
        #    board_size = th.tensor([2], dtype=th.long)
        
        if not self.transform:
            raise ValueError("Transform function is not defined.")
        
        image, labels = self.transform(image, positions)
            
        return image, labels#, is_occluded)#, board_size)
    

if __name__ == "__main__":
    dataset = GOSYNImageDataset(annotations_file="labels.parquet.gz", img_dir="e:\\dev\\pygo_synthetic\\renders_new")
    print(f"Dataset size: {len(dataset)}")
    img, labels = dataset[0]
    print(f"Image shape: {img.shape}, Label: {labels}")