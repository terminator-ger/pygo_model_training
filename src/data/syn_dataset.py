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
from typing import Dict
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
    def __init__(self, annotations_file, img_dir, train=True, return_stone_bbox=True, return_board_corners=True):
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
        self.return_stone_bbox = return_stone_bbox
        self.return_board_corners = return_board_corners
        self.num_classes = num_classes
        self.random_background = True

    def image_transformer(self, data: Dict):
        # scale down image size
        image = data['pixel_values']
        
        c,w,h = image.shape
        bbox = data['bbox']
        bbox = bbox / th.tensor([w,h,w,h])
        

        image = torchvision.transforms.Resize((512, 512))(image)
        
        h,w = image.shape[1], image.shape[2]
        is_occluded = th.tensor([0], dtype=th.int32)

        background_path = "/home/michael/data/unlabeled2017"
        background_images = os.listdir(background_path)
        if self.random_background:
            bg_file = os.path.join(background_path, background_images[np.random.randint(len(background_images))])
        else:
            bg_file = os.path.join(background_path, background_images[0])
            
        bg = cv2.resize(cv2.imread(bg_file), (h,w)) 
        #bg = torchvision.transforms.Resize((h,w))(bg)
        #image_fg = image[3].numpy()  # Convert to grayscale
        image_fg = image[:,:,3]
        ret, thresh = cv2.threshold(image_fg, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #mask = image[3] != 0  # Alpha channel mask
        mask = image[:,:,3] != 0  # Alpha channel mask
        mask = np.stack([mask,mask,mask],-1)
        #image = th.where(mask, image[:3], bg)
        image = np.where(mask, image[:,:,:3], bg)
        
        if False: #th.rand(1).item() < self.occlude_factor:
            is_occluded = th.tensor([1], dtype=th.int32)
            label = th.zeros_like(label, dtype=th.int32)
            # Apply occlusion
            # sample a random point within the image bounds
            pts = [(random() / 2 + 0.5) * cmath.exp(2j*np.pi*i/7) for i in range(7)]
            pts = convexHull([(pt.real, pt.imag ) for pt in pts])
            xs, ys = [interpolateSmoothly(zs, 30) for zs in zip(*pts)]
            scale_x = np.random.randint(w//16, w//4)
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            
            while cv2.pointPolygonTest(contours[0], (center_x, center_y), False) < 0:
                center_x = np.random.randint(0, w)
                center_y = np.random.randint(0, h)
            
            xs = [int((x * scale_x) + center_x) for x in xs]
            ys = [int((y * scale_x) + center_y) for y in ys]
            
            cnt = np.stack((xs,ys))[:,:,None].transpose(1,2,0)
            color = np.random.randint(0,255,size=(3,)).tolist()
            img_np = image.permute(1,2,0).detach().numpy().copy()
            image = cv2.drawContours(img_np, (cnt,), 0, color=color, thickness=cv2.FILLED)
            image = th.tensor(image).permute(2,0,1)
            label = th.ones_like(label, dtype=th.int32) * 3
        
        image = image / 255.0  # Normalize to [0, 1]
        data['pixel_values'] = image
        data['bbox'] = bbox
        return data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['filename'])
        image = decode_image(img_path)
        c,h,w = image.shape
        positions = th.from_numpy(np.stack(self.img_labels.iloc[idx]['labels'])).to(th.int32)
        if self.return_board_corners:
            bbox = th.from_numpy(np.stack(self.img_labels.iloc[idx]['bbox'])).to(th.float32)
            cls = th.from_numpy(np.stack(self.img_labels.iloc[idx]['cls'])).to(th.float32)[:,None]
            cls -= 1
        else:
            bbox = th.zeros((0,4), dtype=th.float32)
            cls = th.zeros((0,1), dtype=th.float32)
        if self.return_board_corners:
            corners = th.from_numpy(np.stack(self.img_labels.iloc[idx]['corners'])).to(th.float32)
            cond_a = corners[:,0] < 0 
            cond_b = corners[:,0] > w
            cond_c = corners[:,1] < 0 
            cond_d = corners[:,1] > h 
            
            idx = th.argwhere(th.logical_or(th.logical_or(th.logical_or(cond_a, cond_b), cond_c), cond_d))
            corners[idx] = 0.0
        else:
            corners = None
        #bs = self.img_labels.iloc[idx]['board_size']
        #if bs == 19:
        #    board_size = th.tensor([0], dtype=th.long)
        #elif bs == 13:
        #    board_size = th.tensor([1], dtype=th.long)
        #elif bs == 9:
        #    board_size = th.tensor([2], dtype=th.long)
        
        if not self.transform:
            raise ValueError("Transform function is not defined.")
        
        data =  {"pixel_values": image, 
                "labels": positions, 
                "bbox": bbox, 
                "cls": cls, 
                "corners": corners
        }
        return self.transform(data)
       
if __name__ == "__main__":
    dataset = GOSYNImageDataset(annotations_file="labels.parquet.gz", img_dir="/media/michael/Data/dev/pygo_synthetic/renders_new")
    import cv2
    print(f"Dataset size: {len(dataset)}")
    for i in range(10):
        img, labels = dataset[i]
        cv2.imwrite(f'img_{i}.png', img.permute(1,2,0).detach().numpy()*255)
    print(f"Image shape: {img.shape}, Label: {labels}")