import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import time
st = time.time()

class BuildingDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.buildings = sorted(os.listdir(os.path.join(self.root_dir, 'train', 'drone')))

    def __len__(self):
        return len(self.buildings)

    def __getitem__(self, index):
        building_name = self.buildings[index]
        drone_dir = os.path.join(self.root_dir, 'train', 'drone', building_name)
        satellite_dir = os.path.join(self.root_dir, 'train', 'satellite', building_name)

        satellite_path = os.path.join(satellite_dir, os.listdir(satellite_dir)[0])
        satellite_img = Image.open(satellite_path).convert('RGB')

        def drone_imgs():
            for drone_img_name in os.listdir(drone_dir):
                drone_img_path = os.path.join(drone_dir, drone_img_name)
                drone_img = Image.open(drone_img_path).convert('RGB')
                yield drone_img

        if self.transform:
            satellite_img = self.transform(satellite_img)
            dataset = [(satellite_img, self.transform(drone_img)) for drone_img in drone_imgs()]

        return dataset


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = BuildingDataset(root_dir='University-Release', transform=transform)

import itertools

result_list = list(itertools.chain.from_iterable(dataset))
print(len(result_list))


end = time.time()
print(end - st)
