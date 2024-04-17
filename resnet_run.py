import os
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import itertools
from Classifier import Classifier
from torchvision.models import resnet50


class DualBranchModel(nn.Module):
	def __init__(self, backbone, is_train=True, input_size=1000, num_classes=701):
		super(DualBranchModel, self).__init__()
		self.device = device
		self.backbone_ = backbone
		self.classif = Classifier(num_classes=num_classes, input_size=input_size, is_train=is_train)

	def forward(self, x1, x2):
		x1 = self.backbone_(x1)
		x2 = self.backbone_(x2)

		x1 = self.classif(x1)
		x2 = self.classif(x2)
		return x1, x2


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device, '\n')

backbone = resnet50().to(device)
model = DualBranchModel(backbone, is_train=False).to(device)
model.load_state_dict(torch.load('model_res.pt'))
model.eval()

# transform = transforms.Compose([
# 	transforms.Resize((256, 256)),
# 	transforms.ToTensor(),
# 	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

batch_size = 1


# img1 = Image.open(r'E:\Datasets\University-Release\test\gallery_drone\0001\image-06.jpeg')
# img1 = transform(img1).to(device)
# img2 = Image.open(r'E:\Datasets\University-Release\test\gallery_satellite\0004\0004.jpg')
# img2 = transform(img2).to(device)
# img3 = Image.open(r'E:\Datasets\University-Release\test\gallery_satellite\0005\0005.jpg')
# img3 = transform(img3).to(device)
# img4 = Image.open(r'E:\Datasets\University-Release\test\gallery_satellite\0001\0001.jpg')
# img4 = transform(img4).to(device)
# l = [(img1, img2, 0), (img1, img3, 0),
# 	 (img1, img4, 1)]
# dataloader = DataLoader(l, batch_size=batch_size)

# input_tensor1 = torch.randn(3, 256, 256).to(device)
# input_tensor2 = torch.randn(3, 256, 256).to(device)
# input_tensor3 = torch.randn(3, 256, 256).to(device)
# input_tensor4 = torch.randn(3, 256, 256).to(device)
# l = [(input_tensor1, input_tensor2, 0), (input_tensor3, input_tensor4, 1),
# 	 (input_tensor1, input_tensor4, 0), (input_tensor3, input_tensor2, 0)]
# dataloader = DataLoader(l, batch_size=batch_size)

class BuildingDataset(data.Dataset):
	def __init__(self, root_dir, transform=None, mode='both'):
		self.root_dir = root_dir
		self.transform = transform
		self.buildings = sorted(os.listdir(os.path.join(self.root_dir, 'test', 'gallery_drone')))
		self.mode = mode

	def __len__(self):
		return len(self.buildings)

	def __getitem__(self, index):
		building_name = self.buildings[index]
		drone_dir = os.path.join(self.root_dir, 'test', 'gallery_drone', building_name)
		satellite_dir = os.path.join(self.root_dir, 'test', 'gallery_satellite', building_name)

		satellite_path = os.path.join(satellite_dir, os.listdir(satellite_dir)[0])
		satellite_img = Image.open(satellite_path).convert('RGB')

		if self.mode == 'drone':

			def drone_imgs():
				for drone_img_name in os.listdir(drone_dir):
					drone_img_path = os.path.join(drone_dir, drone_img_name)
					drone_img = Image.open(drone_img_path).convert('RGB')
					yield drone_img

			if self.transform:
				dataset = [self.transform(drone_img) for drone_img in drone_imgs()]

			return dataset

		if self.mode == 'satellite':
			if self.transform:
				satellite_img = self.transform(satellite_img)

			return satellite_img

		else:
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
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dir = r'C:\Users\Insight\Documents\Обучение\Магистратура\2 семестр\Методы машинного обучения в робототехнике\Курсовая работа\University-Release'
dataset_satellite = BuildingDataset(root_dir=dir, transform=transform, mode='satellite')
dataset_drone = BuildingDataset(root_dir=dir, transform=transform, mode='drone')

dataset_satellite_loader = [dataset_satellite[i] for i in range(len(dataset_satellite))]

len_dataset = 35
result_dataset = []
# создаем спсиок для теста
for position in range(len_dataset):
	vector = [0] * len_dataset
	vector[position] = 1
	for j in range(54):
		dataset_drone_loader = [dataset_drone[position][j]] * len(dataset_drone)
		result = [(x, y, z) for x, y, z in zip(dataset_satellite_loader[:len_dataset], dataset_drone_loader, vector)]
		result_dataset.append(result)

print('Старт')
R1 = 0
R5 = 0
R10 = 0
count = 0
for j in range(len(result_dataset)):
	dataloader = DataLoader(result_dataset[j], batch_size=batch_size)

	distances = []
	for i, (sat_img, uav_img, target) in enumerate(dataloader):
		sat_img = sat_img.to(device)
		uav_img = uav_img.to(device)
		target = target.to(device)
		sat, uav = model(sat_img, uav_img)

		distance = torch.linalg.vector_norm(uav - sat, ord=2)
		distances.append((distance.item(), i))
		if target == 1:
			true_index = i

	sorted_distances = sorted(distances, key=lambda x: x[0])
	for distance, idx in itertools.islice(sorted_distances, 5):
		print(f"Distance: {distance}, i: {idx}")
	print()

	count += 1
	if sorted_distances[0][1] == true_index:
		R1 += 1
		print('R1 +')
	for d in range(5):
		if sorted_distances[d][1] == true_index:
			R5 += 1
			print('R5 +')
			break
	for k in range(10):
		if sorted_distances[k][1] == true_index:
			R10 += 1
			print('R10 +')
			break

print(f'R1:{R1 / count}')
print(f'R5:{R5 / count}')
print(f'R10:{R10 / count}')
