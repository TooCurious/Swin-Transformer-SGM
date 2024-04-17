import os
import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
from torch.cuda import amp
from PIL import Image
import itertools
import numpy as np
from torchvision.models import resnet50
from Classifier import Classifier
import matplotlib.pyplot as plt

epochs = 10
dir = r'E:\Datasets\University-Release'
batch_size = 10


# dataset
def create_vectors(length):
	vectors = []
	for i in range(length):
		vector = np.zeros(length)
		vector[i] = 1
		vectors.append(vector)
	return vectors


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

dataset = BuildingDataset(root_dir=dir, transform=transform)

result_list = list(itertools.chain.from_iterable(dataset))
vectors = create_vectors(701)
result_vectors = [torch.tensor(x) for x in vectors for _ in range(54)]

result_list = [x + (y,) for x, y in zip(result_list, result_vectors)]

dataloader = data.DataLoader(result_list, batch_size=batch_size, shuffle=True)


class DualBranchModel(nn.Module):
	def __init__(self, backbone, device='cpu', is_train=True, input_size=1000, num_classes=701):
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

backbone = resnet50(weights='IMAGENET1K_V2')
model = DualBranchModel(backbone, device=device).to(device)

# X = torch.randn(8, 3, 256, 256)
# Y = torch.randn(8, 3, 256, 256)
# target = torch.zeros(8, 701)
# target[0, 42] = 1
# dataset = data.TensorDataset(X, Y, target)
# dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)

lossFunc = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scaler = amp.GradScaler()

train_losses = []
sat_losses = []
uav_losses = []
for epoch in range(epochs):
	trainLoss = 0
	sat_l = 0
	uav_l = 0
	for i, (sat_img, uav_img, target) in enumerate(dataloader):
		sat_img = sat_img.to(device)
		uav_img = uav_img.to(device)
		target = target.to(device)
		sat, uav = model(sat_img, uav_img)

		optimizer.zero_grad()
		with amp.autocast():
			sat_loss = lossFunc(sat, target)
			uav_loss = lossFunc(uav, target)
			loss = sat_loss + uav_loss

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		batch_num = i + 1
		if (batch_num) == 1 or batch_num % 100 == 0:
			print("Epoch {}/{}, Batch {}/{}".format(epoch + 1, epochs, batch_num, len(dataloader)))

		trainLoss += loss.item()
		sat_l += sat_loss.item()
		uav_l += uav_loss.item()

	trainLoss = trainLoss / len(dataloader)
	train_losses.append(trainLoss)

	sat_l /= len(dataloader)
	sat_losses.append(sat_l)

	uav_l /= len(dataloader)
	uav_losses.append(uav_l)

	if min(train_losses) == train_losses[-1]:
		torch.save(model.state_dict(), 'model_res.pt')
		print('weight saved')
	print('epoch: [{}/{}]; train loss: {:.3f}'.format(epoch + 1, epochs, trainLoss))


def plot_help1(epochs, y_train_m, title):
	plt.rc('lines', linewidth=2.5)
	fig, ax = plt.subplots()
	ax.set_title(title)
	# Using set_dashes() and set_capstyle() to modify dashing of an existing line.
	line1, = ax.plot(np.linspace(1, epochs, num=epochs), y_train_m, label='train')
	line1.set_dashes([10, 2, 2, 2])  # 10pt line, 2pt break, 2pt line, 2pt break.
	line1.set_dash_capstyle('round')
	ax.scatter(np.linspace(1, epochs, num=epochs), y_train_m)

	# Using plot(..., dashes=..., gapcolor=...) to set the dashing and
	# alternating color when creating a line.

	ax.legend(handlelength=4)
	plt.grid()
	plt.savefig(f'{title}_res.png')
	return fig, ax


def plot_help2(epochs, l1, l2, title):
	plt.rc('lines', linewidth=2.5)
	fig, ax = plt.subplots()
	ax.set_title(title)
	# Using set_dashes() and set_capstyle() to modify dashing of an existing line.
	line1, = ax.plot(np.linspace(1, epochs, num=epochs), l1, label='sat_loss')
	line1.set_dashes([10, 2, 2, 2])  # 10pt line, 2pt break, 2pt line, 2pt break.
	line1.set_dash_capstyle('round')
	ax.scatter(np.linspace(1, epochs, num=epochs), l1)

	# Using plot(..., dashes=...) to set the dashing when creating a line.
	line2, = ax.plot(np.linspace(1, epochs, num=epochs), l2, label='uav_loss')
	ax.scatter(np.linspace(1, epochs, num=epochs), l2)

	# Using plot(..., dashes=..., gapcolor=...) to set the dashing and
	# alternating color when creating a line.

	ax.legend(handlelength=4)
	plt.grid()
	plt.savefig(f'{title}_res.png')
	return fig, ax


title = 'Loss over Epochs'
fig, ax = plot_help1(epochs, train_losses, title)

title = 'Loss components over Epochs'
fig1, ax1 = plot_help2(epochs, sat_losses, uav_losses, title)
