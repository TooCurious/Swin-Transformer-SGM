import os
import torch
from torch import nn
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import swin_t
import torch.optim as optim
from torch.cuda import amp
import itertools
import numpy as np
from DualBranchModel import DualBranchModel
import matplotlib.pyplot as plt

epochs = 10
dir = r'E:\Datasets\University-Release'
batch_size = 6


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

# NN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device, '\n')

swin_model = swin_t(weights='IMAGENET1K_V1')
backbone = nn.Sequential(*list(swin_model.children())[:-5])
model = DualBranchModel(backbone, device=device).to(device)

# X = torch.randn(8, 3, 256, 256)
# Y = torch.randn(8, 3, 256, 256)
# target = torch.zeros(8, 701)
# target[0, 42] = 1
# dataset = data.TensorDataset(X, Y, target)
# dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)

lossFunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = amp.GradScaler()


train_losses = []
sat_backgr_losses = []
sat_foregr_losses = []
uav_backgr_losses = []
uav_foregr_losses = []
for epoch in range(epochs):
	trainLoss = 0
	sat_backgr_l = 0
	sat_foregr_l = 0
	uav_backgr_l = 0
	uav_foregr_l = 0
	for i, (sat_img, uav_img, target) in enumerate(dataloader):
		sat_img = sat_img.to(device)
		uav_img = uav_img.to(device)
		target = target.to(device)
		sat_backgr_out, sat_foregr_out, uav_backgr_out, uav_foregr_out = model(sat_img, uav_img)

		optimizer.zero_grad()
		with amp.autocast():
			sat_backgr_loss = lossFunc(sat_backgr_out, target)
			sat_foregr_loss = lossFunc(sat_foregr_out, target)
			uav_backgr_loss = lossFunc(uav_backgr_out, target)
			uav_foregr_loss = lossFunc(uav_foregr_out, target)
			loss = sat_backgr_loss + sat_foregr_loss + uav_backgr_loss + uav_foregr_loss

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		batch_num = i + 1
		if (batch_num) == 1 or batch_num % 100 == 0:
			print("Epoch {}/{}, Batch {}/{}".format(epoch + 1, epochs, batch_num, len(dataloader)))

		trainLoss += loss.item()
		sat_backgr_l += sat_backgr_loss.item()
		sat_foregr_l += sat_foregr_loss.item()
		uav_backgr_l += uav_backgr_loss.item()
		uav_foregr_l += uav_foregr_loss.item()

	trainLoss = trainLoss / len(dataloader)
	train_losses.append(trainLoss)

	sat_backgr_l /= len(dataloader)
	sat_backgr_losses.append(sat_backgr_l)

	sat_foregr_l /= len(dataloader)
	sat_foregr_losses.append(sat_foregr_l)

	uav_backgr_l /= len(dataloader)
	uav_backgr_losses.append(uav_backgr_l)

	uav_foregr_l /= len(dataloader)
	uav_foregr_losses.append(uav_foregr_l)

	if min(train_losses) == train_losses[-1]:
		torch.save(model.state_dict(), 'model.pt')
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
	plt.savefig(f'{title}.png')
	return fig, ax


def plot_help2(epochs, l1, l2, l3, l4, title):
	plt.rc('lines', linewidth=2.5)
	fig, ax = plt.subplots()
	ax.set_title(title)
	# Using set_dashes() and set_capstyle() to modify dashing of an existing line.
	line1, = ax.plot(np.linspace(1, epochs, num=epochs), l1, label='sat_backgr_loss')
	line1.set_dashes([10, 2, 2, 2])  # 10pt line, 2pt break, 2pt line, 2pt break.
	line1.set_dash_capstyle('round')
	ax.scatter(np.linspace(1, epochs, num=epochs), l1)

	# Using plot(..., dashes=...) to set the dashing when creating a line.
	line2, = ax.plot(np.linspace(1, epochs, num=epochs), l2, label='sat_foregr_loss')
	ax.scatter(np.linspace(1, epochs, num=epochs), l2)

	line3, = ax.plot(np.linspace(1, epochs, num=epochs), l3, label='uav_backgr_loss')
	line3.set_dashes([10, 2, 2, 2])  # 10pt line, 2pt break, 2pt line, 2pt break.
	line3.set_dash_capstyle('round')
	ax.scatter(np.linspace(1, epochs, num=epochs), l3)

	line4, = ax.plot(np.linspace(1, epochs, num=epochs), l4, label='uav_foregr_loss')
	ax.scatter(np.linspace(1, epochs, num=epochs), l4)

	# Using plot(..., dashes=..., gapcolor=...) to set the dashing and
	# alternating color when creating a line.

	ax.legend(handlelength=4)
	plt.grid()
	plt.savefig(f'{title}.png')
	return fig, ax


title = 'Loss over Epochs'
fig, ax = plot_help1(epochs, train_losses, title)

title = 'Loss components over Epochs'
fig1, ax1 = plot_help2(epochs, sat_backgr_losses, sat_foregr_losses, uav_backgr_losses, uav_foregr_losses, title)
