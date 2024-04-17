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

dataset = BuildingDataset(root_dir=r"C:\Users\Insight\Documents\Обучение\Магистратура\2 семестр\Методы машинного обучения в робототехнике\Курсовая работа\University-Release", transform=transform)

result_list = list(itertools.chain.from_iterable(dataset))
vectors = create_vectors(701)
result_vectors = [torch.tensor(x) for x in vectors for _ in range(54)]

result_list = [x + (y,) for x, y in zip(result_list, result_vectors)]

batch_size = 4
dataloader = data.DataLoader(result_list, batch_size=batch_size, shuffle=True)


# NN
swin_model = swin_t(weights='IMAGENET1K_V1')
backbone = nn.Sequential(*list(swin_model.children())[:-5])


def SGM(f_tens: torch.Tensor, device):
	"""
	Разделение исходного тензора на два - foreground and background

	Input:
	tens: torch.Tensor

	Output:
	Background
	Foreground
	"""

	backgr = torch.zeros(len(f_tens), 768, device=device)
	foregr = torch.zeros(len(f_tens), 768, device=device)

	for i in range(len(f_tens)):
		tens = f_tens[i]
		M_s = torch.sum(tens, dim=1)
		M_border = [min(M_s), max(M_s)]
		M_norm = [(M_s[i] - M_border[0]) / (M_border[1] - M_border[0]) for i in range(len(M_s))]

		sorted_indices = torch.argsort(torch.tensor(M_norm))
		sorted_tens = tens[sorted_indices, :]

		M_norm = sorted(M_norm)

		diff = []
		for j in range(len(M_norm) - 1):
			if M_norm[j] == 0:
				diff.append(0)
			else:
				diff_i = (M_norm[j + 1] - M_norm[j])
				diff.append(diff_i)

		backgr_ = sorted_tens[:torch.argmax(torch.tensor(diff))]
		backgr_ = torch.mean(backgr_, dim=0)
		backgr[i] = backgr_

		foregr_ = sorted_tens[torch.argmax(torch.tensor(diff)):]
		foregr_ = torch.mean(foregr_, dim=0)
		foregr[i] = foregr_

	return backgr, foregr


class Classifier(nn.Module):
	def __init__(self, num_classes, train, input_size=768, hidden_size=512):
		super(Classifier, self).__init__()

		self.fc1 = nn.Linear(input_size, hidden_size)
		self.batch_norm = nn.BatchNorm1d(hidden_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=0.5)
		self.classifier = nn.Linear(hidden_size, num_classes)
		self.softmax = nn.Softmax(dim=0)
		self.train = train

	def forward(self, x):
		x = self.fc1(x)
		if self.train:
			x = x.view(x.size(0), -1)
			try:
				x = self.batch_norm(x)
			except ValueError:
				pass
			x = self.relu(x)
			x = self.dropout(x)
			x = self.classifier(x)
			x = self.softmax(x)
		return x


class DualBranchModel(nn.Module):
	def __init__(self, backbone, train=True, num_classes=701):
		super(DualBranchModel, self).__init__()
		self.backbone_ = backbone
		self.classif = Classifier(num_classes=num_classes, train=train)

	def forward(self, x1, x2):
		x1 = self.backbone_(x1)
		x2 = self.backbone_(x2)

		x1 = x1.reshape(x1.size(0), x1.size(1) * x1.size(2), x1.size(3))
		x2 = x2.reshape(x2.size(0), x2.size(1) * x2.size(2), x2.size(3))

		x1_backgr, x1_foregr = SGM(x1, device)
		x2_backgr, x2_foregr = SGM(x2, device)

		x1_backgr = self.classif(x1_backgr)
		x1_foregr = self.classif(x1_foregr)
		x2_backgr = self.classif(x2_backgr)
		x2_foregr = self.classif(x2_foregr)

		return x1_backgr, x1_foregr, x2_backgr, x2_foregr


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device, '\n')

model = DualBranchModel(backbone).to(device)


# # Создайте входной тензор нужной размерности
# input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)  # Пример размерности изображения
# # Передайте входные данные через модель
# output1, output2, output3, output4 = model(input_tensor, input_tensor)
# # Выведите размерность выхода
# print(output1.size())
# print(output2.size())
# print(output3.size())
# print(output4.size())

# X = torch.randn(100, 3, 256, 256)
# Y = torch.randn(100, 3, 256, 256)
# target = torch.zeros(100, 701)
# target[42] = 1
# dataset = data.TensorDataset(X, Y, target)
# dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

lossFunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = amp.GradScaler()


epochs = 10
train_losses = []
for epoch in range(epochs):
	print('стартуем')
	trainLoss = 0
	for sat_img, uav_img, target in dataloader:
		sat_img = sat_img.to(device)
		uav_img = uav_img.to(device)
		target = target.to(device)
		output1, output2, output3, output4 = model(sat_img, uav_img)

		loss1 = lossFunc(output1, target)
		loss2 = lossFunc(output2, target)
		loss3 = lossFunc(output3, target)
		loss4 = lossFunc(output4, target)
		loss = loss1 + loss2 + loss3 + loss3

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		trainLoss += loss.item()
		print('42')
	trainLoss = trainLoss / len(dataloader)
	train_losses.append(trainLoss)
	print('epoch: [{}/{}]; train loss: {:.3f}'.format(epoch + 1, epochs, trainLoss))
