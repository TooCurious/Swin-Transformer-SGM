from torch import nn


class Classifier(nn.Module):
	def __init__(self, num_classes, is_train, input_size=768, hidden_size=512):
		super(Classifier, self).__init__()

		self.fc1 = nn.Linear(input_size, hidden_size)
		self.batch_norm = nn.BatchNorm1d(hidden_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=0.5)
		self.classifier = nn.Linear(hidden_size, num_classes)
		self.softmax = nn.Softmax(dim=0)
		self.is_train = is_train

	def forward(self, x):
		x = self.fc1(x)
		if self.is_train:
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
