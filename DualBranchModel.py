import torch
from torch import nn
from Classifier import Classifier

class DualBranchModel(nn.Module):
	def __init__(self, backbone, device='cpu', is_train=True, num_classes=701):
		super(DualBranchModel, self).__init__()
		self.device = device
		self.backbone_ = backbone
		self.classif = Classifier(num_classes=num_classes, is_train=is_train)

	def forward(self, x1, x2):
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

				backgr_ = sorted_tens[torch.argmax(torch.tensor(diff)):]
				backgr_ = torch.mean(backgr_, dim=0)
				backgr[i] = backgr_

				foregr_ = sorted_tens[:torch.argmax(torch.tensor(diff))]
				foregr_ = torch.mean(foregr_, dim=0)
				foregr[i] = foregr_

			return backgr, foregr

		x1 = self.backbone_(x1)
		x2 = self.backbone_(x2)

		x1 = x1.reshape(x1.size(0), x1.size(1) * x1.size(2), x1.size(3))
		x2 = x2.reshape(x2.size(0), x2.size(1) * x2.size(2), x2.size(3))

		x1_backgr, x1_foregr = SGM(x1, self.device)
		x2_backgr, x2_foregr = SGM(x2, self.device)

		x1_backgr = self.classif(x1_backgr)
		x1_foregr = self.classif(x1_foregr)
		x2_backgr = self.classif(x2_backgr)
		x2_foregr = self.classif(x2_foregr)

		return x1_backgr, x1_foregr, x2_backgr, x2_foregr