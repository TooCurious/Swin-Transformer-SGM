import torch
from torch import nn
from torchvision.models import swin_t
from DualBranchModel import DualBranchModel


swin_model = swin_t()
backbone = nn.Sequential(*list(swin_model.children())[:-5])
model = DualBranchModel(backbone, is_train=False)
model.load_state_dict(torch.load('model.pt'))
model.eval()

batch_size = 1
input_tensor = torch.randn(batch_size, 3, 256, 256)
output1, output2, output3, output4 = model(input_tensor, input_tensor)
print(output1.size())
print(output2.size())
print(output3.size())
print(output4.size())