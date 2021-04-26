import torch
from models import PreActResNet18
model_path = './model_resnet18_trades/resnet18_trades.pth'
model = PreActResNet18().cuda()
model.load_state_dict(torch.load(model_path))

# Evaluate the Linf robustness of the model using AutoAttack
# autoattack is installed as a dependency of robustbench so there is not need to install it separately
from robustbench.data import load_cifar10
x_test, y_test = load_cifar10(n_examples=50)

from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255)
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test)