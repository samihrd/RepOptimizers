import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from repoptimizer.repoptimizer_utils import RepOptimizerHandler
from repoptimizer.repoptimizer_sgd import RepOptimizerSGD
from repoptimizer.repoptimizer_adamw import RepOptimizerAdamW

num_train_iters = 500
lr = 0.01
momentum = 0.9
weight_decay = 0.1
nest = True

test_scales = (0.233, 0.555)
in_channels = 3
out_channels = 1
in_h, in_w = 8, 8
batch_size = 4

train_data = []
for _ in range(num_train_iters):
    train_data.append(torch.randn(batch_size, in_channels, in_h, in_w))

class TestModel(nn.Module):

    def __init__(self, scales):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        #self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.scales = scales

    def forward(self, x):
        return self.conv1(x) * self.scales[0] + self.conv2(x) * self.scales[1]

def get_equivalent_kernel(model):
    #return model.conv1.weight * test_scales[0] + F.pad(model.conv2.weight * test_scales[1], [1,1,1,1])
    return model.conv1.weight * test_scales[0] + model.conv2.weight * test_scales[1]
    # return {
    #     "weight": model.conv1.weight * test_scales[0] + model.conv2.weight * test_scales[1],
    #     "bias": model.conv1.bias * test_scales[0] + model.conv2.bias * test_scales[1]
    # }

class TestSGDHandler(RepOptimizerHandler):

    #   "model" is simply a 3x3 conv
    def __init__(self, model, scales):
        self.model = model
        self.scales = scales

    def generate_grad_mults(self):
        weight_mask = torch.ones_like(self.model.weight) * self.scales[0] ** 2
        #weight_mask[:, :, 1, 1] += self.scales[1] ** 2
        weight_mask += self.scales[1] ** 2


        bias_mask = torch.zeros_like(self.model.bias)
        bias_mask += self.scales[0] ** 2 + self.scales[1] ** 2

        params = {
            self.model.weight: weight_mask,
            self.model.bias: bias_mask
        }
        return params

class TestAdamWHandler(RepOptimizerHandler):

    #   "model" is simply a 3x3 conv
    def __init__(self, model, scales):
        self.model = model
        self.scales = scales

    def generate_grad_mults(self):
        weight_mask = torch.ones_like(self.model.weight) * self.scales[0]
        weight_mask[:, :, 1, 1] += self.scales[1]
        return {self.model.weight: weight_mask}



def check_equivalency(update_rule):

    assert update_rule in ['sgd', 'adamw']
    print('################################# testing optimizer: ', update_rule)

    model = TestModel(test_scales)
    model.train()

    #   to check the equivalency, we need to record the initial value of the equivalent kernel
    init_weights = get_equivalent_kernel(model)

    if update_rule == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=lr, weight_decay=weight_decay)

    for i in range(num_train_iters):
        x = train_data[i]
        y = model(x)
        optimizer.zero_grad()
        loss = y.var()      #   just an arbitrary loss function.
        loss.backward()
        optimizer.step()

    print('============== finished training the original model')

    eq_model = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
    eq_model.weight.data = init_weights
    #eq_model.bias.data = init_weights["bias"]



    if update_rule == 'sgd':
        handler = TestSGDHandler(eq_model, scales=test_scales)
        eq_optimizer = RepOptimizerSGD(handler.generate_grad_mults(), eq_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        handler = TestAdamWHandler(eq_model, scales=test_scales)
        eq_optimizer = RepOptimizerAdamW(handler.generate_grad_mults(), eq_model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=lr,
                                      weight_decay=weight_decay)

    for i in range(num_train_iters):
        x = train_data[i]
        y = eq_model(x)
        eq_optimizer.zero_grad()
        loss = y.var()
        loss.backward()
        eq_optimizer.step()

    print('============== finished training the equivalent model')
    print('============== the relative difference is ')
    print((eq_model.weight.data - get_equivalent_kernel(model)).abs().sum() / eq_model.weight.abs().sum())


check_equivalency('sgd')
#check_equivalency('adamw')
exit()
