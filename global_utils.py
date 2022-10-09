import torch
import numpy as np
import scipy.stats
import sys
sys.path.append("..")
sys.path.append("../..")

def get_backbone(name, state_dict=None):
    if name == 'conv4':
        from backbones import conv4
        backbone = conv4()
    elif name == 'resnet12':
        from backbones import resnet12
        backbone = resnet12()
    elif name == 'resnet12_wide':
        from backbones import resnet12_wide
        backbone = resnet12_wide()
    else:
        raise ValueError('Non-supported Backbone.')
    if state_dict is not None:
        backbone.load_state_dict(state_dict)

    return backbone
    
class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    def item(self):
        return self.v

class Mean_confidence_interval():
    def __init__(self, confidence=0.95):
        self.list = []
        self.confidence = confidence
        self.n = 0
    def add(self, x):
        self.list.append(x)
        self.n += 1
    def item(self, return_str=False):
        mean, standard_error = np.mean(self.list), scipy.stats.sem(self.list)
        h = standard_error * scipy.stats.t._ppf((1 + self.confidence) / 2, self.n - 1)
        if return_str:
            return '{0:.2f}; {1:.2f}'.format(mean * 100, h * 100)
        else:
            return mean

def count_acc(logits, labels):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(labels).float())

def get_outputs_c_h(backbone, image_len):
    c_dict = {
        'conv4': 64,
        'resnet12': 512,
        'resnet12_wide': 640,
    }
    c = c_dict[backbone]

    h_devisor_dict = {
        'conv4': 16,
        'resnet12': 16,
        'resnet12_wide': 4,
    }

    h = image_len // h_devisor_dict[backbone]
    if image_len == 84 and h_devisor_dict[backbone] == 8:
        h = 11

    return c, h

def set_reproducibility(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False