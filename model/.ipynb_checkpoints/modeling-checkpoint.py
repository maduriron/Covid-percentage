from resnest.torch import resnest50, resnest101, resnest200
from torchvision.models import resnext50_32x4d
#from efficientnet_pytorch import EfficientNet
#import timm

import torch 
from itertools import chain

class DLDModel(torch.nn.Module):
    def __init__(self, n_class_distribution, n_regression):
        super(DLDModel, self).__init__()
        #self.model = resnext50_32x4d(pretrained=True)
        self.model = resnest101(pretrained=True)
        # timm.create_model('resnest50d_4s2x40d', pretrained=True)
        num_fltrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_fltrs, n_class_distribution)
        self.regressor = torch.nn.Linear(n_class_distribution, n_regression)
        
    def forward(self, x):
        class_scores = self.model(x)
        regression_score = self.regressor(class_scores)
        return class_scores, regression_score
    
class DLD2Model(torch.nn.Module):
    def __init__(self, n_class_distribution, n_regression):
        super(DLD2Model, self).__init__()
        self.model = resnest50(pretrained=True)
        num_fltrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_fltrs, n_class_distribution)
        self.regressor = torch.nn.Linear(n_class_distribution, n_regression)
        self.probabilator = torch.nn.Softmax(dim=-1)
        
    def forward(self, x):
        class_scores = self.model(x)
        class_scores = self.probabilator(class_scores)
        regression_score = self.regressor(class_scores)
        return class_scores, regression_score
    
class DLD3Model(torch.nn.Module):
    def __init__(self, n_class_distribution, n_regression):
        super(DLD3Model, self).__init__()
        self.model = resnest50(pretrained=True)
        num_fltrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_fltrs, n_class_distribution)
        self.regressor = torch.nn.Linear(n_class_distribution, n_regression)
        
    def forward(self, x):
        class_scores = self.model(x)
        regression_score = self.regressor(class_scores)
        return class_scores, regression_score