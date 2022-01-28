import os
import torch 
from resnest.torch import resnest101, resnest50, resnest200
from model import DLDModel, DLD3Model
from utils import SAM, LR_Scheduler, get_criterion, LoadingBar, Log, initialize, RandAugment, SoftArgmax1D
from dataset import LeukemiaDataset
import json
from sklearn.metrics import classification_report, roc_auc_score

from torch.utils.data import DataLoader
import torchvision
import pandas as pd

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
model_paths = ["/home/jovyan/checkpoints/model_resnest_fold5_KL_&_exp_&_regr_&_hp/model_5_49.pth",
              "/home/jovyan/checkpoints/model_resnest_fold5_KL_&_exp_&_regr_&_hp/model_5_48.pth",
               "/home/jovyan/checkpoints/model_resnest_fold5_KL_&_exp_&_regr_&_hp/model_5_47.pth",
               "/home/jovyan/checkpoints/model_resnest_fold5_KL_&_exp_&_regr_&_hp/model_5_46.pth",
               "/home/jovyan/checkpoints/model_resnest_fold5_KL_&_exp_&_regr_&_hp/model_5_45.pth",

               "/home/jovyan/checkpoints/model_resnest_fold2_KL_&_exp_&_regr_&_hp/model_2_49.pth",
               "/home/jovyan/checkpoints/model_resnest_fold2_KL_&_exp_&_regr_&_hp/model_2_48.pth",
               "/home/jovyan/checkpoints/model_resnest_fold2_KL_&_exp_&_regr_&_hp/model_2_47.pth",
               "/home/jovyan/checkpoints/model_resnest_fold2_KL_&_exp_&_regr_&_hp/model_2_46.pth",
               "/home/jovyan/checkpoints/model_resnest_fold2_KL_&_exp_&_regr_&_hp/model_2_45.pth",
              ]
               
batch_size = 1
cuda_device_index = 0
n_class = 101 # extend number of classes
fold_id = "0" # the current fold running
root = "/home/jovyan/data/Test/"
num_workers = 1 # workers for dataloader
fold_test_path = "./final_test_folding.json"
device = torch.device("cuda:" + str(cuda_device_index) if torch.cuda.is_available() else "cpu")
prepath = ""
replacer = ""

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 512)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])
    
with open(fold_test_path) as fhandle:
    fold_test = json.load(fhandle)
    
dataset_test = LeukemiaDataset(root=root, 
                         fold_id=fold_id,
                         fold_splitter=fold_test,
                         transforms=transforms_test, 
                         prepath=prepath,
                         replacer=replacer)

dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print("Validation dataset has size {}".format(len(dataset_test)))

from torchvision.models import resnext50_32x4d

softargmax = SoftArgmax1D(device=device)

class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return torch.nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

from tqdm import tqdm 
d = {}
for model_path in tqdm(model_paths):
    model = DLD3Model(n_class, 1).to(device)
    model.avgpool = torch.nn.Sequential(
                                        torch.nn.MaxPool2d(kernel_size=2, stride=2), 
                                        GlobalAvgPool2d(),
                                       )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for batch in dataloader_test:
            inputs, targets = (b.to(device) for b in batch[:2])
            path = batch[3][0]
            prediction_classes, prediction_regression = model(inputs)
            
            score = torch.squeeze(prediction_regression, dim=1).item() # score predicted
            
            if os.path.basename(path) not in d:
                d[os.path.basename(path)] = []
                
            int_score = int(score) 
            if score <= 0:
                final_score = 0
            if score - int_score >= 0.5:
                final_score = int_score + 1
            elif (int_score <= score) and (score - int_score < 0.5):
                final_score = int_score
            else:
                if score > 0:
                    print("OMG", score, int_score)
                    
            d[os.path.basename(path)].append(final_score)
            
submission_d = {"filename": [], "percentage": []}
for key in d:
    submission_d["filename"].append(key)
    percentage = float(int(sum(d[key]) / len(d[key])))
    submission_d["percentage"].append(percentage)
    
submission_data = pd.DataFrame(data=submission_d)
submission_data.to_csv("predictions.csv", header=False, index=False)

with open("./dicts/d_resnest_KL&exp&regr&hp_1_05_finaltest.json", "w") as fhandle:
    json.dump(d, fhandle)