import os
import torch 
from resnest.torch import resnest101, resnest50, resnest200
from utils import SAM, LR_Scheduler, get_criterion, LoadingBar, Log, initialize, RandAugment
from dataset import LeukemiaDataset
import json
from sklearn.metrics import classification_report, roc_auc_score

from torch.utils.data import DataLoader
import torchvision
import pandas as pd

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
model_paths = ["/home/jovyan/checkpoints/model_resnext50_fold1_basic_smoothL1/model_1_29.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold2_basic_smoothL1/model_2_29.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold3_basic_smoothL1/model_3_29.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold4_basic_smoothL1/model_4_29.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold5_basic_smoothL1/model_5_29.pth",
               
             "/home/jovyan/checkpoints/model_resnext50_fold1_basic_smoothL1/model_1_28.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold2_basic_smoothL1/model_2_28.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold3_basic_smoothL1/model_3_28.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold4_basic_smoothL1/model_4_28.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold5_basic_smoothL1/model_5_28.pth",
               
             "/home/jovyan/checkpoints/model_resnext50_fold1_basic_smoothL1/model_1_27.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold2_basic_smoothL1/model_2_27.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold3_basic_smoothL1/model_3_27.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold4_basic_smoothL1/model_4_27.pth",
             "/home/jovyan/checkpoints/model_resnext50_fold5_basic_smoothL1/model_5_27.pth",
              ]
               
batch_size = 1
cuda_device_index = 1
n_class = 1 # extend number of classes
fold_id = "0" # the current fold running
root = "/home/jovyan/data/Val/"
num_workers = 1 # workers for dataloader
fold_test_path = "./test_folding.json"
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

from tqdm import tqdm 
d = {}
for model_path in tqdm(model_paths):
    
    model = resnext50_32x4d(pretrained=True).to(device)

    num_fltrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_fltrs, n_class).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for batch in dataloader_test:
            inputs, targets = (b.to(device) for b in batch[:2])
            path = batch[2][0]
            predictions = model(inputs)
            if os.path.basename(path) not in d:
                d[os.path.basename(path)] = []
            score = torch.squeeze(predictions, dim=1).item()
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

with open("./dicts/d_resnext_primordial.json", "w") as fhandle:
    json.dump(d, fhandle)
