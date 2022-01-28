import os
import torch 
from resnest.torch import resnest101, resnest50, resnest200
from utils import SAM, LR_Scheduler, get_criterion, LoadingBar, Log, initialize, RandAugment
from dataset import LeukemiaDataset
import json

from torch.utils.data import DataLoader
import torchvision

batch_size = 8
cuda_device_index = 1
rho = 0.05
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.005
warmup_epochs = 2
epochs = 30
n_class = 1 
fold_id = "2" 
root = "/home/jovyan/data/Train/"
num_workers = 1 # workers for dataloader
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
n = 2 # for randaugment
m = 15 # for randaugment
fold_train_path = "./train_folding.json"
fold_valid_path = "./valid_folding.json"

checkpoint_dir = "/home/jovyan/checkpoints/model_resnest50_fold2_basic_smoothL1"
if os.path.isdir(checkpoint_dir) == False:
    os.makedirs(checkpoint_dir)
    
device = torch.device("cuda:" + str(cuda_device_index) if torch.cuda.is_available() else "cpu")
prepath = ""
replacer = ""

model = resnest50(pretrained=True).to(device) 


num_fltrs = model.fc.in_features
model.fc = torch.nn.Linear(num_fltrs, n_class).to(device)

transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

transforms_train.transforms.insert(0, RandAugment(n, m))

transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])


with open(fold_train_path) as fhandle:
    fold_train = json.load(fhandle)
    
with open(fold_valid_path) as fhandle:
    fold_valid = json.load(fhandle)
    
dataset_train = LeukemiaDataset(root=root, 
                         fold_id=fold_id,
                         fold_splitter=fold_train,
                         transforms=transforms_train,
                         prepath=prepath,
                         replacer=replacer)

dataset_valid = LeukemiaDataset(root=root, 
                         fold_id=fold_id,
                         fold_splitter=fold_valid,
                         transforms=transforms_valid, 
                         prepath=prepath,
                         replacer=replacer)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers)

base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

scheduler = LR_Scheduler('cos',
                        base_lr=learning_rate,
                        num_epochs=epochs,
                        iters_per_epoch=len(dataloader_train),
                        warmup_epochs=warmup_epochs)

criterion = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
criterion_eval = torch.nn.L1Loss()
log = Log(log_each=10)

saving_epochs = list(range(epochs))

best_pred = 0

for epoch in range(epochs):
    model.train()
    log.train(len_dataset=len(dataloader_train))
    
    for ix, batch in enumerate(dataloader_train):
        scheduler(optimizer, ix, epoch, best_pred)
        inputs, targets = (b.to(device) for b in batch[:2])
        predictions = model(inputs)
        predictions = torch.squeeze(predictions, dim=1)
        if predictions.shape != targets.shape:
            print("Warning! Different shapes for SmoothL1Loss")
        loss = criterion(predictions, targets)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        criterion(torch.squeeze(model(inputs), dim=1), targets).mean().backward()
        optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            correct = torch.unsqueeze(loss, dim=0)
            log(model, loss.cpu(), correct.cpu(), optimizer.param_groups[0]["lr"])
                
    model.eval()
    log.eval(len_dataset=len(dataloader_valid))

    with torch.no_grad():
        for batch in dataloader_valid:
            inputs, targets = (b.to(device) for b in batch[:2])

            predictions = model(inputs)
            predictions = torch.squeeze(predictions, dim=1)
            if predictions.shape != targets.shape:
                print("Warning! Different shapes for SmoothL1Loss")
            loss = criterion_eval(predictions, targets)
            correct = torch.unsqueeze(loss, dim=0)
            log(model, loss.cpu(), correct.cpu())
            
    if epoch in saving_epochs:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 
                                                    "model_" + fold_id + "_" + str(epoch) + ".pth")
                  )

log.flush()   