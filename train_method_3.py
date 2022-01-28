import os
import torch 
from itertools import chain

from model import DLDModel, DLD2Model, DLD3Model
from utils import SAM, LR_Scheduler, get_criterion, LoadingBar, Log, initialize, RandAugment, SoftArgmax1D
from dataset import LeukemiaDataset
import json

from torch.utils.data import DataLoader
import torchvision


batch_size = 8
cuda_device_index = 2
rho = 0.05
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.005
warmup_epochs = 2
epochs = 50
n_class = 101
fold_id = "2" 
root = "/home/jovyan/data/Train/"
num_workers = 1 # workers for dataloader
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
n = 2 # for randaugment
m = 15 # for randaugment
fold_train_path = "./extended_train_folding.json"
fold_valid_path = "./valid_folding.json"

checkpoint_dir = "/home/jovyan/checkpoints/model_extended_resnest_fold2_KL_&_exp_&_regr_&_hp"
if os.path.isdir(checkpoint_dir) == False:
    os.makedirs(checkpoint_dir)
    
device = torch.device("cuda:" + str(cuda_device_index) if torch.cuda.is_available() else "cpu")
prepath = ""
replacer = ""

model = DLD3Model(n_class, 1).to(device)

class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return torch.nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)
    
model.avgpool = torch.nn.Sequential(
                                    torch.nn.MaxPool2d(kernel_size=2, stride=2), 
                                    GlobalAvgPool2d(),
                                   )

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

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             drop_last=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers)

base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), 
                base_optimizer, rho=rho, lr=learning_rate, momentum=momentum, 
                weight_decay=weight_decay)

scheduler = LR_Scheduler('cos',
                        base_lr=learning_rate,
                        num_epochs=epochs,
                        iters_per_epoch=len(dataloader_train),
                        warmup_epochs=warmup_epochs)

criterion = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
criterion_distribution = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
criterion_eval = torch.nn.L1Loss()
log = Log(log_each=10)
softargmax = SoftArgmax1D(device=device, do_softmax=False)
probabilator = torch.nn.Softmax(dim=1)

saving_epochs = list(range(epochs))

best_pred = 0

for epoch in range(epochs):
    model.train()
    log.train(len_dataset=len(dataloader_train))
    
    for ix, batch in enumerate(dataloader_train):
        scheduler(optimizer, ix, epoch, best_pred)
        inputs, targets1, targets2 = (b.to(device) for b in batch[:3])
        predictions_classes, predictions_regression = model(inputs)
        
        # preparing and making first forward-backward step
        predicted_probabilities = probabilator(predictions_classes)
        expected_value = softargmax(predicted_probabilities)
        loss1 = criterion_distribution(predicted_probabilities.log(), targets1)
        loss2 = criterion(torch.squeeze(predictions_regression, dim=1), targets2)
        loss3 = criterion(expected_value, targets2)
        
        loss = loss1 + loss2 + loss3
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        predictions_2_classes, predictions_2_regression = model(inputs)
        predicted_probabilities_2 = probabilator(predictions_2_classes)
        
        (criterion_distribution(predicted_probabilities_2.log(), targets1) + \
        criterion(torch.squeeze(predictions_2_regression, dim=1), targets2) + \
        criterion(softargmax(predicted_probabilities_2), targets2)).mean().backward()
        optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            correct = torch.unsqueeze(loss2, dim=0)
            log(model, loss1.cpu(), correct.cpu(), optimizer.param_groups[0]["lr"])
                
    model.eval()
    log.eval(len_dataset=len(dataloader_valid))

    with torch.no_grad():
        for batch in dataloader_valid:
            inputs, targets1, targets2 = (b.to(device) for b in batch[:3])
            predictions_classes, predictions_regression = model(inputs)
            predicted_probabilities = probabilator(predictions_classes)
            expected_value = softargmax(predicted_probabilities)
            
            loss1 = criterion_distribution(predicted_probabilities.log(), targets1)
            loss2 = criterion_eval(torch.squeeze(predictions_regression, dim=1), targets2)
            loss3 = criterion_eval(expected_value, targets2)
            
            correct = torch.unsqueeze(loss2, dim=0) 
            log(model, loss1.cpu(), correct.cpu())
            
    if epoch in saving_epochs:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 
                                                    "model_" + fold_id + "_" + str(epoch) + ".pth")
                  )

log.flush()