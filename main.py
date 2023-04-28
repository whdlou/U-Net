import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision
from data import CustomDataset
import cfg
import tools
from models.unet import UNet
from train import train
from inference import inference


device = tools.get_device()
train_transforms = transforms.Compose(
    [transforms.ToImageTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(15),
     transforms.Resize(cfg.size, antialias=True),
     transforms.CenterCrop(cfg.size)]
)
eval_transforms = transforms.Compose(
    [transforms.ToImageTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.Resize(cfg.size, antialias=True),
     transforms.CenterCrop(cfg.size)]
)
train_set = CustomDataset(mode='train', transforms=train_transforms)
data = train_set[5]

val_set = CustomDataset(mode='val', transforms=eval_transforms)
test_set = CustomDataset(mode='test', transforms=eval_transforms)
model = UNet(cfg.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.CrossEntropyLoss()
# train(model=model,
#       train_set=train_set,
#       val_set=val_set,
#       device=device,
#       optimizer=optimizer,
#       criterion=criterion)
model.load_state_dict(torch.load(cfg.weight))
inference(model=model,
          test_set=test_set,
          device=tools.get_device())
