import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms ,models
import argparse
import smdebug
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook
import os
import re
import numpy as np
def get_model():
    model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False 
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.BatchNorm1d(2048),
        nn.Linear(2048 ,4)
    )
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    return model , optimizer ,loss_fn

class DataSet(torch.utils.data.Dataset):
    def __init__(self, path,transform):
        from random import shuffle , seed
        maps = {'sunrise': 0, 'cloudy': 1, 'rain': 2, 'shine': 3}
        self.path = path
        self.images = os.listdir(path)
        self.labels = self._get_labels()
        self.transform  = transform
        self.labels = [maps[i] for i in self.labels]
        seed(42)
        shuffle(self.images)
        shuffle(self.labels)
        self.labels = torch.Tensor(self.labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self ,index):
        image_dir = os.path.join(self.path , self.images[index])
        image = plt.imread(image_dir)
        image=image.astype(np.uint8)
        image = self.transform(image)
        return image ,self.labels[index]
    def _get_labels(self):
        labels =[]
        for file in os.listdir(self.path):
            label =re.findall(r"([a-zA-Z]+)\d+\.j\w+g" ,file)[0]
            labels.append(label)
        return labels
        
train_transform = transforms.Compose([transforms.ToPILImage(),
                                
                                transforms.Resize((300 , 500)),
                                #transforms.RandomCrop((340,510)),
                                #transforms.RandomAffine(0, scale=(0.8,1.2)),
                                #transforms.RandomHorizontalFlip(),
                                #transforms.RandomRotation((-30 ,30)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x :x[:3]),
                                transforms.Lambda(lambda x :x/255.0),
                                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                               ])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Resize((300 , 500)),
                                transforms.Lambda(lambda x :x[:3]),
                                transforms.Lambda(lambda x :x/255.0),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406] , 
                                                     std =[0.229, 0.224, 0.225])
                               ])
 
    

def get_data_loader(train_path , valid_path):
    train_set = DataSet(train_path , train_transform)
    valid_set = DataSet(valid_path ,valid_transform)
    train_loader = torch.utils.data.DataLoader(train_set , batch_size=64 , shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set ,batch_size=len(valid_set) ,shuffle=True)
    return train_loader , valid_loader

def train_batch(model , x , y ,loss_fn , optimizer):
    model.train()
    y_pred = model(x)
    #add l1 regularization
    l1_reg = 0
    for param in model.parameters():
        l1_reg+=torch.norm(param,1)
    #calc loss
    loss = loss_fn(y_pred , y)+0.0001*l1_reg
    #calc gradients
    loss.backward()
    #make update 
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
@torch.no_grad()
def accuracy(model , x , y):
    model.eval()
    y_pred = model(x)
    y_pred = y_pred.argmax(dim=1)
    correct = y_pred == y
    acc = correct.float().mean().item()
    return acc

@torch.no_grad()
def calc_loss(model, x, y, loss_fn):
    model.eval()
    y_pred = model(x)
    loss = loss_fn(y_pred , y)
    return loss.item()
def train(model , train_loader , test_loader , loss_fn , optimizer , epochs,device):
    model.train()
    train_acc , test_acc =[], []
    train_loss , test_loss =[] , []
    for e in range(epochs):
        #hook for monitor training
        hook = get_hook(create_if_not_exists=True)
        if hook:
            hook.set_mode(modes.TRAIN)
            hook.register_loss(loss_fn)
        print(f"Epoch {e+1} training...")
        train_loss_epoch , train_acc_epoch =[],[]
        for x , y in train_loader:
            y = y.type(torch.LongTensor)
            x , y =  x.to(device), y.to(device)
            loss = train_batch(model , x, y, loss_fn , optimizer)
            acc = accuracy(model , x, y)
            train_acc_epoch.append(acc)
            train_loss_epoch.append(loss)
        train_acc.append(np.array(train_acc_epoch).mean())
        train_loss.append(np.array(train_loss_epoch).mean())
        #hook for validation
        hook = get_hook(create_if_not_exists=True)
        if hook :
            hook.set_mode(modes.EVAL)
            hook.register_loss(loss_fn)
        for x , y in test_loader:
            y = y.type(torch.LongTensor)
            x , y =  x.to(device), y.to(device)
            loss = calc_loss(model , x, y , loss_fn)
            acc = accuracy(model , x, y)
        test_acc.append(acc)
        test_loss.append(loss)
        print(f"Epoch {e+1} Train Loss : {train_loss[-1]:0.3f} Train Acc : {train_acc[-1]:0.2f} Valid Loss : {test_loss[-1]:0.2f} Valid Acc :{test_acc[-1]:0.2f}")
    return train_acc , test_acc , train_loss , test_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train" , type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--valid" , type=str, default=os.environ['SM_CHANNEL_VALID'])
    parser.add_argument("--epochs",type = int , default =60)
    parser.add_argument("--gpu",type = bool , default =True)
    args = parser.parse_args()
    model , optimizer ,loss_fn = get_model()
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_hook(model)
        hook.register_loss(loss_fn)
    device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    print(device +" is the device that used")
    model = model.to(device)
    print("Data start loading...")
    train_loader , test_loader = get_data_loader(args.train , args.valid)
    print("Data is loaded")
    train(model , train_loader , test_loader , loss_fn , optimizer , args.epochs, device)
    import pickle
    arch_path =  os.path.join(os.environ["SM_MODEL_DIR"] , "model_arch.pkl")
    with open(arch_path , "wb") as f:
        pickle.dump(model.cpu() , f)
    path = os.path.join(os.environ["SM_MODEL_DIR"] , "model.pth")
    torch.save(model.cpu().state_dict(),path)
        
if __name__ =="__main__":
    main()
    
    
    
    
    


