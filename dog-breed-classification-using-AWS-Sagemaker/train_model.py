#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import os
import cv2
######sagemaker debuder ######3
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook
#####sagemaker profiler#####3
@torch.no_grad()
def test(model, test_loader, loss_fn):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    losses =[]
    accuracy = []
    model.eval()
    #test hook to register events
    hook = get_hook(create_if_not_exists=True)
    if hook :
        hook.set_mode(modes.EVAL)
        hook.register_loss(loss_fn)
    for x_batch , y_batch in test_loader:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred , y_batch)
        losses.append(loss.item())
        max_values, argmaxes = y_pred.max(-1)
        is_correct = argmaxes == y_batch
        accuracy.append(is_correct.cpu().numpy().mean())
    loss = np.array(losses).mean()
    acc = np.array(accuracy).mean()
    return loss , acc 
    

def train(model, train_loader, criterion, optimizer):
    """
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    """
  
    model.train()
    #create for train hook
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.set_mode(modes.TRAIN)
        hook.register_loss(criterion)
    print("Start training")
    epoch_losses = []
    epoch_acc = []
    all_pathces =len(train_loader)
    for x_batch , y_batch in train_loader:
        y_pred = model(x_batch)
        loss = criterion(y_pred , y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_losses.append(loss.item())
        max_values, argmaxes = y_pred.max(-1)
        is_correct = argmaxes == y_batch
        epoch_acc.append(is_correct.cpu().numpy().mean())
        
    return np.array(epoch_losses).mean() , np.array(epoch_acc).mean()



def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = torchvision.models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False 
    num_features=model.fc.in_features
    model.fc = torch.nn.Sequential( torch.nn.Dropout(0.5),
                                  torch.nn.Linear(num_features , 133)   )
    return model


class DogBreed(Dataset):
    def __init__(self , data_path):
        self.targets = os.listdir(data_path)
        self.y = []
        self.x =[]
        for i ,target in enumerate(self.targets):
            for  file in os.listdir(os.path.join(data_path , target)):
                self.x.append(os.path.join(data_path,target,file))
                self.y.append(i)
        self.y = torch.Tensor(self.y).long()
        np.random.seed(42)
        np.random.shuffle(self.x)
        np.random.shuffle(self.y)
        self.x = self.x[:500]
        self.y = self.y[:500]
    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        sample = self.x[index]
        target = self.y[index]
        image = cv2.imread(sample)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image , (600 ,600))
        image = torch.Tensor(image)/255.0
        image = image.float().permute(2 ,0 ,1)
        return image , target
    

def create_data_loaders(train_path , test_path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_set = DogBreed(train_path)
    test_set = DogBreed(test_path)
    train_loader = DataLoader(train_set , batch_size=batch_size , shuffle=True)
    test_loader = DataLoader(test_set , batch_size = batch_size )
    return train_loader , test_loader
    

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type = int , default =2 )
    parser.add_argument("--batch-size" , type=int , default = 64)
    parser.add_argument("--lr" , type = float , default = 0.001 )
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_VALID'])
    args = parser.parse_args()
    model=net()
   
    loss_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    #create hook 
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_hook(model)
        hook.register_loss(loss_criterion)
    print("train path ", args.train)
    print("test path ", args.test)
    print(os.environ['SM_CHANNEL_TRAIN'])
    print(os.environ['SM_CHANNEL_VALID'])
    
    
    train_loader , test_loader = create_data_loaders(args.train , args.test, args.batch_size)
    print("Data is Loaded ")
    for e in range(args.epochs):
        train_loss , train_acc = train(model, train_loader, loss_criterion, optimizer )
        valid_loss , valid_acc = test(model, test_loader, loss_criterion )
        print(f"Epoch {e+1} Train:loss = {train_loss} Train:acc = {train_acc*100:0.4f} Valid:loss = {valid_loss} Valid:acc = {valid_acc*100}")
    path = os.path.join(os.environ["SM_MODEL_DIR"] , "model.pth")
    torch.save(model.state_dict(),path)
    
if __name__=='__main__':

    main()
