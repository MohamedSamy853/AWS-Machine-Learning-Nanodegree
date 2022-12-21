import json 
import os
import pickle
import sys
import PIL
import io
import torch
import torchvision
from torchvision import transforms ,models
import torch.nn as nn
import torch.nn.functional as F
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

def get_model():
    model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2) 
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.BatchNorm1d(2048),
        nn.Linear(2048 ,4)
    )
    model.eval()
    return model
    
def model_fn(model_dir):
    model = get_model()
    print("model arc is loaded ",type(model))
    path = os.path.join(model_dir , "model.pth")
    with open(path , "rb") as f:
        state_dict = torch.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(state_dict)
    print("model is loaded")
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    print("Data is recieved")
    if content_type ==JPEG_CONTENT_TYPE:
        print("data is loaded in jpeg type")
        return PIL.Image.open(io.BytesIO(request_body))
    
    if content_type == JSON_CONTENT_TYPE:
        print("Data is loaded in json format")
        request = json.loads(request_body)
        data = request['data']
        image =PIL.Image.open(io.BytesIO(data))
        return image
    
def predict_fn(input_data , model):
    print("inside predict function")
    test_transform = transforms.Compose([
    transforms.Resize((300 , 500)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x :x[:3]),
    transforms.Lambda(lambda x :x/255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406] , 
                         std =[0.229, 0.224, 0.225])
                               ])
    image = test_transform(input_data)
    image = image.unsqueeze(0)
    print("image shape is " , image.shape)
    model.eval()
    with torch.no_grad():
        print("calling model")
        prediction = model(image)
    #print(prediction)
    #print(prediction.shape)
    value = prediction.argmax(dim=1).item()
    #print(value)
    maps = { 0:'sunrise', 1:'cloudy', 2:'rain', 3:'shine'}
    type_ = maps[value]
    return json.dumps({"type":type_})

    
    
            
            
        
        
        
    
