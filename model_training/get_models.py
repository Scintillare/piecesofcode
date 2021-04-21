import torch
import torch.nn as nn
from torchvision import transforms, datasets, models

def get_resnet(num_classes):
    model = models.resnet101(pretrained=True)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_classes))#, nn.LogSoftmax(dim=1))
    # model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model)
    return model

def get_densenet(num_classes):
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    # model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Softmax(dim=1))
    model_ft.classifier = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_classes))
    return model

def set_parameter_requires_grad(model, feature_extracting):
    '''
    This helper function sets the .requires_grad attribute of the parameters 
    in the model to False when we are feature extracting. By default, when we load 
    a pretrained model all of the parameters have .requires_grad=True, 
    which is fine if we are training from scratch or finetuning.
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False