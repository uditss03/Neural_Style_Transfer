import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variable Initialization
imsize = 512
image_folder = ''
style_path = 'style.jpg'
content_path = 'content.jpeg'

loader = transforms.Compose([transforms.Resize((imsize,imsize)), transforms.ToTensor()])

def image_loader(image_path, imsize):
  loader = transforms.Compose([transforms.Resize((imsize,imsize)), transforms.ToTensor()])
  image = Image.open(image_path)
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

def imshow(tensor, title=None):
  unloader = transforms.ToPILImage()
  image = tensor.cpu().clone()
  image = image.squeeze(0)
  image = unloader(image)
  plt.imshow(image)
  if title is not None:
    plt.title(title)
  plt.pause(0.001)

style_img = image_loader(style_path, imsize=imsize)
content_img = image_loader(content_path, imsize=imsize)


assert (style_img.size() == content_img.size(),"same size of both image")

class ContentLoss(nn.Module):
  def __init__(self, target,):
    super(ContentLoss, self).__init__()
    self.target = target.detach()
  def forward(self, input):
    self.loss = F.mse_loss(input, self.target)
    return input

# Dimensions of input of gram matrix = Channels X (height X Width)
# Gram Matrix = matrix_multiplication(activation_output(Image), activation_layer_output(Image).Transpose)
# Dimensions of output of gram matrix = (height X Width) X Channels

def gram_matrix(input):
  a,b,c,d = input.size()
  features = input.view(a*b, c*d)
  G = torch.mm(features, features.t())
  return G.div(a * b * c * d) 

# Style loss = MSE(Gram_matrix(activation_output(Style_Image)), Gram_matrix(activation_output(Gen_Image)))


class StyleLoss(nn.Module):
  def __init__(self, target_feature):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target_feature).detach()
  def forward(self, input):
    G = gram_matrix(input)
    self.loss = F.mse_loss(G, self.target)
    return input

# Helper class for normalization [(x-mean)/std]

class Normalization(nn.Module):
  def __init__(self, mean, std):
    super(Normalization, self).__init__()
    self.mean = torch.tensor(mean).view(-1,1,1)
    self.std = torch.tensor(std).view(-1,1,1)
  def forward(self, img):
    return (img - self.mean)/self.std 


cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def compute_loses(cnn ,normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers, style_layers=style_layers):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0  
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses


input_img = content_img.clone()

def compute_optimizer(input_img):
  optimizer = optim.LBFGS([input_img.requires_grad_()])
  return optimizer


def run(cnn ,normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers, style_layers=style_layers, num_steps = 300, style_weight=10000, content_weight = 1):
  model, style_losses, content_losses = compute_loses(cnn ,normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers, style_layers=style_layers)
  optimizer = compute_optimizer(input_img)
  print('optimizing')
  run = [0]
  while run[0] <= num_steps:
    def closure():
      input_img.data.clamp_(0, 1)
      optimizer.zero_grad()
      model(input_img)
      style_score =0
      content_score = 0 
      for style_layer in style_losses:
        style_score+= (1/5)*style_layer.loss 
      for content_layer in content_losses:
        content_score+= content_layer.loss 
      style_score *= style_weight
      content_score *= content_weight
      loss = style_score + content_score
      loss.backward()
      run[0]+=1
      if run[0] % 5 ==0:
        imshow(input_img, title='Output Image')
        
      if run[0] % 50 ==0:
        print("run {}".format(run))
        print('Style ',style_score.item(),' content ', content_score.item())
        imshow(input_img, title='Output Image')
        print()
      return style_score + content_score
    optimizer.step(closure)

  input_img.data.clamp(0,1)
  return input_img


output = run(cnn ,cnn_normalization_mean, cnn_normalization_std, style_img, content_img, content_layers=content_layers, style_layers=style_layers, num_steps = 300, style_weight=10000, content_weight = 1)
