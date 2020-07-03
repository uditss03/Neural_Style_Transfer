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

import os
from flask import Flask, render_template, request
from style_transfer import image_loader, run

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")
'''
@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/uploads')
    if not os.path.isdir(target):
        os.mkdir(target)
    for f in os.listdir(target):
        os.remove(os.path.join(target,f))
    content_img = request.files.get("content_file")
    style_img = request.files.get("style_file")

    content_name = content_img.filename
    style_name = style_img.filename

    name,content_ext = os.path.splitext(content_name)
    print(content_ext)
    name,style_ext = os.path.splitext(style_name)
    print(style_ext)
    content_path = "/".join([target, content_name])
    style_path = "/".join([target, style_name])
    content_img.save(content_path)
    style_img.save(style_path)

    

    return render_template("upload.html", content_name=content_name, style_name=style_name)
'''

@app.route("/upload", methods=['POST'])
def upload():
    target_content = os.path.join(APP_ROOT, 'static/content')
    if not os.path.isdir(target_content):
        os.mkdir(target_content)
    for f in os.listdir(target_content):
        os.remove(os.path.join(target_content,f))
    target_style = os.path.join(APP_ROOT, 'static/style')
    if not os.path.isdir(target_style):
        os.mkdir(target_style)
    for f in os.listdir(target_style):
        os.remove(os.path.join(target_style,f))
    content_img = request.files.get("content_file")
    style_img = request.files.get("style_file")

    content_name = content_img.filename
    style_name = style_img.filename

    content_path = "/".join([target_content, content_name])
    style_path = "/".join([target_style, style_name])
    content_img.save(content_path)
    style_img.save(style_path)
    return render_template("upload.html", content_name=content_name, style_name=style_name)

@app.route("/style_transfer")
def style_transfer():
    target_content = os.path.join(APP_ROOT, 'static/content/')
    target_style = os.path.join(APP_ROOT,'static/style/')
    
    target_img = os.path.join(APP_ROOT, 'static/result/')
    if not os.path.isdir(target_img):
        os.mkdir(target_img)
    for f in os.listdir(target_img):
        os.remove(os.path.join(target_img,f))
    

    content_name = os.listdir(target_content)
    style_name = os.listdir(target_style)
    
    content_path = "/".join([target_content, content_name[0]])
    style_path = "/".join([target_style, style_name[0]])

    style_img = image_loader(style_path, imsize=512)
    content_img = image_loader(content_path, imsize=512)
    input_img = content_img.clone()

    #assert (style_img.size() == content_img.size(),"same size of both image")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    output = run(cnn ,cnn_normalization_mean, cnn_normalization_std, style_img, content_img, input_img, content_layers=content_layers, style_layers=style_layers, num_steps = 300, style_weight=10000, content_weight = 1)
    #output.save(content_path+"/result.img")
    
    

    return render_template("style_transfer.html")

if __name__ == "__main__":
    app.run(debug=True)
