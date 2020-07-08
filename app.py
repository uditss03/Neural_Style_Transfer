import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy as np

import os
from flask import Flask, render_template, request
from style_transfer import image_loader, run, save_img

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")


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
    
    #solving memory cache 
    name = content_name.split(".")
    content_name = name[0]+str(time.time()) + name[1]
    name = style_name.split(".")
    style_name = name[0]+str(time.time()) + name[1] 

    content_path = "/".join([target_content, content_name])
    style_path = "/".join([target_style, style_name])
    content_img.save(content_path)
    style_img.save(style_path)
    return render_template("upload.html", content_name=content_name, style_name=style_name)


@app.route("/style_transfer")
def style_transfer():
    target_content = os.path.join(APP_ROOT, 'static/content/')
    target_style = os.path.join(APP_ROOT,'static/style/')

    content_name = os.listdir(target_content)
    style_name = os.listdir(target_style)
    
    content_path = "/".join([target_content, content_name[0]])
    style_path = "/".join([target_style, style_name[0]])

    style_img = image_loader(style_path, imsize=512)
    content_img = image_loader(content_path, imsize=512)
    input_img = content_img.clone()

    assert (style_img.size() == content_img.size(),"same size of both image")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    output = run(cnn ,cnn_normalization_mean, cnn_normalization_std, style_img, content_img, input_img, content_layers=content_layers, style_layers=style_layers, num_steps = 50, style_weight=1000000, content_weight = 1)

    #solving memory cache
    for filename in os.listdir('static/result/'):
        if filename.startswith('result'):  # not to remove other images
            os.remove('static/result/' + filename)
    filename = "result" + str(time.time()) + ".jpg"
    target_path = os.path.join(APP_ROOT,"static/result/"+filename)
    save_img(target_path, output)
    return render_template("style_transfer.html", image_name = filename)



if __name__ == "__main__":
    app.run(debug=True)
