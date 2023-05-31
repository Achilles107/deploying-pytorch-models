import argparse
import os
import sys
import time
import re

import numpy as np
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

from transformer_net import MaskNoMaskClassifier
import io
import streamlit as st
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('category.json') as f:
    cat_to_name = json.load(f)

@st.cache
def load_model(model_path):
    print('load model')
    with torch.no_grad():
        style_model = MaskNoMaskClassifier()
        state_dict = torch.load(model_path, map_location=device)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model

@st.cache
def stylize(style_model, content_image):
    #content_image = utils.load_image(content_image)
    transformation = transforms.Compose([transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                                      transforms.Resize((150, 150)),  # resize to input shape of our CNN
                                      transforms.ToTensor()  # convert PIL to Tensor
                                      ])
    content_image = transformation(content_image)
    output = content_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = style_model(output).cpu()
    output_pred = torch.sigmoid(output)
    output_pred = torch.round(output_pred)
    output_pred = str(int(output_pred.item()))
    return cat_to_name[output_pred]
