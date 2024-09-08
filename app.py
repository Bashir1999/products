import cv2
import gradio as gr
import os
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.system("git clone https://github.com/xuebinqin/DIS")
os.system("mv DIS/IS-Net/* .")

from data_loader_cache import normalize, im_reader, im_preprocess 
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")
    os.system("mv isnet.pth saved_models/")

class GOSNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image

transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])

def load_image(im_path, hypar):
    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)

def build_model(hypar, device):
    net = hypar["model"]
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)
    if hypar["restore_model"] != "":
        net.load_state_dict(torch.load(hypar["model_path"] + "/" + hypar["restore_model"], map_location=device))
    net.eval()
    return net

def predict(net, inputs_val, shapes_val, hypar, device):
    net.eval()
    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)
    ds_val = net(inputs_val_v)[0]
    pred_val = ds_val[0][0, :, :, :]
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)

hypar = {}
hypar["model_path"] = "./saved_models"
hypar["restore_model"] = "isnet.pth"
hypar["interm_sup"] = False
hypar["model_digit"] = "full"
hypar["seed"] = 0
hypar["cache_size"] = [1024, 1024]
hypar["input_size"] = [1024, 1024]
hypar["crop_size"] = [1024, 1024]
hypar["model"] = ISNetDIS()

net = build_model(hypar, device)

def inference(image):
    image_path = image
    image_tensor, orig_size = load_image(image_path, hypar)
    mask = predict(net, image_tensor, orig_size, hypar, device)
    pil_mask = Image.fromarray(mask).convert('L')
    im_rgb = Image.open(image).convert("RGB")
    im_rgba = im_rgb.copy()
    im_rgba.putalpha(pil_mask)
    return [im_rgba, pil_mask]

# Functions Added From Team
def rotate_image(image, degrees):
    img = Image.open(image).rotate(degrees)
    return img

def resize_image(image, width, height):
    img = Image.open(image).resize((width, height))
    return img

def convert_to_grayscale(image):
    img = Image.open(image).convert('L')
    return img

def adjust_brightness(image, brightness_factor):
    img = Image.open(image)
    enhancer = ImageEnhance.Brightness(img)
    img_enhanced = enhancer.enhance(brightness_factor)
    return img_enhanced

# Custom CSS Added From Team
custom_css = """
body {
    background-color: #f0f0f0;
}
.gradio-container {
    max-width: 900px;
    margin: auto;
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.2);
}
button.lg {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    transition-duration: 0.4s;
    cursor: pointer;
    border-radius: 8px;
}
button.lg:hover {
    background-color: #45a049;
    color: white;
}
"""

# Used Some Codes From Yang's Chatbot
with gr.Blocks(css=custom_css) as interface:
    gr.Markdown(f"# {title}")
    gr.Markdown("<h1 style='text-align: center;'>🚩 Image Processor with Brightness Adjustment 🚩</h1>")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type='filepath')
            rotate_button = gr.Button("Rotate Image")
            resize_button = gr.Button("Resize Image")
            grayscale_button = gr.Button("Convert to Grayscale")
            brightness_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Adjust Brightness")
            submit_button = gr.Button("Submit", variant="primary")
            clear_button = gr.Button("Clear", variant="secondary")
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            mask_image = gr.Image(label="Mask")

    # AI Generated: Use Gradio Blocks to organize the interface with buttons
    rotate_button.click(rotate_image, inputs=[input_image, gr.Slider(minimum=0, maximum=360, step=1, value=90, label="Rotation Degrees")], outputs=output_image)
    resize_button.click(resize_image, inputs=[input_image, gr.Number(value=512, label="Width"), gr.Number(value=512, label="Height")], outputs=output_image)
    grayscale_button.click(convert_to_grayscale, inputs=input_image, outputs=output_image)
    
    brightness_slider.change(adjust_brightness, inputs=[input_image, brightness_slider], outputs=output_image)

    submit_button.click(inference, inputs=input_image, outputs=[output_image, mask_image])

    clear_button.click(lambda: (None, None, None), inputs=None, outputs=[input_image, output_image, mask_image])

interface.launch(share=True)
