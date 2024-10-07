import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image, ImageEnhance
import torch
import os
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

# --- Model 1: AI Chatbot Setup ---
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3") # HuggingFaceH4/zephyr-7b-beta

# Personalities for AI Chatbot
PERSONALITIES = {
    "Friendly": "You are a friendly and helpful assistant.",
    "Professional": "You are a professional and concise assistant.",
    "Humorous": "You are a witty and humorous assistant.",
    "Empathetic": "You are a compassionate and empathetic assistant."
}

# Chatbot Functions
def respond(message, history, personality):
    system_message = PERSONALITIES[personality]
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    response = client.chat_completion(messages, max_tokens=1024)
    bot_message = response["choices"][0]["message"]["content"]
    history.append((message, bot_message))
    return history, ""

def generate_fun_fact(history):
    message = "Give me a fun fact."
    system_message = "You are a helpful assistant that shares fun facts when asked."
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    response = client.chat_completion(messages, max_tokens=256)
    fun_fact = response["choices"][0]["message"]["content"]
    history.append((message, fun_fact))
    return history

def generate_daily_challenge(history):
    message = "Give me a daily challenge."
    system_message = "You are a helpful assistant that gives fun or motivational daily challenges."
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    response = client.chat_completion(messages, max_tokens=256)
    challenge = response["choices"][0]["message"]["content"]
    history.append((message, challenge))
    return history

def generate_inspiration(history):
    message = "Give me an inspirational quote or motivational message."
    system_message = "You are a helpful assistant that provides inspiring or motivational quotes when asked."
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    response = client.chat_completion(messages, max_tokens=256)
    inspiration = response["choices"][0]["message"]["content"]
    history.append((message, inspiration))
    return history

def clear_conversation():
    return [], ""




#######

os.system("git clone https://github.com/xuebinqin/DIS")
os.system("mv DIS/IS-Net/* .")

from data_loader_cache import normalize, im_reader, im_preprocess 
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")
    os.system("mv /home/student-admin/isnet.pth saved_models/")

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
with gr.Blocks(css=custom_css) as background_remover_interface:
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
    # input_image
    brightness_slider.change(adjust_brightness, inputs=[input_image, brightness_slider], outputs=output_image)

    submit_button.click(inference, inputs=input_image, outputs=[output_image, mask_image])

    clear_button.click(lambda: (None, None, None), inputs=None, outputs=[input_image, output_image, mask_image])





#####


# --- Gradio Interfaces ---
# AI Chatbot Interface
with gr.Blocks(css=custom_css) as chatbot_interface:
    gr.Markdown("### AI Chatbot - Choose a personality and start chatting")
    personality = gr.Radio(choices=["Friendly", "Professional", "Humorous", "Empathetic"], value="Friendly", label="Personality")
    chatbot = gr.Chatbot(label="Chatbot", height=300)
    message = gr.Textbox(placeholder="Type your message here...")
    history = gr.State([])
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear")
    fun_fact_btn = gr.Button("Fun Fact")
    inspire_me_btn = gr.Button("Inspire Me")
    challenge_btn = gr.Button("Daily Challenge")
    send_btn.click(respond, inputs=[message, history, personality], outputs=[chatbot, message])
    clear_btn.click(clear_conversation, outputs=[chatbot, message])
    fun_fact_btn.click(generate_fun_fact, inputs=history, outputs=chatbot)
    challenge_btn.click(generate_daily_challenge, inputs=history, outputs=chatbot)
    inspire_me_btn.click(generate_inspiration, inputs=history, outputs=chatbot)




# Combine the two interfaces into a single app with tabs
app = gr.TabbedInterface([chatbot_interface, background_remover_interface], ["AI Chatbot", "Background Remover"])

# Launch the app
#app.launch(share=True)  
app.launch(server_name="0.0.0.0", server_port=8040, share=True, enable_queue=True)
