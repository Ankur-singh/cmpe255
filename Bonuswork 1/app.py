import json
import logging
import gradio as gr
from pathlib import Path
from utils import ModelsEngine


# load labels
LABELS = Path("class_names.txt").read_text().splitlines()

# load model
path = Path('model_weights')
# formats = ['pytorch', 'torch_script', 'onnx'
formats = ['openvino_onnx', 'openvino_ir']
engine = ModelsEngine(path, formats)
logging.info('Done loading models . . . .')


def get_prediction(format, device, image):
    model = engine.get_model(format, device)
    image = image / 255.0
    output = model.predict(image)
    logging.info(f'output: {output.shape}')
    confidence = {LABELS[i]: float(output[i]) for i in range(len(LABELS))}
    return confidence


# gradio interface for image classification
title = "Pictionary"
description = "A simple image classification app using PyTorch, ONNX, and OpenVINO. \
                Select `format`, `device` and draw a picture. Let the model guess what it is."

demo = gr.Interface(fn=get_prediction, 
             inputs=[
                gr.Dropdown(formats, value=formats[0], label='Format'),
                gr.Dropdown(['cpu', 'gpu'], value='cpu', label='Device'),
                gr.Sketchpad(type="numpy", label='Image')
            ],
             outputs=gr.Label(num_top_classes=5),
             title=title, description=description,
            )

demo.launch()