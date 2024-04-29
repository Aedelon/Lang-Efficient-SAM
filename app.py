import os
import warnings

import gradio as gr
import numpy as np
from PIL import Image

from lang_efficient_sam.lang_efficient_sam import LangEfficientSAM
from lang_efficient_sam.utils.draw_image import draw_image

warnings.filterwarnings("ignore")

model = LangEfficientSAM()


def predict(box_threshold, text_threshold, image_path, text_prompt):
    print("Predicting... ", box_threshold, text_threshold, image_path, text_prompt)

    image_pil = Image.open(image_path).convert("RGB")

    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]

    image_array = np.asarray(image_pil)
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")

    return image


title = "LangEfficientSAM"

inputs = [
    gr.Slider(0, 1, value=0.3, label="Box threshold"),
    gr.Slider(0, 1, value=0.25, label="Text threshold"),
    gr.Image(type="filepath", label='Image'),
    gr.Textbox(lines=1, label="Text Prompt"),
]

outputs = [gr.Image(type="pil", label="Output Image")]

examples = [
    [
        0.20,
        0.20,
        os.path.join(os.path.dirname(__file__), "images", "living.jpg"),
        "fabric",
    ],
    [
        0.36,
        0.25,
        os.path.join(os.path.dirname(__file__), "images", "fruits.jpg"),
        "apple",
    ],
    [
        0.20,
        0.20,
        os.path.join(os.path.dirname(__file__), "images", "street.jpg"),
        "car",
    ]
]

demo = gr.Interface(fn=predict,
                    inputs=inputs,
                    outputs=outputs,
                    examples=examples,
                    title=title)

demo.launch(debug=False, share=False)
