import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
from PIL import Image

model_name = 'Kontawat/vit-diabetic-retinopathy-classification'
processor = ViTImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, output_attentions=True)
model.config.output_attentions = True

img = Image.new('RGB', (224, 224))
inputs = processor(images=img, return_tensors='pt')
outputs = model(**inputs, output_attentions=True)

print("Type of outputs.attentions:", type(outputs.attentions))
if outputs.attentions is not None:
    print("Length:", len(outputs.attentions))
    if len(outputs.attentions) > 0:
        print("Shape of last attention:", outputs.attentions[-1].shape)
