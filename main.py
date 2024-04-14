from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from model.create_model import create_the_model

app = FastAPI()

model = create_the_model()[0]
model.load_state_dict(torch.load("mnist_model.pt"))
model.eval()

@app.get("/")
async def read_root():
    return {"message": "to make prediction access /docs and try out."}

def transform_image(image_bytes):
    # Resize and normalize the image
    img = Image.open(BytesIO(image_bytes))
    img = img.resize((28, 28), Image.NEAREST)  # or Image.BILINEAR
    img = img.convert('L')  # convert to grayscale
    img = np.array(img) / 255.0  # normalize
    img = img.reshape(1, -1)  # flatten
    img = torch.from_numpy(img).float()  # convert to tensor
    return img


@app.post("/predict/")
async def upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    transformed_image = transform_image(image_bytes)

    # Make a prediction
    with torch.no_grad():
        output = model(transformed_image)

    # Get the predicted class
    _, predicted = torch.max(output, 1)
    return {"prediction": predicted.item()}
