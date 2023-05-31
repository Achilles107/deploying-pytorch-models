# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py
import streamlit as st
from PIL import Image
import gdown
import style
import io
import torch
import os

st.title('PyTorch Mask Prediction')

# downloading the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

url = "https://drive.google.com/uc?export=download&id=1Vp9eIBB-8ScHQaYdpwDGTk2doOSBMQUq"
model = "masking.pth"
# Check if the model file already exists locally
if not os.path.isfile(model):
    # If the file doesn't exist, download it from the URL
    st.write("Downloading the model file...")
    gdown.download(url, model, quiet=False)
    st.write("Model file downloaded successfully!")


uploaded_file = st.file_uploader("Upload a close up image")

if uploaded_file is not None:
    # Process the uploaded file
    file_contents = uploaded_file.read()
    st.write("Uploaded file contents:")
    image = Image.open(io.BytesIO(file_contents))
    st.image(image, width=400)
    #st.image(file_contents, width=400)



clicked = st.button('Detect')

if clicked:
    model = style.load_model(model)
    output = style.stylize(model, image)
    if output == 'Mask':
        st.write('### Mask On')
    else:
        st.write('### Mask off')
    print(output)
    

