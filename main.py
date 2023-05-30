# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py
import streamlit as st
from PIL import Image

import style

st.title('PyTorch Style Transfer')

img = st.sidebar.selectbox(
    'Select Image',
    ('girl_mask.png', 'selena.png')
)

# style_name = st.sidebar.selectbox(
#     'Select Style',
#     ('candy', 'mosaic', 'rain_princess', 'udnie')
# )

model = "mask_pred_weights.pth"
#model= "saved_models/" + style_name + ".pth"
input_image = "images/content-images/" + img
#output_image = "images/output-images/" + style_name + "-" + img

st.write('### Source image:')
image = Image.open(input_image)
st.image(image, width=400) # image: numpy array

clicked = st.button('Stylize')

if clicked:
    model = style.load_model(model)
    output = style.stylize(model, input_image)
    print(output)
    

