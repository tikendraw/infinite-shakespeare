import streamlit as st
from model import encode, GPT
from config import GPTConfig
import os, random, pickle
import torch
import time, sys
import base64
import webbrowser
from pathlib import Path

about_text = """
Welcome to the world of Shakespearean charm and textual artistry with our transformative text generative model! 
Crafted with the essence of William Shakespeare's literary genius, our transformer-based system delves deep into 
the vast treasure trove of the bard's previous works to bring forth a new era of language generation.

 we have embraced the brilliance of Shakespeare and harnessed the power of artificial intelligence to create a
transformative text generative model. Inspired by the elegance, wit, and linguistic prowess of the great bard, 
our model seamlessly blends Shakespearean aesthetics with the ingenuity of modern technology.

Using transformer architecture, our model has been trained on a entire life work of Shakespearean works. 
It has absorbed the intricacies of his language, the beauty of his prose, and the essence of his theatrical expressions. 
By learning from Shakespeare's writings, the model has developed an intricate understanding of his style, themes, and linguistic patterns.

Experience the magic of Shakespearean language revitalized for the digital age. Let our transformative text generative model unleash your 
creativity, ignite your passion for literature, and immerse you in the resplendent beauty of Shakespeare's world. Welcome to a new era of linguistic enchantment."""

disclaimer = """
It is important to note that while the model is trained on Shakespeare's works, it is still an artificial intelligence model and 
may not perfectly replicate his style or creativity. The model can generate text that resembles Shakespeare's writing to a certain extent, 
but it may produce occasional inconsistencies or deviations from his original work."""

github_url = "https://github.com/tikendraw/infinite-shakespeare"
github_star = '<iframe src="https://ghbtns.com/github-btn.html?user=twbs&repo=bootstrap&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>'
star = """<!-- Place this tag in your head or just before your close body tag. -->
<script async defer src="https://buttons.github.io/buttons.js"></script>

<!-- Place this tag where you want the button to render. -->
<a class="github-button"
   href="https://github.com/tikendraw/infinite-shakespeare"
   data-icon="octicon-star"
   data-size="large"
   data-show-count="true"
   aria-label="Star ntkme/github-buttons on GitHub">Star</a>"""


def load_pickles():
    with open("./components/itos.bin", "rb") as f:
        itos = pickle.load(f)  # index to string lookup

    with open("./components/stoi.bin", "rb") as f:
        stoi = pickle.load(f)  # string to index lookup

    return itos, stoi


@st.cache_resource
def load_model(itos):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(device=device)
    gpt = GPT(config, itos=itos).to(config.device)

    # Load weights (if exists)
    if os.path.isfile("./model_weights/gpt.pth"):
        try:
            gpt.load_state_dict( torch.load("./model_weights/gpt.pth") if device == "cuda" else torch.load("./model_weights/gpt.pth", map_location=torch.device("cpu")))
            print("Weights loaded!")
        except Exception as e:
            st.error("Loading weights failed!")
            st.text(e)
            print(e)

    return gpt, config


def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def typing_effect(x):
    # container = st.empty()
    with st.empty():
        rest = ""
        for char in x:
            rest += char
            st.code(rest, language=None, line_numbers=False)
            # container.code(rest, language=None, line_numbers=False)
            time.sleep(0.01)


def main():

    col1, col2, col12 = st.columns([4, 1.3, 1.5])

    with col1:
        input_string = st.text_input(
            " ", placeholder="Give Shakespeare something to start with..."
        ).strip()
    with col2:
        max_len = int(st.number_input("Predict tokens", min_value=100, value=500))

    with col12:
        temperature = float(
            st.number_input(
                "Temperature", min_value=0.5, max_value=10.0, value=1.0, step=0.1
            )
        )

    col3, col4, col5 = st.columns([1, 1, 1])

    with col3:
        generate = st.button("Generate")

    with col4:
        code = st.button("Code/Github")
        
        if code :
            webbrowser(github_url)

    with col5:
        st.components.v1.html(star, width=None, height=None, scrolling=False)

    with st.sidebar:
        st.header("About")
        st.write(about_text)

    itos, stoi = load_pickles()
    gpt, config = load_model(itos)

    if generate and input_string:
        try:
            x = encode(input_string, stoi=stoi)
        except KeyError as e:
            st.error("Try different words....")

        x = torch.Tensor(x).reshape(1, -1)
        x = x.type(torch.LongTensor)

        out = gpt.write(
            x.to(config.device), max_new_tokens=max_len, temperature=temperature
        )


        typing_effect(out[0])

        st.header("Limitations")
        st.error(disclaimer)


set_background("./components/spfitdark.jpg")

if __name__ == "__main__":
    st.title('ShakespeareGPT')
    main()
