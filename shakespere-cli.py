import argparse
import os
import random
import pickle
import torch
import time
import sys
from pathlib import Path
from model import encode, GPT
from config import GPTConfig
import warnings
from warnings import filterwarnings


def load_pickles():
    with open(Path(__file__).parent / 'components/itos.bin', 'rb') as f:
        itos = pickle.load(f)  # index to string lookup

    with open(Path(__file__).parent / 'components/stoi.bin', 'rb') as f:
        stoi = pickle.load(f)  # string to index lookup

    return itos, stoi

def load_model(itos):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GPTConfig(device=device)
    gpt = GPT(config, itos=itos).to(config.device)

    # Load weights (if exists)
    if os.path.isfile(Path(__file__).parent / "model_weights/gpt.pth"):
        try:
            gpt.load_state_dict(torch.load(Path(__file__).parent / "model_weights/gpt.pth") if device == 'cuda' else torch.load(
                Path(__file__).parent / "model_weights/gpt.pth", map_location=torch.device('cpu')))
        except Exception as e:
            print("Loading weights failed!")
            print(e)

    return gpt, config

    
def typing_effect(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Generate Shakespearean text.')
    parser.add_argument('input_string', type=str, help='Input text for generation')
    parser.add_argument('--tokens', type=int, default=500, help='Number of tokens to predict')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for text generation')
    args = parser.parse_args()

    itos, stoi = load_pickles()
    gpt, config = load_model(itos)

    try:
        x = encode(args.input_string, stoi=stoi)
    except KeyError as e:
        print('Try different words....')
        return

    x = torch.Tensor(x).reshape(1, -1)
    x = x.type(torch.LongTensor)

    out = gpt.write(x.to(config.device), max_new_tokens=args.tokens, temperature=args.temperature)

    typing_effect(out[0])

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
