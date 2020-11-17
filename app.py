import argparse
import os
from tqdm import tqdm

import torch
import pytorch_lightning as pl
import streamlit as st

from vocab import Vocab
from utils import load_data, load_sent, load_model, makedir, write
from dataset import get_eval_dataloader


st.sidebar.write("## Parameters")

device = st.sidebar.selectbox("Device",
    ["cpu"] + ["cuda:{}".format(i) for i in range(torch.cuda.device_count())],
    0,
    lambda key: key if key == "cpu" else "GPU {}".format(key)
)

st.write('# Blank Language Models: Demo')

st.write('## Load  model')

yelp_neg = "checkpoints/yelp/neg/lightning_logs/version_0/checkpoints/model.ckpt"
yelp_pos = "checkpoints/yelp/pos/lightning_logs/version_0/checkpoints/model.ckpt"

if not os.path.exists(yelp_neg) or not os.path.exists(yelp_pos):
    st.write(":warning: Default models not found. Run `get_model.sh` to download models trained on Yelp.")
    checkpoint = st.radio("Load checkpoint", ("Custom model", ))

else:
    checkpoint = st.radio("Load checkpoint", ("Yelp positive reviews", "Yelp negative reviews", "Custom model"))

if checkpoint == "Custom model":
    checkpoint_file = st.text_input("Path to `model.ckpt` file", value=yelp_pos)
else:
    checkpoint_file = yelp_pos if "Yelp positive" in checkpoint else yelp_neg

@st.cache
def get_model(checkpoint_file, device):
    model = load_model(checkpoint_file).to(device)
    model.eval()
    vocab = Vocab(os.path.join(model.hparams.root_dir, 'vocab.txt'))
    return model, vocab


model, vocab = get_model(checkpoint_file, device)

decode = st.sidebar.radio("Decoding", ("Greedy", "Sample")).lower()

mode = st.sidebar.radio("Task", ('Infilling', 'Sample'))


if mode == "Sample":
    _, full = model.generate([model.init_canvas()], decode, device)
    full = [[vocab.idx2word[id] for id in ids] for ids in full]
    for step in full:
        st.write(" ".join(step).replace("<blank>", "\_\_\_"))

if mode == "Infilling":
    st.write('## Load infilling data')
    text_input = st.text_input("Blanked input", value="___ place ___ and ___ food ___ .").lower()
    s = text_input.replace("___", "<blank>").split()
    s += ['<eos>'] if model.hparams.add_eos else []
    s = [vocab.word_to_idx(w) for w in s]
    _, full = model.generate(s, decode, device)
    full = [[vocab.idx2word[id] for id in ids] for ids in full]
    for step in full:
        st.write(" ".join(step).replace("<blank>", "\_\_\_"))

if st.button("Rerun"):
  pass
