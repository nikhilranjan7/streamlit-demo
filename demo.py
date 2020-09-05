import streamlit as st
import time
import numpy as np
from transformers import *
import torch

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

st.title('This is BERT tokenization example')

x = st.text_input('Type any sentence here:')
output = tokenizer.tokenize(x)

output
