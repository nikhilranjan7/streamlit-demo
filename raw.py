import streamlit as st
import time
import numpy as np
from transformers import *
import torch

st.title('This is language generation example using OpenAI GPT2')

tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.

x = st.text_input('Type any sentence here:', value='')
input_context = x
input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
sample_bool = st.sidebar.checkbox('Do sample',value=True)
top_k = st.sidebar.slider('top_k parameter', value=50, min_value=10, max_value=500)
top_p = st.sidebar.slider('top_p parameter', value=1.0, min_value=0.1, max_value=1.0)
num_beams = st.sidebar.slider('beam length parameter', value=2, min_value=1, max_value=10)
output_length = st.sidebar.slider('output length parameter', value=50, min_value=20, max_value=1000, step=10)

outputs = model.generate(input_ids=input_ids, max_length=output_length, do_sample=sample_bool, num_beams=num_beams, top_k=top_k, top_p=top_p, num_return_sequences=5)

count = 1
for i in outputs:
    st.markdown('### Output number: {}'.format(count))
    st.write(tokenizer.decode(i))
    count += 1