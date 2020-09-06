import streamlit as st
import time
import numpy as np
from transformers import *
import torch


st.title('OpenAI GPT2 Language generation')

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
    model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
    return model, tokenizer

model, tokenizer = load_model()

x = st.text_input('Type any sentence here:', value='')

sample_bool = st.sidebar.checkbox('Sampling == True',value=True)
top_k = st.sidebar.slider('top_k parameter', value=50, min_value=10, max_value=500)
top_p = st.sidebar.slider('top_p parameter', value=1.0, min_value=0.1, max_value=1.0)
num_beams = st.sidebar.slider('beam length parameter', value=2, min_value=1, max_value=10)
output_length = st.sidebar.slider('output length parameter', value=50, min_value=20, max_value=1000, step=10)

if len(x) > 0:
    input_context = x
    input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
    outputs = model.generate(input_ids=input_ids, max_length=output_length, do_sample=sample_bool, num_beams=num_beams, top_k=top_k, top_p=top_p, num_return_sequences=5)
    ans_tuple = []

    for i in outputs:
        loss=model(i, labels=i)[0].item()
        text = tokenizer.decode(i,skip_special_tokens=True)
        ans_tuple.append([np.exp(loss / len(i)), text])

    ans_tuple.sort()
    for i in range(len(ans_tuple)):
        st.markdown('## Output {}:'.format(i+1))
        st.markdown('#### Perplexity: {}'.format(round(ans_tuple[i][0],4)))
        st.text('\n{}'.format(ans_tuple[i][1]))

