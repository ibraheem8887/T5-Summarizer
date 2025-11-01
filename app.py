import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# -------------------------
# Load Model and Tokenizer
# -------------------------
MODEL_PATH = "t5_final_summarizer_"  # Your model folder

@st.cache_resource(show_spinner=True)
def load_model(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model(MODEL_PATH)

# -------------------------
# Summarization Function
# -------------------------
def summarize_text(text, max_input_length=512, max_output_length=150):
    input_text = "summarize: " + text.strip()
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    )
    summary_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -------------------------
# Streamlit App Interface
# -------------------------
st.set_page_config(page_title="T5 News Summarizer", layout="wide")
st.title("📰 T5 News Article Summarizer")
st.write("Enter a long news article and get a concise summary.")

# Text input
user_input = st.text_area("Paste your news article here:", height=300)

# Optional: slider for summary length
max_len = st.slider("Max summary length:", 50, 300, 150)

# Summarize button
if st.button("Generate Summary"):
    if user_input.strip() != "":
        with st.spinner("Generating summary..."):
            summary = summarize_text(user_input, max_output_length=max_len)
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
