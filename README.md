# 🧠 Project 03 — Fine-Tuning T5 for Text Summarization (Encoder–Decoder Model)

**Course Project:** Fine-Tuning Using Different Architectures  
**Task 2:** Encoder–Decoder Model (T5) — Text Summarization  
**Frameworks:** Hugging Face Transformers • PyTorch • Streamlit  
**Dataset:** [CNN/DailyMail News Summarization](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

---

## 📋 Project Overview

This project focuses on fine-tuning a **T5 (Text-to-Text Transfer Transformer)** model for **abstractive summarization** of long news articles. The model is trained to generate concise and coherent summaries from full-length articles using transfer learning.

Key highlights:
- Data preprocessing and tokenization
- Fine-tuning a pre-trained **T5-small** or **T5-base** model
- Evaluation with **ROUGE metrics**
- Sample qualitative summaries
- Streamlit-based interactive demo

---

## 📁 Folder Structure

```
T5-SUMMARIZER/
├── model_trainer/
│   └── summarization-using-t5-huggingface.ipynb  # Fine-tuning notebook
├── t5_final_summarizer_/  # Saved fine-tuned model
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
│   ├── added_tokens.json
│   └── training_args.bin
├── app.py  # Streamlit app for interactive summarization
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/T5-SUMMARIZER.git
cd T5-SUMMARIZER
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Dataset
Download the CNN/DailyMail News Summarization dataset from Kaggle:  
[https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

Use it directly in the notebook: `model_trainer/summarization-using-t5-huggingface.ipynb`.

---

## 🧩 Fine-Tuning the Model

All training steps are in the Jupyter notebook:
- Loading and cleaning the CNN/DailyMail dataset
- Tokenization with the T5 tokenizer
- Fine-tuning using Hugging Face `Trainer`
- Evaluation using ROUGE metrics
- Saving the fine-tuned model and tokenizer to `t5_final_summarizer_/`

**Running training:**
1. Open the notebook in VS Code or Jupyter.
2. Adjust file paths if necessary.
3. Run all cells sequentially.
4. Checkpoints and final model will be saved automatically.

---

## 📊 Evaluation Metrics (ROUGE)

| Metric   | Description                  | Example Score |
|----------|-----------------------------|---------------|
| ROUGE-1  | Unigram overlap             | 0.42          |
| ROUGE-2  | Bigram overlap              | 0.19          |
| ROUGE-L  | Longest common subsequence  | 0.38          |

> Scores may vary depending on model size, epochs, and hyperparameters.

---

## 🌐 Streamlit Demo

Run the local demo:
```bash
streamlit run app.py
```
Open (https://t5-summarizer.streamlit.app/) in your browser.


---

## 📌 Notes

- Ensure `model.safetensors` is tracked via **Git LFS** before pushing to GitHub.
- Use GPU acceleration if available for faster fine-tuning.
- Adjust hyperparameters in the notebook for optimal performance.
