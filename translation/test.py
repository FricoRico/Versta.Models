# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Neurora/firefox-mt-en-ko")
model = AutoModelForSeq2SeqLM.from_pretrained("Neurora/firefox-mt-en-ko")

translate = "High-performance deep learning models for text-to-speech tasks."

inputs = tokenizer(translate, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.tokenize(translate))
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))