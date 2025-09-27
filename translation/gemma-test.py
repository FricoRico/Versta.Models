from os import environ

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

login(token=environ.get("HUGGINGFACE_TOKEN"))

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
messages = [
    {
        "role": "system",
        "content": "You are a helpful translator. Only respond with the translation of the input text."
    },
    {
        "role": "user",
        "content": "Translate the following text from English to Dutch: 'In southwest Arkansas, the state government runs what might be the world's most unusual diamond mine. For the price of a movie ticket, anyone can dig for diamonds at Crater of Diamonds State Park—and keep whatever they find.'"
    },
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
