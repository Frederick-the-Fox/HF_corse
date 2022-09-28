from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I hate this so much!",
    "I've been waiting for a HuggingFace course my whole life.",
    "Bullshit",
    "LadyGaga",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# from transformers import AutoModel

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModel.from_pretrained(checkpoint)

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# print('origin output:{}; logits: {}'.format(outputs, outputs.logits))

# print(outputs.last_hidden_state.shape)

import torch

pre = torch.nn.functional.softmax(outputs.logits)
print('prediction:{}'.format(pre))

result = torch.argmax(pre, dim = -1)
print('result:{}'.format(result))
