from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
result = generator(
    "Renmin University of China is famous around China",
    max_length=30,
    num_return_sequences=5,
)

for each in result:
    print(each['generated_text'])