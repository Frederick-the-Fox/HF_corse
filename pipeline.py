from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")# 选定任务， 指定模型

result = generator(
    "Renmin University of China is famous around China",
    max_length=30,
    num_return_sequences=5,
) #不同的pipeline有不同的使用方法和参数，这里输出的result是一个list, 每个list是一个dictionary，每个dictionary只有一个元素'generated_text'

print(result)
# for each in result:
#     print(each['generated_text']) 