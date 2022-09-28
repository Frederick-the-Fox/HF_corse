# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, cache_dir='/data/wangyuanchun/fromHF/')

# sequence = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!",
# ]
# # print("raw_data:{}".format(sequence))

# id_list = []

# max_lenth = 0
# #tokenization

# #手动实现sequence->ids->padding过程
# # for each in sequence:

# #     tokens = tokenizer.tokenize(each)
# #     ids = tokenizer.convert_tokens_to_ids(tokens)
# #     id_list.append(ids)
# #     if len(ids) > max_lenth:
# #         max_lenth = len(ids)

# # print("max_lenth:{}".format(max_lenth))
# # print("id_list:{}".format(id_list))

# # i = 0
# # for each in id_list:
# #     # print(each)
# #     id_list[i] = each + [tokenizer.pad_token_id] * (max_lenth - len(each))
# #     i = i + 1

# # print('id_list_later:{}'.format(id_list))
# # input_ids = torch.tensor(id_list)
# # print("Input IDs:", input_ids)

# #利用自动的tokenizer
# input_ids = tokenizer(sequence, padding=True, truncation=True, return_tensors="pt")['input_ids']

# print("input_ids:{}".format(input_ids))
# # print(tokenizer.decode(input_ids))
# print(tokenizer.decode(input_ids))

# #inference
# output = model(input_ids)
# print("Logits:", output.logits)

# #post-processing
# pre = torch.nn.functional.softmax(output.logits, dim=-1)
# print("prediction:{}".format(pre))
# result = torch.argmax(pre, dim=-1)
# print("result:{}".format(result))

# print("padding_id:{}".format(tokenizer.pad_token_id))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#tokenize
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print('tokens:{}'.format(tokens))
#infer
output = model(**tokens)
print("output:{}".format(output))
#post_process
pre = torch.nn.functional.softmax(output.logits, dim = -1)
print('pre:{}'.format(pre))
result = torch.argmax(pre, dim = -1)
print ('result:{}'.format(result))