from transformers import AutoTokenizer
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

#手动实现sequence->ids->padding过程
sequence = raw_inputs
id_list = []
max_lenth = 0
for each in sequence:
    each = tokenizer.decode([101]) + each + tokenizer.decode([102])
    tokens = tokenizer.tokenize(each) #sequence->tokens 得到一个list里面的元素是不同的字符串元素
    #conver函数将tokens转化成索引id，ids是一个lsit里面元素为int
    ids = tokenizer.convert_tokens_to_ids(tokens)#添加[CLS]和[SEP]
    id_list.append(ids)#得到的id_list是一个元素为list的list
    if len(ids) > max_lenth:
        max_lenth = len(ids)

print("max_lenth:{}".format(max_lenth))
print("id_list:{}".format(id_list))

# 一下为padding的过程，将所有序列的长度补全到长度最长的序列的长素
i = 0
for each in id_list:
    # print(each)
    id_list[i] = each + [tokenizer.pad_token_id] * (max_lenth - len(each))
    i = i + 1

print('id_list_paddinged:{}'.format(id_list))
input_ids = torch.tensor(id_list)
print("Input IDs:", input_ids)

# print(tokenizer.decode([102]))
print('手动与自动实现的结果相同：{}'.format(input_ids.equal(inputs['input_ids'])))
