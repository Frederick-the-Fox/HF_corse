from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc", cache_dir='/data/wangyuanchun/fromHF')
print(raw_datasets)

raw_train_dataset = raw_datasets['train']
print(raw_train_dataset[0])

checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='/data/wangyuanchun/fromHF')

# #自动方法直接tokenize一条数据
# inputs = tokenizer([raw_train_dataset[0]['sentence1'], raw_train_dataset[1]['sentence1']], [raw_train_dataset[0]['sentence2'], raw_train_dataset[1]['sentence2']])
# print('inputs:{}'.format(inputs))

# #自动方法方法直接tokenize一个batch出来
# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )
# print(tokenized_dataset.features)

#利用map函数加速的方法
def tokenize_function(example):
    #这里可以对example的内容继续添加操作，灵活度比上一个高，而且不会占用太大内存。
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(raw_datasets)
print(tokenized_datasets)

origin_sample = tokenized_datasets["train"]
# print('origin_sample:{}'.format(origin_sample))
samples = tokenized_datasets["train"][:8] #可以直接这样按照顺序选取sample的条数，此处选取了前8条
# print("[:8] : {}".format(samples))
# print('num of samples in [:8] :{}'.format(len(samples['sentence1'])))
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}#选取那些k不在列表中的kv对组成新的字典（{k: v}这种写法组成字典）

# print([len(x) for x in samples["input_ids"]])#可以学习一下这种写法
# print('after selection:{}'.format(samples.keys()))
# print(raw_datasets['train'][0]['sentence1'])
# print(tokenized_datasets['train'][0]['sentence1'])

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch = data_collator(samples)#将选中数据封装成一个batch，这个过程中加入padding
print({k: v.shape for k, v in batch.items()})