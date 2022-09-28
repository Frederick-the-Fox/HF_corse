from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc", cache_dir='/data/wangyuanchun/fromHF')

print(raw_datasets)

raw_train_dataset = raw_datasets['train']
print(raw_train_dataset[0])

from transformers import AutoTokenizer

checkpoint = 'bert-base-cased'

tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='/data/wangyuanchun/fromHF')

inputs = tokenizer(raw_train_dataset[0]['sentence1'], raw_train_dataset[0]['sentence2'])
print('inputs:{}'.format(inputs))

print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
