#!/usr/bin/env python
# coding: utf-8

import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_data(filepath):

    with open(filepath, "r") as f:
      lines = f.readlines()

    inputs = []
    for line in lines:
      inputs.append(json.loads(line))

    input_str = []
    labels = []
    for inp in inputs:
      cur_inp = f"wic pos:  sentence1: {inp['sentence1']} sentence2: {inp['sentence2']} word: {inp['word']}"
      input_str.append(cur_inp)
      labels.append(str(inp['label']))

    return input_str, labels

train_input_str, train_labels = get_data("./train.jsonl")
val_input_str, val_labels = get_data("./val.jsonl")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to('cpu')

results = []

for inp in val_input_str:
  input_ids = tokenizer(inp, return_tensors="pt").input_ids
  outputs = model.generate(input_ids)
  output_str= tokenizer.decode(outputs[0], skip_special_tokens=True)
  results.append(output_str == 'True')

correct_count = 0
for i in range(len(val_labels)):
    if val_labels[i] == str(results[i]):
        correct_count += 1

print(correct_count, correct_count/len(val_labels))

