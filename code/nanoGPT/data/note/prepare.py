import os
import requests
import tiktoken
import numpy as np
print('welcome!')
input_file_path_dis = os.path.join(os.path.dirname(__file__), 'discharge.csv/discharge.csv')

with open(input_file_path_dis, 'r') as f:
    data = f.read()
n = len(data)
print(f'size of data: {n}')
train_data = data[:int(n*0.9995)]
val_data = data[int(n*0.9995):]
print(len(train_data), len(val_data))

input_file_path_rad = os.path.join(os.path.dirname(__file__), 'radiology.csv/radiology.csv')

with open(input_file_path_rad, 'r') as f:
    data = f.read()
n = len(data)
print(f'size of data: {n}')
train_data_rad = data[:int(n*0.9995)]
val_data_rad = data[int(n*0.9995):]
print(len(train_data_rad), len(val_data_rad))

train_data += train_data_rad
val_data += val_data_rad

print(len(train_data), len(val_data))

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
