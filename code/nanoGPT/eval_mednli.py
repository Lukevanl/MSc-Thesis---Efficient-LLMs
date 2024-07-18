"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import time
import json
import tiktoken
from model import GPTConfig, GPT
from vanilla_transformer import TransformerLM, TransformerConfig
from retnet import RetNet, retnet_1_3b, RetNetConfig

# -----------------------------------------------------------------------------
out_dir = 'out'
num_samples = 1 # number of samples to draw
max_new_tokens = 1 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 10
model_name = 'ckpt.pt'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
isRetnet = True
isTransformer = False
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
ckpt_path = os.path.join(out_dir, model_name)
checkpoint = torch.load(ckpt_path, map_location=device)
if isRetnet:
    conf = RetNetConfig(**checkpoint['model_args'])
    conf.n_embd = int(conf.n_embd)
    conf.dropout = 0.0
    print(conf)
    model = RetNet(conf, 
    num_tokens=50304,
    d_model=conf.n_embd,
    nhead=conf.n_head,
    num_layers=conf.n_layer,
    dim_feedforward=conf.n_embd * 2,
    device=device)
elif isTransformer:
    conf = TransformerConfig(**checkpoint['model_args'])
    print(conf)
    model = TransformerLM(conf, 
    num_tokens=50304,
    d_model=conf.n_embd,
    nhead=conf.n_head,
    num_layers=conf.n_layer,
    dim_feedforward=conf.n_embd * 2,
    device=device)
else:
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


dataset = 'mednli'
data_dir = os.path.join('data', dataset)
with open(os.path.join(data_dir, 'test_gpt.json')) as f_train:
        data = json.loads(f_train.read())

correct = 0
incorrect = 0
for i in range(len(data['context'])):
    data_prompt = data['context'][i] + data['target'][i][:-1]
    x = (torch.tensor(data_prompt, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate_parallel(x, max_new_tokens, temperature=temperature, top_k=top_k)
            pred = decode([y[0][-1]])
            label = decode([data['target'][i][-1]])
            if pred == label:
                correct += 1
            else: 
                incorrect += 1
print(correct, incorrect)
accuracy = correct / (correct + incorrect)
print(accuracy)

