"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import time
import tiktoken
from model import GPTConfig, GPT
from vanilla_transformer import TransformerLM, TransformerConfig
from retnet import RetNet, retnet_1_3b, RetNetConfig

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "Premise: No history of blood clots or DVTs, has never had chest pain prior to one week ago. Hypothesis:  Patient has angina. Target: the label for this hypothesis with respect to the premise is:" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 1 # number of tokens generated in each sample
temperature = 1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 10
model_name = 'ckpt_medical_finetune.pt'
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
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, model_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isRetnet:
        conf = RetNetConfig(**checkpoint['model_args'])
        conf.n_embd = int(conf.n_embd)
        print(conf)
        model = RetNet(conf, 
        num_tokens=50304,
        d_model=conf.n_embd,
        nhead=conf.n_head,
        num_layers=conf.n_layer,
        dim_feedforward=conf.n_embd * 2)
        device=device
    elif isTransformer:
        conf = TransformerConfig(**checkpoint['model_args'])
        print(conf)
        model = TransformerLM(conf, 
        num_tokens=50304,
        d_model=conf.n_embd,
        nhead=conf.n_head,
        num_layers=conf.n_layer,
        dim_feedforward=conf.n_embd * 2)
        device=device
    else:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
time_per_sample = []
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            start = time.time()
            y = model.generate_parallel(x, max_new_tokens, temperature=temperature, top_k=top_k)
            #print(y)
            print(decode(y[0].tolist()))
            end = time.time()
            print('---------------')
            duration = end - start
            print(f"Elapsed time: {duration}")
            time_per_sample.append(duration)
print(time_per_sample)
print(sum(time_per_sample) / len(time_per_sample))
