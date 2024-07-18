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
out_dir = 'results' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 2 # number of samples to draw
max_new_tokens = 40000 # number of tokens generated in each sample
temperature = 1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 50 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 10
block_sizes = [256, 1024, 2048, 4096]
architectures = ['tf', 'retnet']
#model_name = 'ckpt_tf_4096_p.pt'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
isRetnet = False
isTransformer = False
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
                        
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 1  # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
n_layer = 25
n_head = 25
n_embd = 1000
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging





torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



results = {}
for block_size in block_sizes:
    results[block_size] = {}
    for architecture in architectures:
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
        if architecture == 'retnet':
            isRetnet = True
            isTransformer = False
            retnetConfig = RetNetConfig(**model_args)
            model = RetNet(retnetConfig, 
            num_tokens=50304,
            d_model=n_embd,
            nhead=n_head,
            num_layers=n_layer,
            dim_feedforward=n_embd * 2,
            device=device,
            dtype=torch.bfloat16,
        )
        else:
            isRetnet = False
            isTransformer = True
            tfConfig = TransformerConfig(**model_args)
            model = TransformerLM(tfConfig,
            num_tokens=50304,
            d_model=n_embd,
            nhead=n_head,
            num_layers=n_layer,
            dim_feedforward=n_embd * 2,
            max_batch_size = batch_size,
            max_seq_length = block_size,
            device=device,
            dtype=torch.bfloat16,
            )
        model.eval()
        model.to(device)
        if compile:
            model = torch.compile(model) # requires PyTorch 2.0 (optional)

        # ok let's assume gpt-2 encodings by default
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

        # encode the beginning of the prompt
        start = "\n" 
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
                    y = model.generate(x, max_new_tokens, temperature=temperature)
                    #print(y)
                    end = time.time()
                    duration = end - start
                    print(f"Elapsed time: {duration}")
                    time_per_sample.append(duration)
        print(f"Results for {architecture} with block size {block_size}")
        print(time_per_sample)
        print(sum(time_per_sample) / len(time_per_sample))
        results[block_size][architecture] = sum(time_per_sample) / len(time_per_sample)

print("*****************FINAL RESULTS************************")
print(results)
