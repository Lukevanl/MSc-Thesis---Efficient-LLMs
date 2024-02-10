# This adapts GPTQ's quantization process: https://github.com/IST-DASLab/gptq/
# E. Frantar et al GPTQ: Accurate Post-training Compression for GPT, arXiv:2210.17323
# portions copyright by the authors licensed under the Apache License 2.0
import gc
import sys
import time
from pathlib import Path
from typing import Optional
import os
import numpy as np
import pickle
from contextlib import nullcontext
import time
import tiktoken
from model import GPTConfig, GPT
from vanilla_transformer import TransformerLM, TransformerConfig
from retnet import RetNet, retnet_1_3b, RetNetConfig

import torch
from datasets import load_dataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import GPTConfig, GPT
from quantization import GPTQQuantizer
#from lit_llama.utils import EmptyInitOnDevice, llama_model_lookup


model_name = 'ckpt_r_2048.pt'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
isRetnet = False
isTransformer = False
out_dir = 'out'
NR_SAMPLES = 128


def get_sample_data():
        # traindata = load_dataset(
    #     "allenai/c4",
    #     "allenai--c4",
    #     data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
    #     split="train",
    # )
    traindata = load_dataset(
        "wikitext",
        "wikitext-2-v1",
        #data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    # heuristic for the data size?
    txt = "\n".join(
        traindata[i]["text"] for i in torch.randperm(len(traindata))[:5000].tolist()
    )
    return txt


@torch.no_grad()
def llama_blockwise_quantization(
    model, sample_inputs, working_device, *, bits=4, groupsize=-1
):
    """
    This is the classic post-training quantization of all linear layers.
    We quantize in order, i.e. when observing the inputs, we use the outputs of the previously quantized layers rather
    than doing them all at once.
    """
    print(model)
    print(model.config)

    print("Getting inputs for first block")
    model.transformer.wte.to(working_device)
    sample_inputs = sample_inputs.to(working_device)
    inps = model.transformer.wte(sample_inputs)
    model.transformer.wte.to("cpu")
    torch.cuda.empty_cache()

    # rope_cache = model.build_rope_cache(sample_inputs)
    # mask_cache = model.build_mask_cache(sample_inputs)
    rope_cache = None
    mask_cache = None

    print("Starting to quantize blocks")
    outs = torch.zeros_like(inps)

    # better than relying on enumeration? originally the code bundled
    # the two mlp fc layers
    # we could automate this with a lot of hooks and another iteration
    submodules_to_process = [
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    ]

    for i, block in enumerate(model.transformer.h):
        block.to(working_device)

        for name in submodules_to_process:
            print(i, name, end=" ")
            t0 = time.perf_counter()
            print("collecting stats", end=" ")
            sys.stdout.flush()
            module = block.get_submodule(name)

            gptq = GPTQQuantizer(
                module,
                bits=bits,
                groupsize=groupsize,
                actorder=(groupsize == -1),
            )
            handle = module.register_forward_hook(gptq.collect_input_stats)
            for j in range(inps.size(0)):
                outs[j : j + 1] = block(
                    inps[j : j + 1]
                )

            handle.remove()

            print("quantizing", end=" ")
            sys.stdout.flush()
            q_module, error = gptq.quantize()

            # replace the linear module with the quantized module
            pname, dname = name.rsplit(".", 1)
            setattr(block.get_submodule(pname), dname, q_module)

            # cleanup in an attempt to not run out of memory
            del gptq
            gc.collect()
            torch.cuda.empty_cache()
            t1 = time.perf_counter()
            print(f"time {int(t1 - t0 + 0.5)}s quantization error {error:.1f}")

        for j in range(inps.size(0)):
            outs[j : j + 1] = block(
                inps[j : j + 1],
            )

        block.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # the outputs are the next block's inputs and we'll reuse the old inputs
        inps, outs = outs, inps

    model.transformer.ln_f.to(working_device)
    for j in range(inps.size(0)):
        outs[j : j + 1] = model.transformer.ln_f(inps[j : j + 1])
    model.transformer.ln_f.to("cpu")
    inps, outs = outs, inps

    model.lm_head.to(working_device)
    gptq = GPTQQuantizer(
        model.lm_head,
        bits=bits,
        groupsize=groupsize,
        actorder=(groupsize == -1),
    )
    handle = model.lm_head.register_forward_hook(gptq.collect_input_stats)
    for j in range(inps.size(0)):
        model.lm_head(inps[j : j + 1])
    handle.remove()
    q_module, error = gptq.quantize()
    model.lm_head = q_module
    model.lm_head.to("cpu")

def quantizing(
    *,
    output_path: Optional[Path] = None,
    n_samples: int = NR_SAMPLES,
    dtype: str = "float32",
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        checkpoint_path: The checkpoint path to load.
        output_path: Path to write the quantized model's state dict to.
        tokenizer_path: The tokenizer path to load.
        n_samples: Number of example inputs to use for statistics (default: 128)
        dtype: The dtype to use to load the model.
        quantize: Mode to quantize the model to:
            ``"gptq.int4"``: GPTQ 4-bit mode.
            Note that ``"llm.int8"```does not need a quantization step.
    """
    device = "cuda"

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    if quantize == "gptq.int4":
        bits = 4
    elif quantize == "gptq.int8":
        bits = 8
    else:
        raise RuntimeError(f"unknown/unsupported quantization mode {quantize}")

    # we avoid loading the entire model on the GPU and do this block by block
    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    ckpt_path = os.path.join(out_dir, model_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(checkpoint['model_args'])
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return ids

    # tokenize the dataset
    text = get_sample_data()
    encoded_text = torch.tensor(process(text))

    # tokenizer = Tokenizer(tokenizer_path)

    # test_string = get_sample_data()
    # encoded_text = tokenizer.encode(
    #     test_string,
    #     bos=True,
    #     eos=False,
    # )
    block_size = 2048  # this is for compat with gptq, and indeed we get much worse beyond this (https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L30)
    encoded_text = encoded_text[: n_samples * block_size].reshape(n_samples, block_size)

    t0 = time.perf_counter()
    llama_blockwise_quantization(model, encoded_text, device, bits=bits)
    t = time.perf_counter() - t0

    print(
        f"\n\nTime for quantization: {t:.02f} sec total",
        file=sys.stderr,
    )
    print(
        f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB",
        file=sys.stderr,
    )
    print(model.state_dict().keys())
    checkpoint_2 = {
                    'model': model.state_dict(),
                    'optimizer': checkpoint['optimizer'],
                    'model_args': checkpoint['model_args'],
                    'iter_num': checkpoint['iter_num'],
                    'best_val_loss': checkpoint['best_val_loss'],
                    'config': checkpoint['config'],
                }
    print(f"saving quantized model to {output_path}")
    torch.save(checkpoint_2, os.path.join(out_dir, output_path))
    return model


if __name__ == "__main__":
    model = quantizing(output_path=model_name[:-3] + '_q.pt', quantize='gptq.int4')
    model.to(device)
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
    start = "\nAs the man walked down the stairs, " # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 1 # number of samples to draw
    max_new_tokens = 10000 # number of tokens generated in each sample
    temperature = 1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 50 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 10
    model_name = 'ckpt_r_1024.pt'
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
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
                y = model.generate(x, max_new_tokens, temperature=temperature)
                #print(y)
                print(decode(y[0].tolist()))
                end = time.time()
                print('---------------')
                duration = end - start
                print(f"Elapsed time: {duration}")
                time_per_sample.append(duration)
    print(time_per_sample)
    print(sum(time_per_sample) / len(time_per_sample))
