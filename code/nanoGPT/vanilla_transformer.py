import os
from typing import List, Optional, Sequence, Tuple, Union
import inspect
import plotly.graph_objects as go
import torch
import json
from torch import Tensor, nn
from torch.nn import functional as F
from dataclasses import dataclass
from retnet import RetNet, retnet_1_3b
import time

@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class TransformerLM(nn.Module):
    def __init__(
        self,
        config,
        num_tokens: int,  # usually obtained from the tokenizer
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_batch_size: int = 1,
        max_seq_length: int = 8192,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_tokens, d_model, device=device, dtype=dtype)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.config = config
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers
        )
        self.out = nn.Linear(d_model, num_tokens, device=device, dtype=dtype)

        # TODO: This may not be identical to what was benchmarked in the paper.
        # They specifically mention a KV cache, and this isn't technically that.
        # (The key/value projections aren't being cached, just the memory values
        # before KV projection.)  However, this seemed easier to implement, since
        # I don't have to fiddle with Flash Attention or custom Transformer layer
        # implementations.  If time allows, consider implementing a KV cache.
        #
        # a rough-and-dirty memory (KV) cache
        self.cache = torch.zeros(
            (max_batch_size, max_seq_length, d_model),
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor, targets=None, start_pos=0):
        batch_size, seq_len = x.shape
        x = self.embeddings(x)

        # memory cache
        #self.cache[:batch_size, start_pos : start_pos + seq_len] = x
        memory = x[:batch_size, start_pos : start_pos + seq_len]
        x = self.decoder.forward(x, memory)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.out(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.out(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
    
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return 0
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, prev_state=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



def transformer_1_3b(
    num_tokens: int,  # usually obtained from the tokenizer
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> TransformerLM:
    """Transformer configuration to match RetNet 1.3B from the paper:
    https://arxiv.org/pdf/2307.08621v3.pdf
    """
    return TransformerLM(
        num_tokens=num_tokens,
        d_model=2048,
        nhead=8,
        num_layers=24,
        dim_feedforward=4096,
        device=device,
        dtype=dtype,
    )


# @torch.no_grad()
# def benchmark_inference_throughput(
#     retnet: RetNet, transformer: TransformerLM, seq_lengths: Sequence[int]
# ) -> Tuple[List[float], List[float]]:
#     retnet_throughputs: List[float] = []
#     transformer_throughputs: List[float] = []

#     print("\nBenchmarking inference throughput...")
#     for seq_length in seq_lengths:
#         torch.cuda.empty_cache()
#         print(f"seq_length: {seq_length}")
#         x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, seq_length), device=DEVICE)

#         # Benchmark *recurrent* RetNet formulation for inference
#         retnet_result = benchmark(
#             retnet.forward_recurrent, x[:, 0], seq_idx=0, prev_states=[]
#         )
#         retnet_throughput = BATCH_SIZE / retnet_result.mean
#         print(f"RetNet throughput: {retnet_throughput:.3f} tokens/s")
#         # Benchmark *parallel* transformer for inference (with memory cache)
#         _ = transformer(x, start_pos=0)  # warmup memory cache
#         transformer_result = benchmark(transformer, x[:, -1:], start_pos=seq_length - 1)
#         transformer_throughput = BATCH_SIZE / transformer_result.mean
#         print(f"Transformer throughput: {transformer_throughput:.3f} tokens/s")

#         retnet_throughputs.append(retnet_throughput)
#         transformer_throughputs.append(transformer_throughput)

#     return retnet_throughputs, transformer_throughputs


# @torch.no_grad()
# def measure_inference_memory(
#     retnet: RetNet, transformer: TransformerLM, seq_lengths: Sequence[int]
# ) -> Tuple[List[float], List[float]]:
#     retnet_memories: List[float] = []
#     transformer_memories: List[float] = []

#     print("\nMeasuring inference memory...")
#     for seq_length in seq_lengths:
#         torch.cuda.empty_cache()
#         print(f"seq_length: {seq_length}")
#         x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, seq_length), device=DEVICE)

#         # Measure *recurrent* RetNet formulation for inference
#         retnet_result = profile(
#             retnet.forward_recurrent, x[:, 0], seq_idx=0, prev_states=[]
#         )
#         retnet_memory_gib = retnet_result.peak / 2**30
#         print(f"RetNet memory: {retnet_memory_gib:.3f} GiB")
#         # Measure *parallel* transformer for inference (with memory cache)
#         _ = transformer(x, start_pos=0)  # warmup memory cache
#         transformer_result = profile(transformer, x[:, -1:], start_pos=seq_length - 1)
#         transformer_memory_gib = transformer_result.peak / 2**30
#         print(f"Transformer memory: {transformer_memory_gib:.3f} GiB")

#         retnet_memories.append(retnet_memory_gib)
#         transformer_memories.append(transformer_memory_gib)

#     return retnet_memories, transformer_memories


# if __name__ == "__main__":
#     retnet = retnet_1_3b(NUM_TOKENS, device=DEVICE, dtype=DTYPE).eval()
#     transformer = transformer_1_3b(NUM_TOKENS, device=DEVICE, dtype=DTYPE).eval()

#     retnet_footprints, transformer_footprints = measure_inference_memory(
#         retnet, transformer, seq_lengths=SEQ_LENGTHS
#     )
#     retnet_throughputs, transformer_throughputs = benchmark_inference_throughput(
#         retnet, transformer, seq_lengths=SEQ_LENGTHS
#     )

#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter(
#             x=SEQ_LENGTHS,
#             y=retnet_footprints,
#             name="RetNet",
#             mode="lines+markers",
#             line={"color": "blue"},
#             marker={"color": "blue"},
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=SEQ_LENGTHS,
#             y=transformer_footprints,
#             name="Transformer",
#             mode="lines+markers",
#             line={"color": "red"},
#             marker={"color": "red"},
#         )
#     )
#     fig.update_layout(
#         title="Inference Memory Footprint",
#         xaxis_title="Sequence Length",
#         yaxis_title="GPU Memory (GiB)",
#         xaxis={"tickmode": "array", "tickvals": SEQ_LENGTHS},
#         # place legend at center-left
#         legend={"x": 0.1, "y": 0.5},
#     )
#     fig.write_image(os.path.join("doc", "inference-memory.png"))

#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter(
#             x=SEQ_LENGTHS,
#             y=retnet_throughputs,
#             name="RetNet",
#             mode="lines+markers",
#             line={"color": "blue"},
#             marker={"color": "blue"},
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=SEQ_LENGTHS,
#             y=transformer_throughputs,
#             name="Transformer",
#             mode="lines+markers",
#             line={"color": "red"},
#             marker={"color": "red"},
#         )
#     )
#     fig.update_layout(
#         title="Inference Throughput",
#         xaxis_title="Sequence Length",
#         yaxis_title="Throughput (tokens/s)",
#         xaxis={"tickmode": "array", "tickvals": SEQ_LENGTHS},
#         # place legend at center-left
#         legend={"x": 0.1, "y": 0.5},
#     )
#     fig.write_image(os.path.join("doc", "inference-throughput.png"))