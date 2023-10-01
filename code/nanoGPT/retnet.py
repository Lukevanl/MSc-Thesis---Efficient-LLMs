from copy import deepcopy
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from retention import (
    ActivationString,
    MultiScaleRetention,
    _get_activation_fn,
)


class RetNetDecoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerDecoderLayer', but with changes:
    #   - use MultiScaleRetention instead of MultiheadAttention
    #   - no cross-attention layer, since retention doesn't play well with that

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        # retention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.retention = MultiScaleRetention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        # feedforward block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Check that we're following the same initialization as the paper
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward_parallel(self, x: Tensor) -> Tensor:
        def _retention_block(x: Tensor) -> Tensor:
            x, _ = self.retention.forward_parallel(x, x, x)
            return self.dropout(x)

        if self.norm_first:
            x = x + _retention_block(self.norm1(x))
            x = x + self._feedforward_block(self.norm2(x))
        else:
            x = x + self.norm1(_retention_block(x))
            x = x + self.norm2(self._feedforward_block(x))

        return x

    def forward_recurrent(
        self, x: Tensor, seq_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_recurrent(
                x, x, x, seq_idx=seq_idx, prev_state=prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state

    def forward_chunkwise(
        self, x: Tensor, start_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_chunkwise(
                x, x, x, start_idx=start_idx, prev_state=prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)


class RetNetDecoder(nn.Module):
    def __init__(self, decoder_layer: RetNetDecoderLayer, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward_parallel(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            assert isinstance(layer, RetNetDecoderLayer)
            x = layer.forward_parallel(x)
        return x

    def forward_recurrent(
        self, x: Tensor, seq_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        if not prev_states:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, RetNetDecoderLayer)
            x, state = layer.forward_recurrent(x, seq_idx, prev_state)
            states.append(state)
        return x, states

    def forward_chunkwise(
        self, x: Tensor, start_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        if not prev_states:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, RetNetDecoderLayer)
            x, state = layer.forward_chunkwise(x, start_idx, prev_state)
            states.append(state)
        return x, states

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)


class RetNet(nn.Module):
    def __init__(
        self,
        num_tokens: int,  # usually obtained from the tokenizer
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
        dim_feedforward: int = 2048,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_tokens, d_model, device=device, dtype=dtype)
        decoder_layer = RetNetDecoderLayer(
            d_model,
            nhead,
            dropout=dropout,
            activation=activation,
            dim_feedforward=dim_feedforward,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.decoder = RetNetDecoder(decoder_layer, num_layers)
        self.out = nn.Linear(d_model, num_tokens, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward_parallel(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.decoder.forward_parallel(x)
        x = self.out(x)
        return x

    def forward_recurrent(
        self, x: Tensor, seq_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        x = self.embedding(x)
        x, states = self.decoder.forward_recurrent(
            x, seq_idx=seq_idx, prev_states=prev_states
        )
        x = self.out(x)
        return x, states

    def forward_chunkwise(
        self, x: Tensor, start_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        x = self.embedding(x)
        x, states = self.decoder.forward_chunkwise(
            x, start_idx=start_idx, prev_states=prev_states
        )
        x = self.out(x)
        return x, states

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        logits = self.forward_parallel(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
    
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
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


def retnet_1_3b(
    num_tokens: int,  # usually obtained from the tokenizer
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> RetNet:
    """RetNet 1.3B configuration from the paper:
    https://arxiv.org/pdf/2307.08621v3.pdf
    """
    return RetNet(
        num_tokens=num_tokens,
        d_model=2048,
        nhead=8,
        num_layers=24,
        dim_feedforward=4096,
        device=device,
        dtype=dtype,
    )


def retnet_2_7b(
    num_tokens: int,  # usually obtained from the tokenizer
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> RetNet:
    """RetNet 2.7B configuration from the paper:
    https://arxiv.org/pdf/2307.08621v3.pdf
    """
    return RetNet(
        num_tokens=num_tokens,
        d_model=2560,
        nhead=10,
        num_layers=32,
        dim_feedforward=5120,
        device=device,
        dtype=dtype,
    )


def retnet_6_7b(
    num_tokens: int,  # usually obtained from the tokenizer
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> RetNet:
    """RetNet 6.7B configuration from the paper:
    https://arxiv.org/pdf/2307.08621v3.pdf
    """
    return RetNet(
        num_tokens=num_tokens,
        d_model=4096,
        nhead=16,
        num_layers=32,
        dim_feedforward=8192,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    num_tokens = 1000
    batch_size = 1
    seq_len = 8
    d_model = 32
    nhead = 2
    num_layers = 2
    device = "cuda"
    dtype = torch.float32

    size = (batch_size, seq_len)
    x = torch.randint(0, num_tokens, size=size, device=device)
    net = RetNet(
        num_tokens=num_tokens,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    ).eval()

    y_parallel = net.forward_parallel(x)

    y_recurrent = torch.zeros_like(y_parallel)
    prev_states: Sequence[Optional[Tensor]] = [None] * num_layers
    for i in range(seq_len):
        xr = x[:, i]
        y_recurrent[:, i], prev_states = net.forward_recurrent(
            xr, seq_idx=i, prev_states=prev_states
        )

    torch.testing.assert_close(y_parallel, y_recurrent)
