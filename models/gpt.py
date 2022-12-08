"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import sys

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MultiHeadSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_head, n_embd, resid_pdrop, attn_pdrop, block_size, right_masking=False):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # dropout regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence (GPT-specific)
        if right_masking:
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
        else:
            self.register_buffer("mask", None)

        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        batch_size, L, d_embd = x.size() # batch size, sequence length, embedding dimensionality (d_embd)
        assert d_embd == self.n_embd, "Embedding size mismatch!!"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, L, self.n_head, d_embd // self.n_head).transpose(1, 2) # (bs, nh, L, hs)
        q = q.view(batch_size, L, self.n_head, d_embd // self.n_head).transpose(1, 2) # (bs, nh, L, hs)
        v = v.view(batch_size, L, self.n_head, d_embd // self.n_head).transpose(1, 2) # (bs, nh, L, hs)

        # causal self-attention; Self-attend: (bs, nh, L, hs) x (bs, nh, hs, L) -> (bs, nh, L, L)
        att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        if self.mask is not None:
            att = att.masked_fill(self.mask[:,:,:L,:L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (bs, nh, L, L) x (bs, nh, L, hs) -> (bs, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(batch_size, L, d_embd) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_head, n_embd, resid_pdrop, attn_pdrop, block_size, right_masking):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_head, n_embd, resid_pdrop, attn_pdrop, block_size, right_masking=right_masking)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model """

    def __init__(self, model_type=None, n_layer=None, n_head=None, n_embd=None, vocab_size=None, block_size=None, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, right_masking=False):
        """
        :param model_type: select model type for pre-defined num. layer, num. heads, embedding dimension.
        :param n_layer: number of transformer blocks.
        :param n_head: number of heads in multi-head block.
        :param n_embd: number of embedding dimension throughout the architecture.
        :param vocab_size: size of the input vocabulary.
        :param block_size: input dimension (e.g. sentence length).
        :param embd_pdrop: embedding layer dropout probability.
        :param resid_pdrop: attention output dropout probability.
        :param attn_pdrop: attention score dropout probability (randomly set some scores to zero).
        """
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size

        type_given = model_type is not None
        params_given = all([n_layer is not None, n_head is not None, n_embd is not None])
        assert type_given ^ params_given, "One and only one between model type and (n_layer, n_head, n_embd) must be provided." # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            n_layer, n_head, n_embd = {
                # names follow the huggingface naming conventions
                # GPT-1         (n_layer, n_head, n_embd)
                'openai-gpt':   (12, 12, 768),  # 117M params
                # GPT-2 configs
                'gpt2':         (12, 12, 768),  # 124M params
                'gpt2-medium':  (24, 16, 1024), # 350M params
                'gpt2-large':   (36, 20, 1280), # 774M params
                'gpt2-xl':      (48, 25, 1600), # 1558M params
                # Gophers
                'gopher-44m':   (8, 16, 512),
                # Tiny versions
                'gpt-mini':     (6, 6, 192),
                'gpt-micro':    (4, 4, 128),
                'gpt-nano':     (3, 3, 48),
            }[model_type]

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd), # TODO: why not the usual positional embedding (sin and cos)?
            drop = nn.Dropout(embd_pdrop),
            blocks = nn.ModuleList([TransformerBlock(n_head, n_embd, resid_pdrop, attn_pdrop, block_size, right_masking) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0., std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, x, y=None):
        device = x.device
        b, t = x.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(x) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if y is not None:
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # as in BERT, loss is computed only on the masked inputs
            loss = F.cross_entropy(logits[range(len(logits)), y[:, 0]], y[:, 1], ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx