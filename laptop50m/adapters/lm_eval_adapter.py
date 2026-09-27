"""lm-evaluation-harness adapter: exposes a Laptop-50M GPT + `tokenizers` BPE as an lm_eval `TemplateLM`.

Scoring is plain causal log-likelihood: for (context, continuation) the input is the last `max_length + 1`
tokens of context+continuation minus the final token; sequences are right-padded (causal attention makes
right padding invisible to the scored positions). Empty contexts are conditioned on <|endoftext|>.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
from tqdm import tqdm

EOT = "<|endoftext|>"


class Laptop50MLM(TemplateLM):
    def __init__(self, model, tokenizer, device: str = "mps", batch_size: int = 32, max_length: int = 512):
        super().__init__()
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self._device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._eot = tokenizer.token_to_id(EOT)

    # --- TemplateLM contract -------------------------------------------------
    @property
    def eot_token_id(self) -> int:
        return self._eot

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return "laptop50m-bpe16k"

    def tok_encode(self, string: str, add_special_tokens=None, **kwargs) -> list[int]:
        return self.tok.encode(string, add_special_tokens=False).ids

    def tok_decode(self, ids) -> str:
        return self.tok.decode(list(ids))

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = True, **kwargs):
        results: list = [None] * len(requests)
        # longest first -> stable padding, OOM (if any) shows up on the first batch
        order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        for chunk in tqdm(batches, disable=disable_tqdm, desc="loglikelihood"):
            inps, conts = [], []
            for i in chunk:
                _, ctx, cont = requests[i]
                assert 0 < len(cont) <= self.max_length, "continuation must fit in the window"
                full = (list(ctx) + list(cont))[-(self.max_length + 1):]
                inps.append(full[:-1])
                conts.append(list(cont))
            T = max(len(x) for x in inps)
            x = torch.full((len(inps), T), self._eot, dtype=torch.long)
            for j, inp in enumerate(inps):
                x[j, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            logits, _ = self.model(x.to(self._device))
            logp = F.log_softmax(logits.float(), dim=-1)
            for j, i in enumerate(chunk):
                L, n = len(inps[j]), len(conts[j])
                lp = logp[j, L - n:L]
                tgt = torch.tensor(conts[j], dtype=torch.long, device=lp.device)
                ll = float(lp.gather(-1, tgt[:, None]).sum())
                greedy = bool((lp.argmax(-1) == tgt).all())
                results[i] = (ll, greedy)
                if self.cache_hook is not None and requests[i][0] is not None:
                    self.cache_hook.add_partial("loglikelihood", requests[i][0], (ll, greedy))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        out = []
        for req in tqdm(requests, disable=disable_tqdm, desc="loglikelihood_rolling"):
            (string,) = req.args
            windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
                token_list=self.tok_encode(string), prefix_token=self.prefix_token_id,
                max_seq_len=self.max_length, context_len=1)]
            res = self._loglikelihood_tokens([(None, c, t) for c, t in windows])
            ll = sum(r[0] for r in res)
            out.append(ll)
            if self.cache_hook is not None:
                self.cache_hook.add_partial("loglikelihood_rolling", (string,), ll)
        return out

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Laptop-50M is evaluated on log-likelihood tasks only")
