"""A pretrained language model with a compiled exact-arithmetic stream.

ExactArithmeticLM runs a Hugging Face causal model and the compiled stream of
machines/arithmetic.py on the same token ids. The compiled stream reads each
token through an embedding indexed by the language model's vocabulary. Where it
emits a digit or a space, that token is forced; where it emits PASS, the
language model's logits are used unchanged. The streams do not share residual
dimensions, since the language model's normalization and precision would break
the compiled stream's exact integers. The tokenizer must write one digit per token.

    lm = ExactArithmeticLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    lm.generate_text("What is 48291 * 77153?")
"""

import torch
from torch import nn

from . import compile_machine
from .machines.arithmetic import PASS, ArithmeticSpec, code_of
from .torch_export import to_torch


class ExactArithmeticLM(nn.Module):
    def __init__(self, language_model, tokenizer, *, digits=12, max_context=1024):
        super().__init__()
        self.language_model, self.tokenizer = language_model, tokenizer
        spec = ArithmeticSpec(digits=digits, max_context=max_context)
        self.spec = spec
        self.compiled = compile_machine(spec.machine())
        self.stream = to_torch(self.compiled, attention="softmax")
        vocabulary = len(tokenizer)
        codes = [code_of(tokenizer.decode([i])) for i in range(vocabulary)]
        self.register_buffer("codes", torch.tensor(codes, dtype=torch.int64))
        # The token each emitted code forces: the plain digit, or a single space.
        forced = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(d)))[0] for d in range(10)]
        forced.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" "))[0])
        self.register_buffer("forced", torch.tensor(forced, dtype=torch.int64))

    @classmethod
    def from_pretrained(cls, name, *, dtype=torch.float16, device="cuda", **options):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        language_model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        return cls(language_model, tokenizer, **options).to(device)

    def stream_codes(self, input_ids):
        """The compiled stream's output code at every position of a 1-D token tensor."""
        if len(input_ids) > self.spec.max_context:
            raise ValueError("context longer than the compiled stream's")
        return self.stream(self.codes[input_ids]).argmax(dim=1)

    def merge(self, logits, code):
        """Language-model logits for one position, with the stream's token forced
        when it emits a digit (0..9) or a space."""
        if code == PASS:
            return logits
        token = self.forced[int(code) if code <= 9 else 10]
        merged = torch.full_like(logits, torch.finfo(logits.dtype).min)
        merged[token] = 0
        return merged

    @torch.no_grad()
    def forward(self, input_ids):
        """Merged next-token logits at every position of a 1-D token tensor."""
        logits = self.language_model(input_ids[None]).logits[0].float()
        codes = self.stream_codes(input_ids)
        return torch.stack([self.merge(logits[i], int(codes[i])) for i in range(len(input_ids))])

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, *, use_stream=True):
        """Greedy decoding with a cache per stream. Returns (new token ids, number the stream forced)."""
        from .runtime import _KVCache

        ids = input_ids.clone()
        if use_stream and len(ids) + max_new_tokens - 1 > self.spec.max_context:
            raise ValueError("context longer than the compiled stream's")
        cache = _KVCache(self.compiled, self.codes[ids].tolist(), len(ids) + max_new_tokens - 1) if use_stream else None
        past, step_input, forced = None, ids[None], 0
        # Chat models list their end-of-turn tokens in their generation config.
        config = getattr(self.language_model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None) or self.tokenizer.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        for step in range(max_new_tokens):
            out = self.language_model(step_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            if use_stream:
                if step:
                    cache.append(int(self.codes[ids[-1]]))  # The token chosen last step.
                code = int(cache.logits().argmax()) + self.compiled.config["lower"]
                forced += code != PASS
                logits = self.merge(logits, code)
            token = int(logits.argmax())
            ids = torch.cat([ids, torch.tensor([token], device=ids.device)])
            step_input = ids[-1:][None]
            if token in stops:
                break
        return ids[len(input_ids) :], forced

    def chat_ids(self, question, system=None):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"][0].to(self.codes.device)

    def generate_text(self, question, system=None, max_new_tokens=96, use_stream=True):
        new, _ = self.generate(self.chat_ids(question, system), max_new_tokens, use_stream=use_stream)
        return self.tokenizer.decode(new, skip_special_tokens=True)
