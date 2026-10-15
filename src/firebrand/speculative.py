"""Speculative decoding with any draft.

generate(model, prompt, draft, max_steps) returns exactly model.generate(prompt,
max_steps); the draft only affects speed. Drafted tokens are checked in one
parallel pass on the GPU, the first wrong one is replaced by the model's own
token, and a drafter (anything with run(prefix, max_steps), such as draft.Draft)
is asked again from there. Whatever the draft cannot cover is generated serially.
"""

import numpy as np

from .gpu import Verifier


def generate(model, prompt, draft, max_steps, *, verifier=None, backend="c", attempts=8):
    """model.generate(prompt, max_steps), using `draft` (proposed tokens after the
    prompt, or a drafter) to skip serial steps. A drafter gets `attempts` rounds
    that each end at a wrong token before the rest is generated serially."""
    tokens = [int(t) for t in prompt]
    halts = set(model.config["halt_values"])
    drafter = draft if hasattr(draft, "run") else None
    proposal = None if drafter else [int(t) for t in np.asarray(draft).reshape(-1)]
    verifier = verifier or Verifier(model)
    for _ in range(attempts if drafter else 1):
        remaining = max_steps - (len(tokens) - len(prompt))
        if remaining <= 0 or tokens[-1] in halts:
            return tokens
        if drafter:
            proposal = drafter.run(tokens, remaining)[len(tokens) :]
        proposal = proposal[:remaining]
        # Generation stops after a halt token, so a draft continuing past one is cut there.
        for i, token in enumerate(proposal):
            if token in halts:
                proposal = proposal[: i + 1]
                break
        if not proposal:
            break
        wrong, token = verifier.check(tokens + proposal, len(tokens))
        if wrong is None:
            tokens += proposal
            # The check also yields the transformer's token after the proposal.
            if proposal[-1] not in halts and len(tokens) - len(prompt) < max_steps:
                tokens.append(token)
            continue
        tokens += proposal[: wrong - len(tokens)] + [token]
    remaining = max_steps - (len(tokens) - len(prompt))
    if remaining <= 0 or tokens[-1] in halts:
        return tokens
    return model.generate(tokens, remaining, backend=backend)
