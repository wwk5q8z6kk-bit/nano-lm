"""Controlled objective experiment. All test-time inputs are plain transcripts plus a slot query."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from research.cycle3 import exp_nanoscribe_p1_synthesis_01 as old

ROOT = Path(__file__).resolve().parent
SEEDS = (1481, 3527, 6029)
ARMS = ('full', 'answer', 'aligned')


def read_data(split):
    return [json.loads(line) for line in (ROOT / 'data' / (split + '.jsonl')).read_text().splitlines()]


def prompt(text, slot):
    return f'{old.BOS_CH}{text} QRY {slot} -> '


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def training_examples(train):
    tok = old.DeterministicBPETokenizer(vocab_size=384, protect_alnum_spans=True)
    tok.fit([prompt(e['transcript'], slot) + e['gold_state'][slot] + '\n' for e in train for slot in old.SLOTS])
    rows = []
    for enc in train:
        text = enc['transcript']
        for slot in old.SLOTS:
            pre = prompt(text, slot)
            value = enc['gold_state'][slot]
            ids = tok.encode(pre + value + '\n')
            pre_ids = tok.encode(pre)
            value_ids = tok.encode(value)
            assert ids == pre_ids + value_ids + [10]
            source_start = None
            if value != 'ABSTAIN':
                # Choose latest matching complete, whitespace-delimited transcript value.
                char_start = text.rindex(' ' + value + ' ') + 1
                source_start = len(tok.encode(old.BOS_CH + text[:char_start]))
                assert ids[source_start:source_start + len(value_ids)] == value_ids
            rows.append((torch.tensor(ids, dtype=torch.long), len(pre_ids), source_start))
    return tok, rows


def make_batch(rows):
    width = max(len(ids) for ids, _, _ in rows) - 1
    x = torch.zeros((len(rows), width), dtype=torch.long)
    y = torch.full_like(x, -100)
    answers = torch.full_like(x, -100)
    source_keys = torch.full_like(x, -100)
    for row, (ids, start, source) in enumerate(rows):
        length = len(ids) - 1
        x[row, :length] = ids[:-1]
        y[row, :length] = ids[1:]
        answers[row, start-1:length] = ids[start:]
        if source is not None:
            # Pointer key j selects value x[j+1]. Final newline is a vocabulary target.
            n_value = length - start
            source_keys[row, start-1:length-1] = torch.arange(source-1, source-1+n_value)
    return x, y, answers, source_keys


def alignment_loss(model, h, targets):
    # These are exact original pointer scores at supervised answer positions.
    b, t = torch.where(targets != -100)
    if len(b) == 0:
        return h.sum() * 0
    pointer = model.pointer
    queries = F.normalize(pointer.ptr_q(h[b, t]), dim=-1, eps=1e-6)
    keys = F.normalize(pointer.ptr_k(h[:, :-1]), dim=-1, eps=1e-6)
    cols = torch.arange(h.shape[1] - 1)
    scores = (queries[:, None, :] * keys[b]).sum(-1) * pointer.ptr_scale.exp().clamp(max=32)
    scores = scores + F.softplus(pointer.recency_param) * (cols[None, :] - t[:, None])
    scores = scores.masked_fill(cols[None, :] >= t[:, None], -1e9)
    return F.cross_entropy(scores, targets[b, t]) + F.softplus(-pointer.gate_proj(h[b, t])).mean()


@torch.no_grad()
def decode(model, tok, transcripts):
    model.eval()
    grouped = defaultdict(list)
    for i, tx in enumerate(transcripts):
        for slot in old.SLOTS:
            ids = tok.encode(prompt(tx, slot))
            grouped[len(ids)].append((i, slot, ids))
    answers = [{} for _ in transcripts]
    for group in grouped.values():
        cur = torch.tensor([row[2] for row in group])
        live = [True] * len(group)
        generated = [''] * len(group)
        for _ in range(16):
            nxt = model(cur)[:, -1].argmax(-1)
            cur = torch.cat((cur, nxt[:, None]), dim=1)
            for i, idx in enumerate(nxt.tolist()):
                if live[i]:
                    token = tok.decode([idx])
                    generated[i] += token.partition('\n')[0]
                    live[i] = '\n' not in token
            if not any(live):
                break
        for (i, slot, _), value in zip(group, generated):
            answers[i][slot] = value.strip()
    return answers


def run(seed, arm, output, steps=240, batch_size=24, evaluate=True):
    start_time = time.monotonic()
    output.mkdir(exist_ok=True, parents=True)
    train = read_data('train')
    tok, rows = training_examples(train)
    old.seed_everything(seed)
    reference = old.HybridSeqModel(384, 48, 4, 96, ['attn', 'attn'], 'none')
    ref_non_emb = old.count_trainable_params(reference) - reference.embed.weight.numel()
    dff = old._match_hybrid_d_ff(ref_non_emb, 384, 48, 4, ['hdcd', 'attn'], 'standard')
    model = old.HybridSeqModel(384, 48, 4, dff, ['hdcd', 'attn'], 'standard')
    initial = state_hash(model)
    opt = old.make_adamw_no_embed_decay(model, lr=4e-3, weight_decay=0.01)
    batch_ids = np.random.default_rng(91_000 + seed).integers(0, len(rows), size=(steps, batch_size))
    observed = {}
    if arm == 'aligned':
        def keep_features(module, args):
            observed['h'] = args[0]
        hook = model.pointer.register_forward_pre_hook(keep_features)
    losses = []
    train_tokens = answer_tokens = aligned_tokens = 0
    model.train()
    for step, indices in enumerate(batch_ids, start=1):
        x, full, answer, source = make_batch([rows[int(i)] for i in indices])
        train_tokens += int((full != -100).sum())
        answer_tokens += int((answer != -100).sum())
        aligned_tokens += int((source != -100).sum())
        log_probs = model(x)
        labels = full if arm == 'full' else answer
        nll = F.nll_loss(log_probs.flatten(0, 1), labels.flatten(), ignore_index=-100)
        loss = nll
        if arm == 'aligned':
            loss = loss + alignment_loss(model, observed.pop('h'), source)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 20 == 0 or step == steps:
            sample = {'step':step, 'loss':loss.item(), 'nll':nll.item()}
            losses.append(sample)
            print(json.dumps({'seed':seed,'arm':arm,**sample}), flush=True)
    if arm == 'aligned':
        hook.remove()
    tag = arm + '-' + str(seed)
    trained = state_hash(model)
    checkpoint = output / (tag + '.pt')
    torch.save(model.state_dict(), checkpoint)
    result = {'seed':seed,'arm':arm,'steps':steps,'batch_size':batch_size,'d_ff':dff,
              'parameters':old.count_trainable_params(model),'initial_state_sha256':initial,
              'trained_state_sha256':trained,'sample_schedule_sha256':hashlib.sha256(batch_ids.astype('<i8').tobytes()).hexdigest(),
              'sampled_full_tokens':train_tokens,'sampled_answer_tokens':answer_tokens,'sampled_copy_tokens':aligned_tokens,
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'train_seconds':time.monotonic()-start_time,'loss_samples':losses,
              'torch':str(torch.__version__),'numpy':np.__version__,'python':platform.python_version(),
              'architecture':platform.machine(),'dtype':'float32','threads':torch.get_num_threads()}
    if evaluate:
        result['predictions'] = {}
        for split in ('dev', 'test'):
            # No split state, offsets, answer pool, or oracle candidates passed into decoding.
            result['predictions'][split] = decode(model, tok, [e['transcript'] for e in read_data(split)])
    result['total_seconds'] = time.monotonic()-start_time
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (output/(tag+'.json')).write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('seed', type=int, choices=SEEDS)
    args.add_argument('arm', choices=ARMS)
    args.add_argument('output', type=Path)
    args.add_argument('--smoke', action='store_true')
    a=args.parse_args()
    run(a.seed, a.arm, a.output, steps=1 if a.smoke else 240,
        batch_size=2 if a.smoke else 24, evaluate=not a.smoke)
