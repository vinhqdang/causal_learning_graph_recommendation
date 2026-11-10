"""DIN and SIM-hard target attention over the users' pre-cut lifelong sequences (lifelong-aware deep baselines).

Sequences (built by experiments/v12_build_seq.py): for every dataset user, the last L events strictly before the
cut-off (drup.leaf.CUT_MS) on videos outside the random pool, right-aligned and time-ordered. A token is the content
key codes of the event's video (tag, author, music; 0 = missing) and its engagement signal (2 * click + long view).
Video ids are not embedded: out-of-pool videos are never candidates, so an id embedding of them would never be
trained, and the key codes are the representation that transfers to the pool (as in drup/leaf.py).

Score of a user u and a candidate item i (DIN, Zhou et al. 2018, activation unit over [h, q, h - q, h * q]):
    q_i      = W_q [e_i, e_tag(i), e_author(i), e_music(i)]
    a_t      = softmax_t act([h_t, q_i, h_t - q_i, h_t * q_i])          over the valid tokens of u
    interest = sum_t a_t h_t,   h_t = e_tag + e_author + e_music + e_signal of token t
    s(u, i)  = MLP([interest, q_i, e_u]) + <e_u, e_i>
SIM-hard (Pi et al. 2020, hard search): before the attention, keep the K most recent tokens whose tag or author equals
the candidate's (none: zero interest). The collaborative part <e_u, e_i> and the ID embeddings are trained with BPR on
the pre-cut standard log (clicks as positives, uniform negatives outside the clicks), as LightGCN is.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

L_DEFAULT = 200
SIG_N = 4          # signal = 2 * is_click + long_view
NA_TAG = ("nan", "<NA>")


def item_codes(item_feat, vocab_tag, vocab_author, vocab_music):
    """(n, 3) int64 key codes of the dataset items (tag, author, music), in the code space of the sequence file;
    0 = missing or not in the vocabulary."""
    tag = item_feat["tag"].astype(str).values
    tag = np.where(np.isin(tag, NA_TAG), "", tag)
    it = pd.Index(np.asarray(vocab_tag).astype(str)).get_indexer(tag)
    out = [np.where(it < 0, 0, it + 1)]
    for col, vocab in (("author_id", vocab_author), ("music_id", vocab_music)):
        v = pd.to_numeric(item_feat[col], errors="coerce").fillna(-1).astype(np.int64).values
        vv = np.asarray(vocab, dtype=np.int64)
        pos = np.searchsorted(vv, v)
        pos_c = np.minimum(pos, max(len(vv) - 1, 0))
        hit = (len(vv) > 0) & (v >= 0) & (vv[pos_c] == v)
        out.append(np.where(hit, pos_c + 1, 0))
    return np.stack(out, 1).astype(np.int64)


def load_sequences(path, users_raw, item_feat, L=L_DEFAULT):
    """Token tables aligned with the dataset rows (users_raw).

    Returns dict: tok (m, L, 4) int64 [tag, author, music, signal] codes, right-aligned (the last event is the last
    position), valid (m, L) bool, length (m,) int64 (users without a sequence have length 0), cand (n, 3) int64 item
    codes, vocab sizes n_tag, n_author, n_music (codes 0..size-1, 0 = missing), and the sequence-file cut-off."""
    Z = np.load(path, allow_pickle=True)
    L_file = int(Z["L"])
    L = min(int(L), L_file)
    seq_users = np.asarray(Z["users"], dtype=np.int64)
    rows = pd.Index(seq_users).get_indexer(np.asarray(users_raw, dtype=np.int64))
    have = rows >= 0
    src = np.where(have, rows, 0)
    m = len(users_raw)
    tok = np.zeros((m, L, 4), dtype=np.int64)
    for c, key in enumerate(("tag", "author", "music", "sig")):
        arr = np.asarray(Z[key])[:, L_file - L:]          # the last L events of the stored window
        tok[:, :, c] = np.where(have[:, None], arr[src], 0)
    length = np.where(have, np.minimum(np.asarray(Z["length"], dtype=np.int64)[src], L), 0)
    valid = np.arange(L)[None, :] >= (L - length)[:, None]
    vocab = {"tag": np.asarray(Z["vocab_tag"]).astype(str), "author": np.asarray(Z["vocab_author"], dtype=np.int64),
             "music": np.asarray(Z["vocab_music"], dtype=np.int64)}
    cand = item_codes(item_feat, vocab["tag"], vocab["author"], vocab["music"])
    return {"tok": torch.as_tensor(tok), "valid": torch.as_tensor(valid), "length": length, "cand": torch.as_tensor(cand),
            "n_tag": len(vocab["tag"]) + 1, "n_author": len(vocab["author"]) + 1,
            "n_music": len(vocab["music"]) + 1, "L": L, "cut_ms": int(Z["cut_ms"])}


class DIN(nn.Module):
    """DIN target attention over the lifelong tokens (sim_k = 0), or SIM-hard retrieval of the sim_k most recent
    matching tokens before the attention (sim_k > 0). See the module docstring."""

    def __init__(self, m, n, n_tag, n_author, n_music, cand, d=16, hidden=16, mlp=32, sim_k=0, use_id=True):
        super().__init__()
        self.d, self.sim_k, self.use_id = int(d), int(sim_k), bool(use_id)
        self.user = nn.Embedding(m, d, sparse=True)
        self.item = nn.Embedding(n, d, sparse=True)
        self.tag = nn.Embedding(n_tag, d, padding_idx=0, sparse=True)
        self.author = nn.Embedding(n_author, d, padding_idx=0, sparse=True)
        self.music = nn.Embedding(n_music, d, padding_idx=0, sparse=True)
        self.sig = nn.Embedding(SIG_N, d, sparse=True)
        self.query = nn.Linear(4 * d, d)
        self.act = nn.Sequential(nn.Linear(4 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(3 * d, mlp), nn.ReLU(), nn.Linear(mlp, 1))
        for e in (self.user, self.item, self.tag, self.author, self.music, self.sig):
            nn.init.normal_(e.weight, std=0.05)
            if e.padding_idx is not None:
                with torch.no_grad():
                    e.weight[e.padding_idx].zero_()
        self.register_buffer("cand_codes", torch.as_tensor(cand, dtype=torch.long))

    def sparse_parameters(self):
        """Sparse embedding tables (SparseAdam). The user and item ID tables are included even when use_id is False
        (they are then unused and receive no gradient)."""
        return [self.user.weight, self.item.weight, self.tag.weight, self.author.weight, self.music.weight,
                self.sig.weight]

    def score(self, rows, cand, tok, valid):
        """rows (B,) dataset rows, cand (B, C) item ids, tok (B, L, 4) token codes of the rows, valid (B, L) bool.
        Returns (B, C) scores."""
        B, C = cand.shape
        L, d = tok.shape[1], self.d
        cc = self.cand_codes[cand]                                            # (B, C, 3)
        ie = self.item(cand) if self.use_id else torch.zeros(B, C, d)         # (B, C, d); zero without ID embeddings
        x = torch.cat([ie, self.tag(cc[..., 0]), self.author(cc[..., 1]), self.music(cc[..., 2])], -1)
        q = self.query(x)                                                     # (B, C, d)
        h = self.tag(tok[..., 0]) + self.author(tok[..., 1]) + self.music(tok[..., 2]) + self.sig(tok[..., 3])
        if self.sim_k > 0:
            ctag, cauth = cc[..., 0:1], cc[..., 1:2]                          # (B, C, 1)
            match = ((tok[:, None, :, 0] == ctag) & (ctag > 0)) | ((tok[:, None, :, 1] == cauth) & (cauth > 0))
            match = match & valid[:, None, :]                                 # (B, C, L)
            recency = torch.arange(1, L + 1, device=tok.device, dtype=torch.float32)
            key = match.float() * recency
            k = min(self.sim_k, L)
            top = key.topk(k, dim=-1)
            idx = top.indices                                                 # (B, C, k)
            hs = torch.gather(h[:, None].expand(B, C, L, d), 2, idx[..., None].expand(B, C, k, d))
            ms = top.values > 0
        else:
            hs = h[:, None].expand(B, C, L, d)
            ms = valid[:, None, :].expand(B, C, L)
        qe = q[:, :, None, :].expand_as(hs)
        s = self.act(torch.cat([hs, qe, hs - qe, hs * qe], -1)).squeeze(-1)  # (B, C, T)
        s = s.masked_fill(~ms, -1e9)
        w = torch.softmax(s, -1) * ms.float()                                 # all-zero when nothing matches
        interest = (w[..., None] * hs).sum(2)                                 # (B, C, d)
        e_u = (self.user(rows) if self.use_id else torch.zeros(B, d))[:, None, :].expand(B, C, d)
        out = self.head(torch.cat([interest, q, e_u], -1)).squeeze(-1)
        return out + (e_u * ie).sum(-1)


@torch.no_grad()
def score_rows(model, rows, cand, tok, valid, chunk=16):
    """(U, C) scores of the candidate matrix cand (U, C) of the users rows (U,), chunked over users."""
    out = torch.empty(cand.shape, dtype=torch.float32)
    for r0 in range(0, len(rows), chunk):
        sl = slice(r0, r0 + chunk)
        out[sl] = model.score(rows[sl], cand[sl], tok[rows[sl]], valid[rows[sl]]).float()
    return out


def fit_bpr(model, X, tok, valid, lr, wd, epochs, batch, on_epoch, seed=0, log=None):
    """BPR on the clicked pairs X (bool (m, n), the pre-cut in-pool clicks): one uniform negative per positive outside
    X (one resample if it falls in X). Sparse embeddings use SparseAdam, the MLPs Adam. on_epoch(model, ep) is called
    after every epoch (model in eval mode) and returns True to stop. Returns the number of epochs run."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    pu, pi = torch.nonzero(X, as_tuple=True)
    n = X.shape[1]
    sp = model.sparse_parameters()
    spid = {id(p) for p in sp}
    dense = [p for p in model.parameters() if id(p) not in spid]
    opt_s = torch.optim.SparseAdam(sp, lr=lr)
    opt_d = torch.optim.Adam(dense, lr=lr)
    ran = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(pu), generator=g)
        for s0 in range(0, len(perm), batch):
            b = perm[s0:s0 + batch]
            u, i = pu[b], pi[b]
            j = torch.randint(0, n, (len(u),), generator=g)
            j = torch.where(X[u, j], torch.randint(0, n, (len(u),), generator=g), j)
            cand = torch.stack([i, j], 1)
            s = model.score(u, cand, tok[u], valid[u])
            loss = -F.logsigmoid(s[:, 0] - s[:, 1]).mean()
            reg = wd * (model.user(u).pow(2).sum() + model.item(cand).pow(2).sum()) / len(u) if model.use_id else 0.0
            opt_s.zero_grad()
            opt_d.zero_grad()
            (loss + reg).backward()
            opt_s.step()
            opt_d.step()
        model.eval()
        ran = ep + 1
        with torch.no_grad():
            stop = on_epoch(model, ep)
        if log:
            log(f"  epoch {ep} loss {float(loss):.4f}")
        if stop:
            break
    return ran
