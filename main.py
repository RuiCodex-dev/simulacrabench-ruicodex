"""SimulacraBench submission (Rui Codex).

predict(frame, schema) fits, at call time, conditional models of every PREDICT item given the GIVEN block on the
visible (complete) respondents, chooses hyper-parameters and ensemble weights by K-fold cross-validation on those
same visible rows (out-of-fold log score with the grader's floor mixing), refits on all visible rows and returns
one probability vector per blank cell in canonical order.

Models (all multi-task over the items of one instrument, trained jointly on GPU when present):
  * crowd marginal (smoothed counts)            - the reference
  * multinomial logistic regression, L2         - one-hot + ordinal codes of the GIVEN block
  * multi-task MLP                              - shared hidden layer, one softmax head per item
  * calibrated naive Bayes over GIVEN features  - per-item temperature
Stacking: per-item convex weights over the models (EM on out-of-fold probabilities, shrunk to the instrument-wide
weights), then a per-item temperature. Gate cells whose parent is visible use the observed gate rule.
Every stage is guarded: if anything fails, the marginal is returned for the affected item.
"""
import math
import os
import time

import numpy as np
import pandas as pd

try:
    import torch
    TORCH = True
except Exception:  # pragma: no cover
    TORCH = False

FLOOR = 1e-3
SEED = 20261029
T_IMPORT = time.time()
KNOWN_J = [12, 76, 213]          # PREDICT item counts of the three instruments (UNICEF, World Bank, UNHCR)
_DONE_J = []                     # instruments already handled in this container


def _allowance(schema, n_visible, J):
    """Seconds this predict() call may spend: the phase budget (900 s dev, 3600 s final, detected from the split
    counts) minus what is already spent, shared over the instruments still to come by their item counts."""
    sp = schema.get("split", {})
    total = 3600.0 if n_visible > sp.get("n_train", 10 ** 9) + 0.5 * sp.get("n_dev", 0) else 900.0
    total = float(os.environ.get("RC_TOTAL_BUDGET", total))
    left = 0.78 * total - (time.time() - T_IMPORT)
    rest = [j for j in KNOWN_J if j not in _DONE_J and j != J]
    return max(min(90.0, 0.25 * left), left * J / max(1, J + sum(rest)))

if TORCH:
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


# ----------------------------------------------------------------------------------------------------- helpers
def _options(schema, name):
    rec = schema["items"][name]
    return list(rec["values"]) + ([schema["gated_value"]] if rec.get("gate") else [])


def _floor_mix(p):
    """The grader's renormalise + flat mix, applied to a (..., K) array with a validity mask of -inf/0 handled by caller."""
    p = p / p.sum(-1, keepdims=True)
    k = p.shape[-1]
    return FLOOR + (1 - k * FLOOR) * p


def _encode(frame, schema, given, pred):
    """Integer codes of every GIVEN and PREDICT column (-1 = NaN / unknown level)."""
    codes = {}
    for name in given + pred:
        opts = _options(schema, name)
        idx = {str(o): i for i, o in enumerate(opts)}
        col = frame[name]
        out = np.full(len(col), -1, dtype=np.int64)
        for i, v in enumerate(col.tolist()):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out[i] = idx.get(str(v), -1)
        codes[name] = out
    return codes


def _features(codes, schema, given):
    """One-hot of each GIVEN level (+ an 'unknown' slot) and an ordinal code per GIVEN item."""
    blocks, ords = [], []
    for name in given:
        k = len(_options(schema, name))
        c = codes[name]
        oh = np.zeros((len(c), k + 1), dtype=np.float32)
        oh[np.arange(len(c)), np.where(c >= 0, c, k)] = 1.0
        blocks.append(oh)
        o = np.where(c >= 0, c / max(1, k - 1), 0.5).astype(np.float32)
        ords.append(o[:, None])
    X = np.concatenate(blocks + ords, axis=1) if blocks else np.zeros((len(next(iter(codes.values()))), 1), np.float32)
    keep = X.std(0) > 0
    return X[:, keep] if keep.any() else X


# ------------------------------------------------------------------------------------------------ torch models
class _Heads:
    """Packs J items with K_j options into an (n, J, Kmax) layout with -inf padding."""

    def __init__(self, ks):
        self.ks = np.array(ks)
        self.J, self.Kmax = len(ks), int(max(ks))
        mask = np.full((self.J, self.Kmax), -np.inf, dtype=np.float32)
        for j, k in enumerate(ks):
            mask[j, :k] = 0.0
        self.mask = torch.tensor(mask, device=DEV)

    def logp(self, logits):
        z = logits.view(-1, self.J, self.Kmax) + self.mask
        return torch.log_softmax(z, -1)


def _nll(logp, Y):
    """Mean negative log-lik over observed (Y>=0) cells. logp (n,J,K), Y (n,J) long."""
    m = Y >= 0
    g = torch.gather(logp, 2, Y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return -(g * m).sum() / m.sum().clamp(min=1)


def _fit_linear(X, Y, heads, wd, iters=150):
    torch.manual_seed(SEED)
    n, d = X.shape
    W = torch.zeros(d, heads.J * heads.Kmax, device=DEV, requires_grad=True)
    b = torch.zeros(heads.J * heads.Kmax, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _nll(heads.logp(X @ W + b), Y) + wd * (W * W).sum() / heads.J
        loss.backward()
        return loss

    opt.step(closure)
    W, b = W.detach(), b.detach()
    return lambda Xn: heads.logp(Xn @ W + b).exp()


def _fit_mlp(X, Y, heads, hidden, wd, epochs, drop=0.2, lr=3e-3, Xval=None, Yval=None, seed=SEED):
    """Returns (predict_fn, per-epoch validation nll list or None)."""
    torch.manual_seed(seed)
    n, d = X.shape
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(drop),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(drop),
                              torch.nn.Linear(hidden, heads.J * heads.Kmax)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    bs = min(1024, n)
    hist = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = _nll(heads.logp(net(X[idx])), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xval is not None:
            net.eval()
            with torch.no_grad():
                hist.append(_nll(torch.log(_mix_t(heads.logp(net(Xval)).exp(), heads)), Yval).item())
    net.eval()

    def f(Xn):
        with torch.no_grad():
            return heads.logp(net(Xn)).exp()
    return f, hist


def _mix_t(P, heads):
    """Grader floor mixing on padded probs (n,J,K)."""
    valid = (heads.mask == 0).float()
    P = P * valid
    P = P / P.sum(-1, keepdim=True).clamp(min=1e-30)
    k = torch.tensor(heads.ks, device=DEV, dtype=P.dtype).view(1, -1, 1)
    return (FLOOR + (1 - k * FLOOR) * P) * valid + (1 - valid)



# ------------------------------------------------------------------------------------------------- LightGBM
try:
    import lightgbm as lgb
    LGB = True
except Exception:  # pragma: no cover
    LGB = False

GBM_PARAMS = dict(objective="multiclass", learning_rate=0.08, num_leaves=15, min_child_samples=25, reg_lambda=1.0,
                  subsample=0.85, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8, max_cat_to_onehot=8,
                  cat_smooth=10, min_data_per_group=20)


def _gbm_one(Xc, y, k, Xp, rounds):
    m = y >= 0
    yy = y[m]
    if len(yy) < 30 or len(np.unique(yy)) < 2:
        c = np.bincount(yy, minlength=k)[:k] + 0.5 if len(yy) else np.ones(k)
        return np.repeat((c / c.sum())[None], len(Xp), 0)
    present = np.unique(yy)
    remap = {c: i for i, c in enumerate(present)}
    clf = lgb.LGBMClassifier(num_class=len(present), n_estimators=rounds, **GBM_PARAMS) if len(present) > 2 else         lgb.LGBMClassifier(n_estimators=rounds, **{**GBM_PARAMS, "objective": "binary"})
    clf.fit(Xc[m], np.array([remap[v] for v in yy]), categorical_feature=list(range(Xc.shape[1])))
    pp = clf.predict_proba(Xp)
    out = np.full((len(Xp), k), 0.0)
    out[:, present] = pp
    # options never seen in training get a small share, as the marginal would
    unseen = np.setdiff1d(np.arange(k), present)
    if len(unseen):
        out[:, unseen] = 0.5 / (len(yy) + 0.5 * k)
        out /= out.sum(1, keepdims=True)
    return out


def _gbm_order(Yv, ks):
    """Items most worth a tree model first: observed count x marginal entropy."""
    pri = []
    for j, k in enumerate(ks):
        y = Yv[:, j]
        y = y[y >= 0]
        if len(y) == 0:
            pri.append(0.0); continue
        p = np.bincount(y, minlength=k)[:k] / len(y)
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        pri.append(h * len(y))
    return list(np.argsort(pri)[::-1])



# ------------------------------------------------------------------------------------------- profile kNN
def _onehot_y(Yv, heads):
    n = len(Yv)
    oh = torch.zeros(n, heads.J, heads.Kmax, device=DEV)
    m = torch.tensor(Yv >= 0, device=DEV)
    idx = torch.tensor(np.clip(Yv, 0, None), device=DEV)
    oh.scatter_(2, idx.unsqueeze(-1), m.unsqueeze(-1).float())
    return oh.view(n, -1)


def _hamming(Ca, Cb):
    """(len(Ca), len(Cb)) number of GIVEN features that differ, computed in row chunks."""
    A = torch.tensor(Ca, device=DEV)
    B = torch.tensor(Cb, device=DEV)
    out = torch.empty(len(A), len(B), dtype=torch.int16, device=DEV)
    step = max(1, int(2e8 // max(1, len(B) * A.shape[1])))
    for i in range(0, len(A), step):
        out[i:i + step] = (A[i:i + step, None, :] != B[None, :, :]).sum(-1).to(torch.int16)
    return out


def _knn_counts(D, Yoh, c1, c2, mask=None):
    """Neighbour-weighted answer counts: weight 1 at distance 0, c1 at distance 1, c2 at distance 2."""
    W = (D == 0).float() + c1 * (D == 1).float() + c2 * (D == 2).float()
    if mask is not None:
        W = W * mask
    out = torch.empty(D.shape[0], Yoh.shape[1], device=DEV)
    step = 2048
    for i in range(0, D.shape[0], step):
        out[i:i + step] = W[i:i + step] @ Yoh
    return out


def _knn_post(N, base, heads, alpha):
    N = N.view(-1, heads.J, heads.Kmax)
    tot = N.sum(-1, keepdim=True)
    P = (N + alpha * base) / (tot + alpha)
    return P


# ----------------------------------------------------------------------------------------------- count models
def _marginal(Y, heads):
    P = np.zeros((heads.J, heads.Kmax), np.float64)
    for j, k in enumerate(heads.ks):
        y = Y[:, j]
        c = np.bincount(y[y >= 0], minlength=k)[:k] + 0.5
        P[j, :k] = c / c.sum()
    return P


def _naive_bayes(codes_tr, Ytr, codes_te, heads, given, alpha=1.0):
    """log p(y) + sum_f log p(x_f | y), returned as per-feature log-likelihood stacks so a temperature can be applied."""
    J, K = heads.J, heads.Kmax
    n_te = len(codes_te[given[0]]) if given else 0
    prior = np.log(_marginal(Ytr, heads) + 1e-12)
    L = np.zeros((n_te, J, K), np.float64)
    for f in given:
        xt, xe = codes_tr[f], codes_te[f]
        kf = int(max(xt.max(), xe.max())) + 2
        xt = np.where(xt >= 0, xt, kf - 1)
        xe = np.where(xe >= 0, xe, kf - 1)
        for j in range(J):
            y = Ytr[:, j]
            m = y >= 0
            tab = np.zeros((heads.ks[j], kf)) + alpha
            np.add.at(tab, (y[m], xt[m]), 1.0)
            tab = np.log(tab / tab.sum(1, keepdims=True))          # log p(x_f | y)
            L[:, j, :heads.ks[j]] += tab[:, xe].T
    return prior, L


def _nb_probs(prior, L, heads, temp):
    z = prior[None] + temp * L
    out = np.zeros_like(z)
    for j, k in enumerate(heads.ks):
        zz = z[:, j, :k]
        zz = zz - zz.max(1, keepdims=True)
        e = np.exp(zz)
        out[:, j, :k] = e / e.sum(1, keepdims=True)
    return out


def _cell_logscore(P, Y, ks):
    """Per-item mean log score (grader mixing). P (n,J,K) numpy, Y (n,J)."""
    n, J, K = P.shape
    res = np.zeros(J)
    cnt = np.zeros(J)
    for j, k in enumerate(ks):
        y = Y[:, j]
        m = y >= 0
        if not m.any():
            continue
        p = P[m, j, :k]
        p = p / p.sum(1, keepdims=True)
        p = FLOOR + (1 - k * FLOOR) * p
        res[j] = np.log(p[np.arange(m.sum()), y[m]]).sum()
        cnt[j] = m.sum()
    return res, cnt


# ------------------------------------------------------------------------------------------------- stacking
def _stack_weights(oof, Y, ks, prior_strength=10.0, iters=60):
    """Per-item convex weights over M models by EM on out-of-fold probabilities, shrunk to global weights."""
    M = len(oof)
    n, J, K = oof[0].shape
    # per-model probability of the observed answer, mixed like the grader
    q = np.zeros((M, n, J))
    obs = Y >= 0
    for mi, P in enumerate(oof):
        for j, k in enumerate(ks):
            p = P[:, j, :k] / P[:, j, :k].sum(1, keepdims=True)
            p = FLOOR + (1 - k * FLOOR) * p
            yy = np.clip(Y[:, j], 0, None)
            q[mi, :, j] = p[np.arange(n), yy]
    q = np.where(obs[None], q, 1.0)
    wg = np.full(M, 1.0 / M)
    for _ in range(iters):                       # global weights
        r = wg[:, None, None] * q
        r = r / r.sum(0, keepdims=True)
        wg = (r * obs[None]).sum((1, 2)) / obs.sum()
    W = np.tile(wg[:, None], (1, J))
    for _ in range(iters):                       # per-item weights with Dirichlet(prior_strength * wg) prior
        r = W[:, None, :] * q
        r = r / r.sum(0, keepdims=True)
        num = (r * obs[None]).sum(1) + prior_strength * wg[:, None]
        W = num / num.sum(0, keepdims=True)
    return wg, W


def _blend(preds, W):
    out = np.zeros_like(preds[0])
    for mi, P in enumerate(preds):
        out += W[mi][None, :, None] * P
    return out


def _temper(P, ks, t):
    out = np.zeros_like(P)
    for j, k in enumerate(ks):
        z = np.log(np.clip(P[:, j, :k], 1e-12, None)) * t[j]
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        out[:, j, :k] = e / e.sum(1, keepdims=True)
    return out


def _fit_temps(P, Y, ks, grid=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)):
    best = np.ones(len(ks))
    bests = np.full(len(ks), -np.inf)
    for t in grid:
        s, c = _cell_logscore(_temper(P, ks, np.full(len(ks), t)), Y, ks)
        better = s > bests
        best[better], bests[better] = t, s[better]
    return best



# ------------------------------------------------------------------------------------------- gate factorisation
def _gate_rates(schema, pred, opts, Yv):
    """For PREDICT items gated on a PREDICT parent: r[v] = P(child is the sentinel | parent answered v), from the
    visible rows (smoothed towards the child's overall sentinel rate)."""
    idx = {n: j for j, n in enumerate(pred)}
    out = {}
    for j, n in enumerate(pred):
        g = schema["items"][n].get("gate")
        if not g or g.get("parent") not in idx:
            continue
        pj = idx[g["parent"]]
        kp, kc = len(opts[g["parent"]]), len(opts[n])
        yc, yp = Yv[:, j], Yv[:, pj]
        m = (yc >= 0) & (yp >= 0)
        if m.sum() < 20:
            continue
        sent = (yc[m] == kc - 1).astype(float)
        glob = (sent.sum() + 0.5) / (len(sent) + 1.0)
        r = np.zeros(kp)
        for v in range(kp):
            mv = m & (yp == v)
            r[v] = ((yc[mv] == kc - 1).sum() + 2.0 * glob) / (mv.sum() + 2.0)
        out[j] = (pj, r, kc)
    return out


def _factor_one(F, P, j, pj, r, kc):
    kp = len(r)
    ps = np.clip(F[:, pj, :kp] @ r, 1e-6, 1 - 1e-6)
    cond = P[:, j, :kc - 1]
    cond = cond / np.clip(cond.sum(1, keepdims=True), 1e-12, None)
    return np.concatenate([(1 - ps)[:, None] * cond, ps[:, None]], 1)


def _apply_factor(P, rates, weights):
    """Children in schema order (parents first): P(sentinel) from the (already factorised) parent distribution,
    the answer distribution given 'asked' from the child's own model; mixed with the direct estimate by weight."""
    F = P.copy()
    for j in sorted(rates):
        w = weights.get(j, 0.0)
        if w > 0:
            pj, r, kc = rates[j]
            F[:, j, :kc] = w * _factor_one(F, P, j, pj, r, kc) + (1 - w) * P[:, j, :kc]
    return F


def _fit_factor_weights(P, Y, ks, rates, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    F = P.copy()
    weights = {}
    for j in sorted(rates):
        pj, r, kc = rates[j]
        fac = _factor_one(F, P, j, pj, r, kc)
        best = None
        for w in grid:
            mix = w * fac + (1 - w) * P[:, j, :kc]
            tmp = np.zeros((len(P), 1, P.shape[2])); tmp[:, 0, :kc] = mix
            s, _ = _cell_logscore(tmp, Y[:, [j]], ks[[j]])
            if best is None or s[0] > best[0] + 1e-9:
                best = (s[0], w, mix)
        weights[j] = best[1]
        F[:, j, :kc] = best[2]
    return weights



def _pair_features(codes_a, codes_b_list, given, min_count=10, max_cols=6000):
    """One-hot of pairwise GIVEN interactions (columns seen >= min_count times in the visible rows).
    codes_a: dict of visible codes (defines the columns); returns a builder for any codes dict."""
    cols = []
    for i in range(len(given)):
        for j in range(i + 1, len(given)):
            a, b = codes_a[given[i]], codes_a[given[j]]
            kb = int(max(b.max(), 0)) + 2
            key = (a + 1) * kb + (b + 1)
            vals, cnt = np.unique(key, return_counts=True)
            for v in vals[cnt >= min_count]:
                cols.append((i, j, kb, int(v)))
    if len(cols) > max_cols:
        cols = cols[:max_cols]

    def build(codes):
        n = len(codes[given[0]])
        X = np.zeros((n, len(cols)), np.float32)
        cache = {}
        for c, (i, j, kb, v) in enumerate(cols):
            if (i, j) not in cache:
                cache[(i, j)] = (codes[given[i]] + 1) * kb + (codes[given[j]] + 1)
            X[:, c] = cache[(i, j)] == v
        return X
    return build


# --------------------------------------------------------------------------------------------------- pipeline
def _models_cv(Xv, Yv, codes_v, heads, given, folds, t_end):
    """Out-of-fold probabilities for each candidate model; returns (names, oof list, config dict)."""
    n = len(Yv)
    rng = np.random.RandomState(SEED)
    fold = rng.permutation(n) % folds
    Xt = torch.tensor(Xv, device=DEV)
    Yt = torch.tensor(Yv, device=DEV)
    cfg = {}
    names, oof = [], []

    # logistic regression: OOF for every weight decay, then the best decay chosen PER ITEM
    WDS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    t_lr0 = time.time()
    Ps = {}
    for wd in WDS:
        P = np.zeros((n, heads.J, heads.Kmax))
        for k in range(folds):
            tr, va = fold != k, fold == k
            P[va] = _fit_linear(Xt[tr], Yt[tr], heads, wd)(Xt[va]).cpu().numpy()
        Ps[wd] = P
    sc = np.stack([_cell_logscore(Ps[wd], Yv, heads.ks)[0] for wd in WDS])      # (n_wd, J)
    pick = sc.argmax(0)
    cfg["lr_wd_item"] = [WDS[i] for i in pick]
    cfg["t_lr"] = time.time() - t_lr0
    cfg["lr_wd"] = WDS[int(np.bincount(pick, minlength=len(WDS)).argmax())]
    P = np.zeros((n, heads.J, heads.Kmax))
    for j in range(heads.J):
        P[:, j] = Ps[WDS[pick[j]]][:, j]
    names.append("lr"); oof.append(P)



    # MLP: epochs chosen on fold 0 by validation curve, then OOF
    if time.time() < t_end:
        best = None
        for hidden, wd, drop in ((256, 1e-4, 0.2), (512, 1e-3, 0.3)):
            tr, va = fold != 0, fold == 0
            _, hist = _fit_mlp(Xt[tr], Yt[tr], heads, hidden, wd, 60, drop, Xval=Xt[va], Yval=Yt[va])
            e = int(np.argmin(hist)) + 1
            if best is None or min(hist) < best[0]:
                best = (min(hist), hidden, wd, drop, e)
        cfg["mlp"] = best[1:]
        P = np.zeros((n, heads.J, heads.Kmax))
        for k in range(folds):
            tr, va = fold != k, fold == k
            t_fold = time.time()
            for s in range(cfg.setdefault("mlp_seeds", 1)):
                f, _ = _fit_mlp(Xt[tr], Yt[tr], heads, cfg["mlp"][0], cfg["mlp"][1], cfg["mlp"][3], cfg["mlp"][2], seed=SEED + s)
                P[va] += f(Xt[va]).cpu().numpy() / cfg["mlp_seeds"]
            if k == 0 and cfg["mlp_seeds"] == 1:
                # bag more seeds when the allowance easily covers it (cheap on the H100)
                per = time.time() - t_fold
                cfg["mlp_seeds"] = 3 if time.time() + per * 3 * (folds + 1) < t_end else 1
                if cfg["mlp_seeds"] > 1:
                    P[va] = 0.0
                    for s in range(cfg["mlp_seeds"]):
                        f, _ = _fit_mlp(Xt[tr], Yt[tr], heads, cfg["mlp"][0], cfg["mlp"][1], cfg["mlp"][3], cfg["mlp"][2], seed=SEED + s)
                        P[va] += f(Xt[va]).cpu().numpy() / cfg["mlp_seeds"]
        names.append("mlp"); oof.append(P)

    # naive Bayes with a per-instrument temperature
    if given and time.time() < t_end:
        P = {t: np.zeros((n, heads.J, heads.Kmax)) for t in (0.3, 0.5, 0.7, 1.0)}
        for k in range(folds):
            tr, va = fold != k, fold == k
            ctr = {g: codes_v[g][tr] for g in given}
            cva = {g: codes_v[g][va] for g in given}
            prior, L = _naive_bayes(ctr, Yv[tr], cva, heads, given)
            for t in P:
                P[t][va] = _nb_probs(prior, L, heads, t)
        sc = {t: _cell_logscore(P[t], Yv, heads.ks)[0].sum() for t in P}
        cfg["nb_t"] = max(sc, key=sc.get)
        names.append("nb"); oof.append(P[cfg["nb_t"]])

    # LightGBM per item (3 groups of the same folds), as many items as the time allowance permits
    if LGB and given and time.time() < t_end:
        Xc = np.stack([codes_v[g] for g in given], 1).astype(np.int32)
        grp = fold % 3
        P = oof[names.index("lr")].copy()
        done_items = []
        t_item = None
        for j in _gbm_order(Yv, heads.ks):
            if t_item is not None and time.time() + 1.3 * t_item > t_end:
                break
            ti = time.time()
            k = int(heads.ks[j])
            for g in range(3):
                tr, va = grp != g, grp == g
                P[va, j, :] = 0.0
                P[va, j, :k] = _gbm_one(Xc[tr], Yv[tr, j], k, Xc[va], cfg.get("gbm_rounds", 120))
            done_items.append(int(j))
            dt = time.time() - ti
            t_item = dt if t_item is None else 0.8 * t_item + 0.2 * dt
        cfg["gbm_items"] = done_items
        if done_items:
            names.append("gbm"); oof.append(P)

    # profile kNN on the GIVEN block, shrunk to the logistic prediction
    if given and time.time() < t_end:
        try:
            C = np.stack([codes_v[g] for g in given], 1).astype(np.int64)
            D = _hamming(C, C)
            ft = torch.tensor(fold, device=DEV)
            mask = (ft[:, None] != ft[None, :]).float()
            Yoh = _onehot_y(Yv, heads)
            base = torch.tensor(oof[names.index("lr")], device=DEV, dtype=torch.float32)
            best = None
            for c1, c2 in ((0.0, 0.0), (0.2, 0.0), (0.5, 0.05), (1.0, 0.2)):
                N = _knn_counts(D, Yoh, c1, c2, mask)
                for alpha in (1.0, 3.0, 10.0, 30.0):
                    P = _knn_post(N, base, heads, alpha)
                    s = _nll(torch.log(_mix_t(P, heads)), torch.tensor(Yv, device=DEV)).item()
                    if best is None or s < best[0]:
                        best = (s, c1, c2, alpha)
            cfg["knn"] = best[1:]
            N = _knn_counts(D, Yoh, best[1], best[2], mask)
            names.append("knn"); oof.append(_knn_post(N, base, heads, best[3]).cpu().numpy())
            del D, mask
        except Exception as e:
            cfg["knn_error"] = repr(e)[:120]

    # logistic regression on one-hot + pairwise interactions, per-item decay (strong decays only)
    if given and len(given) > 1 and time.time() < t_end:
        try:
            build = _pair_features(codes_v, None, given)
            X2 = torch.tensor(np.concatenate([Xv, build(codes_v)], 1), device=DEV)
            WD2 = (3e-3, 1e-2)
            est = cfg["t_lr"] / 6 * len(WD2) * X2.shape[1] / max(1, Xv.shape[1])
            if time.time() + 1.5 * est > t_end:
                raise RuntimeError("no time for lr2 (est %.0fs)" % est)
            Ps = {}
            for wd in WD2:
                P = np.zeros((n, heads.J, heads.Kmax))
                for k in range(folds):
                    tr, va = fold != k, fold == k
                    P[va] = _fit_linear(X2[tr], Yt[tr], heads, wd)(X2[va]).cpu().numpy()
                Ps[wd] = P
                if time.time() > t_end:
                    break
            WD2 = tuple(Ps)
            sc = np.stack([_cell_logscore(Ps[wd], Yv, heads.ks)[0] for wd in WD2])
            pick = sc.argmax(0)
            cfg["lr2_wd_item"] = [WD2[i] for i in pick]
            P = np.zeros((n, heads.J, heads.Kmax))
            for j in range(heads.J):
                P[:, j] = Ps[WD2[pick[j]]][:, j]
            names.append("lr2"); oof.append(P)
            del X2
        except Exception as e:
            cfg["lr2_error"] = repr(e)[:120]

    # marginal (fold-wise)
    P = np.zeros((n, heads.J, heads.Kmax))
    for k in range(folds):
        tr, va = fold != k, fold == k
        P[va] = _marginal(Yv[tr], heads)[None]
    names.append("marg"); oof.append(P)
    return names, oof, cfg


def _fit_full(name, cfg, Xv, Yv, Xh, codes_v, codes_h, heads, given):
    Xt, Yt, Xht = (torch.tensor(Xv, device=DEV), torch.tensor(Yv, device=DEV), torch.tensor(Xh, device=DEV))
    if name == "lr":
        out = None
        for wd in sorted(set(cfg["lr_wd_item"])):
            Pw = _fit_linear(Xt, Yt, heads, wd)(Xht).cpu().numpy()
            if out is None:
                out = Pw.copy()
            for j, w in enumerate(cfg["lr_wd_item"]):
                if w == wd:
                    out[:, j] = Pw[:, j]
        return out
    if name == "lr2":
        build = _pair_features(codes_v, None, given)
        X2 = torch.tensor(np.concatenate([Xv, build(codes_v)], 1), device=DEV)
        X2h = torch.tensor(np.concatenate([Xh, build(codes_h)], 1), device=DEV)
        out = None
        for wd in sorted(set(cfg["lr2_wd_item"])):
            Pw = _fit_linear(X2, Yt, heads, wd)(X2h).cpu().numpy()
            if out is None:
                out = Pw.copy()
            for j, w in enumerate(cfg["lr2_wd_item"]):
                if w == wd:
                    out[:, j] = Pw[:, j]
        return out
    if name == "mlp":
        hidden, wd, drop, e = cfg["mlp"]
        out = 0.0
        for s in range(cfg.get("mlp_seeds", 1)):
            f, _ = _fit_mlp(Xt, Yt, heads, hidden, wd, max(1, int(round(e * 1.1))), drop, seed=SEED + s)
            out = out + f(Xht).cpu().numpy() / cfg.get("mlp_seeds", 1)
        return out
    if name == "nb":
        prior, L = _naive_bayes(codes_v, Yv, codes_h, heads, given)
        return _nb_probs(prior, L, heads, cfg["nb_t"])
    if name == "gbm":
        P = _fit_linear(Xt, Yt, heads, cfg["lr_wd"])(Xht).cpu().numpy()
        Xc = np.stack([codes_v[g] for g in given], 1).astype(np.int32)
        Xch = np.stack([codes_h[g] for g in given], 1).astype(np.int32)
        for j in cfg["gbm_items"]:
            k = int(heads.ks[j])
            P[:, j, :] = 0.0
            P[:, j, :k] = _gbm_one(Xc, Yv[:, j], k, Xch, cfg.get("gbm_rounds", 120))
        return P
    if name == "knn":
        c1, c2, alpha = cfg["knn"]
        C = np.stack([codes_v[g] for g in given], 1).astype(np.int64)
        Ch = np.stack([codes_h[g] for g in given], 1).astype(np.int64)
        D = _hamming(Ch, C)
        base = _fit_linear(Xt, Yt, heads, cfg["lr_wd"])(Xht)
        N = _knn_counts(D, _onehot_y(Yv, heads), c1, c2)
        return _knn_post(N, base, heads, alpha).cpu().numpy()
    if name == "marg":
        return np.repeat(_marginal(Yv, heads)[None], len(Xh), 0)
    raise KeyError(name)


def _gate_rules(schema, frame, codes, pred, given, visible):
    """For PREDICT items gated on a GIVEN parent: empirical P(sentinel | parent level) from the visible rows,
    used only where it is (near) deterministic."""
    rules = {}
    for name in pred:
        g = schema["items"][name].get("gate")
        if not g or g.get("parent") not in given:
            continue
        par = codes[g["parent"]]
        y = codes[name]
        k = len(_options(schema, name))
        sent = k - 1
        tab = {}
        for lv in np.unique(par[visible]):
            m = visible & (par == lv) & (y >= 0)
            if m.sum() >= 10:
                rate = float((y[m] == sent).mean())
                if rate <= 0.0 or rate >= 1.0:
                    tab[int(lv)] = rate
        rules[name] = (g["parent"], tab)
    return rules


def predict(frame, schema):
    t_start = time.time()
    items = schema["items"]
    keys = [n for n, r in items.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in keys if items[n]["class"] == "GIVEN"]
    pred = [n for n in keys if items[n]["class"] == "PREDICT"]
    opts = {n: _options(schema, n) for n in keys}

    values = frame[keys].to_numpy(dtype=object)
    blank = pd.isna(frame[keys]).to_numpy()
    cells = [(r, c) for r in range(values.shape[0]) for c in range(len(keys)) if blank[r, c]]

    # crowd marginal for every shipped item: the fallback for everything
    marg = {}
    for n in keys:
        counts = frame[n].value_counts()
        w = np.array([counts.get(o, 0) + 0.5 for o in opts[n]], float)
        marg[n] = w / w.sum()
    result = {(r, c): marg[keys[c]] for r, c in cells}

    try:
        if not pred or not TORCH:
            raise RuntimeError("nothing to model")
        codes = _encode(frame, schema, given, pred)
        Yall = np.stack([codes[n] for n in pred], 1)
        pcols = [keys.index(n) for n in pred]
        visible = ~blank[:, pcols].any(1)
        held = blank[:, pcols].any(1)
        X = _features(codes, schema, given).astype(np.float32)
        heads = _Heads([len(opts[n]) for n in pred])
        Xv, Yv = X[visible], Yall[visible]
        Xh = X[held]
        cv_v = {g: codes[g][visible] for g in given}
        cv_h = {g: codes[g][held] for g in given}
        allow = _allowance(schema, int(visible.sum()), len(pred))
        t_end = t_start + 0.6 * allow
        folds = 5
        names, oof, cfg = _models_cv(Xv, Yv, cv_v, heads, given, folds, t_end)
        wg, W = _stack_weights(oof, Yv, heads.ks)
        blend_oof = _blend(oof, W)
        temps = _fit_temps(blend_oof, Yv, heads.ks)
        full = [_fit_full(nm, cfg, Xv, Yv, Xh, cv_v, cv_h, heads, given) for nm in names]
        P = _temper(_blend(full, W), heads.ks, temps)
        fw = {}
        try:
            rates = _gate_rates(schema, pred, opts, Yv)
            if rates and time.time() < t_start + 0.9 * allow:
                fw = _fit_factor_weights(_temper(blend_oof, heads.ks, temps), Yv, heads.ks, rates)
                P = _apply_factor(P, rates, fw)
        except Exception as e:
            cfg["factor_error"] = repr(e)[:120]
        rules = _gate_rules(schema, frame, codes, pred, given, visible)
        hrows = np.flatnonzero(held)
        rowpos = {r: i for i, r in enumerate(hrows)}
        for j, n in enumerate(pred):
            k = len(opts[n])
            c = keys.index(n)
            rule = rules.get(n)
            for r in hrows:
                if not blank[r, c]:
                    continue
                v = P[rowpos[r], j, :k].copy()
                if rule:
                    lv = int(codes[rule[0]][r])
                    if lv in rule[1]:
                        rate = rule[1][lv]
                        if rate >= 1.0:
                            v = np.zeros(k); v[k - 1] = 1.0
                        else:
                            v[k - 1] = 0.0
                            if v.sum() <= 0:
                                v = marg[n].copy(); v[k - 1] = 0.0
                if np.all(np.isfinite(v)) and v.sum() > 0:
                    result[(r, c)] = v
        _DONE_J.append(len(pred))
        predict.last = dict(factor_w=round(float(np.mean(list(fw.values()))) if fw else -1, 3), n_factor=len(fw), allow=round(allow), gbm_items=len(cfg.get("gbm_items", [])), names=names, global_weights=wg.round(3).tolist(), cfg=cfg,
                            temps_mean=float(temps.mean()), seconds=round(time.time() - t_start, 1))
    except Exception as e:  # never fail the submission
        predict.last = dict(error=repr(e)[:200])

    out = []
    for r, c in cells:
        v = np.asarray(result[(r, c)], float)
        v = np.where(np.isfinite(v) & (v >= 0), v, 0.0)
        if v.sum() <= 0:
            v = marg[keys[c]]
        out.append([float(x) for x in v])
    return out
