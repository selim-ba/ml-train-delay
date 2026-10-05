"""Graph transformer on per-point subgraphs (research experiment, one horizon).

Each prediction point is a small graph (``swissdelay.features.graph``): the stations from
the current one to the target plus their 1-hop neighbours, with window features on nodes
and segments. A [CLS] token carries the train's own tabular features (default: the
champion set ``full_network_plus``) and is read out to predict Δd with an L1 loss.

Architecture (Graphormer-style, pre-LayerNorm):

- node token = linear(node features) + station embedding; [CLS] = MLP(tabular features
  + categorical embeddings);
- attention scores get additive biases per head: one per shortest-path distance between
  nodes (hops in the undirected subgraph), one from the features of each directed edge
  (separately for the two directions), and a learned bias between [CLS] and the nodes;
  padded nodes are masked;
- ``layers`` blocks of multi-head attention + feed-forward, then an MLP head on [CLS].

Variants (same code, same data):

- ``graph``: as above;
- ``nograph``: [CLS] token only (an MLP-like transformer on the tabular features), the
  control that separates the gain of the graph from the gain of a neural model;
- ``random``: same nodes and node features, but every edge is rewired between random
  nodes of the same subgraph (and distances follow): tests whether the real topology
  matters.

Data use as for XGBoost: fit on Aug 2025 – Mar 2026, early stopping on April 2026,
evaluation on May 2026 (``--test``: June, only at the end). ``--refit``: like XGBoost's
refit, train on Aug 2025 – Apr 2026 for a fixed ``--epochs`` (15, the schedule chosen with
early stopping) and keep the last epoch; outputs are named ``gt_<variant>_refit``.

Usage::

    uv sync --group dev --group dl
    uv run python -m swissdelay.models.graph_transformer --variant graph --sample 0.1 --epochs 2
    uv run python -m swissdelay.models.graph_transformer --variant graph
    uv run python -m swissdelay.models.graph_transformer --variant nograph
    uv run python -m swissdelay.models.graph_transformer --variant random
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from swissdelay import config
from swissdelay.features import graph as gr
from swissdelay.features.build import CATEGORICAL_FEATURES
from swissdelay.models import tabular as tb

log = logging.getLogger("swissdelay.gt")

KEYS = ["trip_key", "stop_seq", "horizon_min"]
FIT_MONTHS = ["2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02",
              "2026-03"]  # fmt: skip
TUNE_MONTHS = ["2026-04"]
VALID_MONTHS = ["2026-05"]
TEST_MONTHS = ["2026-06"]
VARIANTS = ("graph", "nograph", "random")
MAX_SPD = 6  # distances above are grouped with "unreachable"
CLIP = 10.0  # standardised inputs are clipped to ±CLIP


@dataclass
class TrainConfig:
    variant: str = "graph"
    horizon: int = 15
    feature_set: str = "full_network_plus"
    d_model: int = 64
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1
    batch_size: int = 1024
    block: int = 128  # contiguous points read together from disk (batch = several blocks)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 2
    sample: float = 1.0  # share of fitting blocks used
    seed: int = 0
    refit: bool = False  # train on fit + tune months for ``epochs`` epochs, no early stopping
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- data


class Graphs:
    """Memory-mapped subgraphs of several months, addressed by a global row index."""

    def __init__(self, months: list[str], horizon: int):
        self.parts = [gr.load(gr.GRAPH_DIR / f"h{horizon}" / m) for m in months]
        sizes = [len(p["points"]) for p in self.parts]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.points = pd.concat([p["points"] for p in self.parts], ignore_index=True)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def gather(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays for global ``rows`` (any order; read month by month in sorted order)."""
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        month = np.searchsorted(self.offsets, sorted_rows, side="right") - 1
        out = {name: [] for name in gr.ARRAYS}
        for m in np.unique(month):
            local = sorted_rows[month == m] - self.offsets[m]
            for name in gr.ARRAYS:
                out[name].append(np.asarray(self.parts[m][name][local]))
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return {name: np.concatenate(v)[inverse] for name, v in out.items()}


@dataclass
class Tabular:
    """Preprocessed [CLS] inputs: standardised numbers and categorical codes."""

    num: np.ndarray  # float32 (P, F)
    cat: np.ndarray  # int64 (P, C)
    y: np.ndarray  # float32 (P,)
    pred_historical: np.ndarray


class TabularPrep:
    """Imputation (fit median + missing indicators), standardisation and category codes,
    all fitted on the fitting months only."""

    def __init__(self, cols: list[str]):
        self.num_cols = [c for c in cols if c not in CATEGORICAL_FEATURES]
        self.cat_cols = [c for c in cols if c in CATEGORICAL_FEATURES]

    def fit(self, df: pd.DataFrame) -> TabularPrep:
        x = df[self.num_cols].astype("float32")
        self.median = x.median().fillna(0)
        self.indicator_cols = [c for c in self.num_cols if x[c].isna().any()]
        filled = x.fillna(self.median)
        self.mean, self.std = filled.mean(), filled.std().replace(0, 1).fillna(1)
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].dropna().unique()))}
                      for c in self.cat_cols}  # fmt: skip
        return self

    @property
    def n_num(self) -> int:
        return len(self.num_cols) + len(self.indicator_cols)

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.cat_cols]

    def transform(self, df: pd.DataFrame) -> Tabular:
        x = df[self.num_cols].astype("float32")
        ind = x[self.indicator_cols].isna().astype("float32").to_numpy()
        z = ((x.fillna(self.median) - self.mean) / self.std).clip(-CLIP, CLIP).to_numpy()
        num = np.concatenate([z, ind], axis=1).astype(np.float32)
        cat = np.stack([df[c].map(self.vocab[c]).fillna(0).astype("int64").to_numpy()
                        for c in self.cat_cols], axis=1) if self.cat_cols else \
            np.zeros((len(df), 0), np.int64)  # fmt: skip
        return Tabular(num, cat, df["delta_min"].to_numpy(np.float32),
                       df["pred_historical"].to_numpy(np.float32))  # fmt: skip


def load_tabular(
    con: duckdb.DuckDBPyConnection, where: str, cols: list[str], points: pd.DataFrame
) -> pd.DataFrame:
    """Tabular rows in the order of the graph ``points`` (same keys, same order)."""
    need = [*KEYS, "delta_min", "pred_historical", *cols]
    df = tb.load(con, where, need)
    out = points[KEYS].merge(df, on=KEYS, how="left", validate="one_to_one")
    missing = out["delta_min"].isna().sum()
    if missing:
        raise ValueError(f"{missing} graph points without tabular features")
    return out


def feature_stats(graphs: Graphs, n: int = 200_000, seed: int = 0) -> dict[str, np.ndarray]:
    """Mean and std of node and edge features over real (non-padded) rows of a sample."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(graphs), size=min(n, len(graphs)), replace=False))
    g = graphs.gather(rows)
    node_mask = np.arange(gr.N_MAX)[None, :] < g["node_n"][:, None]
    edge_mask = np.arange(gr.E_MAX)[None, :] < g["edge_n"][:, None]
    nodes = g["node_x"][node_mask].astype(np.float32)
    edges = g["edge_x"][edge_mask].astype(np.float32)
    return dict(node_mean=nodes.mean(0), node_std=nodes.std(0) + 1e-6,
                edge_mean=edges.mean(0), edge_std=edges.std(0) + 1e-6)  # fmt: skip


def to_batch(g: dict[str, np.ndarray], tab: Tabular, rows: np.ndarray, device) -> dict:

    def t(a, dtype):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

    node_mask = np.arange(gr.N_MAX)[None, :] < g["node_n"][:, None]
    edge_mask = np.arange(gr.E_MAX)[None, :] < g["edge_n"][:, None]
    return dict(
        node_x=t(g["node_x"], torch.float32), node_station=t(g["node_station"], torch.long),
        node_mask=t(node_mask, torch.bool), edge_x=t(g["edge_x"], torch.float32),
        edge_index=t(g["edge_index"], torch.long), edge_mask=t(edge_mask, torch.bool),
        num=t(tab.num[rows], torch.float32), cat=t(tab.cat[rows], torch.long),
        y=t(tab.y[rows], torch.float32),
    )  # fmt: skip


def batches(n: int, batch_size: int, block: int, shuffle: bool, rng=None, sample: float = 1.0):
    """Row indices per batch. Shuffled mode: random blocks of ``block`` contiguous rows
    (fast reads from the memory-mapped files), ``batch_size // block`` blocks per batch."""
    if not shuffle:
        for start in range(0, n, batch_size):
            yield np.arange(start, min(start + batch_size, n))
        return
    starts = np.arange(0, n, block)
    starts = rng.permutation(starts)[: max(1, int(len(starts) * sample))]
    per = max(1, batch_size // block)
    for i in range(0, len(starts), per):
        yield np.concatenate([np.arange(s, min(s + block, n)) for s in starts[i : i + per]])


# --------------------------------------------------------------------------- model


def shortest_paths(edge_index: torch.Tensor, edge_mask: torch.Tensor, n: int) -> torch.Tensor:
    """Hop distances in the undirected subgraph, (B, n, n), MAX_SPD + 1 = farther or
    unreachable. Breadth-first search with boolean matrix products."""
    b = edge_index.shape[0]
    adj = torch.zeros(b, n * n, device=edge_index.device)
    src, dst = edge_index[..., 0].clamp(min=0), edge_index[..., 1].clamp(min=0)
    adj.scatter_add_(1, src * n + dst, edge_mask.float())
    adj = adj.view(b, n, n)
    adj = ((adj + adj.transpose(1, 2)) > 0).float()
    eye = torch.eye(n, device=adj.device).expand(b, n, n)
    dist = torch.full((b, n, n), MAX_SPD + 1, device=adj.device, dtype=torch.long)
    dist[eye.bool()] = 0
    reach = eye.clone()
    for k in range(1, MAX_SPD + 1):
        new = ((reach @ adj + reach) > 0).float()
        dist[(new > 0) & (reach == 0)] = k
        reach = new
    return dist


def random_edges(edge_index: torch.Tensor, edge_mask: torch.Tensor,
                 node_n: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """Rewire every real edge between two random nodes of the same subgraph."""
    n = node_n.clamp(min=1).float()[:, None]
    r = torch.rand(edge_index.shape, device=edge_index.device)
    out = (r * n[..., None]).long()
    return torch.where(edge_mask[..., None], out, edge_index)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * d, d))  # fmt: skip
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        b, n, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(b, n, 3, self.heads, self.dk).unbind(2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # (B, H, n, dk)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if bias is not None:
            scores = scores + bias
        att = self.drop(scores.softmax(-1))
        h = h + self.drop(self.out((att @ v).transpose(1, 2).reshape(b, n, d)))
        return h + self.drop(self.ff(self.ln2(h)))


class GraphTransformer(nn.Module):
    def __init__(self, cfg: TrainConfig, n_num: int, cardinalities: list[int], n_stations: int,
                 stats: dict[str, np.ndarray]):  # fmt: skip
        super().__init__()
        d, h = cfg.d_model, cfg.heads
        self.variant, self.heads = cfg.variant, h
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        self.cat_emb = nn.ModuleList(nn.Embedding(c, 8) for c in cardinalities)
        self.cls_in = nn.Sequential(nn.Linear(n_num + 8 * len(cardinalities), 2 * d), nn.GELU(),
                                    nn.Linear(2 * d, d))  # fmt: skip
        self.node_in = nn.Linear(len(gr.NODE_FEATURES), d)
        self.station_emb = nn.Embedding(n_stations + 1, d)
        self.spd_bias = nn.Embedding(MAX_SPD + 2, h)
        self.edge_bias = nn.Sequential(nn.Linear(len(gr.EDGE_FEATURES), 32), nn.GELU(),
                                       nn.Linear(32, 2 * h))  # fmt: skip
        self.cls_bias = nn.Parameter(torch.zeros(2, h))  # CLS→node, node→CLS
        self.blocks = nn.ModuleList(Block(d, h, cfg.dropout) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def attention_bias(self, batch: dict, edge_index: torch.Tensor) -> torch.Tensor:
        b, n = batch["node_mask"].shape
        h = self.heads
        dist = shortest_paths(edge_index, batch["edge_mask"], n)
        bias = self.spd_bias(dist).permute(0, 3, 1, 2)  # (B, H, n, n)
        # edge features: query = target node, key = source node (and the reverse)
        ex = ((batch["edge_x"] - self.edge_mean) / self.edge_std).clamp(-CLIP, CLIP)
        eb = self.edge_bias(ex) * batch["edge_mask"][..., None]  # (B, E, 2H)
        src, dst = edge_index[..., 0].clamp(min=0), edge_index[..., 1].clamp(min=0)
        flat = bias.reshape(b, h, n * n)
        flat = flat.scatter_add(2, (dst * n + src)[:, None, :].expand(b, h, -1),
                                eb[..., :h].transpose(1, 2))  # fmt: skip
        flat = flat.scatter_add(2, (src * n + dst)[:, None, :].expand(b, h, -1),
                                eb[..., h:].transpose(1, 2))  # fmt: skip
        bias = flat.view(b, h, n, n)
        # add the [CLS] row and column, then mask padded keys
        full = torch.zeros(b, h, n + 1, n + 1, device=bias.device)
        full[:, :, 1:, 1:] = bias
        full[:, :, 0, 1:] = self.cls_bias[0][None, :, None]
        full[:, :, 1:, 0] = self.cls_bias[1][None, :, None]
        pad = ~batch["node_mask"]
        full[:, :, :, 1:] = full[:, :, :, 1:].masked_fill(pad[:, None, None, :], -1e4)
        return full

    def forward(self, batch: dict) -> torch.Tensor:
        cats = [emb(batch["cat"][:, i]) for i, emb in enumerate(self.cat_emb)]
        cls = self.cls_in(torch.cat([batch["num"], *cats], dim=1))[:, None, :]
        if self.variant == "nograph":
            h, bias = cls, None
        else:
            nx = ((batch["node_x"] - self.node_mean) / self.node_std).clamp(-CLIP, CLIP)
            nodes = self.node_in(nx) + self.station_emb(batch["node_station"])
            h = torch.cat([cls, nodes], dim=1)
            edge_index = batch["edge_index"]
            if self.variant == "random":
                edge_index = random_edges(edge_index, batch["edge_mask"],
                                          batch["node_mask"].sum(1))  # fmt: skip
            bias = self.attention_bias(batch, edge_index)
        for blk in self.blocks:
            h = blk(h, bias)
        return self.head(self.ln(h[:, 0])).squeeze(-1)


# --------------------------------------------------------------------------- training


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, graphs: Graphs, tab: Tabular, dev, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for rows in batches(len(graphs), batch_size, 0, shuffle=False):
        g = graphs.gather(rows)
        out.append(model(to_batch(g, tab, rows, dev)).float().cpu().numpy())
    return np.concatenate(out)


def run(cfg: TrainConfig, eval_months: list[str], out_dir: Path = config.PROCESSED,
        model_dir: Path = tb.MODELS_DIR) -> dict:  # fmt: skip
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = device()
    con = duckdb.connect()
    cols = tb.feature_columns(cfg.feature_set)
    h = cfg.horizon

    # refit: the tuning month joins the training data; no early stopping
    train_months = FIT_MONTHS + TUNE_MONTHS if cfg.refit else FIT_MONTHS
    fit_g, tune_g, eval_g = (Graphs(m, h) for m in (train_months, TUNE_MONTHS, eval_months))
    log.info("graphs: train %d (%s to %s), tune %d, eval %d points", len(fit_g),
             train_months[0], train_months[-1], len(tune_g), len(eval_g))  # fmt: skip
    where = f"horizon_min = {h} AND operating_day "

    def span(months: list[str]) -> str:
        return f"BETWEEN DATE '{months[0]}-01' AND last_day(DATE '{months[-1]}-01')"

    fit_df = load_tabular(con, where + span(train_months), cols, fit_g.points)
    prep = TabularPrep(cols).fit(fit_df)
    fit_t = prep.transform(fit_df)
    del fit_df
    tune_t = prep.transform(load_tabular(con, where + span(TUNE_MONTHS), cols, tune_g.points))
    eval_t = prep.transform(load_tabular(con, where + span(eval_months), cols, eval_g.points))

    n_stations = int(pd.read_parquet(gr.STATIONS_PATH)["idx"].max())
    stats = feature_stats(fit_g, seed=cfg.seed)
    model = GraphTransformer(cfg, prep.n_num, prep.cardinalities, n_stations, stats).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model %s: %d parameters, device %s", cfg.variant, n_params, dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(fit_g) * cfg.sample / cfg.batch_size)
    total = cfg.epochs * (steps_per_epoch + 2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, pct_start=0.05,
                                                total_steps=total)  # fmt: skip
    best, best_state, bad, history = math.inf, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for rows in batches(len(fit_g), cfg.batch_size, cfg.block, True, rng, cfg.sample):
            batch = to_batch(fit_g.gather(rows), fit_t, rows, dev)
            loss = F.l1_loss(model(batch), batch["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if cfg.refit:  # April is training data now: no tuning score, keep the last epoch
            history.append(dict(epoch=epoch, train_l1=float(np.mean(losses)),
                                seconds=time.time() - t0))  # fmt: skip
            log.info("epoch %d: train L1 %.4f (%.0fs)", epoch, np.mean(losses), time.time() - t0)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue
        tune_mae = float(np.mean(np.abs(predict(model, tune_g, tune_t, dev) - tune_t.y)))
        history.append(dict(epoch=epoch, train_l1=float(np.mean(losses)), tune_mae=tune_mae,
                            seconds=time.time() - t0))  # fmt: skip
        log.info("epoch %d: train L1 %.4f, tune MAE %.4f (%.0fs)", epoch, np.mean(losses),
                 tune_mae, time.time() - t0)  # fmt: skip
        if tune_mae < best - 1e-4:
            best, bad = tune_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    pred = predict(model, eval_g, eval_t, dev)

    name = f"gt_{cfg.variant}" + ("_refit" if cfg.refit else "")
    split = "test" if eval_months == TEST_MONTHS else "valid"
    preds = eval_g.points[KEYS].assign(delta_min=eval_t.y, **{f"pred_{name}": pred})
    out_path = out_dir / f"{name}_predictions_{split}_h{h}.parquet"
    preds.to_parquet(out_path, index=False)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), config=asdict(cfg), prep=prep.__dict__),
               model_dir / f"{name}_h{h}.pt")  # fmt: skip
    info = dict(variant=cfg.variant, horizon=h, params=n_params, best_tune_mae=best,
                eval_mae=float(np.mean(np.abs(pred - eval_t.y))), epochs=len(history),
                history=history, config=asdict(cfg))  # fmt: skip
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / f"{name}_fit_info_{split}_h{h}.json").write_text(json.dumps(info, indent=1))
    log.info("%s on %s: MAE %.4f (all labelled points), wrote %s", name, split,
             info["eval_mae"], out_path.name)  # fmt: skip
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=VARIANTS, default="graph")
    parser.add_argument("--horizon", type=int, default=15, choices=config.HORIZONS_MIN)
    parser.add_argument("--feature-set", default="full_network_plus", choices=list(tb.FEATURE_SETS))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--sample", type=float, default=1.0, help="share of fitting data")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--layers", type=int, default=TrainConfig.layers)
    parser.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refit", action="store_true",
                        help="train on Aug-Apr, fixed --epochs, no early stopping")  # fmt: skip
    parser.add_argument("--test", action="store_true", help="evaluate on June (final run only)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainConfig(
        variant=args.variant, horizon=args.horizon, feature_set=args.feature_set,
        epochs=args.epochs, sample=args.sample, batch_size=args.batch_size, lr=args.lr,
        layers=args.layers, d_model=args.d_model, seed=args.seed, refit=args.refit,
    )  # fmt: skip
    run(cfg, TEST_MONTHS if args.test else VALID_MONTHS)


if __name__ == "__main__":
    main()
