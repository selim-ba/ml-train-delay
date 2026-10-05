"""Tests for the graph transformer: distances, masking, variants, batching, preprocessing.
Skipped when PyTorch is not installed (``uv sync --group dl``)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from swissdelay.features import graph as gr  # noqa: E402
from swissdelay.models import graph_transformer as gt  # noqa: E402

N, E, N_NUM, CARDS = 8, 10, 5, [4, 3]


def make_batch(b: int = 3, seed: int = 0) -> dict:
    """Paths 0 → 1 → … → n−1 with n = 3, 5, 6 nodes; the rest is padding."""
    g = torch.Generator().manual_seed(seed)
    node_n = torch.tensor([3, 5, 6][:b])
    node_mask = torch.arange(N)[None, :] < node_n[:, None]
    edge_index = torch.full((b, E, 2), -1, dtype=torch.long)
    edge_mask = torch.zeros(b, E, dtype=torch.bool)
    for i, n in enumerate(node_n.tolist()):
        for j in range(n - 1):
            edge_index[i, j] = torch.tensor([j, j + 1])
            edge_mask[i, j] = True
    return dict(
        node_x=torch.randn(b, N, len(gr.NODE_FEATURES), generator=g) * node_mask[..., None],
        node_station=torch.randint(1, 20, (b, N), generator=g) * node_mask,
        node_mask=node_mask, edge_x=torch.randn(b, E, len(gr.EDGE_FEATURES), generator=g),
        edge_index=edge_index, edge_mask=edge_mask, num=torch.randn(b, N_NUM, generator=g),
        cat=torch.cat([torch.randint(0, c, (b, 1), generator=g) for c in CARDS], 1),
        y=torch.randn(b, generator=g),
    )  # fmt: skip


def make_model(variant: str) -> "gt.GraphTransformer":
    torch.manual_seed(0)
    cfg = gt.TrainConfig(variant=variant, d_model=16, heads=2, layers=2, dropout=0.0)
    nf, ef = len(gr.NODE_FEATURES), len(gr.EDGE_FEATURES)
    stats = dict(node_mean=np.zeros(nf), node_std=np.ones(nf),
                 edge_mean=np.zeros(ef), edge_std=np.ones(ef))  # fmt: skip
    return gt.GraphTransformer(cfg, N_NUM, CARDS, n_stations=20, stats=stats).eval()


def test_shortest_paths():
    batch = make_batch()
    dist = gt.shortest_paths(batch["edge_index"], batch["edge_mask"], N)
    assert dist[0, 0, 2] == 2 and dist[0, 2, 0] == 2  # undirected
    assert dist[2, 0, 5] == 5 and dist[2, 5, 5] == 0
    assert dist[0, 0, 3] == gt.MAX_SPD + 1  # padding: unreachable
    assert dist[1, 0, 4] == 4


@pytest.mark.parametrize("variant", gt.VARIANTS)
def test_forward_shapes(variant):
    out = make_model(variant)(make_batch())
    assert out.shape == (3,) and torch.isfinite(out).all()


def test_padding_is_ignored():
    model, batch = make_model("graph"), make_batch()
    with torch.no_grad():
        ref = model(batch)
        changed = dict(batch)
        changed["node_x"] = batch["node_x"] + 5.0 * ~batch["node_mask"][..., None]
        changed["node_station"] = torch.where(batch["node_mask"], batch["node_station"], 7)
        torch.testing.assert_close(model(changed), ref)


def test_graph_uses_nodes_and_nograph_does_not():
    batch = make_batch()
    changed = dict(batch, node_x=batch["node_x"] + 1.0 * batch["node_mask"][..., None])
    with torch.no_grad():
        g = make_model("graph")
        assert not torch.allclose(g(changed), g(batch))
        ng = make_model("nograph")
        torch.testing.assert_close(ng(changed), ng(batch))


def test_edges_change_the_output():
    batch, model = make_batch(), make_model("graph")
    no_edges = dict(batch, edge_mask=torch.zeros_like(batch["edge_mask"]))
    with torch.no_grad():
        assert not torch.allclose(model(no_edges), model(batch))


def test_random_edges_stay_inside_the_subgraph():
    batch = make_batch()
    torch.manual_seed(1)
    ei = gt.random_edges(batch["edge_index"], batch["edge_mask"], batch["node_mask"].sum(1))
    n = batch["node_mask"].sum(1)
    real = batch["edge_mask"]
    assert (ei[real] >= 0).all()
    assert (ei[..., 0] < n[:, None])[real].all() and (ei[..., 1] < n[:, None])[real].all()
    assert (ei[~real] == -1).all()


def test_batches_cover_every_row_once():
    rows = np.concatenate(list(gt.batches(1000, 64, 0, shuffle=False)))
    assert (rows == np.arange(1000)).all()
    rng = np.random.default_rng(0)
    rows = np.concatenate(list(gt.batches(1000, 64, 16, shuffle=True, rng=rng)))
    assert sorted(rows.tolist()) == list(range(1000))
    half = np.concatenate(list(gt.batches(1000, 64, 16, True, np.random.default_rng(0), 0.5)))
    assert 450 <= len(half) <= 550 and len(set(half.tolist())) == len(half)


def test_tabular_prep():
    fit = pd.DataFrame({"a": [1.0, 2.0, np.nan, 3.0], "b": [0, 1, 1, 0], "category": list("xyxy"),
                        "delta_min": 0.0, "pred_historical": -0.8})  # fmt: skip
    prep = gt.TabularPrep(["a", "b", "category"]).fit(fit)
    assert prep.indicator_cols == ["a"] and prep.n_num == 3 and prep.cardinalities == [3]
    new = fit.assign(category=["x", "z", "y", "x"])  # "z" unseen
    t = prep.transform(new)
    assert t.num.shape == (4, 3) and t.cat[:, 0].tolist() == [1, 0, 2, 1]
    assert t.num[2, 2] == 1.0 and t.num[0, 2] == 0.0  # missing indicator
    assert abs(t.num[:, 1].mean()) < 1e-6  # standardised on the fitting rows
