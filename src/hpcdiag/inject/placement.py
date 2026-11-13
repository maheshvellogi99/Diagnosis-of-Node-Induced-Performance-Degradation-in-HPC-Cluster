"""Random-placement base (D29): each job's node set is replaced by |N_j| distinct nodes drawn
uniformly at random from the nodes of its partition (seeded, one draw per base seed)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def partition_nodes(job_nodes: pd.DataFrame, jobs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sorted node ids observed for each partition in the real placement."""
    x = job_nodes[["row_id", "node_id"]].merge(jobs[["row_id", "partition"]], on="row_id")
    return {p: np.sort(g["node_id"].unique()) for p, g in x.groupby("partition")}


def random_placement(job_nodes: pd.DataFrame, jobs: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Same rows/columns as `job_nodes` (one row per job x node), node_id redrawn per job.
    Jobs are processed in row_id order with one generator, so the result depends only on seed."""
    pnodes = partition_nodes(job_nodes, jobs)
    part = jobs.set_index("row_id")["partition"]
    jn = job_nodes.sort_values(["row_id", "node_id"], kind="stable").reset_index(drop=True)
    sizes = jn.groupby("row_id", sort=True).size()
    rng = np.random.default_rng(seed)
    new = np.empty(len(jn), dtype="int64")
    pos = 0
    for rid, k in sizes.items():
        pool = pnodes[part[rid]]
        if k > len(pool):
            raise ValueError(f"job {rid} needs {k} nodes, partition has {len(pool)}")
        new[pos : pos + k] = np.sort(rng.choice(pool, size=k, replace=False))
        pos += k
    out = jn.copy()
    out["node_id"] = new
    return out
