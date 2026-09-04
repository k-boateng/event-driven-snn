"""Frozen network and stimulus on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from .network import Network, build as build_net
from .spec import Config
from .stimulus import Stimulus, build as build_stim


def data_root() -> Path:
    return Path(os.environ.get("SNNKERN_DATA", "data"))


def paths(cfg: Config, root: Path | None = None) -> tuple[Path, Path]:
    root = root or data_root()
    return root / f"{cfg.tag()}.net.npz", root / f"{cfg.tag()}.stim.npz"


def load_or_build(cfg: Config, root: Path | None = None,
                  verbose: bool = False) -> tuple[Network, Stimulus]:
    net_p, stim_p = paths(cfg, root)
    if net_p.exists() and stim_p.exists():
        if verbose:
            print(f"load {net_p.name}")
        return Network.load(net_p), Stimulus.load(stim_p)

    if verbose:
        print(f"build {cfg.tag()}")
    net_p.parent.mkdir(parents=True, exist_ok=True)
    net = build_net(cfg.net)
    stim = build_stim(cfg.net.n, cfg.stim)
    # Write to a temporary name and rename, so an interrupted job cannot leave
    # a truncated .npz that later loads as a valid-looking network. The temp
    # name must itself end in .npz -- np.savez appends the extension when it is
    # missing, which would put the file somewhere the rename cannot find it.
    tmp_n = net_p.with_name(net_p.name + ".tmp.npz")
    tmp_s = stim_p.with_name(stim_p.name + ".tmp.npz")
    net.save(tmp_n); stim.save(tmp_s)
    tmp_n.replace(net_p); tmp_s.replace(stim_p)
    return net, stim