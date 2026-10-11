"""
Model artifact manifest tool (Phase 2 I-3).

A strategy only opens new positions when every model artifact it needs is
listed in <model dir>/manifest.json and the file's SHA-256 matches. Models
produced by retraining are registered automatically; models produced
elsewhere (e.g. copied out of the old container during the cutover) must
be registered here, deliberately, after a human has checked them.

    python -m tools.model_manifest status [--dir DIR]
        every expected artifact for the enabled strategies, verified; exits 1
        if any is not usable

    python -m tools.model_manifest register NAME --file PATH --version V --note "..." [--dir DIR] [--confirm]
        validates PATH (loads it and checks the strategy's own requirements),
        copies it into the model directory under a content-addressed name and
        records its SHA-256. Without --confirm it is a dry run. The source
        file is never modified.

WARNING: registering loads (unpickles) the file. Only register files you
produced or obtained from your own deployment.

The model directory is SINGHQUANT_MODEL_DIR (default: models/).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SINGHQUANT_NO_AUTO_MIGRATE", "1")

from models import artifacts  # noqa: E402


def _enabled_strategies():
    from core.config import FEATURE_FLAGS
    return ["stable", "risky1"] + (["risky2"] if FEATURE_FLAGS.get("risky2_enabled") else [])


def _validator(name):
    """(loader, check) for a logical artifact name."""
    if name.endswith(".pkl"):
        from models.train import _load_pickle, check_feature_width
        return _load_pickle, check_feature_width
    if name.endswith(".zip"):
        from models.rl_train import _load_ppo, _check_ppo
        return _load_ppo, _check_ppo
    raise ValueError(f"unknown artifact type for {name}")


def _strategy_of(name):
    for s in ("stable", "risky1", "risky2"):
        if name in artifacts.expected_artifacts(s):
            return s
    return None


def cmd_status(directory):
    bad = 0
    for strategy in _enabled_strategies():
        for name in artifacts.expected_artifacts(strategy):
            loader, check = _validator(name)
            h = artifacts.ArtifactHandle(name, loader, check, directory)
            status = h.refresh()
            bad += status != artifacts.OK
            print(f"{strategy:7s} {name:24s} {status:16s} {h.detail}")
    print(f"model directory: {directory or artifacts.model_dir()}")
    return 1 if bad else 0


def cmd_register(name, src, version, note, directory, confirm):
    strategy = _strategy_of(name)
    if strategy is None:
        print(f"ERROR: {name} is not an artifact any strategy uses (expected e.g. stable_model.pkl)")
        return 2
    if not Path(src).is_file():
        print(f"ERROR: {src} is not a file")
        return 2
    loader, check = _validator(name)
    try:
        model = loader(src)
    except Exception as e:
        print(f"ERROR: {src} cannot be loaded as {name}: {e}")
        return 2
    problem = check(model) if check else None
    if problem:
        print(f"ERROR: {src} is not usable as {name}: {problem}")
        return 2
    digest = artifacts.sha256_file(src)
    print(f"{name}: {src} validated (strategy {strategy}, sha256 {digest})")
    if not confirm:
        print("DRY RUN: nothing registered. Add --confirm to copy it into the model directory and record it.")
        return 0
    entry = artifacts.register_existing(name, src, strategy, version,
                                        meta={"source": str(Path(src).resolve()), "note": note,
                                              "registered_by": "tools/model_manifest.py"},
                                        directory=directory)
    print(f"registered {name} -> {entry['file']} version {entry['version']}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None, help="model directory (default: SINGHQUANT_MODEL_DIR)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("register")
    r.add_argument("name", help="logical artifact name, e.g. stable_model.pkl or risky2_BTCUSD.zip")
    r.add_argument("--file", required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--note", required=True, help="where the file came from and who checked it")
    r.add_argument("--confirm", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "status":
        return cmd_status(a.dir)
    return cmd_register(a.name, a.file, a.version, a.note, a.dir, a.confirm)


if __name__ == "__main__":
    sys.exit(main())
