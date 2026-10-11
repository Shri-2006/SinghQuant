"""
Image smoke test (Phase 2 I-5). Runs as a Docker build step on every
architecture (amd64 CI, arm64 CI, the Oracle ARM host). No network, no
credentials, no broker, and nothing is written outside a temporary
directory.

    python -m tools.smoke_imports [--require-rl]

Checks that every runtime module imports, that the paper-trading guard
resolves the paper endpoint, that the native wheels on this CPU actually
work (numpy / pandas / XGBoost train and predict; torch if --require-rl),
and that the database layer can create, migrate and read a schema.
"""
import os
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tmp = tempfile.mkdtemp(prefix="singhquant-smoke-")
    os.environ["SINGHQUANT_DB_PATH"] = os.path.join(tmp, "smoke.db")
    os.environ["SINGHQUANT_MODEL_DIR"] = os.path.join(tmp, "models")
    os.environ["SINGHQUANT_NO_AUTO_MIGRATE"] = "1"
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "POLYGON_API_KEY"):
        os.environ.setdefault(key, "smoke-test-not-a-key")
    os.environ.pop("SINGHQUANT_ALLOW_LIVE_TRADING", None)

    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost

    import run
    import core.execution, core.portfolio, core.lifecycle, core.storage            # noqa: F401
    import strategies.stable, strategies.risky1, strategies.risky2                  # noqa: F401
    import models.train, models.artifacts, models.retrain                           # noqa: F401
    import data.polygon_fetcher, data.sentiment_fetcher, data.macro_fetcher         # noqa: F401
    import tools.healthcheck, tools.db_maintenance, tools.model_manifest            # noqa: F401
    import alpaca_trade_api                                                          # noqa: F401

    from paper_trading.alpaca_paper import resolve_base_url
    from core.config import ALPACA_PAPER_URL
    for s in ("stable", "risky1", "risky2"):
        assert resolve_base_url(s) == ALPACA_PAPER_URL, f"{s} does not resolve to the paper endpoint"

    from core import logger
    assert logger.init_db() == logger.SCHEMA_VERSION
    assert logger.schema_ready()
    assert run.storage_and_schema_preflight() == []

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(models.train.FEATURE_COLUMNS)))
    y = (X[:, 0] > 0).astype(int)
    model = models.train.build_model().fit(X, y)
    assert set(np.unique(model.predict(X[:5]))) <= {0, 1}
    assert pd.DataFrame(X).rolling(5).mean().iloc[-1].notna().all()

    versions = {"python": platform.python_version(), "machine": platform.machine(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
    if "--require-rl" in argv:
        import torch
        import stable_baselines3
        assert float(torch.ones(3).sum()) == 3.0
        versions.update(torch=torch.__version__, stable_baselines3=stable_baselines3.__version__,
                        torch_cuda=bool(torch.version.cuda))
    print("smoke test ok: " + ", ".join(f"{k}={v}" for k, v in versions.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
