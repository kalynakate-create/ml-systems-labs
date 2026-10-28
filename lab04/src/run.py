"""Lab 4: the whole pipeline in one command.

    python lab04/src/run.py      (from the repo root; any working directory works)
"""
import sys

import features
import finalize
import ingest
import label_audit
import transform
import validate
from common import LAB_DIR, get_logger, load_config

STEPS = [("ingest", ingest), ("validate", validate), ("transform", transform),
         ("features", features), ("label_audit", label_audit), ("finalize", finalize)]


def main():
    (LAB_DIR / "results").mkdir(exist_ok=True)
    (LAB_DIR / "results" / "pipeline.log").write_text("", encoding="utf-8")  # one log per run
    cfg = load_config()
    log = get_logger()
    for name, step in STEPS:
        log.info("--- %s ---", name)
        step.main(cfg)
    log.info("done")


if __name__ == "__main__":
    sys.exit(main())
