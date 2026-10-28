"""Task 7: write the processed dataset, hash it and store the hash with the config."""
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from common import add_derived, get_logger, lab_path, load_config, results_dir, sha256_file
from features import load_clean


def main(cfg):
    log = get_logger()
    train, test = load_clean()
    processed = pd.concat([add_derived(train).assign(split="train"),
                           add_derived(test).assign(split="test")]).sort_index()

    path = lab_path(cfg["data"]["processed_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # fixed float format and line endings: the bytes, and so the hash, do not depend on the OS
    processed.to_csv(path, index_label="row_id", float_format="%.10g", lineterminator="\n")
    digest = sha256_file(path)

    out = results_dir(cfg)
    raw_info = yaml.safe_load(open(out / "ingest.json", encoding="utf-8"))  # JSON is valid YAML
    manifest = {
        "processed_sha256": digest,
        "processed_file": cfg["data"]["processed_path"],
        "processed_rows": len(processed),
        "processed_columns": list(processed.columns),
        "raw_sha256": raw_info["raw_sha256"],
        "preprocessor_sha256": sha256_file(out / "preprocessor.joblib"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                        "joblib": joblib.__version__, "pyyaml": yaml.__version__},
        "config": cfg,
    }
    with open(out / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True, width=100)
    log.info("finalize: %d rows, sha256 %s", len(processed), digest)
    return manifest


if __name__ == "__main__":
    main(load_config())
