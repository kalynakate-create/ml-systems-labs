"""Task 2: fetch UCI Adult from OpenML and store an untouched raw copy."""
from sklearn.datasets import fetch_openml

from common import get_logger, lab_path, load_config, results_dir, save_json, sha256_file


def main(cfg):
    log = get_logger()
    d = cfg["data"]
    bunch = fetch_openml(name=d["openml_name"], version=d["openml_version"],
                         as_frame=True, data_home=d["data_home"])
    raw = bunch.frame  # 14 features + target, exactly as OpenML serves them

    rows, cols = raw.shape
    log.info("ingest: %d rows, %d columns (OpenML id %s)", rows, cols, bunch.details["id"])
    assert (rows, cols) == (d["expected_rows"], d["expected_cols"]), (rows, cols)

    # no renaming, no filling, no type changes: the file is the raw source for every later step
    path = lab_path(d["raw_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, index=False, lineterminator="\n")

    info = {
        "source": f"https://www.openml.org/d/{bunch.details['id']}",
        "openml_id": bunch.details["id"],
        "openml_version": bunch.details["version"],
        "licence": bunch.details.get("licence"),
        "rows": rows,
        "columns": cols,
        "column_names": list(raw.columns),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "raw_file": str(d["raw_path"]),
        "raw_sha256": sha256_file(path),
    }
    save_json(info, results_dir(cfg) / "ingest.json")
    return info


if __name__ == "__main__":
    main(load_config())
