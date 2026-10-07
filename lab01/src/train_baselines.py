"""Step 2: train both baselines and save test accuracy to results/baseline_accuracy.csv."""
import csv

from sklearn.metrics import accuracy_score

from common import RESULTS_DIR, load_split, make_models, set_seeds


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train size: {len(X_train)}, test size: {len(X_test)}")

    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})
        print(f"{name}: test accuracy = {acc:.4f}")

    out = RESULTS_DIR / "baseline_accuracy.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
