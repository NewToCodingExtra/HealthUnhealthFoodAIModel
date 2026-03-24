import pathlib
import sys
import joblib
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transforms import log1p_added_sugar  # pickle compatibility for loader


def get_coef(pipe):
    try:
        return pipe.named_steps["clf"].calibrated_classifiers_[0].estimator.coef_[0]
    except Exception:
        return pipe.named_steps["clf"].estimator.coef_[0]


def main():
    model_path = pathlib.Path(r"c:\Python\HealthUnhealthFoodAIModel\trained_model.pkl")
    out_path = pathlib.Path(r"c:\Python\HealthUnhealthFoodAIModel\FEATURE_IMPORTANCE_REPORT.md")
    data = joblib.load(model_path)
    pipes = data["pipelines"]

    rows = []
    for model_key, model_label in [("core", "Core Model (9 features)"), ("all", "All-Features Model (14 features)")]:
        feats = data["core_features"] if model_key == "core" else data["all_features"]
        coef = get_coef(pipes[model_key])
        for f, c in zip(feats, coef):
            c = float(c)
            rows.append(
                {
                    "model": model_label,
                    "feature": f,
                    "coef_healthy": c,
                    "coef_unhealthy": -c,
                    "abs_coef": abs(c),
                    "direction": "Healthy (+)" if c > 0 else ("Unhealthy (-)" if c < 0 else "Neutral"),
                }
            )

    df = pd.DataFrame(rows)

    lines = [
        "# NutriScan Feature Importance Report",
        "",
        "Generated from current `trained_model.pkl` coefficients (calibrated logistic base estimator).",
        "",
        "## Core Model (9 features)",
        "",
        "| Feature | Coef (Healthy) | Coef (Unhealthy) | Direction | |coef| |",
        "|---|---:|---:|---|---:|",
    ]

    core = df[df["model"] == "Core Model (9 features)"].sort_values("abs_coef", ascending=False)
    for _, r in core.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['coef_healthy']:.6f} | {r['coef_unhealthy']:.6f} | {r['direction']} | {r['abs_coef']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## All-Features Model (14 features)",
            "",
            "| Feature | Coef (Healthy) | Coef (Unhealthy) | Direction | |coef| |",
            "|---|---:|---:|---|---:|",
        ]
    )

    allf = df[df["model"] == "All-Features Model (14 features)"].sort_values("abs_coef", ascending=False)
    for _, r in allf.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['coef_healthy']:.6f} | {r['coef_unhealthy']:.6f} | {r['direction']} | {r['abs_coef']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Positive coefficient means higher values push toward `Healthy`.",
            "- Negative coefficient means higher values push toward `Unhealthy`.",
            "- Magnitude shows relative influence after preprocessing.",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
