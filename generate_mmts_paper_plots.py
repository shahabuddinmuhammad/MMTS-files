"""
Generate publication-quality plots for the MMTS PRISM experiments.

Expected input files in the same folder as this script:
  - result10.1.1.txt                  # resilience with congestion
  - result10.1.2.txt                  # resilience baseline
  - result10.2.1.txt                  # infrastructure / allStations / lanes4Bus / time
  - result10.2.1 time-Cost.txt        # Pareto: time-cost
  - result10.2.1 time-Emission.txt    # Pareto: time-emission
  - result10.3 time(1).txt            # incident experiments
  - result10.4 delays(1).txt          # random delay experiments

The script saves all figures into ./mmts_plots/ as PNG and PDF.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "mmts_plots"
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "resilience_congestion": BASE_DIR / "result10.1.1.txt",
    "resilience_baseline": BASE_DIR / "result10.1.2.txt",
    "infrastructure_time": BASE_DIR / "result10.2.1.txt",
    "pareto_cost": BASE_DIR / "result10.2.1 time-Cost.txt",
    "pareto_emission": BASE_DIR / "result10.2.1 time-Emission.txt",
    "incidents": BASE_DIR / "result10.3 time.txt",
    "delays": BASE_DIR / "result10.4 delays.txt",
}

STATION_LABELS = {
    1: "Base stations",
    2: "Cycle stations everywhere",
    3: "Bus stations everywhere",
}

LANE_LABELS = {
    0: "Shared lanes",
    1: "Reserved bus lanes",
}

INCIDENT_LABELS = {
    1: "Normal",
    2: "Heavy traffic",
    3: "Accident",
    4: "Breakdown",
    5: "Blockage",
}

LINK_LABELS = {
    1: "1-2", 2: "2-3", 3: "1-4", 4: "2-5", 5: "3-7",
    6: "4-5", 7: "5-6", 8: "6-7", 9: "4-8", 10: "5-9",
    11: "7-10", 12: "8-9", 13: "9-10", 14: "8-11", 15: "9-12",
    16: "9-13", 17: "10-13", 18: "10-17", 19: "11-12", 20: "11-14",
    21: "12-15", 22: "13-16", 23: "14-15", 24: "15-16", 25: "16-17",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
})


def savefig(name: str) -> None:
    """Save current figure as PNG and PDF."""
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    for col in df.columns:
        if col != "Result":
            
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    if "Result" in df.columns:
        df["Result"] = df["Result"].replace("Infinity", np.inf)
        # Numeric result files become numeric; Pareto result files remain strings.
        df["Result_num"] = pd.to_numeric(df["Result"], errors="coerce")
    return df


def parse_pareto_result(value: str) -> list[tuple[float, float]]:
    """Parse PRISM multi-objective output: [(time, metric), ...]."""
    try:
        parsed = ast.literal_eval(str(value))
        return [(float(x), float(y)) for x, y in parsed]
    except Exception:
        return []


def pareto_expand(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for time_value, metric_value in parse_pareto_result(r["Result"]):
            rows.append({
                "allStations": int(r["allStations"]),
                "lanes4Bus": int(r["lanes4Bus"]),
                "pCongFixed": float(r["pCongFixed"]),
                "time": time_value,
                metric_name: metric_value,
            })
    return pd.DataFrame(rows)


def finite_only(df: pd.DataFrame, col: str = "Result_num") -> pd.DataFrame:
    return df[np.isfinite(df[col])].copy()


# ---------------------------------------------------------------------
# Plot group 1: infrastructure design / reserved-lane effects
# ---------------------------------------------------------------------
def plot_infrastructure_time(df: pd.DataFrame) -> None:
    df = finite_only(df)

    # All strategies, separate shared/reserved-lane curves.
    plt.figure(figsize=(8, 5))
    for (station, lane), g in df.groupby(["allStations", "lanes4Bus"]):
        g = g.sort_values("pCongFixed")
        label = f"{STATION_LABELS.get(station, f'allStations={station}')} - {LANE_LABELS.get(lane, lane)}"
        plt.plot(g["pCongFixed"], g["Result_num"], marker="o", label=label)
    plt.xlabel("Congestion probability")
    plt.ylabel("Minimum expected travel time (min)")
    plt.title("Infrastructure strategies under congestion")
    plt.legend(ncol=1, frameon=True)
    savefig("fig01_infrastructure_time_all_strategies")

    # Reserved-lane benefit by strategy.
    pivot = df.pivot_table(index=["allStations", "pCongFixed"], columns="lanes4Bus", values="Result_num").reset_index()
    if 0 in pivot.columns and 1 in pivot.columns:
        pivot["saving"] = pivot[0] - pivot[1]
        plt.figure(figsize=(8, 5))
        for station, g in pivot.groupby("allStations"):
            g = g.sort_values("pCongFixed")
            plt.plot(g["pCongFixed"], g["saving"], marker="o", label=STATION_LABELS.get(station, f"allStations={station}"))
        plt.axhline(0, linewidth=1)
        plt.xlabel("Congestion probability")
        plt.ylabel("Time saving from reserved bus lanes (min)")
        plt.title("Reserved bus-lane benefit")
        plt.legend(frameon=True)
        savefig("fig02_reserved_lane_time_saving")

    # Bar chart at selected congestion levels.
    selected = df[df["pCongFixed"].isin([0.0, 0.5, 1.0])].copy()
    selected["strategy"] = selected["allStations"].map(STATION_LABELS) + "\n" + selected["lanes4Bus"].map(LANE_LABELS)
    selected["p_label"] = "p=" + selected["pCongFixed"].astype(str)
    table = selected.pivot_table(index="strategy", columns="p_label", values="Result_num")
    plt.figure(figsize=(9, 5))
    table.plot(kind="bar", ax=plt.gca())
    plt.xlabel("Infrastructure configuration")
    plt.ylabel("Minimum expected travel time (min)")
    plt.title("Infrastructure comparison at representative congestion levels")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    savefig("fig03_infrastructure_bar_selected_congestion")


# ---------------------------------------------------------------------
# Plot group 2: Pareto fronts
# ---------------------------------------------------------------------
def plot_pareto(df_raw: pd.DataFrame, metric_name: str, metric_label: str, filename_prefix: str) -> None:
    df = pareto_expand(df_raw, metric_name)
    if df.empty:
        return

    # Selected congestion fronts keep figures readable.
    selected_p = [0.0, 0.5, 1.0]
    for p in selected_p:
        subset = df[np.isclose(df["pCongFixed"], p)]
        if subset.empty:
            continue
        plt.figure(figsize=(8, 5))
        for (station, lane), g in subset.groupby(["allStations", "lanes4Bus"]):
            g = g.sort_values(metric_name)
            label = f"{STATION_LABELS.get(station, f'allStations={station}')} - {LANE_LABELS.get(lane, lane)}"
            plt.plot(g[metric_name], g["time"], marker="o", label=label)
        plt.xlabel(metric_label)
        plt.ylabel("Expected travel time (min)")
        plt.title(f"Pareto front: time vs {metric_label.lower()} at p={p:g}")
        plt.legend(frameon=True)
        savefig(f"{filename_prefix}_pareto_p_{str(p).replace('.', '_')}")

    # Combined front for base stations only, shared vs reserved lanes.
    subset = df[df["allStations"] == 1]
    plt.figure(figsize=(8, 5))
    for (lane, p), g in subset.groupby(["lanes4Bus", "pCongFixed"]):
        if p not in selected_p:
            continue
        g = g.sort_values(metric_name)
        plt.plot(g[metric_name], g["time"], marker="o", label=f"{LANE_LABELS.get(lane, lane)}, p={p:g}")
    plt.xlabel(metric_label)
    plt.ylabel("Expected travel time (min)")
    plt.title(f"Base-station Pareto fronts: time vs {metric_label.lower()}")
    plt.legend(frameon=True)
    savefig(f"{filename_prefix}_base_combined")


# ---------------------------------------------------------------------
# Plot group 3: incidents
# ---------------------------------------------------------------------
def plot_incidents(df: pd.DataFrame) -> None:
    df = finite_only(df)

    # Congestion curves for each incident type, averaged over incident nodes.
    avg = df.groupby(["incidentType", "lanes4Bus", "pCongFixed"], as_index=False)["Result_num"].mean()
    for lane in sorted(avg["lanes4Bus"].unique()):
        plt.figure(figsize=(8, 5))
        sub = avg[avg["lanes4Bus"] == lane]
        for itype, g in sub.groupby("incidentType"):
            g = g.sort_values("pCongFixed")
            plt.plot(g["pCongFixed"], g["Result_num"], marker="o", label=INCIDENT_LABELS.get(itype, f"Type {itype}"))
        plt.xlabel("Congestion probability")
        plt.ylabel("Average travel time (min)")
        plt.title(f"Incident severity under congestion ({LANE_LABELS.get(lane, lane)})")
        plt.legend(frameon=True)
        savefig(f"fig04_incident_congestion_curves_lane_{lane}")

    # Incident-node heatmap at high congestion, no reserved lanes.
    for lane in [0, 1]:
        for p in [0.3, 0.6, 0.9]:
            sub = df[(df["lanes4Bus"] == lane) & (np.isclose(df["pCongFixed"], p))]
            if sub.empty:
                continue
            heat = sub.pivot_table(index="incidentType", columns="incidentNode", values="Result_num", aggfunc="mean")
            plt.figure(figsize=(10, 4.8))
            im = plt.imshow(heat.values, aspect="auto")
            plt.colorbar(im, label="Travel time (min)")
            plt.xticks(range(len(heat.columns)), heat.columns)
            plt.yticks(range(len(heat.index)), [INCIDENT_LABELS.get(i, i) for i in heat.index])
            plt.xlabel("Incident node")
            plt.ylabel("Incident type")
            plt.title(f"Incident impact heatmap, p={p:g}, {LANE_LABELS.get(lane, lane)}")
            savefig(f"fig05_incident_heatmap_lane_{lane}_p_{str(p).replace('.', '_')}")

    # Critical incident nodes at high congestion.
    sub = df[np.isclose(df["pCongFixed"], 0.9)]
    rank = sub.groupby("incidentNode", as_index=False)["Result_num"].mean().sort_values("Result_num", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(rank["incidentNode"].astype(str), rank["Result_num"])
    plt.xlabel("Incident node")
    plt.ylabel("Average travel time at p=0.9 (min)")
    plt.title("Critical incident-node ranking")
    plt.grid(axis="y", alpha=0.3)
    savefig("fig06_critical_incident_nodes")


# ---------------------------------------------------------------------
# Plot group 4: random delays
# ---------------------------------------------------------------------
def plot_delays(df: pd.DataFrame) -> None:
    df = finite_only(df)
    delay_cols = ["p_car_delay", "p_bus_delay", "p_train_delay", "p_cycle_delay"]

    # Average effect of delay probability by transport mode.
    for p in sorted(df["pCongFixed"].unique()):
        if p not in [0.0, 0.3, 0.6, 0.9]:
            continue
        plt.figure(figsize=(8, 5))
        sub = df[np.isclose(df["pCongFixed"], p)]
        for col in delay_cols:
            g = sub.groupby(col, as_index=False)["Result_num"].mean().sort_values(col)
            plt.plot(g[col], g["Result_num"], marker="o", label=col.replace("p_", "").replace("_delay", ""))
        plt.xlabel("Delay probability")
        plt.ylabel("Average travel time (min)")
        plt.title(f"Mode-specific random delay sensitivity at p={p:g}")
        plt.legend(title="Mode", frameon=True)
        savefig(f"fig07_delay_probability_sensitivity_p_{str(p).replace('.', '_')}")

    # Delay-node ranking.
    rank = df.groupby("delayNode", as_index=False)["Result_num"].mean().sort_values("Result_num", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(rank["delayNode"].astype(str), rank["Result_num"])
    plt.xlabel("Delay node")
    plt.ylabel("Average travel time (min)")
    plt.title("Critical delay-node ranking")
    plt.grid(axis="y", alpha=0.3)
    savefig("fig08_critical_delay_nodes")

    # Delay amount sensitivity.
    g = df.groupby(["delayInMin", "pCongFixed"], as_index=False)["Result_num"].mean()
    plt.figure(figsize=(8, 5))
    for p, gp in g.groupby("pCongFixed"):
        gp = gp.sort_values("delayInMin")
        plt.plot(gp["delayInMin"], gp["Result_num"], marker="o", label=f"p={p:g}")
    plt.xlabel("Injected delay duration (min)")
    plt.ylabel("Average travel time (min)")
    plt.title("Sensitivity to delay duration")
    plt.legend(frameon=True)
    savefig("fig09_delay_duration_sensitivity")


# ---------------------------------------------------------------------
# Plot group 5: disabled nodes / disabled links resilience
# ---------------------------------------------------------------------
def plot_resilience(df_cong: pd.DataFrame, df_base: pd.DataFrame) -> None:
    df_cong = finite_only(df_cong)
    df_base = finite_only(df_base)

    # Heatmaps for disabled nodes and links at each congestion level.
    for p in sorted(df_cong["pCongFixed"].unique()):
        sub = df_cong[np.isclose(df_cong["pCongFixed"], p)]
        heat = sub.pivot_table(index="disabledNode", columns="disabledLink", values="Result_num", aggfunc="mean")
        plt.figure(figsize=(11, 5.5))
        im = plt.imshow(heat.values, aspect="auto")
        plt.colorbar(im, label="Travel time (min)")
        plt.xticks(range(len(heat.columns)), [LINK_LABELS.get(int(c), str(c)) for c in heat.columns], rotation=90)
        plt.yticks(range(len(heat.index)), heat.index)
        plt.xlabel("Disabled link")
        plt.ylabel("Disabled node")
        plt.title(f"Resilience heatmap under disabled node/link, p={p:g}")
        savefig(f"fig10_resilience_heatmap_p_{str(p).replace('.', '_')}")

    # Rank disabled nodes by worst/average impact.
    node_rank = df_cong.groupby("disabledNode", as_index=False)["Result_num"].mean().sort_values("Result_num", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(node_rank["disabledNode"].astype(str), node_rank["Result_num"])
    plt.xlabel("Disabled node")
    plt.ylabel("Average travel time across links/congestion (min)")
    plt.title("Disabled-node criticality ranking")
    plt.grid(axis="y", alpha=0.3)
    savefig("fig11_disabled_node_ranking")

    # Rank disabled links by average impact.
    link_rank = df_cong.groupby("disabledLink", as_index=False)["Result_num"].mean().sort_values("Result_num", ascending=False)
    plt.figure(figsize=(9, 5))
    labels = [LINK_LABELS.get(int(x), str(x)) for x in link_rank["disabledLink"]]
    plt.bar(labels, link_rank["Result_num"])
    plt.xlabel("Disabled link")
    plt.ylabel("Average travel time across nodes/congestion (min)")
    plt.title("Disabled-link criticality ranking")
    plt.xticks(rotation=70, ha="right")
    plt.grid(axis="y", alpha=0.3)
    savefig("fig12_disabled_link_ranking")

    # Baseline heatmap without congestion dimension.
    heat = df_base.pivot_table(index="disabledNode", columns="disabledLink", values="Result_num", aggfunc="mean")
    plt.figure(figsize=(11, 5.5))
    im = plt.imshow(heat.values, aspect="auto")
    plt.colorbar(im, label="Travel time (min)")
    plt.xticks(range(len(heat.columns)), [LINK_LABELS.get(int(c), str(c)) for c in heat.columns], rotation=90)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("Disabled link")
    plt.ylabel("Disabled node")
    plt.title("Baseline resilience heatmap")
    savefig("fig13_baseline_resilience_heatmap")


# ---------------------------------------------------------------------
# Plot group 6: Random Forest surrogate feature importance
# ---------------------------------------------------------------------
def plot_random_forest_feature_importance(datasets: dict[str, pd.DataFrame]) -> None:
    if not SKLEARN_AVAILABLE:
        print("scikit-learn is not installed. Skipping Random Forest feature-importance plots.")
        return

    # Combine compatible scalar-output experiments. Missing columns are filled with 0.
    scalar_frames = []
    for name in ["infrastructure_time", "incidents", "delays", "resilience_congestion"]:
        df = datasets[name].copy()
        if "Result_num" not in df.columns:
            continue
        df = finite_only(df)
        df["experiment"] = name
        scalar_frames.append(df)

    data = pd.concat(scalar_frames, ignore_index=True, sort=False)
    y = data["Result_num"]

    feature_cols = [
        "allStations", "lanes4Bus", "pCongFixed",
        "incidentNode", "incidentType",
        "delayNode", "delayInMin", "p_car_delay", "p_bus_delay", "p_train_delay", "p_cycle_delay",
        "disabledNode", "disabledLink",
    ]
    X = data.reindex(columns=feature_cols).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    scores = pd.DataFrame({
        "metric": ["R2", "MAE"],
        "value": [r2_score(y_test, pred), mean_absolute_error(y_test, pred)],
    })
    scores.to_csv(OUT_DIR / "rf_scores.csv", index=False)

    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=True)
    imp.to_csv(OUT_DIR / "rf_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(imp["feature"], imp["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Feature importance learned by Random Forest surrogate")
    plt.grid(axis="x", alpha=0.3)
    savefig("fig14_rf_feature_importance")

    # Predicted vs actual plot.
    plt.figure(figsize=(5.8, 5.8))
    plt.scatter(y_test, pred, alpha=0.45, s=18)
    lo = min(y_test.min(), pred.min())
    hi = max(y_test.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("Actual PRISM travel time (min)")
    plt.ylabel("Predicted travel time (min)")
    plt.title("Random Forest surrogate: predicted vs actual")
    savefig("fig15_rf_predicted_vs_actual")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    datasets = {name: read_tsv(path) for name, path in FILES.items()}

    plot_infrastructure_time(datasets["infrastructure_time"])
    plot_pareto(datasets["pareto_cost"], "cost", "Travel cost", "fig16_time_cost")
    plot_pareto(datasets["pareto_emission"], "emission", "Emission", "fig17_time_emission")
    plot_incidents(datasets["incidents"])
    plot_delays(datasets["delays"])
    plot_resilience(datasets["resilience_congestion"], datasets["resilience_baseline"])
    plot_random_forest_feature_importance(datasets)

    print(f"Done. Figures saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
