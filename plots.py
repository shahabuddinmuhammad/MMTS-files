import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Baseline: congestion vs minimum travel time
# ============================================================

baseline = pd.DataFrame({
    "pCongFixed": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "min_time": [32.58, 40.46, 44.50, 48.67, 61.17, 66.17]
})

plt.figure(figsize=(7, 4))
plt.plot(baseline["pCongFixed"], baseline["min_time"], marker="o")
plt.xlabel("Congestion probability")
plt.ylabel("Minimum travel time (minutes)")
plt.title("Effect of Congestion on Minimum Travel Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_congestion_time.png", dpi=300)
plt.show()


# ============================================================
# 2. Reserved bus lane comparison
# ============================================================

lanes = pd.DataFrame({
    "pCongFixed": [0.2, 0.5, 0.8],
    "without_reserved_lane": [48.8, 61.7, 97.7],
    "with_reserved_lane": [48.8, 61.7, 85.1]
})

plt.figure(figsize=(7, 4))
plt.plot(lanes["pCongFixed"], lanes["without_reserved_lane"], marker="o", label="Without reserved bus lane")
plt.plot(lanes["pCongFixed"], lanes["with_reserved_lane"], marker="s", label="With reserved bus lane")
plt.xlabel("Congestion probability")
plt.ylabel("Travel time (minutes)")
plt.title("Impact of Reserved Bus Lanes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reserved_bus_lane_comparison.png", dpi=300)
plt.show()


# ============================================================
# 3. Pareto front: time vs cost
# ============================================================

pareto = pd.DataFrame({
    "case": ["No lanes", "No lanes", "No lanes", "Reserved lanes", "Reserved lanes", "Reserved lanes"],
    "time": [61.8, 78.1, 140.2, 61.8, 67.2, 85.2],
    "cost": [40, 10, 6, 40, 15, 6]
})

plt.figure(figsize=(7, 4))
for case, group in pareto.groupby("case"):
    group = group.sort_values("cost")
    plt.plot(group["cost"], group["time"], marker="o", label=case)

plt.xlabel("Travel cost")
plt.ylabel("Travel time (minutes)")
plt.title("Pareto Front: Travel Time vs Cost")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pareto_time_cost.png", dpi=300)
plt.show()


# ============================================================
# 4. Time vs hops trade-off
# ============================================================

hops = pd.DataFrame({
    "pCongFixed": [0, 0, 0.5, 0.5, 0.9, 0.9],
    "time": [39.1, 42.1, 61.8, 75.3, 97.7, 225.3],
    "hops": [6, 5, 6, 5, 6, 5]
})

plt.figure(figsize=(7, 4))
for p, group in hops.groupby("pCongFixed"):
    plt.scatter(group["hops"], group["time"], label=f"p={p}", s=80)

plt.xlabel("Number of hops")
plt.ylabel("Travel time (minutes)")
plt.title("Travel Time vs Number of Hops")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("time_vs_hops.png", dpi=300)
plt.show()


# ============================================================
# 5. Infrastructure strategy comparison
# ============================================================

infra = pd.DataFrame({
    "strategy": ["Base", "Cycle stations everywhere", "Bus stations everywhere", "Car stations everywhere"],
    "low_congestion": [40.5, 85.0, 48.0, 32.6],
    "medium_congestion": [61.7, 85.0, 62.0, 48.7],
    "high_congestion": [97.7, 85.0, 90.0, 66.2]
})

infra_plot = infra.set_index("strategy")

infra_plot.plot(kind="bar", figsize=(9, 5))
plt.xlabel("Infrastructure strategy")
plt.ylabel("Minimum travel time (minutes)")
plt.title("Infrastructure Design Under Different Congestion Levels")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("infrastructure_strategy_comparison.png", dpi=300)
plt.show()


# ============================================================
# 6. Incident severity plot
# ============================================================

incident = pd.DataFrame({
    "incident_type": ["Normal", "Heavy traffic", "Accident", "Breakdown", "Blockage"],
    "without_reserved_lane": [48.8, 58.0, 70.5, 84.2, 160.0],
    "with_reserved_lane": [48.8, 54.0, 62.5, 73.0, 120.0]
})

plt.figure(figsize=(8, 4))
plt.plot(incident["incident_type"], incident["without_reserved_lane"], marker="o", label="Without reserved lane")
plt.plot(incident["incident_type"], incident["with_reserved_lane"], marker="s", label="With reserved lane")
plt.xlabel("Incident type")
plt.ylabel("Travel time (minutes)")
plt.title("Impact of Incident Severity on Travel Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("incident_severity_time.png", dpi=300)
plt.show()


# ============================================================
# 7. Disabled node resilience ranking
# ============================================================

disabled_nodes = pd.DataFrame({
    "node": list(range(2, 17)),
    "travel_time": [55, 52, 70, 80, 60, 68, 75, 110, 95, 65, 72, 90, 58, 77, 88]
})

disabled_nodes = disabled_nodes.sort_values("travel_time", ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(disabled_nodes["node"].astype(str), disabled_nodes["travel_time"])
plt.xlabel("Disabled node")
plt.ylabel("Travel time (minutes)")
plt.title("Resilience Analysis: Impact of Disabled Nodes")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("disabled_node_ranking.png", dpi=300)
plt.show()


# ============================================================
# 8. Disabled link resilience ranking
# ============================================================

disabled_links = pd.DataFrame({
    "link": list(range(1, 26)),
    "travel_time": [
        50, 52, 55, 60, 70,
        58, 65, 68, 75, 95,
        85, 80, 115, 78, 88,
        105, 92, 130, 76, 84,
        98, 110, 82, 100, 125
    ]
})

disabled_links = disabled_links.sort_values("travel_time", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(disabled_links["link"].astype(str), disabled_links["travel_time"])
plt.xlabel("Disabled link ID")
plt.ylabel("Travel time (minutes)")
plt.title("Resilience Analysis: Impact of Disabled Links")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("disabled_link_ranking.png", dpi=300)
plt.show()


# ============================================================
# 9. Weather heatmap
# ============================================================

import numpy as np

cloud_radius = [5, 10, 20, 30, 40, 50]
rain_severity = [0, 0.2, 0.4, 0.6, 0.8, 1]

weather_time = np.array([
    [45, 48, 51, 55, 60, 68],
    [47, 50, 54, 59, 66, 75],
    [50, 55, 61, 68, 78, 90],
    [54, 60, 68, 77, 90, 105],
    [58, 66, 75, 88, 104, 122],
    [62, 72, 84, 100, 120, 145]
])

plt.figure(figsize=(7, 5))
plt.imshow(weather_time, aspect="auto")
plt.colorbar(label="Travel time (minutes)")
plt.xticks(range(len(rain_severity)), rain_severity)
plt.yticks(range(len(cloud_radius)), cloud_radius)
plt.xlabel("Rain severity")
plt.ylabel("Cloud radius")
plt.title("Weather Impact on Travel Time")
plt.tight_layout()
plt.savefig("weather_heatmap.png", dpi=300)
plt.show()


# ============================================================
# 10. Random Forest feature importance
# ============================================================

features = pd.DataFrame({
    "feature": [
        "pCongFixed",
        "lanes4Bus",
        "incidentType",
        "incidentNode",
        "cloudRadius",
        "disabledNode",
        "disabledLink",
        "everyNodeIsStation"
    ],
    "importance": [0.31, 0.08, 0.18, 0.06, 0.12, 0.10, 0.11, 0.04]
})

features = features.sort_values("importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(features["feature"], features["importance"])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Feature Importance Learned by Random Forest Model")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=300)
plt.show()