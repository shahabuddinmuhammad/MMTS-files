import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load data
# -----------------------------
incident_file = "result10.3 time.txt"
infra_file = "result10.2.1.txt"

incident_df = pd.read_csv(incident_file, sep="\t")
infra_df = pd.read_csv(infra_file, sep="\t")

incident_df["Result"] = pd.to_numeric(incident_df["Result"], errors="coerce")
infra_df["Result"] = pd.to_numeric(infra_df["Result"], errors="coerce")

incident_df = incident_df.replace([np.inf, -np.inf], np.nan).dropna()
infra_df = infra_df.replace([np.inf, -np.inf], np.nan).dropna()

# -----------------------------
# Train ML surrogate models
# -----------------------------
rf_incident = RandomForestRegressor(
    n_estimators=600,
    random_state=42,
    n_jobs=-1
)

rf_incident.fit(
    incident_df[["incidentNode", "incidentType", "lanes4Bus", "pCongFixed"]],
    incident_df["Result"]
)

rf_infra = RandomForestRegressor(
    n_estimators=600,
    random_state=42,
    n_jobs=-1
)

rf_infra.fit(
    infra_df[["allStations", "lanes4Bus", "pCongFixed"]],
    infra_df["Result"]
)

# -----------------------------
# Congestion range
# -----------------------------
p = np.linspace(0, 1, 101)

# -----------------------------
# Generate response curves
# -----------------------------

curves = {}

# bus lanes
X = pd.DataFrame({
    "incidentNode": 2,
    "incidentType": 1,
    "lanes4Bus": 1,
    "pCongFixed": p
})
curves["bus lanes"] = rf_incident.predict(X)

# incident
X = pd.DataFrame({
    "incidentNode": 2,
    "incidentType": 5,
    "lanes4Bus": 0,
    "pCongFixed": p
})
curves["incident"] = rf_incident.predict(X)

# incident + bus lanes
X = pd.DataFrame({
    "incidentNode": 2,
    "incidentType": 5,
    "lanes4Bus": 1,
    "pCongFixed": p
})
curves["incident + bus lanes"] = rf_incident.predict(X)

# delay approximation from incident model
X = pd.DataFrame({
    "incidentNode": 4,
    "incidentType": 4,
    "lanes4Bus": 0,
    "pCongFixed": p
})
#curves["delay"] = rf_incident.predict(X)

# failure approximation
X = pd.DataFrame({
    "incidentNode": 4,
    "incidentType": 5,
    "lanes4Bus": 0,
    "pCongFixed": p
})
curves["failure"] = rf_incident.predict(X)

# infrastructure
X = pd.DataFrame({
    "allStations": 2,
    "lanes4Bus": 0,
    "pCongFixed": p
})
curves["infrastructure"] = rf_infra.predict(X)

# infrastructure + bus lanes
X = pd.DataFrame({
    "allStations": 2,
    "lanes4Bus": 1,
    "pCongFixed": p
})
curves["infrastructure + bus lanes"] = rf_infra.predict(X)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6), dpi=200)

styles = {
    "bus lanes": ("orange", "-"),
    "incident": ("green", "--"),
    "incident + bus lanes": ("red", "-"),
#    "delay": ("purple", "--"),
    "failure": ("brown", "--"),
    "infrastructure": ("deeppink", "--"),
    "infrastructure + bus lanes": ("dimgray", "-")
}

for label, y in curves.items():
    color, linestyle = styles[label]
    plt.plot(
        p,
        y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=2.5
    )

plt.xlabel(r"Congestion probability ($p_{cong}$)", fontsize=13)
plt.ylabel("ML-predicted expected time (min)", fontsize=13)

plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(frameon=False, fontsize=10, loc="upper left")

plt.xlim(0, 1.05)
plt.ylim(35, 200)

plt.tight_layout()
plt.savefig("fig_ml_surrogate_response_curves.png", dpi=300, bbox_inches="tight")
plt.savefig("fig_ml_surrogate_response_curves.pdf", bbox_inches="tight")
plt.show()