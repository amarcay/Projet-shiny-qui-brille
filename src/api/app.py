"""
Flask product app for the GBP/USD trading system.
Exposes the best model from registry.json with 4 pages:
Home, Performance, Signal (live), Technology.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, send_from_directory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

FASTAPI_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
TEMPLATES_DIR = BASE_DIR / "src" / "app" / "templates"
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Static file serving for report images
# ---------------------------------------------------------------------------
@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_csv(path, **kwargs):
    """Read a CSV if it exists, else return empty DataFrame."""
    if Path(path).exists():
        return pd.read_csv(path, **kwargs)
    return pd.DataFrame()


def _read_json(path):
    """Read a JSON file if it exists."""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _get_best_model():
    """Read registry.json and return the best model info (pluggable)."""
    registry = _read_json(MODELS_DIR / "registry.json")
    best_ver = registry.get("best_version", "v1")
    return registry.get("models", {}).get(best_ver, {})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    model = _get_best_model()
    test_metrics = model.get("metrics", {}).get("test_2025_2026", {})

    sim = _read_csv(REPORTS_DIR / "simulation" / "simulation_10k_history.csv")

    kpis = {
        "model_name": model.get("model_name", "N/A"),
        "version": model.get("version", "N/A"),
        "sharpe": round(test_metrics.get("sharpe", 0), 2),
        "profit_factor": round(test_metrics.get("profit_factor", 0), 4),
        "n_trades": test_metrics.get("n_trades", 0),
    }

    chart_json = ""
    if not sim.empty:
        kpis["capital_final"] = round(sim["capital"].iloc[-1], 2)
        kpis["max_drawdown"] = round(
            (sim["capital"] - sim["capital"].cummax()).min(), 2
        )

        import plotly.graph_objects as go

        ts = sim["timestamp"]
        cap = sim["capital"]
        baseline = 10000
        cap_above = np.clip(cap, baseline, None)
        cap_below = np.clip(cap, None, baseline)

        fig = go.Figure()
        # Baseline at 10k (invisible, used as fill reference)
        fig.add_trace(go.Scatter(
            x=ts, y=[baseline] * len(ts),
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False, hoverinfo="skip",
        ))
        # Green area (above 10k)
        fig.add_trace(go.Scatter(
            x=ts, y=cap_above,
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tonexty", fillcolor="rgba(0,212,170,0.25)",
            showlegend=False, hoverinfo="skip",
        ))
        # Baseline again for red fill reference
        fig.add_trace(go.Scatter(
            x=ts, y=[baseline] * len(ts),
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False, hoverinfo="skip",
        ))
        # Red area (below 10k)
        fig.add_trace(go.Scatter(
            x=ts, y=cap_below,
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tonexty", fillcolor="rgba(255,77,106,0.25)",
            showlegend=False, hoverinfo="skip",
        ))
        # Main capital line
        fig.add_trace(go.Scatter(
            x=ts, y=cap,
            mode="lines", name="Capital",
            line=dict(color="#e6edf3", width=2),
        ))
        fig.add_hline(y=baseline, line_dash="dash", line_color="#555", line_width=1)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#000",
            plot_bgcolor="#000",
            margin=dict(l=50, r=20, t=40, b=40),
            title="Simulation 10 000 EUR - 2025-2026 (Test)",
            yaxis_title="Capital (EUR)",
            yaxis_range=[8000, 12000],
            xaxis_title="",
            height=420,
        )
        chart_json = fig.to_json()

    return render_template("home.html", kpis=kpis, chart_json=chart_json)


@app.route("/performance")
def performance():
    model = _get_best_model()
    model_name = model.get("model_name", "N/A")
    strategy_name = model.get("eval_strategy_name", "RL (DQN)")

    sim = _read_csv(REPORTS_DIR / "simulation" / "simulation_10k_history.csv")
    ev = _read_csv(REPORTS_DIR / "evaluation" / "evaluation_metrics.csv")

    chart_json = ""
    monthly = []
    metrics = {}

    if not sim.empty:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        ts = sim["timestamp"]
        cap = sim["capital"]
        baseline = 10000
        cap_above = np.clip(cap, baseline, None)
        cap_below = np.clip(cap, None, baseline)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
            vertical_spacing=0.06,
            subplot_titles=("Capital (EUR)", "Position"),
        )

        # Baseline (invisible, fill reference)
        fig.add_trace(go.Scatter(
            x=ts, y=[baseline] * len(ts),
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        # Green area (above 10k)
        fig.add_trace(go.Scatter(
            x=ts, y=cap_above,
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tonexty", fillcolor="rgba(0,212,170,0.25)",
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        # Baseline again for red fill
        fig.add_trace(go.Scatter(
            x=ts, y=[baseline] * len(ts),
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        # Red area (below 10k)
        fig.add_trace(go.Scatter(
            x=ts, y=cap_below,
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tonexty", fillcolor="rgba(255,77,106,0.25)",
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        # Main capital line
        fig.add_trace(go.Scatter(
            x=ts, y=cap,
            mode="lines", name="Capital",
            line=dict(color="#e6edf3", width=2),
            showlegend=False,
        ), row=1, col=1)
        fig.add_hline(y=baseline, line_dash="dash", line_color="#555",
                       line_width=1, row=1, col=1)

        pos_map = {"HOLD": 0, "BUY": 1, "SELL": -1}
        fig.add_trace(go.Scatter(
            x=ts, y=sim["action"].map(pos_map),
            mode="markers", name="Action",
            marker=dict(
                size=2,
                color=sim["action"].map(
                    {"BUY": "#00d4aa", "SELL": "#ff4d6a", "HOLD": "#555"}
                ),
            ),
            showlegend=False,
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#000",
            plot_bgcolor="#000",
            margin=dict(l=50, r=20, t=40, b=40),
            height=560, showlegend=False,
        )
        fig.update_yaxes(range=[8000, 12000], row=1, col=1)
        chart_json = fig.to_json()

        # Monthly summary
        grp = (
            sim.groupby("month")
            .agg(
                capital_end=("capital", "last"),
                pnl=("pnl_eur", "sum"),
                trades=("action", lambda x: (x != "HOLD").sum()),
            )
            .reset_index()
        )
        monthly = grp.to_dict("records")

        # Metrics
        metrics["capital_final"] = round(sim["capital"].iloc[-1], 2)
        metrics["pnl_total"] = round(sim["capital"].iloc[-1] - 10000, 2)
        metrics["max_capital"] = round(sim["capital"].max(), 2)
        metrics["min_capital"] = round(sim["capital"].min(), 2)
        metrics["max_dd_eur"] = round(
            (sim["capital"] - sim["capital"].cummax()).min(), 2
        )
        metrics["nb_trades"] = int((sim["action"] != "HOLD").sum())
        metrics["total_costs"] = round(sim["trade_cost"].sum(), 2)

    # Comparison vs Buy & Hold (2024 Test only)
    comparison = []
    if not ev.empty:
        test_rows = ev[ev["Split"] == "2025 & 2026 (Test)"]
        for _, row in test_rows.iterrows():
            strat = row["Strategie"] if "Strategie" in row.index else row.get("Strat\u00e9gie", "")
            if strat in (strategy_name, "Buy & Hold"):
                comparison.append(row.to_dict())

    return render_template(
        "performance.html",
        model_name=model_name,
        chart_json=chart_json,
        monthly=monthly,
        metrics=metrics,
        comparison=comparison,
    )


@app.route("/signal", methods=["GET", "POST"])
def signal():
    model = _get_best_model()
    features = model.get("features", [])

    result = None
    error = None
    model_info = None

    # Try to get model info from FastAPI
    try:
        resp = requests.get(f"{FASTAPI_URL}/model/info", timeout=3)
        if resp.status_code == 200:
            model_info = resp.json()
    except requests.exceptions.ConnectionError:
        pass

    if request.method == "POST":
        try:
            feature_values = {}
            for f in features:
                val = request.form.get(f, "")
                feature_values[f] = float(val) if val else 0.0

            resp = requests.post(
                f"{FASTAPI_URL}/predict",
                json=feature_values,
                timeout=5,
            )
            if resp.status_code == 200:
                result = resp.json()
            else:
                error = f"API error ({resp.status_code}): {resp.text}"
        except requests.exceptions.ConnectionError:
            error = "API not available. Start the FastAPI server on port 8000."
        except ValueError as e:
            error = f"Invalid input: {e}"

    return render_template(
        "signal.html",
        features=features,
        result=result,
        error=error,
        model_info=model_info,
        model=model,
    )


@app.route("/technology")
def technology():
    model = _get_best_model()
    return render_template("technology.html", model=model)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
