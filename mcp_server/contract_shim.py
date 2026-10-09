"""
CRRA Lab C2 - Mock Contract Management API

Stands in for the real BizOps contract system so the Lab C3 agent can be run
again and again without touching production data.

Run from the project root:  python mcp_server/contract_shim.py
"""

import csv
from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "contracts.csv"

# Everyone analyses the portfolio against the same fixed "today", so results
# are identical whichever day you run the lab.
SIMULATED_TODAY = date(2025, 4, 1)
EDITABLE_FIELDS = ("status", "owner", "proposed_uplift_pct")

app = Flask(__name__)


def enrich(row: dict) -> dict:
    """Convert CSV text to proper types and add the four derived fields."""
    for key in ("annual_value_inr", "notice_days", "seats_purchased",
                "seats_active", "proposed_uplift_pct"):
        row[key] = int(row[key])
    row["auto_renew"] = row["auto_renew"].strip().upper() == "Y"

    renewal = date.fromisoformat(row["renewal_date"])
    deadline = renewal - timedelta(days=row["notice_days"])
    row["notice_deadline"] = deadline.isoformat()
    row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
    row["days_to_notice_deadline"] = (deadline - SIMULATED_TODAY).days

    # EXPIRED is checked first so a lapsed contract never shows as INSIDE_WINDOW.
    if renewal < SIMULATED_TODAY:
        row["notice_state"] = "EXPIRED"
    elif deadline < SIMULATED_TODAY:
        row["notice_state"] = "INSIDE_WINDOW"      # deadline passed, renewal not yet
    elif row["days_to_notice_deadline"] <= 30:
        row["notice_state"] = "APPROACHING"        # deadline is today or within 30 days
    else:
        row["notice_state"] = "OPEN"

    # AMC and cloud-support contracts have no seats, so utilisation is unknown.
    if row["seats_purchased"] > 0:
        row["utilisation_pct"] = round(100 * row["seats_active"] / row["seats_purchased"])
    else:
        row["utilisation_pct"] = None

    value = row["annual_value_inr"]
    row["approval_band"] = "A" if value < 1_000_000 else "B" if value <= 5_000_000 else "C"
    return row


def load_contracts() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return [enrich(row) for row in csv.DictReader(f)]


CONTRACTS = load_contracts()  # loaded once, at startup


def find(contract_id: str):
    return next((c for c in CONTRACTS
                 if c["contract_id"].upper() == contract_id.upper()), None)


def not_found(contract_id: str):  # 404 as JSON, so the agent can read it
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


@app.get("/health")
def health():
    return jsonify({"status": "ok", "contracts_loaded": len(CONTRACTS),
                    "simulated_today": SIMULATED_TODAY.isoformat()})


@app.get("/api/contracts")
def list_contracts():
    """All contracts, with optional ?category= ?band= ?notice_state= filters."""
    results = CONTRACTS
    for param, field in (("category", "category"), ("band", "approval_band"),
                         ("notice_state", "notice_state")):
        wanted = request.args.get(param)
        if wanted:
            results = [c for c in results if c[field].lower() == wanted.lower()]
    return jsonify({"count": len(results), "contracts": results})


@app.get("/api/contracts/expiring")
def expiring():
    """Contracts renewing within ?days= (default 90), soonest first."""
    window = request.args.get("days", default=90, type=int)
    results = sorted((c for c in CONTRACTS if 0 <= c["days_to_renewal"] <= window),
                     key=lambda c: c["days_to_renewal"])
    return jsonify({"count": len(results), "window_days": window, "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    contract = find(contract_id)
    return jsonify(contract) if contract else not_found(contract_id)


@app.get("/api/categories")
def categories():
    """Vendors grouped by category with total annual value: the overlap view."""
    grouped: dict[str, list[dict]] = {}
    for c in CONTRACTS:
        grouped.setdefault(c["category"], []).append(
            {key: c[key] for key in ("contract_id", "vendor",
                                     "annual_value_inr", "utilisation_pct")})
    summary = [{"category": cat, "vendor_count": len(items),
                "total_annual_value_inr": sum(i["annual_value_inr"] for i in items),
                "vendors": items}
               for cat, items in sorted(grouped.items())]
    return jsonify({"count": len(summary), "categories": summary})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """Update status, owner or proposed_uplift_pct. In memory only: a restart resets it."""
    if not (contract := find(contract_id)):
        return not_found(contract_id)
    payload = request.get_json(silent=True) or {}
    changes = {k: v for k, v in payload.items() if k in EDITABLE_FIELDS}
    if not changes:
        return jsonify({"error": f"Send JSON with any of: {', '.join(EDITABLE_FIELDS)}"}), 400
    if "proposed_uplift_pct" in changes:
        try:
            changes["proposed_uplift_pct"] = float(changes["proposed_uplift_pct"])
        except (TypeError, ValueError):
            return jsonify({"error": "proposed_uplift_pct must be a number"}), 400
    contract.update(changes)
    return jsonify({"updated": True, "contract": contract})


if __name__ == "__main__":
    print("=" * 60)
    print("  Mock Contract API")
    print(f"  {len(CONTRACTS)} contracts loaded from {CSV_PATH.name}")
    print(f"  Simulated today: {SIMULATED_TODAY.isoformat()}   http://localhost:5001/health")
    print("=" * 60)
    app.run(port=5001, debug=False)
