"""
CRRA Lab C3 - Renewal Analysis Agent

A tool-calling agent that reviews one vendor contract at a time and submits a
RENEW / RENEGOTIATE / CONSOLIDATE / TERMINATE recommendation with a policy
citation.

Before running:
  1. Tab 1: python mcp_server/contract_shim.py   (leave it running)
  2. Put ANTHROPIC_API_KEY in .env (or set it in the terminal)

Run from the project root (Tab 2):
    python agents/renewal_agent.py
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
import requests

# Python only puts this script's own folder on the import path. Add the project
# root so we can reuse the chunking code from Lab C1.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.kb_setup import COLLECTION_NAME, KB_DIR, chunk_markdown  # noqa: E402

try:  # load ANTHROPIC_API_KEY from .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

API_BASE = "http://localhost:5001"
MODEL = "claude-opus-5"
MAX_ROUNDS = 5  # hard stop so the agent can never loop forever
# A deliberate spread (same as the lab guide): healthy renewal, high uplift,
# low utilisation with an overlap, an orphaned contract, and a Band C contract
# inside its notice window.
TEST_CONTRACTS = ["CTR-1003", "CTR-1012", "CTR-1005", "CTR-1006", "CTR-1004"]

SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps procurement.
You review ONE vendor contract and recommend exactly one action:
RENEW, RENEGOTIATE, CONSOLIDATE or TERMINATE.

How to work, in order:
1. Call get_contract to get the facts. Never guess contract data. If it
   returns an error, do not submit a recommendation: explain the problem in
   plain text and stop.
2. Call search_policy for the governing rule. Call it at most twice.
3. Call find_category_overlap only if you are considering CONSOLIDATE.
4. Finish by calling submit_recommendation exactly once.

Decision guidance:
- Utilisation below 40% with a viable overlapping vendor points to CONSOLIDATE.
- Utilisation below 20% with no business owner (UNASSIGNED) is NOT an automatic
  TERMINATE. An absent owner means nobody has confirmed the capability is
  unneeded, so escalate to a human.
- Proposed uplift above 15% is never accepted at first offer: RENEGOTIATE.
- High utilisation with a modest uplift is a healthy RENEW.
- TERMINATE only where the capability itself is no longer required.

Confidence:
- HIGH: the numbers and the policy point the same way with no ambiguity.
- MEDIUM: sound, but rests on an assumption you must name in the rationale.
- LOW: genuinely unclear. LOW confidence is a valid and useful answer. Say so
  rather than inventing certainty, and do not keep calling tools hoping for a
  cleaner picture.

Set human_approval_required to true whenever policy demands it: approval
Bands B and C, anything INSIDE_WINDOW, and every TERMINATE.

estimated_annual_impact_inr is a rough rupee change per year versus renewing
as quoted: negative means a saving, 0 for RENEW at existing terms.

Always cite the specific policy file and section, e.g.
"auto_renewal_rules.md - Escalation trigger"."""

TOOLS = [
    {
        "name": "get_contract",
        "description": "Fetch one contract from the contract API, including derived "
                       "fields: approval_band, notice_state, notice_deadline, "
                       "utilisation_pct (null when the contract has no seats).",
        "input_schema": {
            "type": "object",
            "properties": {"contract_id": {"type": "string", "description": "e.g. CTR-1004"}},
            "required": ["contract_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search the BizOps procurement policy knowledge base. Returns the "
                       "best matching section from each of the top 2 policy files, with "
                       "source, section heading, confidence (0-1) and text.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A plain-English policy question"}},
            "required": ["query"],
        },
    },
    {
        "name": "find_category_overlap",
        "description": "List every vendor in a category with its annual value and "
                       "utilisation, plus the category total. Use before recommending CONSOLIDATE.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "e.g. Observability"}},
            "required": ["category"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Submit the final recommendation for the contract. Call exactly once, last.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {"type": "string",
                                   "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {"type": "string"},
                "policy_citation": {"type": "string",
                                    "description": "e.g. approval_thresholds.md - Annual value bands"},
                "estimated_annual_impact_inr": {"type": "integer",
                                                "description": "Negative means a saving"},
                "human_approval_required": {"type": "boolean"},
            },
            "required": ["contract_id", "recommendation", "confidence", "rationale",
                         "policy_citation", "estimated_annual_impact_inr",
                         "human_approval_required"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------- tools ----

def build_policy_collection():
    """Lab C1's KB lives in memory, so rebuild it in this process (takes a second)."""
    client = chromadb.Client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for md_file in sorted(KB_DIR.glob("*.md")):
        for c in chunk_markdown(md_file.read_text(encoding="utf-8"), md_file.name):
            ids.append(c["id"])
            docs.append(c["document"])
            metas.append(c["metadata"])
    collection.add(ids=ids, documents=docs, metadatas=metas)
    return collection


POLICY = build_policy_collection()


def api_get(path: str) -> dict:
    """GET from the contract API; return an error dict instead of raising."""
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=5)
    except requests.exceptions.RequestException:
        return {"error": "Contract API unreachable at http://localhost:5001. Start it in "
                         "another terminal with: python mcp_server/contract_shim.py"}
    try:
        return resp.json()
    except ValueError:
        return {"error": f"Contract API returned HTTP {resp.status_code} with no JSON. "
                         "Is another program using port 5001?"}


def get_contract(contract_id: str) -> dict:
    return api_get(f"/api/contracts/{contract_id}")


def search_policy(query: str) -> dict:
    res = POLICY.query(query_texts=[query], n_results=8)
    best_per_file: dict[str, dict] = {}
    # Results come back best-first, so the first hit seen for a file is its best section
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if meta["source"] not in best_per_file:
            best_per_file[meta["source"]] = {
                "source": meta["source"],
                "section": meta["heading"],
                "confidence": round(1 - dist, 2),
                "text": doc,
            }
    return {"query": query, "results": list(best_per_file.values())[:2]}


def find_category_overlap(category: str) -> dict:
    data = api_get("/api/categories")
    if "error" in data:
        return data
    for entry in data.get("categories", []):
        if entry["category"].lower() == category.lower():
            return entry
    known = [c["category"] for c in data.get("categories", [])]
    return {"error": f"No category named '{category}'. Known categories: {known}"}


def run_tool(name: str, args: dict) -> dict:
    if name == "get_contract":
        return get_contract(args["contract_id"])
    if name == "search_policy":
        return search_policy(args["query"])
    if name == "find_category_overlap":
        return find_category_overlap(args["category"])
    return {"error": f"Unknown tool {name}"}


# ---------------------------------------------------------------- agent ----

def first_text(response) -> str:
    """Return the first text block. A thinking block may come before it."""
    for block in response.content:
        if getattr(block, "text", None):
            return block.text
    return ""


def analyse_contract(client: anthropic.Anthropic, contract_id: str) -> dict:
    messages = [{"role": "user",
                 "content": f"Review contract {contract_id} and submit your recommendation."}]

    for round_no in range(1, MAX_ROUNDS + 1):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},
            # If a safety classifier declines, retry on Anthropic's recommended model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        print(f"    round {round_no}: stop_reason={response.stop_reason}")

        if response.stop_reason == "refusal":
            return {"contract_id": contract_id, "recommendation": "NO DECISION",
                    "confidence": "-", "human_approval_required": True,
                    "rationale": "Model declined the request."}

        # Keep the full content (including thinking blocks) in the history
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            # Stopped without submitting (e.g. the contract does not exist)
            text = first_text(response)
            print(f"    agent said: {text[:300]}")
            return {"contract_id": contract_id, "recommendation": "NO DECISION",
                    "confidence": "-", "human_approval_required": True,
                    "rationale": text or "Model stopped without a recommendation."}

        results = []
        for tu in tool_uses:
            if tu.name == "submit_recommendation":
                return dict(tu.input)
            result = run_tool(tu.name, tu.input)
            print(f"    -> {tu.name}({json.dumps(tu.input)})"
                  + (f"  ERROR: {result['error']}" if "error" in result else ""))
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": json.dumps(result)})
        messages.append({"role": "user", "content": results})  # all results in one message

    return {"contract_id": contract_id, "recommendation": "NO DECISION",
            "confidence": "-", "human_approval_required": True,
            "rationale": f"Stopped after MAX_ROUNDS={MAX_ROUNDS} without a recommendation."}


def enforce_approval_rule(rec: dict) -> dict:
    """Code-side guardrail: never trust the model alone on when a human must approve."""
    if rec.get("human_approval_required") or rec["recommendation"] == "NO DECISION":
        return rec
    contract = get_contract(rec["contract_id"])
    reasons = []
    if contract.get("approval_band") in ("B", "C"):
        reasons.append(f"Band {contract['approval_band']}")
    if contract.get("notice_state") == "INSIDE_WINDOW":
        reasons.append("inside notice window")
    if rec["recommendation"] == "TERMINATE":
        reasons.append("termination")
    if reasons:
        rec["human_approval_required"] = True
        rec["rationale"] += f" [Approval forced by code: {', '.join(reasons)}.]"
    return rec


def main() -> None:
    if "error" in api_get("/health"):
        raise SystemExit("Contract API is not running. In another terminal run:\n"
                         "    python mcp_server/contract_shim.py")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Add it to .env in the project root.")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    results = []
    for contract_id in TEST_CONTRACTS:
        print(f"\nAnalysing {contract_id}")
        rec = enforce_approval_rule(analyse_contract(client, contract_id))
        results.append(rec)
        print(f"    {rec['recommendation']} ({rec['confidence']}): {rec['rationale'][:300]}")

    print("\n" + "=" * 96)
    print(f"{'Contract':<10} {'Recommendation':<15} {'Conf':<7} {'Approval':<9} "
          f"{'Impact INR':>12}  Policy citation")
    print("-" * 96)
    for r in results:
        impact = r.get("estimated_annual_impact_inr")
        impact_txt = f"{impact:>12,.0f}" if isinstance(impact, (int, float)) else f"{'-':>12}"
        approval = "YES" if r.get("human_approval_required") else "no"
        print(f"{r['contract_id']:<10} {r['recommendation']:<15} {r['confidence']:<7} "
              f"{approval:<9} {impact_txt}  {r.get('policy_citation', '-')[:40]}")
    print("-" * 96)
    needs_human = sum(1 for r in results if r.get("human_approval_required"))
    print(f"{len(results)} analysed, {needs_human} need human approval before action")
    print("=" * 96)


if __name__ == "__main__":
    main()
