"""Governed fleet campaign: POLICY over the campaign MECHANISM.

The campaign boundary (driver/protocols) owns the *mechanism* — config-hashed
identity + idempotent skip (A/B), event streaming (C), fleet fan-out + preemption
recovery (D), host probing (E), cloud brokering (F). This module adds the
*policy* that makes a certificate lineage trustworthy, physics-agnostically:

  - envelope **preflight BEFORE rent** (don't pay for a config that can't hold);
  - a **launch-completeness gate** (the launched config-hash set must equal the
    planned one — no silently-dropped legs);
  - **shipment verification**: product-hash sidecars + worker **engine-SHA
    attestation** against a global tag (the host attests the blob it actually
    ran, never copied from the request);
  - a typed **cut-flow** so silent attrition is structurally impossible;
  - idempotent **corpus ingest** with conflict-never-overwrite.

Everything domain-specific is INJECTED: `preflight(cfg) -> [violations]`,
`stage_validate(plan, cfg) -> [violations]`, and `ingest(record)`. A caller with
no policy gets a thin, correct wrapper around `execute_config`.

The RunFn shipment contract (what a governed `run_fn` returns)::

    {"products": {name: <json-able record>, ...},
     "sidecar":  {name: sha256(record)},
     "attested_shas": {slot: <worker-measured blob sha>}}

receipts stream through `ctx.emit` as event records (P6: records, not fields).
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

from jax_solitons.campaign.driver import execute_config
from jax_solitons.campaign.reference import FileRunRegistry, JsonlEventSink
from jax_solitons.runs import RunConfig

FLEET_STAGES = ["planned", "launch-gate", "completed", "shipped",
                "verified", "registered"]


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class CutFlow:
    """Per-leg accounting: every leg enters at stage 0; each stage records
    survivors; a drop carries a typed reason and forces every later stage
    False. `render()` shows the full waterfall — silent attrition is impossible."""

    def __init__(self, stages: list[str]):
        self.stages = stages
        self.rows: dict[str, list] = {}
        self.reasons: dict[tuple, str] = {}

    def enter(self, leg: str):
        self.rows[leg] = [None] * len(self.stages)

    def mark(self, leg: str, stage: str, ok: bool, reason: str = ""):
        i = self.stages.index(stage)
        self.rows[leg][i] = ok
        if not ok:
            self.reasons[(leg, stage)] = reason or "unspecified"
            for j in range(i + 1, len(self.stages)):
                self.rows[leg][j] = False
        else:
            self.reasons.pop((leg, stage), None)

    def table(self) -> dict:
        counts = [{"stage": s, "survivors": sum(1 for r in self.rows.values() if r[i])}
                  for i, s in enumerate(self.stages)]
        drops = [{"leg": leg, "stage": st, "reason": rs}
                 for (leg, st), rs in sorted(self.reasons.items())]
        return {"entered": len(self.rows), "waterfall": counts, "drops": drops}

    def render(self) -> str:
        t = self.table()
        out = [f"cut-flow: {t['entered']} legs entered"]
        out += [f"  {c['stage']:<24}{c['survivors']}" for c in t["waterfall"]]
        out += [f"  DROP {d['leg']} @ {d['stage']}: {d['reason']}" for d in t["drops"]]
        return "\n".join(out)


def leg_to_config(leg: dict, gtag: str, required_shas: dict) -> RunConfig:
    """A farm leg -> a campaign RunConfig. Identity = config_hash (mechanism A),
    which replaces any hand-rolled spec hash. `leg` carries at least
    {rid, cfg:{L, dx, ...}}; everything else rides in `params` verbatim."""
    cfg = leg["cfg"]
    return RunConfig(
        model=leg.get("model", "farm"),
        N=int(leg.get("N", round(cfg["L"] / cfg["dx"]))),
        L=float(cfg["L"]), seed=int(leg.get("seed", 0)),
        params={"rid": leg["rid"], "cfg": cfg, "plan": leg.get("plan", []),
                "gtag": gtag, "required_shas": required_shas,
                **{k: leg[k] for k in leg
                   if k not in ("rid", "cfg", "plan", "N", "seed", "model")}})


def launch_gate(planned: list, launched: list) -> list[str]:
    """Refuse the WHOLE launch unless the launched config-hash set equals the
    planned one — a missing leg at launch (an argparse/config slip) is caught
    before any host is billed, not discovered in a short results table."""
    p = {c.config_hash(): c.params["rid"] for c in planned}
    q = {c.config_hash(): c.params["rid"] for c in launched}
    v = []
    if set(p) - set(q):
        v.append(f"MISSING legs at launch: {sorted(p[h] for h in set(p) - set(q))}")
    if set(q) - set(p):
        v.append(f"UNPLANNED legs at launch: {sorted(q[h] for h in set(q) - set(p))}")
    return v


def verify_shipment(shipment: dict, required_shas: dict) -> list[str]:
    """Product-hash sidecars + worker engine-SHA attestation vs a global tag."""
    v = []
    for name, obj in shipment.get("products", {}).items():
        want, got = shipment.get("sidecar", {}).get(name), sha256_json(obj)
        if want != got:
            v.append(f"HASH_MISMATCH {name}: sidecar {str(want)[:12]} != shipped {got[:12]}")
    for slot, sha in required_shas.items():
        att = shipment.get("attested_shas", {}).get(slot)
        if att != sha:
            v.append(f"SHA_MISMATCH {slot}: host ran {att}, tag requires {sha}")
    return v


class FarmCampaign:
    """Chamber-style policy for one fleet campaign: plan (pre-rent) -> gate ->
    execute via the mechanism -> verify -> ingest, over a typed cut-flow.

    Injected policy (all optional; omitting one makes that gate a no-op):
      preflight(cfg)         -> [violations]  (envelope walls, before rent)
      stage_validate(plan,cfg)-> [violations] (staging geometry admissibility)
      ingest(record)         -> None          (corpus persistence; default in-mem)

    The executor is campaign seam D: `execute_leg` runs the run_fn in-process
    (execute_config) by default; pass `run=` to dispatch through an Executor /
    ProviderExecutor / FleetExecutor instead. All mechanism guarantees (A idempotent
    skip, B restart-after-death, C flushed events) are inherited unchanged."""

    def __init__(self, gtag: str, required_shas: dict, work_dir: str, *,
                 preflight: Callable[[dict], list] | None = None,
                 stage_validate: Callable[[list, dict], list] | None = None,
                 ingest: Callable[[dict], None] | None = None):
        self.gtag, self.required_shas = gtag, required_shas
        self.registry = FileRunRegistry(work_dir)        # mechanism A/B
        self.sink = JsonlEventSink()                     # mechanism C
        self.preflight = preflight or (lambda cfg: [])
        self.stage_validate = stage_validate or (lambda plan, cfg: [])
        self.ingest = ingest
        self.cf = CutFlow(FLEET_STAGES)
        self.configs: list = []
        self.ingested: dict = {}                         # rid -> content hash

    def plan(self, legs: list[dict]) -> dict:
        ok = []
        for leg in legs:
            self.cf.enter(leg["rid"])
            pv = self.preflight(leg["cfg"]) + self.stage_validate(
                leg.get("plan", []), leg["cfg"])
            self.cf.mark(leg["rid"], "planned", not pv, "; ".join(pv))
            if not pv:
                ok.append(leg_to_config(leg, self.gtag, self.required_shas))
        self.configs = ok
        return {"planned": len(legs), "valid": len(ok)}

    def gate_launch(self, launched: list) -> list[str]:
        v = launch_gate(self.configs, launched)
        for c in self.configs:
            self.cf.mark(c.params["rid"], "launch-gate", not v, "; ".join(v) if v else "")
        return v

    def execute_leg(self, config: RunConfig, run_fn, *, run=None) -> dict:
        rid = config.params["rid"]
        runner = run or (lambda: execute_config(
            config, run_fn, registry=self.registry, sink=self.sink))
        try:
            shipment = runner()
        except Exception as e:                            # preemption / host death
            self.cf.mark(rid, "completed", False, f"HOST_LOST: {e}")
            return {"rid": rid, "status": "LOST", "reason": str(e)}
        if shipment is None:                              # idempotent skip (A)
            if rid in self.ingested:
                return {"rid": rid, "status": "SKIP_OK", "reason": "already complete"}
            done = self.registry.register(config).dir / "DONE.json"
            shipment = json.loads(done.read_text()) if done.exists() else None
            if shipment is None:
                return {"rid": rid, "status": "SKIP_OK", "reason": "complete, no result"}
        self.cf.mark(rid, "completed", True)
        self.cf.mark(rid, "shipped", True)
        v = verify_shipment(shipment, self.required_shas)
        self.cf.mark(rid, "verified", not v, "; ".join(v))
        if v:
            return {"rid": rid, "status": "REJECTED", "violations": v}
        return self._ingest(rid, shipment)

    def _ingest(self, rid: str, shipment: dict) -> dict:
        content = sha256_json(shipment["products"])
        if rid in self.ingested:
            if self.ingested[rid] == content:
                return {"rid": rid, "status": "SKIP_OK",
                        "reason": "already in corpus, identical content"}
            self.cf.enter(rid + ".conflict")
            for s in FLEET_STAGES[:-1]:
                self.cf.mark(rid + ".conflict", s, True)
            self.cf.mark(rid + ".conflict", "registered", False,
                         "CONFLICT: rid in corpus with DIFFERENT content — never overwritten")
            return {"rid": rid, "status": "CONFLICT"}
        if self.ingest is not None:
            self.ingest(shipment["products"].get("record", shipment["products"]))
        self.ingested[rid] = content
        self.cf.mark(rid, "registered", True)
        return {"rid": rid, "status": "REGISTERED"}
