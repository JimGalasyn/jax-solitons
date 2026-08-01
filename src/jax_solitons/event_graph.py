"""event_graph — one causal event graph behind scene / calorimeter / ECS views.

Ported from the retired program's engine dogfood layer (event_graph.py) and
generalized: the attribute namespace, conserved-charge keys, radiated-receipt
sector, and label->PDG resolution are all injected instead of hardcoded, so the
graph serves any model/preset rather than one theory's ledger.

Three roles on a single HepMC3-shaped causal graph:

    persistent structure   -> Particle (edge / world-line)
    interaction event      -> Vertex   (node)
    radiated quanta        -> outgoing Particle (status 1, the "receipt")
    conserved charge vector-> particle charge attributes ({ns}.*)
    conservation law       -> per-vertex closure  ==  calorimeter check
    proper-time TICK       -> per-particle {ns}.ticks (volume ledger = sum ticks)

Roles on the same object:
    EventGraph.from_scene(scene)      seeds the in-state          (scene staging)
    EventGraph.from_ecs_trace(trace)  appends events + receipts   (ECS recorder)
    EventGraph.check_conservation()   per-vertex charge residual  (calorimeter)
    EventGraph.to_hepmc3()            HepMC3 Asciiv3 + {ns}.*      (interop)

Dependency-light: stdlib only. The ECS adapter reads a trace's Node/Edge
attributes duck-typed; charge resolution is an injected callback so callers'
physics stays an optional dependency, not ours.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# defaults — override per-instance for a preset's own ledger
DEFAULT_CHARGE_KEYS = ("E", "Q", "B", "f", "logdet", "Lk")
DEFAULT_RADIATED_KEYS = ("E", "f")     # massless sector carried by receipts
DEFAULT_RECEIPT_PDG = {"GAMMA": 22, "NU": 12}
PDG_PRIVATE = 99000000                 # PDG private/user range for composites


@dataclass
class Particle:  # reconcile: allow-dup — HepMC3-style event-record particle
    #                (world-line), NOT the nwt_substrate physics Particle;
    #                genuine name clash per the gate's protocol.
    pid: int
    pdg: int
    status: int                              # 4 incoming, 1 final, 2 intermediate
    charges: dict = field(default_factory=dict)   # subset of the graph's charge_keys
    attrs: dict = field(default_factory=dict)     # {ns}.*: labels, ticks, ...
    momentum: tuple = (0.0, 0.0, 0.0)
    prod_vtx: int | None = None
    end_vtx: int | None = None


@dataclass
class Vertex:
    vid: int
    kind: str                                # COMPOSE|DECAY|RECONNECT|CREATE|...|STAGE
    incoming: list = field(default_factory=list)   # particle ids
    outgoing: list = field(default_factory=list)   # particle ids
    handedness: int = 0
    attrs: dict = field(default_factory=dict)
    position: tuple = (0.0, 0.0, 0.0, 0.0)


class EventGraph:
    def __init__(self, name="event", universe=None, frame="cm", ns="eg",
                 charge_keys=DEFAULT_CHARGE_KEYS,
                 radiated_keys=DEFAULT_RADIATED_KEYS,
                 receipt_pdg=None, pdg_of=None):
        self.name = name
        self.universe = universe
        self.frame = frame
        self.ns = ns
        self.charge_keys = tuple(charge_keys)
        self.radiated_keys = tuple(radiated_keys)
        self.receipt_pdg = dict(DEFAULT_RECEIPT_PDG if receipt_pdg is None
                                else receipt_pdg)
        self.pdg_of = pdg_of or (lambda label, charges: PDG_PRIVATE)
        self.particles: dict[int, Particle] = {}
        self.vertices: dict[int, Vertex] = {}
        self._pid = 0
        self._vid = 0

    def _k(self, key):
        return f"{self.ns}.{key}"

    # ---- construction ----
    def add_particle(self, pdg, status, charges=None, attrs=None, momentum=(0, 0, 0)):
        self._pid += 1
        self.particles[self._pid] = Particle(
            self._pid, int(pdg), int(status), dict(charges or {}),
            dict(attrs or {}), tuple(momentum))
        return self._pid

    def add_vertex(self, kind, incoming, outgoing, handedness=0, attrs=None):
        self._vid += 1
        v = Vertex(self._vid, kind, list(incoming), list(outgoing),
                   handedness, dict(attrs or {}))
        for pid in incoming:
            self.particles[pid].end_vtx = self._vid   # beams keep status 4
        for pid in outgoing:
            self.particles[pid].prod_vtx = self._vid
        self.vertices[self._vid] = v
        return self._vid

    # ---- view 1: scene in-state ----
    @classmethod
    def from_scene(cls, scene, attr_keys=("knot", "framing", "chirality", "core"),
                   **kwargs):
        g = cls(scene.get("scene", "scene"),
                universe=scene.get("universe"), frame=scene.get("frame", "cm"),
                **kwargs)
        pids = []
        for a in scene["actors"]:
            top = a.get("topology", {})
            charges = {k: top[k] for k in g.charge_keys if k in top}
            attrs = {g._k(k): top[k] for k in attr_keys if k in top}
            if "catalog" in a:
                attrs[g._k("catalog")] = a["catalog"]
            # carry the staged boost into the 3-momentum so a collide scene's
            # HepMC view isn't zeroed out
            pids.append(g.add_particle(a["pdg"], 4, charges, attrs,
                                       momentum=_scene_momentum(a, top)))
        # one staging vertex; outgoing is filled once the run produces products
        g.add_vertex(scene.get("stage", {}).get("interaction", "STAGE").upper(),
                     incoming=pids, outgoing=[])
        return g

    # ---- view 3: ECS committed trace ----
    @classmethod
    def from_ecs_trace(cls, trace, charges_of=None, name="ecs", **kwargs):
        """Duck-typed over an ECS trace: nodes carry .kind/.parents/.children/
        .handedness, edges carry .kind/.dE/.df. TICK -> {ns}.ticks on the chain
        particle; interaction node -> a Vertex with children + its receipt (one
        edge per interaction, zipped in order). charges_of(name)->dict resolves
        a chain label's charge vector."""
        charges_of = charges_of or (lambda n: {})
        g = cls(name, **kwargs)
        nodes = list(trace.nodes)
        edges = list(trace.edges)
        live: dict[str, int] = {}          # chain label -> current particle id

        def ensure(label):
            if label not in live:
                ch = charges_of(label)
                live[label] = g.add_particle(g.pdg_of(label, ch), 4, ch,
                                             {g._k("chain"): label,
                                              g._k("ticks"): 0})
            return live[label]

        inter = [n for n in nodes if n.kind != "TICK"]
        if len(inter) != len(edges):       # should be equal; be loud, not wrong
            g._warn = f"edge/interaction mismatch: {len(edges)} edges, {len(inter)} events"
        ei = 0
        for n in nodes:
            if n.kind == "TICK":
                pid = ensure(n.parents[0])
                p = g.particles[pid]
                p.attrs[g._k("ticks")] = p.attrs.get(g._k("ticks"), 0) + 1
                continue
            in_pids = [ensure(p) for p in n.parents]
            out_pids = []
            for c in n.children:
                ch = charges_of(c)
                cid = g.add_particle(g.pdg_of(c, ch), 2, ch,
                                     {g._k("chain"): c, g._k("ticks"): 0})
                live[c] = cid
                out_pids.append(cid)
            # the receipt (radiated quanta) — status 1 final. Its charge is the
            # RESIDUAL (sum_in - sum_children) restricted to the radiated sector,
            # so the vertex closes by construction for EVERY channel. The ECS
            # edge's stated dE/df are kept as provenance and cross-checked.
            if ei < len(edges):
                e = edges[ei]; ei += 1
                resid = {}
                for k in g.radiated_keys:
                    cin = sum(g.particles[p].charges.get(k, 0.0) for p in in_pids)
                    cout = sum(g.particles[p].charges.get(k, 0.0) for p in out_pids)
                    r = cin - cout
                    if abs(r) > 1e-9:
                        resid[k] = r
                for k, ek in (("E", "dE"), ("f", "df")):
                    if k not in g.radiated_keys:
                        continue
                    stated = float(getattr(e, ek, 0.0))
                    if abs(stated - resid.get(k, 0.0)) > 1e-6:
                        g._warn = (getattr(g, "_warn", "") + f" edge {e.kind} "
                                   f"{ek}={stated:g} != residual {k}={resid.get(k, 0.0):g};")
                rid = g.add_particle(g.receipt_pdg.get(e.kind, 0), 1, resid,
                                     {g._k("receipt"): e.kind,
                                      g._k("edge_dE"): float(getattr(e, "dE", 0.0)),
                                      g._k("edge_df"): float(getattr(e, "df", 0.0))})
                out_pids.append(rid)
            g.add_vertex(n.kind, in_pids, out_pids,
                         handedness=getattr(n, "handedness", 0))
        # HepMC status: initial chains stay beam(4); a produced particle never
        # consumed by a later vertex is a final product(1), else intermediate(2)
        for p in g.particles.values():
            if p.status == 2 and p.end_vtx is None:
                p.status = 1
        return g

    # ---- role: the calorimeter (per-vertex charge closure) ----
    def check_conservation(self, tol=1e-6):
        """For each vertex: sum(charges_in) - sum(charges_out) per key — the
        calorimeter closure. Returns {vid: {key: residual}}; keys absent on a
        particle count as 0 (partial ledgers still checkable)."""
        out = {}
        for vid, v in self.vertices.items():
            if not v.outgoing:            # a bare in-state (scene) vertex: nothing to close
                continue
            res = {}
            for k in self.charge_keys:
                cin = sum(self.particles[p].charges.get(k, 0.0) for p in v.incoming)
                cout = sum(self.particles[p].charges.get(k, 0.0) for p in v.outgoing)
                r = cin - cout
                if abs(r) > tol:
                    res[k] = r
            out[vid] = res
        return out

    def volume(self):
        """P-even ledger = sum {ns}.ticks over all particles."""
        k = self._k("ticks")
        return sum(p.attrs.get(k, 0) for p in self.particles.values())

    def net_handedness(self):
        return sum(v.handedness for v in self.vertices.values())

    # ---- interop: HepMC3 Asciiv3 ----
    def to_hepmc3(self):
        ns = self.ns
        lines = ["HepMC::Version 3.02.06", "HepMC::Asciiv3-START_EVENT_LISTING",
                 f"E 0 {len(self.vertices)} {len(self.particles)}", "U GEV MM"]
        if self.universe is not None:
            lines.append(f"A 0 {ns}.universe {json.dumps(self.universe, separators=(',', ':'))}")
        lines.append(f"A 0 {ns}.scene {self.name}")
        lines.append(f"A 0 {ns}.frame {self.frame}")
        lines.append(f"A 0 {ns}.volume {self.volume()}")
        for vid, v in self.vertices.items():
            inref = ",".join(str(p) for p in v.incoming)
            lines.append(f"V -{vid} 0 [{inref}] @ 0 0 0 0")
            if v.kind:
                lines.append(f"A -{vid} {ns}.event {v.kind}")
        for pid, p in self.particles.items():
            px, py, pz = p.momentum
            echg = float(p.charges.get("E", 0.0))
            massless = (p.attrs.get(self._k("receipt")) is not None) or p.pdg in (22,)
            m = 0.0 if massless else echg               # E charge = rest energy
            psq = px * px + py * py + pz * pz
            e = (psq + m * m) ** 0.5
            if massless and psq == 0.0:                 # receipt energy lives in its E charge
                e = echg
            prod = f"-{p.prod_vtx}" if p.prod_vtx else "0"   # Asciiv3: production-vertex ref
            lines.append(f"P {pid} {prod} {p.pdg} {px:.6g} {py:.6g} {pz:.6g} {e:.6g} {m:.6g} {p.status}")
            for k, val in p.charges.items():
                lines.append(f"A {pid} {ns}.{k} {val}")
            for k, val in p.attrs.items():
                lines.append(f"A {pid} {k} {val}")
        lines.append("HepMC::Asciiv3-END_EVENT_LISTING")
        return "\n".join(lines) + "\n"


def _scene_momentum(actor, top):
    """Staged 3-momentum from an actor's kinematics: boost.speed is the
    authoritative Galilean speed; energy is a flagged toy (u=sqrt(2KE/m))."""
    kin = actor.get("kinematics")
    if not kin:
        return (0.0, 0.0, 0.0)
    m = float(top.get("E", top.get("rest_energy", 1.0)) or 1.0)
    if "boost" in kin:
        u = float(kin["boost"]["speed"])
    else:
        u = (2.0 * float(kin["energy"]) / m) ** 0.5
    d = kin.get("direction") or [0.0, 0.0, 0.0]
    nrm = sum(c * c for c in d) ** 0.5 or 1.0
    return tuple(m * u * c / nrm for c in d)
