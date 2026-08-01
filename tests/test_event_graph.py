"""event_graph: the three roles (scene seed, ECS trace, calorimeter closure)."""
from dataclasses import dataclass, field

from jax_solitons.event_graph import EventGraph, PDG_PRIVATE


# ---- duck-typed ECS trace fixtures ----

@dataclass
class Node:
    kind: str
    parents: list = field(default_factory=list)
    children: list = field(default_factory=list)
    handedness: int = 0


@dataclass
class Edge:
    kind: str
    dE: float = 0.0
    df: float = 0.0


@dataclass
class Trace:
    nodes: list
    edges: list


def _charges(label):
    return {"A": {"E": 10.0, "Q": 1.0},
            "B": {"E": 4.0, "Q": 1.0},
            "C": {"E": 3.0, "Q": 0.0}}.get(label, {})


def test_manual_graph_conservation_closure():
    g = EventGraph("t", charge_keys=("E", "Q"))
    a = g.add_particle(PDG_PRIVATE, 4, {"E": 5.0, "Q": 1.0})
    b = g.add_particle(PDG_PRIVATE, 1, {"E": 3.0, "Q": 1.0})
    c = g.add_particle(22, 1, {"E": 2.0})
    g.add_vertex("DECAY", [a], [b, c])
    res = g.check_conservation()
    assert res == {1: {}}                    # closes exactly


def test_manual_graph_reports_violation():
    g = EventGraph("t", charge_keys=("E",))
    a = g.add_particle(PDG_PRIVATE, 4, {"E": 5.0})
    b = g.add_particle(PDG_PRIVATE, 1, {"E": 3.0})
    g.add_vertex("DECAY", [a], [b])
    res = g.check_conservation()
    assert abs(res[1]["E"] - 2.0) < 1e-12    # missing energy is loud


def test_ecs_trace_receipt_closes_vertex():
    # A -> B + C releases dE = 3; the receipt must carry the residual so the
    # committed vertex closes by construction.
    trace = Trace(nodes=[Node("TICK", parents=["A"]),
                         Node("DECAY", parents=["A"], children=["B", "C"])],
                  edges=[Edge("GAMMA", dE=3.0)])
    g = EventGraph.from_ecs_trace(trace, charges_of=_charges,
                                  charge_keys=("E", "Q"), radiated_keys=("E",))
    res = g.check_conservation()
    assert all(not r for r in res.values())
    assert not hasattr(g, "_warn")           # stated dE matches residual
    receipts = [p for p in g.particles.values()
                if p.attrs.get("eg.receipt") == "GAMMA"]
    assert len(receipts) == 1 and abs(receipts[0].charges["E"] - 3.0) < 1e-9
    assert g.volume() == 1                   # one TICK on A


def test_ecs_trace_cross_check_flags_mismatch():
    trace = Trace(nodes=[Node("DECAY", parents=["A"], children=["B", "C"])],
                  edges=[Edge("GAMMA", dE=999.0)])  # lies about the energy
    g = EventGraph.from_ecs_trace(trace, charges_of=_charges,
                                  charge_keys=("E", "Q"), radiated_keys=("E",))
    assert "999" in getattr(g, "_warn", "")


def test_scene_seed_and_namespace():
    scene = {"scene": "collide", "frame": "cm",
             "stage": {"interaction": "collide"},
             "actors": [
                 {"pdg": PDG_PRIVATE,
                  "topology": {"E": 2.0, "Lk": 3, "knot": "trefoil"},
                  "kinematics": {"boost": {"speed": 0.5},
                                 "direction": [1, 0, 0]}},
                 {"pdg": PDG_PRIVATE,
                  "topology": {"E": 2.0, "Lk": 3, "knot": "trefoil"},
                  "kinematics": {"boost": {"speed": 0.5},
                                 "direction": [-1, 0, 0]}}]}
    g = EventGraph.from_scene(scene, ns="zoo", charge_keys=("E", "Lk"))
    assert len(g.particles) == 2
    (v,) = g.vertices.values()
    assert v.kind == "COLLIDE" and not v.outgoing
    p = g.particles[1]
    assert p.attrs["zoo.knot"] == "trefoil"
    assert abs(p.momentum[0] - 1.0) < 1e-12          # m*u = 2.0*0.5
    # in-state-only vertex is not checkable — calorimeter skips it
    assert g.check_conservation() == {}


def test_hepmc3_output_uses_namespace():
    g = EventGraph("t", ns="zoo", charge_keys=("E",))
    a = g.add_particle(PDG_PRIVATE, 4, {"E": 1.0})
    b = g.add_particle(PDG_PRIVATE, 1, {"E": 1.0})
    g.add_vertex("DECAY", [a], [b])
    text = g.to_hepmc3()
    assert text.startswith("HepMC::Version")
    assert "zoo.scene t" in text and "zoo.event DECAY" in text
    assert "nwt." not in text
