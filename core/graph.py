"""Interaction graph: people as nodes, candidate pairs as edges.

ARCHITECTURE STATEMENT (read before extending): ST-GCN++ estimates pairwise
violent interaction. Graph reasoning aggregates simultaneous violent pair
relationships into scene-level multi-person violence candidates. There is NO
pretrained "mob"/"group assault" neural class anywhere in this system; group
assault is a deterministic graph pattern (one person with strong violent
edges to 2+ others), not a recognized action. Never report otherwise.

numpy-only. Thresholds come from config, never hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from core.logging_utils import get_logger
from core.pairs import PairTrack, pair_key_name

log = get_logger("graph")


class SceneCandidate(Enum):
    NO_FIGHT = "NO_FIGHT"
    FIGHT = "FIGHT"
    MULTI_PERSON_FIGHT = "MULTI_PERSON_FIGHT"
    GROUP_ASSAULT = "GROUP_ASSAULT"


_SEVERITY = {
    SceneCandidate.NO_FIGHT: 0,
    SceneCandidate.FIGHT: 1,
    SceneCandidate.MULTI_PERSON_FIGHT: 2,
    SceneCandidate.GROUP_ASSAULT: 3,
}


@dataclass
class EdgeInfo:
    key: Tuple[int, int, int, int]
    person_ids: Tuple[int, int] = (0, 0)
    candidate: bool = True
    score: Optional[float] = None
    frames_observed: int = 0
    last_seen: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ComponentVerdict:
    members: List[int]
    violent_edges: List[Tuple[int, int, int, int]]
    candidate: SceneCandidate
    hub_person: Optional[int] = None


@dataclass
class SceneVerdict:
    scene: SceneCandidate
    incidents: List[ComponentVerdict] = field(default_factory=list)
    features: dict = field(default_factory=dict)


@dataclass
class GraphConfig:
    violence_edge_threshold: float = 0.5
    minimum_edges_for_group_assault: int = 2
    minimum_people_for_multi_person_fight: int = 3
    minimum_violent_degree_for_group_assault: int = 2


class InteractionGraph:
    def __init__(self):
        self.nodes: Set[int] = set()
        self.edges: Dict[Tuple[int, int, int, int], EdgeInfo] = {}

    def reset(self) -> None:
        self.nodes.clear()
        self.edges.clear()

    def ingest_tracks(self, tracks: List[PairTrack], timestamp: float) -> None:
        for t in tracks:
            self.nodes.add(t.id_a)
            self.nodes.add(t.id_b)
            e = self.edges.get(t.key)
            if e is None:
                e = EdgeInfo(key=t.key, person_ids=(t.id_a, t.id_b))
                self.edges[t.key] = e
            e.candidate = True
            e.frames_observed = t.frame_count
            e.last_seen = float(timestamp)
            e.metadata = dict(t.latest_features)

    def set_score(self, key: Tuple[int, int, int, int], score: float) -> None:
        e = self.edges.get(key)
        if e is not None:
            e.score = float(score)

    def violent_keys(self, threshold: float) -> List[Tuple[int, int, int, int]]:
        return [k for k, e in self.edges.items()
                if e.score is not None and e.score >= threshold]

    def violent_degree(self, threshold: float) -> Dict[int, int]:
        deg: Dict[int, int] = {n: 0 for n in self.nodes}
        for k in self.violent_keys(threshold):
            e = self.edges[k]
            deg[e.person_ids[0]] = deg.get(e.person_ids[0], 0) + 1
            deg[e.person_ids[1]] = deg.get(e.person_ids[1], 0) + 1
        return deg

    def violent_components(self, threshold: float) -> List[Set[int]]:
        adj: Dict[int, Set[int]] = {n: set() for n in self.nodes}
        for k in self.violent_keys(threshold):
            a, b = self.edges[k].person_ids
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        seen: Set[int] = set()
        comps: List[Set[int]] = []
        for n in adj:
            if n in seen or not adj[n]:
                continue
            stack, comp = [n], set()
            while stack:
                m = stack.pop()
                if m in seen:
                    continue
                seen.add(m)
                comp.add(m)
                stack.extend(adj[m] - seen)
            comps.append(comp)
        return comps


def edge_set_for(members: Set[int], violent: List[Tuple[int, int, int, int]],
                 edges: Dict[Tuple[int, int, int, int], EdgeInfo]) -> list:
    return [k for k in violent
            if edges[k].person_ids[0] in members and edges[k].person_ids[1] in members]


def _spoke_edges(hub: int, members: Set[int],
                 violent: List[Tuple[int, int, int, int]],
                 edges: Dict[Tuple[int, int, int, int], EdgeInfo]) -> int:
    others = set(members) - {hub}
    n = 0
    for k in violent:
        a, b = edges[k].person_ids
        if a in others and b in others:
            n += 1
    return n


def _best_hub(members: Set[int], deg: Dict[int, int],
              violent: List[Tuple[int, int, int, int]],
              edges: Dict[Tuple[int, int, int, int], EdgeInfo],
              cfg: GraphConfig) -> Optional[int]:
    cands = [p for p in members if deg.get(p, 0) >= cfg.minimum_violent_degree_for_group_assault]
    cands = [p for p in cands if _spoke_edges(p, members, violent, edges) == 0]
    if not cands:
        return None
    return sorted(cands, key=lambda p: (-deg.get(p, 0), p))[0]


def reason_scene(graph: InteractionGraph, cfg: GraphConfig) -> SceneVerdict:
    thr = cfg.violence_edge_threshold
    violent = graph.violent_keys(thr)
    deg = graph.violent_degree(thr)
    comps = graph.violent_components(thr)
    verdicts: List[ComponentVerdict] = []
    for comp in comps:
        members = sorted(comp)
        vedges = edge_set_for(comp, violent, graph.edges)
        hub = _best_hub(comp, deg, violent, graph.edges, cfg)
        if hub is not None and len(vedges) >= cfg.minimum_edges_for_group_assault:
            verdicts.append(ComponentVerdict(members, vedges,
                                              SceneCandidate.GROUP_ASSAULT, hub))
        elif len(members) >= cfg.minimum_people_for_multi_person_fight:
            verdicts.append(ComponentVerdict(members, vedges,
                                              SceneCandidate.MULTI_PERSON_FIGHT))
        else:
            verdicts.append(ComponentVerdict(members, vedges, SceneCandidate.FIGHT))
    scene = SceneCandidate.NO_FIGHT
    for v in verdicts:
        if _SEVERITY[v.candidate] > _SEVERITY[scene]:
            scene = v.candidate
    feats = {
        "number_of_active_people": len(graph.nodes),
        "number_of_candidate_pairs": len(graph.edges),
        "number_of_high_score_edges": len(violent),
        "connected_components": len(comps),
        "violent_connected_components": len(verdicts),
        "violent_degree_per_person": {str(k): v for k, v in sorted(deg.items())},
    }
    return SceneVerdict(scene=scene, incidents=verdicts, features=feats)


def overlay_lines(verdict: SceneVerdict, graph: InteractionGraph) -> List[str]:
    lines = [f"scene_candidate={verdict.scene.value}"]
    for e in sorted(graph.edges.values(), key=lambda x: pair_key_name(x.key)):
        s = f"{e.score:.2f}" if e.score is not None else "n/a"
        lines.append(f"P{e.person_ids[0]}<->P{e.person_ids[1]} "
                     f"pair={pair_key_name(e.key)} score={s} buf={e.frames_observed}")
    for v in verdict.incidents:
        lines.append(f"component={sorted(v.members)} violent_edges={len(v.violent_edges)} "
                     f"candidate={v.candidate.value}"
                     + (f" hub=P{v.hub_person}" if v.hub_person else ""))
    deg = verdict.features.get("violent_degree_per_person", {})
    lines.append("degrees=" + ",".join(f"P{k}:{v}" for k, v in sorted(deg.items())))
    return lines
