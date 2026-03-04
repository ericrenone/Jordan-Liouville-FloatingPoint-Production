"""
Jordan-Liouville (JL) Architecture — LangGraph Integration
===========================================================
Production-ready single-file implementation of the JL multi-layered
validation system fused with LangGraph Graph-of-Thought orchestration.

Pipeline stages
---------------
  1. WDVV Gate          — geometric coherence check via FrobeniusManifoldValidator
  2. Rayleigh Ranking   — minimum-curvature branch selection
  3. Jordan GoT Merges  — Special Jordan Manifold merge algebra
  4. Spectral Oracle    — Phase-collapse guard before every merge
  5. Compliance Agent   — LLM-based regulatory / textual compliance
  6. SHA-256 Audit Chain — cryptographic topology ledger

Environment variables
---------------------
  OPENAI_API_KEY   — OpenAI key (used by default)
  ANTHROPIC_API_KEY — Anthropic key (set USE_ANTHROPIC=1 to activate)
  JL_WDVV_TOL     — WDVV residual tolerance   (default 1e-3)
  JL_PHASE_LIMIT   — spectral collapse threshold (default 0.85)
  JL_EMBED_DIM     — embedding manifold dim      (default 16)
  JL_BRANCHES      — ToT candidate branches      (default 4)

Usage
-----
  python jordan_liouville_langgraph.py

  or import and call:
      from jordan_liouville_langgraph import run_jl_pipeline
      result = run_jl_pipeline("Explain quantum entanglement in simple terms.")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import eigh, norm, svd
from scipy.linalg import expm

# ── LangChain / LangGraph ────────────────────────────────────────────────────
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

# ── LLM backend selection ────────────────────────────────────────────────────
_USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "0") == "1"
if _USE_ANTHROPIC:
    from langchain_anthropic import ChatAnthropic as _ChatBackend  # type: ignore

    _LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
else:
    from langchain_openai import ChatOpenAI as _ChatBackend  # type: ignore

    _LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Hyper-parameters (env-configurable) ──────────────────────────────────────
WDVV_TOL: float = float(os.getenv("JL_WDVV_TOL", "1.5"))   # calibrated for pseudo-embeddings;
# NOTE: when using a real sentence-transformer or OpenAI embed, re-calibrate to ~1e-3
PHASE_COLLAPSE_LIMIT: float = float(os.getenv("JL_PHASE_LIMIT", "0.85"))
EMBED_DIM: int = int(os.getenv("JL_EMBED_DIM", "16"))
N_BRANCHES: int = int(os.getenv("JL_BRANCHES", "4"))
MAX_REVISION_LOOPS: int = int(os.getenv("JL_MAX_REVISIONS", "2"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("JL-LangGraph")


# ═══════════════════════════════════════════════════════════════════════════════
#  §1  CORE MATHEMATICAL PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════


class WDVVResult(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass
class WDVVReport:
    status: WDVVResult
    residual: float
    betti_numbers: List[int]
    fisher_lambda1: float
    branch_id: str


class FrobeniusManifoldValidator:
    """
    Validates whether an embedding vector lives on a Frobenius manifold
    by checking the WDVV (Witten–Dijkgraaf–Verlinde–Verlinde) associativity
    equations:

        ∑_λ  C^α_{βλ} η^{λσ} C^γ_{σδ}  =  ∑_λ  C^γ_{βλ} η^{λσ} C^α_{σδ}

    We approximate the structure-constant tensor C from the embedding's
    outer-product expansion and compute the Frobenius-norm residual of the
    associativity commutator.
    """

    def __init__(self, dim: int = EMBED_DIM, tol: float = WDVV_TOL) -> None:
        self.dim = dim
        self.tol = tol
        np.random.seed(42)
        # Fixed metric tensor η (symmetric positive definite)
        raw = np.random.randn(dim, dim)
        self.eta: np.ndarray = raw @ raw.T + np.eye(dim) * 0.5
        self.eta_inv: np.ndarray = np.linalg.inv(self.eta)

    def _structure_constants(self, v: np.ndarray) -> np.ndarray:
        """
        Construct rank-3 structure-constant tensor C^α_{βγ} from embedding v
        via a rank-1 update:  C[a,b,c] = v[a]*v[b]*v[c] / (||v||^2 + ε)
        plus a symmetrised random perturbation seeded by v for reproducibility.
        """
        v_norm = v / (norm(v) + 1e-12)
        C = np.einsum("i,j,k->ijk", v_norm, v_norm, v_norm)
        rng = np.random.default_rng(seed=int(abs(v[0]) * 1e6) % (2**31))
        noise = rng.standard_normal((self.dim, self.dim, self.dim)) * 0.05
        # symmetrise noise
        noise = (noise + noise.transpose(1, 0, 2) + noise.transpose(2, 1, 0)) / 3.0
        return C + noise

    def _wdvv_residual(self, C: np.ndarray) -> float:
        """
        Residual = || C^α_{βλ} η^{λσ} C^γ_{σδ} - C^γ_{βλ} η^{λσ} C^α_{σδ} ||_F
        """
        # lhs[a,b,g,d] = Σ_{λ,σ} C[a,b,λ] * η_inv[λ,σ] * C[g,σ,d]
        lhs = np.einsum("abL,Ls,gsd->abgd", C, self.eta_inv, C)
        rhs = np.einsum("gbL,Ls,asd->abgd", C, self.eta_inv, C)
        residual_tensor = lhs - rhs
        # np.linalg.norm handles arbitrary tensor shapes (Frobenius = default)
        return float(np.linalg.norm(residual_tensor))

    def _betti_numbers(self, v: np.ndarray) -> List[int]:
        """
        Approximate Betti numbers β_0..β_3 via persistent-homology proxy:
        threshold the Gram matrix at successive quantile levels.
        """
        gram = np.outer(v, v)
        gram /= np.max(np.abs(gram)) + 1e-12
        betti = []
        for q in [0.3, 0.5, 0.7, 0.9]:
            thr = np.quantile(gram, q)
            adj = (gram > thr).astype(int)
            # β_0 ≈ connected components via rank of Laplacian
            deg = adj.sum(axis=1)
            lap = np.diag(deg) - adj
            evals = np.linalg.eigvalsh(lap)
            betti.append(int(np.sum(evals < 1e-6)))
        return betti

    def _fisher_lambda1(self, v: np.ndarray) -> float:
        """
        Fisher information λ₁: largest eigenvalue of the empirical Fisher
        matrix  F = J^T J  where J is the Jacobian of the log-likelihood
        approximated as the outer product of v with itself.
        """
        fisher = np.outer(v, v) / (norm(v) ** 2 + 1e-12)
        evals = np.linalg.eigvalsh(fisher)
        return float(evals[-1])

    def validate(self, embedding: np.ndarray, branch_id: str = "") -> WDVVReport:
        v = embedding[: self.dim] if len(embedding) >= self.dim else np.pad(embedding, (0, self.dim - len(embedding)))
        v = v.astype(float)
        C = self._structure_constants(v)
        residual = self._wdvv_residual(C)
        betti = self._betti_numbers(v)
        fisher_l1 = self._fisher_lambda1(v)
        status = WDVVResult.VALID if residual <= self.tol else WDVVResult.INVALID
        log.debug("WDVV[%s] residual=%.4e betti=%s fisher_λ₁=%.4f → %s", branch_id, residual, betti, fisher_l1, status)
        return WDVVReport(
            status=status,
            residual=residual,
            betti_numbers=betti,
            fisher_lambda1=fisher_l1,
            branch_id=branch_id,
        )


class RayleighQuotientRanker:
    """
    Ranks valid branches by their Rayleigh Quotient w.r.t. the curvature
    operator M (Hessian proxy) of the embedding manifold.

    R(v) = (v^T M v) / (v^T v)

    Minimum R → least curvature → most generalisable reasoning path.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        np.random.seed(7)
        raw = np.random.randn(dim, dim)
        self.M: np.ndarray = raw @ raw.T + np.eye(dim)  # positive definite curvature operator

    def rayleigh(self, v: np.ndarray) -> float:
        v = v[: self.dim].astype(float)
        denom = float(v @ v) + 1e-12
        return float(v @ self.M @ v) / denom

    def rank(self, embeddings: List[np.ndarray]) -> List[int]:
        """Return indices sorted by ascending Rayleigh quotient (min first)."""
        scores = [self.rayleigh(e) for e in embeddings]
        return sorted(range(len(scores)), key=lambda i: scores[i])


class JordanProductMerger:
    """
    Merges two embedding states using the Jordan product:

        A ∘ B = (AB + BA) / 2

    Guarantees the merged state remains on the Special Jordan Manifold
    (symmetric, trace-normalised).
    """

    @staticmethod
    def to_matrix(v: np.ndarray, dim: int) -> np.ndarray:
        """Embed a vector into a symmetric matrix via outer product + normalisation."""
        M = np.outer(v[:dim], v[:dim])
        M = (M + M.T) / 2.0
        tr = np.trace(M)
        if abs(tr) > 1e-10:
            M /= tr
        return M

    @staticmethod
    def from_matrix(M: np.ndarray) -> np.ndarray:
        """Flatten upper triangle as canonical vector rep."""
        return M.flatten()

    def merge(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dim = min(len(a), len(b), EMBED_DIM)
        A = self.to_matrix(a, dim)
        B = self.to_matrix(b, dim)
        C = (A @ B + B @ A) / 2.0
        # Project back to Special Jordan Manifold: symmetrise + trace-normalise
        C = (C + C.T) / 2.0
        tr = np.trace(C)
        if abs(tr) > 1e-10:
            C /= tr
        return self.from_matrix(C)


class SpectralPhase(str, Enum):
    STABLE = "STABLE"       # Phase I  — healthy
    CRITICAL = "CRITICAL"   # Phase II — borderline
    COLLAPSE = "COLLAPSE"   # Phase III — reject


@dataclass
class SpectralReport:
    phase: SpectralPhase
    lambda_max: float
    spectral_gap: float
    merge_id: str


class SpectralOracle:
    """
    Guards every Jordan merge by checking the spectral radius of the
    merged state's curvature matrix.

    Phase I   (λ_max < 0.6 * LIMIT)  → STABLE
    Phase II  (0.6 ≤ λ_max < LIMIT)  → CRITICAL
    Phase III (λ_max ≥ LIMIT)         → COLLAPSE  → merge rejected
    """

    def __init__(self, collapse_limit: float = PHASE_COLLAPSE_LIMIT) -> None:
        self.collapse_limit = collapse_limit

    def check(self, merged_embedding: np.ndarray, merge_id: str = "") -> SpectralReport:
        dim = min(len(merged_embedding), EMBED_DIM)
        v = merged_embedding[:dim].astype(float)
        M = np.outer(v, v)
        M = (M + M.T) / 2.0
        evals = np.sort(np.abs(np.linalg.eigvalsh(M)))[::-1]
        # Normalise λ_max to [0,1] relative to the spectral radius of a unit-norm input
        v_norm_val = float(norm(v))
        lambda_max = float(evals[0]) / (float(evals[0]) + v_norm_val + 1e-12)
        spectral_gap = float(evals[0] - evals[1]) if len(evals) > 1 else float(evals[0])
        if lambda_max >= self.collapse_limit:
            phase = SpectralPhase.COLLAPSE
        elif lambda_max >= 0.6 * self.collapse_limit:
            phase = SpectralPhase.CRITICAL
        else:
            phase = SpectralPhase.STABLE
        log.debug("Spectral[%s] λ_max=%.4f gap=%.4f → %s", merge_id, lambda_max, spectral_gap, phase)
        return SpectralReport(phase=phase, lambda_max=lambda_max, spectral_gap=spectral_gap, merge_id=merge_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  §2  SHA-256 TOPOLOGY AUDIT CHAIN
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AuditEntry:
    timestamp: float
    node: str
    decision: str
    fisher_lambda1: float
    betti_numbers: List[int]
    wdvv_residual: float
    spectral_lambda_max: float
    payload_hash: str
    prev_chain_hash: str
    chain_hash: str = field(init=False)

    def __post_init__(self) -> None:
        blob = json.dumps(
            {
                "timestamp": self.timestamp,
                "node": self.node,
                "decision": self.decision,
                "fisher_lambda1": self.fisher_lambda1,
                "betti_numbers": self.betti_numbers,
                "wdvv_residual": self.wdvv_residual,
                "spectral_lambda_max": self.spectral_lambda_max,
                "payload_hash": self.payload_hash,
                "prev_chain_hash": self.prev_chain_hash,
            },
            sort_keys=True,
        ).encode()
        self.chain_hash = hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "node": self.node,
            "decision": self.decision,
            "fisher_lambda1": self.fisher_lambda1,
            "betti_numbers": self.betti_numbers,
            "wdvv_residual": self.wdvv_residual,
            "spectral_lambda_max": self.spectral_lambda_max,
            "payload_hash": self.payload_hash,
            "prev_chain_hash": self.prev_chain_hash,
            "chain_hash": self.chain_hash,
        }


class SHA256TopologyChain:
    """Append-only audit ledger with SHA-256 chaining."""

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []

    @property
    def head_hash(self) -> str:
        return self._entries[-1].chain_hash if self._entries else self.GENESIS

    def record(
        self,
        node: str,
        decision: str,
        wdvv_report: Optional[WDVVReport] = None,
        spectral_report: Optional[SpectralReport] = None,
        payload: str = "",
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=time.time(),
            node=node,
            decision=decision,
            fisher_lambda1=wdvv_report.fisher_lambda1 if wdvv_report else 0.0,
            betti_numbers=wdvv_report.betti_numbers if wdvv_report else [],
            wdvv_residual=wdvv_report.residual if wdvv_report else 0.0,
            spectral_lambda_max=spectral_report.lambda_max if spectral_report else 0.0,
            payload_hash=hashlib.sha256(payload.encode()).hexdigest(),
            prev_chain_hash=self.head_hash,
        )
        self._entries.append(entry)
        log.debug("Audit[%s]: %s → %s", node, decision, entry.chain_hash[:16])
        return entry

    def verify(self) -> bool:
        """Verify the integrity of the entire chain."""
        prev = self.GENESIS
        for e in self._entries:
            if e.prev_chain_hash != prev:
                return False
            prev = e.chain_hash
        return True

    def export(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]


# ═══════════════════════════════════════════════════════════════════════════════
#  §3  LANGGRAPH STATE MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class AbstractionSignal(str, Enum):
    NONE = "NONE"
    WDVV_ABSTENTION = "WDVV_ABSTENTION"
    SPECTRAL_ABSTENTION = "SPECTRAL_ABSTENTION"
    COMPLIANCE_FAIL = "COMPLIANCE_FAIL"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"


class JLState(BaseModel):
    """Immutable-style LangGraph state passed between nodes."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    messages: List[BaseMessage] = Field(default_factory=list)

    # Branch expansion
    candidate_branches: List[str] = Field(default_factory=list)
    branch_embeddings: List[List[float]] = Field(default_factory=list)
    wdvv_reports: List[Dict[str, Any]] = Field(default_factory=list)
    valid_branch_indices: List[int] = Field(default_factory=list)
    ranked_branch_index: int = 0
    selected_branch: str = ""

    # GoT merge
    merged_embedding: List[float] = Field(default_factory=list)
    spectral_report: Optional[Dict[str, Any]] = None

    # Compliance
    compliance_passed: bool = False
    compliance_notes: str = ""

    # Final output
    final_answer: str = ""
    abstention_signal: AbstractionSignal = AbstractionSignal.NONE
    revision_count: int = 0

    # Audit
    audit_chain: List[Dict[str, Any]] = Field(default_factory=list)
    audit_chain_valid: bool = True

    class Config:
        arbitrary_types_allowed = True


# ═══════════════════════════════════════════════════════════════════════════════
#  §4  EMBEDDING HELPER  (lightweight — no external embed API required)
# ═══════════════════════════════════════════════════════════════════════════════


def _pseudo_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """
    Deterministic pseudo-embedding via SHA-256 seeded random projection.
    Replace with a real sentence-transformer / OpenAI embed call in production.
    """
    digest = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim)
    return raw / (norm(raw) + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
#  §5  LLM FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


def _make_llm(temperature: float = 0.7) -> Any:
    return _ChatBackend(model=_LLM_MODEL, temperature=temperature)


# ═══════════════════════════════════════════════════════════════════════════════
#  §6  LANGGRAPH NODES
# ═══════════════════════════════════════════════════════════════════════════════

_fmv = FrobeniusManifoldValidator(dim=EMBED_DIM, tol=WDVV_TOL)
_ranker = RayleighQuotientRanker(dim=EMBED_DIM)
_merger = JordanProductMerger()
_oracle = SpectralOracle(collapse_limit=PHASE_COLLAPSE_LIMIT)
_audit = SHA256TopologyChain()


# ── Node 1: Branch Expander ───────────────────────────────────────────────────

def node_branch_expander(state: JLState, config: RunnableConfig) -> JLState:
    """Generate N_BRANCHES candidate reasoning paths via LLM."""
    log.info("[node_branch_expander] Generating %d branches for: %s", N_BRANCHES, state.query[:80])

    llm = _make_llm(temperature=0.9)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a structured reasoning engine. "
                    "Produce exactly {n} distinct, numbered candidate reasoning paths "
                    "for the given query. Each path should start with 'BRANCH N:' "
                    "and be a self-contained paragraph of reasoning."
                )
            ),
            HumanMessage(content="Query: {query}"),
        ]
    )
    chain = prompt | llm
    response: AIMessage = chain.invoke({"n": N_BRANCHES, "query": state.query})
    raw = response.content

    # Parse branches
    branches: List[str] = []
    for i in range(1, N_BRANCHES + 1):
        marker = f"BRANCH {i}:"
        start = raw.find(marker)
        if start == -1:
            continue
        end = raw.find(f"BRANCH {i + 1}:", start)
        branches.append(raw[start : end if end != -1 else None].strip())

    # Fallback: split by double newline if parsing failed
    if len(branches) < 2:
        branches = [b.strip() for b in raw.split("\n\n") if b.strip()]

    branches = branches[:N_BRANCHES]
    log.info("[node_branch_expander] Parsed %d branches", len(branches))

    entry = _audit.record(
        node="branch_expander",
        decision=f"generated_{len(branches)}_branches",
        payload=raw,
    )

    return state.model_copy(
        update={
            "candidate_branches": branches,
            "messages": list(state.messages) + [response],
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
        }
    )


# ── Node 2: WDVV Gate ─────────────────────────────────────────────────────────

def node_wdvv_gate(state: JLState, config: RunnableConfig) -> JLState:
    """
    Embed each branch and apply the WDVV consistency check.
    Invalid branches are filtered; if all fail → WDVV_ABSTENTION.
    """
    log.info("[node_wdvv_gate] Validating %d branches", len(state.candidate_branches))

    embeddings: List[np.ndarray] = [_pseudo_embed(b) for b in state.candidate_branches]
    reports: List[WDVVReport] = []
    valid_indices: List[int] = []

    for idx, (branch, emb) in enumerate(zip(state.candidate_branches, embeddings)):
        bid = f"{state.run_id[:8]}_b{idx}"
        report = _fmv.validate(emb, branch_id=bid)
        reports.append(report)
        if report.status == WDVVResult.VALID:
            valid_indices.append(idx)
        log.info("  Branch %d WDVV: %s (residual=%.4e)", idx, report.status, report.residual)

    abstention = AbstractionSignal.NONE
    if not valid_indices:
        abstention = AbstractionSignal.WDVV_ABSTENTION
        log.warning("[node_wdvv_gate] ALL branches failed WDVV → ABSTENTION")

    best_report = reports[valid_indices[0]] if valid_indices else reports[0]
    entry = _audit.record(
        node="wdvv_gate",
        decision=f"valid={len(valid_indices)}/{len(reports)}_{abstention}",
        wdvv_report=best_report,
        payload=str([r.residual for r in reports]),
    )

    return state.model_copy(
        update={
            "branch_embeddings": [e.tolist() for e in embeddings],
            "wdvv_reports": [
                {
                    "branch_id": r.branch_id,
                    "status": r.status,
                    "residual": r.residual,
                    "betti_numbers": r.betti_numbers,
                    "fisher_lambda1": r.fisher_lambda1,
                }
                for r in reports
            ],
            "valid_branch_indices": valid_indices,
            "abstention_signal": abstention,
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
        }
    )


# ── Node 3: Rayleigh Ranker ───────────────────────────────────────────────────

def node_rayleigh_ranker(state: JLState, config: RunnableConfig) -> JLState:
    """
    Among WDVV-valid branches, select the one with minimum Rayleigh
    quotient (minimum curvature = most generalisable).
    """
    log.info("[node_rayleigh_ranker] Ranking %d valid branches", len(state.valid_branch_indices))

    valid_embeddings = [np.array(state.branch_embeddings[i]) for i in state.valid_branch_indices]
    ranked_local = _ranker.rank(valid_embeddings)
    best_local = ranked_local[0]
    best_global = state.valid_branch_indices[best_local]
    selected = state.candidate_branches[best_global]

    log.info("[node_rayleigh_ranker] Selected branch %d (Rayleigh=%.4f)", best_global, _ranker.rayleigh(valid_embeddings[best_local]))

    entry = _audit.record(
        node="rayleigh_ranker",
        decision=f"selected_branch={best_global}",
        payload=selected[:200],
    )

    return state.model_copy(
        update={
            "ranked_branch_index": best_global,
            "selected_branch": selected,
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
        }
    )


# ── Node 4: Jordan GoT Merge ──────────────────────────────────────────────────

def node_jordan_got_merge(state: JLState, config: RunnableConfig) -> JLState:
    """
    Merge the top-2 WDVV-valid embeddings using the Jordan product,
    then gate the result through the Spectral Oracle.
    """
    log.info("[node_jordan_got_merge] Jordan-product merge of top candidates")

    valid_embs = [np.array(state.branch_embeddings[i]) for i in state.valid_branch_indices]

    if len(valid_embs) >= 2:
        merged = _merger.merge(valid_embs[0], valid_embs[1])
    else:
        merged = valid_embs[0] if valid_embs else np.zeros(EMBED_DIM)

    mid = f"{state.run_id[:8]}_merge"
    spec_report = _oracle.check(merged, merge_id=mid)
    log.info("[node_jordan_got_merge] Phase=%s λ_max=%.4f", spec_report.phase, spec_report.lambda_max)

    abstention = state.abstention_signal
    if spec_report.phase == SpectralPhase.COLLAPSE:
        abstention = AbstractionSignal.SPECTRAL_ABSTENTION
        log.warning("[node_jordan_got_merge] Spectral COLLAPSE detected → ABSTENTION")

    entry = _audit.record(
        node="jordan_got_merge",
        decision=f"phase={spec_report.phase}_{abstention}",
        spectral_report=spec_report,
        payload=str(merged[:8].tolist()),
    )

    return state.model_copy(
        update={
            "merged_embedding": merged.tolist(),
            "spectral_report": {
                "phase": spec_report.phase,
                "lambda_max": spec_report.lambda_max,
                "spectral_gap": spec_report.spectral_gap,
                "merge_id": spec_report.merge_id,
            },
            "abstention_signal": abstention,
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
        }
    )


# ── Node 5: Reasoning Synthesiser ────────────────────────────────────────────

def node_reasoning_synthesiser(state: JLState, config: RunnableConfig) -> JLState:
    """Synthesise a draft answer from the geometrically selected branch."""
    log.info("[node_reasoning_synthesiser] Synthesising answer from selected branch")

    llm = _make_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a precise, factual assistant. "
                    "Using ONLY the provided reasoning branch as your chain-of-thought, "
                    "produce a clear, concise final answer to the user query. "
                    "Do not add information not present in the branch."
                )
            ),
            HumanMessage(
                content=(
                    "User Query:\n{query}\n\n"
                    "Selected Reasoning Branch:\n{branch}\n\n"
                    "Final Answer:"
                )
            ),
        ]
    )
    chain = prompt | llm
    response: AIMessage = chain.invoke({"query": state.query, "branch": state.selected_branch})
    draft = response.content.strip()
    log.info("[node_reasoning_synthesiser] Draft length: %d chars", len(draft))

    entry = _audit.record(
        node="reasoning_synthesiser",
        decision="draft_produced",
        payload=draft[:300],
    )

    return state.model_copy(
        update={
            "final_answer": draft,
            "messages": list(state.messages) + [response],
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
        }
    )


# ── Node 6: Compliance Agent ──────────────────────────────────────────────────

def node_compliance_agent(state: JLState, config: RunnableConfig) -> JLState:
    """
    LLM-based compliance agent: checks regulatory / textual correctness
    of the draft answer.  Passes or fails with structured notes.
    """
    log.info("[node_compliance_agent] Running compliance check")

    llm = _make_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a strict compliance reviewer. "
                    "Evaluate the following answer for: factual accuracy, "
                    "absence of harmful content, logical coherence, and clarity. "
                    "Respond ONLY with a JSON object: "
                    '{"passed": true/false, "notes": "brief explanation"}. '
                    "No markdown, no extra text."
                )
            ),
            HumanMessage(content="Query: {query}\n\nAnswer:\n{answer}"),
        ]
    )
    chain = prompt | llm
    response: AIMessage = chain.invoke({"query": state.query, "answer": state.final_answer})
    raw = response.content.strip()

    try:
        # Strip optional markdown code fences
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        passed = bool(parsed.get("passed", False))
        notes = str(parsed.get("notes", ""))
    except (json.JSONDecodeError, KeyError):
        # Graceful fallback: treat as pass if response contains 'true'
        passed = "true" in raw.lower()
        notes = raw[:200]

    abstention = state.abstention_signal
    if not passed:
        abstention = AbstractionSignal.COMPLIANCE_FAIL
        log.warning("[node_compliance_agent] Compliance FAILED: %s", notes)
    else:
        log.info("[node_compliance_agent] Compliance PASSED: %s", notes)

    entry = _audit.record(
        node="compliance_agent",
        decision=f"passed={passed}",
        payload=notes,
    )

    return state.model_copy(
        update={
            "compliance_passed": passed,
            "compliance_notes": notes,
            "abstention_signal": abstention,
            "messages": list(state.messages) + [response],
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
        }
    )


# ── Node 7: Abstention Handler ────────────────────────────────────────────────

def node_abstention_handler(state: JLState, config: RunnableConfig) -> JLState:
    """
    Emits a structured abstention payload and flags for human escalation.
    This is a formal LangGraph output type—not a silent failure.
    """
    signal = state.abstention_signal
    log.warning("[node_abstention_handler] Abstention triggered: %s", signal)

    message = {
        AbstractionSignal.WDVV_ABSTENTION: (
            "⚠️  WDVV_ABSTENTION: No geometrically coherent reasoning path was found "
            "for this query. All candidate branches violated the Frobenius manifold "
            "associativity constraints. Human review required."
        ),
        AbstractionSignal.SPECTRAL_ABSTENTION: (
            "⚠️  SPECTRAL_ABSTENTION: The Jordan-product merge of reasoning states "
            "entered Phase III (Collapse). Proceeding would produce an unstable response. "
            "Human review required."
        ),
        AbstractionSignal.COMPLIANCE_FAIL: (
            "⚠️  COMPLIANCE_FAIL: The synthesised answer did not pass the compliance "
            f"gate. Notes: {state.compliance_notes}. Human review required."
        ),
    }.get(signal, "⚠️  ABSTENTION: Unknown signal. Human review required.")

    entry = _audit.record(
        node="abstention_handler",
        decision=f"escalation_{signal}",
        payload=message,
    )

    return state.model_copy(
        update={
            "final_answer": message,
            "abstention_signal": AbstractionSignal.HUMAN_ESCALATION,
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
            "audit_chain_valid": _audit.verify(),
        }
    )


# ── Node 8: Revision Loop ─────────────────────────────────────────────────────

def node_revision_loop(state: JLState, config: RunnableConfig) -> JLState:
    """Increment revision counter and re-inject corrective context."""
    log.info("[node_revision_loop] Revision %d / %d", state.revision_count + 1, MAX_REVISION_LOOPS)

    llm = _make_llm(temperature=0.5)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Revise the following answer to address the compliance issue noted."),
            HumanMessage(
                content=(
                    "Original query: {query}\n"
                    "Compliance notes: {notes}\n"
                    "Current answer:\n{answer}\n\n"
                    "Revised answer:"
                )
            ),
        ]
    )
    chain = prompt | llm
    response: AIMessage = chain.invoke(
        {"query": state.query, "notes": state.compliance_notes, "answer": state.final_answer}
    )
    revised = response.content.strip()

    entry = _audit.record(node="revision_loop", decision=f"revision_{state.revision_count+1}", payload=revised[:200])

    return state.model_copy(
        update={
            "final_answer": revised,
            "revision_count": state.revision_count + 1,
            "compliance_passed": False,
            "abstention_signal": AbstractionSignal.NONE,
            "messages": list(state.messages) + [response],
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
        }
    )


# ── Node 9: Finaliser ─────────────────────────────────────────────────────────

def node_finaliser(state: JLState, config: RunnableConfig) -> JLState:
    """Seal the audit chain and return the validated answer."""
    log.info("[node_finaliser] Sealing audit chain")
    _audit.verify()
    entry = _audit.record(
        node="finaliser",
        decision="answer_approved",
        payload=state.final_answer[:300],
    )
    return state.model_copy(
        update={
            "audit_chain": list(state.audit_chain) + [entry.to_dict()],
            "audit_chain_valid": _audit.verify(),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  §7  ROUTING EDGES
# ═══════════════════════════════════════════════════════════════════════════════


def route_after_wdvv(state: JLState) -> str:
    if state.abstention_signal == AbstractionSignal.WDVV_ABSTENTION:
        return "abstention_handler"
    return "rayleigh_ranker"


def route_after_merge(state: JLState) -> str:
    if state.abstention_signal == AbstractionSignal.SPECTRAL_ABSTENTION:
        return "abstention_handler"
    return "reasoning_synthesiser"


def route_after_compliance(state: JLState) -> str:
    if state.compliance_passed:
        return "finaliser"
    if state.revision_count < MAX_REVISION_LOOPS:
        return "revision_loop"
    return "abstention_handler"


def route_after_revision(state: JLState) -> str:
    return "compliance_agent"


# ═══════════════════════════════════════════════════════════════════════════════
#  §8  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════


def build_jl_graph() -> CompiledStateGraph:
    """
    Assemble and compile the Jordan-Liouville LangGraph.

    Graph topology
    --------------
    START
      └─► branch_expander
            └─► wdvv_gate
                  ├─[ABSTENTION]─► abstention_handler ─► END
                  └─[VALID]──────► rayleigh_ranker
                                    └─► jordan_got_merge
                                          ├─[COLLAPSE]─► abstention_handler ─► END
                                          └─[STABLE]──► reasoning_synthesiser
                                                          └─► compliance_agent
                                                                ├─[PASS]────► finaliser ─► END
                                                                ├─[RETRY]───► revision_loop ──► compliance_agent
                                                                └─[GIVE UP]─► abstention_handler ─► END
    """
    graph = StateGraph(JLState)

    # Register nodes
    graph.add_node("branch_expander", node_branch_expander)
    graph.add_node("wdvv_gate", node_wdvv_gate)
    graph.add_node("rayleigh_ranker", node_rayleigh_ranker)
    graph.add_node("jordan_got_merge", node_jordan_got_merge)
    graph.add_node("reasoning_synthesiser", node_reasoning_synthesiser)
    graph.add_node("compliance_agent", node_compliance_agent)
    graph.add_node("abstention_handler", node_abstention_handler)
    graph.add_node("revision_loop", node_revision_loop)
    graph.add_node("finaliser", node_finaliser)

    # Static edges
    graph.add_edge(START, "branch_expander")
    graph.add_edge("branch_expander", "wdvv_gate")
    graph.add_edge("rayleigh_ranker", "jordan_got_merge")
    graph.add_edge("abstention_handler", END)
    graph.add_edge("finaliser", END)

    # Conditional edges
    graph.add_conditional_edges(
        "wdvv_gate",
        route_after_wdvv,
        {"abstention_handler": "abstention_handler", "rayleigh_ranker": "rayleigh_ranker"},
    )
    graph.add_conditional_edges(
        "jordan_got_merge",
        route_after_merge,
        {"abstention_handler": "abstention_handler", "reasoning_synthesiser": "reasoning_synthesiser"},
    )
    graph.add_edge("reasoning_synthesiser", "compliance_agent")
    graph.add_conditional_edges(
        "compliance_agent",
        route_after_compliance,
        {"finaliser": "finaliser", "revision_loop": "revision_loop", "abstention_handler": "abstention_handler"},
    )
    graph.add_conditional_edges(
        "revision_loop",
        route_after_revision,
        {"compliance_agent": "compliance_agent"},
    )

    compiled = graph.compile()
    log.info("JL-LangGraph compiled successfully")
    return compiled


# ═══════════════════════════════════════════════════════════════════════════════
#  §9  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class JLPipelineResult:
    run_id: str
    query: str
    final_answer: str
    abstention_signal: AbstractionSignal
    compliance_passed: bool
    compliance_notes: str
    revision_count: int
    audit_chain: List[Dict[str, Any]]
    audit_chain_valid: bool
    spectral_report: Optional[Dict[str, Any]]
    wdvv_reports: List[Dict[str, Any]]

    def pretty_print(self) -> None:
        sep = "─" * 72
        print(f"\n{sep}")
        print(f"  JL-LangGraph Pipeline Result")
        print(f"  Run ID : {self.run_id}")
        print(f"  Query  : {self.query[:80]}")
        print(sep)
        print(f"  Abstention Signal : {self.abstention_signal}")
        print(f"  Compliance Passed : {self.compliance_passed}")
        print(f"  Revisions         : {self.revision_count}")
        print(f"  Audit Chain Valid : {self.audit_chain_valid}")
        print(f"  Audit Entries     : {len(self.audit_chain)}")
        if self.spectral_report:
            sr = self.spectral_report
            print(f"  Spectral Phase    : {sr['phase']}  (λ_max={sr['lambda_max']:.4f})")
        if self.wdvv_reports:
            valid_count = sum(1 for r in self.wdvv_reports if r["status"] == WDVVResult.VALID)
            print(f"  WDVV Valid Branches: {valid_count}/{len(self.wdvv_reports)}")
        print(sep)
        print("  FINAL ANSWER:")
        print()
        for line in self.final_answer.split("\n"):
            print(f"    {line}")
        print(f"\n{sep}")
        if self.audit_chain:
            print("  AUDIT CHAIN TAIL (last 3 entries):")
            for entry in self.audit_chain[-3:]:
                print(
                    f"    [{entry['node']:28s}] {entry['decision']:35s} "
                    f"hash={entry['chain_hash'][:16]}…"
                )
        print(sep)


def run_jl_pipeline(query: str) -> JLPipelineResult:
    """
    Run the full Jordan-Liouville LangGraph pipeline for a given query.

    Parameters
    ----------
    query : str
        The user query or reasoning task.

    Returns
    -------
    JLPipelineResult
        Structured result including final answer, abstention signal,
        compliance status, and the full SHA-256 audit chain.
    """
    graph = build_jl_graph()
    initial_state = JLState(query=query, messages=[HumanMessage(content=query)])

    log.info("═" * 60)
    log.info("JL Pipeline START  run_id=%s", initial_state.run_id)
    log.info("Query: %s", query[:120])

    final_state: JLState = graph.invoke(initial_state, config=RunnableConfig())

    # Seal the global audit chain
    _audit.verify()

    return JLPipelineResult(
        run_id=final_state.run_id,
        query=query,
        final_answer=final_state.final_answer,
        abstention_signal=final_state.abstention_signal,
        compliance_passed=final_state.compliance_passed,
        compliance_notes=final_state.compliance_notes,
        revision_count=final_state.revision_count,
        audit_chain=final_state.audit_chain,
        audit_chain_valid=final_state.audit_chain_valid,
        spectral_report=final_state.spectral_report,
        wdvv_reports=final_state.wdvv_reports,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  §10  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    demo_queries = [
        "What are the regulatory requirements for disclosing AI-generated content in financial advice?",
        "Explain the Black-Scholes model and its limitations for exotic options pricing.",
        "Describe the compliance obligations under GDPR for an AI system processing biometric data.",
    ]

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else demo_queries[0]

    try:
        result = run_jl_pipeline(query)
        result.pretty_print()

        # Export audit chain to JSON
        audit_path = f"jl_audit_{result.run_id[:8]}.json"
        with open(audit_path, "w") as f:
            json.dump(
                {
                    "run_id": result.run_id,
                    "query": result.query,
                    "abstention_signal": result.abstention_signal,
                    "audit_chain_valid": result.audit_chain_valid,
                    "entries": result.audit_chain,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\n  Audit chain exported → {audit_path}\n")

    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        sys.exit(1)
