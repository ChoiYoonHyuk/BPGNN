import argparse
import copy
import math
import os
import random
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from torch_geometric.transforms import NormalizeFeatures
except ImportError as exc:
    _PYG_IMPORT_ERROR = exc

    def NormalizeFeatures():
        raise ImportError("Dataset loading requires torch-geometric.") from _PYG_IMPORT_ERROR

EPS = 1e-12


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_split_masks(data, split_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    if train_mask.dim() == 1:
        return train_mask, val_mask, test_mask
    if split_idx < 0 or split_idx >= train_mask.size(1):
        raise ValueError(f"split_idx={split_idx} out of range for mask shape {train_mask.shape}")
    return train_mask[:, split_idx], val_mask[:, split_idx], test_mask[:, split_idx]


def dedup_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    key = src * num_nodes + dst
    key_sorted, perm = torch.sort(key)
    src, dst = src[perm], dst[perm]
    mask = torch.ones_like(key_sorted, dtype=torch.bool)
    mask[1:] = key_sorted[1:] != key_sorted[:-1]
    return torch.stack([src[mask], dst[mask]], dim=0)


def load_dataset(name: str, root: str = "./data", normalize_features: bool = True):
    name_raw = name
    name = name.lower()

    transform = NormalizeFeatures() if normalize_features else None
    is_webkb = name in ["texas", "wisconsin", "cornell"]

    if name in ["cora", "citeseer", "pubmed"]:
        from torch_geometric.datasets import Planetoid
        proper = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name]
        dataset = Planetoid(root=os.path.join(root, proper), name=proper, split="public", transform=transform)
        data = dataset[0]
        return data, dataset.num_features, dataset.num_classes

    if name in ["chameleon", "squirrel"]:
        try:
            from torch_geometric.datasets import WikipediaNetwork
            dataset = WikipediaNetwork(
                root=os.path.join(root, name_raw),
                name=name,
                geom_gcn_preprocess=True,
                transform=transform,
            )
            data = dataset[0]
            return data, dataset.num_features, dataset.num_classes
        except Exception as e:
            print(f"[WARN] WikipediaNetwork load failed ({e}). Trying fallback...")

    if is_webkb:
        try:
            from torch_geometric.datasets import WebKB
            dataset = WebKB(root=os.path.join(root, name_raw), name=name, transform=transform)
            data = dataset[0]
            return data, dataset.num_features, dataset.num_classes
        except Exception as e:
            print(f"[WARN] WebKB load failed ({e}). Trying fallback...")

    try:
        from torch_geometric.datasets import HeterophilousGraphDataset
        dataset = HeterophilousGraphDataset(root=os.path.join(root, name_raw), name=name, transform=transform)
        data = dataset[0]
        return data, dataset.num_features, dataset.num_classes
    except Exception as e:
        raise RuntimeError(
            f"Could not load dataset '{name_raw}'. Planetoid/WikipediaNetwork/WebKB/HeterophilousGraphDataset failed.\nLast error: {e}"
        )


def center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def normalized_probabilities(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1).clamp(min=EPS, max=1.0)
    return p / p.sum(dim=-1, keepdim=True)


def centered_log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return center(normalized_probabilities(logits).log())


def compute_reverse_edge(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.new_empty(0)
    src, dst = edge_index
    key = src * num_nodes + dst
    sorted_key, perm = torch.sort(key)
    rev_key = dst * num_nodes + src
    pos = torch.searchsorted(sorted_key, rev_key)
    safe_pos = pos.clamp(max=sorted_key.numel() - 1)
    if not bool(((pos < sorted_key.numel()) & (sorted_key[safe_pos] == rev_key)).all()):
        raise ValueError("Every message edge must have a reverse edge.")
    return perm[safe_pos]


@dataclass
class MessageGraph:
    edge_index: torch.Tensor
    rev: torch.Tensor
    degree: torch.Tensor
    representatives: torch.Tensor
    pair_index: torch.Tensor
    num_nodes: int


def make_message_graph(edge_index: torch.Tensor, num_nodes: int) -> MessageGraph:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    ei = edge_index.long()
    if ei.numel() and (int(ei.min()) < 0 or int(ei.max()) >= num_nodes):
        raise ValueError("edge_index contains an invalid node index.")
    ei = ei[:, ei[0] != ei[1]]
    ei = dedup_edge_index(torch.cat((ei, ei.flip(0)), dim=1), num_nodes)
    rev = compute_reverse_edge(ei, num_nodes)
    ids = torch.arange(ei.shape[1], device=ei.device)
    representatives = ids[ids < rev]
    pair_index = torch.empty_like(ids)
    pair_ids = torch.arange(representatives.numel(), device=ei.device)
    pair_index[representatives] = pair_ids
    pair_index[rev[representatives]] = pair_ids
    degree = torch.bincount(ei[0], minlength=num_nodes)
    return MessageGraph(ei, rev, degree, representatives, pair_index, num_nodes)


def sum_at_nodes(values: torch.Tensor, indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return values.new_zeros((num_nodes, values.shape[-1])).index_add(0, indices, values)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)


@dataclass
class InferenceConfig:
    T: int = 10
    damping: str = "adaptive"
    eta: float = 0.5
    eta_min: float = 0.05
    eta_max: float = 1.0
    K: int = 10
    N_probe: int = 8
    lanczos_tol: float = 1e-6
    conv_tol: float = 1e-4
    conv_window: int = 5

    def __post_init__(self):
        if self.T < 0 or self.K < 1 or self.N_probe < 1:
            raise ValueError("T must be nonnegative; K and N_probe must be positive.")
        if self.damping not in ("adaptive", "fixed"):
            raise ValueError("damping must be adaptive or fixed.")
        if not 0.0 < self.eta_min <= self.eta_max <= 1.0:
            raise ValueError("Damping bounds must satisfy 0 < eta_min <= eta_max <= 1.")
        if not self.eta_min <= self.eta <= self.eta_max:
            raise ValueError("Fixed eta must lie within the damping bounds.")
        if self.lanczos_tol <= 0 or self.conv_tol <= 0 or self.conv_window < 1:
            raise ValueError("Numerical tolerances and conv_window must be positive.")


class JacobianOperator:
    def __init__(self, function: Callable, state: torch.Tensor, create_graph: bool):
        self.create_graph = create_graph
        with torch.enable_grad():
            self.state = state if create_graph and state.requires_grad else state.detach().requires_grad_(True)
            self.output = function(self.state)
            self.dual = torch.zeros_like(self.output, requires_grad=True)
            self.pullback = None
            if self.output.requires_grad:
                self.pullback = torch.autograd.grad(
                    self.output, self.state, grad_outputs=self.dual,
                    create_graph=True, retain_graph=True, allow_unused=True,
                )[0]

    def jvp(self, vector: torch.Tensor) -> torch.Tensor:
        if self.pullback is None or not self.pullback.requires_grad:
            return torch.zeros_like(vector)
        with torch.enable_grad():
            value = torch.autograd.grad(
                self.pullback, self.dual, grad_outputs=vector,
                create_graph=self.create_graph, retain_graph=True, allow_unused=True,
            )[0]
        return torch.zeros_like(vector) if value is None else value

    def vjp(self, vector: torch.Tensor) -> torch.Tensor:
        if self.pullback is None:
            return torch.zeros_like(vector)
        with torch.enable_grad():
            value = torch.autograd.grad(
                self.output, self.state, grad_outputs=vector,
                create_graph=self.create_graph, retain_graph=True, allow_unused=True,
            )[0]
        return torch.zeros_like(vector) if value is None else value

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        return vector - self.vjp(self.jvp(vector))


def lanczos_margin(
    operator: JacobianOperator,
    initial_vector: torch.Tensor,
    steps: int,
    tolerance: float,
) -> torch.Tensor:
    if initial_vector.numel() == 0:
        return initial_vector.new_ones(())
    norm = torch.linalg.vector_norm(initial_vector)
    if not bool(torch.isfinite(norm)) or float(norm.detach()) == 0.0:
        raise ValueError("Lanczos requires a finite nonzero starting vector.")
    q = initial_vector / norm
    basis = []
    diagonal = []
    off_diagonal = []
    previous_q = torch.zeros_like(q)
    previous_beta = q.new_zeros(())
    for k in range(min(steps, initial_vector.numel())):
        basis.append(q)
        product = operator.matvec(q)
        alpha = (q * product).sum()
        diagonal.append(alpha)
        residual = product - alpha * q - previous_beta * previous_q
        for _ in range(2):
            for basis_vector in basis:
                residual = residual - (basis_vector * residual).sum() * basis_vector
        beta = torch.linalg.vector_norm(residual)
        scale = max(1.0, float(torch.linalg.vector_norm(product.detach())))
        if not bool(torch.isfinite(alpha.detach())) or not bool(torch.isfinite(beta.detach())):
            raise FloatingPointError("Nonfinite values encountered during Lanczos estimation.")
        if k + 1 == min(steps, initial_vector.numel()) or float(beta.detach()) <= tolerance * scale:
            break
        off_diagonal.append(beta)
        previous_q, q = q, residual / beta
        previous_beta = beta
    tridiagonal = torch.diag(torch.stack(diagonal))
    if off_diagonal:
        off = torch.stack(off_diagonal)
        tridiagonal = tridiagonal + torch.diag(off, 1) + torch.diag(off, -1)
    return torch.linalg.eigvalsh(tridiagonal)[0]


def sensitivity_diagonal(
    operator: JacobianOperator,
    state: torch.Tensor,
    num_probes: int,
    probes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if probes is not None and tuple(probes.shape) != (num_probes,) + tuple(state.shape):
        raise ValueError("Probe tensor shape must be [N_probe, *state.shape].")
    diagonal = torch.zeros_like(state)
    for r in range(num_probes):
        if probes is None:
            z = torch.empty_like(state).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        else:
            z = probes[r].detach()
        response = operator.jvp(z)
        diagonal = diagonal + response.square()
    return diagonal / num_probes


class BPGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        edge_hidden_dim: int = 64,
        dropout: float = 0.6,
        edge_dropout: float = 0.0,
        cmp_margin: float = 0.05,
        tau: float = 0.1,
        inference: str = "bp",
        calibration: str = "sensitivity",
        edge_chunk_size: int = 1024,
        checkpoint_edges: bool = True,
        temperature_a: float = 0.1,
        temperature_b: float = -2.0,
    ):
        super().__init__()
        if inference not in ("bp", "mean_field", "unary"):
            raise ValueError("Unsupported inference map.")
        if calibration not in ("sensitivity", "global", "entropy", "none"):
            raise ValueError("Unsupported calibration map.")
        if min(in_dim, hidden_dim, edge_hidden_dim, num_classes, edge_chunk_size) < 1:
            raise ValueError("Model dimensions and edge_chunk_size must be positive.")
        if not 0 <= dropout < 1 or not 0 <= edge_dropout < 1:
            raise ValueError("Dropout probabilities must lie in [0, 1).")
        if cmp_margin <= 0 or tau <= 0:
            raise ValueError("cmp_margin and tau must be positive.")
        self.num_classes = num_classes
        self.inference = inference
        self.calibration = calibration
        self.cmp_margin = cmp_margin
        self.tau = tau
        self.edge_chunk_size = edge_chunk_size
        self.checkpoint_edges = checkpoint_edges
        self.encoder = MLP(in_dim, hidden_dim, num_classes, dropout)
        self.edge_mlp = MLP(2 * in_dim + 2, edge_hidden_dim, 1, edge_dropout)
        self.R_raw = nn.Parameter(torch.empty(num_classes, num_classes))
        nn.init.xavier_uniform_(self.R_raw)
        self.a_q = nn.Parameter(torch.tensor(float(temperature_a)))
        self.b_q = nn.Parameter(torch.tensor(float(temperature_b)))
        self._cached_edge_index = None
        self._cached_graph = None
        self._cached_version = None

    def _sym_R(self) -> torch.Tensor:
        return 0.5 * (self.R_raw + self.R_raw.t())

    def compatibility_prior(self) -> torch.Tensor:
        R = self._sym_R()
        if self.inference == "unary":
            return R.sum() * 0.0
        penalties = F.relu(R.diagonal()[:, None] - R + self.cmp_margin)
        mask = ~torch.eye(self.num_classes, dtype=torch.bool, device=R.device)
        return penalties[mask].sum()

    def _get_graph(self, edge_index: torch.Tensor, num_nodes: int) -> MessageGraph:
        if (
            self._cached_edge_index is not edge_index
            or self._cached_version != edge_index._version
            or self._cached_graph.num_nodes != num_nodes
        ):
            self._cached_graph = make_message_graph(edge_index, num_nodes)
            self._cached_edge_index = edge_index
            self._cached_version = edge_index._version
        return self._cached_graph

    def _edge_strength_chunk(
        self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, structural: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat((x[src] + x[dst], (x[src] - x[dst]).abs(), structural), dim=-1)
        return self.edge_mlp(inputs).squeeze(-1)

    def _potentials(self, x: torch.Tensor, graph: MessageGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        representatives = graph.representatives
        src, dst = graph.edge_index[:, representatives]
        log_degree = torch.log1p(graph.degree.to(x.dtype))
        structural = torch.stack(
            (log_degree[src] + log_degree[dst], (log_degree[src] - log_degree[dst]).abs()), dim=-1
        )
        values = []
        for start in range(0, representatives.numel(), self.edge_chunk_size):
            stop = start + self.edge_chunk_size
            arguments = (x, src[start:stop], dst[start:stop], structural[start:stop])
            if self.checkpoint_edges and self.training and torch.is_grad_enabled():
                value = checkpoint(self._edge_strength_chunk, *arguments, use_reentrant=False)
            else:
                value = self._edge_strength_chunk(*arguments)
            values.append(value)
        unique_w = torch.cat(values) if values else x.new_empty(0)
        w = unique_w[graph.pair_index]
        log_psi = w[:, None, None] * self._sym_R()[None, :, :]
        return w, log_psi

    def _incoming(self, u: torch.Tensor, log_psi: torch.Tensor) -> torch.Tensor:
        log_m = normalized_probabilities(u).log()
        return torch.logsumexp(log_m[:, :, None] + log_psi, dim=1)

    def _bp_map(
        self, u: torch.Tensor, logits: torch.Tensor, log_psi: torch.Tensor, graph: MessageGraph
    ) -> torch.Tensor:
        src, dst = graph.edge_index
        incoming = self._incoming(u, log_psi)
        total = sum_at_nodes(incoming, dst, graph.num_nodes)
        cavity = logits[src] + total[src] - incoming[graph.rev]
        return centered_log_probabilities(cavity)

    def _mean_field_map(
        self, u: torch.Tensor, logits: torch.Tensor, log_psi: torch.Tensor, graph: MessageGraph
    ) -> torch.Tensor:
        src, dst = graph.edge_index
        beliefs = normalized_probabilities(u)
        expected = torch.einsum("ed,edc->ec", beliefs[src], log_psi)
        evidence = logits + sum_at_nodes(expected, dst, graph.num_nodes)
        return centered_log_probabilities(evidence)

    def _beliefs(
        self, u: torch.Tensor, logits: torch.Tensor, log_psi: torch.Tensor, graph: MessageGraph
    ) -> torch.Tensor:
        if self.inference == "unary":
            return normalized_probabilities(logits)
        if self.inference == "mean_field":
            return normalized_probabilities(u)
        incoming = self._incoming(u, log_psi)
        return normalized_probabilities(logits + sum_at_nodes(incoming, graph.edge_index[1], graph.num_nodes))

    def _node_sensitivity(self, diagonal: torch.Tensor, graph: MessageGraph) -> torch.Tensor:
        if self.inference == "mean_field":
            return diagonal.sum(dim=-1)
        src, dst = graph.edge_index
        edge_sensitivity = diagonal.sum(dim=-1, keepdim=True)
        incident = sum_at_nodes(edge_sensitivity, src, graph.num_nodes)
        incident = incident + sum_at_nodes(edge_sensitivity, dst, graph.num_nodes)
        return incident.squeeze(-1) / graph.degree.to(diagonal.dtype).clamp(min=1.0)

    def _calibrate(self, beliefs: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entropy = -(beliefs * beliefs.log()).sum(dim=-1)
        entropy = entropy / math.log(self.num_classes) if self.num_classes > 1 else entropy * 0.0
        if self.calibration == "none":
            return beliefs, torch.ones_like(q), entropy
        if self.calibration == "global":
            temperature = (1.0 + F.softplus(self.b_q)).expand_as(q)
        else:
            feature = q if self.calibration == "sensitivity" else entropy
            temperature = 1.0 + F.softplus(self.a_q * feature + self.b_q)
        probabilities = torch.softmax(torch.log(beliefs + EPS) / temperature[:, None], dim=-1)
        return probabilities, temperature, entropy

    def forward(
        self,
        data,
        inf_cfg: InferenceConfig,
        edge_index: Optional[torch.Tensor] = None,
        rev: Optional[torch.Tensor] = None,
        x_override: Optional[torch.Tensor] = None,
        diagnostics: bool = False,
        initial_vector: Optional[torch.Tensor] = None,
        probes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Jacobian monitoring needs autograd; use torch.no_grad(), not torch.inference_mode().")
        x = data.x if x_override is None else x_override
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64 for Jacobian-based inference.")
        graph = self._get_graph(data.edge_index if edge_index is None else edge_index, data.num_nodes)
        logits = self.encoder(x)
        create_graph = torch.is_grad_enabled()
        zero = logits.new_zeros(())
        q = logits.new_zeros(data.num_nodes)
        margin_history = []
        damping_history = []
        change_history = []
        consecutive = 0
        first_convergence = 0
        sign_changes = zero.clone()
        previous_increment = None
        if self.inference == "unary":
            w = logits.new_empty(0)
            log_psi = logits.new_empty((0, self.num_classes, self.num_classes))
            u = centered_log_probabilities(logits)
            margin = logits.new_tensor(float("nan"))
            stability_loss = zero
            diagonal = torch.zeros_like(u)
            residual = logits.new_tensor(float("nan"))
        else:
            w, log_psi = self._potentials(x, graph)
            if self.inference == "bp":
                u = centered_log_probabilities(logits[graph.edge_index[0]])
                update = self._bp_map
            else:
                u = centered_log_probabilities(logits)
                update = self._mean_field_map
            if initial_vector is None:
                initial_vector = torch.randn_like(u)
            if initial_vector.shape != u.shape:
                raise ValueError("Lanczos initial_vector must have the same shape as the inference state.")
            initial_vector = initial_vector.detach()
            function = lambda state: update(state, logits, log_psi, graph)
            controller_logits, controller_psi = logits.detach(), log_psi.detach()
            control_function = lambda state: update(state, controller_logits, controller_psi, graph)
            for step in range(1, inf_cfg.T + 1):
                if inf_cfg.damping == "adaptive" and u.numel():
                    with torch.no_grad():
                        controller = JacobianOperator(control_function, u.detach(), create_graph=False)
                        delta = lanczos_margin(controller, initial_vector, inf_cfg.K, inf_cfg.lanczos_tol)
                        amplification = (1.0 - delta).clamp(min=0.0).sqrt()
                        eta = (1.0 / (1.0 + amplification)).clamp(inf_cfg.eta_min, inf_cfg.eta_max)
                        margin_history.append(delta.detach())
                    del controller
                elif inf_cfg.damping == "adaptive":
                    eta = logits.new_tensor(inf_cfg.eta_max)
                    margin_history.append(logits.new_ones(()))
                else:
                    eta = logits.new_tensor(inf_cfg.eta)
                eta = eta.detach()
                u_new = function(u)
                updated = centered_log_probabilities((1.0 - eta) * u + eta * u_new)
                damping_history.append(eta)
                if diagnostics:
                    increment = updated.detach() - u.detach()
                    change = increment.abs().mean() if increment.numel() else zero
                    change_history.append(change)
                    consecutive = consecutive + 1 if float(change) < inf_cfg.conv_tol else 0
                    if first_convergence == 0 and consecutive >= inf_cfg.conv_window:
                        first_convergence = step
                    if previous_increment is not None and step < inf_cfg.T:
                        sign_changes = sign_changes + (increment.sign() != previous_increment.sign()).sum()
                    previous_increment = increment
                u = updated
            if u.numel():
                terminal_function = function if create_graph else control_function
                terminal = JacobianOperator(terminal_function, u, create_graph=create_graph)
                margin = lanczos_margin(terminal, initial_vector, inf_cfg.K, inf_cfg.lanczos_tol)
                diagonal = sensitivity_diagonal(terminal, u, inf_cfg.N_probe, probes)
                q = self._node_sensitivity(diagonal, graph)
                stability_loss = F.softplus(self.tau - margin)
                residual = torch.linalg.vector_norm(terminal.output.detach() - u.detach())
                del terminal
            else:
                margin = logits.new_ones(())
                stability_loss = F.softplus(self.tau - margin)
                diagonal = torch.zeros_like(u)
                residual = zero
        beliefs = self._beliefs(u, logits, log_psi, graph)
        probabilities, temperature, entropy = self._calibrate(beliefs, q)
        if self.calibration == "sensitivity":
            uncertainty = q
        elif self.calibration == "entropy":
            uncertainty = entropy
        else:
            uncertainty = 1.0 - probabilities.max(dim=-1).values
        oscillation = sign_changes / (u.numel() * (inf_cfg.T - 2)) if (
            diagnostics and self.inference != "unary" and u.numel() and inf_cfg.T > 2
        ) else logits.new_tensor(float("nan"))
        extras = {
            "beliefs": beliefs,
            "messages": normalized_probabilities(u),
            "state": u,
            "q": q,
            "sensitivity_diagonal": diagonal,
            "temperature": temperature,
            "entropy": entropy,
            "uncertainty": uncertainty,
            "delta_hat": margin,
            "stability_loss": stability_loss,
            "undamped_residual": residual,
            "w": w,
            "R": self._sym_R(),
            "eta_history": torch.stack(damping_history) if damping_history else logits.new_empty(0),
            "margin_history": torch.stack(margin_history) if margin_history else logits.new_empty(0),
            "change_history": torch.stack(change_history) if change_history else logits.new_empty(0),
            "converged": logits.new_tensor(float(first_convergence > 0)),
            "convergence_iteration": logits.new_tensor(float(first_convergence) if first_convergence else float("nan")),
            "oscillation": oscillation,
        }
        return probabilities, extras


CertBP = BPGNN


def training_objective(
    model: BPGNN,
    probabilities: torch.Tensor,
    extras: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    lambda_brier: float,
    lambda_stab: float,
    lambda_cmp: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    p, y = probabilities[mask], labels[mask]
    if y.numel() == 0:
        raise ValueError("The training split is empty.")
    supervised = F.nll_loss(p.log(), y, reduction="sum")
    one_hot = F.one_hot(y, num_classes=model.num_classes).to(p.dtype)
    brier = (p - one_hot).square().sum()
    stability = extras["stability_loss"]
    compatibility = model.compatibility_prior()
    loss = supervised + lambda_brier * brier + lambda_stab * stability + lambda_cmp * compatibility
    return loss, {"supervised": supervised, "brier": brier, "stability": stability, "compatibility": compatibility}


@torch.no_grad()
def accuracy(probabilities: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return (probabilities[mask].argmax(dim=-1) == labels[mask]).to(torch.float64).mean().item()


@torch.no_grad()
def nll_loss(probabilities: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return F.nll_loss(probabilities[mask].log(), labels[mask]).item()


@torch.no_grad()
def ece_score(probabilities: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, n_bins: int = 15) -> float:
    p, y = probabilities[mask], labels[mask]
    confidence, prediction = p.max(dim=-1)
    correct = (prediction == y).to(p.dtype)
    bin_ids = torch.ceil(confidence * n_bins).long().sub(1).clamp(0, n_bins - 1)
    ece = p.new_zeros(())
    for bin_index in range(n_bins):
        members = bin_ids == bin_index
        if bool(members.any()):
            ece = ece + members.to(p.dtype).mean() * (correct[members].mean() - confidence[members].mean()).abs()
    return ece.item()


@torch.no_grad()
def error_detection_auroc(scores: torch.Tensor, errors: torch.Tensor) -> float:
    scores, errors = scores.flatten(), errors.flatten().bool()
    positives = errors.sum().item()
    negatives = errors.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    sorted_scores, order = torch.sort(scores)
    sorted_errors = errors[order].to(torch.float64)
    _, group, counts = torch.unique_consecutive(sorted_scores, return_inverse=True, return_counts=True)
    end_rank = counts.cumsum(0).to(torch.float64)
    start_rank = end_rank - counts.to(torch.float64) + 1.0
    average_rank = (start_rank + end_rank) / 2.0
    rank_sum = (average_rank[group] * sorted_errors).sum().item()
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


@torch.no_grad()
def prediction_metrics(
    probabilities: torch.Tensor, extras: Dict[str, torch.Tensor], labels: torch.Tensor, mask: torch.Tensor
) -> Dict[str, float]:
    p, y = probabilities[mask], labels[mask]
    if y.numel() == 0:
        raise ValueError("Cannot evaluate an empty split.")
    target = F.one_hot(y, num_classes=p.shape[-1]).to(p.dtype)
    return {
        "accuracy": accuracy(probabilities, labels, mask),
        "ece": ece_score(probabilities, labels, mask),
        "nll": nll_loss(probabilities, labels, mask),
        "brier": (p - target).square().sum(dim=-1).mean().item(),
        "auroc": error_detection_auroc(extras["uncertainty"][mask], p.argmax(-1) != y),
    }


@torch.no_grad()
def predict_probs(
    model: BPGNN, data, inf_cfg: InferenceConfig, diagnostics: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.eval()
    probabilities, extras = model(data, inf_cfg, diagnostics=diagnostics)
    return probabilities.detach(), {key: value.detach() for key, value in extras.items()}


def train_one_run(args) -> Dict[str, float]:
    from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, to_undirected

    device = torch.device(args.device)
    set_seed(args.seed, deterministic=args.deterministic)

    data, in_dim, num_classes = load_dataset(args.dataset, root=args.root, normalize_features=args.normalize_features)

    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=data.num_nodes)
    edge_index = dedup_edge_index(edge_index, data.num_nodes)

    data.edge_index = edge_index
    data.rev_edge = compute_reverse_edge(data.edge_index, data.num_nodes)
    data = data.to(device)

    train_mask, val_mask, test_mask = get_split_masks(data, args.split)

    if not all(bool(mask.any()) for mask in (train_mask, val_mask, test_mask)):
        raise ValueError("Training, validation, and test splits must all be nonempty.")
    if any(bool((a & b).any()) for a, b in ((train_mask, val_mask), (train_mask, test_mask), (val_mask, test_mask))):
        raise ValueError("Training, validation, and test masks must be disjoint.")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    data.x = data.x.to(dtype=dtype)
    data.y = data.y.long().view(-1)
    model = BPGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        edge_hidden_dim=args.edge_hidden_dim,
        dropout=args.dropout,
        edge_dropout=args.edge_dropout,
        cmp_margin=args.cmp_margin,
        tau=args.tau,
        inference=args.inference,
        calibration=args.calibration,
        edge_chunk_size=args.edge_chunk_size,
        checkpoint_edges=args.checkpoint_edges,
        temperature_a=args.temperature_a,
        temperature_b=args.temperature_b,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * args.cosine_min_lr_scale
    ) if args.use_cosine else None
    inf_cfg = InferenceConfig(
        T=args.T,
        damping=args.damping,
        eta=args.eta,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        K=args.K,
        N_probe=args.N_probe,
        lanczos_tol=args.lanczos_tol,
    )
    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        probabilities, extras = model(data, inf_cfg)
        loss, terms = training_objective(
            model, probabilities, extras, data.y, train_mask,
            args.lambda_brier, args.lambda_stab, args.lambda_cmp,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}.")
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        elif any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite parameter gradient at epoch {epoch}.")
        optimizer.step()
        training_loss = float(loss.detach())
        term_values = {key: float(value.detach()) for key, value in terms.items()}
        del probabilities, extras, loss, terms
        if scheduler is not None:
            scheduler.step()
        probabilities_eval, extras_eval = predict_probs(model, data, inf_cfg)
        val_accuracy = accuracy(probabilities_eval, data.y, val_mask)
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[{args.dataset} split={args.split} seed={args.seed}] "
                f"ep={epoch:04d} loss={training_loss:.6f} "
                f"sup={term_values['supervised']:.6f} brier={term_values['brier']:.6f} "
                f"stab={term_values['stability']:.6f} cmp={term_values['compatibility']:.6f} "
                f"tr={100 * accuracy(probabilities_eval, data.y, train_mask):.2f} "
                f"va={100 * val_accuracy:.2f} "
                f"vaNLL={nll_loss(probabilities_eval, data.y, val_mask):.6f} "
                f"vaECE={ece_score(probabilities_eval, data.y, val_mask):.6f} "
                f"delta_hat={extras_eval['delta_hat'].item():.6f}",
                flush=True,
            )
        del probabilities_eval, extras_eval
        if patience >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")
    model.load_state_dict(best_state)
    probabilities, extras = predict_probs(model, data, inf_cfg)
    metrics = prediction_metrics(probabilities, extras, data.y, test_mask)
    metrics["best_epoch"] = float(best_epoch)
    metrics["delta_hat"] = extras["delta_hat"].item()
    print(
        f"\n[FINAL] {args.dataset} split={args.split} seed={args.seed} best_epoch={best_epoch} "
        f"TEST Acc={100 * metrics['accuracy']:.2f} ECE={metrics['ece']:.6f} "
        f"NLL={metrics['nll']:.6f} Brier={metrics['brier']:.6f} AUROC={metrics['auroc']:.6f} "
        f"delta_hat={metrics['delta_hat']:.6f}",
        flush=True,
    )
    if args.diagnostics and args.inference != "unary":
        _, diagnostic = predict_probs(model, data, replace(inf_cfg, T=args.T_eval), diagnostics=True)
        metrics.update({
            "converged": diagnostic["converged"].item(),
            "convergence_iteration": diagnostic["convergence_iteration"].item(),
            "oscillation": diagnostic["oscillation"].item(),
            "diagnostic_delta_hat": diagnostic["delta_hat"].item(),
            "undamped_residual": diagnostic["undamped_residual"].item(),
        })
        print(
            f"[STABILITY] T_eval={args.T_eval} converged={int(metrics['converged'])} "
            f"first_iteration={metrics['convergence_iteration']:.0f} Osc={metrics['oscillation']:.6f} "
            f"delta_hat={metrics['diagnostic_delta_hat']:.6f} "
            f"undamped_residual={metrics['undamped_residual']:.6e}",
            flush=True,
        )
    elif args.inference == "unary":
        print("[STABILITY] N/A for unary-only inference.", flush=True)
    if args.save_path:
        path = args.save_path
        if args.runs > 1:
            base, extension = os.path.splitext(path)
            path = f"{base}_seed{args.seed}{extension or '.pt'}"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model": best_state, "args": vars(args), "metrics": metrics}, path)
    return metrics


def run_self_test() -> None:
    set_seed(7)
    torch.set_num_threads(1)
    x = torch.randn(5, 4, dtype=torch.float64)
    edges = torch.tensor([[0, 1, 1, 2, 2, 0, 2, 3, 0, 4], [1, 0, 2, 1, 0, 2, 3, 2, 0, 4]])
    data = SimpleNamespace(x=x, edge_index=edges, num_nodes=5)
    graph = make_message_graph(edges, 5)
    assert not bool((graph.edge_index[0] == graph.edge_index[1]).any())
    assert torch.equal(graph.rev[graph.rev], torch.arange(graph.rev.numel()))
    model = BPGNN(4, 8, 3, edge_hidden_dim=7, dropout=0.0, checkpoint_edges=False).double()
    logits = model.encoder(x)
    w, log_psi = model._potentials(x, graph)
    torch.testing.assert_close(w, w[graph.rev], rtol=0.0, atol=0.0)
    torch.testing.assert_close(log_psi, log_psi[graph.rev].transpose(1, 2), rtol=0.0, atol=0.0)
    state = centered_log_probabilities(logits[graph.edge_index[0]])
    function = lambda u: model._bp_map(u, logits, log_psi, graph)
    operator = JacobianOperator(function, state, create_graph=True)
    dense = torch.autograd.functional.jacobian(function, state).reshape(state.numel(), state.numel())
    vector = torch.randn_like(state)
    torch.testing.assert_close(operator.jvp(vector).flatten(), dense @ vector.flatten(), rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(operator.vjp(vector).flatten(), dense.t() @ vector.flatten(), rtol=1e-8, atol=1e-10)
    margin = lanczos_margin(operator, vector, state.numel(), 1e-12)
    exact = 1.0 - torch.linalg.matrix_norm(dense, ord=2).square()
    torch.testing.assert_close(margin, exact, rtol=1e-7, atol=1e-9)
    eps = 1e-5
    u0 = state.detach().clone().requires_grad_(True)
    parameter = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    direction = torch.randn_like(u0)
    probes = torch.empty((4,) + tuple(u0.shape), dtype=u0.dtype).bernoulli_(0.5).mul_(2).sub_(1)

    def differentiable_statistic(u, parameter_value):
        map_function = lambda v: torch.tanh(parameter_value * v)
        op = JacobianOperator(map_function, u, create_graph=True)
        delta = lanczos_margin(op, vector, 4, 1e-12)
        diagonal = sensitivity_diagonal(op, u, 4, probes)
        return F.softplus(0.1 - delta) + 0.01 * diagonal.sum()

    value = differentiable_statistic(u0, parameter)
    state_gradient, parameter_gradient = torch.autograd.grad(value, (u0, parameter))
    numeric_state = (differentiable_statistic(u0 + eps * direction, parameter) - differentiable_statistic(u0 - eps * direction, parameter)) / (2 * eps)
    numeric_parameter = (differentiable_statistic(u0, parameter + eps) - differentiable_statistic(u0, parameter - eps)) / (2 * eps)
    torch.testing.assert_close((state_gradient * direction).sum(), numeric_state, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(parameter_gradient, numeric_parameter, rtol=1e-4, atol=1e-6)
    configuration = InferenceConfig(T=3, K=4, N_probe=4)
    model.train()
    probabilities, extras = model(data, configuration)
    labels = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.ones(5, dtype=torch.bool)
    loss, _ = training_objective(model, probabilities, extras, labels, mask, 0.5, 0.1, 0.01)
    loss.backward()
    for parameter_value in model.parameters():
        assert parameter_value.grad is not None
        assert bool(torch.isfinite(parameter_value.grad).all())
    assert float(extras["q"][4].detach()) == 0.0
    torch.testing.assert_close(extras["beliefs"][4], normalized_probabilities(model.encoder(x))[4])
    assert not extras["eta_history"].requires_grad
    assert extras["stability_loss"].requires_grad
    assert extras["q"].requires_grad
    assert bool((extras["eta_history"] >= configuration.eta_min).all())
    assert bool((extras["eta_history"] <= configuration.eta_max).all())
    assert bool((extras["temperature"] > 1.0).all())
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(5, dtype=x.dtype))
    assert torch.equal(probabilities.argmax(-1), extras["beliefs"].argmax(-1))
    probabilities_eval, extras_eval = predict_probs(model, data, configuration, diagnostics=True)
    assert not probabilities_eval.requires_grad
    assert not extras_eval["q"].requires_grad
    empty_data = SimpleNamespace(x=x, edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=5)
    empty_p, empty_extras = predict_probs(model, empty_data, configuration)
    assert bool(torch.isfinite(empty_p).all()) and float(empty_extras["q"].sum()) == 0.0
    for inference in ("mean_field", "unary"):
        variant = BPGNN(4, 8, 3, dropout=0.0, inference=inference, checkpoint_edges=False).double()
        p, extra = variant(data, configuration)
        variant_loss, _ = training_objective(variant, p, extra, labels, mask, 0.5, 0.1, 0.01)
        variant_loss.backward()
        assert bool(torch.isfinite(p).all())
    print("[SELF-TEST] PASS: cavity Jacobian, JVP/VJP, Ritz estimate, higher-order gradients, calibration, damping, isolated nodes, empty graphs, and inference variants.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stability-aware BPGNN with differentiable terminal Jacobian monitoring.")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--normalize_features", dest="normalize_features", action="store_true", default=True)
    parser.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--edge_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--edge_dropout", type=float, default=0.0)
    parser.add_argument("--edge_chunk_size", type=int, default=1024)
    parser.add_argument("--checkpoint_edges", dest="checkpoint_edges", action="store_true", default=True)
    parser.add_argument("--no_checkpoint_edges", dest="checkpoint_edges", action="store_false")
    parser.add_argument("--inference", choices=["bp", "mean_field", "unary"], default="bp")
    parser.add_argument("--calibration", choices=["sensitivity", "global", "entropy", "none"], default="sensitivity")
    parser.add_argument("--cmp_margin", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--temperature_a", type=float, default=0.1)
    parser.add_argument("--temperature_b", type=float, default=-2.0)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--T_eval", type=int, default=100)
    parser.add_argument("--damping", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--eta_min", type=float, default=0.05)
    parser.add_argument("--eta_max", type=float, default=1.0)
    parser.add_argument("--K", "--lanczos_steps", dest="K", type=int, default=10)
    parser.add_argument("--N_probe", "--n_probe", dest="N_probe", type=int, default=8)
    parser.add_argument("--lanczos_tol", type=float, default=1e-6)
    parser.add_argument("--lambda_brier", type=float, default=1.0)
    parser.add_argument("--lambda_stab", type=float, default=0.1)
    parser.add_argument("--lambda_cmp", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_cosine", dest="use_cosine", action="store_true", default=False)
    parser.add_argument("--no_use_cosine", dest="use_cosine", action="store_false")
    parser.add_argument("--cosine_min_lr_scale", type=float, default=0.1)
    parser.add_argument("--diagnostics", dest="diagnostics", action="store_true", default=True)
    parser.add_argument("--no_diagnostics", dest="diagnostics", action="store_false")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.dataset:
        parser.error("--dataset is required unless --self_test is used.")
    if min(args.runs, args.epochs, args.patience, args.log_every) < 1:
        parser.error("runs, epochs, patience, and log_every must be positive.")
    if min(args.lambda_brier, args.lambda_stab, args.lambda_cmp, args.weight_decay, args.grad_clip) < 0 or args.lr <= 0:
        parser.error("Loss weights, weight_decay, and grad_clip must be nonnegative; lr must be positive.")
    if args.diagnostics and args.T_eval < 3:
        parser.error("T_eval must be at least 3 when diagnostics are enabled.")
    results = []
    for run in range(args.runs):
        run_args = copy.copy(args)
        run_args.seed = args.seed + run
        results.append(train_one_run(run_args))
    if len(results) > 1:
        print(f"\n[SUMMARY] dataset={args.dataset} split={args.split} runs={len(results)}")
        for metric in results[0]:
            values = np.asarray([result[metric] for result in results], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                print(f"{metric}=N/A")
            elif metric == "converged":
                print(f"convergence_rate={values.mean():.6f}")
            else:
                deviation = values.std(ddof=1) if values.size > 1 else 0.0
                print(f"{metric}={values.mean():.6f} +/- {deviation:.6f} n={values.size}")


if __name__ == "__main__":
    main()
