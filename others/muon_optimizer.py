"""Muon optimizer with Polar Express for computing the polar decomposition.

Muon (MomentUm Orthogonalized by Newton-schulz) replaces the standard gradient
update with the polar factor of the momentum buffer, computing the nearest
semi-orthogonal matrix to the gradient direction.

The polar factor is approximated using the Polar Express algorithm, which applies
a composition of optimal degree-5 polynomials using only matrix-matrix products.
The products run in float32 by default (``polar_dtype``). bfloat16 is the
alternative, with little margin: the iteration tolerates about 1 % of relative
error per product before a singular value leaves its stable range and blows up.

A guard (``polar_guard``) leaves out, for one step, the update of any matrix
whose polar factor comes out non-finite or oversized, and reports it.

References:
    - Muon: https://github.com/KellerJordan/Muon
    - Polar Express: https://arxiv.org/abs/2505.16932
"""

import logging
import math
import socket
from itertools import repeat

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.adamw import adamw as functional_adamw

# ---------------------------------------------------------------------------
# Polar Express: precomputed optimal degree-5 polynomial coefficients
# Computed with l=1e-3, 10 iterations, safety_factor_eps=1e-2, cushion=0.02
# From https://github.com/NoahAmsel/PolarExpress
# ---------------------------------------------------------------------------
_POLAR_EXPRESS_COEFFS_RAW = [
    (8.287212018145630, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.947849916737911, 0.544843108292660),
    (3.948690853482295, -2.908902115962949, 0.551819139437014),
    (3.318419657370602, -2.488488024314874, 0.510048940123720),
    (2.300652019954817, -1.668903984574749, 0.418807311952567),
    (1.891301407787398, -1.267995827194587, 0.376804089485248),
    (1.875001480853448, -1.250001645399949, 0.375000164547425),
    (1.875, -1.25, 0.375),
]

# Safety factor for numerical stability (exclude last polynomial)
_POLAR_EXPRESS_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in _POLAR_EXPRESS_COEFFS_RAW[:-1]
] + [_POLAR_EXPRESS_COEFFS_RAW[-1]]


_POLAR_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _polar_express(G: torch.Tensor, steps: int, dtype: torch.dtype) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.to(dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = _POLAR_EXPRESS_COEFFS[:steps] + list(
        repeat(_POLAR_EXPRESS_COEFFS[-1], max(0, steps - len(_POLAR_EXPRESS_COEFFS)))
    )
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def polar_express(G: torch.Tensor, steps: int = 6, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Approximate polar(G) = U @ V^T via Polar Express (Algorithm 1).

    Applies a composition of optimal degree-5 polynomials to the normalized
    input. All computation is done in ``dtype``.

    Args:
        G: Input tensor with ndim >= 2.
        steps: Number of polynomial iterations (recommended: 5-8).
        dtype: Precision of the products (float32 or bfloat16).

    Returns:
        Approximation to polar(G) in ``dtype``.
    """
    return _polar_express(G, steps, dtype)


@torch.compile
def guarded_polar_express(G: torch.Tensor, steps: int, dtype: torch.dtype, max_mean_sq: float):
    """``polar_express`` with a check of its output, per matrix.

    A polar factor has singular values near 1, so its squared Frobenius norm is
    about ``min(m, n)``. When a singular value escapes the iteration's stable
    range it ends in the thousands, or as inf / NaN. A matrix whose mean squared
    singular value ``||X||_F^2 / min(m, n)`` is above ``max_mean_sq`` or
    non-finite is returned as zeros.

    Returns:
        The polar factor (zeros where the check failed) and the mean squared
        singular value of each matrix, shape ``G.shape[:-2]``, before zeroing.
    """
    X = _polar_express(G, steps, dtype)
    # Two single-axis sums: one sum over both axes makes inductor warn about an
    # unknown symbol range when the leading dimension is dynamic (torch 2.4).
    mean_sq = X.square().sum(-1, dtype=torch.float32).sum(-1) / min(G.size(-2), G.size(-1))
    ok = mean_sq <= max_mean_sq  # False for NaN too
    return torch.where(ok[..., None, None], X, 0.0), mean_sq


_UPDATE_SCALES = {"none", "moonlight", "keller"}


def _update_scale(p: torch.Tensor, chunks: int, mode: str) -> float:
    """Multiplier of the polar update of ``p`` (see :class:`Muon`)."""
    if mode == "none" or p.ndim < 2:
        return 1.0
    rows = p.shape[0] // chunks
    cols = math.prod(p.shape[1:])
    if mode == "moonlight":
        return 0.2 * math.sqrt(max(rows, cols))
    if mode == "keller":
        return math.sqrt(max(1.0, rows / cols))
    raise ValueError(f"unknown update_scale {mode!r}")


class Muon(Optimizer):
    """Muon optimizer: MomentUm Orthogonalized by Polar Express.

    Applies the Muon update (momentum + polar decomposition) to parameter groups
    with ``use_muon=True``, and a standard AdamW update to groups with
    ``use_muon=False``.

    The parameter routing (which params get Muon vs AdamW) is handled externally
    by ``configure_optimizers`` in the training module. Each param group dict
    should include a ``use_muon`` boolean key. Muon groups may additionally set
    ``muon_chunks`` (int, default 1): their tensors fuse that many logical
    matrices along dim 0 (e.g. 3 for a fused QKV weight) and each row-block is
    orthogonalized separately.

    Weight decay on BOTH sides is lr-coupled, PyTorch AdamW convention:
    ``p *= 1 - lr * wd`` (the Complete(d)P weight-decay scaling rules assume
    this convention).

    Update scale (``update_scale``, per Muon group): the polar factor of an
    ``m x n`` matrix has per-element RMS ``1/sqrt(max(m, n))``, so the raw
    update (``"none"``) moves wide/tall matrices less than square ones.
    ``"moonlight"`` multiplies it by ``0.2 * sqrt(max(m, n))`` (Moonshot's
    "Muon is Scalable for LLM Training"): every matrix then gets an update RMS
    of ``0.2 * lr``, close to AdamW's, so Muon and AdamW groups can share one
    learning rate and weight decay. ``"keller"`` is ``sqrt(max(1, m / n))``
    (KellerJordan/Muon). ``m`` is the row count of one logical matrix
    (``rows / muon_chunks``) and ``n`` the product of the other dimensions.

    Args:
        params: Iterable of parameter groups with ``use_muon`` flags.
        lr: Learning rate for Muon parameters.
        momentum: Momentum coefficient for Muon (SGD-style accumulator:
            ``buf = momentum * buf + grad``, not an EMA).
        nesterov: Use Nesterov-style momentum for Muon.
        polar_express_steps: Number of Polar Express iterations.
        polar_dtype: ``"float32"`` (default) or ``"bfloat16"``: precision of
            the polar products. Under ``allow_tf32`` float32 products run in
            TF32 on the GPU, which is accurate enough.
        polar_guard: Leave out, for one step, the update of a matrix whose
            polar factor is non-finite or oversized (weight decay still
            applies, the momentum is kept; a non-finite momentum is reset).
            Each event is logged, see ``pop_guard_report``.
        polar_guard_max_rms: "Oversized" = RMS singular value of the polar
            factor above this (it is 1 for an exact polar factor).
        weight_decay: Coupled weight decay for Muon parameters.
        update_scale: ``"none"`` (default), ``"moonlight"`` or ``"keller"``;
            see above.
        adamw_lr: Learning rate for AdamW parameters (groups with
            ``use_muon=False`` that do not set their own ``lr``).
        adamw_betas: Beta coefficients for AdamW.
        adamw_eps: Epsilon for AdamW numerical stability.
        adamw_weight_decay: Coupled weight decay for AdamW parameters (groups
            with ``use_muon=False`` that do not set their own
            ``weight_decay``).
        adamw_impl: auto (fused on CUDA, foreach on CPU), fused, foreach, or
            loop. All implementations retain the legacy checkpoint schema.
        distributed_muon_group_size: Number of ranks per host sharing whole
            matrix polar work. Requires already-synchronized DDP gradients;
            all ranks must have the same pattern of missing gradients. Falls
            back to replicated execution outside distributed training.
        distributed_muon_bucket_cap_mb: Maximum send workspace per collective.
            Two slots each hold this amount plus group_size times as much for
            received updates. Momentum remains replicated for exact resume.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        polar_express_steps: int = 6,
        weight_decay: float = 0.0,
        update_scale: str = "none",
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
        adamw_impl: str = "auto",
        distributed_muon_group_size: int = 1,
        distributed_muon_bucket_cap_mb: float = 16.0,
        polar_dtype: str = "float32",
        polar_guard: bool = True,
        polar_guard_max_rms: float = 1.5,
    ):
        if polar_dtype not in _POLAR_DTYPES:
            raise ValueError(f"polar_dtype must be one of {sorted(_POLAR_DTYPES)}, got {polar_dtype!r}")
        if not math.isfinite(polar_guard_max_rms) or polar_guard_max_rms <= 0:
            raise ValueError("polar_guard_max_rms must be positive and finite")
        if adamw_impl not in {"auto", "fused", "foreach", "loop"}:
            raise ValueError("adamw_impl must be auto, fused, foreach, or loop")
        if (isinstance(distributed_muon_group_size, bool)
                or int(distributed_muon_group_size) != distributed_muon_group_size
                or distributed_muon_group_size < 1):
            raise ValueError("distributed_muon_group_size must be a positive integer")
        if not math.isfinite(distributed_muon_bucket_cap_mb) or distributed_muon_bucket_cap_mb <= 0:
            raise ValueError("distributed_muon_bucket_cap_mb must be positive and finite")
        if update_scale not in _UPDATE_SCALES:
            raise ValueError(f"update_scale must be one of {sorted(_UPDATE_SCALES)}, got {update_scale!r}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            polar_express_steps=polar_express_steps,
            weight_decay=weight_decay,
            update_scale=update_scale,
            use_muon=True,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        # AdamW groups that do not set their own lr / weight decay take the
        # adamw_* values, not the Muon defaults filled in by Optimizer.
        params = list(params)
        if params and isinstance(params[0], dict):
            params = [dict(g) for g in params]
            for g in params:
                if not g.get("use_muon", True):
                    g.setdefault("lr", adamw_lr)
                    g.setdefault("weight_decay", adamw_weight_decay)
        super().__init__(params, defaults)
        # Execution settings are deliberately NOT parameter-group fields. Loading
        # an old checkpoint must preserve the requested implementation, while
        # restoring all mathematical hyperparameters from the checkpoint.
        self.adamw_impl = adamw_impl
        self.distributed_muon_group_size = int(distributed_muon_group_size)
        self._distributed_muon_bucket_cap_bytes = max(1, int(distributed_muon_bucket_cap_mb * 2**20))
        self._distributed_muon_group = None
        self._distributed_muon_group_ranks = None
        self._distributed_muon_plans = {}
        self._distributed_muon_workspaces = {}
        self._distributed_muon_collective_count = 0
        self._adamw_step_tensors = {}
        # The polar precision and the guard are execution settings too: a run
        # resumed from an older checkpoint gets them, and they add nothing to
        # the saved state.
        self.polar_dtype = _POLAR_DTYPES[polar_dtype]
        self.polar_guard = bool(polar_guard)
        self._guard_max_mean_sq = float(polar_guard_max_rms) ** 2
        self._param_names = {}
        self._guard_entries = []    # (parameter, mean squared singular values) of the step being run
        self._guard_layout = None   # one slot per logical matrix, built at the first guarded step
        self._guard_pending = None  # last step's result, still on the device
        self._guard_peak = None     # per slot: largest value of an applied update since the last report
        self._guard_counts = {}     # matrix label -> updates left out since the last report
        self._guard_steps = 0
        logging.info(
            f"[muon] polar products in {polar_dtype}; guard "
            + (f"on (a matrix update whose polar factor is non-finite or has an RMS singular value above "
               f"{polar_guard_max_rms:g} is left out and logged)" if self.polar_guard else "off")
        )

    def set_parameter_names(self, named_parameters):
        """Names for the guard's log lines: ``(name, parameter)`` pairs, e.g. ``model.named_parameters()``."""
        self._param_names = {id(p): str(n) for n, p in named_parameters}

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._adamw_step_tensors.clear()
        self._distributed_muon_plans.clear()
        # Legacy checkpoints use Python integers. Accept native Adam-style
        # scalar tensors too, without a device synchronization in each step.
        for state in self.state.values():
            if "step" in state:
                state["step"] = int(state["step"])
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if (self.distributed_muon_group_size > 1 and dist.is_initialized()
                and dist.get_world_size() > 1):
            self._init_distributed_muon_group()

        # Outcome of the previous step's guard. Read here, where the caller has
        # just synchronized with the device (GradScaler reads its inf check
        # before calling step), so it costs no wait of its own.
        self._guard_collect()

        for group in self.param_groups:
            if group.get("use_muon", True):
                self._muon_step(group)
            else:
                self._adamw_step(group)

        if self._guard_entries:
            self._guard_commit()

        return loss

    def _polar_update(self, p, g, steps):
        """Polar factor of ``g``, whose leading dimensions index the logical matrices of ``p``."""
        if not self.polar_guard:
            return polar_express(g, steps=steps, dtype=self.polar_dtype)
        update, mean_sq = guarded_polar_express(g, steps, self.polar_dtype, self._guard_max_mean_sq)
        self._guard_entries.append((p, mean_sq))
        return update

    def _build_guard_layout(self):
        slots, index, size = [], {}, 0
        for group in self.param_groups:
            if not group.get("use_muon", True):
                continue
            chunks = group.get("muon_chunks", 1)
            for p in group["params"]:
                if p.ndim < 2:
                    continue
                shape = (chunks, p.shape[0] // chunks, *p.shape[1:]) if chunks > 1 else tuple(p.shape)
                count = math.prod(shape[:-2])
                slots.append((p, size, count))
                index[id(p)] = (size, count)
                size += count
        return {"slots": slots, "index": index, "size": size, "partial": {}}

    def _guard_commit(self):
        """Keep this step's guard result on the device; no read here."""
        entries, self._guard_entries = self._guard_entries, []
        layout = self._guard_layout
        if layout is None:
            layout = self._guard_layout = self._build_guard_layout()
        values = torch.cat([v.reshape(-1) for _, v in entries])
        slots = layout["slots"]
        if len(entries) != len(slots) or any(e[0] is not s[0] for e, s in zip(entries, slots)):
            # Some matrices were not updated here this step (no gradient, or
            # owned by another rank): their slots stay at 0.
            key = tuple(id(p) for p, _ in entries)
            where = layout["partial"].get(key)
            if where is None:
                where = torch.cat([
                    torch.arange(start, start + count, device=values.device)
                    for start, count in (layout["index"][i] for i in key)
                ])
                layout["partial"][key] = where
            values = values.new_zeros(layout["size"]).index_copy_(0, where, values)
        bad = ~(values <= self._guard_max_mean_sq)
        self._guard_pending = (values, bad, bad.sum())
        applied = values.masked_fill(bad, 0.0)
        self._guard_peak = applied if self._guard_peak is None else torch.maximum(self._guard_peak, applied)
        self._guard_steps += 1

    def _guard_label(self, p, count, j):
        name = self._param_names.get(id(p)) or f"<parameter of shape {tuple(p.shape)}>"
        return name if count == 1 else f"{name}[part {j + 1}/{count}]"

    def _guard_collect(self):
        """Read the last step's guard result; log and count what it left out."""
        pending, self._guard_pending = self._guard_pending, None
        if pending is None:
            return
        values, bad, n_bad = pending
        if not int(n_bad):  # the only device read of an ordinary step
            return
        values, bad = values.tolist(), bad.tolist()
        described, resets = [], []
        for p, start, count in self._guard_layout["slots"]:
            hit = [j for j in range(count) if bad[start + j]]
            if not hit:
                continue
            state = self.state[p]
            buf = state.get("momentum_buffer")
            if buf is not None and not bool(torch.isfinite(buf).all()):
                buf.zero_()
                resets.append(self._guard_label(p, 1, 0))
            for j in hit:
                label = self._guard_label(p, count, j)
                self._guard_counts[label] = self._guard_counts.get(label, 0) + 1
                v = values[start + j]
                rms = f"{math.sqrt(v):.4g}" if math.isfinite(v) and v >= 0 else str(v)
                described.append(f"{label} (RMS singular value {rms}, update #{state.get('step', '?')})")
        rank = dist.get_rank() if dist.is_initialized() else 0
        shown = "; ".join(described[:8]) + (f"; and {len(described) - 8} more" if len(described) > 8 else "")
        logging.warning(
            f"[muon-guard] rank {rank}: the previous optimizer step left out {len(described)} matrix update(s) "
            f"whose polar factor was non-finite or oversized: {shown}"
            + (f". Non-finite momentum reset to zero: {'; '.join(resets)}" if resets else "")
        )

    def pop_guard_report(self):
        """Guard activity since the last call, then reset (one device read).

        Returns ``None`` when the guard is off or has not run, else a dict:
        ``steps`` (guarded optimizer steps), ``skipped`` (``{matrix: updates
        left out}``), ``peak_rms`` / ``peak_name`` (largest RMS singular value
        of an applied update, and its matrix; about 1 when all is well).
        """
        self._guard_collect()
        if self._guard_steps == 0:
            return None
        peak_rms, peak_name = 0.0, None
        if self._guard_peak is not None:
            i = int(self._guard_peak.argmax())
            peak_rms = math.sqrt(max(0.0, float(self._guard_peak[i])))
            for p, start, count in self._guard_layout["slots"]:
                if start <= i < start + count:
                    peak_name = self._guard_label(p, count, i - start)
                    break
        report = {"steps": self._guard_steps, "skipped": dict(self._guard_counts),
                  "peak_rms": peak_rms, "peak_name": peak_name}
        self._guard_peak, self._guard_counts, self._guard_steps = None, {}, 0
        return report

    def _muon_step(self, group):
        if self._distributed_muon_group is not None:
            self._distributed_muon_step(group)
            return
        lr = group["lr"]
        beta = group["momentum"]
        nesterov = group["nesterov"]
        wd = group["weight_decay"]
        steps = group["polar_express_steps"]
        chunks = group.get("muon_chunks", 1)
        scale_mode = group.get("update_scale", "none")

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(grad)

            state["step"] += 1
            buf = state["momentum_buffer"]
            buf.mul_(beta).add_(grad)

            if nesterov:
                g = grad.add(buf, alpha=beta)
            else:
                g = buf

            if p.ndim >= 2:
                if chunks > 1:
                    # fused logical matrices (e.g. QKV): polar per row-block
                    g = g.reshape(chunks, g.size(0) // chunks, *g.shape[1:])
                    update = self._polar_update(p, g, steps).reshape(p.shape)
                else:
                    update = self._polar_update(p, g, steps)
            else:
                update = g

            if wd != 0:
                p.mul_(1 - lr * wd)

            p.add_(update, alpha=-lr * _update_scale(p, chunks, scale_mode))

    def _init_distributed_muon_group(self):
        """Discover node-local groups, including interleaved global ranks.

        Requires DDP-replicated parameters AND already-reduced gradients. This
        is compute sharding of whole matrices, not support for FSDP parameters.
        Every world rank must construct/step this optimizer in the same order.
        """
        if self._distributed_muon_group is not None:
            return
        schema = (self._distributed_muon_bucket_cap_bytes, [
            (g.get("use_muon", True), g.get("muon_chunks", 1), g["polar_express_steps"],
             [(tuple(p.shape), str(p.dtype), p.device.type, p.is_contiguous()) for p in g["params"]])
            for g in self.param_groups
        ])
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, (socket.gethostname(), self.distributed_muon_group_size, schema))
        if any(size != self.distributed_muon_group_size or spec != schema for _, size, spec in records):
            raise RuntimeError("distributed Muon requires identical parameter layouts, polar steps, group sizes, and bucket capacities on all ranks")
        if any(not p.is_contiguous() for g in self.param_groups if g.get("use_muon", True) for p in g["params"]):
            raise RuntimeError("distributed Muon requires contiguous parameters")
        for group in self.param_groups:
            if group.get("use_muon", True):
                self._validate_distributed_chunks(group)
        devices = {p.device for g in self.param_groups for p in g["params"]}
        if len(devices) > 1:
            raise RuntimeError("distributed Muon requires one device per rank")
        hosts = {}
        for rank, (host, _, _) in enumerate(records):
            hosts.setdefault(host, []).append(rank)
        size = self.distributed_muon_group_size
        if any(len(ranks) % size for ranks in hosts.values()):
            raise RuntimeError("distributed_muon_group_size must divide the number of ranks on every host")
        rank_groups = [ranks[start:start + size] for ranks in hosts.values() for start in range(0, len(ranks), size)]
        for ranks in sorted(rank_groups, key=lambda ranks: ranks[0]):
            pg = dist.new_group(ranks=ranks)
            if dist.get_rank() in ranks:
                self._distributed_muon_group = pg
                self._distributed_muon_group_ranks = ranks
                self._distributed_muon_local_index = ranks.index(dist.get_rank())

    @staticmethod
    def _validate_distributed_chunks(group):
        # Validate on every rank before owners start computing: otherwise one
        # owner's reshape error would strand its peers inside an all-gather.
        chunks = group.get("muon_chunks", 1)
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
            raise RuntimeError("distributed Muon requires positive integer muon_chunks")
        if any(p.ndim >= 2 and p.shape[0] % chunks for p in group["params"]):
            raise RuntimeError("distributed Muon requires muon_chunks to divide each matrix's first dimension")

    @staticmethod
    def _polar_cost(p, steps, chunks):
        if p.ndim < 2:
            return p.numel()
        shape = (chunks, p.shape[0] // chunks, *p.shape[1:]) if chunks > 1 else p.shape
        m, n = sorted(shape[-2:])
        return math.prod(shape[:-2]) * steps * (4 * m * m * n + 2 * m * m * m)

    def _distributed_workspace(self, device, dtype, count, slot):
        key = (device, dtype, slot)
        send, receive = self._distributed_muon_workspaces.get(key, (None, None))
        size = len(self._distributed_muon_group_ranks)
        if send is None or send.numel() < count:
            send = torch.empty(count, device=device, dtype=dtype)
            receive = torch.empty(size * count, device=device, dtype=dtype)
            self._distributed_muon_workspaces[key] = (send, receive)
        return send[:count], receive[:size * count]

    def _distributed_muon_step(self, group):
        beta, lr, wd = group["momentum"], group["lr"], group["weight_decay"]
        steps, chunks = group["polar_express_steps"], group.get("muon_chunks", 1)
        scale_mode = group.get("update_scale", "none")
        plan_key = (id(group), steps, chunks)
        if plan_key not in self._distributed_muon_plans:
            self._validate_distributed_chunks(group)
        active = [p for p in group["params"] if p.grad is not None]
        # All ranks retain exactly the same momentum/step state. Foreach uses
        # the same separate multiply and add as the serial reference.
        buffers, grads = [], []
        for p in active:
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            state["step"] += 1
            buffers.append(state["momentum_buffer"])
            grads.append(p.grad)
        if buffers:
            torch._foreach_mul_(buffers, beta)
            torch._foreach_add_(buffers, grads)

        if plan_key not in self._distributed_muon_plans:
            ordered = sorted(enumerate(group["params"]), key=lambda pair: (-self._polar_cost(pair[1], steps, chunks), pair[0]))
            by_dtype = {}
            # Include parameters with None gradients: collective layouts must
            # remain fixed when an entire parameter is unused on a step.
            for _, p in ordered:
                dtype = self.polar_dtype if p.ndim >= 2 else p.dtype
                by_dtype.setdefault(dtype, []).append(p)
            self._distributed_muon_plans[plan_key] = by_dtype

        size = len(self._distributed_muon_group_ranks)
        local = self._distributed_muon_local_index
        pending = None
        sequence = 0

        def finish(entry):
            work, receive, params, offset, count = entry
            work.wait()
            received = receive.view(size, count)
            for owner, p in enumerate(params):
                take = min(count, max(0, p.numel() - offset))
                if take and p.grad is not None:
                    p.view(-1)[offset:offset + take].add_(
                        received[owner, :take], alpha=-lr * _update_scale(p, chunks, scale_mode))

        for dtype, params in self._distributed_muon_plans[plan_key].items():
            cap = max(1, self._distributed_muon_bucket_cap_bytes // torch.empty((), dtype=dtype).element_size())
            for start in range(0, len(params), size):
                round_params = params[start:start + size]
                owned = None
                if local < len(round_params):
                    p = round_params[local]
                    if p.grad is not None:
                        buf = self.state[p]["momentum_buffer"]
                        g = p.grad.add(buf, alpha=beta) if group["nesterov"] else buf
                        if p.ndim >= 2:
                            logical = g.reshape(chunks, g.shape[0] // chunks, *g.shape[1:]) if chunks > 1 else g
                            owned = self._polar_update(p, logical, steps).reshape(-1)
                        else:
                            owned = g.reshape(-1)
                for p in round_params:
                    if wd != 0 and p.grad is not None:
                        p.mul_(1 - lr * wd)
                maximum = max(p.numel() for p in round_params)
                for offset in range(0, maximum, cap):
                    count = min(cap, maximum - offset)
                    send, receive = self._distributed_workspace(round_params[0].device, dtype, count, sequence % 2)
                    take = min(count, max(0, owned.numel() - offset)) if owned is not None else 0
                    if take:
                        send[:take].copy_(owned[offset:offset + take])
                    if take < count:
                        send[take:].zero_()
                    work = dist.all_gather_into_tensor(receive, send, group=self._distributed_muon_group, async_op=True)
                    self._distributed_muon_collective_count += 1
                    # Two workspaces permit the next polar calculation / send
                    # to overlap the preceding NCCL collective. Never overwrite
                    # send/receive storage while a collective still uses it.
                    if pending is not None:
                        finish(pending)
                    pending = (work, receive, round_params, offset, count)
                    sequence += 1
        if pending is not None:
            finish(pending)

    def _adamw_step(self, group):
        if self.adamw_impl == "loop":
            self._adamw_loop_step(group)
            return
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        fused = self.adamw_impl == "fused" or (self.adamw_impl == "auto" and all(p.is_cuda for p in params))
        if fused and not all(p.is_cuda for p in params):
            raise RuntimeError("adamw_impl=fused requires CUDA parameters; use auto or foreach on CPU")
        grads, avgs, squares, counters = [], [], [], []
        for p in params:
            if p.grad.is_sparse or p.is_complex():
                raise RuntimeError("Muon AdamW requires dense real gradients")
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            device = p.device if fused else torch.device("cpu")
            counter = self._adamw_step_tensors.get(p)
            if counter is None or counter.device != device:
                counter = torch.tensor(float(state["step"]), dtype=torch.float32, device=device)
                self._adamw_step_tensors[p] = counter
            state["step"] += 1
            grads.append(p.grad)
            avgs.append(state["exp_avg"])
            squares.append(state["exp_avg_sq"])
            counters.append(counter)
        beta1, beta2 = group.get("adamw_betas", (0.9, 0.95))
        functional_adamw(
            params, grads, avgs, squares, [], counters,
            amsgrad=False, beta1=beta1, beta2=beta2, lr=group["lr"],
            weight_decay=group["weight_decay"], eps=group.get("adamw_eps", 1e-8),
            maximize=False, foreach=not fused, fused=fused,
        )

    def _adamw_loop_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group.get("adamw_betas", (0.9, 0.95))
        eps = group.get("adamw_eps", 1e-8)
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            # A caller may switch implementations between steps. The native
            # counter must restart from the saved integer after a loop update.
            self._adamw_step_tensors.pop(p, None)

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if wd != 0:
                p.mul_(1 - lr * wd)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            p.addcdiv_(exp_avg, denom, value=-step_size)
