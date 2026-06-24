from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class KernelShape:
    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int = 2
    dtype: str = "bfloat16"
    activation: str = "silu"


@dataclass(frozen=True)
class ParallelPoint:
    tp: int
    ep: int

    @property
    def world_size(self) -> int:
        return self.tp * self.ep

    @property
    def label(self) -> str:
        return f"tp{self.tp}-ep{self.ep}"


@dataclass(frozen=True)
class SlurmConfig:
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    time: str = "02:00:00"
    cpus_per_task: int = 8
    gpus_per_node: int = 8
    gpu_type: str | None = None
    use_gres: bool = False
    mem: str = "64G"
    constraint: str | None = None
    modules: list[str] = field(default_factory=list)
    venv: str | None = None
    workdir: str | None = None
    uv_env_dir: str | None = None
    extra_sbatch_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StudyConfig:
    study_name: str
    all2all_backend: str
    kernel_shapes: list[KernelShape]
    parallel_points: list[ParallelPoint]
    alphas: list[float]
    tokens: list[int]
    sweep_modes: list[str] = field(default_factory=lambda: ["one_at_a_time", "full_factorial"])
    warmup_iters: int = 10
    measure_iters: int = 50
    seed: int = 0
    apply_router_weight_on_input: bool = False
    hot_expert_count: int = 1
    output_root: str = "runs"
    routing_weight_mode: str = "uniform"
    collect_buckets: bool = True
    bucket_profile_iters: int = 5
    bucket_full_events: bool = True
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    baseline: dict[str, Any] = field(default_factory=dict)

    def max_tokens(self) -> int:
        return max(self.tokens)

    def baseline_alpha(self) -> float:
        if "alpha" in self.baseline:
            return float(self.baseline["alpha"])
        return float(self.alphas[0])

    def baseline_tokens(self) -> int:
        if "tokens" in self.baseline:
            return int(self.baseline["tokens"])
        return int(self.tokens[0])

    def baseline_parallel(self) -> ParallelPoint:
        baseline_parallel = self.baseline.get("parallel")
        if baseline_parallel:
            return ParallelPoint(tp=int(baseline_parallel["tp"]), ep=int(baseline_parallel["ep"]))
        return self.parallel_points[0]


@dataclass(frozen=True)
class BenchmarkCondition:
    shape: KernelShape
    parallel: ParallelPoint
    tokens: int
    alpha: float
    mode: str

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["shape"] = asdict(self.shape)
        out["parallel"] = asdict(self.parallel)
        return out


VALID_SWEEP_MODES = {"one_at_a_time", "full_factorial"}


def _require_keys(data: dict[str, Any], keys: list[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def load_study_config(path: str | Path) -> StudyConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    _require_keys(raw, ["study_name", "all2all_backend", "kernel_shapes", "parallel_points", "alphas", "tokens"])

    shapes = [KernelShape(**item) for item in raw["kernel_shapes"]]
    parallel_points = [ParallelPoint(**item) for item in raw["parallel_points"]]
    slurm = SlurmConfig(**(raw.get("slurm") or {}))

    cfg = StudyConfig(
        study_name=raw["study_name"],
        all2all_backend=raw["all2all_backend"],
        kernel_shapes=shapes,
        parallel_points=parallel_points,
        alphas=[float(x) for x in raw["alphas"]],
        tokens=[int(x) for x in raw["tokens"]],
        sweep_modes=list(raw.get("sweep_modes") or ["one_at_a_time", "full_factorial"]),
        warmup_iters=int(raw.get("warmup_iters", 10)),
        measure_iters=int(raw.get("measure_iters", 50)),
        seed=int(raw.get("seed", 0)),
        apply_router_weight_on_input=bool(raw.get("apply_router_weight_on_input", False)),
        hot_expert_count=int(raw.get("hot_expert_count", 1)),
        output_root=str(raw.get("output_root", "runs")),
        routing_weight_mode=str(raw.get("routing_weight_mode", "uniform")),
        collect_buckets=bool(raw.get("collect_buckets", True)),
        bucket_profile_iters=int(raw.get("bucket_profile_iters", 5)),
        bucket_full_events=bool(raw.get("bucket_full_events", True)),
        slurm=slurm,
        baseline=dict(raw.get("baseline") or {}),
    )
    invalid_modes = [m for m in cfg.sweep_modes if m not in VALID_SWEEP_MODES]
    if invalid_modes:
        raise ValueError(f"Unsupported sweep_modes: {invalid_modes}")
    return cfg


def make_conditions(cfg: StudyConfig, parallel: ParallelPoint) -> list[BenchmarkCondition]:
    conditions: list[BenchmarkCondition] = []
    baseline_alpha = cfg.baseline_alpha()
    baseline_tokens = cfg.baseline_tokens()
    baseline_parallel = cfg.baseline_parallel()

    for shape in cfg.kernel_shapes:
        if "one_at_a_time" in cfg.sweep_modes:
            for alpha in cfg.alphas:
                conditions.append(BenchmarkCondition(shape=shape, parallel=parallel, tokens=baseline_tokens, alpha=alpha, mode="one_at_a_time"))
            for tokens in cfg.tokens:
                conditions.append(BenchmarkCondition(shape=shape, parallel=parallel, tokens=tokens, alpha=baseline_alpha, mode="one_at_a_time"))
            if parallel == baseline_parallel:
                pass
            else:
                conditions.append(BenchmarkCondition(shape=shape, parallel=parallel, tokens=baseline_tokens, alpha=baseline_alpha, mode="one_at_a_time"))

        if "full_factorial" in cfg.sweep_modes:
            for alpha in cfg.alphas:
                for tokens in cfg.tokens:
                    conditions.append(BenchmarkCondition(shape=shape, parallel=parallel, tokens=tokens, alpha=alpha, mode="full_factorial"))

    dedup: dict[tuple[str, int, float, int, int, str], BenchmarkCondition] = {}
    for cond in conditions:
        key = (cond.shape.name, cond.tokens, cond.alpha, cond.parallel.tp, cond.parallel.ep, cond.mode)
        dedup[key] = cond
    return list(dedup.values())
