# flagtree tle
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class _AllTiles:
    """Sentinel for read-all task-grid dimensions."""

    def __repr__(self) -> str:
        return "ALL"


ALL = _AllTiles()
_MISSING = object()
_RESERVED_NAMES = {"commit", "epoch", "fields", "name", "scope", "shape", "tile_id", "type"}


def _validate_public_name(kind: str, value: Any) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"tlem.{kind} name must be a Python identifier, got {value!r}")
    if value.startswith("_"):
        raise ValueError(f"tlem.{kind} name must not start with '_', got {value!r}")
    if value in _RESERVED_NAMES:
        raise ValueError(f"tlem.{kind} name {value!r} is reserved")
    return value


def _normalize_shape(kind: str, shape: Any) -> tuple[int, ...]:
    if isinstance(shape, int):
        shape = (shape, )
    elif isinstance(shape, (tuple, list)):
        shape = tuple(shape)
    else:
        raise ValueError(f"tlem.{kind} shape/domain must be a static int or tuple/list of ints")
    if not shape:
        raise ValueError(f"tlem.{kind} shape/domain must not be empty")
    for dim in shape:
        if not isinstance(dim, int):
            raise ValueError(f"tlem.{kind} shape/domain dimensions must be ints, got {type(dim).__name__}")
        if dim <= 0:
            raise ValueError(f"tlem.{kind} shape/domain dimensions must be positive, got {dim}")
    return tuple(shape)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    mlir_type: str


@dataclass(frozen=True)
class GraphArgSpec:
    name: str
    mlir_type: str


@dataclass(frozen=True)
class GridSpec:
    name: str
    shape: tuple[int, ...]
    fields: tuple[FieldSpec, ...]
    scope: str = "device"


@dataclass(frozen=True)
class AffineMapSpec:
    domain_rank: int
    projections: tuple[int, ...]
    wildcard_dims: tuple[int, ...] = ()

    @property
    def target_rank(self) -> int:
        return len(self.projections) + len(self.wildcard_dims)


@dataclass(frozen=True)
class TaskMapSpec:
    grid: str
    map: AffineMapSpec


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: tuple[int, ...]
    reads: tuple[TaskMapSpec, ...]
    writes: tuple[TaskMapSpec, ...]
    fn: Any = None


def affine_map(domain_rank: int, *outputs: Any) -> AffineMapSpec:
    """Create a canonical affine task map.

    ``outputs`` is a sequence of task-domain dimension indices or ``tlem.ALL``.
    For example, ``affine_map(2, 0, 1)`` represents ``(d0, d1) -> (d0, d1)``
    and ``affine_map(1, 0, ALL)`` represents ``(d0) -> (d0, *)``.
    """
    if len(outputs) == 1 and isinstance(outputs[0], (tuple, list)):
        outputs = tuple(outputs[0])
    if not isinstance(domain_rank, int) or domain_rank <= 0:
        raise ValueError("tlem.affine_map domain_rank must be a positive int")
    if not outputs:
        raise ValueError("tlem.affine_map requires at least one output dimension")

    projections = []
    wildcard_dims = []
    for grid_dim, output in enumerate(outputs):
        if output is ALL:
            wildcard_dims.append(grid_dim)
            continue
        if not isinstance(output, int):
            raise ValueError("tlem.affine_map outputs must be task dimension indices or tlem.ALL")
        if output < 0 or output >= domain_rank:
            raise ValueError("tlem.affine_map output dimension is outside the task domain rank")
        projections.append(output)
    return AffineMapSpec(domain_rank=domain_rank, projections=tuple(projections), wildcard_dims=tuple(wildcard_dims))


def _normalize_fields(fields: Any) -> tuple[FieldSpec, ...]:
    if isinstance(fields, Mapping):
        items = tuple(fields.items())
    elif isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)):
        items = tuple(fields)
    else:
        raise ValueError("tlem.grid fields must be a mapping or sequence of (name, mlir_type) pairs")
    if not items:
        raise ValueError("tlem.grid requires at least one field")

    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("tlem.grid fields must contain (name, mlir_type) pairs")
        name = _validate_public_name("field", item[0])
        mlir_type = item[1]
        if not isinstance(mlir_type, str) or not mlir_type:
            raise ValueError(f"tlem.grid field {name!r} requires a non-empty MLIR type string")
        if name in seen:
            raise ValueError(f"tlem.grid field {name!r} is duplicated")
        seen.add(name)
        normalized.append(FieldSpec(name=name, mlir_type=mlir_type))
    return tuple(normalized)


def _format_array_i64(values: Sequence[int]) -> str:
    return "array<i64: " + ", ".join(str(value) for value in values) + ">"


def _format_affine_map(spec: AffineMapSpec) -> str:
    dims = ", ".join(f"d{i}" for i in range(spec.domain_rank))
    results = ", ".join(f"d{i}" for i in spec.projections)
    return f"affine_map<({dims}) -> ({results})>"


def _format_task_map(spec: TaskMapSpec) -> str:
    parts = [f'grid = "{spec.grid}"', f"map = {_format_affine_map(spec.map)}"]
    if spec.map.wildcard_dims:
        parts.append(f"wildcard_dims = {_format_array_i64(spec.map.wildcard_dims)}")
    return "{" + ", ".join(parts) + "}"


class MegaGraph:
    """Host-side container for formal TLE megakernel task graph metadata."""

    def __init__(self, name: str = "mega_graph"):
        self.name = _validate_public_name("graph", name)
        self.args: list[GraphArgSpec] = []
        self.grids: list[GridSpec] = []
        self.tasks: list[TaskSpec] = []

    def arg(self, name: str, mlir_type: str) -> GraphArgSpec:
        name = _validate_public_name("arg", name)
        if any(existing.name == name for existing in self.args):
            raise ValueError(f"tlem.graph duplicate arg name {name!r}")
        if not isinstance(mlir_type, str) or not mlir_type:
            raise ValueError(f"tlem.graph arg {name!r} requires a non-empty MLIR type string")
        spec = GraphArgSpec(name=name, mlir_type=mlir_type)
        self.args.append(spec)
        return spec

    def grid(self, name: str, *, shape: Any, fields: Any, scope: str = "device") -> GridSpec:
        name = _validate_public_name("grid", name)
        if scope not in {"device", "cta"}:
            raise ValueError(f"tlem.grid scope must be 'device' or 'cta', got {scope!r}")
        if any(existing.name == name for existing in self.grids):
            raise ValueError(f"tlem.graph duplicate grid name {name!r}")
        spec = GridSpec(name=name, shape=_normalize_shape("grid", shape), fields=_normalize_fields(fields), scope=scope)
        self.grids.append(spec)
        return spec

    def add_grid(self, grid: GridSpec) -> GridSpec:
        if not isinstance(grid, GridSpec):
            raise TypeError("tlem.add_grid expects a GridSpec returned by graph.grid")
        if any(existing.name == grid.name for existing in self.grids):
            raise ValueError(f"tlem.graph duplicate grid name {grid.name!r}")
        self.grids.append(grid)
        return grid

    def task(self, fn: Any = None, *, name: str | None = None, domain: Any, reads=_MISSING,
             writes=_MISSING) -> TaskSpec:
        task_name = name
        if task_name is None:
            if isinstance(fn, str):
                task_name = fn
            elif fn is not None:
                task_name = getattr(fn, "__name__", None)
        task_name = _validate_public_name("task", task_name)
        if any(existing.name == task_name for existing in self.tasks):
            raise ValueError(f"tlem.graph duplicate task name {task_name!r}")
        if reads is _MISSING:
            raise ValueError("tlem.task requires explicit reads, use reads={} for no inputs")
        if writes is _MISSING:
            raise ValueError("tlem.task requires explicit writes")

        domain_shape = _normalize_shape("task", domain)
        spec = TaskSpec(
            name=task_name,
            domain=domain_shape,
            reads=self._normalize_task_maps(reads, len(domain_shape), "read", allow_wildcards=True),
            writes=self._normalize_task_maps(writes, len(domain_shape), "write", allow_wildcards=False),
            fn=fn,
        )
        if not spec.writes:
            raise ValueError("tlem.task writes must contain at least one map")
        self.tasks.append(spec)
        return spec

    def add_task(self, task: TaskSpec) -> TaskSpec:
        if not isinstance(task, TaskSpec):
            raise TypeError("tlem.add_task expects a TaskSpec returned by graph.task")
        if any(existing.name == task.name for existing in self.tasks):
            raise ValueError(f"tlem.graph duplicate task name {task.name!r}")
        self.tasks.append(task)
        return task

    def _grid_by_name(self) -> dict[str, GridSpec]:
        return {grid.name: grid for grid in self.grids}

    def _normalize_task_maps(self, maps: Any, domain_rank: int, kind: str,
                             allow_wildcards: bool) -> tuple[TaskMapSpec, ...]:
        if isinstance(maps, Mapping):
            items = tuple(maps.items())
        elif isinstance(maps, Sequence) and not isinstance(maps, (str, bytes)):
            items = []
            for item in maps:
                if isinstance(item, TaskMapSpec):
                    items.append((item.grid, item.map))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    items.append((item[0], item[1]))
                else:
                    raise ValueError(f"tlem.task {kind}s must be a mapping or sequence of (grid, map) pairs")
            items = tuple(items)
        else:
            raise ValueError(f"tlem.task {kind}s must be a mapping or sequence of (grid, map) pairs")

        grids = self._grid_by_name()
        normalized = []
        for grid_ref, map_spec in items:
            if isinstance(grid_ref, GridSpec):
                grid_name = grid_ref.name
            else:
                grid_name = _validate_public_name("grid", grid_ref)
            grid = grids.get(grid_name)
            if grid is None:
                raise ValueError(f"tlem.task {kind} map references unknown grid {grid_name!r}")
            if not isinstance(map_spec, AffineMapSpec):
                raise ValueError(f"tlem.task {kind} map for grid {grid_name!r} must be created with tlem.affine_map")
            if map_spec.domain_rank != domain_rank:
                raise ValueError(
                    f"tlem.task {kind} map for grid {grid_name!r} has domain rank {map_spec.domain_rank}, expected {domain_rank}"
                )
            if map_spec.target_rank != len(grid.shape):
                raise ValueError(
                    f"tlem.task {kind} map for grid {grid_name!r} targets rank {map_spec.target_rank}, expected {len(grid.shape)}"
                )
            if map_spec.wildcard_dims and not allow_wildcards:
                raise ValueError("tlem.task write maps must not contain tlem.ALL")
            normalized.append(TaskMapSpec(grid=grid_name, map=map_spec))
        return tuple(normalized)

    def validate(self) -> None:
        if not self.grids:
            raise ValueError("tlem.graph requires at least one grid")
        if not self.tasks:
            raise ValueError("tlem.graph requires at least one task")

    def _task_callee_name(self, task: TaskSpec) -> str | None:
        if task.fn is None:
            return None
        if isinstance(task.fn, str):
            if task.fn == task.name:
                return None
            return _validate_public_name("callee", task.fn.removeprefix("@"))
        repr_fn = getattr(task.fn, "repr", None)
        if callable(repr_fn):
            callee = repr_fn(None)
            return _validate_public_name("callee", str(callee).removeprefix("@"))
        callee = getattr(task.fn, "__name__", None)
        if callee is None:
            raise ValueError(f"tlem.task {task.name!r} callable does not expose __name__ for callee emission")
        return _validate_public_name("callee", callee)

    def to_mlir_function(self, func_name: str | None = None) -> str:
        """Lower the canonical host graph to a single TLE graph function."""
        self.validate()
        func_name = _validate_public_name("function", func_name or self.name)

        args = [(arg.name, arg.mlir_type) for arg in self.args]
        seen_args = {arg.name for arg in self.args}
        arg_by_field: dict[tuple[str, str], str] = {}
        for grid in self.grids:
            for field in grid.fields:
                arg_name = f"{grid.name}_{field.name}"
                if arg_name in seen_args:
                    raise ValueError(f"tlem.graph arg name {arg_name!r} collides with generated grid field argument")
                seen_args.add(arg_name)
                arg_by_field[(grid.name, field.name)] = arg_name
                args.append((arg_name, field.mlir_type))

        lines = []
        arg_text = ", ".join(f"%{name}: {mlir_type}" for name, mlir_type in args)
        lines.append(f"  tt.func @{func_name}({arg_text}) {{")
        for grid in self.grids:
            operands = ", ".join(f"%{arg_by_field[(grid.name, field.name)]}" for field in grid.fields)
            field_names = ", ".join(f'"{field.name}"' for field in grid.fields)
            field_types = ", ".join(field.mlir_type for field in grid.fields)
            lines.append(
                f"    tle.task_grid.create {operands} "
                f'{{field_names = [{field_names}], grid_name = "{grid.name}", scope = "{grid.scope}", '
                f"shape = {_format_array_i64(grid.shape)}}} : {field_types}"
            )
        for task in self.tasks:
            reads = ", ".join(_format_task_map(item) for item in task.reads)
            writes = ", ".join(_format_task_map(item) for item in task.writes)
            attrs = []
            callee = self._task_callee_name(task)
            if callee is not None:
                attrs.append(f"callee = @{callee}")
            attrs.extend([
                f"domain_shape = {_format_array_i64(task.domain)}",
                f"reads = [{reads}]",
                f'task_name = "{task.name}"',
                f"writes = [{writes}]",
            ])
            lines.append(f"    tle.task.declare {{{', '.join(attrs)}}}")
        lines.append("    tt.return")
        lines.append("  }")
        return "\n".join(lines) + "\n"

    def to_mlir(self, func_name: str | None = None) -> str:
        """Lower the canonical host graph to verifiable TLE task metadata IR."""
        lines = ['module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {']
        lines.append(self.to_mlir_function(func_name).rstrip())
        lines.append("}")
        return "\n".join(lines) + "\n"

    def compile(self, *args, **kwargs):
        self.validate()
        raise NotImplementedError("TLE mega graph scheduler codegen is not implemented yet; use to_mlir() for metadata")


def mega_graph(name: str = "mega_graph") -> MegaGraph:
    """Create a host-side TLE megakernel task graph."""
    return MegaGraph(name=name)
