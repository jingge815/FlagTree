# flagtree tle
from __future__ import annotations

from typing import List, Tuple

import triton.language.core as tl
from triton._C.libtriton import ir


def _unwrap_task_grid_constexpr(value):
    if isinstance(value, tl.constexpr):
        value = value.value
    if isinstance(value, tl.tuple):
        return tuple(_unwrap_task_grid_constexpr(item) for item in value.values)
    if isinstance(value, (tuple, list)):
        return type(value)(_unwrap_task_grid_constexpr(item) for item in value)
    if value is None or isinstance(value, str):
        return value
    value = tl._unwrap_if_constexpr(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_unwrap_task_grid_constexpr(item) for item in value)
    return value


def _validate_public_name(kind, value):
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"tle.task_grid {kind} name must be a Python identifier, got {value!r}")
    if value.startswith("_"):
        raise ValueError(f"tle.task_grid {kind} name must not start with '_', got {value!r}")
    if value in {"commit", "epoch", "fields", "name", "scope", "shape", "tile_id", "type"}:
        raise ValueError(f"tle.task_grid {kind} name {value!r} is reserved")


def _validate_shape(shape):
    shape = _unwrap_task_grid_constexpr(shape)
    if shape is None:
        return None
    if isinstance(shape, int):
        shape = (shape, )
    elif isinstance(shape, (tuple, list)):
        shape = tuple(_unwrap_task_grid_constexpr(dim) for dim in shape)
    else:
        raise ValueError("tle.task_grid shape must be a compile-time int or tuple/list of ints")
    if not shape:
        raise ValueError("tle.task_grid shape must not be empty")
    for dim in shape:
        if not isinstance(dim, int):
            raise ValueError(f"tle.task_grid shape dimensions must be compile-time ints, got {type(dim).__name__}")
        if dim <= 0:
            raise ValueError(f"tle.task_grid shape dimensions must be positive, got {dim}")
    return tuple(shape)


def _get_builder_method(_semantic, name):
    builder = getattr(_semantic, "builder", None)
    if builder is None:
        return None
    return getattr(builder, name, None)


class task_grid_value_type(tl.base_type):

    def __init__(self, scope: str, name: str, shape, fields):
        self.scope = scope
        self.name = name
        self.shape = None if shape is None else tuple(shape)
        self.fields = list(fields)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple["task_grid_value", int]:
        values = {}
        for name, ty in self.fields:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values[name] = value
        return task_grid_value(self.scope, self.name, self.shape, values), cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        for _, ty in self.fields:
            ty._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        shape = "dynamic" if self.shape is None else "x".join(str(dim) for dim in self.shape)
        fields = "_".join(f"{name}_{ty.mangle()}" for name, ty in self.fields)
        return f"task_grid_{self.scope}_{self.name}_{shape}_{fields}"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.scope == other.scope and self.name == other.name
                and self.shape == other.shape and self.fields == other.fields)

    def __str__(self) -> str:
        fields = ", ".join(f"{name}: {ty}" for name, ty in self.fields)
        return f"task_grid<{self.scope}, {self.name}, shape={self.shape}, {fields}>"


class task_grid_value(tl.base_value):

    def __init__(self, scope: str, name: str, shape, fields, epoch=None):
        super().__init__()
        self.scope = scope
        self.name = name
        self.shape = None if shape is None else tuple(shape)
        self.fields = dict(fields)
        self.epoch = epoch
        for field_name, value in self.fields.items():
            setattr(self, field_name, value)

    def _flatten_ir(self, handles) -> None:
        for field in self.fields.values():
            field._flatten_ir(handles)

    @property
    def type(self):
        return task_grid_value_type(self.scope, self.name, self.shape,
                                    [(name, value.type) for name, value in self.fields.items()])

    def _field_handles(self):
        return [field.handle for field in self.fields.values()]

    def _field_names(self):
        return list(self.fields.keys())

    def _ir_name(self):
        return self.name

    @tl.builtin
    def tile_id(self, _semantic=None):
        create = _get_builder_method(_semantic, "create_task_grid_tile_id")
        shape = list(self.shape or ())
        rank = len(shape)
        if create is None:
            if _semantic is None:
                return tuple()
            values = tuple(_semantic.to_tensor(0) for _ in range(rank))
        else:
            handles = create(self._field_handles(), self.scope, self._ir_name(), self._field_names(), shape)
            values = tuple(tl.tensor(handle, tl.int32) for handle in handles)
        if rank == 1:
            return values[0]
        return values

    @tl.builtin
    def commit(self, tile=None, _semantic=None):
        tile = _unwrap_task_grid_constexpr(tile)
        if tile is not None:
            if self.shape is None:
                raise ValueError("tle.task_grid.commit(tile=...) requires task_grid shape")
            if len(self.shape) == 1 and not isinstance(tile, (tuple, list, tl.tuple)):
                tile = (tile, )
            elif isinstance(tile, tl.tuple):
                tile = tuple(tile.values)
            elif isinstance(tile, (tuple, list)):
                tile = tuple(tile)
            else:
                raise ValueError("tle.task_grid.commit tile must be a scalar for rank-1 grids or a tuple/list")
            if len(tile) != len(self.shape):
                raise ValueError(f"tle.task_grid.commit tile rank must be {len(self.shape)}, got {len(tile)}")

        create = _get_builder_method(_semantic, "create_task_grid_commit")
        if create is not None:
            create(self._field_handles(), self.scope, self._ir_name(), self._field_names())


@tl.builtin
def task_grid(
    *,
    scope="device",
    name=None,
    shape=None,
    epoch=None,
    _semantic=None,
    **fields,
) -> task_grid_value:
    """
    Create a typed task-grid dependency edge descriptor.

    ``task_grid`` is a scheduler dependency key space. User kernels can query
    the current logical tile with ``grid.tile_id()`` and publish completion with
    ``grid.commit()``. Acquire/wait/release belong to ``tle.pipe`` and are not
    part of this API.
    """
    scope = _unwrap_task_grid_constexpr(scope)
    name = _unwrap_task_grid_constexpr(name)
    shape = _validate_shape(shape)

    if scope not in {"device", "cta"}:
        raise ValueError(f"tle.task_grid scope must be 'device' or 'cta', got {scope!r}")
    if not isinstance(name, str):
        raise ValueError(f"tle.task_grid name must be a compile-time string, got {type(name).__name__}")
    _validate_public_name("edge", name)
    if not fields:
        raise ValueError("tle.task_grid requires at least one payload field")

    for field_name, field in fields.items():
        _validate_public_name("field", field_name)
        if field is None:
            raise ValueError(f"tle.task_grid field {field_name!r} must not be None")
        if not hasattr(field, "handle"):
            raise ValueError(
                f"tle.task_grid field {field_name!r} must be a Triton/TLE value with an IR handle, got {type(field).__name__}"
            )

    create = _get_builder_method(_semantic, "create_task_grid_create")
    if create is not None:
        create([field.handle for field in fields.values()], scope, name, list(fields.keys()), list(shape or ()))
    return task_grid_value(scope, name, shape, fields, epoch=epoch)
