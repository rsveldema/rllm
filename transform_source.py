from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

import offloadize_common as common


class TransformSourceRewriter:
    def __init__(
        self,
        src_text: str,
        rel_path: str,
        backend_namespace: str,
        include_line: str,
        emit_named_kernels: bool = False,
        on_emit_loop: Callable[[common.LoopContext], None] | None = None,
        parfor_dump_dir: Path | None = None,
        symbol_values: dict[str, str] | None = None,
    ):
        self.in_lines = src_text.splitlines()
        self.rel_path = rel_path
        self.backend_namespace = backend_namespace
        self.include_line = include_line
        self.emit_named_kernels = emit_named_kernels
        self.on_emit_loop = on_emit_loop
        self.parfor_dump_dir = parfor_dump_dir
        self.symbol_values = symbol_values

        self.out_lines: list[str] = []
        self.loop_stack: list[common.LoopContext] = []
        self.changed = False
        self.defined_macros = common._defined_macros_for_backend(backend_namespace)
        self.conditional_stack: list[tuple[bool, bool]] = []
        self.active_code = True
        self.active_offload_param_types: dict[str, str] = {}
        self.active_offload_param_lines: list[str] = []
        self.collecting_offload_params = False
        self.active_raw_body_lines: list[str] = []
        self.pending_parfor_invocation: str | None = None
        self.collecting_param_names: list[str] = []
        self.collecting_param_types: dict[str, str] = {}
        self.collecting_param_lines: list[str] = []
        self.collecting_shared_vars = False
        self.collecting_shared_names: list[str] = []
        self.collecting_shared_types: dict[str, str] = {}
        self.pending_shared_vars: dict[str, str] | None = None

    def transform(self) -> tuple[str, bool]:
        for lineno, line in enumerate(self.in_lines, start=1):
            self._process_line(lineno, line)

        if self.changed:
            self.out_lines = common.inject_kernel_include(self.out_lines, self.include_line)

        return "\n".join(self.out_lines) + "\n", self.changed

    def _process_line(self, lineno: int, line: str) -> None:
        stripped = line.strip()

        if stripped.startswith("#") and self._handle_preprocessor(line, stripped):
            return

        if not self.active_code:
            self.out_lines.append(line)
            return

        if self.collecting_offload_params:
            self._process_offload_param_line(line)
            return

        if self.collecting_shared_vars:
            self._process_shared_var_line(line)
            return

        if self._try_start_offload_params(line):
            return

        if self._try_start_shared_vars(line):
            return

        parsed = common.parse_macro_invocation(line)
        if stripped == "ENDFOR" and self.loop_stack:
            self._finish_loop()
            return

        if parsed is None:
            self._append_to_current(line)
            return

        macro, args, indent = parsed
        self.pending_parfor_invocation = line.strip()

        if self._try_start_1d_param(lineno, macro, args, indent):
            return
        if self._try_start_2d(lineno, macro, args, indent):
            return
        if self._try_start_2d_param(lineno, macro, args, indent):
            return
        if self._try_start_3d_param(lineno, macro, args, indent):
            return
        if macro in common.OFFLOAD_3D_TRIANGULAR_PARAM_MACROS and len(args) >= 6:
            assert False, "obsolete"
        if self._try_start_2d_triangular_param(lineno, macro, args, indent, upper=False):
            return
        if self._try_start_2d_triangular_param(lineno, macro, args, indent, upper=True):
            return

        self._append_to_current(line)

    def _handle_preprocessor(self, line: str, stripped: str) -> bool:
        if_match = re.match(r"^#\s*if\s+(.*)$", stripped)
        ifdef_match = re.match(r"^#\s*ifdef\s+([A-Za-z_]\w*)\s*$", stripped)
        ifndef_match = re.match(r"^#\s*ifndef\s+([A-Za-z_]\w*)\s*$", stripped)
        elif_match = re.match(r"^#\s*elif\s+(.*)$", stripped)
        else_match = re.match(r"^#\s*else\s*$", stripped)
        endif_match = re.match(r"^#\s*endif\s*$", stripped)

        if if_match:
            parent_active = self.active_code
            branch_active = parent_active and common._eval_preprocessor_expr(if_match.group(1), self.defined_macros)
            self.conditional_stack.append((parent_active, branch_active))
            self.active_code = branch_active
            self._append_directive(line)
            return True

        if ifdef_match:
            parent_active = self.active_code
            branch_active = parent_active and (ifdef_match.group(1) in self.defined_macros)
            self.conditional_stack.append((parent_active, branch_active))
            self.active_code = branch_active
            self._append_directive(line)
            return True

        if ifndef_match:
            parent_active = self.active_code
            branch_active = parent_active and (ifndef_match.group(1) not in self.defined_macros)
            self.conditional_stack.append((parent_active, branch_active))
            self.active_code = branch_active
            self._append_directive(line)
            return True

        if elif_match and self.conditional_stack:
            parent_active, any_taken = self.conditional_stack[-1]
            branch_active = parent_active and (not any_taken) and common._eval_preprocessor_expr(elif_match.group(1), self.defined_macros)
            self.conditional_stack[-1] = (parent_active, any_taken or branch_active)
            self.active_code = branch_active
            self._append_directive(line)
            return True

        if else_match and self.conditional_stack:
            parent_active, any_taken = self.conditional_stack[-1]
            branch_active = parent_active and (not any_taken)
            self.conditional_stack[-1] = (parent_active, True)
            self.active_code = branch_active
            self._append_directive(line)
            return True

        if endif_match and self.conditional_stack:
            parent_active, _ = self.conditional_stack.pop()
            self.active_code = parent_active
            self._append_directive(line)
            return True

        return False

    def _process_offload_param_line(self, line: str) -> None:
        if common._OFFLOAD_PARAMETERS_END_RE.match(line.strip()):
            self.collecting_offload_params = False
            self.active_offload_param_types.update(self.collecting_param_types)
            self.active_offload_param_lines = list(self.collecting_param_lines)
            self.collecting_param_names = []
            self.collecting_param_types = {}
            self.collecting_param_lines = []
            self._append_to_current(line)
            return

        self.collecting_param_types.update(
            common.parse_offload_param_types_from_declaration_line(line, self.collecting_param_names)
        )
        self.collecting_param_lines.append(line)
        self._append_to_current(line)

    def _process_shared_var_line(self, line: str) -> None:
        if common._SHARED_VARIABLES_END_RE.match(line.strip()):
            self.collecting_shared_vars = False
            self.pending_shared_vars = dict(self.collecting_shared_types) if self.collecting_shared_types else None
            self.collecting_shared_names = []
            self.collecting_shared_types = {}
            return

        self.collecting_shared_types.update(
            common.parse_offload_param_types_from_declaration_line(line, self.collecting_shared_names)
        )

    def _try_start_offload_params(self, line: str) -> bool:
        match = common._OFFLOAD_PARAMETERS_START_RE.match(line.strip())
        if match is None:
            return False

        self.collecting_offload_params = True
        self.collecting_param_names = common.parse_identifier_list(match.group("names"))
        self.collecting_param_types = {}
        self._append_to_current(line)
        return True

    def _try_start_shared_vars(self, line: str) -> bool:
        match = common._SHARED_VARIABLES_START_RE.match(line.strip())
        if match is None or self.loop_stack:
            return False

        self.collecting_shared_vars = True
        self.collecting_shared_names = common.parse_identifier_list(match.group("names"))
        self.collecting_shared_types = {}
        return True

    def _finish_loop(self) -> None:
        ctx = self.loop_stack.pop()
        ctx.range_expr = common.apply_symbol_values(ctx.range_expr, self.symbol_values)
        if ctx.extra_params is not None:
            ctx.extra_params = common.apply_symbol_values(ctx.extra_params, self.symbol_values)
        if ctx.extra_param_types:
            ctx.extra_param_types = {
                name: common.apply_symbol_values(type_name, self.symbol_values)
                for name, type_name in ctx.extra_param_types.items()
            }
        if ctx.queue_expr is not None:
            ctx.queue_expr = common.apply_symbol_values(ctx.queue_expr, self.symbol_values)

        shared_vars, filtered_body = common.extract_shared_variables_from_body_lines(ctx.body_lines)
        if shared_vars:
            merged_shared_vars = dict(ctx.shared_vars or {})
            merged_shared_vars.update(shared_vars)
            ctx.shared_vars = merged_shared_vars
        ctx.body_lines = filtered_body

        ctx.body_lines = [
            common.apply_symbol_values(body_line, self.symbol_values)
            for body_line in ctx.body_lines
        ]
        ctx.body_lines = common.rewrite_enum_iterator_loops(ctx.body_lines, self.symbol_values)
        ctx.raw_body_lines = list(self.active_raw_body_lines)

        if self.parfor_dump_dir is not None and ctx.parfor_invocation is not None:
            common._write_parfor_dump(ctx, self.parfor_dump_dir, self.symbol_values)

        if self.on_emit_loop is not None:
            self.on_emit_loop(ctx)
        self._append_many_to_current(common._emit_loop_invocation(ctx))
        self.changed = True

    def _try_start_1d_param(self, lineno: int, macro: str, args: list[str], indent: str) -> bool:
        if macro not in common.OFFLOAD_1D_PARAM_MACROS or len(args) < 3:
            return False
        extra_param_names = common.parse_extra_param_names(", ".join(args[2:]))
        self._push_loop(
            lineno,
            indent,
            is_2d=False,
            is_3d=False,
            vars=[args[1]],
            range_expr=common.apply_symbol_values(args[2], self.symbol_values),
            kernel_guard_expr=None,
            extra_params=", ".join(args[3:]),
            extra_param_names=extra_param_names,
            queue_expr=args[0],
        )
        return True

    def _try_start_2d(self, lineno: int, macro: str, args: list[str], indent: str) -> bool:
        if macro not in common.OFFLOAD_2D_MACROS or len(args) < 3:
            return False
        self._push_loop(
            lineno,
            indent,
            is_2d=True,
            is_3d=False,
            vars=[args[1], args[2]],
            range_expr=common.apply_symbol_values(", ".join(args[3:]), self.symbol_values),
            kernel_guard_expr=None,
            extra_params=None,
            extra_param_names=None,
            queue_expr=args[0],
        )
        return True

    def _try_start_2d_param(self, lineno: int, macro: str, args: list[str], indent: str) -> bool:
        if macro not in common.OFFLOAD_2D_PARAM_MACROS or len(args) < 4:
            return False
        extra_param_names = common.parse_extra_param_names(", ".join(args[3:]))
        self._push_loop(
            lineno,
            indent,
            is_2d=True,
            is_3d=False,
            vars=[args[1], args[2]],
            range_expr=common.apply_symbol_values(args[3], self.symbol_values),
            kernel_guard_expr=None,
            extra_params=", ".join(args[4:]),
            extra_param_names=extra_param_names,
            queue_expr=args[0],
        )
        return True

    def _try_start_3d_param(self, lineno: int, macro: str, args: list[str], indent: str) -> bool:
        if macro not in common.OFFLOAD_3D_PARAM_MACROS or len(args) < 5:
            return False
        extra_param_names = common.parse_extra_param_names(", ".join(args[4:]))
        self._push_loop(
            lineno,
            indent,
            is_2d=False,
            is_3d=True,
            vars=[args[1], args[2], args[3]],
            range_expr=common.apply_symbol_values(args[4], self.symbol_values),
            kernel_guard_expr=None,
            extra_params=", ".join(args[5:]),
            extra_param_names=extra_param_names,
            queue_expr=args[0],
        )
        return True

    def _try_start_2d_triangular_param(
        self,
        lineno: int,
        macro: str,
        args: list[str],
        indent: str,
        upper: bool,
    ) -> bool:
        macro_set = common.OFFLOAD_2D_UPPER_TRIANGULAR_PARAM_MACROS if upper else common.OFFLOAD_2D_TRIANGULAR_PARAM_MACROS
        if macro not in macro_set or len(args) < 4:
            return False

        extra_param_names = common.parse_extra_param_names(", ".join(args[4:]))
        bound_param_name = common.parse_bound_param_name(args[3])
        if bound_param_name and bound_param_name not in extra_param_names:
            extra_param_names.append(bound_param_name)
        bound_expr = common.apply_symbol_values(args[3], self.symbol_values)
        guard_expr = f"{args[1]} > {args[2]}" if upper else f"{args[1]} > {args[1]}"
        if bound_param_name:
            guard_expr = (
                f"{args[1]} >= int({bound_param_name}) || "
                f"{args[2]} >= int({bound_param_name}) || "
                f"{guard_expr}"
            )

        self._push_loop(
            lineno,
            indent,
            is_2d=True,
            is_3d=False,
            vars=[args[1], args[2]],
            range_expr=f"rllm::enum_iterator2D<decltype({bound_expr}), decltype({bound_expr})>({bound_expr})",
            kernel_guard_expr=guard_expr,
            extra_params=", ".join(extra_param_names),
            extra_param_names=extra_param_names,
            queue_expr=args[0],
            triangular_kind="upper" if upper else "lower",
        )
        return True

    def _push_loop(
        self,
        lineno: int,
        indent: str,
        is_2d: bool,
        is_3d: bool,
        vars: list[str],
        range_expr: str,
        kernel_guard_expr: str | None,
        extra_params: str | None,
        extra_param_names: list[str] | None,
        queue_expr: str | None,
        triangular_kind: str | None = None,
    ) -> None:
        extra_param_types = None
        if extra_param_names is not None:
            extra_param_types = {
                name: self.active_offload_param_types[name]
                for name in extra_param_names
                if name in self.active_offload_param_types
            }

        self.loop_stack.append(
            common.LoopContext(
                indent=indent,
                backend_namespace=self.backend_namespace,
                rel_path=self.rel_path,
                lineno=lineno,
                is_2d=is_2d,
                is_3d=is_3d,
                vars=vars,
                range_expr=range_expr,
                kernel_guard_expr=kernel_guard_expr,
                extra_params=extra_params,
                extra_param_types=extra_param_types,
                offload_param_lines=list(self.active_offload_param_lines),
                emit_named_kernel=self.emit_named_kernels,
                body_lines=[],
                raw_body_lines=list(self.active_raw_body_lines),
                parfor_invocation=self.pending_parfor_invocation,
                shared_vars=dict(self.pending_shared_vars) if self.pending_shared_vars else None,
                queue_expr=queue_expr,
                triangular_kind=triangular_kind,
            )
        )
        self.active_raw_body_lines.clear()
        self.pending_shared_vars = None
        self.changed = True

    def _append_to_current(self, line: str) -> None:
        if self.loop_stack:
            self.loop_stack[-1].body_lines.append(line)
            self.active_raw_body_lines.append(line)
        else:
            self.out_lines.append(line)

    def _append_many_to_current(self, lines: list[str]) -> None:
        if self.loop_stack:
            self.loop_stack[-1].body_lines.extend(lines)
        else:
            self.out_lines.extend(lines)

    def _append_directive(self, line: str) -> None:
        self.out_lines.append(line)
