"""Unit tests for Vulkan compute shader C++ stub generation.

Each test:
1. Parses a .kernel file with the Lark-based parser
2. Generates a C++ stub header via VulkanCppStubVisitor
3. Wraps it in a small test harness that includes mock Vulkan types
4. Compiles the combined code with g++ to verify correctness
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


KERNEL_DIR = _ROOT / "codegen" / "testdata"
MULTI_ARG_KERNEL = KERNEL_DIR / "multi-arg.kernel"
SINGLE_ASSIGN_KERNEL = KERNEL_DIR / "single-assign.kernel"


def _generate_and_compile(kernel_path: Path):
    """Parse the kernel, generate a stub header, wrap in mock Vulkan types,
    and attempt to compile. Returns (success: bool, stderr: str)."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    stub_content = program.accept(visitor)

    kernel_name = kernel_path.stem
    has_2d = program.space_dim >= 2 and len(program.loop_vars) >= 2

    # Classify params for main() generation
    from codegen.ast import FlexibleRowsMatrix, FixedSizeMatrix, FixedSizeVector, Int, Float

    matrix_params = []
    vector_params = []
    scalar_count = 0

    for param in program.params:
        vt = param.var_type
        if isinstance(vt, (FlexibleRowsMatrix, FixedSizeMatrix)):
            sname = f"FRM_float"
            matrix_params.append((sname, "const"))
        elif isinstance(vt, FixedSizeVector):
            vec_name = f"vec_{param.name}"
            vector_params.append((vec_name, "const" if param.is_const else ""))
        elif isinstance(vt, (Int, Float)):
            scalar_count += 1

    # Mock Vulkan types
    mock_vulkan = textwrap.dedent("""\
        #include <cstdint>
        struct VkDevice_T{}; using VkDevice = VkDevice_T*;
        struct VkPipelineLayout_T{}; using VkPipelineLayout = VkPipelineLayout_T*;
        struct VkDescriptorSetLayout_T{}; using VkDescriptorSetLayout = VkDescriptorSetLayout_T*;
        struct VkCommandBuffer_T{}; using VkCommandBuffer = VkCommandBuffer_T*;
        struct VkDescriptorSet_T{}; using VkDescriptorSet = VkDescriptorSet_T*;
        inline void vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t){}
    """)

    harness_lines = [mock_vulkan]
    for line in stub_content.splitlines():
        if "#include <vulkan/vulkan.h>" in line:
            continue
        harness_lines.append(line)

    # Build main() with correct number of args
    init_code = []
    call_args = ["nullptr", "nullptr", "nullptr", "nullptr", "nullptr"]  # device, pipeline_layout, dsl, cb, ds

    if has_2d:
        call_args.extend(["1u", "1u"])
    else:
        call_args.append("1u")

    for i, (sname, _cv) in enumerate(matrix_params):
        vname = f"_m{i}"
        init_code.append(f"    {sname} {vname}{{nullptr, 0, 0}};")
        call_args.append(vname)

    for i, (vec_name, _cv) in enumerate(vector_params):
        vname = f"_v{i}"
        init_code.append(f"    {vec_name} {vname}{{nullptr, 0}};")
        call_args.append(vname)

    for i in range(scalar_count):
        vname = f"_sc{i}"
        init_code.append(f"    int32_t {vname} = 0;")
        call_args.append(vname)

    # Find the dispatch function name from stub
    func_name = None
    for line in stub_content.splitlines():
        if "inline void" in line and "(" in line:
            func_part = line.replace("inline void ", "").strip().split("(")[0]
            func_name = func_part.rstrip(";{")
            break

    if not func_name:
        return False, "Could not find dispatch function name in stub"

    main_func = (
        f"int main() {{\n"
        + "\n".join(init_code) + "\n"
        f"    {func_name}({', '.join(call_args)});\n"
        "    return 0;\n"
        "}\n"
    )

    harness_lines.append(main_func)
    combined_source = "\n".join(harness_lines) + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        test_cpp = tmppath / "test_stub.cpp"
        test_cpp.write_text(combined_source)

        for compiler in ("g++", "clang++", "c++"):
            try:
                r = subprocess.run(
                    [compiler, "-std=c++17", "-x", "c++", str(test_cpp),
                     "-o", str(tmppath / "test_out")],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    return True, ""
            except FileNotFoundError:
                continue

        r = subprocess.run(
            ["g++", "-std=c++17", "-x", "c++", str(test_cpp),
             "-o", str(tmppath / "test_out")],
            capture_output=True, text=True, timeout=30,
        )
        return False, r.stderr


# ---- tests ----

@pytest.mark.parametrize("kernel_path", [
    MULTI_ARG_KERNEL,
    SINGLE_ASSIGN_KERNEL,
], ids=["multi-arg", "single-assign"])
def test_stub_compiles(kernel_path: Path):
    """Verify that the generated C++ stub compiles with mocked Vulkan types."""
    success, stderr = _generate_and_compile(kernel_path)
    assert success, (
        f"Generated C++ stub for {kernel_path.name} failed to compile.\n"
        f"stderr:\n{stderr}"
    )


@pytest.mark.parametrize("kernel_path", [
    MULTI_ARG_KERNEL,
], ids=["multi-arg"])
def test_stub_has_vkcmddispatch_call(kernel_path: Path):
    """Verify the stub contains a vkCmdDispatch call."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)
    assert "vkCmdDispatch" in content, "Missing vkCmdDispatch in generated stub"


@pytest.mark.parametrize("kernel_path", [
    SINGLE_ASSIGN_KERNEL,
], ids=["single-assign"])
def test_stub_single_assign_has_vector_struct(kernel_path: Path):
    """Verify single-assign kernel stub has vector SSBO struct with size."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)

    assert "vkCmdDispatch" in content
    assert "vec_dst" in content, "Missing vec_dst struct for fixed_size_vector param"
    assert "uint32_t size" in content, "Missing size field in vector struct"


@pytest.mark.parametrize("kernel_path", [
    MULTI_ARG_KERNEL,
], ids=["multi-arg"])
def test_stub_multi_arg_has_struct_defs(kernel_path: Path):
    """Verify multi-arg kernel stub has matrix struct definitions."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)

    assert "vkCmdDispatch" in content
    assert "struct FRM_float" in content, "Missing FRM_float struct"


@pytest.mark.parametrize("kernel_path", [
    MULTI_ARG_KERNEL,
], ids=["multi-arg"])
def test_stub_multi_arg_dispatch_dimensions(kernel_path: Path):
    """Verify 2D parfor kernel dispatches with correct x/y workgroup division."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)

    assert "dispatch_rows" in content, "Missing dispatch_rows param for 2D kernel"
    assert "dispatch_cols" in content, "Missing dispatch_cols param for 2D kernel"
    assert "(dispatch_rows + 8 - 1)" in content, "Missing workgroup X division"
    assert "(dispatch_cols + 8 - 1)" in content, "Missing workgroup Y division"
