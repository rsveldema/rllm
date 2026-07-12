# AGENTS.md

## Repository expectations

- Run '. .venv/bin/activate' first
- Document public utilities in 'docs/' when changing behaviour
- Run 'build_debug.sh' to build with debugging enabled, 'build_release.sh' for an optimized version
- Create tkernel files to transform the AST for optimizations
- Do not perform optimization by emitting different code in the vulkan_kernel_visitor