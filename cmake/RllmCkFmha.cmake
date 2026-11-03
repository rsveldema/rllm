# Build the narrow CK-Tile FMHA kernel set used by rllm.  CK's example target
# generates thousands of variants, so invoke its generator directly instead.

function(rllm_add_ck_fmha_target)
    if(NOT OFFLOAD_BACKEND STREQUAL "hip")
        message(FATAL_ERROR "RLLM_ENABLE_CK_FMHA requires OFFLOAD_BACKEND=hip")
    endif()

    set(RLLM_CK_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/composable_kernel")
    set(RLLM_CK_FMHA_DIR "${RLLM_CK_SOURCE_DIR}/example/ck_tile/01_fmha")
    set(RLLM_CK_GENERATOR "${RLLM_CK_FMHA_DIR}/generate.py")
    if(NOT EXISTS "${RLLM_CK_GENERATOR}")
        message(FATAL_ERROR
            "Composable Kernel submodule is missing. Run: "
            "git submodule update --init --recursive")
    endif()

    if(NOT GPU_TARGETS)
        message(FATAL_ERROR "RLLM_ENABLE_CK_FMHA requires GPU_TARGETS (for example gfx1201)")
    endif()

    string(REPLACE ";" "," RLLM_CK_TARGETS_ARG "${GPU_TARGETS}")
    set(RLLM_CK_GEN_DIR "${CMAKE_BINARY_DIR}/generated/ck_fmha")
    file(MAKE_DIRECTORY "${RLLM_CK_GEN_DIR}")

    # rllm uses packed/group mode, causal masking, FP16 Q/K/V, saved LSE,
    # no bias/dropout/logit cap, and a 64-element head dimension.
    set(RLLM_CK_FWD_FILTER
        "*d64_fp16_group*nlogits_nbias_mask_lse_ndropout_nskip_nqscale_ntrload_nsink")
    # Backward uses three cooperating stages; `@` supplies one fnmatch pattern
    # for each stage in dot-do-o, convert-dq, dq-dk-dv order.
    set(RLLM_CK_BWD_FILTER
        "fmha_bwd_dot_do_o_d64_fp16_b64_group_o2_ps@fmha_bwd_convert_dq_d64_fp16_b64_group_o2_ps_ndeterministic@fmha_bwd_d64_fp16_group*_npad_nbias_ndbias_mask_ndropout_ndeterministic_ntrload")

    set(RLLM_CK_FWD_LIST "${RLLM_CK_GEN_DIR}/fwd_blobs.txt")
    set(RLLM_CK_BWD_LIST "${RLLM_CK_GEN_DIR}/bwd_blobs.txt")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${RLLM_CK_GENERATOR}"
            --targets "${RLLM_CK_TARGETS_ARG}" --api fwd --optdim 64
            --filter "${RLLM_CK_FWD_FILTER}" --list_blobs "${RLLM_CK_FWD_LIST}"
        RESULT_VARIABLE RLLM_CK_FWD_RESULT
    )
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${RLLM_CK_GENERATOR}"
            --targets "${RLLM_CK_TARGETS_ARG}" --api bwd --receipt 3 --optdim 64
            --filter "${RLLM_CK_BWD_FILTER}" --list_blobs "${RLLM_CK_BWD_LIST}"
        RESULT_VARIABLE RLLM_CK_BWD_RESULT
    )
    if(NOT RLLM_CK_FWD_RESULT EQUAL 0 OR NOT RLLM_CK_BWD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to enumerate CK-Tile FMHA kernels")
    endif()

    file(STRINGS "${RLLM_CK_FWD_LIST}" RLLM_CK_FWD_SOURCES)
    file(STRINGS "${RLLM_CK_BWD_LIST}" RLLM_CK_BWD_SOURCES)
    set(RLLM_CK_FMHA_SOURCES ${RLLM_CK_FWD_SOURCES} ${RLLM_CK_BWD_SOURCES})

    file(GLOB_RECURSE RLLM_CK_CODEGEN_SCRIPTS CONFIGURE_DEPENDS
        "${RLLM_CK_FMHA_DIR}/generate.py"
        "${RLLM_CK_FMHA_DIR}/codegen/*.py")
    add_custom_command(
        OUTPUT ${RLLM_CK_FWD_SOURCES}
        COMMAND "${Python3_EXECUTABLE}" "${RLLM_CK_GENERATOR}"
            --targets "${RLLM_CK_TARGETS_ARG}" --api fwd --optdim 64
            --filter "${RLLM_CK_FWD_FILTER}" --output_dir "${RLLM_CK_GEN_DIR}"
        DEPENDS ${RLLM_CK_CODEGEN_SCRIPTS}
        COMMENT "Generating CK-Tile FMHA forward kernels for rllm"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${RLLM_CK_BWD_SOURCES}
        COMMAND "${Python3_EXECUTABLE}" "${RLLM_CK_GENERATOR}"
            --targets "${RLLM_CK_TARGETS_ARG}" --api bwd --receipt 3 --optdim 64
            --filter "${RLLM_CK_BWD_FILTER}" --output_dir "${RLLM_CK_GEN_DIR}"
        DEPENDS ${RLLM_CK_CODEGEN_SCRIPTS}
        COMMENT "Generating CK-Tile FMHA backward kernels for rllm"
        VERBATIM
    )

    add_library(rllm_ck_fmha STATIC EXCLUDE_FROM_ALL ${RLLM_CK_FMHA_SOURCES})
    set_source_files_properties(${RLLM_CK_FMHA_SOURCES} PROPERTIES LANGUAGE HIP GENERATED TRUE)
    set_property(TARGET rllm_ck_fmha PROPERTY HIP_ARCHITECTURES "${GPU_TARGETS}")
    target_include_directories(rllm_ck_fmha PUBLIC
        "${RLLM_CK_SOURCE_DIR}/include"
        "${RLLM_CK_FMHA_DIR}"
    )
    target_compile_features(rllm_ck_fmha PUBLIC cxx_std_20)
    target_compile_definitions(rllm_ck_fmha PUBLIC
        CK_ENABLE_FP16
        CK_TILE_FMHA_FWD_SPLITKV_API=0
        CK_TILE_FMHA_FWD_APPENDKV_API=0
        CK_TILE_FMHA_FWD_PAGEDKV_API=0
        CK_TILE_FMHA_FWD_BATCH_PREFILL_API=0
    )
    if(GPU_TARGETS MATCHES "gfx12")
        target_compile_definitions(rllm_ck_fmha PUBLIC
            CK_GFX12_SUPPORT
            CK_TILE_USE_WMMA=1
            CK_USE_WMMA
            CK_USE_WMMA_FP8
            CK_USE_XDL
            CK_TILE_USE_OCP_FP8
            CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=5)
    endif()
    if(GPU_TARGETS STREQUAL "gfx1201")
        # CK needs the architecture to agree in both the host and device passes.
        target_compile_definitions(rllm_ck_fmha PUBLIC CK_CMAKE_GPU_TARGET_IDS=0x1201)
    endif()
    target_compile_options(rllm_ck_fmha PRIVATE
        -O3
        -g0
        -Wno-undefined-func-template
        -Wno-float-equal
        -Wno-unknown-warning-option
        -Wno-pass-failed
        -Wno-bit-int-extension
        -fbracket-depth=1024
        -fno-offload-uniform-block
        -fgpu-flush-denormals-to-zero
        "SHELL:-mllvm -amdgpu-early-inline-all=true"
        "SHELL:-mllvm -amdgpu-function-calls=false"
    )
    target_link_libraries(rllm_ck_fmha PUBLIC hip::host)
endfunction()
