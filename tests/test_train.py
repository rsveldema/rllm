import os
import json
from pathlib import Path

import pytest

import train
import generate_embedding_prefill


def valid_config() -> dict[str, object]:
    return {
        "build_type": "release",
        "model_directory": "models-test",
        "layers": 3,
        "window_size": 12,
        "window_stride": 6,
        "learn_depth": 1,
        "sources": [],
        "cmake_arguments": [],
        "training_arguments": [],
    }


def test_comment_removal_defaults_to_disabled(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(valid_config()))

    config = train.load_config(config_file)

    assert config["strip_comments"] is False


def test_embedding_prefill_generator_adds_only_exact_token_declarations(tmp_path):
    base = tmp_path / "base.json"
    token_map = tmp_path / "token-map.json"
    source = tmp_path / "source"
    source.mkdir()
    base.write_text(json.dumps({
        "concepts": ["class", "class-name", "struct"],
        "relationships": [["class", "struct", 0.5]],
    }))
    token_map.write_text(json.dumps({"Widget": 1, "class": 2, "struct": 3}))
    (source / "sample.cpp").write_text(
        "class Widget {}; struct FragmentedName {}; // class CommentOnly {}\nclass X {};"
    )

    result = generate_embedding_prefill.generate(base, token_map, [source])

    assert "Widget" in result["concepts"]
    assert "FragmentedName" not in result["concepts"]
    assert "CommentOnly" not in result["concepts"]
    assert "X" not in result["concepts"]
    assert ["Widget", "class-name", 0.9] in result["relationships"]
    assert {
        "subject": "FragmentedName", "relation": "is-a",
        "object": "class-name", "exact_token": False,
    } in result["discovered_facts"]


def test_incremental_training_removes_concept_embedding_prefill():
    arguments = (
        "--epochs", "4", "--concept-embeddings", "concepts.json",
        "--micro-batch-size", "64",
    )

    assert train.remove_embedding_prefill(arguments) == (
        "--epochs", "4", "--micro-batch-size", "64",
    )


def test_comment_removal_launcher_override_can_enable_or_disable():
    enabled = train.parse_launcher_arguments(["config.json", "--strip-comments"])
    disabled = train.parse_launcher_arguments(["config.json", "--no-strip-comments"])

    assert enabled.strip_comments is True
    assert disabled.strip_comments is False


def test_latest_selects_newest_numbered_checkpoint_not_best(tmp_path, monkeypatch):
    older = tmp_path / "checkpoint-100.st"
    newer = tmp_path / "checkpoint-200.st"
    best = tmp_path / "checkpoint-best-window.st"
    for checkpoint in (older, newer, best):
        checkpoint.touch()
    # Set deterministic mtimes without relying on creation-time resolution.
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(best, (3, 3))
    monkeypatch.setattr(train, "compatible_model", lambda *args, **kwargs: True)

    selected = train.select_resume_model(
        tmp_path, None, Path("runtime.hpp"),
        fresh_start=False, latest=True, resume_model=None,
    )

    assert selected == newer


def test_default_resume_selects_newest_numbered_checkpoint_before_best(tmp_path, monkeypatch):
    older = tmp_path / "checkpoint-100.st"
    newer = tmp_path / "checkpoint-200.st"
    best = tmp_path / "checkpoint-best-window.st"
    for checkpoint in (older, newer, best):
        checkpoint.touch()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(best, (3, 3))
    monkeypatch.setattr(train, "compatible_model", lambda *args, **kwargs: True)

    selected = train.select_resume_model(
        tmp_path, None, Path("runtime.hpp"),
        fresh_start=False, latest=False, resume_model=None,
    )

    assert selected == newer


def test_latest_requires_a_latest_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "checkpoint-best-window.st").touch()
    monkeypatch.setattr(train, "compatible_model", lambda *args, **kwargs: True)

    with pytest.raises(FileNotFoundError, match="No latest checkpoint"):
        train.select_resume_model(
            tmp_path, None, Path("runtime.hpp"),
            fresh_start=False, latest=True, resume_model=None,
        )


def test_latest_is_mutually_exclusive_with_other_resume_modes():
    with pytest.raises(SystemExit):
        train.parse_launcher_arguments(["config.json", "--latest", "--fresh-start"])

    with pytest.raises(SystemExit):
        train.parse_launcher_arguments(["config.json", "--latest", "--lowest-loss"])


def test_learning_rate_launcher_options_are_independent_of_resume_mode():
    options = train.parse_launcher_arguments([
        "config.json", "--latest",
        "--restart-learning-rate-schedule", "--skip-warmup",
    ])

    assert options.latest
    assert options.restart_learning_rate_schedule
    assert options.skip_warmup


def test_compatible_model_rejects_tokenizer_signature_mismatch(tmp_path):
    runtime_header = tmp_path / "tokenizer_map.hpp"
    runtime_source = tmp_path / "tokenizer_map.cc"
    runtime_header.write_text("enum class TokenID { TOK_0 = 0, MAX = 1 };\n")
    runtime_source.write_text(
        'std::map<TokenID, TokenInfo> tokenizer_map = {\n'
        '    {TokenID::TOK_0, {"abc", true}},\n'
        '};\n'
    )
    checkpoint = tmp_path / "model.json"
    checkpoint.write_text(json.dumps({
        "tokenizer_vocab_size": 1,
        "tokenizer_signature": 1234,
    }))

    assert not train.compatible_model(checkpoint, runtime_header)


def test_incremental_window_accepts_latest():
    options = train.parse_launcher_arguments([
        "config.json", "--incremental-window", "--latest",
    ])

    assert options.incremental_window
    assert options.latest


def test_incremental_window_rejects_other_resume_selection():
    with pytest.raises(SystemExit):
        train.parse_launcher_arguments([
            "config.json", "--incremental-window", "--lowest-loss",
        ])


def test_incremental_window_stages_grow_depth_and_context():
    stages = train.incremental_window_stages(8, 96, 48, 44)

    assert stages == [
        (3, 12, 6, 4),
        (4, 16, 8, 4),
        (5, 24, 12, 6),
        (6, 48, 24, 8),
        (7, 72, 36, 10),
        (8, 96, 48, 12),
    ]


def test_incremental_steps_train_new_block_then_all_blocks_at_each_upgrade():
    stages = train.incremental_window_stages(8, 96, 48, 44, 4, 2)

    assert train.incremental_training_steps(stages, 2) == [
        (3, 12, 6, 4, "bootstrap"),
        (4, 16, 8, 2, "growth"),
        (4, 16, 8, 2, "all-blocks"),
        (5, 24, 12, 4, "growth"),
        (5, 24, 12, 2, "all-blocks"),
        (6, 48, 24, 6, "growth"),
        (6, 48, 24, 2, "all-blocks"),
        (7, 72, 36, 8, "growth"),
        (7, 72, 36, 2, "all-blocks"),
        (8, 96, 48, 10, "growth"),
        (8, 96, 48, 2, "all-blocks"),
    ]


def test_incremental_stages_reject_budget_too_small_for_growing_new_block_time():
    with pytest.raises(ValueError, match="at least 44 epochs"):
        train.incremental_window_stages(8, 96, 48, 40, 4, 2)


def test_three_layer_incremental_target_uses_the_complete_budget():
    assert train.incremental_window_stages(3, 12, 6, 7) == [(3, 12, 6, 7)]


def test_incremental_controller_state_round_trip(tmp_path):
    stages = train.incremental_window_stages(8, 96, 48, 40, 3)
    steps = train.incremental_training_steps(stages, 2)
    state_file = tmp_path / "incremental-controller.json"
    state = {
        "version": 1,
        "status": "running",
        "active_step": 3,
        "plan": [list(step) for step in steps],
        "config": {"learning_rate": 0.1},
        "input_checkpoint": "previous.st",
    }

    train.write_incremental_controller_state(state_file, state)

    assert train.load_incremental_controller_state(
        state_file, steps, {"learning_rate": 0.1}) == state


def test_incremental_controller_rejects_changed_plan(tmp_path):
    state_file = tmp_path / "incremental-controller.json"
    train.write_incremental_controller_state(state_file, {
        "version": 1, "status": "running", "active_step": 0, "plan": [],
    })
    stages = train.incremental_window_stages(8, 96, 48, 40, 3)

    with pytest.raises(ValueError, match="does not match the current configuration"):
        train.load_incremental_controller_state(
            state_file, train.incremental_training_steps(stages, 2))


def test_incremental_controller_rejects_changed_training_parameters(tmp_path):
    stages = train.incremental_window_stages(8, 96, 48, 40, 3)
    steps = train.incremental_training_steps(stages, 2)
    state_file = tmp_path / "incremental-controller.json"
    train.write_incremental_controller_state(state_file, {
        "version": 1, "status": "running", "active_step": 0,
        "plan": [list(step) for step in steps], "config": {"learning_rate": 0.1},
    })

    with pytest.raises(ValueError, match="different training parameters"):
        train.load_incremental_controller_state(
            state_file, steps, {"learning_rate": 0.2})


def test_incremental_controller_logs_active_step_and_command(tmp_path):
    train.log_incremental_controller_step(
        tmp_path, 2, 11, 4, 16, 8, 2, "growth",
        ["rllm", "--train", "--layers", "4"],
    )

    log = (tmp_path / "train.log").read_text()
    assert "Incremental controller step active: growth" in log
    assert "2/11, layers 4, window 16, stride 8, epochs 2" in log
    assert '"--layers", "4"' in log


def test_incremental_resume_step_uses_checkpoint_training_shape_and_phase(tmp_path):
    checkpoint = tmp_path / "checkpoint-latest.st"
    checkpoint.touch()
    Path(f"{checkpoint}.training.json").write_text(json.dumps({
        "layers": 5,
        "window_size": 24,
        "window_stride": 12,
        "incremental_step": "all-blocks",
    }))
    stages = train.incremental_window_stages(8, 96, 48, 40, 3)
    steps = train.incremental_training_steps(stages, 2)

    assert train.incremental_resume_step(checkpoint, steps) == 4


def test_incremental_resume_step_rejects_unknown_shape(tmp_path):
    checkpoint = tmp_path / "checkpoint-latest.st"
    checkpoint.touch()
    Path(f"{checkpoint}.training.json").write_text(json.dumps({
        "layers": 3,
        "window_size": 24,
        "window_stride": 12,
    }))

    with pytest.raises(ValueError, match="does not match the configured incremental plan"):
        stages = train.incremental_window_stages(8, 96, 48, 40, 3)
        train.incremental_resume_step(
            checkpoint, train.incremental_training_steps(stages, 2))


def test_incremental_config_enables_mode_without_cli_flag():
    config = train.load_config(Path("configs/incremental.json"))
    options = train.parse_launcher_arguments(["configs/incremental.json"])

    assert config["incremental_window"] is True
    assert config["incremental_stage_epochs"] == 4
    assert config["incremental_new_block_epochs"] == 2
    assert config["incremental_upgrade_output_scale"] == 0.1
    assert "curriculum/algos:0.10" in config["sources"]
    assert config["model_directory"] == "models-8-incremental"
    assert not options.incremental_window
    assert options.incremental_window or config["incremental_window"]


def test_incremental_stage_marker_is_visible_as_epoch_offset(tmp_path):
    progress = tmp_path / "train.json"
    progress.write_text(json.dumps([
        {"item_type": "window", "epoch": 0, "training_loss": 5.0},
    ]))

    train.append_incremental_stage_marker(progress, 2, 6, 3, 4, 40, 20, 3)
    entries = json.loads(progress.read_text())

    assert entries[-1]["item_type"] == "incremental_stage"
    assert entries[-1]["epoch_offset"] == 3
    assert entries[-1]["layers"] == 4
    assert entries[-1]["window_size"] == 40


def test_clean_preserves_newest_checkpoint_and_its_sidecar(tmp_path):
    older = tmp_path / "checkpoint-100.st"
    older_sidecar = tmp_path / "checkpoint-100.st.training.json"
    newer = tmp_path / "checkpoint-200.st"
    newer_sidecar = tmp_path / "checkpoint-200.st.training.json"
    orphan_sidecar = tmp_path / "checkpoint-300.st.training.json"
    for artifact in (older, older_sidecar, newer, newer_sidecar, orphan_sidecar):
        artifact.write_text(artifact.name)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    preserved = train.clean_intermediate_checkpoints(tmp_path)

    assert preserved == tmp_path / "checkpoint-latest.st"
    assert preserved.read_text() == newer.name
    assert Path(f"{preserved}.training.json").read_text() == newer_sidecar.name
    assert not older.exists()
    assert not older_sidecar.exists()
    assert not orphan_sidecar.exists()


def test_latest_selects_cleaned_checkpoint(tmp_path, monkeypatch):
    cleaned = tmp_path / "checkpoint-latest.st"
    cleaned.touch()
    monkeypatch.setattr(train, "compatible_model", lambda *args, **kwargs: True)

    selected = train.select_resume_model(
        tmp_path, None, Path("runtime.hpp"),
        fresh_start=False, latest=True, resume_model=None,
    )

    assert selected == cleaned


def test_lowest_loss_selects_checkpoint_paired_with_best_validation(tmp_path, monkeypatch):
    worse = tmp_path / "checkpoint-1003.st"
    better = tmp_path / "checkpoint-2004.st"
    worse.touch()
    better.touch()
    (tmp_path / "train.json").write_text(json.dumps([
        {"item_type": "validation", "timestamp_ms": 1000, "validation_loss": 5.2},
        {"item_type": "validation", "timestamp_ms": 2000, "validation_loss": 4.8},
    ]))
    monkeypatch.setattr(train, "compatible_model", lambda *args, **kwargs: True)

    selected = train.select_resume_model(
        tmp_path, None, Path("runtime.hpp"),
        fresh_start=False, latest=False, resume_model=None, lowest_loss=True,
    )

    assert selected == better


def test_lowest_loss_rejects_unpaired_validation(tmp_path):
    (tmp_path / "checkpoint-10000.st").touch()
    (tmp_path / "train.json").write_text(json.dumps([
        {"item_type": "validation", "timestamp_ms": 1000, "validation_loss": 4.8},
    ]))

    with pytest.raises(FileNotFoundError, match="paired with a validation"):
        train.lowest_loss_checkpoint(tmp_path)


def test_clean_preserves_newer_existing_stable_checkpoint(tmp_path):
    cleaned = tmp_path / "checkpoint-latest.st"
    cleaned_sidecar = tmp_path / "checkpoint-latest.st.training.json"
    numbered = tmp_path / "checkpoint-100.st"
    numbered_sidecar = tmp_path / "checkpoint-100.st.training.json"
    for artifact in (cleaned, cleaned_sidecar, numbered, numbered_sidecar):
        artifact.write_text(artifact.name)
    os.utime(numbered, (1, 1))
    os.utime(cleaned, (2, 2))

    preserved = train.clean_intermediate_checkpoints(tmp_path)

    assert preserved == cleaned
    assert cleaned.read_text() == cleaned.name
    assert cleaned_sidecar.read_text() == cleaned_sidecar.name
    assert not numbered.exists()
    assert not numbered_sidecar.exists()


def test_selecting_default_checkpoint_preserves_numbered_checkpoints_and_sidecars(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint-100.st"
    sidecar = tmp_path / "checkpoint-100.st.training.json"
    best = tmp_path / "checkpoint-best-window.st"
    for artifact in (checkpoint, sidecar, best):
        artifact.touch()
    monkeypatch.setattr(train, "compatible_model", lambda *args, **kwargs: True)

    selected = train.select_resume_model(
        tmp_path, None, Path("runtime.hpp"),
        fresh_start=False, latest=False, resume_model=None,
    )

    assert selected == checkpoint
    assert checkpoint.exists()
    assert sidecar.exists()


def test_default_resume_rejects_incompatible_same_directory_checkpoints(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint-100.st"
    checkpoint.touch()
    monkeypatch.setattr(train, "compatible_model", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="none are compatible"):
        train.select_resume_model(
            tmp_path, None, Path("runtime.hpp"),
            fresh_start=False, latest=False, resume_model=None,
        )
