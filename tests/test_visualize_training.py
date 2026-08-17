import visualize_training


def test_restart_baseline_is_rebased_to_latest_training_position(tmp_path):
    progress = tmp_path / "train.json"
    progress.write_text("""[
      {"item_type":"incremental_stage","epoch_offset":0,"stage":1,"layers":3},
      {"item_type":"window","epoch":0,"range_end":75,"total_items":100},
      {"item_type":"validation","phase":"baseline","epoch":0,"epoch_progress":0},
      {"item_type":"window","epoch":0,"range_end":10,"total_items":100}
    ]""")

    entries = visualize_training.load_entries(progress)

    assert visualize_training.position(entries[2], True) == 0.75
    assert visualize_training.position(entries[3], False) == 0.85
import matplotlib.pyplot as plt


def test_incremental_epoch_tick_includes_active_layer_count():
    stages = [(0.0, 3), (3.0, 4), (6.0, 5)]

    assert visualize_training.epoch_layer_label(2.0, stages) == "2 (3L)"
    assert visualize_training.epoch_layer_label(3.0, stages) == "3 (4L)"
    assert visualize_training.epoch_layer_label(7.5, stages) == "7.5 (5L)"


def test_regular_epoch_tick_has_no_layer_suffix():
    assert visualize_training.epoch_layer_label(4.0, []) == "4"


def test_gradient_plot_is_hidden_without_diagnostic_records(tmp_path):
    fig, axes = plt.subplot_mosaic([
        ["loss", "perplexity"],
        ["mtp", "probability"],
        ["gradient", "gradient"],
    ])
    entries = [{"item_type": "window", "epoch": 0, "training_loss": 5.0}]

    visualize_training.render_metrics(fig, axes, [(tmp_path / "train.json", entries)])

    assert not axes["gradient"].get_visible()
    plt.close(fig)


def test_gradient_plot_is_visible_with_diagnostic_records(tmp_path):
    fig, axes = plt.subplot_mosaic([
        ["loss", "perplexity"],
        ["mtp", "probability"],
        ["gradient", "gradient"],
    ])
    entries = [{
        "item_type": "window",
        "epoch": 0,
        "training_loss": 5.0,
        "optimizer_diagnostics": [{
            "parameter_group": "transformer layer 0",
            "normalized_global_gradient_norm": 0.5,
        }],
    }]

    visualize_training.render_metrics(fig, axes, [(tmp_path / "train.json", entries)])

    assert axes["gradient"].get_visible()
    plt.close(fig)
