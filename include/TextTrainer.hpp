#pragma once

#include <Corpus.hpp>
#include <array>
#include <InputLayer.hpp>
#include <LearningRate.hpp>
#include <OptimizerDiagnostics.hpp>
#include <TransformerBlock.hpp>
#include <LayerPrimitives.hpp>
#include <OutputLayer.hpp>
#include <Statistics.hpp>
#include <cpu/cpu_flex_rows_matrix.hpp>
#include <string>

#include <fixed_size_obj_vector.hpp>

#include <nlohmann/json_fwd.hpp>
#include <chrono>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>


namespace rllm
{
    struct TextTrainerForwardWorkspace;
    struct BackwardPropWorkspace;
    struct GradientAccumulationWorkspace;
    enum class TrainingStepOutcome
    {
        Continue,
        Converged,
        Failed,
    };

    struct TrainingStepTiming
    {
        double forward_ms = 0.0;
        double backward_ms = 0.0;
        double apply_ms = 0.0;
        double backward_output_ms = 0.0;
        double backward_transformer_ms = 0.0;
        double backward_input_ms = 0.0;
    };

    enum class TrainingMethod
    {
        WINDOW,
        REVERSE_WINDOW,
    };

    const char* training_method_to_string(TrainingMethod method);

    class TextTrainer
    {
      public:
        /** Denominator for convergence threshold: fires_nothing_ce_loss / k.
         * Higher k = tighter threshold = more gradient steps per example.
         * NOTE: if set too high, we drive a specific example to high confidence but fail to learn from other examples, harming generalization.
         *
         * - FIRES_NOTHONG_CE_LOSS(std::log(static_cast<float>(TokenID::MAX)))
         * - With TokenID::MAX = 621:
         * - Baseline uniform CE loss: ln(621) ≈ 6.431
         * - CONVERGENCE_THRESHOLD(FIRES_NOTHONG_CE_LOSS / K_CONVERGENCE_DIVISOR)
         * - K_CONVERGENCE_DIVISOR = 2.0 → convergence threshold ≈ 3.215   = exp(-3.215) ≈ 0.040 = 4% chance of predicting the correct token
         * - K_CONVERGENCE_DIVISOR = 4.0 → convergence threshold ≈ 1.608 = exp(-1.608) ≈ 0.200 = 20% chance of predicting the correct token
         * - K_CONVERGENCE_DIVISOR = 6.0 → convergence threshold ≈ 1.072 = exp(-1.072) ≈ 0.342 = 34% chance of predicting the correct token
         * - K_CONVERGENCE_DIVISOR = 10.0 → convergence threshold ≈
         * - K_CONVERGENCE_DIVISOR = 50.0 → convergence threshold ≈ 0.129 = exp(-0.129) ≈ 0.879 = 88% chance of predicting the correct token
         * - K_CONVERGENCE_DIVISOR = 100.0 → convergence threshold ≈ 0.0643 = exp(-0.0643) ≈ 0.938 = 94% chance of predicting the correct token
         * - K_CONVERGENCE_DIVISOR = 200.0 → convergence threshold ≈ 0.0321 = exp(-0.0321) ≈ 0.969 = 97% chance of predicting the correct token
         * - K_CONVERGENCE_DIVISOR = 500.0 → convergence threshold ≈ 0.0129 = exp(-0.0129) ≈ 0.987 = 98% chance of predicting the correct token
         * - K_CONVERGENCE_DIVISOR = 1000.0 → convergence threshold ≈ 0.00643 = exp(-0.00643) ≈ 0.9936 = 99% chance of predicting the correct token
         * - K_CONVERGENCE_DIVISOR = 2000.0 → convergence threshold ≈ 0.003215 = exp(-0.003215) ≈ 0.9968 = 99.7% chance of predicting the
         */
        static constexpr float K_CONVERGENCE_DIVISOR = 50.0f;
        static constexpr size_t DEFAULT_LEARN_DEPTH = 16;
        static constexpr float DEFAULT_LEARNING_RATE = 0.0003f;
        static constexpr float FIRES_NOTHONG_CE_LOSS = std::log(static_cast<float>(TokenID::MAX));
        static constexpr float CONVERGENCE_THRESHOLD = FIRES_NOTHONG_CE_LOSS / K_CONVERGENCE_DIVISOR;
        static constexpr float PROBABILITY_OF_CORRECT_TOKEN = std::exp(-CONVERGENCE_THRESHOLD) * 100.0f;

        TextTrainer(size_t num_layers, Corpus& corpus, Statistics& stats);
        ~TextTrainer();
        TextTrainer(const TextTrainer&) = delete;
        TextTrainer& operator=(const TextTrainer&) = delete;

        const Corpus& get_corpus() const { return m_corpus; }
        Statistics&   get_statistics() const { return m_stats; }
        const fixed_size_obj_vector<OutputLayer, MultiTokenPredictionIndex>& get_output_layers() const { return m_output_layers; }
        const OutputLayer& get_output_layer(MultiTokenPredictionIndex idx) const { return m_output_layers[idx]; }
        const InputLayer& get_input_layer() const { return m_input_layer; }
        size_t get_transformer_block_count() const { return m_transformer_blocks.size(); }

        void set_training_method(TrainingMethod m) { m_training_method = m; }
        void set_window_size(int n) { assert(n >= 2); m_window_size = n; }
        void set_window_stride(size_t n) { assert(n > 0); m_window_stride = n; }
        void set_learn_depth(size_t n) { assert(n > 0); m_learn_depth = n; }
        void set_learning_rate(float rate) { assert(rate > 0.0f); m_learning_rate = rate; }
        void set_layer_learning_rate_multiplier(float multiplier) { assert(multiplier >= 1.0f && multiplier < 2.0f); m_layer_learning_rate_multiplier = multiplier; }
        void set_learning_rate_schedule(LearningRateSchedule schedule) { m_learning_rate_schedule = schedule; }
        void set_simulated_annealing_decay_factor(float factor) { assert(factor > 0.0f && factor < 1.0f); m_simulated_annealing_decay_factor = factor; }
        void set_simulated_annealing_initial_multiplier(float multiplier) { assert(multiplier > 0.0f); m_simulated_annealing_initial_multiplier = multiplier; }
        void set_simulated_annealing_decay_epochs(size_t epochs) { assert(epochs > 0); m_simulated_annealing_decay_epochs = epochs; }
        void set_simulated_annealing_min_multiplier(float multiplier) { assert(multiplier > 0.0f); m_simulated_annealing_min_multiplier = multiplier; }
        void set_micro_batch_size(size_t n);
        void set_max_validation_windows(size_t n)
        {
            assert(n > 0);
            m_max_validation_windows = n;
        }
        void set_validation_worst_count(size_t n)
        {
            assert(n > 0);
            m_validation_worst_count = n;
        }
        void set_early_stopping_enabled(bool enabled) { m_early_stopping_enabled = enabled; }
        void set_example_convergence_enabled(bool enabled) { m_example_convergence_enabled = enabled; }
        void set_training_diagnostics_enabled(bool enabled) { m_training_diagnostics_enabled = enabled; }
        void set_reset_optimizer_state_on_load(bool enabled) { m_reset_optimizer_state_on_load = enabled; }
        void set_restart_learning_rate_schedule_on_load(bool enabled) { m_restart_learning_rate_schedule_on_load = enabled; }
        void set_weight_initializer(WeightInitializerType type) { m_weight_initializer = type; }
        void set_ffn_initializer(FFNInitializerType type) { m_ffn_initializer = type; }
        void set_embedding_initializer(EmbeddingInitializerType type) { m_embedding_initializer = type; }
        void set_training_parameters_json(std::string json) { m_training_parameters_json = std::move(json); }
        void set_training_progress_filename(std::string filename) { m_training_progress_filename = std::move(filename); }

        void propagate_forward();

        // Returns the top-K output tokens with the highest activation.
        std::vector<OutputToken> get_best_output_token_ids(size_t top_k, MultiTokenPredictionIndex head) const;

        void train(
            bool verbose,
            size_t num_epochs,
            const std::optional<std::string>& input_filename,
            const std::optional<std::chrono::seconds>& checkpointing_interval = std::nullopt,
            std::optional<size_t> epoch_size = std::nullopt
        );

        void set_random_weights_and_connections();

        // returns true on success, false on failure (e.g. file not found or parse error)
        bool load(const std::string& filename);
        void checkpoint() const;
        void save(const std::string& filename) const;
        // Safetensors serialization
        void save_to_safetensors(const std::string& filename) const;
        bool load_from_safetensors(const std::string& filename);

        // Mean-pool the last transformer block's hidden state over the sequence dimension.
        // Equivalent to last_hidden_state.mean(dim=1) in PyTorch.
        // Must be called after propagate_forward().
        cpu_fixed_vector<float, EmbeddingDimension> get_last_hidden_mean(VulkanQueue& queue) const;

        /** set the input for the neural network.
         * Call this just before    propagate_forward() and then call propagate_backward_mtp() to train on this input.
         */
        CpuInputLine& get_last_input() {
            return m_last_input;
        }
        

      private:
                static constexpr size_t VALIDATION_PERCENT = 20;
                static constexpr size_t EARLY_STOPPING_PATIENCE = 3;
                static constexpr float  VALIDATION_IMPROVEMENT_EPSILON = 1e-4f;

        Corpus&    m_corpus;
        Statistics& m_stats;
        InputLayer  m_input_layer;
        CpuInputLine   m_last_input;   // saved in propagate_forward for use in propagate_backward
        std::vector<TransformerBlock> m_transformer_blocks;
        fixed_size_obj_vector<OutputLayer, MultiTokenPredictionIndex> m_output_layers;
        fixed_size_obj_vector<Score, MultiTokenPredictionIndex> m_training_scores;
        Score m_evaluation_score;
        bool m_early_stopping_enabled = true;
        bool m_example_convergence_enabled = true;
        bool m_training_diagnostics_enabled = true;
        bool m_reset_optimizer_state_on_load = false;
        bool m_restart_learning_rate_schedule_on_load = false;
        bool m_optimizer_diagnostics_pending = false;
        bool m_backward_diagnostics_pending = false;

        // Hidden state at the final position after the last transformer block.
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> m_last_hidden;
        PositionIndex m_seq_len{PositionIndex::START};
        std::unique_ptr<TextTrainerForwardWorkspace> m_forward_workspace;
        std::unique_ptr<BackwardPropWorkspace> m_backward_workspace;
        std::unique_ptr<GradientAccumulationWorkspace> m_gradient_accumulation_workspace;
        std::unique_ptr<GpuPackedBatchInput> m_gpu_packed_batch;
        std::unique_ptr<BatchedOutputWorkspace> m_batched_output_workspace;

        void reset_workspaces();
        void reset_gradient_accumulators();
        void apply_accumulated_gradients(float learning_rate_scale);
        void dump_top_predictions();
        void trace_probes_for_example(const char* phase, size_t iter, float loss_value, const std::string& full_string);
        TrainingStepOutcome do_training_step(
            const CpuInputLine& train_output,
            bool verbose,
            size_t iteration_index,
            float learning_rate_scale = 1.0f,
            bool manage_accumulator = true,
            TrainingStepTiming* timing = nullptr
        );
        size_t do_training(const CpuInputLine& train_output, bool verbose, size_t max_iterations, float learning_rate_scale = 1.0f, bool manage_accumulator = true);
        // Accumulates gradients from all valid MTP heads and backpropagates once.
        void propagate_backward_mtp(
            const fixed_size_obj_vector<Score, MultiTokenPredictionIndex>& scores,
            MultiTokenPredictionIndex num_valid,
            TrainingStepTiming* timing = nullptr
        );
        struct EvaluationMetrics
        {
            // Head zero is the next-token completion objective and the metric
            // used for checkpointing and early stopping.
            float average_loss = 0.0f;
            double perplexity = 0.0;
            double average_correct_token_probability = 0.0;
            double mtp_average_loss = 0.0;
            double mtp_average_correct_token_probability = 0.0;
            std::array<double, static_cast<size_t>(MultiTokenPredictionIndex::MAX)> per_head_loss{};
            std::array<size_t, static_cast<size_t>(MultiTokenPredictionIndex::MAX)> per_head_count{};
        };
        EvaluationMetrics evaluate(const std::vector<CpuInputLine>& evaluation_lines);
        EvaluationMetrics evaluate(
            const std::vector<WindowExample>& evaluation_windows,
            bool report_worst_predictions = false
        );

        TrainingMethod m_training_method = TrainingMethod::WINDOW;
        int m_window_size = 4;
        size_t m_window_stride = 1;
        size_t m_learn_depth = DEFAULT_LEARN_DEPTH;
        float m_learning_rate = DEFAULT_LEARNING_RATE;
        float m_layer_learning_rate_multiplier = DEFAULT_DEPTH_LEARNING_RATE_MULTIPLIER;
        LearningRateSchedule m_learning_rate_schedule = LearningRateSchedule::Lowering;
        float m_simulated_annealing_decay_factor = 0.8f;
        float m_simulated_annealing_initial_multiplier = 50.0f;
        size_t m_simulated_annealing_decay_epochs = 2;
        float m_simulated_annealing_min_multiplier = SimulatedAnnealingLearningRate::DEFAULT_MIN_MULTIPLIER;
        std::unique_ptr<ILearningRate> m_learning_rate_provider;
        size_t m_micro_batch_size = 1;
        size_t m_max_validation_windows = 4096;
        size_t m_validation_worst_count = 5;
        size_t m_optimizer_step = 0;
        size_t m_learning_rate_schedule_steps = 0;
        float m_last_logged_learning_rate = std::numeric_limits<float>::quiet_NaN();
        bool m_has_loaded_training_state = false;
        size_t m_loaded_learning_rate_step = 0;
        float m_loaded_current_learning_rate = 0.0f;
        size_t m_loaded_epochs_at_current_rate = 0;
        size_t m_checkpoint_epoch = 0;
        size_t m_checkpoint_window = 0;
        size_t m_checkpoint_window_count = 0;
        std::string m_checkpoint_rng_state;
        WeightInitializerType m_weight_initializer = WeightInitializerType::XavierInputProjections;
        FFNInitializerType m_ffn_initializer = FFNInitializerType::XavierInputProjections;
        EmbeddingInitializerType m_embedding_initializer = EmbeddingInitializerType::LegacyUniform;
        std::string m_training_parameters_json;
        std::string m_training_progress_filename = "train.json";

        void train_with_window(
            int window_size,
            bool verbose,
            size_t num_epochs,
            const std::optional<std::chrono::seconds>& checkpointing_interval
        );
        struct BatchTrainingItem
        {
            CpuInputLine line;
            bool finished = false;
            std::optional<PositionIndex> context_length;
        };
        struct BatchTrainingTiming
        {
            double forward_ms = 0.0;
            double backward_ms = 0.0;
            double apply_ms = 0.0;
            double backward_output_ms = 0.0;
            double backward_transformer_ms = 0.0;
            double backward_input_ms = 0.0;
            double primary_loss_sum = 0.0;
            size_t primary_loss_count = 0;
            size_t rounds = 0;

            double average_primary_loss() const
            {
                return primary_loss_count == 0
                    ? std::numeric_limits<double>::quiet_NaN()
                    : primary_loss_sum / static_cast<double>(primary_loss_count);
            }
        };
        void initialize_training_progress_log();
        void log_training_progress(
            const char* item_type,
            size_t epoch,
            size_t num_epochs,
            size_t range_start,
            size_t range_end,
            size_t total_items,
            size_t batch_size,
            size_t iterations,
            double batch_ms,
            const BatchTrainingTiming& timing
        );
        void log_validation_progress(
            const char* phase,
            size_t epoch,
            size_t num_epochs,
            double epoch_progress,
            size_t sample_count,
            const EvaluationMetrics& metrics
        );
        std::unique_ptr<nlohmann::json> m_training_progress_entries;
        std::chrono::steady_clock::time_point m_last_training_progress_flush{};
        std::vector<OptimizerDiagnosticMetrics> m_pending_optimizer_diagnostics;
        void flush_training_progress_log() const;
        void flush_training_progress_log_if_due(bool force = false);
        void apply_loaded_training_state_resets();
        size_t train_batch_items(std::vector<BatchTrainingItem>& batch, size_t steps, BatchTrainingTiming& timing);
        size_t train_batch_step(std::vector<BatchTrainingItem>& batch, size_t step, BatchTrainingTiming& timing);
        void collect_active_batch_inputs(
            const std::vector<BatchTrainingItem>& batch,
            std::vector<size_t>& active_indices,
            std::vector<CpuInputLine>& contexts
        ) const;
        void train_batch_output_heads(
            const std::vector<BatchTrainingItem>& batch,
            const std::vector<size_t>& active_indices,
            const std::vector<MultiTokenPredictionIndex>& valid_heads,
            BatchIndex packed_batch_size,
            VulkanQueue& queue,
            std::vector<float>& primary_losses,
            BatchTrainingTiming& timing
        );
        void train_window_all_positions(
            const std::vector<BatchTrainingItem>& batch,
            const std::vector<size_t>& active_indices,
            const PackedBatchInput& packed,
            VulkanQueue& queue,
            std::vector<float>& primary_losses,
            BatchTrainingTiming& timing
        );
        void train_batch_transformer_backward(PositionIndex packed_rows, BatchTrainingTiming& timing);
        size_t finish_converged_batch_items(
            std::vector<BatchTrainingItem>& batch,
            const std::vector<size_t>& active_indices,
            const std::vector<float>& primary_losses,
            size_t step
        );

        void do_whole_corpus_window_based_training(
            bool verbose,
            size_t num_epochs,
            const std::optional<std::chrono::seconds>& checkpointing_interval
        );
    };

} // namespace rllm
