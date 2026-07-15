#include <Trainer.hpp>

#include <algorithm>
#include <fstream>
#include <print>
#include <string>

#include <vulkan_session.hpp>

namespace rllm
{
    Trainer::Trainer(const std::vector<std::string>& filters)
        : m_filters(filters)
    {}

    void Trainer::train_mode(
        const std::optional<std::string>& input_filename,
        const std::string& output_filename,
        size_t num_layers,
        bool verbose,
        TrainingMethod method,
        std::optional<std::chrono::seconds> checkpointing_interval,
        int window_size,
        size_t window_stride,
        size_t learn_depth,
        float learning_rate,
        LearningRateSchedule learning_rate_schedule,
        float simulated_annealing_decay_factor,
        float simulated_annealing_initial_multiplier,
        size_t simulated_annealing_decay_epochs,
        float simulated_annealing_min_multiplier,
        WeightInitializerType weight_initializer,
        FFNInitializerType ffn_initializer,
        EmbeddingInitializerType embedding_initializer,
        size_t micro_batch_size,
        size_t num_epochs,
        std::optional<size_t> epoch_size,
        const std::string& train_corpus_dir
    )
    {
        const float effective_learning_rate = learning_rate / static_cast<float>(std::max<size_t>(1, num_layers));
        std::println(
            "Training mode: depth {}, learning rate {} (effective per-layer {}), micro-batch size {}",
            learn_depth,
            learning_rate,
            effective_learning_rate,
            micro_batch_size
        );
        if (learning_rate > 0.25f)
        {
            std::println(
                "Warning: base learning rate {} is high for the current clipped optimizer and may saturate logits. "
                "Try --learning-rate 0.2 or lower if losses jump to ~38400.",
                learning_rate
            );
        }
        set_nn_log_file("train.log");
        ComputeKernelRegistry::instance().enableRegistrationLog("training-log.txt");

        Corpus corpus{m_filters};
        corpus.load_files_from_dir(train_corpus_dir);
        Statistics stats;

        auto nn = std::make_unique<TextTrainer>(num_layers, corpus, stats);
        nn->set_training_method(method);
        nn->set_window_size(window_size);
        nn->set_window_stride(window_stride);
        nn->set_learn_depth(learn_depth);
        nn->set_learning_rate(learning_rate);
        nn->set_learning_rate_schedule(learning_rate_schedule);
        nn->set_simulated_annealing_decay_factor(simulated_annealing_decay_factor);
        nn->set_simulated_annealing_initial_multiplier(simulated_annealing_initial_multiplier);
        nn->set_simulated_annealing_decay_epochs(simulated_annealing_decay_epochs);
        nn->set_simulated_annealing_min_multiplier(simulated_annealing_min_multiplier);
        nn->set_weight_initializer(weight_initializer);
        nn->set_ffn_initializer(ffn_initializer);
        nn->set_embedding_initializer(embedding_initializer);
        nn->set_micro_batch_size(micro_batch_size);

        nn->train(verbose, num_epochs, input_filename, checkpointing_interval, epoch_size);

        stats.print_statistics();
        parallel::statistics.print_statistics();

        nn->save(output_filename);
    }
} // namespace rllm
