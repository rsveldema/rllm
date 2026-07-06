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
        size_t learn_depth,
        float learning_rate,
        size_t num_epochs,
        const std::string& train_corpus_dir
    )
    {
        const float effective_learning_rate = learning_rate / static_cast<float>(std::max<size_t>(1, num_layers));
        std::println("Training mode: depth {}, learning rate {} (effective per-layer {})", learn_depth, learning_rate, effective_learning_rate);
        if (effective_learning_rate > 0.01f)
        {
            std::println(
                "Warning: effective learning rate {} is high for the current clipped optimizer and may saturate logits. "
                "Try --learning-rate 0.01 or lower if losses jump to ~38400.",
                effective_learning_rate
            );
        }
        set_nn_log_file("train.log");
        ComputeKernelRegistry::instance().enableRegistrationLog("training-log.txt");

        Corpus corpus{m_filters};
        corpus.load_files_from_dir(train_corpus_dir);
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(num_layers, corpus, stats);
        nn->set_training_method(method);
        nn->set_window_size(window_size);
        nn->set_learn_depth(learn_depth);
        nn->set_learning_rate(learning_rate);

        nn->train(verbose, num_epochs, input_filename, checkpointing_interval);

        stats.print_statistics();
        parallel::statistics.print_statistics();

        nn->save(output_filename);
    }
} // namespace rllm
