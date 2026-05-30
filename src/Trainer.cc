#include <Trainer.hpp>

#include <print>
#include <string>

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
        size_t num_epochs,
        const std::string& train_corpus_dir
    )
    {
        std::println("Training mode");
        set_nn_log_file("train.log");

        Corpus corpus{m_filters};
        corpus.load_files_from_dir(train_corpus_dir);
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(num_layers, corpus, stats);
        nn->set_training_method(method);
        nn->set_window_size(window_size);

        nn->train(verbose, num_epochs, input_filename, checkpointing_interval);

        stats.print_statistics();

        nn->save(output_filename);
    }
} // namespace rllm