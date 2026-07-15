#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string_view>
#include <RandomHelpers.hpp>

namespace rllm
{
    enum class WeightInitializerType { XavierUniform, XavierInputProjections, LegacyUniform };
    enum class FFNInitializerType { XavierUniform, XavierInputProjections, LegacyUniform };
    enum class EmbeddingInitializerType { VarianceScaledUniform, LegacyUniform };

    inline std::string_view initializer_name(WeightInitializerType type)
    {
        if (type == WeightInitializerType::XavierUniform)
            return "xavier-uniform";
        if (type == WeightInitializerType::XavierInputProjections)
            return "xavier-input-projections";
        return "legacy-uniform";
    }

    inline std::string_view initializer_name(FFNInitializerType type)
    {
        if (type == FFNInitializerType::XavierUniform)
            return "xavier-uniform";
        if (type == FFNInitializerType::XavierInputProjections)
            return "xavier-input-projections";
        return "legacy-uniform";
    }

    inline std::string_view initializer_name(EmbeddingInitializerType type)
    {
        return type == EmbeddingInitializerType::VarianceScaledUniform ? "variance-scaled-uniform" : "legacy-uniform";
    }
    class WeightInitializer
    {
      public:
        virtual ~WeightInitializer() = default;
        virtual float getNextValue() = 0;
    };

    class FFNInitializer
    {
      public:
        virtual ~FFNInitializer() = default;
        virtual float getNextValue() = 0;
    };

    class EmbeddingInitializer
    {
      public:
        virtual ~EmbeddingInitializer() = default;
        virtual float getNextValue() = 0;
    };

    inline float xavier_uniform_bound(size_t fan_in, size_t fan_out)
    {
        assert(fan_in > 0 && fan_out > 0);
        return std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    }

    inline float embedding_uniform_bound(size_t dimension)
    {
        assert(dimension > 0);
        // Uniform(-a, a) has variance a^2 / 3. This gives variance 1 / dimension.
        return std::sqrt(3.0f / static_cast<float>(dimension));
    }

    class XavierUniformWeightInitializer final : public WeightInitializer
    {
      public:
        XavierUniformWeightInitializer(size_t fan_in, size_t fan_out)
            : m_bound(xavier_uniform_bound(fan_in, fan_out))
        {}

        float getNextValue() override { return get_random_value(-m_bound, m_bound); }

      private:
        float m_bound;
    };

    class XavierUniformFFNInitializer final : public FFNInitializer
    {
      public:
        XavierUniformFFNInitializer(size_t fan_in, size_t fan_out)
            : m_bound(xavier_uniform_bound(fan_in, fan_out))
        {}

        float getNextValue() override { return get_random_value(-m_bound, m_bound); }

      private:
        float m_bound;
    };

    class VarianceScaledEmbeddingInitializer final : public EmbeddingInitializer
    {
      public:
        explicit VarianceScaledEmbeddingInitializer(size_t dimension)
            : m_bound(embedding_uniform_bound(dimension))
        {}

        float getNextValue() override { return get_random_value(-m_bound, m_bound); }

      private:
        float m_bound;
    };

    class LegacyUniformWeightInitializer final : public WeightInitializer
    {
      public:
        explicit LegacyUniformWeightInitializer(size_t fan_in)
            : m_bound(1.0f / std::sqrt(static_cast<float>(fan_in)))
        { assert(fan_in > 0); }
        float getNextValue() override { return get_random_value(-m_bound, m_bound); }
      private:
        float m_bound;
    };

    class LegacyUniformFFNInitializer final : public FFNInitializer
    {
      public:
        explicit LegacyUniformFFNInitializer(size_t fan_in)
            : m_bound(1.0f / std::sqrt(static_cast<float>(fan_in)))
        { assert(fan_in > 0); }
        float getNextValue() override { return get_random_value(-m_bound, m_bound); }
      private:
        float m_bound;
    };

    class LegacyUniformEmbeddingInitializer final : public EmbeddingInitializer
    {
      public:
        float getNextValue() override { return get_random_value(-0.1f, 0.1f); }
    };

    inline std::unique_ptr<WeightInitializer> make_weight_initializer(
        WeightInitializerType type, size_t fan_in, size_t fan_out)
    {
        if (type == WeightInitializerType::XavierUniform)
            return std::make_unique<XavierUniformWeightInitializer>(fan_in, fan_out);
        return std::make_unique<LegacyUniformWeightInitializer>(fan_in);
    }

    inline std::unique_ptr<FFNInitializer> make_ffn_initializer(
        FFNInitializerType type, size_t fan_in, size_t fan_out)
    {
        if (type == FFNInitializerType::XavierUniform)
            return std::make_unique<XavierUniformFFNInitializer>(fan_in, fan_out);
        return std::make_unique<LegacyUniformFFNInitializer>(fan_in);
    }

    inline std::unique_ptr<EmbeddingInitializer> make_embedding_initializer(
        EmbeddingInitializerType type, size_t dimension)
    {
        if (type == EmbeddingInitializerType::VarianceScaledUniform)
            return std::make_unique<VarianceScaledEmbeddingInitializer>(dimension);
        return std::make_unique<LegacyUniformEmbeddingInitializer>();
    }
}
