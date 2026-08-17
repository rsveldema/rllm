#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cassert>
#include <numbers>

namespace rllm
{
    inline constexpr float DEFAULT_DEPTH_LEARNING_RATE_MULTIPLIER = 1.05f;

    inline float depth_learning_rate_multiplier(
        size_t stage_index,
        size_t stage_count,
        float output_multiplier = DEFAULT_DEPTH_LEARNING_RATE_MULTIPLIER)
    {
        if (stage_count <= 1)
            return 1.0f;
        assert(stage_index < stage_count);
        assert(output_multiplier >= 1.0f && output_multiplier < 2.0f);
        const float input_multiplier = 2.0f - output_multiplier;
        const float depth = static_cast<float>(stage_index) / static_cast<float>(stage_count - 1);
        return input_multiplier + depth * (output_multiplier - input_multiplier);
    }

    class ILearningRate
    {
      public:
        virtual ~ILearningRate() = default;
        virtual float get_rate() = 0;
        virtual float current_rate() const = 0;
        virtual void advance_epoch() {}
        virtual size_t step() const { return 0; }
        virtual size_t epochs_at_current_rate() const { return 0; }
        virtual void restore(size_t, float, size_t) {}
    };

    class ConstantLearningRate final : public ILearningRate
    {
      public:
        explicit ConstantLearningRate(float rate)
            : m_rate(rate)
        {}

        float get_rate() override { return m_rate; }
        float current_rate() const override { return m_rate; }

      private:
        float m_rate;
    };

    class LoweringLearningRate final : public ILearningRate
    {
      public:
        static constexpr float DEFAULT_WARMUP_PERCENT = 5.0f;
        static constexpr float MIN_SCALE = 0.10f;

        LoweringLearningRate(
            float rate, size_t total_steps,
            float warmup_percent = DEFAULT_WARMUP_PERCENT,
            bool skip_warmup = false)
            : m_rate(rate)
            , m_total_steps(total_steps)
            , m_warmup_percent(warmup_percent)
            , m_skip_warmup(skip_warmup)
        {
            assert(warmup_percent > 0.0f && warmup_percent <= 100.0f);
        }

        float get_rate() override
        {
            ++m_step;
            return m_rate * scale_for_step(m_step, m_total_steps, m_warmup_percent, m_skip_warmup);
        }

        size_t step() const override { return m_step; }
        float current_rate() const override { return m_rate * scale_for_step(std::max<size_t>(m_step, 1), m_total_steps, m_warmup_percent, m_skip_warmup); }
        void restore(size_t step, float, size_t) override { m_step = step; }

        static size_t warmup_steps_for(size_t total_steps, float warmup_percent)
        {
            assert(warmup_percent > 0.0f && warmup_percent <= 100.0f);
            if (total_steps == 0)
                return 0;
            return std::max<size_t>(1, static_cast<size_t>(std::ceil(
                static_cast<double>(total_steps) * static_cast<double>(warmup_percent) / 100.0)));
        }

        static float scale_for_step(
            size_t step, size_t total_steps,
            float warmup_percent = DEFAULT_WARMUP_PERCENT,
            bool skip_warmup = false)
        {
            if (total_steps == 0)
                return 1.0f;

            step = std::clamp(step, size_t{1}, total_steps);
            const size_t warmup_steps = warmup_steps_for(total_steps, warmup_percent);
            if (step <= warmup_steps)
                return skip_warmup ? 1.0f :
                    static_cast<float>(step) / static_cast<float>(warmup_steps);
            if (warmup_steps == total_steps)
                return 1.0f;

            const float decay_progress = static_cast<float>(step - warmup_steps) /
                static_cast<float>(total_steps - warmup_steps);
            const float cosine = 0.5f * (1.0f + std::cos(std::numbers::pi_v<float> * decay_progress));
            return MIN_SCALE + (1.0f - MIN_SCALE) * cosine;
        }

      private:
        float m_rate;
        size_t m_total_steps;
        float m_warmup_percent;
        bool m_skip_warmup;
        size_t m_step = 0;
    };

    class SimulatedAnnealingLearningRate final : public ILearningRate
    {
      public:
        static constexpr float DEFAULT_MIN_MULTIPLIER = 1.0f / 50.0f;
        SimulatedAnnealingLearningRate(
            float rate, float decay_factor, float initial_multiplier,
            size_t epochs_per_decay, float min_multiplier = DEFAULT_MIN_MULTIPLIER)
            : m_min_rate(rate * min_multiplier)
            , m_current_rate(std::max(rate * initial_multiplier, m_min_rate))
            , m_decay_factor(decay_factor)
            , m_epochs_per_decay(epochs_per_decay)
        { assert(min_multiplier > 0.0f); }

        float get_rate() override
        {
            return m_current_rate;
        }
        float current_rate() const override { return m_current_rate; }

        void advance_epoch() override
        {
            ++m_epochs_at_current_rate;
            if (m_epochs_at_current_rate == m_epochs_per_decay)
            {
                m_current_rate = std::max(m_current_rate * m_decay_factor, m_min_rate);
                m_epochs_at_current_rate = 0;
            }
        }

        size_t epochs_at_current_rate() const override { return m_epochs_at_current_rate; }
        void restore(size_t, float current_rate, size_t epochs) override
        {
            m_current_rate = std::max(current_rate, m_min_rate);
            m_epochs_at_current_rate = epochs % m_epochs_per_decay;
        }

      private:
        float m_min_rate;
        float m_current_rate;
        float m_decay_factor;
        size_t m_epochs_per_decay;
        size_t m_epochs_at_current_rate = 0;
    };

    enum class LearningRateSchedule
    {
        Constant,
        Lowering,
        SimulatedAnnealing,
    };
}
