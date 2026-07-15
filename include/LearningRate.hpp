#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace rllm
{
    class ILearningRate
    {
      public:
        virtual ~ILearningRate() = default;
        virtual float get_rate() = 0;
        virtual void advance_epoch() {}
    };

    class ConstantLearningRate final : public ILearningRate
    {
      public:
        explicit ConstantLearningRate(float rate)
            : m_rate(rate)
        {}

        float get_rate() override { return m_rate; }

      private:
        float m_rate;
    };

    class LoweringLearningRate final : public ILearningRate
    {
      public:
        static constexpr float WARMUP_FRACTION = 0.05f;
        static constexpr float MIN_SCALE = 0.10f;

        LoweringLearningRate(float rate, size_t total_steps)
            : m_rate(rate)
            , m_total_steps(total_steps)
        {}

        float get_rate() override
        {
            ++m_step;
            return m_rate * scale_for_step(m_step, m_total_steps);
        }

        static float scale_for_step(size_t step, size_t total_steps)
        {
            if (total_steps == 0)
                return 1.0f;

            step = std::clamp(step, size_t{1}, total_steps);
            const size_t warmup_steps = std::max<size_t>(
                1, total_steps / 20 + (total_steps % 20 != 0 ? 1 : 0));
            if (step <= warmup_steps)
                return static_cast<float>(step) / static_cast<float>(warmup_steps);
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

        void advance_epoch() override
        {
            ++m_epochs_at_current_rate;
            if (m_epochs_at_current_rate == m_epochs_per_decay)
            {
                m_current_rate = std::max(m_current_rate * m_decay_factor, m_min_rate);
                m_epochs_at_current_rate = 0;
            }
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
