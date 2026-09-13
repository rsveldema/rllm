#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <vector>
#include <nlohmann/json.hpp>

namespace rllm
{
    /** A learned categorical projection over source-local string-table indices.
     * Rows grow with the supplied tables, independently of the token vocabulary.
     * The CPU implementation also returns gradients for the shared hidden state.
     */
    class StringIndexHead
    {
      public:
        explicit StringIndexHead(size_t dimensions) : m_dimensions(dimensions) {}

        void reserve_entries(size_t count)
        {
            const size_t old_size = m_weights.size();
            if (count <= entries()) return;
            m_weights.resize(count * m_dimensions);
            m_first.resize(m_weights.size());
            m_second.resize(m_weights.size());
            for (size_t i = old_size; i < m_weights.size(); ++i)
            {
                // Stable initialization: table growth does not reset learned rows.
                uint64_t bits = (i + 1) * 0x9e3779b97f4a7c15ULL;
                bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
                bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
                bits ^= bits >> 31;
                m_weights[i] = (float(bits & 0xffff) / 65535.0f - 0.5f) /
                    std::sqrt(static_cast<float>(m_dimensions));
            }
        }

        size_t entries() const { return m_weights.size() / m_dimensions; }

        std::vector<float> logits(std::span<const float> hidden, size_t count) const
        {
            if (hidden.size() != m_dimensions || count > entries())
                std::abort();
            std::vector<float> result(count);
            for (size_t row = 0; row < count; ++row)
                for (size_t d = 0; d < m_dimensions; ++d)
                    result[row] += m_weights[row * m_dimensions + d] * hidden[d];
            return result;
        }

        size_t predict(std::span<const float> hidden, size_t count) const
        {
            if (!count) return static_cast<size_t>(-1);
            const auto scores = logits(hidden, count);
            return std::max_element(scores.begin(), scores.end()) - scores.begin();
        }

        float loss(std::span<const float> hidden, size_t count, size_t expected,
                   std::vector<float>& delta, float scale = 1.0f) const
        {
            if (expected >= count) std::abort();
            delta = logits(hidden, count);
            const float maximum = *std::max_element(delta.begin(), delta.end());
            const float target_logit = delta[expected];
            double sum = 0;
            for (auto& value : delta) { value = std::exp(value - maximum); sum += value; }
            for (auto& value : delta) value = static_cast<float>(value / sum) * scale;
            delta[expected] -= scale;
            return maximum - target_logit + static_cast<float>(std::log(sum));
        }

        void backward(std::span<const float> hidden, std::span<const float> delta,
                      std::span<float> dh, std::vector<float>& gradient) const
        {
            gradient.resize(m_weights.size());
            for (size_t row = 0; row < delta.size(); ++row)
                for (size_t d = 0; d < m_dimensions; ++d)
                {
                    const size_t i = row * m_dimensions + d;
                    dh[d] += m_weights[i] * delta[row];
                    gradient[i] += hidden[d] * delta[row];
                }
        }

        void update(const std::vector<float>& gradient, float rate, float correction1, float correction2)
        {
            double norm = 0;
            for (const auto value : gradient) norm += double(value) * value;
            const float scale = norm > 1 ? static_cast<float>(1 / std::sqrt(norm)) : 1;
            for (size_t i = 0; i < gradient.size(); ++i)
            {
                const float g = gradient[i] * scale;
                m_first[i] = .9f * m_first[i] + .1f * g;
                m_second[i] = .999f * m_second[i] + .001f * g * g;
                m_weights[i] -= rate * (m_first[i] / correction1) /
                    (std::sqrt(m_second[i] / correction2) + 1e-8f);
            }
        }

        void reset_optimizer()
        {
            std::fill(m_first.begin(), m_first.end(), 0);
            std::fill(m_second.begin(), m_second.end(), 0);
        }

        const std::vector<float>& weights() const { return m_weights; }
        const std::vector<float>& first_moment() const { return m_first; }
        const std::vector<float>& second_moment() const { return m_second; }
        size_t dimensions() const { return m_dimensions; }
        void load_state(std::vector<float> weights, std::vector<float> first, std::vector<float> second)
        {
            if (weights.size() % m_dimensions || first.size() != weights.size() || second.size() != weights.size())
                std::abort();
            m_weights = std::move(weights); m_first = std::move(first); m_second = std::move(second);
        }

        nlohmann::json save() const
        { return {{"dimensions", m_dimensions}, {"weights", m_weights}, {"first", m_first}, {"second", m_second}}; }
        void load(const nlohmann::json& json)
        {
            if (json.at("dimensions").get<size_t>() != m_dimensions)
                std::abort();
            auto weights = json.at("weights").get<std::vector<float>>();
            auto first = json.at("first").get<std::vector<float>>();
            auto second = json.at("second").get<std::vector<float>>();
            load_state(std::move(weights), std::move(first), std::move(second));
        }

      private:
        size_t m_dimensions;
        std::vector<float> m_weights, m_first, m_second;
    };
}
