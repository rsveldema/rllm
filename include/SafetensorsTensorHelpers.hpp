#pragma once

#include <safetensors.hh>

#include <LayerPrimitives.hpp>
#include <cpu/cpu_fixed_matrix.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace rllm::safetensors_helpers
{
    // Convert a cpu_fixed_matrix to a safetensors tensor (always kFLOAT32).
    template <typename T, typename X, typename Y>
    safetensors::tensor_t
    serialize_matrix(const cpu_fixed_matrix<T, X, Y>& m, std::vector<uint8_t>& storage)
    {
        safetensors::tensor_t tensor;
        tensor.dtype = safetensors::dtype::kFLOAT32;
        tensor.shape.resize(2);
        tensor.shape[0] = static_cast<size_t>(X::MAX);
        tensor.shape[1] = static_cast<size_t>(Y::MAX);

        // Compute the flat buffer size for float representation.
        size_t n_elements = static_cast<size_t>(X::MAX) * static_cast<size_t>(Y::MAX);
        size_t data_size = n_elements * sizeof(float);
        size_t dst_offset = storage.size();

        // Copy matrix as floats into the safetensors storage.
        std::vector<float> flat(n_elements);
        for (size_t x = 0; x < static_cast<size_t>(X::MAX); ++x)
            for (size_t y = 0; y < static_cast<size_t>(Y::MAX); ++y)
                flat[x * static_cast<size_t>(Y::MAX) + y] = static_cast<float>(m.get(static_cast<X>(x), static_cast<Y>(y)));

        size_t prev_size = storage.size();
        storage.resize(prev_size + data_size);
        std::copy(flat.begin(), flat.end(), reinterpret_cast<float*>(storage.data() + dst_offset));

        tensor.data_offsets[0] = dst_offset;
        tensor.data_offsets[1] = dst_offset + data_size;
        return tensor;
    }

    // Deserialize a safetensors tensor into a cpu_fixed_matrix.
    template <typename T, typename X, typename Y>
    void deserialize_matrix(const safetensors::tensor_t& tensor, cpu_fixed_matrix<T, X, Y>& m, std::string* err)
    {
        if (tensor.dtype != safetensors::dtype::kFLOAT32)
        {
            if (err) *err += "Expected kFLOAT32 tensor, got " + safetensors::get_dtype_str(tensor.dtype);
            return;
        }

        auto expected_rows = static_cast<size_t>(X::MAX);
        auto expected_cols = static_cast<size_t>(Y::MAX);
        if (tensor.shape.size() != 2 || tensor.shape[0] != expected_rows || tensor.shape[1] != expected_cols)
        {
            if (err) *err += "Tensor shape mismatch for matrix";
            return;
        }

        auto* data = reinterpret_cast<float const*>(static_cast<uint8_t const*>(nullptr) + tensor.data_offsets[0]);
        // Note: this function doesn't have direct access to the storage buffer.
        // The caller must ensure the tensor is still valid (e.g., safetensors_t kept alive).
        for (size_t x = 0; x < expected_rows; ++x)
            for (size_t y = 0; y < expected_cols; ++y)
                m.set(static_cast<X>(x), static_cast<Y>(y), static_cast<T>(data[x * expected_cols + y]));
    }

    // Convert a fixed_size_vector to a safetensors tensor (kFLOAT32, shape [N]).
    template <typename Enum, typename T>
    safetensors::tensor_t
    serialize_vector(const fixed_size_vector<T, Enum>& v, std::vector<uint8_t>& storage)
    {
        safetensors::tensor_t tensor;
        tensor.dtype = safetensors::dtype::kFLOAT32;
        tensor.shape.resize(1);
        tensor.shape[0] = static_cast<size_t>(Enum::MAX);

        size_t n_elements = static_cast<size_t>(Enum::MAX);
        size_t data_size = n_elements * sizeof(float);
        size_t dst_offset = storage.size();

        std::vector<float> flat(n_elements);
        for (size_t i = 0; i < n_elements; ++i)
            flat[i] = static_cast<float>(v[static_cast<Enum>(i)]);

        size_t prev_size = storage.size();
        storage.resize(prev_size + data_size);
        std::copy(flat.begin(), flat.end(), reinterpret_cast<float*>(storage.data() + dst_offset));

        tensor.data_offsets[0] = dst_offset;
        tensor.data_offsets[1] = dst_offset + data_size;
        return tensor;
    }

    // Deserialize a safetensors 1D tensor into a fixed_size_vector.
    template <typename Enum, typename T>
    void deserialize_vector(const safetensors::tensor_t& tensor, fixed_size_vector<T, Enum>& v, std::string* err)
    {
        if (tensor.dtype != safetensors::dtype::kFLOAT32)
        {
            if (err) *err += "Expected kFLOAT32 tensor";
            return;
        }

        auto expected_size = static_cast<size_t>(Enum::MAX);
        if (tensor.shape.size() != 1 || tensor.shape[0] != expected_size)
        {
            if (err) *err += "Tensor shape mismatch for vector";
            return;
        }

        auto* data = reinterpret_cast<float const*>(static_cast<uint8_t const*>(nullptr) + tensor.data_offsets[0]);
        for (size_t i = 0; i < expected_size; ++i)
            v[static_cast<Enum>(i)] = static_cast<T>(data[i]);
    }

    // Check if a safetensors file is readable.
    bool is_safetensors_file(const std::string& filename, std::string* err);

    // Create an empty safetensors object with metadata version=2.
    safetensors::safetensors_t create_metadata();
} // namespace rllm::safetensors_helpers
