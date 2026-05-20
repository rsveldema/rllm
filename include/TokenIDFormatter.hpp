#pragma once
#include <Corpus.hpp>
#include <format>

namespace std {
    template<>
    struct formatter<rllm::TokenID> {
        constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
        auto format(const rllm::TokenID& id, std::format_context& ctx) const {
            return std::format_to(ctx.out(), "{}", static_cast<std::underlying_type_t<rllm::TokenID>>(id));
        }
    };
}
