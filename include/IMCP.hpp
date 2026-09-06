#pragma once

#include <cstddef>
#include <compare>
#include <string>
#include <string_view>

namespace rllm
{
    /** Stable, owning location of a concrete value in source text. */
    class SourceContext
    {
      public:
        SourceContext(std::string_view text, size_t offset, size_t length)
            : m_text(text), m_offset(offset), m_length(length)
        {}

        const std::string& text() const { return m_text; }
        size_t offset() const { return m_offset; }
        size_t length() const { return m_length; }
        std::string_view before() const { return std::string_view{m_text}.substr(0, m_offset); }
        std::string_view value() const { return std::string_view{m_text}.substr(m_offset, m_length); }
        std::string_view after() const { return std::string_view{m_text}.substr(m_offset + m_length); }

        auto operator<=>(const SourceContext&) const = default;

      private:
        std::string m_text;
        size_t m_offset;
        size_t m_length;
    };

    /** Location of a placeholder within generated model output. */
    class MCPContext
    {
      public:
        MCPContext(std::string_view text, size_t offset, size_t length)
            : m_text(text), m_offset(offset), m_length(length)
        {}

        std::string_view text() const { return m_text; }
        size_t offset() const { return m_offset; }
        size_t length() const { return m_length; }
        std::string_view before() const { return m_text.substr(0, m_offset); }
        std::string_view placeholder() const { return m_text.substr(m_offset, m_length); }
        std::string_view after() const { return m_text.substr(m_offset + m_length); }

      private:
        std::string_view m_text;
        size_t m_offset;
        size_t m_length;
    };

    /** Bridge between abstract model tokens and concrete source spellings.
     * Implementations may use prompt context, symbol lookup, or an external MCP
     * service to choose a concrete value for each generated placeholder. */
    class IMCP
    {
      public:
        virtual ~IMCP() = default;

        virtual void record_seen_identifier(
            const SourceContext& context, std::string_view identifier) = 0;
        virtual void record_seen_string(
            const SourceContext& context, std::string_view literal) = 0;
        virtual void record_seen_mcp(
            const SourceContext& context, std::string_view expression) = 0;

        virtual std::string map_identifier(const MCPContext& context) = 0;
        virtual std::string map_string(const MCPContext& context) = 0;
        virtual std::string map_mcp(const MCPContext& context) = 0;
    };

}
