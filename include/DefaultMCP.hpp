#pragma once

#include <IMCP.hpp>

#include <map>
#include <string>
#include <string_view>

namespace rllm
{
    /** Default MCP used when no external resolver is configured. */
    class DefaultMCP final : public IMCP
    {
      public:
        void record_seen_identifier(
            const SourceContext& context, std::string_view value) override
        { m_identifiers[context] = value; }
        void record_seen_string(
            const SourceContext& context, std::string_view value) override
        { m_strings[context] = value; }
        void record_seen_mcp(
            const SourceContext& context, std::string_view value) override
        { m_mcp_sections[context] = value; }

        std::string map_identifier(const MCPContext&) override { return "???"; }
        std::string map_string(const MCPContext&) override { return "???"; }
        std::string map_mcp(const MCPContext&) override { return "???"; }

        const std::map<SourceContext, std::string>& identifiers() const { return m_identifiers; }
        const std::map<SourceContext, std::string>& strings() const { return m_strings; }
        const std::map<SourceContext, std::string>& mcp_sections() const { return m_mcp_sections; }

      private:
        std::map<SourceContext, std::string> m_identifiers;
        std::map<SourceContext, std::string> m_strings;
        std::map<SourceContext, std::string> m_mcp_sections;
    };
}
