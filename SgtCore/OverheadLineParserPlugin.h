#ifndef OVERHEAD_LINE_PARSER_PLUGIN
#define OVERHEAD_LINE_PARSER_PLUGIN

#include <SgtCore/NetworkParser.h>

namespace Sgt
{
    class Network;
    class OverheadLine;

    /// @brief ParserPlugin that parses OverheadLine objects.
    class OverheadLineParserPlugin : public NetworkParserPlugin
    {
        public:
            virtual const char* key()
            {
                return "overhead_line";
            }

            virtual void parse(const YAML::Node& nd, Network& netw, const ParserBase& parser) const override;

            std::unique_ptr<OverheadLine> parseOverheadLine(const YAML::Node& nd, const ParserBase& parser) const;
    };
}

#endif // OVERHEAD_LINE_PARSER_PLUGIN
