#ifndef SIM_NETWORK_PARSER_PLUGIN_DOT_H
#define SIM_NETWORK_PARSER_PLUGIN_DOT_H

#include <SgtSim/SimNetwork.h>
#include <SgtSim/SimParser.h>

namespace Sgt
{
    class Simulation;

    /// @brief ParserPlugin that parses the network keyword, adding a new SimNetwork to the model.
    class SimNetworkParserPlugin : public SimParserPlugin
    {
        public:
            virtual const char* key() override
            {
                return "network";
            }

        public:
            virtual void parse(const YAML::Node& nd, Simulation& sim, const ParserBase& parser) const override;
    };
}

#endif // SIM_NETWORK_PARSER_PLUGIN_DOT_H
