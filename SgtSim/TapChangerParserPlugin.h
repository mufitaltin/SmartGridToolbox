// Copyright 2015 National ICT Australia Limited (NICTA)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TAP_CHANGER_PARSER_PLUGIN_DOT_H
#define TAP_CHANGER_PARSER_PLUGIN_DOT_H

#include <SgtSim/SimParser.h>

namespace Sgt
{
    class Simulation;

    /// @addtogroup SimYamlSpec
    /// @{
    /// <b>YAML schema for `tap_changer` keyword.</b>
    ///
    /// The `tap_changer` keyword adds a TapChanger to the Simulation.
    ///
    /// ~~~{.yaml}
    /// - tap_changer:
    ///     id:                     <string>        # Component ID.
    ///     sim_network_id:         <string>        # SimNetwork ID.
    ///     transformer_id:         <string>        # ID of transformer to which I'm attached.
    ///     taps:                   <real_array>    # Array of off nominal tap ratios. 
    ///     setpoint:               <real>          # Tap changer target/setpoint.
    ///     tolerance:              <real>          # Deadband is setpoint +- tolerance.
    ///     ctrl_side_idx:          <int>           # Monitor bus 0 or 1 of transformer?
    ///     winding_idx:            <int>           # Monitor which winding of transformer?
    ///     ratio_idx:              <int>           # Control ratio of which terminal of transformer?
    ///     # The following LDC (Line Drop Compensation) parameters are optional:
    ///     ldc_impedance:          <int>           # Fictitious impedance for applying LDC.
    ///     ldc_top_factor:         <int>           # Topological factor for LDC, multiplies winding I to give line I. 
    /// ~~~
    /// @}

    /// @brief Parses the `tap_changer` keyword, adding a TapChanger to the simulation.
    /// @ingroup SimParserPlugins
    class TapChangerParserPlugin : public SimParserPlugin
    {
        public:
        virtual const char* key() const override
        {
            return "tap_changer";
        }

        public:
        virtual void parse(const YAML::Node& nd, Simulation& sim, const ParserBase& parser) const override;
    };
}

#endif // TAP_CHANGER_PARSER_PLUGIN_DOT_H
