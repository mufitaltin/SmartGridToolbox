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

#include "WeatherParserPlugin.h"

#include "Simulation.h"
#include "Weather.h"

namespace Sgt
{
    void WeatherParserPlugin::parse(const YAML::Node& nd, Simulation& sim, const ParserBase& parser) const
    {
        assertFieldPresent(nd, "id");

        string id = parser.expand<std::string>(nd["id"]);
        auto weather = sim.newSimComponent<Weather>(id);

        weather->data.latLong = sim.latLong(); // TODO: could be restrictive?

        if (nd["dt"])
        {
            weather->setDt(parser.expand<Time>(nd["dt"]));
        }
        else
        {
            weather->setDt(posix_time::minutes(5.0));
        }

        const auto& temperatureNd = nd["temperature"];
        if (temperatureNd)
        {
            std::string nd1Key = nd.begin()->first.as<std::string>();
            YAML::Node nd1 = nd.begin()->second;
            if (nd1Key == "series")
            {
                std::string id = parser.expand<std::string>(nd1);
                auto series = sim.timeSeries<TimeSeries<Time, double>>(id);
                sgtAssert(series != nullptr, "Parsing weather: couldn't find time series " << id << ".");
                weather->data.setTemperature(series);
            }
            else if (nd1Key == "const")
            {
                weather->data.setTemperature(nd1.as<double>());
            }
        }

        const auto& irradianceNd = nd["irradiance"];
        if (irradianceNd)
        {
            std::string nd1Key = nd.begin()->first.as<std::string>();
            YAML::Node nd1 = nd.begin()->second;
            if (nd1Key == "series")
            {
                std::string id = parser.expand<std::string>(nd1);
                auto series = sim.timeSeries<TimeSeries<Time, arma::Col<double>>>(id);
                sgtAssert(series != nullptr, "Parsing weather: couldn't find time series " << id << ".");
                weather->data.setIrradiance(series);
            }
            else if (nd1Key == "const")
            {
                auto vec = nd1.as<arma::Col<double>>();
                weather->data.setIrradiance({{{vec(0), vec(1), vec(2)}}, vec(3), vec(4)}); 
            }
            else if (nd1Key == "solar_model" && nd1.as<std::string>() == "standard")
            {
                weather->data.setIrradianceToSunModel();
            }
        }

        const auto& cloudAttenNd = nd["cloud_attenuation_factors"];
        if (cloudAttenNd)
        {
            std::string nd1Key = nd.begin()->first.as<std::string>();
            YAML::Node nd1 = nd.begin()->second;
            if (nd1Key == "series")
            {
                std::string id = parser.expand<std::string>(nd1);
                auto series = sim.timeSeries<TimeSeries<Time, arma::Col<double>>>(id);
                sgtAssert(series != nullptr, "Parsing weather: couldn't find time series " << id << ".");
                weather->data.setCloudAttenuationFactors(series);
            }
            else if (nd1Key == "const")
            {
                auto vec = nd1.as<arma::Col<double>>();
                weather->data.setCloudAttenuationFactors({{vec(0), vec(1), vec(2)}}); 
            }
        }

        const auto& windNd = nd["wind"];
        if (windNd)
        {
            std::string nd1Key = nd.begin()->first.as<std::string>();
            YAML::Node nd1 = nd.begin()->second;
            if (nd1Key == "series")
            {
                std::string id = parser.expand<std::string>(nd1);
                auto series = sim.timeSeries<TimeSeries<Time, arma::Col<double>>>(id);
                sgtAssert(series != nullptr, "Parsing weather: couldn't find time series " << id << ".");
                weather->data.setWindVector(series);
            }
            else if (nd1Key == "const")
            {
                auto vec = nd1.as<arma::Col<double>>();
                weather->data.setWindVector({{vec(0), vec(1), vec(2)}}); 
            }
        }

    }
}
