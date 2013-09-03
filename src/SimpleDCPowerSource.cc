#include "SimpleDCPowerSource.h"

#include "Model.h"
#include "InverterBase.h"

namespace SmartGridToolbox
{
   void SimpleDCPowerSourceParser::parse(const YAML::Node & nd, Model & mod) const
   {
      SGT_DEBUG(debug() << "SimpleDCPowerSource : parse." << std::endl);

      assertFieldPresent(nd, "name");
      assertFieldPresent(nd, "inverter");
      assertFieldPresent(nd, "dc_power");
      const std::string nameStr = nd["name"].as<std::string>();
      auto & comp = mod.newComponent<SimpleDCPowerSource>(nameStr);
      comp.setPDC(nd["dc_power"].as<double>());
   }

   void SimpleDCPowerSourceParser::postParse(const YAML::Node & nd, Model & mod) const
   {
      SGT_DEBUG(debug() << "SimpleDCPowerSource : postParse." << std::endl);

      const std::string nameStr = nd["name"].as<std::string>();
      SimpleDCPowerSource & comp = *mod.componentNamed<SimpleDCPowerSource>(nameStr);

      const std::string inverterStr = nd["inverter"].as<std::string>();
      InverterBase * inverterComp = mod.componentNamed<InverterBase>(inverterStr);
      if (inverterComp != nullptr)
      {
         inverterComp->addDCPowerSource(comp);
      }
      else
      {
         error() << "For component " << nameStr << ", inverter " << inverterStr << " was not found in the model." 
                 << std::endl;
         abort();
      }
   }
}