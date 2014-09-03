#ifndef SIM_BRANCH_DOT_H
#define SIM_BRANCH_DOT_H

#include <SgtSim/SimComponent.h>

#include <SgtCore/Branch.h>

namespace SmartGridToolbox
{
   class SimBranchAbc : public SimComponentAbc, public BranchAbc 
   {
      public:
         SimBranchAbc(const std::string& id, Phases phases0, Phases phases1) : Branch(id, phases0, phases1) {}
   };
}

#endif // SIM_BRANCH_DOT_H
