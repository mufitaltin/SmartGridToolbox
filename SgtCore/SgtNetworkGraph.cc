#include "SgtNetworkGraph.h"

namespace SmartGridToolbox
{
   void SgtNetworkGraph::create(const Network& nw)
   {
      for (auto bus : nw.busses())
      {
         this->addNode(bus->id(), SgtGraphNodeInfo{0.0, 0.0, 10.0, 10.0, bus.get()});
      }
      for (auto branch : nw.branches())
      {
         this->addArc(branch->id(), branch->bus0()->id(), branch->bus1()->id(), SgtGraphArcInfo{0.0, branch.get()});
      }
   };
};