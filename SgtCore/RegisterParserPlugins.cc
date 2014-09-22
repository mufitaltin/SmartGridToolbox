#include "Parser.h"
#include "BusParserPlugin.h"
#include "CommonBranchParserPlugin.h"
#include "DgyTransformerParserPlugin.h"
#include "GenericBranchParserPlugin.h"
#include "GenericGenParserPlugin.h"
#include "GenericZipParserPlugin.h"
#include "MatpowerParserPlugin.h"
#include "OverheadLineParserPlugin.h"

namespace SmartGridToolbox
{
   class Network;
   template<> void registerParserPlugins<Network>(Parser<Network>& p)
   {
      p.registerParserPlugin<BusParserPlugin>();
      p.registerParserPlugin<CommonBranchParserPlugin>();
      p.registerParserPlugin<DgyTransformerParserPlugin>();
      p.registerParserPlugin<GenericBranchParserPlugin>();
      p.registerParserPlugin<GenericGenParserPlugin>();
      p.registerParserPlugin<GenericZipParserPlugin>();
      p.registerParserPlugin<MatpowerParserPlugin>();
      p.registerParserPlugin<OverheadLineParserPlugin>();
   }
}
