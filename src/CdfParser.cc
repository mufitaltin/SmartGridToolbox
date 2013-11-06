#include "CdfParser.h"
#include <SmartGridToolbox/Bus.h>
#include <SmartGridToolbox/Network.h>
#include <SmartGridToolbox/PowerFlow.h>
#include <SmartGridToolbox/ZipToGround.h>

#include <fstream>
#include <list>
#include <map>

namespace SmartGridToolbox
{
   // All quantities expressed as injections:
   struct CdfBusInfo
   {
      std::string name;
      int CdfType;

      double VBase;  // SI.
      Complex VPu;   // Per unit.
      Complex SLoad; // SI.
      Complex Sg;    // SI.
      double QMin;   // SI.
      double QMax;   // SI.
      Complex ysPu;  // Per unit.
   };

   struct CdfBranchInfo
   {
      int bus0Id; 
      int bus1Id;

      double RsPu;
      double XsPu;
      double BcPu;
      double tap;
      double thetaRad;
   };

   static std::string busName(const std::string & prefix, int id)
   {
      return prefix + "_bus_" + std::to_string(id);
   }

   static std::string zipName(const std::string & prefix, int id)
   {
      return prefix + "_zip_" + std::to_string(id);
   }

   static std::string branchName(const std::string & prefix, int nBranch, int id1, int id2)
   {
      return prefix + "_branch_" + std::to_string(nBranch) + "_" + std::to_string(id1) + "_" + std::to_string(id2);
   }

   static void errInvalidCdf()
   {
      error() << "The CDF file is not in the correct format." << std::endl;
      abort();
   }

   void CdfParser::parse(const YAML::Node & nd, Model & mod, const ParserState & state) const
   {
      const double MW = 1e6;

      SGT_DEBUG(debug() << "CDF : parse." << std::endl);

      assertFieldPresent(nd, "input_file");
      assertFieldPresent(nd, "network_name");
      assertFieldPresent(nd, "default_V_base");

      std::string inputName = state.expandName(nd["input_file"].as<std::string>());
      std::string networkName = state.expandName(nd["network_name"].as<std::string>());

      Phases phases;
      const YAML::Node ndPhases = nd["phases"];
      if (ndPhases)
      {
         phases = ndPhases.as<Phases>();
      }
      else
      {
         phases = Phase::BAL;
      }

      const YAML::Node ndPerUnit = nd["use_per_unit"];
      bool usePerUnit = ndPerUnit ? ndPerUnit.as<bool>() : false;
      
      double defaultVBase = nd["default_V_base"].as<double>();

      const YAML::Node ndFreq = nd["freq_Hz"];
      double freq = ndFreq ? ndFreq.as<double>() : 50.0;

      std::map<int, CdfBusInfo> busMap;
      std::list<CdfBranchInfo> branchList;
      double SBase;
      {
         // Parse in the raw CDF data.
         std::string line;
         std::fstream infile(inputName);
         if (!infile.is_open())
         {
            error() << "Could not open the CDF input file " << inputName << "." << std::endl;
            abort();
         }

         if (!std::getline(infile, line)) errInvalidCdf();
         double MVABase;
         std::istringstream(line.substr(31, 6)) >> MVABase;

         SBase = MW * MVABase;

         if (!std::getline(infile, line)) errInvalidCdf();
         if (line.substr(0, 3) != "BUS") errInvalidCdf();
         int i = 0;
         for (std::getline(infile, line); line.substr(0, 4) != std::string("-999"); std::getline(infile, line), ++i)
         {
            int busId;
            std::istringstream(line.substr(0, 4)) >> busId;
            CdfBusInfo & busInfo = busMap[busId];

            busInfo.name = busName(networkName, busId);

            std::istringstream(line.substr(24, 2)) >> busInfo.CdfType;

            double KVBase;
            std::istringstream(line.substr(76, 7)) >> KVBase;
            busInfo.VBase = KVBase == 0 ? defaultVBase : KVBase * 1000;

            double VmPu;
            std::istringstream(line.substr(27, 6)) >> VmPu;
            double VaDeg;
            std::istringstream(line.substr(33, 7)) >> VaDeg;
            busInfo.VPu = polar(VmPu, VaDeg * pi/180);

            double PdMW;
            std::istringstream(line.substr(40, 9)) >> PdMW;
            double QdMW;
            std::istringstream(line.substr(49, 10)) >> QdMW;
            busInfo.SLoad = -Complex(PdMW, QdMW) * MW; // Injection is -ve load.

            double PgMW;
            std::istringstream(line.substr(59, 8)) >> PgMW;
            double QgMW;
            std::istringstream(line.substr(67, 8)) >> QgMW;
            busInfo.Sg = Complex(PgMW, QgMW) * MW; // Injection is -ve load.
            
            double QMinMW;
            std::istringstream(line.substr(98, 8)) >> QMinMW;
            busInfo.QMin = QMinMW * MW;

            double QMaxMW;
            std::istringstream(line.substr(90, 8)) >> QMaxMW;
            busInfo.QMax = QMaxMW * MW;

            double GsPu;
            std::istringstream(line.substr(106, 8)) >> GsPu;
            double BsPu;
            std::istringstream(line.substr(114, 8)) >> BsPu;
            busInfo.ysPu = Complex(GsPu, BsPu);
         }

         std::getline(infile, line); 
         if (line.substr(0, 6) != "BRANCH") errInvalidCdf();

         for (std::getline(infile, line); line.substr(0, 4) != std::string("-999"); std::getline(infile, line))
         {
            branchList.push_back(CdfBranchInfo());
            CdfBranchInfo & branchInfo = branchList.back();

            std::istringstream(line.substr(0, 4)) >> branchInfo.bus0Id;
            std::istringstream(line.substr(5, 4)) >> branchInfo.bus1Id;
            std::istringstream(line.substr(19, 10)) >> branchInfo.RsPu;
            std::istringstream(line.substr(29, 11)) >> branchInfo.XsPu;
            std::istringstream(line.substr(40, 10)) >> branchInfo.BcPu;
            std::istringstream(line.substr(76, 6)) >> branchInfo.tap;
            if (branchInfo.tap == 0.0)
            {
               branchInfo.tap = 1.0; // 0 means "default", i.e. 1.
            }
            std::istringstream(line.substr(83, 7)) >> branchInfo.thetaRad;
            branchInfo.thetaRad *= pi / 180.0;
         }
      }
      
      {
         Network & netw = mod.newComponent<Network>(networkName, freq);
         for (const auto pair : busMap)
         {
            const int & busId = pair.first;
            const CdfBusInfo & info = pair.second;
            BusType type = BusType::BAD;
            switch (info.CdfType)
            {
               case 0 :
               case 1 :
                  type = BusType::PQ;
                  break;
               case 2 :
                  type = BusType::PV;
                  break;
               case 3 :
                  type = BusType::SL;
                  break;
               case 4:
                  error() << "CDF isolated bus type not supported." << std::endl; 
                  abort();
                  break;
               default:
                  error() << "Bad CDF bus type (type = " << info.CdfType << ") encountered." << std::endl;
                  abort();
                  break;
            }

            Complex V = usePerUnit ? info.VPu : info.VPu * info.VBase;
            Complex SLoad = usePerUnit ? info.SLoad / SBase : info.SLoad;
            Complex Sg = usePerUnit ? info.Sg / SBase : info.Sg;
            Complex ys = usePerUnit ? info.ysPu : info.ysPu * SBase / (info.VBase * info.VBase);
            double QMin = usePerUnit ? info.QMin / SBase : info.QMin;
            double QMax = usePerUnit ? info.QMax / SBase : info.QMax;

            ublas::vector<Complex> VVec(phases.size(), V);
            ublas::vector<Complex> SLoadVec(phases.size(), SLoad);
            ublas::vector<Complex> SgVec(phases.size(), Sg);
            ublas::vector<Complex> ysVec(phases.size(), ys);
            ublas::vector<double> QMinVec(phases.size(), QMin);
            ublas::vector<double> QMaxVec(phases.size(), QMax);

            Bus & bus = mod.newComponent<Bus>(busName(networkName, busId), type, phases, VVec, VVec, SgVec);
            bus.QMin() = QMinVec; // For PV busses, place bound on Q.
            bus.QMax() = QMaxVec; // For PV busses, place bound on Q.

            ZipToGround & zip = mod.newComponent<ZipToGround>(zipName(networkName, busId), phases);
            zip.S() = SLoadVec;
            zip.Y() = ysVec;
            bus.addZipToGround(zip);

            netw.addBus(bus);
         }

         int i = 0;
         for (const CdfBranchInfo & info : branchList)
         {
            const CdfBusInfo & busInfo0 = busMap[info.bus0Id];
            const CdfBusInfo & busInfo1 = busMap[info.bus1Id];

            double VBase0 = busInfo0.VBase;
            double VBase1 = busInfo1.VBase;
            double Rs = usePerUnit ? info.RsPu : info.RsPu * SBase / (VBase0 * VBase1);
            double Xs = usePerUnit ? info.XsPu : info.XsPu * SBase / (VBase0 * VBase1);
            double Bc = usePerUnit ? info.BcPu : info.BcPu * SBase / (VBase0 * VBase1);

            Complex cTap = polar(info.tap, info.thetaRad);

            Bus * bus0 = mod.componentNamed<Bus>(busName(networkName, info.bus0Id));
            if (bus0 == nullptr)
            {
               error() << "CDF: for branch, from bus " << info.bus0Id << " was not found." << std::endl;
               abort();
            }
            Bus * bus1 = mod.componentNamed<Bus>(busName(networkName, info.bus1Id));
            if (bus1 == nullptr)
            {
               error() << "CDF: for branch, to bus " << info.bus0Id << " was not found." << std::endl;
               abort();
            }

            Complex ys = 1.0 / Complex{Rs, Xs};

            Complex Y11 = (ys + Complex{0, 0.5 * Bc});
            Complex Y00 = Y11 / (info.tap * info.tap);
            Complex Y01 = -(ys / conj(cTap));
            Complex Y10 = -(ys / cTap);

            Branch & branch = mod.newComponent<Branch>(branchName(networkName, i, info.bus0Id, info.bus1Id), 
                              phases, phases);

            branch.setBus0(*bus0);
            branch.setBus1(*bus1);

            for (int k = 0; k < phases.size(); ++k)
            {
               branch.Y()(k, k) = Y00;
               branch.Y()(k + phases.size(), k + phases.size()) = Y11;
               branch.Y()(k, k + phases.size()) = Y01;
               branch.Y()(k + phases.size(), k) = Y10;
            }

            netw.addBranch(branch);
            ++i;
         }
      }
   }
}