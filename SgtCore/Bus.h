#ifndef BUS_DOT_H
#define BUS_DOT_H

#include <SgtCore/Common.h>
#include <SgtCore/Component.h>
#include <SgtCore/Event.h>
#include <SgtCore/PowerFlow.h>

#include<iostream>
#include<map>
#include<vector>

namespace Sgt
{
   class GenAbc;
   class ZipAbc;

   /// @brief A Bus is a grouped set of conductors / terminals, one per phase.
   class Bus : public Component, public HasProperties<Bus>
   {
      friend class Network;

      public:

         typedef std::shared_ptr<GenAbc> GenPtr;
         typedef std::vector<GenPtr> GenVec;

         typedef std::shared_ptr<ZipAbc> ZipPtr;
         typedef std::vector<ZipPtr> ZipVec;

         SGT_PROPS_INIT(Bus);
         SGT_PROPS_INHERIT(Bus, Component);

      /// @name Static member functions:
      /// @{
         
         static const std::string& sComponentType()
         {
            static std::string result("bus");
            return result;
         }
      
      /// @}

      /// @name Lifecycle:
      /// @{

         Bus(const std::string& id, const Phases& phases, const arma::Col<Complex>& VNom, double VBase);

      /// @}

      /// @name ComponentInterface virtual overridden functions.
      /// @{
        
         virtual const std::string& componentType() const override
         {
            return sComponentType();
         }

         virtual void print(std::ostream& os) const override;

      /// @}

      /// @name Basic identity and type:
      /// @{

         virtual const Phases& phases() const
         {
            return phases_;
         }

         SGT_PROP_GET(phases, Bus, Phases, GetByConstRef, phases);

         virtual arma::Col<Complex> VNom() const
         {
            return VNom_;
         }

         SGT_PROP_GET(VNom, Bus, arma::Col<Complex>, GetByVal, VNom);

         virtual double VBase() const
         {
            return VBase_;
         }

         SGT_PROP_GET(VBase, Bus, double, GetByVal, VBase);

      /// @}

      /// @name Generators
      /// @{
         
         const GenVec gens() const {return genVec_;}
         void addGen(GenPtr gen) {genVec_.push_back(gen);}
         void removeAllGens() {genVec_.clear();}
         int nInServiceGens() const;
         arma::Col<Complex> SGenRequested() const; ///< Requested power of gens.
         arma::Col<Complex> SGen() const;  ///< Actual power of gens, including possible unserved generation.
         double JGen() const;

      /// @}

      /// @name Zips
      /// @{
         
         const ZipVec zips() const {return zipVec_;}
         void addZip(ZipPtr zip) {zipVec_.push_back(zip);}
         void removeAllZips() {zipVec_.clear();}
         int nInServiceZips() const;

         arma::Col<Complex> YConst() const; ///< Constant impedance component of zip.
         arma::Col<Complex> SYConst() const; ///< Complex power from constant impedance component.

         arma::Col<Complex> IConst() const; ///< Constant current component of zip.
         arma::Col<Complex> SIConst() const; ///< Complex power from constant current component of zip.

         arma::Col<Complex> SConst() const; ///< Complex power component of zip.
         
         arma::Col<Complex> SZipRequested() const; ///< Requested power of zips.
         arma::Col<Complex> SZip() const; ///< Actual power of zips, including possible unserved load.

      /// @}
      
      /// @name Control and limits:
      /// @{

         virtual BusType type() const
         {
            return type_;
         }

         virtual void setType(const BusType type)
         {
            type_ = type;
            setpointChanged_.trigger();
         }

         SGT_PROP_GET_SET(type, Bus, BusType, GetByVal, type, setType);

         virtual arma::Col<double> VMagSetpoint() const
         {
            return VMagSetpoint_;
         }

         virtual void setVMagSetpoint(const arma::Col<double>& VMagSetpoint)
         {
            VMagSetpoint_ = VMagSetpoint;
            setpointChanged_.trigger();
         }
         
         virtual arma::Col<double> VAngSetpoint() const
         {
            return VAngSetpoint_;
         }

         virtual void setVAngSetpoint(const arma::Col<double>& VAngSetpoint)
         {
            VAngSetpoint_ = VAngSetpoint;
            setpointChanged_.trigger();
         }

         virtual void applyVSetpoints();

         virtual double VMagMin() const
         {
            return VMagMin_;
         }

         virtual void setVMagMin(double VMagMin)
         {
            VMagMin_ = VMagMin;
            setpointChanged_.trigger();
         }

         virtual double VMagMax() const
         {
            return VMagMax_;
         }

         virtual void setVMagMax(double VMagMax)
         {
            VMagMax_ = VMagMax;
            setpointChanged_.trigger();
         }

      /// @}

      /// @name State
      /// @{

         virtual bool isInService() const
         {
            return isInService_;
         }

         virtual void setIsInService(bool isInService)
         {
            isInService_ = isInService;
            isInServiceChanged_.trigger();
         }

         SGT_PROP_GET_SET(isInService, Bus, bool, GetByVal, isInService, setIsInService);

         virtual const arma::Col<Complex>& V() const
         {
            return V_;
         }

         virtual void setV(const arma::Col<Complex>& V)
         {
            V_ = V;
            voltageUpdated_.trigger();
         }

         virtual const arma::Col<Complex>& SGenUnserved() const
         {
            return SGenUnserved_;
         }

         virtual void setSGenUnserved(const arma::Col<Complex>& SGenUnserved)
         {
            SGenUnserved_ = SGenUnserved;
         }

         virtual const arma::Col<Complex>& SZipUnserved() const
         {
            return SZipUnserved_;
         }

         virtual void setSZipUnserved(const arma::Col<Complex>& SZipUnserved)
         {
            SZipUnserved_ = SZipUnserved;
         }

      /// @}
      
      /// @name Events.
      /// @{
         
         /// @brief Event triggered when I go in or out of service.
         virtual Event& isInServiceChanged()
         {
            return isInServiceChanged_;
         }

         /// @brief Event triggered when bus setpoint has changed.
         virtual Event& setpointChanged()
         {
            return setpointChanged_;
         }
         
         /// @brief Event triggered when bus state (e.g. voltage) has been updated.
         virtual Event& voltageUpdated()
         {
            return voltageUpdated_;
         }
      
      /// @}
     
      private:

         Phases phases_{Phase::BAL};
         arma::Col<Complex> VNom_;
         double VBase_{1.0};

         GenVec genVec_;

         ZipVec zipVec_;

         BusType type_{BusType::NA};
         arma::Col<double> VMagSetpoint_;
         arma::Col<double> VAngSetpoint_;
         double VMagMin_{-infinity};
         double VMagMax_{infinity};

         bool isInService_{true};

         arma::Col<Complex> V_;
         arma::Col<Complex> SGenUnserved_; // SGen + SGenUnserved = SGenRequested
         arma::Col<Complex> SZipUnserved_; // SZip + SZipUnserved = SZipRequested

         Event isInServiceChanged_{std::string(sComponentType()) + " : Is in service changed"};
         Event setpointChanged_{std::string(sComponentType()) + " : Setpoint changed"};
         Event voltageUpdated_{std::string(sComponentType()) + " : Voltage updated"};
   };
}

#endif // BUS_DOT_H
