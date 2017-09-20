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

#include "DcPowerSource.h"
#include "Inverter.h"

#include <SgtCore/Zip.h>

using namespace arma;

namespace Sgt
{
    void InverterAbc::addDcPowerSource(const ConstSimComponentPtr<DcPowerSourceAbc>& source)
    {
        sources_.push_back(source);
        dependsOn(source, true);
    }

    double InverterAbc::availableP() const
    {
        double PDcA = PDc();
        return PDcA >= 0 ? PDcA * efficiency(PDcA) : PDcA / efficiency(PDcA);
    }
        
    void Inverter::addDcPowerSource(const ConstSimComponentPtr<DcPowerSourceAbc>& source)
    {
        InverterAbc::addDcPowerSource(source);
    }
    
    void Inverter::updateState(const Time& t)
    {
        zip()->setSConst(SConst());
    }

    Mat<Complex> Inverter::SConst() const
    {
        uword nPhase = zip()->phases().size();
        double PPerPh = availableP() / nPhase;
        double P2PerPh = PPerPh * PPerPh;
        double reqQPerPh = requestedQ_ / nPhase;
        double reqQ2PerPh = reqQPerPh * reqQPerPh;
        double maxSMagPerPh =  maxSMag_ / nPhase;
        double maxSMag2PerPh = maxSMagPerPh * maxSMagPerPh;
        double SMag2PerPh = std::min(P2PerPh + reqQ2PerPh, maxSMag2PerPh);
        double QPerPh = sqrt(SMag2PerPh - P2PerPh);
        if (requestedQ_ < 0.0)
        {
            QPerPh *= -1;
        }
        Complex SLoadPerPh{-PPerPh, -QPerPh}; // Load = -ve gen.
        return diagmat(Col<Complex>(nPhase, fill::none).fill(SLoadPerPh));
    }
}
