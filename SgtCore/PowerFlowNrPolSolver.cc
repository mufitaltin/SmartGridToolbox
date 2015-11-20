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

#include "config.h"
#include "Network.h"
#include "PowerFlowNrPolSolver.h"
#include "SparseHelper.h"
#ifdef WITH_KLU
#include "KluSolver.h"
#endif

#include <algorithm>
#include <ostream>
#include <sstream>

using namespace Sgt;
using namespace arma;

namespace Sgt
{
    namespace
    {
        Col<Complex> calcS(
                const Col<Complex>& Scg,
                const Col<Complex>& Ic,
                const Col<Complex>& V,
                const Col<double>& M,
                const SpMat<Complex>& Y)
        {
            return (Scg + conj(Ic) % M - V % conj(Y * V));
        }

        void updateQcgPv(
                Col<Complex>& Scg,
                arma::span selPv,
                const Col<Complex>& Ic,
                const Col<Complex>& V,
                const Col<double>& M,
                const SpMat<Complex>& Y)
        {
            // TODO: may unnessecarily calculate P.
            // Note: set_imag doesn't work for subview.
            Col<double> PSv = real(Scg(selPv)); // Temporary works.
            Col<double> QNew = imag(Ic(selPv)) % M(selPv) + imag(V(selPv) % conj(Y.rows(selPv.a, selPv.b) * V)); 
            Scg(selPv) = cx_mat(PSv, QNew); 
        }

        void updateScgSl(
                Col<Complex>& Scg,
                arma::span selSl,
                const Col<Complex>& Ic,
                const Col<Complex>& V,
                const Col<double>& M,
                const SpMat<Complex>& Y)
        {
            // TODO: may unnessecarily calculate P.
            // Note: set_imag doesn't work for subview.
            Scg(selSl) = Ic(selSl) % M(selSl) + V(selSl) % conj(Y.rows(selSl.a, selSl.b) * V);
        }

        template<typename T> SpMat<typename T::elem_type> spDiag(const T& v)
        {
            uword n = v.size();
            Mat<uword> locs(2, n, arma::fill::none);
            locs.row(0) = linspace<Row<uword>>(0, n - 1, n);
            locs.row(1) = locs.row(0);
            return SpMat<typename T::elem_type>(locs, v, false);
        }

        // Used for the Jacobian. See Matpower's dSbus_dV.
        // Note however that we're using a negative definition for S compared to matpower.
        // First returned element is dS_dVm, second is dS_dVa. 
        std::pair<SpMat<Complex>, SpMat<Complex>> dSdV(const SpMat<Complex>& Y, const Col<Complex>& V)
        {
            Col<Complex> I = Y * V;
            SpMat<Complex> diagI = spDiag(I);
            SpMat<Complex> diagV = spDiag(V);
            Col<Complex> VNorm = V / abs(V);
            SpMat<Complex> diagVNorm = spDiag(VNorm);
            return {
                -(diagV * conj(Y * diagVNorm) + conj(diagI) * diagVNorm),
                -Complex{0, 1} * diagV * conj(diagI - Y * diagV)
            };
        }
        
        SpMat<double> calcJ(
                const uword nPq,
                const uword nPv,
                const Col<Complex>& V,
                const SpMat<Complex>& Y)
        {
            uword nPqPv = nPq + nPv;

            auto iPq = [&](uword i){return i;};
            auto iPV = [&](uword i){return i + nPq;};

            auto allPq = span(iPq(0), iPq(nPq - 1));
            auto allPqPV = span(iPq(0), iPV(nPv - 1));

            uword szJ = 2 * nPq + nPv;

            auto dSdV_ = dSdV(Y, V);
            const auto& dSdM = dSdV_.first;
            const auto& dSdT = dSdV_.second;

            SpMat<double> dPdM = real(dSdM.submat(allPqPV, allPq));
            SpMat<double> dQdM = imag(dSdM.submat(allPq, allPq));
            SpMat<double> dPdT = real(dSdT.submat(allPqPV, allPqPV));
            SpMat<double> dQdT = imag(dSdT.submat(allPq, allPqPV));
            
            SparseHelper<double> h(szJ, szJ);
            for (auto it = dPdM.begin(); it != dPdM.end(); ++it)
            {
                uword i = it.row(); uword k = it.col(); double val = *it;
                h.insert(i, k, val);
            }
            for (auto it = dQdM.begin(); it != dQdM.end(); ++it)
            {
                uword i = it.row(); uword k = it.col(); double val = *it;
                h.insert(i + nPqPv, k, val);
            }
            for (auto it = dPdT.begin(); it != dPdT.end(); ++it)
            {
                uword i = it.row(); uword k = it.col(); double val = *it;
                h.insert(i, k + nPq, val);
            }
            for (auto it = dQdT.begin(); it != dQdT.end(); ++it)
            {
                uword i = it.row(); uword k = it.col(); double val = *it;
                h.insert(i + nPqPv, k + nPq, val);
            }

            return h.get();
        }

        // Solve Jx + f = 0
        bool solveSparseSystem(const SpMat<double>& J, const Col<double>& f, Col<double>& x)
        {
            bool ok;
#ifdef WITH_KLU
            ok = kluSolve(J, -f, x);
#else
            ok = spsolve(x, J, -f, "superlu");
#endif
            return ok;
        }
    }

    bool PowerFlowNrPolSolver::solveProblem()
    {
        sgtLogDebug() << "PowerFlowNrPolSolver : solve." << std::endl;
        LogIndent indent;

        mod_ = buildModel(*netw_);
        
        // Set up data structures for the calculation.
        // Model indexing is [0 ... nPq - 1] = Pq, [nPq ... nPq + nPv - 1] = PV, [nPq + nPv ... nPq + nPv + nSl] = SL

        uword nNode = mod_->nNode();
        uword nPq = mod_->nPq();
        uword nPv = mod_->nPv();
        uword nSl = mod_->nSl();
        uword nPqPv = nPq + nPv;

        const SpMat<Complex>& Y = mod_->Y(); // Model indexing. Includes shunts (const Y in ZIPs).
        SpMat<double> G = real(Y); // Model indexing. Includes shunts (const Y in ZIPs).
        SpMat<double> B = imag(Y); // Model indexing. Includes shunts (const Y in ZIPs).

        Col<Complex> V = mod_->V(); // Model indexing.
        Col<double> M = abs(V); // Model indexing.

        Col<double> theta(nNode, fill::none); // Model indexing.
        for (uword i = 0; i < nNode; ++i) theta(i) = std::arg(mod_->V()(i));

        Col<Complex> Scg = mod_->S(); // Model indexing. S_cg = S_c + S_g.

        const Col<Complex>& Ic = mod_->IConst(); // Model indexing. P_c + P_g.

        bool wasSuccessful = false;
        double err = 0;
        unsigned int niter;
        bool ok;

        for (niter = 0; niter < maxiter_; ++niter)
        {
            sgtLogMessage(LogLevel::VERBOSE) << "iter " << niter << std::endl;
            LogIndent _;

            auto S = calcS(Scg, Ic, V, M, Y);
            arma::Col<double> f = join_vert(real(S.subvec(0, nPqPv - 1)), imag(S.subvec(0, nPq - 1)));
            
            err = norm(f, "inf");
            sgtLogMessage(LogLevel::VERBOSE) << "Err = " << err << std::endl;
            if (err <= tol_) 
            {
                wasSuccessful = true;
                break;
            }

            Col<double> x; // Delta theta.
            SpMat<double> J = calcJ(nPq, nPv, V, Y);

            ok = solveSparseSystem(J, f, x);
            if (!ok)
            {
                sgtLogWarning() << "Solve failed." << std::endl;
                break;
            }
            sgtLogDebug() << "--------------------" << std::endl;
            sgtLogDebug() << "V = " << V << std::endl;
            sgtLogDebug() << "S = " << S << std::endl;
            sgtLogDebug() << "f = " << f << std::endl;
            sgtLogDebug() << "|f| = " << err << std::endl;
            sgtLogDebug() << "x = " << x << std::endl;
            sgtLogDebug() << "J =\n" << std::setw(10) << arma::Mat<double>(J) << std::endl;
            sgtLogDebug() << "--------------------" << std::endl;

            // Update M, theta, V.
            M.subvec(0, nPq - 1) += x.subvec(0, nPq - 1);
            theta.subvec(0, nPqPv - 1) += x.subvec(nPq, 2 * nPq + nPv - 1);
            V.subvec(0, nPqPv - 1) = M.subvec(0, nPqPv - 1) % exp(Complex{0, 1} * theta.subvec(0, nPqPv - 1));
        }

        if (!wasSuccessful)
        {
            sgtLogWarning() << "PowerFlowFdSolver: failed to converge." << std::endl;
            for (std::size_t i = 0; i < mod_->nNode(); ++i)
            {
                // TODO: this should be part of PowerFlowModel.
                auto node = mod_->nodeVec()[i];
                node->V_ = 0;
                node->S_ = 0;
                node->bus_->V_[node->phaseIdx_] = node->V_;
                node->bus_->S_[node->phaseIdx_] = node->S_;
            }
        }

        if (nPv > 0)
        {
            updateQcgPv(Scg, mod_->selPv(), Ic, V, M, Y);
        }
        if (nSl > 0)
        {
            updateScgSl(Scg, mod_->selPq(), Ic, V, M, Y);
        }

        mod_->V() = V;
        mod_->S() = Scg;

        // Update nodes and busses.
        for (uword i = 0; i < mod_->nNode(); ++i)
        {
            auto node = mod_->nodeVec()[i];
            node->V_ = mod_->V()(i);
            node->S_ = mod_->S()(i);
            node->bus_->V_[node->phaseIdx_] = node->V_;
            node->bus_->S_[node->phaseIdx_] = node->S_;
        }

        return wasSuccessful;
    }
}