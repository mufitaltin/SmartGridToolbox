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

#include "KluSolver.h"

#include <klu.h>

#include <armadillo>

using namespace arma;

bool kluSolve(const arma::SpMat<double>& a, const arma::Col<double>& b, arma::Col<double>& result)
{
    auto n = b.size();
    auto nnz = a.n_nonzero;

    int* ap = new int[n + 1]; for (size_t i = 0; i <= n; ++i) ap[i] = static_cast<int>(nnz);
    // TODO: Normally all but last element will get overwritten - but if not, is nnz OK / correct?
    int* ai = new int[nnz];
    double* ax = new double[nnz];
    uword prevCol = n; // Force first element to be added to ap.
    uword nCol = 0; // Number of entries in ap.
    uword iNz = 0;
    for (auto x = a.begin(); x != a.end(); ++x, ++iNz)
    {
        ax[iNz] = *x;
        ai[iNz] = static_cast<int>(x.row());
        if (x.col() != prevCol)
        {
            prevCol = x.col();
            ap[nCol] = static_cast<int>(iNz);
            nCol += 1;
        }
    }

    double* b1 = new double[n];
    for (unsigned int i = 0; i < n; ++i) b1[i] = b(i);

    klu_symbolic *Symbolic;
    klu_numeric *Numeric;
    klu_common Common;

    klu_defaults (&Common);
    Symbolic = klu_analyze (static_cast<int>(n), ap, ai, &Common);
    Numeric = klu_factor (ap, ai, ax, Symbolic, &Common);
    bool ok = klu_solve(Symbolic, Numeric, static_cast<int>(n), 1, b1, &Common) == 1;

    if (!ok)
    {
        std::cerr << "KLU failed." << std::endl;
        std::cerr << "Status = " << Common.status << std::endl;
    }

    klu_free_symbolic (&Symbolic, &Common);
    klu_free_numeric (&Numeric, &Common);

    result = arma::Col<double>(n, fill::none);
    for (arma::uword i = 0; i < n; ++i)
    {
        result(i) = b1[i];
    }

    delete[] ap;
    delete[] ai;
    delete[] ax;
    delete[] b1;

    return ok;
}
