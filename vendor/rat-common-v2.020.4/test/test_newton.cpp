// Copyright 2025 Jeroen van Nugteren

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// general headers
#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

// specific headers
#include "error.hh"
#include "newtonraphson.hh"

// main
int main(){
	// settings
	const arma::uword num_equations = 10;

	// create a log
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();

	// create solver object
	rat::cmn::ShNewtonRaphsonPr mysolver = rat::cmn::NewtonRaphson::create();
	mysolver->set_tolfun(RAT_CONST(1e-8));

	// system function
	rat::cmn::NRSysFun sysfun = [&](const arma::Col<rat::fltp> &x){
		return (x%arma::flipud(x) + x - RAT_CONST(1.0)).eval();
	};

	// set objective function
	mysolver->set_systemfun(sysfun);
	mysolver->set_finite_difference();
	mysolver->set_use_central_diff(true);

	// initial guess
	const arma::Col<rat::fltp> x0(num_equations,arma::fill::zeros);
	mysolver->set_initial(x0);

	// solve
	mysolver->solve(lg);

	// get x
	const arma::Col<rat::fltp> x = mysolver->get_result();
	const arma::Col<rat::fltp> res = arma::abs(sysfun(x));

	// check result
	if(arma::any(res>1e-5))rat_throw_line("tolerance not achieved");
}