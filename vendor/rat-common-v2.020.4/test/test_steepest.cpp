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
#include "log.hh"
#include "steepest.hh"

// main
int main(){
	// settings
	const rat::fltp tol = RAT_CONST(1e-3);

	// create log
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();

	// create solver object
	rat::cmn::ShSteepestPr myoptim = rat::cmn::Steepest::create();
	//mysolver->set_tolfun(1e-8);
	myoptim->set_gamma(RAT_CONST(0.2));
	myoptim->set_use_central_diff(true);
	myoptim->set_use_linesearch(true);

	// system function
	rat::cmn::StSysFun sysfun = [&](const arma::Col<rat::fltp> &x){
		return arma::sum(RAT_CONST(0.1)*x%x%x%x + 2*x%x + 2);
	};

	// jacobian function
	rat::cmn::StGradFun gradfun = [&](const arma::Col<rat::fltp> &x){
		return (0.4*x%x%x + 4*x).eval();
	};

	// set objective function
	myoptim->set_systemfun(sysfun);
	//myoptim->set_finite_difference();
	myoptim->set_gradientfun(gradfun);

	// initial guess
	const arma::Col<rat::fltp> x0 = {
		RAT_CONST(-2.65),RAT_CONST(1.25),
		RAT_CONST(-3.1),RAT_CONST(4.2)};
	myoptim->set_initial(x0);

	// solve
	myoptim->optimise(lg);

	// check tolerance
	const arma::Col<rat::fltp> x = myoptim->get_result();
	if(arma::any(arma::abs(x)>tol))
		rat_throw_line("tolerance not achieved");
}