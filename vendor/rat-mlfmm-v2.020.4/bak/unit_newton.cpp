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

#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

#include "newtonraphson.hh"

// main
int main(){
	// create solver object
	ShNewtonRaphsonPr mysolver = NewtonRaphson::create();
	//mysolver->set_tolfun(1e-8);

	// system function
	NRSysFun sysfun = [&](const arma::Col<double> &x){
		// create output vector
		arma::Col<double> out = x%x%x + x%x + x + 1;
		return out;
	};

	// set objective function
	mysolver->set_systemfun(sysfun);
	mysolver->set_jacfun();
	
	// initial guess
	arma::Col<double> x0 = {4,2,3};
	mysolver->set_initial(x0);

	// solve
	mysolver->solve();

	// get x
	arma::Col<double> x = mysolver->get_result();
	arma::Col<double> res = arma::abs(sysfun(x));

	// display result
	std::cout<<mysolver->get_message()<<std::endl;
	std::cout<<std::endl;
	std::cout<<" solution: "<<x(0)<<" "<<x(1)<<" "<<x(2)<<std::endl;
	std::cout<<"residuals: "<<res(0)<<" "<<res(1)<<" "<<res(2)<<std::endl;

	// check result
	assert(arma::all(res<1e-5));
}