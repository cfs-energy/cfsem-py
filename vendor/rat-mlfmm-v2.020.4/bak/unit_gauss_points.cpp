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

#include "common/gauss.hh"

// main
int main(){
	// settings
	arma::uword max_N = 6;

	// calculate gauss points
	for(arma::uword i=1;i<=max_N;i++){
		// make gauss point calculator
		Gauss gp(i);

		// extract abscissae and weights
		arma::Row<double> x = gp.get_abscissae(); 
		arma::Row<double> w = gp.get_weights();

		// allocate cross checking table
		arma::Row<double> xexpect;
		arma::Row<double> wexpect;

		// check table
		// https://pomax.github.io/bezierinfo/legendre-gauss.html
		if(i==2){
			xexpect = {-0.5773502691896257,0.5773502691896257};
			wexpect = {1,1};
		}
		if(i==3){
			xexpect = {-0.7745966692414834,0,0.7745966692414834};
			wexpect = {0.5555555555555556,0.8888888888888888,0.5555555555555556};
		}
		if(i==4){
			xexpect = {-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526};
			wexpect = {0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538};
		}
		if(i==5){
			xexpect = {-0.9061798459386640,-0.5384693101056831,0,0.5384693101056831,0.9061798459386640};
			wexpect = {0.2369268850561891,0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891};
		}
		if(i==6){
			xexpect = {-0.9324695142031521,-0.6612093864662645,-0.2386191860831969,0.2386191860831969,0.6612093864662645,0.9324695142031521};
			wexpect = {0.1713244923791704,0.3607615730481386,0.4679139345726910,0.4679139345726910,0.3607615730481386,0.1713244923791704};
		}

		// check results
		if(!xexpect.is_empty()){
			assert(arma::all(arma::abs(x-xexpect)<1e-7));
			assert(arma::all(arma::abs(w-wexpect)<1e-7));
		}
	}

	// return
	return 0;
}