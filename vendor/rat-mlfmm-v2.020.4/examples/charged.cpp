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

// solver
#include "chargedpolyhedron.hh"

// main
int main(){
	//Define parameters (modify as needed)
	const rat::fltp a = 0.01;      // [m] extent in the x-direction
	const rat::fltp b = 0.02;      // [m] extent in the y-direction
	const rat::fltp c = 0.005;      // [m] observation height
	const rat::fltp sigma0 = 1.0;      // source parameter sigma0
	const rat::fltp sigma1 = 0.0;      // source parameter sigma1
	const rat::fltp sigma2 = 0.0;      // source parameter sigma2

	std::cout<<(1.0/(4*arma::Datum<rat::fltp>::pi*arma::Datum<rat::fltp>::mu_0))*rat::fmm::ChargedPolyhedron::right_triangle_magnetic_field({a},{b},{c})<<std::endl;
	
}
