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

// specific headers
#include "rat/common/error.hh"
#include "soleno.hh"

// cross check between soleno and old field-soleno

// main
int main(){
	// geometry
	const rat::fltp Rin = 0.1;
	const rat::fltp Rout = 0.11;
	const rat::fltp height = 0.1;
	const rat::fltp J = 400e6;
	const arma::uword num_layer = 5;
	const arma::uword num_turns = 1000;
	const rat::fltp tol = 1e-4;

	// setup soleno
	rat::fmm::ShSolenoPr sol = rat::fmm::Soleno::create();
	sol->add_solenoid(Rin,Rout,-height/2,height/2,J*(Rout-Rin)*height/num_turns,num_turns,num_layer);

	// field in middle
	const arma::Col<rat::fltp>::fixed<3> Rt1 = {0,0,0};
	const arma::Col<rat::fltp>::fixed<3> Bref1 = {0,0,2.162044564325736f};
	const arma::Col<rat::fltp> B1 = sol->calc_B(Rt1);

	std::cout<<Bref1<<" "<<B1<<std::endl;

	// check
	if(!arma::all(arma::abs(B1-Bref1)/arma::norm(Bref1)<tol))rat_throw_line(
		"magnetic field at center deviates from reference value");
	
	// field on inside edge of magnet
	const arma::Col<rat::fltp> Rt2 = {0.1,0,0};
	const arma::Col<rat::fltp> Bref2 = {0,0,3.444125885353538f};
	const arma::Col<rat::fltp> B2 = sol->calc_B(Rt2);

	// check
	if(!arma::all(arma::abs(B2-Bref2)/arma::norm(Bref2)<tol))rat_throw_line(
		"magnetic field at inner radius deviates from reference value");

	// field on inside corner of magnet
	const arma::Col<rat::fltp> Rt3 = {0.1,0,0.05};
	const arma::Col<rat::fltp> Bref3 = {2.436076740012795f,0,2.021404683125279f};
	const arma::Col<rat::fltp> B3 = sol->calc_B(Rt3);

	// check
	if(!arma::all(arma::abs(B3-Bref3)/arma::norm(Bref3)<tol))rat_throw_line(
		"magnetic field at inner corner deviates from reference value");

	// field on outside corner of magnet
	const arma::Col<rat::fltp> Rt4 = {0.11,0,0.05};
	const arma::Col<rat::fltp> Bref4 = {2.355577963021144f,0,-0.444462255430037f};
	const arma::Col<rat::fltp> B4 = sol->calc_B(Rt4);

	// check
	if(!arma::all(arma::abs(B4-Bref4)/arma::norm(Bref4)<tol))rat_throw_line(
		"magnetic field at outside corner deviates from reference value");

	// field on outside edge of magnet
	const arma::Col<rat::fltp> Rt5 = {0.11,0,0};
	const arma::Col<rat::fltp> Bref5 = {0,0,-1.317103956518485f};
	const arma::Col<rat::fltp> B5 = sol->calc_B(Rt5);

	// check
	if(!arma::all(arma::abs(B5-Bref5)/arma::norm(Bref5)<tol))rat_throw_line(
		"magnetic field at outer radius deviates from reference value");

	// calculate inductance
	const arma::Mat<rat::fltp> M = sol->calc_M();
	const arma::Mat<rat::fltp> Mref = {0.210800082851227f};

	// check
	if(!arma::all(arma::all(arma::abs(M-Mref)/Mref<tol)))rat_throw_line(
		"inductance deviates from reference value");

	// return normally
	return 0;
}
  