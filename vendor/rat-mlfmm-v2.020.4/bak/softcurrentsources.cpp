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

// include header file
#include "softcurrentsources.hh"

// constructor
SoftCurrentSources::SoftCurrentSources(){
	
}

// factory
ShSoftCurrentSourcesPr SoftCurrentSources::create(){
	//return ShSoftCurrentSourcesPr(new SoftCurrentSources);
	return std::make_shared<SoftCurrentSources>();
}

// setting of the softening factor
void SoftCurrentSources::set_softening(const arma::Row<double> &eps){
	eps_ = eps;
}

// calculation of vector potential for all sources
void SoftCurrentSources::calc_direct(ShTargetsPr &tar) const{
	// get target coordinates
	arma::Mat<double> Rt = tar->get_target_coords();

	// forward calculation of vector potential to extra
	if(tar->has("A")){
		tar->add_field("A", Savart::calc_I2A_s(
			Rs_, Ieff_, eps_, Rt, true), true);
	}

	// forward calculation of magnetic field to extra
	if(tar->has("H")){
		tar->add_field("H", Savart::calc_I2H_s(
			Rs_, Ieff_, eps_, Rt, true), true);
	}

}

// calculation of vector potential for specific sources
void SoftCurrentSources::calc_direct(
	ShTargetsPr &tar, const arma::Row<arma::uword> &tidx, 
	const arma::Row<arma::uword> &sidx) const{

	// get target coordinates
	arma::Mat<double> Rt = tar->get_target_coords(tidx);

	// forward calculation of vector potential to extra
	if(tar->has("A")){
		tar->add_field("A", tidx, Savart::calc_I2A_s(
			Rs_.cols(sidx), Ieff_.cols(sidx), eps_.cols(sidx), Rt, false), true);
	}

	// forward calculation of magnetic field to extra
	if(tar->has("H")){
		tar->add_field("H", tidx, Savart::calc_I2H_s(
			Rs_.cols(sidx), Ieff_.cols(sidx), eps_.cols(sidx), Rt, false), true);
	}
}