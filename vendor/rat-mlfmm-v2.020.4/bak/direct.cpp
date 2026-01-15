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
#include "direct.hh"

// constructor
Direct::Direct(){

}

// factory
ShDirectPr Direct::create(){
	//return ShDirectPr(new Direct);
	return std::make_shared<Direct>();
}

// set settings pointer
void Direct::set_settings(ShSettingsPr stngs){
	assert(stngs!=NULL);
	stngs_ = stngs;
}
	
// calculate magnetic field
void Direct::calculate(){
	// check if sources and targets set
	assert(!src_.is_empty());
	assert(!tar_.is_empty());
	assert(stngs_!=NULL);

	// walk over targets and allocate
	for(arma::uword i=0;i<tar_.n_elem;i++){
		tar_(i)->allocate();
	}

	// walk over all sources
	for(arma::uword i=0;i<src_.n_elem;i++){
		// walk over all targets
		for(arma::uword j=0;j<tar_.n_elem;j++){
			// perform calculation
			src_(i)->calc_direct(tar_(j));
		}
	}	
}





