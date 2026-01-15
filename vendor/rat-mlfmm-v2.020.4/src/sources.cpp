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
#include "sources.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	Sources::Sources(){

	}

	// all source types should have these methods
	// in order to commmunicate with MLFMM
	arma::Mat<fltp> Sources::get_source_coords() const{
		return arma::Mat<fltp>(3,0);
	}

	// will always be called
	void Sources::calc_external(
		const ShTargetsPr &/*tar*/, 
		const ShSettingsPr &/*stngs*/,
		const cmn::ShLogPr &/*lg*/) const{

	}

	// field calculation from specific sources
	void Sources::calc_direct(
		const ShTargetsPr &/*tar*/, 
		const ShSettingsPr &/*stngs*/,
		const cmn::ShLogPr &/*lg*/) const{

	}

	// source to target
	void Sources::source_to_target(
		const ShTargetsPr &/*tar*/, 
		const arma::Col<arma::uword> &/*target_list*/, 
		const arma::field<arma::Col<arma::uword> > &/*source_list*/, 
		const arma::Row<arma::uword> &/*first_source*/, 
		const arma::Row<arma::uword> &/*last_source*/, 
		const arma::Row<arma::uword> &/*first_target*/, 
		const arma::Row<arma::uword> &/*last_target*/, 
		const ShSettingsPr &/*stngs*/,
		const cmn::ShLogPr &/*lg*/) const{

	} 

	// sort sources
	void Sources::sort_sources(
		const arma::Row<arma::uword> &/*sort_idx*/){

	}

	// unsort sources
	void Sources::unsort_sources(
		const arma::Row<arma::uword> &/*sort_idx*/){

	}

	// source to multipole step
	void Sources::setup_source_to_multipole(
		const arma::Mat<fltp> &/*dR*/,
		const ShSettingsPr &/*stngs*/){

	}

	// source to multipole
	void Sources::source_to_multipole(
		arma::Mat<std::complex<fltp> > &/*Mp*/, 
		const arma::Row<arma::uword> &/*first_source*/, 
		const arma::Row<arma::uword> &/*last_source*/, 
		const ShSettingsPr &/*stngs*/) const{

	}

	// setup sources
	void Sources::setup_sources(){

	}

	// get number of dimensions
	arma::uword Sources::get_num_dim() const{
		return 3;
	}

	// getting basic information
	arma::uword Sources::num_sources() const{
		return 0;
	}

	// get typical element size (to limit grid)
	fltp Sources::element_size() const{
		return 0;
	}

	ShSourcesPr Sources::subdivide(const fltp /*grid_size*/){
		return NULL;
	}

}}