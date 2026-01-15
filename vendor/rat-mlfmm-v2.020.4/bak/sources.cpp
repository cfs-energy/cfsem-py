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

	arma::Mat<double> Sources::get_source_coords() const{return arma::Col<double>(3,0);}
	arma::Mat<double> Sources::get_source_coords(const arma::Row<arma::uword> &/*indices*/) const{return arma::Col<double>(3,0);}
	void Sources::calc_direct(ShTargetsPr &/*tar*/) const{}
	void Sources::source_to_target(ShTargetsPr &/*tar*/, const arma::Col<arma::uword> &/*target_list*/, const arma::field<arma::Col<arma::uword> > &/*source_list*/, const arma::Row<arma::uword> &/*first_source*/, const arma::Row<arma::uword> &/*last_source*/, const arma::Row<arma::uword> &/*first_target*/, const arma::Row<arma::uword> &/*last_target*/) const{}
	void Sources::sort(const arma::Row<arma::uword> &/*sort_idx*/){}
	void Sources::unsort(const arma::Row<arma::uword> &/*sort_idx*/){}
	void Sources::setup_source_to_multipole(const arma::Mat<double> &/*dR*/, const arma::uword /*num_exp*/){}
	void Sources::source_to_multipole(arma::Mat<std::complex<double> > &/*Mp*/, const arma::Row<arma::uword> &/*first_source*/, const arma::Row<arma::uword> &/*last_source*/, const arma::uword /*num_exp*/) const{}
	arma::uword Sources::get_num_dim() const{return 3;}
	arma::uword Sources::num_sources() const{return 0;}

}}