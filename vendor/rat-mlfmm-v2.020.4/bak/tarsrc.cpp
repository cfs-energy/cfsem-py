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

#include "tarsrc.hh"

// code specific to Rat
namespace Rat{
	// set new sources
	void TarSrc::set_sources(ShSourcesPr src){
		// check input
		if(src==NULL)rat_throw_line("supplied source is null pointer");

		// reset and resize to one
		src_.set_size(1);

		// add source to list
		src_(0) = src;
	}

	// set new targets
	void TarSrc::set_targets(ShTargetsPr tar){
		// check input
		if(tar==NULL)rat_throw_line("supplied target is null pointer");

		// reset and resize to one
		tar_.set_size(1);

		// add target to list
		tar_(0) = tar;
	}

	// set new sources
	void TarSrc::set_sources(arma::field<ShSourcesPr> src){
		src_ = src;
	}

	// set new targets
	void TarSrc::set_targets(arma::field<ShTargetsPr> tar){
		tar_ = tar;
	}

	// add new sources
	void TarSrc::add_sources(ShSourcesPr src){
		// check input
		if(src==NULL)rat_throw_line("supplied source is null pointer");

		// allocate new source list
		arma::field<ShSourcesPr> new_src(src_.n_elem + 1);

		// set old and new sources
		for(arma::uword i=0;i<src_.n_elem;i++)new_src(i) = src_(i);
		new_src(src_.n_elem) = src;

		// set new source list
		src_ = new_src;
	}

	// add new targets
	void TarSrc::add_targets(ShTargetsPr tar){
		// check input
		if(tar==NULL)rat_throw_line("supplied target is null pointer");

		// allocate new source list
		arma::field<ShTargetsPr> new_tar(tar_.n_elem + 1);

		// set old and new sources
		for(arma::uword i=0;i<tar_.n_elem;i++)new_tar(i) = tar_(i);
		new_tar(tar_.n_elem) = tar;

		// set new source list
		tar_ = new_tar;
	}

	// add new sources
	void TarSrc::add_sources(arma::field<ShSourcesPr> src){
		// check input
		for(arma::uword i=0;i<src.n_elem;i++)if(src(i)==NULL)
			rat_throw_line("one source in the list is a null pointer");

		// allocate new source list
		arma::field<ShSourcesPr> new_src(src_.n_elem + src.n_elem);

		// set old and new sources
		for(arma::uword i=0;i<src_.n_elem;i++)new_src(i) = src_(i);
		for(arma::uword i=0;i<src.n_elem;i++)new_src(src_.n_elem+i) = src(i);

		// set new source list
		src_ = new_src;
	}

	// add new targets
	void TarSrc::add_targets(arma::field<ShTargetsPr> tar){
		// check input
		for(arma::uword i=0;i<tar.n_elem;i++)if(tar(i)==NULL)
			rat_throw_line("one target in the list is a null pointer");

		// allocate new source list
		arma::field<ShTargetsPr> new_tar(tar_.n_elem + tar.n_elem);

		// set old and new sources
		for(arma::uword i=0;i<tar_.n_elem;i++)new_tar(i) = tar_(i);
		for(arma::uword i=0;i<tar.n_elem;i++)new_tar(tar_.n_elem+i) = tar(i);

		// set new source list
		tar_ = new_tar;
	}

	// get sources
	arma::field<ShSourcesPr> TarSrc::get_sources() const{
		return src_;
	}

	// get sources
	arma::field<ShTargetsPr> TarSrc::get_targets() const{
		return tar_;
	}

	// get number of sources
	arma::uword TarSrc::num_source_objects() const{
		return src_.n_elem;
	}

	// get number of targets
	arma::uword TarSrc::num_target_objects() const{
		return tar_.n_elem;
	}
}