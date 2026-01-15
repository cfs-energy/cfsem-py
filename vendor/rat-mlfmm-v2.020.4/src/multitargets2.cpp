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
#include "multitargets2.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	MultiTargets2::MultiTargets2(){

	}
	
	// constructor
	MultiTargets2::MultiTargets2(const ShTargetsPrList& targets){
		tars_ = targets;
	}

	// factory
	ShMultiTargets2Pr MultiTargets2::create(){
		return std::make_shared<MultiTargets2>();
	}

	// factory
	ShMultiTargets2Pr MultiTargets2::create(const ShTargetsPrList& targets){
		return std::make_shared<MultiTargets2>(targets);
	}

	// add targets
	void MultiTargets2::add_targets(const ShTargetsPr& tar){
		// check input
		if(tar==NULL)rat_throw_line("model points to zero");

		// get number of models
		const arma::uword num_targets = tars_.n_elem;

		// allocate new target list
		ShTargetsPrList new_tars(num_targets + 1);

		// set old and new targets
		for(arma::uword i=0;i<num_targets;i++)new_tars(i) = tars_(i);
		new_tars(num_targets) = tar;
		
		// set new target list
		tars_ = new_tars;
	}

	// setup function
	void MultiTargets2::setup_targets(){
		// get counters
		const arma::uword num_target_objects = tars_.n_elem;

		// in case of no target objects
		if(num_target_objects==0)return;

		// setup the children
		for(arma::uword i=0;i<num_target_objects;i++)
			tars_(i)->setup_targets();

		// allocate number of targets
		arma::Row<arma::uword> num_targets(num_target_objects);

		// walk over target objects
		for(arma::uword i=0;i<num_target_objects;i++)
			num_targets(i) = tars_(i)->num_targets();
		
		// calculate indexes
		target_index_.set_size(num_target_objects+1); target_index_(0) = 0; 
		target_index_.cols(1,num_target_objects) = arma::cumsum<arma::Row<arma::uword> >(num_targets);

		// get number of targets
		num_targets_ = target_index_.back();

		// allocate coordinates
		Rt_.set_size(3,num_targets_);
		
		// walk over target objects and collect coordinates
		for(arma::uword i=0;i<tars_.n_elem;i++){
			if(num_targets(i)>0){
				Rt_.cols(target_index_(i),target_index_(i+1)-1) = tars_(i)->get_target_coords();
			}
		}

		// mirror field types
		field_type_.clear();
		for(arma::uword i=0;i<tars_.n_elem;i++){if(tars_(i)->has('A')){add_field_type('A',3); break;}}
		for(arma::uword i=0;i<tars_.n_elem;i++){if(tars_(i)->has('H')){add_field_type('H',3); break;}}
		for(arma::uword i=0;i<tars_.n_elem;i++){if(tars_(i)->has('B')){add_field_type('B',3); break;}}
		for(arma::uword i=0;i<tars_.n_elem;i++){if(tars_(i)->has('M')){add_field_type('M',3); break;}}
	}

	// get number of target objects
	arma::uword MultiTargets2::get_num_target_objects() const{
		return tars_.n_elem;
	}

	// post processing
	void MultiTargets2::post_process(){
		// // allocate children
		// for(arma::uword i=0;i<tars_.n_elem;i++)tars_(i)->allocate();

		// walk over target objects and collect coordinates
		for(auto it = field_type_.begin();it!=field_type_.end();it++){
			char ft = (*it).first;
			for(arma::uword j=0;j<tars_.n_elem;j++){
				if(tars_(j)->has(ft)){
					tars_(j)->add_field(ft, M_[ft].cols(target_index_(j),target_index_(j+1)-1), false);
				}
			}
		}

		// post process children
		for(arma::uword i=0;i<tars_.n_elem;i++)
			tars_(i)->post_process();
	}

	// allocate
	void MultiTargets2::allocate(){
		// allocate self
		MgnTargets::allocate();

		// allocate children
		for(arma::uword i=0;i<tars_.n_elem;i++)tars_(i)->allocate();
	}


}}