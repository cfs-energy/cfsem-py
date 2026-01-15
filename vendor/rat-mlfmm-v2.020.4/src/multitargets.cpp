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
#include "multitargets.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	MultiTargets::MultiTargets(){

	}
	
	// constructor
	MultiTargets::MultiTargets(const ShTargetsPrList &targets){
		tars_ = targets;
	}

	// factory
	ShMultiTargetsPr MultiTargets::create(){
		return std::make_shared<MultiTargets>();
	}

	// factory
	ShMultiTargetsPr MultiTargets::create(const ShTargetsPrList &targets){
		return std::make_shared<MultiTargets>(targets);
	}

	// add targets
	void MultiTargets::add_targets(const ShTargetsPr &tar){
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


	// get number of target objects
	arma::uword MultiTargets::get_num_target_objects() const{
		return tars_.n_elem;
	}


	// setup function
	void MultiTargets::setup_multi_targets(){
		// get counters
		const arma::uword num_target_objects = tars_.n_elem;

		// in case of no target objects
		if(num_target_objects==0)rat_throw_line("target objects are not set");

		// no need for indexing when only one target
		if(num_target_objects==1){
			total_num_targets_ = tars_(0)->num_targets();
			return;
		}

		// allocate number of targets
		num_targets_.set_size(num_target_objects);

		// walk over target objects
		for(arma::uword i=0;i<num_target_objects;i++)
			num_targets_(i) = tars_(i)->num_targets();
		
		// calculate indexes
		target_index_.set_size(num_target_objects+1); target_index_(0) = 0; 
		target_index_.cols(1,num_target_objects) = arma::cumsum<arma::Row<arma::uword> >(num_targets_);

		// total number of targets
		total_num_targets_ = target_index_(num_target_objects);

		// allocate index arrays
		original_index_.set_size(total_num_targets_);
		original_target_.set_size(total_num_targets_);

		// number indexes
		for(arma::uword i=0;i<num_target_objects;i++){
			original_index_.cols(target_index_(i),target_index_(i+1)-1) = 
				arma::regspace<arma::Row<arma::uword> >(0,num_targets_(i)-1);
			original_target_.cols(target_index_(i),target_index_(i+1)-1) = 
				i*arma::Row<arma::uword>(num_targets_(i),arma::fill::ones);
		}

	}

	// acecss sorting index
	arma::Row<arma::uword> MultiTargets::get_original_index()const{
		if(tars_.n_elem==1)return arma::regspace<arma::Row<arma::uword> >(0,total_num_targets_-1);
		return original_index_;
	}
	
	// access original source index
	arma::Row<arma::uword> MultiTargets::get_original_target()const{
		if(tars_.n_elem==1)return arma::Row<arma::uword>(total_num_targets_, arma::fill::zeros);
		return original_target_;
	}

	// get field function
	arma::Mat<fltp> MultiTargets::get_field(const char type) const{
		// single target case
		if(tars_.n_elem==1)return tars_(0)->get_field(type);
		
		// check if all targets have this type
		for(arma::uword i=0;i<tars_.n_elem;i++)
			if(!tars_(i)->has(type))rat_throw_line("target without type");

		// get data
		arma::field<arma::Mat<fltp> > field_data(tars_.n_elem);

		// walk over targets
		for(arma::uword i=0;i<tars_.n_elem;i++){
			field_data(i) = tars_(i)->get_field(type);
		}

		// combine and return
		return rat::cmn::Extra::field2mat(field_data);
	}

	// getting basic information
	bool MultiTargets::has(const char type) const{
		bool has_type = false;
		for(arma::uword i=0;i<tars_.n_elem;i++){
			if(tars_(i)->has(type)){
				has_type=true; break;
			}
		}
		return has_type;
	}

	// allocation of storage matrix
	void MultiTargets::allocate(){
		for(arma::uword i=0;i<tars_.n_elem;i++)tars_(i)->allocate();
	}

	// sort targets using sort index array
	void MultiTargets::sort_targets(const arma::Row<arma::uword> &sort_idx){
		// single target case
		if(tars_.n_elem==1){
			tars_(0)->sort_targets(sort_idx); return;
		}

		// sort indexes
		original_index_ = original_index_.cols(sort_idx);
		original_target_ = original_target_.cols(sort_idx);

		// walk over targets and sort individually
		for(arma::uword i=0;i<tars_.n_elem;i++){
			// find indexes of this target
			const arma::Row<arma::uword> idx_target = arma::find(original_target_==i).t();

			// create sorting index
			const arma::Row<arma::uword> local_sort_idx = original_index_.cols(idx_target);

			// sort at target level
			tars_(i)->sort_targets(local_sort_idx); 

			// unsort indexes at this multitargets level
			original_index_.cols(idx_target.cols(local_sort_idx)) = original_index_.cols(idx_target);
			//original_target_.cols(idx_target.cols(local_sort_idx)) = original_target_.cols(idx_target);
		}
	}

	// unsort targets using (un)sort index array
	void MultiTargets::unsort_targets(const arma::Row<arma::uword> &sort_idx){
		// single target case
		if(tars_.n_elem==1){
			tars_(0)->unsort_targets(sort_idx); return;
		}

		// translate sort index this is where the elements are originint within the target object
		original_index_.cols(sort_idx) = original_index_;
		original_target_.cols(sort_idx) = original_target_;

		// walk over targets and sort individually
		for(arma::uword i=0;i<tars_.n_elem;i++){
			// find indexes of this target
			const arma::Row<arma::uword> idx_target = arma::find(original_target_==i).t();

			// create sorting index
			const arma::Row<arma::uword> local_sort_idx = original_index_.cols(idx_target);

			// sort at target level
			tars_(i)->sort_targets(local_sort_idx); // on purpose

			// unsort at this multitargets level
			original_index_.cols(idx_target.cols(local_sort_idx)) = original_index_.cols(idx_target);
		}
	}

	// // get coordinates of all stored targets
	// arma::Mat<fltp> MultiTargets::get_target_coords() const{
	// 	// single target case
	// 	if(tars_.n_elem==1)return tars_(0)->get_target_coords();

	// 	// get counters
	// 	const arma::uword num_target_objects = tars_.n_elem;

	// 	// check if setup
	// 	if(original_index_.is_empty())rat_throw_line("multi targets not setup");
	// 	if(original_target_.is_empty())rat_throw_line("multi targets not setup");

	// 	// allocate coordinates
	// 	arma::Mat<fltp> Rt(3,total_num_targets_);

	// 	// calculate indexes
	// 	arma::Row<arma::uword> tar_indices(num_target_objects+1); tar_indices(0) = 0; 
	// 	tar_indices.cols(1,num_target_objects) = arma::cumsum<arma::Row<arma::uword> >(num_targets_);

	// 	// walk over target objects and collect coordinates
	// 	for(arma::uword i=0;i<num_target_objects;i++){
	// 		Rt.cols(tar_indices(i),tar_indices(i+1)-1) = tars_(i)->get_target_coords();
	// 	}

	// 	// return coordinates
	// 	return Rt;
	// }

	// get coordinates of all stored targets
	arma::Mat<fltp> MultiTargets::get_target_coords() const{
		// single target case
		if(tars_.n_elem==1)return tars_(0)->get_target_coords();
		
		// check if setup
		if(original_index_.is_empty())rat_throw_line("multi targets not setup");
		if(original_target_.is_empty())rat_throw_line("multi targets not setup");
		
		// allocate coordinates
		arma::Mat<fltp> Rt(3,total_num_targets_);
		
		// walk over target objects and collect coordinates
		for(arma::uword i=0;i<tars_.n_elem;i++){
			if(num_targets_(i)>0){
				Rt.cols(target_index_(i),target_index_(i+1)-1) = tars_(i)->get_target_coords();
			}
		}
		
		// sort and return
		return Rt.cols(original_index_ + target_index_.cols(original_target_));
	}


	// get coordinates with specific indexes
	arma::Mat<fltp> MultiTargets::get_target_coords(
		const arma::uword ft, const arma::uword lt) const{

		// single target case
		if(tars_.n_elem==1)return tars_(0)->get_target_coords(ft,lt);

		// check if setup
		if(original_index_.is_empty())rat_throw_line("multi targets not setup");
		if(original_target_.is_empty())rat_throw_line("multi targets not setup");

		// allocate coordinates
		arma::Mat<fltp> Rt(3,lt-ft+1);

		// collect targets one by one
		for(arma::uword i=0;i<lt-ft+1;i++){
			Rt.col(i) = tars_(original_target_(ft+i))->get_target_coords(original_index_(ft+i),original_index_(ft+i));
		}

		// return coordinates
		return Rt;
	}

	// // get coordinates with specific indexes
	// arma::Mat<fltp> MultiTargets::get_target_coords(
	// 	const arma::Row<arma::uword> &indices) const{
	// 	// single target case
	// 	if(tars_.n_elem==1)return tars_(0)->get_target_coords(indices);

	// 	// check if setup
	// 	if(original_index_.is_empty())rat_throw_line("multi targets not setup");
	// 	if(original_target_.is_empty())rat_throw_line("multi targets not setup");

	// 	// allocate coordinates
	// 	arma::Mat<fltp> Rt(3,indices.n_elem);

	// 	// get original objects and indexes
	// 	const arma::Row<arma::uword> element_original_index = 
	// 		original_index_.cols(indices);
	// 	const arma::Row<arma::uword> element_original_target = 
	// 		original_target_.cols(indices);

	// 	// find unique targets
	// 	const arma::Row<arma::uword> unique_targets = 
	// 		arma::unique(element_original_target);

	// 	// walk over target objects and collect coordinates
	// 	for(arma::uword i=0;i<unique_targets.n_elem;i++){
	// 		const arma::Row<arma::uword> idx = arma::find(element_original_target==unique_targets(i)).t();
	// 		Rt.cols(idx) = tars_(i)->get_target_coords(element_original_index.cols(idx));
	// 	}

	// 	// return coordinates
	// 	return Rt;
	// }


	// getting the number of target points
	arma::uword MultiTargets::num_targets() const{
		arma::uword num_targets = 0;
		for(arma::uword i=0;i<tars_.n_elem;i++)
			num_targets += tars_(i)->num_targets();
		return num_targets;
	}

	// setting of calculated field
	void MultiTargets::add_field(
		const char type, 
		const arma::Mat<fltp> &Madd, 
		const bool with_lock){

		// single target case
		if(tars_.n_elem==1){tars_(0)->add_field(type,Madd,with_lock); return;}

		// check input
		assert(!Madd.is_empty());

		// check input
		assert(Madd.n_cols==total_num_targets_);
		
		// sort
		arma::Mat<fltp> Madd_tar(3,total_num_targets_);
		Madd_tar.cols(original_index_ + target_index_.cols(original_target_)) = Madd;

		// set to targets		
		for(arma::uword i=0;i<tars_.n_elem;i++){
			if(tars_(i)->has(type)){
				if(num_targets_(i)>0){
					tars_(i)->add_field(type, Madd_tar.cols(target_index_(i),target_index_(i+1)-1));
				}
			}
		}

	}


	// add field values
	void MultiTargets::add_field(
		const char /*type*/, 
		const arma::Mat<fltp> &/*Madd*/,
		const arma::Row<arma::uword> &/*ft*/,
		const arma::Row<arma::uword> &/*lt*/,
		const arma::Row<arma::uword> &/*ti*/,
		const bool /*with_lock*/){
		rat_throw_line("this function is not implemented");
	}


	// target to multipole step setup function
	void MultiTargets::setup_localpole_to_target(
		const arma::Mat<fltp> &dR, 
		const arma::uword num_dim,
		const ShSettingsPr &stngs){

		// for single target
		if(tars_.n_elem==1){
			tars_(0)->setup_localpole_to_target(dR,num_dim,stngs); return;
		}

		// sort
		arma::Mat<fltp> dR_tar(3,total_num_targets_);
		dR_tar.cols(original_index_ + target_index_.cols(original_target_)) = dR;
		
		// walk over target objects and
		// forward setup to the respective targets
		for(arma::uword i=0;i<tars_.n_elem;i++){
			if(num_targets_(i)>0){
				tars_(i)->setup_localpole_to_target(dR_tar.cols(target_index_(i),target_index_(i+1)-1),num_dim,stngs);
			}
		}

	}

	// calculate target to multipole
	void MultiTargets::localpole_to_target(
		const arma::Mat<std::complex<fltp> > &Lp, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target, 
		const arma::uword num_dim, 
		const ShSettingsPr &stngs){

		// case of only one target
		if(tars_.n_elem==1){
			tars_(0)->localpole_to_target(Lp,first_target,last_target,num_dim,stngs); 
			return;
		}

		// check input target indexing
		assert(first_target(0)==0);
		assert(arma::all(arma::abs(
			first_target.tail_cols(first_target.n_elem-1)-
			last_target.head_cols(last_target.n_elem-1)-1)==0));
		assert(last_target(last_target.n_elem-1)==total_num_targets_-1);

		// check input multipole matrix
		assert(Lp.n_rows==static_cast<arma::uword>(Extra::polesize(stngs->get_num_exp())));
		assert(Lp.n_cols==num_dim*first_target.n_elem);

		// get counters
		const arma::uword num_target_nodes = first_target.n_elem;

		// walk over objects
		for(arma::uword i=0;i<tars_.n_elem;i++){
			// allocate
			arma::Row<arma::uword> reduced_first_target(num_target_nodes); 
			arma::Row<arma::uword> reduced_last_target(num_target_nodes);

			// translate first and last targets for this object
			//for(arma::uword j=0;j<num_target_nodes;j++){
			cmn::parfor(0,num_target_nodes,use_parallel_,[&](arma::uword j, int){
				// find the targets for this node
				const arma::Row<arma::uword> osidx = original_target_.cols(first_target(j),last_target(j));

				const arma::Row<arma::uword> idx1 = arma::find(osidx==i,1,"first").t();
				const arma::Row<arma::uword> idx2 = arma::find(osidx==i,1,"last").t();
				assert(idx1.n_elem<=1); assert(idx2.n_elem<=1);
				
				// calculate indexes for this box
				if(idx1.n_elem==1 && idx2.n_elem==1){
					assert(arma::as_scalar(idx2>=idx1));
					reduced_first_target(j) = original_index_(first_target(j)+arma::as_scalar(idx1));
					reduced_last_target(j) = original_index_(first_target(j)+arma::as_scalar(idx2));
				}

				// disable if no targets found for this box
				else{
					reduced_first_target(j) = 1; reduced_last_target(j) = 0;
				}
			});
			
			// re-index removing the non-existent boxes for this target
			arma::Row<arma::uword> exist_idx = arma::find(reduced_last_target>=reduced_first_target).t();

			// reduce node indexes
			reduced_first_target = reduced_first_target.cols(exist_idx);
			reduced_last_target = reduced_last_target.cols(exist_idx);

			// call target to target on object
			tars_(i)->localpole_to_target(
				Lp.cols(fmm::Extra::expand_indices(exist_idx,num_dim)), 
				reduced_first_target, reduced_last_target, num_dim, stngs);
		}
	}

	// forward setup to each target
	void MultiTargets::setup_targets(){
		// setup childeren
		for(arma::uword i=0;i<tars_.n_elem;i++)tars_(i)->setup_targets();

		// setup self
		setup_multi_targets();
	}

	// forward post processing to each target
	void MultiTargets::post_process(){
		// setup childeren
		for(arma::uword i=0;i<tars_.n_elem;i++)tars_(i)->post_process();
	}


}}