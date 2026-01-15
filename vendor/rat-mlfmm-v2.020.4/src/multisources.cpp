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
#include "multisources.hh"

#include "gpukernels.hh"
#include "currentsources.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	MultiSources::MultiSources(){

	}
	
	// constructor
	MultiSources::MultiSources(const ShSourcesPrList &sources){
		srcs_ = sources;
	}

	// factory
	ShMultiSourcesPr MultiSources::create(){
		return std::make_shared<MultiSources>();
	}

	// factory
	ShMultiSourcesPr MultiSources::create(const ShSourcesPrList &sources){
		return std::make_shared<MultiSources>(sources);
	}

	// set combine
	void MultiSources::set_combine(const bool combine){
		combine_ = combine;
	}

	// compress sources
	void MultiSources::combine_sources(){
		// create a list of current sources
		arma::field<ShCurrentSourcesPr> current_srcs(srcs_.n_elem);
		arma::field<ShCurrentSourcesPr> current_srcs_vl(srcs_.n_elem);
		arma::field<ShSourcesPr> remaining_srcs(srcs_.n_elem);

		// walk over sources
		arma::uword cnt1 = 0, cnt2 = 0, cnt3 = 0;
		for(arma::uword i=0;i<srcs_.n_elem;i++){
			const ShCurrentSourcesPr current_src = std::dynamic_pointer_cast<CurrentSources>(srcs_(i));
			if(current_src!=NULL){
				if(current_src->get_van_Lanen())current_srcs_vl(cnt3++) = current_src; 
				else current_srcs(cnt1++) = current_src; 
			}else remaining_srcs(cnt2++) = srcs_(i);
		}

		// no current sources found
		if(cnt1==0 && cnt3==0)return;

		// crop lists
		if(cnt1>0)current_srcs = current_srcs.rows(0,cnt1-1);
		if(cnt3>0)current_srcs_vl = current_srcs_vl.rows(0,cnt3-1);
		if(cnt2>0)remaining_srcs = remaining_srcs.rows(0,cnt2-1);

		// clear sources
		srcs_.clear();

		// add combined sources
		if(cnt2!=0)srcs_ = remaining_srcs;
		if(cnt1!=0)add_sources(CurrentSources::create(current_srcs));
		if(cnt3!=0)add_sources(CurrentSources::create(current_srcs_vl));
	}

	// add sources
	void MultiSources::add_sources(const ShSourcesPr &src){
		// check input
		if(src==NULL)rat_throw_line("model points to zero");

		// get number of models
		const arma::uword num_sources = srcs_.n_elem;

		// allocate new source list
		ShSourcesPrList new_srcs(num_sources + 1);

		// set old and new sources
		for(arma::uword i=0;i<num_sources;i++)new_srcs(i) = srcs_(i);
		new_srcs(num_sources) = src;
		
		// set new source list
		srcs_ = new_srcs;
	}

	// get sources
	const ShSourcesPrList& MultiSources::get_sources()const{
		return srcs_;
	}

	// get number of source objects
	arma::uword MultiSources::get_num_source_objects() const{
		return srcs_.n_elem;
	}

	// setup function
	void MultiSources::setup_multi_sources(){
		// get counters
		const arma::uword num_source_objects = srcs_.n_elem;

		// in case of no source objects
		if(num_source_objects==0)rat_throw_line("source objects are not set");

		// no need for indexing when only one source
		if(num_source_objects==1){
			total_num_sources_ = srcs_(0)->num_sources();
			num_dim_ = srcs_(0)->get_num_dim();
			return;
		}

		// allocate number of sources
		num_sources_.set_size(num_source_objects);

		// walk over source objects
		for(arma::uword i=0;i<num_source_objects;i++)
			num_sources_(i) = srcs_(i)->num_sources();

		// calculate indexes
		source_index_.set_size(num_source_objects+1); source_index_(0) = 0; 
		source_index_.cols(1,num_source_objects) = arma::cumsum<arma::Row<arma::uword> >(num_sources_);

		// total number of sources
		total_num_sources_ = source_index_(num_source_objects);

		// allocate index arrays
		original_index_.set_size(total_num_sources_);
		original_source_.set_size(total_num_sources_);

		// number indexes
		for(arma::uword i=0;i<num_source_objects;i++){
			if(num_sources_(i)>0){
				original_index_.cols(source_index_(i),source_index_(i+1)-1) = 
					arma::regspace<arma::Row<arma::uword> >(0,num_sources_(i)-1);
				original_source_.cols(source_index_(i),source_index_(i+1)-1) = 
					i*arma::Row<arma::uword>(num_sources_(i),arma::fill::ones);
			}
		}

		// number of dimensions
		num_dim_ = srcs_(0)->get_num_dim();
		for(arma::uword i=1;i<num_source_objects;i++)
			if(num_dim_!=srcs_(i)->get_num_dim())
				rat_throw_line("inconsistent number of dimensions");
	}

	// get coordinates of all stored sources
	arma::Mat<fltp> MultiSources::get_source_coords() const{
		// single source case
		if(srcs_.n_elem==1)return srcs_(0)->get_source_coords();

		// check total number of sources
		if(total_num_sources_==0)return{};

		// check if setup
		if(original_index_.is_empty())rat_throw_line("multi sources not setup");
		if(original_source_.is_empty())rat_throw_line("multi sources not setup");
		
		// allocate coordinates
		arma::Mat<fltp> Rs(3,total_num_sources_);
		
		// walk over target objects and collect coordinates
		for(arma::uword i=0;i<srcs_.n_elem;i++){
			if(num_sources_(i)>0){
				Rs.cols(source_index_(i),source_index_(i+1)-1) = srcs_(i)->get_source_coords();
			}
		}
		
		// sort and return
		return Rs.cols(original_index_ + source_index_.cols(original_source_));
	}

	// // get coordinates with specific indexes
	// arma::Mat<fltp> MultiSources::get_source_coords(const arma::Row<arma::uword> &indices) const{
	// 	// single source case
	// 	if(srcs_.n_elem==1)return srcs_(0)->get_source_coords(indices);

	// 	// check if setup
	// 	if(original_index_.is_empty())rat_throw_line("multi sources not setup");
	// 	if(original_source_.is_empty())rat_throw_line("multi sources not setup");

	// 	// get counters
	// 	const arma::uword num_indices = indices.n_elem;

	// 	// allocate coordinates
	// 	arma::Mat<fltp> Rs(3,num_indices);

	// 	// get original objects and indexes
	// 	const arma::Row<arma::uword> element_original_index = original_index_.cols(indices);
	// 	const arma::Row<arma::uword> element_original_source = original_source_.cols(indices);

	// 	// walk over source objects and collect coordinates
	// 	for(arma::uword i=0;i<srcs_.n_elem;i++){
	// 		const arma::Row<arma::uword> idx = arma::find(element_original_source==i).t();
	// 		Rs.cols(idx) = srcs_(i)->get_source_coords(element_original_index.cols(idx));
	// 	}

	// 	// return coordinates
	// 	return Rs;
	// }

	// sort sources using sort index array
	void MultiSources::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// single source case
		if(srcs_.n_elem==1){
			srcs_(0)->sort_sources(sort_idx); return;
		}

		// sort indexes
		original_index_ = original_index_.cols(sort_idx);
		original_source_ = original_source_.cols(sort_idx);

		// walk over sources and sort individually
		for(arma::uword i=0;i<srcs_.n_elem;i++){
			// find indexes of this source
			const arma::Row<arma::uword> idx_source = arma::find(original_source_==i).t();

			// create sorting index
			const arma::Row<arma::uword> local_sort_idx = original_index_.cols(idx_source);

			// sort at source level
			srcs_(i)->sort_sources(local_sort_idx);

			// sort indexes at this multisources level
			original_index_.cols(idx_source.cols(local_sort_idx)) = original_index_.cols(idx_source);
			//original_source_.cols(idx_source.cols(local_sort_idx)) = original_source_.cols(idx_source);
		}
	}

	// unsort sources using (un)sort index array
	void MultiSources::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// single source case
		if(srcs_.n_elem==1){
			srcs_(0)->unsort_sources(sort_idx); return;
		}

		// translate sort index this is where the elements are originint within the source object
		original_index_.cols(sort_idx) = original_index_;
		original_source_.cols(sort_idx) = original_source_;

		// walk over sources and sort individually
		for(arma::uword i=0;i<srcs_.n_elem;i++){
			// find indexes of this source
			const arma::Row<arma::uword> idx_source = arma::find(original_source_==i).t();

			// create sorting index
			const arma::Row<arma::uword> local_sort_idx = original_index_.cols(idx_source);

			// sort at source level
			srcs_(i)->sort_sources(local_sort_idx); // this is on purpose

			// unsort at this multisources level
			original_index_.cols(idx_source.cols(local_sort_idx)) = original_index_.cols(idx_source);
		}
	}

	// direct field calculation of all sources to all
	// target points
	void MultiSources::calc_direct(const ShTargetsPr &tar, const ShSettingsPr &stngs, const cmn::ShLogPr &lg) const{
		// walk over all sources and forward calculation
		for(arma::uword i=0;i<srcs_.n_elem;i++)
			srcs_(i)->calc_direct(tar,stngs,lg);
	}

	// acecss sorting index
	arma::Row<arma::uword> MultiSources::get_original_index()const{
		if(srcs_.n_elem==1)return arma::regspace<arma::Row<arma::uword> >(0,total_num_sources_-1);
		return original_index_;
	}
	
	// access original source index
	arma::Row<arma::uword> MultiSources::get_original_source()const{
		if(srcs_.n_elem==1)return arma::Row<arma::uword>(total_num_sources_, arma::fill::zeros);
		return original_source_;
	}

	// field calculation from specific sources
	void MultiSources::source_to_target(const ShTargetsPr &tar, 
		const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list, 
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const ShSettingsPr &stngs,
		const cmn::ShLogPr &lg) const{

		// check input source indexing
		assert(first_source(0)==0);
		assert(arma::all(arma::abs(
			first_source.tail_cols(first_source.n_elem-1)-
			last_source.head_cols(last_source.n_elem-1)-1)==0));
		assert(last_source(last_source.n_elem-1)==total_num_sources_-1);

		// // get number of cpu's
		// #ifdef ENABLE_CUDA_KERNELS
		// // get number of cpu's
		// const std::set<int> gpu_devices = GpuKernels::get_devices();

		// // check if any gpu device set
		// // if so this is the end of this function
		// // otherwise we use the non-CUDA implementation
		// if(!gpu_devices.empty() && srcs_.n_elem>1){
		// 	// get number of available CPU cores
		// 	// int num_cpus = std::thread::hardware_concurrency();
		// 	std::list<std::future<void> > futures;

		// 	// make index
		// 	std::atomic<arma::uword> idx; idx = 0;

		// 	// walk over available cpus
		// 	for(auto it = gpu_devices.begin();it!=gpu_devices.end();it++){
		// 		// get device index
		// 		const int gpu = *it;

		// 		// start new thread and capture device index
		// 		futures.push_back(std::async(std::launch::async,[&,gpu](){
		// 			// set gpu device for this thread
		// 			GpuKernels::set_device(gpu);

		// 			// keep firing tasks untill run out of idx
		// 			for(;;){
		// 				// atomic increment of index so that no two threads can do same task
		// 				const arma::uword source_index = idx++; 

		// 				// check if last index
		// 				if(source_index>=srcs_.n_elem)break;

		// 				// run source to target for this specific source
		// 				source_to_target_core(source_index,tar,target_list,source_list,first_source,last_source,first_target,last_target);
		// 			}
		// 		}));
		// 	}

		// 	// make sure all threads are finished
		// 	for(auto it=futures.begin();it!=futures.end();it++)(*it).get();

		// 	// return to avoid using the regular kernel
		// 	return;
		// }

		// // case of only one source with GPU enabled
		// if(!gpu_devices.empty() && srcs_.n_elem==1)
		// 	GpuKernels::set_device(*gpu_devices.begin());
		// #endif

		// case of single source
		if(srcs_.n_elem==1){
			srcs_(0)->source_to_target(tar,target_list,source_list,
				first_source,last_source,first_target,last_target,stngs,lg); 
			return;
		}

		// without cuda
		for(arma::uword source_index=0;source_index<srcs_.n_elem;source_index++)
			source_to_target_core(source_index,tar,
				target_list,source_list,first_source,
				last_source,first_target,last_target,stngs,lg);
	}

	// source to target
	void MultiSources::source_to_target_core(
		const arma::uword source_index, const ShTargetsPr &tar, 
		const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list, 
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const ShSettingsPr &stngs,
		const cmn::ShLogPr &lg) const{

		// check zero sources case
		if(num_sources_(source_index)==0)return;

		// get counters
		const arma::uword num_source_nodes = first_source.n_elem;

		// allocate
		arma::Row<arma::uword> reduced_first_source(num_source_nodes); 
		arma::Row<arma::uword> reduced_last_source(num_source_nodes);

		// translate first and last sources for this object
		for(arma::uword j=0;j<num_source_nodes;j++){
		// cmn::parfor(0,num_source_nodes,use_parallel_,[&](arma::uword j, int){
			// find sources for this node
			const arma::Row<arma::uword> osidx = original_source_.cols(first_source(j),last_source(j));
			const arma::Row<arma::uword> idx1 = arma::find(osidx==source_index,1,"first").t();
			const arma::Row<arma::uword> idx2 = arma::find(osidx==source_index,1,"last").t();

			// calculate indexes for this box
			if(idx1.n_elem==1 && idx2.n_elem==1){
				assert(arma::as_scalar(idx2>=idx1));
				reduced_first_source(j) = original_index_(first_source(j)+arma::as_scalar(idx1));
				reduced_last_source(j) = original_index_(first_source(j)+arma::as_scalar(idx2));
			}

			// disable if no sources found for this box
			else{
				reduced_first_source(j) = 1; reduced_last_source(j) = 0;
			}
		}

		// re-index removing the non-existent boxes for this source
		arma::Row<arma::uword> exist = reduced_last_source>=reduced_first_source;
		arma::Row<arma::uword> exist_idx = arma::find(exist>0).t();
		
		// indexing of boxes
		arma::Row<arma::uword> reindex(exist.n_elem,arma::fill::zeros);
		reindex.cols(exist_idx) = arma::regspace<arma::Row<arma::uword> >(0,exist_idx.n_elem-1);

		// allocate
		arma::field<arma::Col<arma::uword> > reduced_source_list(source_list.n_elem,1);
		arma::Col<arma::uword> reduced_target_list(target_list.n_elem);
		
		// walk over source and target list combinations
		arma::uword cnt = 0;
		for(arma::uword j=0;j<source_list.n_elem;j++){
			// find indexes in source list that exist
			const arma::Row<arma::uword> idx = arma::find(exist(source_list(j))).t();

			// indexes found add it to the reduced lists
			if(!idx.is_empty()){
				reduced_source_list(cnt) = reindex.cols(source_list(j).rows(idx)).t();
				reduced_target_list(cnt) = target_list(j);
				assert(arma::all(reduced_source_list(j)<reduced_first_source.n_elem));
				cnt++;
			}
		}

		// call source to target on object
		if(cnt>0){
			// resize source and target lists
			reduced_target_list.resize(cnt);
			reduced_source_list = reduced_source_list.rows(0,cnt-1);

			// reduce node indexes
			reduced_first_source = reduced_first_source.cols(exist_idx);
			reduced_last_source = reduced_last_source.cols(exist_idx);

			// call source to target on object
			srcs_(source_index)->source_to_target(
				tar, reduced_target_list, reduced_source_list, 
				reduced_first_source,reduced_last_source,
				first_target,last_target,stngs,lg);
		}
	}


	// source to multipole step setup function
	void MultiSources::setup_source_to_multipole(
		const arma::Mat<fltp> &dR, 
		const ShSettingsPr &stngs){

		// for single source
		if(srcs_.n_elem==1){
			srcs_(0)->setup_source_to_multipole(dR,stngs); return;
		}

		// walk over source objects and
		// forward setup to the respective sources
		for(arma::uword i=0;i<srcs_.n_elem;i++){
			// get source index
			const arma::Row<arma::uword> idx_source = arma::find(original_source_==i).t();
			const arma::Mat<fltp> dRsrc = dR.cols(idx_source);
			srcs_(i)->setup_source_to_multipole(dRsrc,stngs);
		}
	}

	// calculate source to multipole
	void MultiSources::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp, 
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const ShSettingsPr &stngs) const{

		// case of only one source
		if(srcs_.n_elem==1){
			srcs_(0)->source_to_multipole(Mp,first_source,last_source,stngs); 
			return;
		}

		// get counters
		const arma::uword num_source_nodes = first_source.n_elem;

		// get number of expansions
		const int num_exp = stngs->get_num_exp();
		
		// check input source indexing
		assert(first_source(0)==0);
		assert(arma::all(arma::abs(
			first_source.tail_cols(first_source.n_elem-1)-
			last_source.head_cols(last_source.n_elem-1)-1)==0));
		assert(last_source(last_source.n_elem-1)==total_num_sources_-1);

		// check input multipole matrix
		assert(Mp.n_rows==static_cast<arma::uword>(Extra::polesize(num_exp)));
		assert(Mp.n_cols==num_dim_*first_source.n_elem);

		

		// walk over objects
		for(arma::uword i=0;i<srcs_.n_elem;i++){
			// allocate
			arma::Row<arma::uword> reduced_first_source(num_source_nodes); 
			arma::Row<arma::uword> reduced_last_source(num_source_nodes);

			// translate first and last sources for this object
			//for(arma::uword j=0;j<num_source_nodes;j++){
			cmn::parfor(0,num_source_nodes,use_parallel_,[&](arma::uword j, int){
				// find the sources for this node
				const arma::Row<arma::uword> osidx = original_source_.cols(first_source(j),last_source(j));
				const arma::Row<arma::uword> idx1 = arma::find(osidx==i,1,"first").t();
				const arma::Row<arma::uword> idx2 = arma::find(osidx==i,1,"last").t();
				assert(idx1.n_elem<=1); assert(idx2.n_elem<=1);
				
				// calculate indexes for this box
				if(idx1.n_elem==1 && idx2.n_elem==1){
					assert(arma::as_scalar(idx2>=idx1));
					reduced_first_source(j) = original_index_(first_source(j)+arma::as_scalar(idx1));
					reduced_last_source(j) = original_index_(first_source(j)+arma::as_scalar(idx2));
				}

				// disable if no sources found for this box
				else{
					reduced_first_source(j) = 1; reduced_last_source(j) = 0;
				}
			});

			// re-index removing the non-existent boxes for this source
			arma::Row<arma::uword> exist_idx = arma::find(reduced_last_source>=reduced_first_source).t();

			// reduce node indexes
			reduced_first_source = reduced_first_source.cols(exist_idx);
			reduced_last_source = reduced_last_source.cols(exist_idx);
			
			// call source to target on object
			arma::Mat<std::complex<fltp> > Mp_reduced(
				Extra::polesize(num_exp),num_dim_*exist_idx.n_elem,arma::fill::zeros);
			srcs_(i)->source_to_multipole(
				Mp_reduced, reduced_first_source, reduced_last_source, stngs);
			
			// add multipole contribution
			Mp.cols(fmm::Extra::expand_indices(exist_idx,num_dim_)) += Mp_reduced;
		}
	}

	// get element size
	fltp MultiSources::element_size() const{
		// default size
		fltp element_size = 0;

		// setup childeren
		for(arma::uword i=0;i<srcs_.n_elem;i++)
			element_size = std::max(element_size, srcs_(i)->element_size());

		// return size
		return element_size;
	}


	// setup function
	void MultiSources::setup_sources(){
		// setup childeren
		for(arma::uword i=0;i<srcs_.n_elem;i++)srcs_(i)->setup_sources();

		// combine sources
		if(combine_)combine_sources();

		// setup self
		setup_multi_sources();
	}

	// get number of dimensions
	arma::uword MultiSources::get_num_dim() const{
		return num_dim_;
	}

	// getting basic information
	arma::uword MultiSources::num_sources() const{
		// no need for indexing when only one source
		return total_num_sources_;
	}

	// subdivide
	ShSourcesPr MultiSources::subdivide(const fltp grid_size){
		std::list<ShSourcesPr> src_list;
		bool is_divided = false;
		for(arma::uword i=0;i<srcs_.n_elem;i++){
			const ShSourcesPr subsrc = srcs_(i)->subdivide(grid_size);
			if(subsrc==NULL)src_list.push_back(srcs_(i));
			else{src_list.push_back(subsrc); is_divided = true;}
		}
		ShSourcesPrList src_fld(src_list.size()); arma::uword cnt = 0;
		for(auto it=src_list.begin();it!=src_list.end();it++){
			src_fld(cnt++) = (*it);
		}
		if(is_divided){
			// setup is called from grid
			return MultiSources::create(src_fld);
		}else{
			return NULL;
		}
	}

	// external calculation
	void MultiSources::calc_external(
		const ShTargetsPr &tar, 
		const ShSettingsPr &stngs, 
		const cmn::ShLogPr &lg) const{

		for(arma::uword i=0;i<srcs_.n_elem;i++)
			srcs_(i)->calc_external(tar,stngs,lg);
	}

}}