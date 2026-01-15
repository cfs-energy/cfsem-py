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
#include "settings.hh"

// common headers
#include "rat/common/error.hh"
#include "rat/common/extra.hh"

// mlfmm headers
#include "extra.hh"
#include "gpukernels.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	Settings::Settings(){
		#ifdef ENABLE_CUDA_KERNELS
		const int num_gpu = GpuKernels::get_num_devices();
		for(auto it=gpu_devices_.begin();it!=gpu_devices_.end();){
			if((*it)>=num_gpu)it = gpu_devices_.erase(it); else it++;
		}
		if(num_gpu==0)set_enable_gpu(false);
		#else
		set_enable_gpu(false);
		#endif
	}

	// factory
	ShSettingsPr Settings::create(){
		return ShSettingsPr(new Settings);
	}

	// always direct calculation
	void Settings::set_direct(const DirectMode direct){
		direct_ = direct;
	}

	// always direct calculation
	void Settings::set_direct_tresh(const fltp direct_tresh){
		direct_tresh_ = direct_tresh;
	}

	// large interaction list
	void Settings::set_large_ilist(const bool large_ilist){
		large_ilist_ = large_ilist;
	}

	// set minimal number of levels
	void Settings::set_min_levels(const arma::uword min_levels){
		// set number of expansions
		min_levels_ = min_levels;
	}

	// set minimal number of levels
	void Settings::set_max_levels(const arma::uword max_levels){
		// set number of expansions
		max_levels_ = max_levels;
	}

	// allow subdivision of mesh type of sources
	void Settings::set_allow_subdivision(
		const bool allow_subdivision){
		allow_subdivision_ = allow_subdivision;
	}

	// subdivision criterion
	arma::uword Settings::get_subdivide_criterion()const{
		return subdivide_criterion_;
	}

	// get if subdivision is allowed
	bool Settings::get_allow_subdivision()const{
		return allow_subdivision_;
	}

	// get maximum number of subdivisions
	arma::uword Settings::get_num_subdivide_max()const{
		return num_subdivide_max_;
	}

	// set maximum number of subdivisions
	void Settings::set_num_subdivide_max(const arma::uword num_subdivide_max){
		num_subdivide_max_ = num_subdivide_max;
	}

	// set number of expansions used for multipoles
	void Settings::set_num_exp(const int num_exp){
		// set number of expansions
		num_exp_ = num_exp;
	}

	// set minimum number of elements for grid refinement
	void Settings::set_num_refine(const fltp num_refine){
		// set numrefine
		num_refine_ = num_refine;
	}

	// set size limit for refinement
	void Settings::set_size_limit(const fltp size_limit){
		// set numrefine
		size_limit_ = size_limit;
	}

	// set minimum number of elements for grid refinement
	void Settings::set_num_refine_min(const fltp num_refine_min){
		// set numrefine
		num_refine_min_ = num_refine_min;
	}

	// enable parallel computing on CPU for source to target
	void Settings::set_parallel_s2t(const bool parallel_s2t){
		parallel_s2t_ = parallel_s2t;
	}

	// enable parallel computing on CPU for source to multipole
	void Settings::set_parallel_s2m(const bool parallel_s2m){
		parallel_s2m_ = parallel_s2m;
	}

	// set memory efficiency (see header)
	void Settings::set_memory_efficient_s2m(const bool memory_efficient_s2m){
		memory_efficient_s2m_ = memory_efficient_s2m;
	}

	// set weights for morton indexes
	void Settings::set_morton_weights(
		const arma::uword w0, 
		const arma::uword w1, 
		const arma::uword w2){
		// call overloaded function
		set_morton_weights({w0,w1,w2});
	}	

	// set weights for morton indexes
	void Settings::set_morton_weights(
		const arma::Col<arma::uword>::fixed<3> &weights){
		// check input
		if(weights.n_elem!=3)rat_throw_line("weight vector must contain three elements");
		if(!arma::any(weights==1))rat_throw_line("weight vector must contain a one");
		if(!arma::any(weights==2))rat_throw_line("weight vector must contain a two");
		if(!arma::any(weights==4))rat_throw_line("weight vector must contain a four");
		 
		// set weights
		morton_weights_ = weights;
	}

	// always direct calculation
	DirectMode Settings::get_direct()const{
		return direct_;
	}

	// return number of expansions used for multipoles
	int Settings::get_num_exp() const{
		return num_exp_;
	}

	// return number of expansions used for multipoles
	int Settings::get_polesize() const{
		return Extra::polesize(int(num_exp_));
	}

	// return minimum number of elements for grid refinement
	fltp Settings::get_num_refine() const{
		return num_refine_;
	}

	// get size limit for refinement
	fltp Settings::get_size_limit() const{
		return size_limit_;
	}

	// return minimum number of elements
	fltp Settings::get_num_refine_min() const{
		return num_refine_min_;
	}

	// set stop criterion for grid refinement
	void Settings::set_refine_stop_criterion(const RefineStopCriterion refine_stop_criterion){
		refine_stop_criterion_ = refine_stop_criterion;
	}

	// get stop criterion for grid refinement
	RefineStopCriterion Settings::get_refine_stop_criterion() const{
		return refine_stop_criterion_;
	}

	// return maximum number of levels
	arma::uword Settings::get_max_levels() const{
		return max_levels_;
	}

	// return minimum number of levels
	arma::uword Settings::get_min_levels() const{
		return min_levels_;
	}

	// persistent memory getting
	bool Settings::get_persistent_memory() const{
		return persistent_memory_;
	}

	// large interaction list setting
	bool Settings::get_large_ilist() const{
		return large_ilist_;
	}

	// persistent memory setting
	void Settings::set_persistent_memory(
		const bool persistent_memory){
		persistent_memory_ = persistent_memory;
	}

	// set the subdivision criterion
	void Settings::set_subdivide_criterion(const arma::uword subdivide_criterion){
		subdivide_criterion_ = subdivide_criterion;
	}

	// set whether source to target step is enabled
	void Settings::set_enable_s2t(const bool enable_s2t){
		enable_s2t_ = enable_s2t;
	}

	// set whether fmm steps are enabled
	void Settings::set_enable_fmm(const bool enable_fmm){
		enable_fmm_ = enable_fmm;
	}

	// check if source to target step enabled
	bool Settings::get_enable_s2t() const{
		return enable_s2t_;
	}

	// check if fmm steps enabled
	bool Settings::get_enable_fmm() const{
		return enable_fmm_;
	}

	// morton weighing array
	arma::Col<arma::uword>::fixed<3> Settings::get_morton_weights() const{
		return morton_weights_;
	}

	// single morton weighing
	arma::uword Settings::get_morton_weights(const arma::uword dim) const{
		return morton_weights_(dim);
	}

	// split source to target thread
	bool Settings::get_split_s2t() const{
		// char* str = std::getenv("RAT_NUM_THREADS");
		// if(str!=NULL)return false;
		// #ifdef ENABLE_CUDA_KERNELS
		// return split_s2t_ || (so2ta_enable_gpu_ && !gpu_devices_.empty());
		// #else
		// return split_s2t_;
		// #endif
		return split_s2t_;
	}

	// split source to target thread
	bool Settings::get_split_m2l() const{
		// char* str = std::getenv("RAT_NUM_THREADS");
		// if(str!=NULL)return false;
		// #ifdef ENABLE_CUDA_KERNELS
		// return split_m2l_ || (fmm_enable_gpu_ && !gpu_devices_.empty());
		// #else
		return split_m2l_;
		// #endif
	}

	// parallel multipole to multipole step
	bool Settings::get_parallel_m2m() const{
		return parallel_m2m_;
	}

	// parallel multipole to localpole step
	bool Settings::get_parallel_m2l() const{
		return parallel_m2l_;
	}

	// parallel localpole to localpole step
	bool Settings::get_parallel_l2l() const{
		return parallel_l2l_;
	}

	// parallel tree setup
	bool Settings::get_parallel_tree_setup() const{
		return parallel_tree_setup_;
	}

	// set allocation mode
	void Settings::set_no_allocate(const bool no_allocate){
		no_allocate_ = no_allocate;
	}

	// get allocation mode
	bool Settings::get_no_allocate() const{
		return no_allocate_;
	}

	// get sorting of m2l kernel
	arma::uword Settings::get_m2l_sorting() const{
		// sorting is always by type for GPU
		if(fmm_enable_gpu_ && !gpu_devices_.empty())return 0;
		return m2l_sorting_;
	}

	// get gpu m2l kernel
	bool Settings::get_fmm_enable_gpu() const{
		return fmm_enable_gpu_ && !gpu_devices_.empty();
	}

	// set source to target gpu enabled
	void Settings::set_so2ta_enable_gpu(const bool so2ta_enable_gpu){
		so2ta_enable_gpu_ = so2ta_enable_gpu;
	}

	// set source to multipole gpu enabled
	void Settings::set_so2mp_enable_gpu(const bool so2mp_enable_gpu){
		so2mp_enable_gpu_ = so2mp_enable_gpu;
	}

	// get treshold between direct and mlfmm calculations
	// when auto is enabled
	fltp Settings::get_direct_tresh() const{
		return direct_tresh_;
	}

	// set gpu m2l kernel
	void Settings::set_fmm_enable_gpu(const bool fmm_enable_gpu){
		fmm_enable_gpu_ = fmm_enable_gpu;
	}

	// get whether batch sorting is enabled
	bool Settings::get_use_m2l_batch_sorting() const{
		return use_m2l_batch_sorting_;
	}

	// get maximum number of rescales
	arma::uword Settings::get_num_rescale_max() const{
		return num_rescale_max_;
	}

	// set number of rescales
	void Settings::set_num_rescale_max(const arma::uword num_rescale_max){
		num_rescale_max_ = num_rescale_max;
	}

	// set m2l list sorting type
	void Settings::set_m2l_sorting(const arma::uword m2l_sorting){
		m2l_sorting_ = m2l_sorting;
	}

	// set sorting for multipole to localpole batches (for optimizing gpu)
	void Settings::set_use_m2l_batch_sorting(const bool use_m2l_batch_sorting){
		use_m2l_batch_sorting_ = use_m2l_batch_sorting;
	}

	// split source to target thread from mlfmm
	void Settings::set_split_s2t(const bool split_s2t){
		split_s2t_ = split_s2t;
	}

	// split multipole to local thread each level
	void Settings::set_split_m2l(const bool split_m2l){
		split_m2l_ = split_m2l;
	}

	// use parallel multipole to multipole
	void Settings::set_parallel_m2m(const bool parallel_m2m){
		parallel_m2m_ = parallel_m2m;
	}

	// use parallel multipole to localpole kernel
	void Settings::set_parallel_m2l(const bool parallel_m2l){
		parallel_m2l_ = parallel_m2l;
	}

	// use parallel localpole to localole kernel
	void Settings::set_parallel_l2l(const bool parallel_l2l){
		parallel_l2l_ = parallel_l2l;
	} 

	// use parallel tree setup kernel
	void Settings::set_parallel_tree_setup(const bool parallel_tree_setup){
		parallel_tree_setup_ = parallel_tree_setup;
	} 

	// get gpu devices
	std::set<int> Settings::get_gpu_devices() const{
		return gpu_devices_;
	}

	// set device indices of used GPU's
	void Settings::set_gpu_devices(const std::set<int> &gpu_devices){
		gpu_devices_ = gpu_devices;
	}

	// get gpu devices
	void Settings::add_gpu_device(const int device_index){
		gpu_devices_.emplace(device_index);
	}

	// get gpu devices
	void Settings::remove_gpu_device(const int device_index){
		auto it = gpu_devices_.find(device_index);
		if(it==gpu_devices_.end())rat_throw_line("device is not on list");
		gpu_devices_.erase(it);
	}

	// fmm switchboard
	bool Settings::get_so2ta_enable_gpu() const{
		return so2ta_enable_gpu_ && !gpu_devices_.empty();
	}

	// fmm switchboard
	bool Settings::get_so2mp_enable_gpu() const{
		return so2mp_enable_gpu_ && !gpu_devices_.empty();
	}

	bool Settings::get_parallel_s2t()const{
		return parallel_s2t_;
	}

	// set memory efficiency (see header)
	bool Settings::get_memory_efficient_s2m() const{
		return memory_efficient_s2m_;
	}

	void Settings::set_memory_efficient_l2t(const bool memory_efficient_l2t){
		memory_efficient_l2t_ = memory_efficient_l2t;
	}


	void Settings::set_parallel_l2t(const bool parallel_l2t){
		parallel_l2t_ = parallel_l2t;
	}

	bool Settings::get_memory_efficient_l2t() const{
		return memory_efficient_l2t_;
	}


	bool Settings::get_parallel_l2t() const{
		return parallel_l2t_;
	}

	bool Settings::get_parallel_s2m()const{
		return parallel_s2m_;
	}

	// enable/disable gpu kernels
	void Settings::set_enable_gpu(const bool enable_gpu){
		set_fmm_enable_gpu(false);
		set_so2ta_enable_gpu(enable_gpu);
		set_so2mp_enable_gpu(false);
		set_split_s2t(false);
		set_split_m2l(false);
		set_use_m2l_batch_sorting(false);
	}

	// disable all paralelism in MLFMM
	void Settings::set_single_threaded(){
		fmm_enable_gpu_ = false;
		so2ta_enable_gpu_ = false;
		so2mp_enable_gpu_ = false;
		use_m2l_batch_sorting_ = false;
		split_s2t_ = false;
		split_m2l_ = false;
		parallel_m2m_ = false;
		parallel_m2l_ = false;
		parallel_l2l_ = false;
		parallel_tree_setup_ = false;

		// source side settings
		parallel_s2t_ = false;
		parallel_s2m_ = false;
		parallel_l2t_ = false;
	}

	// vadidity check
	bool Settings::is_valid(bool enable_throws)const{
		if(num_refine_<=1.0){if(enable_throws){rat_throw_line("minimum refinement number must be larger than one");} return false;}
		if(num_exp_>20){if(enable_throws){rat_throw_line("number of expansions should not exceed twenty");} return false;}
		if(num_exp_<=1){if(enable_throws){rat_throw_line("number of expansions must be larger than one");} return false;}
		if(max_levels_>=100){if(enable_throws){rat_throw_line("maximum number of levels must be less than a hundred");} return false;}
		if(max_levels_<3){if(enable_throws){rat_throw_line("maximum number of levels must be larger than two");} return false;}
		if(min_levels_==0){if(enable_throws){rat_throw_line("maximum number of levels must be larger than two");} return false;}
		return true;
	}

	// display function
	void Settings::display(const cmn::ShLogPr &lg) const{
		// header 
		lg->msg(2,"%sInput Settings%s\n",KBLU,KNRM);

		// debug mode or not
		#ifdef NDEBUG
		lg->msg("armadillo debug mode: %sdisabled%s\n",KYEL,KNRM);
		#else
		lg->msg("armadillo debug mode: %senabled%s\n",KYEL,KNRM);
		#endif

		// precision of RAT
		#ifdef RAT_DOUBLE_PRECISION
		lg->msg("rat precision: %sdouble%s\n",KYEL,KNRM);
		#else
		lg->msg("rat precision: %ssingle%s\n",KYEL,KNRM);
		#endif

		// CUDA and GPU
		lg->msg("gpu enabled (if compiled): %s%i%s\n",KYEL,get_fmm_enable_gpu(),KNRM);
		#ifdef ENABLE_CUDA_KERNELS
		lg->msg("gpu compiled: %strue%s\n",KYEL,KNRM);
		#if defined(RAT_DOUBLE_PRECISION) && defined(RAT_CUDA_DOUBLE_PRECISION)
		lg->msg("gpu precision: %sdouble%s\n",KYEL,KNRM);
		#else
		#if defined(RAT_DOUBLE_PRECISION)
		lg->msg("gpu precision: %ssingle%s %s(with conversion)%s\n",KYEL,KNRM,KRED,KNRM);
		#else
		lg->msg("gpu precision: %ssingle%s\n",KYEL,KNRM);
		#endif
		#endif
		#endif

		// direct mode
		lg->msg("direct mode: %s%i%s\n",KYEL,direct_,KNRM);	
		lg->msg("direct treshold: %s%.2f%s\n",KYEL,direct_tresh_,KNRM);	

		// number of expansions used
		lg->msg("number of expansions: %s%llu%s\n",KYEL,num_exp_,KNRM);
		lg->msg("minimal number of levels: %s%llu%s\n",KYEL,min_levels_,KNRM);
		lg->msg("maximum number of levels: %s%llu%s\n",KYEL,max_levels_,KNRM);
		lg->msg("refinement target: %s%.2f%s\n",KYEL,num_refine_,KNRM);
		lg->msg("refinement limit: %s%.2f%s\n",KYEL,num_refine_min_,KNRM);
		lg->msg("persistent memory: %s%i%s\n",KYEL,persistent_memory_,KNRM);
		lg->msg("no allocation: %s%i%s\n",KYEL,no_allocate_,KNRM);
		lg->msg("large interaction list: %s%i%s\n",KYEL,large_ilist_,KNRM);

		// number of available threads
		const int num_cpus = std::thread::hardware_concurrency();
		char* str = std::getenv("RAT_NUM_THREADS");
		const int num_rat_threads = str!=NULL ? std::stoi(std::string(str)) : -1;
		lg->msg("number of CPU threads available: %s%i%s (env) / %s%i%s (hardware)\n",KYEL,num_rat_threads,KNRM,KYEL,num_cpus,KNRM);

		// parallelism
		//lg->msg("parallel source to target: %s%i%s\n",KYEL,parallel_s2t_,KNRM);
		//lg->msg("parallel source to multipole: %s%i%s\n",KYEL,parallel_s2m_,KNRM);
		lg->msg("parallel multipole to multipole: %s%i%s\n",KYEL,parallel_m2m_,KNRM);
		lg->msg("parallel multipole to localpole: %s%i%s\n",KYEL,parallel_m2l_,KNRM);
		lg->msg("parallel localpole to localpole: %s%i%s\n",KYEL,parallel_l2l_,KNRM);
		//lg->msg("parallel localpole to target: %s%i%s\n",KYEL,parallel_l2t_,KNRM);

		// splitting of threads
		lg->msg("split source to target thread: %s%i%s\n",KYEL,split_s2t_,KNRM);
		lg->msg("split multipole to localpole threads: %s%i%s\n",KYEL,split_m2l_,KNRM);

		// footer
		lg->msg(-2,"\n");
	}

	// get type
	std::string Settings::get_type(){
		return "rat::fmm::settings";
	}

	// method for serialization into json
	void Settings::serialize(Json::Value &js, cmn::SList &) const{
		// properties
		js["type"] = get_type();
		js["persistent_memory"] = persistent_memory_;
		js["large_ilist"] = large_ilist_;
		js["no_allocate"] = no_allocate_;
		js["num_exp"] = static_cast<unsigned int>(num_exp_);
		js["min_levels"] = static_cast<unsigned int>(min_levels_);
		js["max_levels"] = static_cast<unsigned int>(max_levels_);
		js["num_refine"] = num_refine_;
		js["num_refine_min"] = num_refine_min_;
		js["num_rescale_max"] = static_cast<unsigned int>(num_rescale_max_);
		for(arma::uword i=0;i<3;i++)js["morton_weights"].append(static_cast<unsigned int>(morton_weights_(i)));
		js["m2l_sorting"] = static_cast<unsigned int>(m2l_sorting_);
		js["fmm_enable_gpu"] = fmm_enable_gpu_;
		js["use_m2l_batch_sorting"] = use_m2l_batch_sorting_;
		js["split_s2t"] = split_s2t_;
		js["split_m2l"] = split_m2l_;
		js["parallel_m2m"] = parallel_m2m_;
		js["parallel_m2l"] = parallel_m2l_;
		js["parallel_l2l"] = parallel_l2l_;
		js["parallel_tree_setup"] = parallel_tree_setup_;
		js["direct_tresh"] = direct_tresh_;
		js["refine_stop_criterion"] = static_cast<int>(refine_stop_criterion_);
		js["direct"] = static_cast<int>(direct_);
		js["so2ta_enable_gpu"] = so2ta_enable_gpu_;
		js["so2mp_enable_gpu"] = so2mp_enable_gpu_;
		js["memory_efficient_s2m"] = memory_efficient_s2m_;
		js["memory_efficient_l2t"] = memory_efficient_l2t_;
		js["parallel_l2t"] = parallel_l2t_;
		js["subdivide_criterion"] = static_cast<int>(subdivide_criterion_);
		js["gpu_devices"] = Json::arrayValue; // empty array
		for(auto it=gpu_devices_.begin();it!=gpu_devices_.end();it++){
			js["gpu_devices"].append(*it);
		}
		js["allow_subdivision"] = allow_subdivision_;
		js["num_subdivide_max"] = static_cast<int>(num_subdivide_max_);
	}

	// method for deserialisation from json
	void Settings::deserialize(
		const Json::Value &js, cmn::DSList &/*list*/, 
		const cmn::NodeFactoryMap &/*factory_list*/, 
		const boost::filesystem::path &/*pth*/){
		// properties
		if(js.isMember("persistent_memory"))persistent_memory_ = js["persistent_memory"].asBool();
		if(js.isMember("large_ilist"))set_large_ilist(js["large_ilist"].asBool());
		if(js.isMember("no_allocate"))set_no_allocate(js["no_allocate"].asBool());
		if(js.isMember("num_exp"))set_num_exp(js["num_exp"].asInt());
		if(js.isMember("min_levels"))set_min_levels(js["min_levels"].asUInt64());
		if(js.isMember("max_levels"))set_max_levels(js["max_levels"].asUInt64());
		if(js.isMember("num_refine"))set_num_refine(js["num_refine"].ASFLTP());
		if(js.isMember("num_refine_min"))set_num_refine_min(js["num_refine_min"].ASFLTP());
		if(js.isMember("num_rescale_max"))set_num_rescale_max(js["num_rescale_max"].asUInt64());
		if(js.isMember("allow_subdivision"))set_allow_subdivision(js["allow_subdivision"].asBool());
		if(js.isMember("num_subdivide_max"))set_num_subdivide_max(js["num_subdivide_max"].asUInt64());
		if(js.isMember("morton_weights")){
			arma::Col<arma::uword>::fixed<3> morton_weights; arma::uword idx = 0;
			for(auto it = js["morton_weights"].begin();it!=js["morton_weights"].end();it++,idx++)morton_weights(idx) = (*it).asUInt64();
			set_morton_weights(morton_weights);
		}
		
		if(js.isMember("m2l_sorting"))set_m2l_sorting(js["m2l_sorting"].asUInt64());
		if(js.isMember("fmm_enable_gpu"))set_fmm_enable_gpu(js["fmm_enable_gpu"].asBool());
		if(js.isMember("use_m2l_batch_sorting"))set_use_m2l_batch_sorting(js["use_m2l_batch_sorting"].asBool());
		if(js.isMember("split_s2t"))set_split_s2t(js["split_s2t"].asBool());
		if(js.isMember("split_m2l"))set_split_m2l(js["split_m2l"].asBool());
		if(js.isMember("parallel_m2m"))set_parallel_m2m(js["parallel_m2m"].asBool());
		if(js.isMember("parallel_m2l"))set_parallel_m2l(js["parallel_m2l"].asBool());
		if(js.isMember("parallel_l2l"))set_parallel_l2l(js["parallel_l2l"].asBool());
		if(js.isMember("parallel_tree_setup"))set_parallel_tree_setup(js["parallel_tree_setup"].asBool());
		if(js.isMember("refine_stop_criterion"))set_refine_stop_criterion(static_cast<RefineStopCriterion>(js["refine_stop_criterion"].asInt()));
		if(js.isMember("direct_tresh"))set_direct_tresh(js["direct_tresh"].ASFLTP());
		if(js.isMember("direct"))set_direct(static_cast<DirectMode>(js["direct"].asInt()));
		if(js.isMember("so2ta_enable_gpu"))set_so2ta_enable_gpu(js["so2ta_enable_gpu"].asBool());
		if(js.isMember("so2mp_enable_gpu"))set_so2mp_enable_gpu(js["so2mp_enable_gpu"].asBool());
		if(js.isMember("memory_efficient_s2m"))set_memory_efficient_s2m(js["memory_efficient_s2m"].asBool());
		if(js.isMember("memory_efficient_l2t"))set_memory_efficient_l2t(js["memory_efficient_l2t"].asBool());
		if(js.isMember("parallel_l2t"))set_parallel_l2t(js["parallel_l2t"].asBool());
		if(js.isMember("subdivide_criterion"))set_subdivide_criterion(js["subdivide_criterion"].asUInt64());

		// gpu device settings
		gpu_devices_.clear();
		#ifdef ENABLE_CUDA_KERNELS
		if(js.isMember("gpu_devices")){
			const int num_devices = GpuKernels::get_num_devices();
			if(num_devices>0){
				set_enable_gpu();
			}
			for(auto it=js["gpu_devices"].begin();it!=js["gpu_devices"].end();it++){
				const int device_id = (*it).asInt();
				if(device_id<num_devices){
					add_gpu_device(device_id); 
				}
			}
		}
		#endif
		
	}

}}
