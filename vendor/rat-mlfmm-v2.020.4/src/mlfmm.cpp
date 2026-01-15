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

// include header
#include "mlfmm.hh"

// code specific to Rat
namespace rat{namespace fmm{
	
	// constructor
	Mlfmm::Mlfmm(){
		// set default input settings
		stngs_ = Settings::create();

		// this class takes care of allocation
		stngs_->set_no_allocate(true);
	}

	// constructor with settings, source and target input
	Mlfmm::Mlfmm(const ShSourcesPr &src, const ShTargetsPr &tar){
		// set default input settings
		stngs_ = Settings::create();
		
		// this class takes care of allocation
		stngs_->set_no_allocate(true);

		// add sources and targets 
		set_sources(src); set_targets(tar);
	}

	// constructor with settings, source and target input
	Mlfmm::Mlfmm(const ShSourcesPr &src, const ShTargetsPr &tar, const ShSettingsPr &stngs){
		// add sources and targets 
		set_sources(src); set_targets(tar); set_settings(stngs);
	}

	// constructor
	Mlfmm::Mlfmm(const ShSettingsPr &stngs){
		// check input
		if(stngs==NULL)rat_throw_line("supplied settings is null pointer");

		// set supplied input settings
		stngs_ = stngs;

		// this class takes care of allocation
		stngs_->set_no_allocate(true);
	}

	// factory
	ShMlfmmPr Mlfmm::create(){
		return std::make_shared<Mlfmm>();
	}

	// factory with source and target input
	ShMlfmmPr Mlfmm::create(const ShSourcesPr &src, const ShTargetsPr &tar){
		//return ShMlfmmPr(new Mlfmm(src,tar));
		return std::make_shared<Mlfmm>(src,tar);
	}

	// factory with target and source input
	ShMlfmmPr Mlfmm::create(const ShTargetsPr &tar, const ShSourcesPr &src){
		//return ShMlfmmPr(new Mlfmm(tar,src));
		return std::make_shared<Mlfmm>(src,tar);
	}

	// factory with target and source input
	ShMlfmmPr Mlfmm::create(const ShTargetsPr &tar, const ShSourcesPr &src, const ShSettingsPr &stngs){
		//return ShMlfmmPr(new Mlfmm(tar,src));
		return std::make_shared<Mlfmm>(src,tar,stngs);
	}

	// // factory with target and source input
	ShMlfmmPr Mlfmm::create(const ShSourcesPr &src, const ShTargetsPr &tar, const ShSettingsPr &stngs){
		//return ShMlfmmPr(new Mlfmm(tar,src));
		return std::make_shared<Mlfmm>(src,tar,stngs);
	}

	// factory with target and source input
	ShMlfmmPr Mlfmm::create(const ShSettingsPr &stngs){
		//return ShMlfmmPr(new Mlfmm(tar,src));
		return std::make_shared<Mlfmm>(stngs);
	}

	// set new sources
	void Mlfmm::set_sources(const ShSourcesPr &src){
		// check input
		if(src==NULL)rat_throw_line("supplied sources is null pointer");

		// set to self
		src_ = src;
	}

	// set new targets
	void Mlfmm::set_targets(const ShTargetsPr &tar){
		// check input
		if(tar==NULL)rat_throw_line("supplied targets is null pointer");

		// set to self
		tar_ = tar;
	}

	// // set new sources
	// void Mlfmm::set_multi_sources(ShSourcesPr src){
	// 	// check input
	// 	if(src==NULL)rat_throw_line("supplied sources is null pointer");

	// 	// set to self
	// 	src_ = src;
	// }

	// // set new targets
	// void Mlfmm::set_multi_targets(ShTargetsPr tar){
	// 	// check input
	// 	if(tar==NULL)rat_throw_line("supplied targets is null pointer");

	// 	// set to self
	// 	tar_ = tar;
	// }

	// set background field (NULL means there is no background field)
	void Mlfmm::set_background(const ShBackgroundPr &bg){
		bg_ = bg;
	}


	// set new sources
	void Mlfmm::set_settings(const ShSettingsPr &stngs){
		// check input
		if(stngs==NULL)rat_throw_line("supplied settings is null pointer");

		// this class takes care of allocation
		stngs->set_no_allocate(true);

		// reset and resize to one
		stngs_ = stngs;
	}

	// set allocation flag
	void Mlfmm::set_no_allocate(const bool no_allocate){
		no_allocate_ = no_allocate;
	}

	// setting up the grid and oct-tree structure
	void Mlfmm::setup(const cmn::ShLogPr &lg){
		
		// check log
		if(lg==NULL)rat_throw_line("supplied log is null pointer");

		// report MLFMM is being setup
		lg->msg(2,"%s%sSETTING UP ELEMENTS%s\n",KBLD,KGRN,KNRM);

		// call source setup function
		lg->msg("%ssetup sources%s\n",KBLU,KNRM); 
		src_->setup_sources(); 

		// call target setup function
		lg->msg("%ssetup targets%s\n",KBLU,KNRM);
		if(!no_allocate_)tar_->setup_targets();

		// done
		lg->msg(-2,"\n");

		// when there are no sources no need to setup
		if(src_->num_sources()==0)return;
		if(tar_->num_targets()==0)return;

		// when using direct do not setup mlfmm
		if(use_direct()){
			// report MLFMM is being setup
			lg->msg(2,"%s%sRAT-DIRECT SETUP%s\n",KBLD,KGRN,KNRM);
			
			// type
			lg->msg(2,"%sInput Sources and Targets%s\n",KBLU,KNRM);

			// display information on sources and targets
			const arma::uword num_sources = src_->num_sources();
			const arma::uword num_targets = tar_->num_targets();
			const fltp num_interactions = static_cast<fltp>(num_sources)*static_cast<fltp>(num_targets);
			lg->msg("number of sources: %s%04llu%s\n",KYEL,num_sources,KNRM); 
			lg->msg("number of targets: %s%04llu%s\n",KYEL,num_targets,KNRM); 
			lg->msg("number of elements: %s%04llu%s\n",KYEL,num_sources+num_targets,KNRM); 
			lg->msg("number of interactions: %s%07.3fM%s\n",KYEL,1e-6*num_interactions,KNRM); 

			// done
			lg->msg(-4,"\n");

			// no need for further setup
			return;
		}

		// report MLFMM is being setup
		lg->msg(2,"%s%sRAT-MLFMM SETUP%s\n",KBLD,KGRN,KNRM);

		// create armadillo timer
		arma::wall_clock timer;

		// set timer
		timer.tic();

		// display grid
		stngs_->display(lg);
		
		// call GPU once to ensure it is active
		#ifdef ENABLE_CUDA_KERNELS
		if(GpuKernels::get_num_devices()>0)
			GpuKernels::show_device_info(stngs_->get_gpu_devices(), lg);
		#endif

		// create interaction list object
		ilist_ = IList::create(stngs_);

		// setup oct-tree grid
		grid_ = Grid::create(stngs_,ilist_,src_,tar_);
				
		// create root nodelevel object
		root_ = NodeLevel::create(stngs_,ilist_,grid_);

		// setup interaction lists
		ilist_->setup();
		if(lg->is_cancelled())return;
		ilist_->display(lg);
		
		// call setup functions
		grid_->setup(lg);
		if(lg->is_cancelled())return; 
		grid_->setup_matrices(); 
		grid_->display(lg);

		// setup tree
		root_->setup(lg);
		if(lg->is_cancelled())return;

		// get time
		setup_time_ = timer.toc();

		// done
		lg->msg(-2);
	}

	// run mlfmm calculation
	void Mlfmm::calculate(const cmn::ShLogPr &lg){
		// check for cancelled
		if(lg->is_cancelled())return;

		// check log
		if(lg==NULL)rat_throw_line("supplied log is null pointer");

		// intercept direct calculation
		if(use_direct()){
			calculate_direct(lg); 
			return;
		}

		// run setup on the target
		if(!no_allocate_)tar_->allocate();

		// run external field calculation
		src_->calc_external(tar_,stngs_,lg);

		// when there are no sources no need to setup
		if(tar_->num_targets()==0 || src_->num_sources()==0){
			// call post process and return
			if(!no_allocate_)tar_->post_process();
			return;
		}

		// report MLFMM is calculating
		lg->msg(2,"%s%sRAT-MLFMM CALCULATION%s\n",KBLD,KGRN,KNRM);

		// create armadillo timer
		arma::wall_clock timer;

		// set timer
		timer.tic();

		// check if setup was performed
		assert(root_!=NULL);

		// run multipole method recursively from root
		grid_->sort_srctar();
		root_->run_mlfmm(lg);
		grid_->unsort_srctar();
		if(lg->is_cancelled())return;

		// add background field
		if(bg_!=NULL)bg_->calc_field(tar_);

		// get time
		add_calculation_time(timer.toc());

		// done
		lg->msg(-2);

		// call post process on targets
		if(!no_allocate_)tar_->post_process();
	}

	// direct calculation
	// no need to run setup
	void Mlfmm::calculate_direct(const cmn::ShLogPr &lg){
		// check log
		if(lg==NULL)rat_throw_line("supplied log is null pointer");

		// walk over targets and allocate
		if(!no_allocate_)tar_->allocate(); 

		// calculate external
		src_->calc_external(tar_,stngs_,lg);

		// when there are no sources no need to setup
		if(tar_->num_targets()==0 || src_->num_sources()==0){
			if(!no_allocate_)tar_->post_process();
			return;
		}

		// report
		lg->msg(2,"%s%sRAT-DIRECT CALCULATION%s\n",KBLD,KGRN,KNRM);
		lg->msg(2,"%sCalculation performed at sources%s\n",KBLU,KNRM);
		lg->msg("number of interactions: %s%6.2f M%s\n",KYEL,
			1e-6*static_cast<fltp>(src_->num_sources())*static_cast<fltp>(tar_->num_targets()),KNRM);

		// create armadillo timer
		arma::wall_clock timer;

		// set timer
		timer.tic();

		// all combination mode
		src_->calc_direct(tar_, stngs_, lg);
		if(lg->is_cancelled())return;

		// add background field
		if(bg_!=NULL)bg_->calc_field(tar_);

		// get time
		add_calculation_time(timer.toc());

		// done
		lg->msg("time used: %s%.2f [s]%s\n",KYEL,last_calculation_time_,KNRM);
		lg->msg(-4,"\n");

		// call post process on targets
		if(!no_allocate_)tar_->post_process();
	}

	// access settings
	ShSettingsPr& Mlfmm::settings(){
		return stngs_;
	}

	// display function
	void Mlfmm::display(const cmn::ShLogPr &lg) const{
		// check log
		if(lg==NULL)rat_throw_line("supplied log is null pointer");
		lg->newl();
		//stngs_->display(lg);
		if(grid_!=NULL)grid_->display(lg);
		if(ilist_!=NULL)ilist_->display(lg);
		if(root_!=NULL)root_->display(lg);
	}

	// get setup time
	fltp Mlfmm::get_setup_time() const{
		return setup_time_;
	}

	// get calculation time of last iteration
	fltp Mlfmm::get_last_calculation_time() const{
		return last_calculation_time_;
	}

	// get calculation time
	fltp Mlfmm::get_life_calculation_time() const{
		return life_calculation_time_;
	}

	// direct mode
	bool Mlfmm::use_direct() const{
		return stngs_->get_direct()==DirectMode::ALWAYS || 
			(stngs_->get_direct()==DirectMode::TRESHOLD && 
			static_cast<fltp>(src_->num_sources())*static_cast<fltp>(tar_->num_targets())<
			stngs_->get_direct_tresh());
	}

	// get root
	const ShNodeLevelPr& Mlfmm::get_root()const{
		return root_;
	}

	// get grid
	const ShGridPr& Mlfmm::get_grid()const{
		return grid_;
	}

	// add calculation time
	void Mlfmm::add_calculation_time(const fltp time){
		last_calculation_time_ = time;
		life_calculation_time_ += last_calculation_time_;
	}

}}