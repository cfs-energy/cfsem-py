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
#include "grid.hh"

#include "gpukernels.hh"
#include<boost/sort/sort.hpp>

// code specific to Rat
namespace rat{namespace fmm{

	// constructor
	Grid::Grid(){
		
	}

	// constructor with extra input
	Grid::Grid(
		const ShSettingsPr &stngs, const ShIListPr &ilist, 
		const ShSourcesPr &src, const ShTargetsPr &tar){
		
		// set everything
		set_settings(stngs); set_ilist(ilist);
		set_sources(src); set_targets(tar);
	}

			
	// factory
	ShGridPr Grid::create(){
		//return ShIListPr(new IList);
		return std::make_shared<Grid>();
	}

			
	// factory with input
	ShGridPr Grid::create(
		const ShSettingsPr &stngs, const ShIListPr &ilist, 
		const ShSourcesPr &src, const ShTargetsPr &tar){
		//return ShIListPr(new IList);
		return std::make_shared<Grid>(stngs,ilist,src,tar);
	}

	// set settings
	void Grid::set_settings(const ShSettingsPr &stngs){
		// check input
		assert(stngs!=NULL); 

		// set
		stngs_ = stngs;
	}

	// set ilist
	void Grid::set_ilist(const ShIListPr &ilist){
		// check input
		assert(ilist!=NULL);

		// set
		ilist_  = ilist;
	}

	// set new sources
	void Grid::set_sources(const ShSourcesPr &src){
		// check input
		assert(src!=NULL); 

		// add source to list
		src_ = src;
	}

	// set new targets
	void Grid::set_targets(const ShTargetsPr &tar){
		// check input
		assert(tar!=NULL);

		// add target to list
		tar_ = tar;
	}

	// method for setting up the grid
	// finds outer bounds and assigns morton indices 
	// to each element from both sources and targets
	void Grid::setup(const cmn::ShLogPr &lg){
		// check if settings were set
		assert(stngs_!=NULL);
		
		// check if sources and targets set
		assert(src_!=NULL);
		assert(tar_!=NULL);

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// report header
		lg->msg(2,"%sRefining Grid%s\n",KBLU,KNRM);
		
		// re-set number of rescales
		num_rescale_ = 0; num_subdivide_ = 0;

		// set number of levels
		// the refinement will always stop when this number is reached
		// independent of the refinement conditions
		num_levels_ = stngs_->get_max_levels()+1;

		// count number of sources and targets
		num_sources_ = src_->num_sources();
		num_targets_ = tar_->num_targets();

		// allocate coordinates
		arma::Mat<fltp> Re = arma::join_horiz(
			src_->get_source_coords(),
			tar_->get_target_coords());

		// find grid center point and size
		setup_grid_size(Re);

		// get typical element size
		fltp element_size = src_->element_size();

		// keep setting mesh untill requirements met
		while(true){
			// check if there is anything to calculate
			if(num_sources_==0)rat_throw_line("there are no sources");
			if(num_targets_==0)rat_throw_line("there are no targets");

			// count number of sources and targets
			num_elements_ = num_sources_ + num_targets_;

			// set element type
			element_type_ = arma::join_horiz(
				arma::Row<arma::sword>(num_sources_,arma::fill::ones),
				-arma::Row<arma::sword>(num_targets_,arma::fill::ones));

			// set original index
			// original_index_ = arma::join_horiz(
			// 	arma::regspace<arma::Row<arma::uword> >(0,num_sources_-1),
			// 	arma::regspace<arma::Row<arma::uword> >(0,num_targets_-1));

			// weight array (determines order in which the grid 
			// is serialized with the Morton indices, also see grid)
			const arma::Col<arma::uword>::fixed<3> weight = stngs_->get_morton_weights(); // x,y,z

			// set boxsize root level
			fltp boxsize = grid_size_;

			// initialise morton index array
			morton_index_.zeros(1,num_elements_);

			// refine flag
			bool refine_flag = false;

			// refine tree untill maximum levels or refinement condition reached
			for(arma::uword ilevel=0;;ilevel++){
				// check for cancellation
				if(lg->is_cancelled())return;

				// calculate boxsize for this level
				boxsize /= 2;

				// find three dimensional index in grid (incl. conversion to integer values)
				const arma::Mat<arma::uword> grid_index = 
					arma::conv_to<arma::Mat<arma::uword> >::from(
					(Re.each_col() - grid_position_)/boxsize);
				
				// get the modulus this is the position within the parent's box
				const arma::Mat<arma::uword> grid_index_mod = cmn::Extra::modulus(grid_index,2);
				
				// increment morton indices for next level
				morton_index_ = cmn::Extra::bitshift(morton_index_,3);
				morton_index_ += arma::sum(grid_index_mod.each_col()%weight,0);

				// if the level is still below the minimum level
				// it is not necessary to check the other conditions
				if(ilevel+1<stngs_->get_min_levels())continue;

				// using sort as a replacement
				// #ifdef ENABLE_CUDA_KERNELS
				// arma::Row<unsigned int> sorted_source_mortons_int = 
				// 	arma::conv_to<arma::Row<unsigned int> >::from(morton_index_.cols(0,num_sources_-1));
				// arma::Row<unsigned int> sorted_target_mortons_int = 
				// 	arma::conv_to<arma::Row<unsigned int> >::from(morton_index_.cols(num_sources_,num_elements_-1));
				// GpuKernels::sort(sorted_source_mortons_int.memptr(), sorted_source_mortons_int.n_elem);
				// GpuKernels::sort(sorted_target_mortons_int.memptr(), sorted_target_mortons_int.n_elem);
				// const arma::Row<arma::uword> sorted_source_mortons = 
				// 	arma::conv_to<arma::Row<arma::uword> >::from(sorted_source_mortons_int);
				// const arma::Row<arma::uword> sorted_target_mortons = 
				// 	arma::conv_to<arma::Row<arma::uword> >::from(sorted_target_mortons_int);
				// #else

				// sort using boost
				arma::Row<arma::uword> sorted_source_mortons = morton_index_.cols(0,num_sources_-1);
				arma::Row<arma::uword> sorted_target_mortons = morton_index_.cols(num_sources_,num_elements_-1);
				boost::sort::block_indirect_sort(sorted_source_mortons.begin(), sorted_source_mortons.end(), std::thread::hardware_concurrency());
				boost::sort::block_indirect_sort(sorted_target_mortons.begin(), sorted_target_mortons.end(), std::thread::hardware_concurrency());

				// // sort source morton indices and target morton indices
				// const arma::Row<arma::uword> sorted_source_mortons2 = 
				// 	arma::sort(morton_index_.cols(0,num_sources_-1));
				// const arma::Row<arma::uword> sorted_target_mortons2 = 
				// 	arma::sort(morton_index_.cols(num_sources_,num_elements_-1));
				// // #endif
				
				// parallel sort
				// const arma::Row<arma::uword> sorted_source_mortons = 
				// 	cmn::Extra::parsort(morton_index_.cols(0,num_sources_-1));
				// const arma::Row<arma::uword> sorted_target_mortons = 
				// 	cmn::Extra::parsort(morton_index_.cols(num_sources_,num_elements_-1));

				// check sorting
				assert(sorted_source_mortons.is_sorted());
				assert(sorted_target_mortons.is_sorted());

				// find groups of morton indices that have the same value
				const arma::Mat<arma::uword> morton_source_sects = 
					cmn::Extra::find_sections(sorted_source_mortons);
				const arma::Mat<arma::uword> morton_target_sects = 
					cmn::Extra::find_sections(sorted_target_mortons);

				// calculate size of the groups this 
				// is the number of sources and targets in the boxes
				const arma::Row<arma::uword> num_sources_per_box = 
					morton_source_sects.row(1) - morton_source_sects.row(0) + 1;
				const arma::Row<arma::uword> num_targets_per_box = 
					morton_target_sects.row(1) - morton_target_sects.row(0) + 1;

				// average number of sources and targets in all boxes
				mean_num_sources_ = arma::as_scalar(arma::mean(
					arma::conv_to<arma::Row<fltp> >::from(
					num_sources_per_box.cols(arma::find(num_sources_per_box>0))),1));
				mean_num_targets_ = arma::as_scalar(arma::mean(
					arma::conv_to<arma::Row<fltp> >::from(
					num_targets_per_box.cols(arma::find(num_targets_per_box>0))),1));

				// report header
				lg->msg("Level - %s%02llu%s, srcs %s%8.2e%s, tars %s%8.2e%s\n",
					KYEL,ilevel+1,KNRM,KYEL,mean_num_sources_,KNRM,KYEL,mean_num_targets_,KNRM);


				// check for early stop criteria 
				bool cnd1 = false;
				switch(stngs_->get_refine_stop_criterion()){
					// stop when the weighted average number 
					// of sources and average number of targets
					// falls below the refinement target
					case RefineStopCriterion::BOTH:{
						cnd1 = mean_num_sources_<stngs_->get_num_refine() 
							&& mean_num_targets_<stngs_->get_num_refine()
							&& ilevel+1>=stngs_->get_min_levels();
					}break;

					// stop when the weighted average number 
					// of sources or average number of targets
					// falls below the refinement target
					case RefineStopCriterion::EITHER:{
						cnd1 = (mean_num_sources_<stngs_->get_num_refine() 
							|| mean_num_targets_<stngs_->get_num_refine())
							&& ilevel+1>=stngs_->get_min_levels();
					}break;

					// stop when the weighted average number 
					// of particles falls below the refinement target
					case RefineStopCriterion::AVERAGE:{
						const fltp mean_num_particles = 
							(num_sources_per_box.n_elem*mean_num_sources_ + 
							num_targets_per_box.n_elem*mean_num_targets_)/
							(num_sources_per_box.n_elem + num_targets_per_box.n_elem);
						cnd1 = mean_num_particles<stngs_->get_num_refine() 
							&& ilevel+1>=stngs_->get_min_levels();
					}break;

					// stop when the weighted average number 
					// of particles falls below the refinement target
					case RefineStopCriterion::AVERAGE2:{
						const fltp mean_num_particles = 
							(mean_num_sources_+mean_num_targets_)/2;
						cnd1 = mean_num_particles<stngs_->get_num_refine() 
							&& ilevel+1>=stngs_->get_min_levels();
					}break;

					// stop when the average number of interactions 
					// in each box falls below the refinement target
					case RefineStopCriterion::TIMES:{
						const fltp average_num_interactions = mean_num_sources_*mean_num_targets_;
						cnd1 = average_num_interactions<stngs_->get_num_refine()
							&& ilevel+1>=stngs_->get_min_levels();
					}break;

					// stop when the geometric mean falls below refinement target
					case RefineStopCriterion::GEOMETRIC_MEAN:{
						const fltp average_num_interactions = mean_num_sources_*mean_num_targets_;
						cnd1 = std::sqrt(average_num_interactions)<stngs_->get_num_refine()
							&& ilevel+1>=stngs_->get_min_levels();
					}break;

					// error when none of the above
					default: rat_throw_line("refinement condition not recognised");
				}

				// other conditions
				const bool cnd2 = mean_num_sources_<2*stngs_->get_num_refine_min()
					&& ilevel+1>=stngs_->get_min_levels();
				const bool cnd3 = num_levels_==ilevel+1;
				const bool cnd4 = ilevel>=stngs_->get_max_levels();
				// const bool cnd5 = boxsize/2<element_size; // allow element to span maximum 1 box
				const bool cnd5 = boxsize<element_size; // allow element to span maximum 2 boxes
				const bool cnd6 = boxsize<stngs_->get_size_limit(); // allow element to span maximum 2 boxes

				// check if stop triggered
				if(cnd1 || cnd2 || cnd3 || cnd4 || cnd5 || cnd6){
					// report
					if(cnd1)lg->msg("stop criterion reached\n");
					else if(cnd2)lg->msg("min source criterion reached\n");
					else if(cnd3)lg->msg("max number of levels reached\n");
					else if(cnd4)lg->msg("max number of levels reached\n");
					else if(cnd6)lg->msg("settings size criterion reached: %s%2.2f%s/%s%2.2f%s [mm]\n",
						KYEL,1e3*stngs_->get_size_limit(),KNRM,KYEL,1e3*boxsize,KNRM);
					else if(cnd5){
						// lg->msg("element size criterion reached: %s%2.2f%s/%s%2.2f%s [mm] (1 box)\n",
						// 	KYEL,1e3*element_size,KNRM,KYEL,1e3*2*boxsize,KNRM);
						lg->msg("element size criterion reached: %s%2.2f%s/%s%2.2f%s [mm] (2 boxes)\n",
							KYEL,1e3*element_size,KNRM,KYEL,1e3*2*boxsize,KNRM);
						if(mean_num_sources_>stngs_->get_subdivide_criterion())
							// if(mean_num_sources_>mean_num_targets_)
							refine_flag = true;
					}

					// set number of levels
					// if the refinement is run again
					// it will be the new maximum level
					num_levels_ = ilevel+1;

					// calculate position of the nodes
					Rn_ = (arma::conv_to<arma::Mat<fltp> >::from(grid_index)+0.5)*boxsize;
					Rn_.each_col()+=grid_position_;

					// end refining
					// break out of for loop
					break;
				}
			}

			// try subdivide meshes in sources to decrease element size
			// such that the grid can be refined further
			if(refine_flag && stngs_->get_allow_subdivision() && 
				num_subdivide_<stngs_->get_num_subdivide_max()){

				// unflag
				refine_flag = false;

				// subdividing mesh
				lg->msg("subdividing sources: ");

				// subdivide to target grid size
				const ShSourcesPr subsrc = src_->subdivide(boxsize/2); 

				// check if subdivision succeeded
				fltp new_element_size = element_size;
				if(subsrc!=NULL){
					// rerun setup
					subsrc->setup_sources();

					// get new element size
					new_element_size = subsrc->element_size();
				}

				// check if subdivision changed the elements size
				if(std::abs(new_element_size-element_size)<RAT_CONST(1e-9)){
					// report
					lg->msg(0,"%sno effect on element size%s\n",KYEL,KNRM);
				}

				// yes element size changed
				else{
					// report
					lg->msg(0,"%selement size changed%s\n",KGRN,KNRM);
					lg->msg("new element size: %s%10.2e%s [m]\n",KYEL,new_element_size,KNRM);

					// accept new sources
					src_ = subsrc; element_size = new_element_size;

					// combine coordinates of sources and targets
					Re = arma::join_horiz(
						src_->get_source_coords(), 
						tar_->get_target_coords());

					// update grid size
					setup_grid_size(Re);

					// update number of sources and targets
					num_sources_ = src_->num_sources();
					num_targets_ = tar_->num_targets();

					// reset max number of levels
					num_levels_ = stngs_->get_max_levels()+1;

					// increment counter
					num_subdivide_++;

					// have another go
					continue;
				}
			}

			// does the gridsize need changing (in case of overshoot)
			// this because insufficient sources are in box
			// this feature gives more refined control
			// over the average number of elements inside the boxes
			if(mean_num_sources_<stngs_->get_num_refine_min() && 
				num_rescale_<stngs_->get_num_rescale_max()){
				// report
				lg->msg("rescale grid size and restart\n");

				// update grid
				// keep center of grid at same location
				scale_grid_size(RAT_CONST(1.1));

				// have another go
				continue;
			}

			// break out of while loop
			break;
		}

		// check for cancellation
		if(lg->is_cancelled())return;

		// report
		lg->msg("sorting morton indexes\n");

		// setup indices
		setup_indices();

		// calculate relative position between 
		// elements and their respective nodes
		assert(Re.is_finite());
		Re = Re.cols(sort_index_);
		dR_ = Rn_-Re; // after sorting

		// get time used for setting up the grid
		last_setup_time_ = timer.toc();
		life_setup_time_ += last_setup_time_;

		// show time used
		lg->msg("time used: %s%.2f [s]%s\n",KYEL,last_setup_time_,KNRM);

		// grid refinement done
		lg->msg(-2,"\n");
	}

	// determine grid size based on particle coords
	void Grid::setup_grid_size(const arma::Mat<fltp>&Re){
		const arma::Col<fltp>::fixed<3> Rmax = arma::max(Re,1);
		const arma::Col<fltp>::fixed<3> Rmin = arma::min(Re,1);
		grid_size_ = 1.001*arma::max(Rmax - Rmin);
		grid_position_ = (Rmax + Rmin)/2 - grid_size_/2;
	}

	// scale grid
	void Grid::scale_grid_size(const fltp scale_factor){
		grid_position_ += grid_size_/2;
		grid_size_ *= scale_factor;
		grid_position_ -= grid_size_/2;
		num_rescale_++;
	}

	// setup indices
	void Grid::setup_indices(){
		// find sources and targets
		const arma::Row<arma::uword> src_idx = 
			arma::find(element_type_==1).t();
		const arma::Row<arma::uword> tar_idx = 
			arma::find(element_type_==-1).t();

		// morton index of sources and targets separately
		arma::Row<arma::uword> source_morton = 
			morton_index_.cols(src_idx);
		arma::Row<arma::uword> target_morton = 
			morton_index_.cols(tar_idx);

		// sort sources and target by morton
		source_sort_index_ = arma::sort_index(source_morton).t();
		target_sort_index_ = arma::sort_index(target_morton).t();

		// sort morton indexes
		source_morton = source_morton.cols(source_sort_index_);
		target_morton = target_morton.cols(target_sort_index_);

		// find sections with the same morton index for sources
		const arma::Mat<arma::uword> source_sections = 
			cmn::Extra::find_sections(source_morton);
		first_source_ = source_sections.row(0);
		last_source_ = source_sections.row(1);
		num_source_nodes_ = source_sections.n_cols;

		// find sections with the same morton index for targets
		const arma::Mat<arma::uword> target_sections = 
			cmn::Extra::find_sections(target_morton);
		first_target_ = target_sections.row(0);
		last_target_ = target_sections.row(1);
		num_target_nodes_ = target_sections.n_cols;
		
		// create global sorting array for all elements
		sort_index_ = arma::regspace<arma::Row<arma::uword> >(0,num_elements_-1);
		const arma::Row<arma::uword> list1 = sort_index_.cols(src_idx);
		const arma::Row<arma::uword> list2 = sort_index_.cols(tar_idx);
		sort_index_.cols(src_idx) = list1.cols(source_sort_index_);
		sort_index_.cols(tar_idx) = list2.cols(target_sort_index_);

		// sort morton indexes of all elements
		morton_index_ = morton_index_.cols(sort_index_);

		// now sort all elements by morton without 
		// changing the order of the sources and targets
		// this is achieved by stable sort
		const arma::Row<arma::uword> srt = 
			arma::stable_sort_index(morton_index_).t();

		// now the sort indexes need to be sorted
		sort_index_ = sort_index_.cols(srt);

		// sort everything
		morton_index_ = morton_index_.cols(srt);
		//original_index_ = original_index_.cols(srt); 
		// not really original index anymore (just were the element is now)
		element_type_ = element_type_.cols(sort_index_);
		Rn_ = Rn_.cols(sort_index_);

		// make sure sorting worked correctly
		assert(morton_index_.is_sorted());

		// check output
		assert(dR_.is_finite());
		assert(Rn_.is_finite());
	}


	// calculate morton index from grid index
	arma::Mat<arma::uword> Grid::grid2morton(
		const arma::Mat<arma::uword> &grid_index, 
		const arma::uword level){

		// check if grid index is three-dimensional
		assert(grid_index.n_rows==3);

		// check if grid index is in range of this level
		assert(arma::all(arma::all(grid_index>=0)));
		assert(arma::all(arma::all(grid_index<arma::uword(std::pow(2,level)))));

		// get number of indexes
		arma::uword num = grid_index.n_cols;
		arma::Mat<arma::uword> n = grid_index;
		arma::Mat<arma::uword> morton_index(1,num,arma::fill::zeros);

		// walk over levels and calculate morton index contribution
		for(int ilevel=0;ilevel<int(level);ilevel++) {
			morton_index += cmn::Extra::bitshift(
				cmn::Extra::modulus(n.row(0),2), 3*ilevel + 1);
			morton_index += cmn::Extra::bitshift(
				cmn::Extra::modulus(n.row(1),2), 3*ilevel);
			morton_index += cmn::Extra::bitshift(
				cmn::Extra::modulus(n.row(2),2), 3*ilevel + 2);
			n = cmn::Extra::bitshift(n,-1);
		}

		// return list of morton indices
		return morton_index;
	}

	// calculate grid index from morton index
	// for testing only
	arma::Mat<arma::uword> Grid::morton2grid(
		const arma::Mat<arma::uword> &morton_index){
		
		// check input
		assert(morton_index.n_rows==1);

		// get number of morton indices
		arma::uword num = morton_index.n_cols;

		// allocate output grid vector
		arma::Mat<arma::uword> grid_index(3,num,arma::fill::zeros);

		// copy morton indices into n (to allow modifying
		arma::Mat<arma::uword> n = morton_index;
			
		// calculate morton indices
		int k = 0; int i = 0;
		while(arma::as_scalar(arma::any(n!=0,1))){
			int j = 2-k;
			grid_index.row(j)+=(cmn::Extra::modulus(n,2))*(1llu << i);
			n = cmn::Extra::bitshift(n,-1);
			k = (k+1)%3;
			if(k==0)i++;
		}

		// sort output rows
		arma::Mat<arma::uword> idx = {1,2,0};
		grid_index = grid_index.rows(idx);

		// output index in grid
		return grid_index;
	}

	// setup MLFMM matrices corresponding to grid size
	void Grid::setup_matrices(){
		// use threading
		char* str = std::getenv("RAT_NUM_THREADS");
		const bool allow_threads = str==NULL;
		
		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// use thread to setup translation matrices
		// i.e. Mp2Mp and Lp2Lp
		std::future<void> th1 = std::async([&](){
			setup_translation_matrices();
		});

		if(!allow_threads)th1.get(); 

		// use thread to setup transformation matrices
		// i.e. Mp2Lp
		std::future<void> th2 = std::async([&](){
			setup_transformation_matrices();
		});

		if(!allow_threads)th2.get();

		// setup source to multipole matrix in sources
		std::future<void> th3 = std::async([&](){
			// get indices of source elements
			const arma::Row<arma::uword> idx = arma::find(element_type_==1).t();

			// sort relative position vectors in the order of the sources
			//arma::Mat<fltp> dRs(3,num_sources_);
			//dRs.cols(original_index_.cols(idx)) = dR_.cols(idx);
			const arma::Mat<fltp> dRs = dR_.cols(idx);

			// setup source to multipole matrices based on the relative position
			src_->sort_sources(source_sort_index_);
			src_->setup_source_to_multipole(dRs,stngs_);
			src_->unsort_sources(source_sort_index_);
		});

		// setup localpole to to target matrix in targets
		// walk over targets	
		// get indices of target elements
		const arma::Row<arma::uword> idx = arma::find(element_type_==-1).t();

		// sort relative position vectors in the order of the targets
		//arma::Mat<fltp> dRt(3,num_targets_);
		//dRt.cols(original_index_.cols(idx)) = dR_.cols(idx);
		const arma::Mat<fltp> dRt = dR_.cols(idx);

		// setup localpole to target matrices based on the relative position
		tar_->sort_targets(target_sort_index_);
		tar_->setup_localpole_to_target(dRt,src_->get_num_dim(),stngs_);
		tar_->unsort_targets(target_sort_index_);

		// join threads
		if(allow_threads)th1.get(); 
		if(allow_threads)th2.get(); 
		th3.get();

		// get time used for setting up the grid
		last_matrix_time_ = timer.toc();
		life_matrix_time_ += last_matrix_time_;
	}

	// setup multipole to multipole matrices and
	// setup localpole to localpole matrices
	// sets up the translation matrices for each child according to
	// the boxsize specified at the finest grid level.
	void Grid::setup_translation_matrices(){
		// check input
		assert(stngs_!=NULL);
		assert(ilist_!=NULL);
		
		// generate list of coordinates based on interaction list
		// these coordinates are given as Rchild-Rparent
		const fltp box_size = get_box_size();
		arma::Mat<arma::uword> type_nshift = ilist_->get_type_nshift();
		arma::Mat<fltp> dR = arma::conv_to<arma::Mat<fltp> >
			::from(type_nshift.t())*box_size-box_size/2;

		// create m2m matrix
		multipole_to_multipole_matrix_ = FmmMat_Mp2Mp::create();
		
		// set number of expansions
		multipole_to_multipole_matrix_->set_num_exp(stngs_->get_num_exp());

		// calculate matrix from coordinates
		// dR = Rsource (child) - Rtarget (parent) (also see test_matrices.cpp)
		multipole_to_multipole_matrix_->calc_matrix(dR);

		// create l2l matrix
		localpole_to_localpole_matrix_ = FmmMat_Lp2Lp::create();

		// set number of expansions to be used for matrix setup
		localpole_to_localpole_matrix_->set_num_exp(stngs_->get_num_exp());

		// calculate matrix from coordinates
		// dR = Rsource (parent) - Rtarget (child) (also see test_matrices.cpp)
		localpole_to_localpole_matrix_->calc_matrix(-dR);
	}

	// sets up the transformation matrix for each approximate interaction
	// stored in the interaction list and according to
	// the boxsize specified at the finest grid level.
	void Grid::setup_transformation_matrices(){
		// check input
		assert(stngs_!=NULL);
		assert(ilist_!=NULL);
			
		// generate list of coordinates based on approximate interaction list
		// these coordinates are given as: Rbox - Rcenterpoint
		const fltp box_size = get_box_size();
		arma::Mat<arma::sword> approx_nshift = ilist_->get_approx_nshift();
		arma::Mat<fltp> dR = arma::conv_to<arma::Mat<fltp> >
			::from(approx_nshift.t())*box_size;

		// create m2l matrix
		multipole_to_localpole_matrix_ = FmmMat_Mp2Lp::create();

		// set number of expansions to be used for matrix setup
		multipole_to_localpole_matrix_->set_num_exp(stngs_->get_num_exp());

		// calculate matrix from coordinates
		// dR = Rsource - Rtarget (also see test_matrices.cpp)
		multipole_to_localpole_matrix_->calc_matrix(dR);
	}



	// get mp2mp matrix reference for use in nodelevel
	const ShFmmMat_Mp2MpPr& Grid::get_multipole_to_multipole_matrix(){
		return multipole_to_multipole_matrix_;
	}

	// get lp2lp matrix reference for use in nodelevel
	const ShFmmMat_Lp2LpPr& Grid::get_localpole_to_localpole_matrix(){
		return localpole_to_localpole_matrix_;
	}

	// get mp2lp matrix reference for use in nodelevel
	const ShFmmMat_Mp2LpPr& Grid::get_multipole_to_localpole_matrix(){
		return multipole_to_localpole_matrix_;
	}

	// getting morton index
	const arma::Mat<arma::uword>& Grid::get_morton() const{
		return morton_index_;
	}

	// get number of levels
	arma::uword Grid::get_num_levels() const{
		return num_levels_;
	}

	// get number of levels
	fltp Grid::get_grid_size() const{
		return grid_size_;
	}

	// get position
	arma::Col<fltp>::fixed<3> Grid::get_position() const{
		return grid_position_;
	}

	// calculate grid center position
	arma::Col<fltp>::fixed<3> Grid::center_position() const{
		return grid_position_ + grid_size_/2;
	}

	// get is_source boolean array
	arma::Mat<arma::uword> Grid::get_is_source() const{
		return element_type_>0;
	}

	// get target boolean array
	arma::Mat<arma::uword> Grid::get_is_target() const{
		return element_type_<0;
	}

	// get list of first sources
	const arma::Row<arma::uword>& Grid::get_first_source()const{
		return first_source_;
	}

	// get list of last sources
	const arma::Row<arma::uword>& Grid::get_last_source()const{
		return last_source_;
	}

	// get list of first target
	const arma::Row<arma::uword>& Grid::get_first_target()const{
		return first_target_;
	}

	// get list of last target
	const arma::Row<arma::uword>& Grid::get_last_target()const{
		return last_target_;
	}

	// get source sorting indexes 
	const arma::Row<arma::uword>& Grid::get_source_sort_index()const{
		return source_sort_index_;
	}

	// get target sorting indexes 
	const arma::Row<arma::uword>& Grid::get_target_sort_index()const{
		return target_sort_index_;
	}

	// get is_source boolean array
	// arma::Row<arma::uword> Grid::get_original_index() const{
	// 	return original_index_;
	// }

	// get is_source boolean array
	const arma::Row<arma::uword>& Grid::get_sort_index() const{
		return sort_index_;
	}

	// get box size at maximum level
	fltp Grid::get_box_size() const{
		return get_box_size(num_levels_);
	}

	// get box size at specified level
	fltp Grid::get_box_size(
		const arma::uword ilevel) const{

		assert(ilevel<=num_levels_);
		return grid_size_ / std::pow(2,ilevel);
	}

	// get stored locations of the nodes
	arma::Mat<fltp> Grid::get_node_locations(
		const arma::Mat<arma::uword> &indices) const{
		return Rn_.cols(indices);
	}

	// number of sources
	arma::uword Grid::get_num_sources() const{
		return num_sources_;
	}

	// get number of targets
	arma::uword Grid::get_num_targets() const{
		return num_targets_;
	}

	// get number of elements
	arma::uword Grid::get_num_elements() const{
		return num_elements_;
	}

	// get average number of sources in each box
	fltp Grid::get_mean_num_sources() const{
		return mean_num_sources_;
	}

	// get average number of targets in each box
	fltp Grid::get_mean_num_targets() const{
		return mean_num_targets_;
	}

	// get number of rescales
	arma::uword Grid::get_num_rescale() const{
		return num_rescale_;
	}


	// direct calculation of magnetic field and/or vector potential
	// use pre-set target list (normal way of calling)
	void Grid::source_to_target(
		const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const cmn::ShLogPr &lg){
		// run source to target kernel
		src_->source_to_target(tar_, 
			target_list, source_list,
			first_source_, last_source_,  
			first_target_, last_target_, stngs_, lg);
	}

	// execute source to multipole
	void Grid::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp) const{

		// check input
		assert(!Mp.is_empty());
		assert(Mp.n_cols==src_->get_num_dim()*num_source_nodes_);
		assert(Mp.n_rows==static_cast<arma::uword>(stngs_->get_polesize()));

		// forward to sources
		src_->source_to_multipole(Mp,first_source_,
			last_source_,stngs_);

		// check for nans
		assert(!Mp.has_nan());
	}

	// localpole to target calculation
	void Grid::localpole_to_target(
		const arma::Mat<std::complex<fltp> > &Lp){

		// check input
		assert(!Lp.is_empty());
		assert(Lp.n_cols==src_->get_num_dim()*num_target_nodes_);
		assert(Lp.n_rows==static_cast<arma::uword>(stngs_->get_polesize()));
		assert(!Lp.has_nan());

		// send the localpoles to targets to calculate the field there
		tar_->localpole_to_target(Lp, first_target_, last_target_, 
			src_->get_num_dim(), stngs_);
	}

	// fmm setup function
	void Grid::fmm_setup(){
		// allocate targets
		if(!stngs_->get_no_allocate()){
			tar_->allocate();
		}
	}

	// fmm post function
	void Grid::fmm_post(){

	}

	// sort sources and targets
	void Grid::sort_srctar(){
		// check if indexes were setup
		assert(!source_sort_index_.is_empty());
		assert(!target_sort_index_.is_empty());
		src_->sort_sources(source_sort_index_);
		tar_->sort_targets(target_sort_index_);
	}

	// unsort sources and targets
	void Grid::unsort_srctar(){
		// check if indexes were setup
		assert(!source_sort_index_.is_empty());
		assert(!target_sort_index_.is_empty());
		src_->unsort_sources(source_sort_index_);
		tar_->unsort_targets(target_sort_index_);
	}

	// number of dimensions
	arma::uword Grid::get_num_dim() const{
		return src_->get_num_dim();
	}

	// display stored values in terminal
	void Grid::display(const cmn::ShLogPr &lg) const{
		// type
		lg->msg(2,"%sComputing Grid%s\n",KBLU,KNRM);

		// display information on sources and targets
		lg->msg("number of sources: %s%04llu%s\n",KYEL,num_sources_,KNRM); 
		lg->msg("number of targets: %s%04llu%s\n",KYEL,num_targets_,KNRM); 
		lg->msg("number of elements: %s%04llu%s\n",KYEL,num_elements_,KNRM); 

		// display grid positioning and size
		lg->msg(2,"grid position:\n");   
		lg->msg(-2,"x = %s%+.2f%s, y = %s%+.2f%s, z = %s%+.2f [m]%s\n",
			KYEL,grid_position_(0),KNRM,KYEL,grid_position_(1),KNRM,KYEL,grid_position_(2),KNRM); 
		lg->msg(2,"grid center position:\n"); 
		arma::Col<fltp>::fixed<3> Rcen = center_position();  
		lg->msg(-2,"x = %s%+.2f%s, y = %s%+.2f%s, z = %s%+.2f [m]%s\n",
			KYEL,Rcen(0),KNRM,KYEL,Rcen(1),KNRM,KYEL,Rcen(2),KNRM); 
		lg->msg("grid size: %s%.2f [m]%s\n",KYEL,grid_size_,KNRM); 

		// display information on refinement
		lg->msg("refined to %s%03llu%s out of %s%03llu%s levels\n",
			KYEL,num_levels_,KNRM,
			KYEL,stngs_->get_max_levels(),KNRM); 
		lg->msg("number of rescales used: %s%llu%s\n",KYEL,num_rescale_,KNRM);
		lg->msg("number of mesh subdivisions used: %s%llu%s\n",KYEL,num_subdivide_,KNRM);
		lg->msg("average number of sources: %s%.2f%s\n",KYEL,mean_num_sources_,KNRM); 
		lg->msg("average number of targets: %s%.2f%s\n",KYEL,mean_num_targets_,KNRM);
		lg->msg("refinement criterion: %s%i%s\n",KYEL,stngs_->get_refine_stop_criterion(),KNRM);

		// display time
		lg->msg("setup time: %s%.2f / %.2f [s]%s\n",KYEL,last_setup_time_,life_setup_time_,KNRM);
		lg->msg("matrix time: %s%.2f / %.2f [ms]%s\n",KYEL,1e3*last_matrix_time_,1e3*life_matrix_time_,KNRM);

		// grid setup finished
		lg->msg(-2,"\n");
	}

}}