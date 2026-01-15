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
#include "nodelevel.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// constructor
	NodeLevel::NodeLevel(){

	}

	// constructor
	NodeLevel::NodeLevel(const ShSettingsPr &stngs, 
		const ShIListPr &ilist, const ShGridPr &grid){
		// check input
		assert(ilist!=NULL); assert(stngs!=NULL); assert(grid!=NULL);
		
		// store provided pointers in self
		ilist_ = ilist; stngs_ = stngs; grid_ = grid;
	}

	// destructor
	NodeLevel::~NodeLevel(){
		
	}

	// construct from previous level for recursive construction
	NodeLevel::NodeLevel(NodeLevel* prev_level){
		// set previous level
		prev_level_ = prev_level;
		
		// copy pointers from previous level
		grid_ = prev_level->grid_;
		ilist_ = prev_level->ilist_;
		stngs_ = prev_level->stngs_;
	}

	// factory
	ShNodeLevelPr NodeLevel::create(){
		//return UnNodeLevelPr(new NodeLevel);
		return std::make_shared<NodeLevel>();
	}

	// factory with extra inputs
	ShNodeLevelPr NodeLevel::create(const ShSettingsPr &stngs, 
		const ShIListPr &ilist, const ShGridPr &grid){
		// call constructor and return shared pointer
		return std::make_shared<NodeLevel>(stngs,ilist,grid);
	}

	// factory from previous level for recursive construction
	ShNodeLevelPr NodeLevel::create(NodeLevel* prev_level){
		//return UnNodeLevelPr(new NodeLevel(prev_level));
		return std::make_shared<NodeLevel>(prev_level);
	}


	// set grid
	void NodeLevel::set_grid(const ShGridPr &grid){
		assert(grid!=NULL);
		grid_ = grid;
	}

	// set ilist
	void NodeLevel::set_ilist(const ShIListPr &ilist){
		assert(ilist!=NULL);
		ilist_ = ilist;
	}

	// set settings
	void NodeLevel::set_settings(const ShSettingsPr &stngs){
		assert(stngs!=NULL);
		stngs_ = stngs;
	}

	// setup tree
	void NodeLevel::setup(const cmn::ShLogPr &lg){
		// check if ilist and grid are set
		assert(grid_!=NULL); assert(ilist_!=NULL);

		// header
		lg->msg(2,"%sGrowing %sOct-Tree%s structure%s\n",KBLU,KGRN,KBLU,KNRM);

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// ensure this method is called on root
		assert(is_root());

		// get number of levels
		const arma::uword num_levels = grid_->get_num_levels();

		// setting up tree structure
		lg->msg("level %s%02llu%s to %s%02llu%s - growing %sleaves%s, %snodes%s and %sbranches%s\n",
			KYEL,num_levels,KNRM,KYEL,level_,KNRM,KGRN,KNRM,KGRN,KNRM,KGRN,KNRM);
		setup_structure();
		if(lg->is_cancelled())return;

		// coordinate setup		
		lg->msg("level %s%02llu%s to %s%02llu%s - calculating node coordinates\n",
			KYEL,0,KNRM,KYEL,num_levels,KNRM);
		setup_node_locations();
		if(lg->is_cancelled())return;

		// interaction list setup
		lg->msg("level %s%02llu%s to %s%02llu%s - create interaction lists (%sL1%s, %sL2%s)\n",
			KYEL,0,KNRM,KYEL,num_levels,KNRM,KGRN,KNRM,KGRN,KNRM);
		setup_interaction_lists();
		if(lg->is_cancelled())return;

		// multipole allocation
		lg->msg("level %s%02llu%s to %s%02llu%s - allocate multipoles and localpoles\n",
			KYEL,num_levels,KNRM,KYEL,level_,KNRM);
		setup_multipoles();
		if(lg->is_cancelled())return;

		// multipole to localpole lists
		lg->msg("level %s%02llu%s to %s%02llu%s - setup mp2lp lists\n",
			KYEL,0,KNRM,KYEL,num_levels,KNRM);
		setup_mp2lp_lists();
		if(lg->is_cancelled())return;

		// setup source to target lkist for leaf level
		const ShNodeLevelPr& leaf = get_leaf();

		// setting up source to target lists
		lg->msg("level %s%02llu%s - setup so2ta lists\n",
			KYEL,num_levels,KNRM);
		leaf->setup_s2t_lists();
		if(lg->is_cancelled())return;

		// get time
		const fltp tree_setup_time = timer.toc();
		
		// footer
		lg->msg("time used: %s%.2f [s]%s\n",KYEL,tree_setup_time,KNRM);
		lg->msg(-2,"tree fully grown %s(don't forget to water)%s\n\n",KGRN,KNRM);
	}

	// setup tree structure for root level
	// this method is always called on root
	void NodeLevel::setup_structure(){
		// call on root
		setup_structure(0);
	}

	// setup tree structure for specified level
	// this function is called recursively
	arma::uword NodeLevel::setup_structure(const arma::uword ilevel){
		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// set level
		level_ = ilevel;

		// dimensions of the grid at this level
		grid_dim_ = 1llu << ilevel; // bitshift
		
		// allocate children's indices
		arma::Mat<arma::uword> child_morton_indices;

		// allocate children's indices
		arma::Mat<arma::uword> child_num_sources;
		arma::Mat<arma::uword> child_num_targets;

		// create next level untill number of levels in grid reached
		if(ilevel < grid_->get_num_levels()){
			// increment depth of the tree by creating a new level
			next_level_ = NodeLevel::create(this); 

			// call this method on next level
			num_below_ = next_level_->setup_structure(ilevel+1);

			// get morton indices from next level
			child_morton_indices = next_level_->get_morton();		

			// bitshift to get the morton index of the parent
			child_morton_indices = cmn::Extra::bitshift(child_morton_indices,-3);

			// get number of sources
			child_num_sources = next_level_->get_num_sources();
			child_num_targets = next_level_->get_num_targets();
		}

		// this is a leaf
		else{
			// get morton indices from the grid
			child_morton_indices = grid_->get_morton();

			// get number of sources
			child_num_sources = grid_->get_is_source();
			child_num_targets = grid_->get_is_target();

			// no levels below
			num_below_ = 0;
		}

		// find first and last child
		const arma::Mat<arma::uword> idx = cmn::Extra::find_sections(child_morton_indices);
		first_child_ = idx.row(0); last_child_ = idx.row(1);
		
		// for non leaf levels 
		// set parent index of next level
		if(ilevel < grid_->get_num_levels()){
			next_level_->set_parent(cmn::Extra::set_sections(idx));
		}

		// for leaf level
		// set parent index in grid
		// else{
		// 	grid_->set_parent(cmn::Extra::set_sections(idx));
		// }

		// assemble morton indices for this level
		morton_index_ = child_morton_indices.cols(first_child_);
		
		// number of nodes in this level
		num_nodes_ = morton_index_.n_cols;

		// allocate
		num_sources_.set_size(1,num_nodes_);
		num_targets_.set_size(1,num_nodes_);

		// walk over nodes
		for(arma::uword i=0;i<num_nodes_;i++){
			// note square brackets for no out of range check
			// count sources
			num_sources_(i) = arma::accu(child_num_sources.cols(first_child_[i],last_child_[i]));

			// count targets
			num_targets_(i) = arma::accu(child_num_targets.cols(first_child_[i],last_child_[i]));
		}

		// find and index source and target nodes
		source_list_ = arma::find(num_sources_>0).t();
		target_list_ = arma::find(num_targets_>0).t();
		num_source_nodes_ = source_list_.n_elem;
		num_target_nodes_ = target_list_.n_elem;

		// timer
		last_structure_time_ = timer.toc();

		// return number of levels below
		return num_below_ + 1;
	}

	// Locations
	void NodeLevel::setup_node_locations(){
		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// allocate
		grid_location_.zeros(3,num_nodes_);
		
		// root level
		if(!is_root()){
			// get location from parent
			arma::Mat<arma::uword> parent_location = prev_level_->get_grid_location();
			
			// get nshift list from the interaction list
			arma::Mat<arma::uword> nshift = ilist_->get_type_nshift().t();

			// get type
			arma::Mat<arma::uword> type = cmn::Extra::modulus(morton_index_,8);
			
			// derive grid location from parent
			grid_location_ = cmn::Extra::bitshift(
				parent_location.cols(parent_),1) 
				+ nshift.cols(type);

			// check
			//grid_location_ = cmn::Extra::morton2grid(morton_index_);
		}

		// calculate node location
		fltp boxsize = grid_->get_box_size(level_);
		Rn_ = (arma::conv_to<arma::Mat<fltp> >::from(grid_location_)+RAT_CONST(0.5))*boxsize;
		Rn_.each_col() += grid_->get_position();

		// setup next levels
		if(!is_leaf()){
			next_level_->setup_node_locations();
		}

		// check with grid to see if elements are the same
		else{
			// check if the nodes calculated for the grid are 
			// the same as the ones calculated in this leaf-level
			const arma::Mat<fltp> Rn_grid_first = grid_->get_node_locations(first_child_);
			const arma::Mat<fltp> Rn_grid_last = grid_->get_node_locations(last_child_);
			assert(Rn_grid_first.n_cols==Rn_.n_cols); assert(Rn_grid_last.n_cols==Rn_.n_cols);
			assert(arma::all(arma::all(arma::abs(Rn_grid_first - Rn_)<RAT_CONST(1e-9),0)));
			assert(arma::all(arma::all(arma::abs(Rn_grid_last - Rn_)<RAT_CONST(1e-9),0)));
		}

		// timer
		last_node_time_ = timer.toc();
	}

	// Setup interaction lists recursively
	void NodeLevel::setup_interaction_lists(){
		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// make sure this is not called on leaf level
		assert(!is_leaf());

		// root node is only its own neighbour
		// this is manually set
		if(is_root()){
			// direct interaction list
			direct_interactions_.set_size(1,1);
			direct_interactions_(0).zeros(1);

			// approximate interaction list
			approx_interactions_.set_size(1,1);
			approx_interactions_(0).set_size(0);

			// only one interaction with itself
			num_direct_.ones(1,1);
			num_direct_found_.ones(1,1);

			// no approximate interaction with itself
			num_approx_.zeros(1,1);
			num_approx_found_.zeros(1,1);
		}

		// allocate direct interaction lists in next level
		next_level_->direct_interactions_.set_size(1,next_level_->num_nodes_);
		
		// allocate counters for number of direct interactions for next level
		next_level_->num_direct_.zeros(1,next_level_->num_nodes_);
		next_level_->num_direct_found_.zeros(1,next_level_->num_nodes_);

		// allocate approximate interaction list in next level
		next_level_->approx_interactions_.set_size(1,next_level_->num_nodes_);
		next_level_->approx_type_.set_size(1,next_level_->num_nodes_);

		// allocate counters for number of approximate interactions for next level
		next_level_->num_approx_.zeros(1,next_level_->num_nodes_);
		next_level_->num_approx_found_.zeros(1,next_level_->num_nodes_);

		// get direct interaction shift from the interaction list
		// in contrast to the approximate nshift list this does not change
		// with the type of the node
		const arma::Mat<arma::sword> direct_nshift = ilist_->get_direct_nshift().t();
		const arma::Mat<arma::uword> direct_shift_type = ilist_->get_direct_shift_type().t();

		// create atomic vectors for counting the number of times a node was
		// found as a direct neighbour and as an approximate neighbour
		// this part can not be done using armadillo
		#ifndef NDEBUG
		std::vector<std::atomic<arma::uword> > num_direct_found(next_level_->num_nodes_);
		std::fill(num_direct_found.begin(), num_direct_found.end(), 0);
		std::vector<std::atomic<arma::uword> > num_approx_found(next_level_->num_nodes_);
		std::fill(num_approx_found.begin(), num_approx_found.end(), 0);
		#endif

		// walk over nodes in this level
		// this can be parallel
		// (would need atomic add for 
		// num_direct_found_ debugging array
		// or remove this array completely) 
		// (possibly only run this routine on sources)
		// second int is CPU number
		// for(arma::uword i=0;i<num_nodes_;i++){
		cmn::parfor(0, num_nodes_,stngs_->get_parallel_tree_setup(),[&](arma::uword i, int) {
		      //printf("task %d running on cpu %d\n", i, cpu);
			// allocate child list
			arma::field<arma::Mat<arma::uword> > direct_child_list_fld(1,num_direct_(i));
			arma::field<arma::Mat<arma::uword> > direct_child_indexes_fld(1,num_direct_(i));

			// walk over neighbours
			for(arma::uword j=0;j<num_direct_(i);j++){
				// index of the neighbour node
				const arma::uword k = direct_interactions_(i)(j);

				// add grid index of children to direct child list
				direct_child_list_fld(0,j) = 
					next_level_->grid_location_.cols(first_child_(k),last_child_(k));
				
				// add indexes of children to index list
				//direct_child_indexes_fld(0,j) = 
				//	arma::conv_to<arma::Mat<arma::uword> >::from(
				//		arma::regspace(first_child_(k),last_child_(k)).t());
				direct_child_indexes_fld(0,j) = arma::regspace
					<arma::Row<arma::uword> >(first_child_(k),last_child_(k));
			}

			// convert field array to matrix
			// it is not important which parent supplied these children
			arma::Mat<arma::uword> direct_child_list = 
				cmn::Extra::field2mat(direct_child_list_fld);
			arma::Mat<arma::uword> direct_child_indexes = 
				cmn::Extra::field2mat(direct_child_indexes_fld);

			// reset to free up memory
			direct_child_list_fld.reset(); // free up memory
			direct_child_indexes_fld.reset();

			// may want to consider sorting the direct_child_list
			// this allows for even faster searches?
			// arma::Row<arma::uword> sorting_index = 
			//  	cmn::Extra::matrix_sort_index_by_row(direct_child_list);
			// direct_child_list = direct_child_list.cols(sorting_index);
			// direct_child_indexes = direct_child_indexes.cols(sorting_index);

			// walk over child nodes
			for(arma::uword k=first_child_(i);k<=last_child_(i);k++){
				// get my box type
				const arma::uword mytype = next_level_->morton_index_(k)%8;

				// approximate nshift for this child node
				const arma::Mat<arma::sword> approx_nshift = 
					ilist_->get_approx_nshift(mytype).t();

				// get corresponding shift type
				const arma::Mat<arma::uword> approx_shift_type = 
					ilist_->get_approx_shift_type(mytype).t();

				// get grid location of this children
				// convert to signed integer to make compatible with 
				// direct and approximate nshift list
				const arma::Mat<arma::sword> mylocation = 
					arma::conv_to<arma::Mat<arma::sword> >::from(
					next_level_->grid_location_.col(k));

				// making use of same search algorithm 
				// for direct and approximate interactions
				// interaction type: 0=direct, 1=approx
				for(arma::uword interaction_type=0;interaction_type<=1;interaction_type++){
					// when only the S2T part of the MLFMM is used
					// it is not necessary to create the approximate
					// interaction list and thus we skip this loop
					// iteration
					if(stngs_->get_enable_fmm()==false && interaction_type==1)break;

					// allocate search list
					arma::Mat<arma::sword> search_list;
					arma::Row<arma::uword> search_list_shift_type;

					// generate list of possible locations to search for
					// direct interactions
					if(interaction_type==0){
						search_list = direct_nshift.each_col() + mylocation;
						search_list_shift_type = direct_shift_type;
					}

					// generate list of possible locations to search for
					// approximate interactions
					else if(interaction_type==1){
						search_list = approx_nshift.each_col() + mylocation;
						search_list_shift_type = approx_shift_type;
					}

					// thin out search list by removing locations 
					// outside the grid (of the next level)
					const arma::Mat<arma::uword> inside_grid = 
						arma::find(arma::all(search_list>=0 && 
						search_list<next_level_->grid_dim_,0));
					search_list = search_list.cols(inside_grid);
					search_list_shift_type = search_list_shift_type.cols(inside_grid);

					// allocate direct interaction list for this child
					// will resize later to correct size
					if(interaction_type==0){
						next_level_->direct_interactions_(k).set_size(search_list.n_cols);
					}

					// allocate approximate interaction list for this child
					// will resize later to correct size
					else if(interaction_type==1){
						next_level_->approx_interactions_(k).set_size(search_list.n_cols);
						next_level_->approx_type_(k).set_size(search_list.n_cols);
					}

					// find possible indexes (if the node exists) 
					// of the locations in the search list and 
					// store this node in direct interaction list
					// This "for-loop" can possibly be circumvented 
					// using vector math: to be continued.
					for(arma::uword m=0;m<search_list.n_cols;m++){
						// get my index
						const arma::Col<arma::uword>::fixed<3> myidx = 
							arma::conv_to<arma::Mat<arma::uword> >::from(search_list.col(m));

						// find matching grid location in direct child list
						// (for optimization it would be possible here to only search
						// in the parent's neighbour that is relevant. This is, however,
						// rather tricky for implementation)
						arma::Row<arma::uword> idx_match = arma::find(direct_child_list.row(0) == myidx(0)).t();
						arma::Col<arma::uword>::fixed<1> idx_row; idx_row(0) = 1;
						if(idx_match.n_cols>0){
							idx_match = idx_match(arma::find(
								direct_child_list.submat(idx_row,idx_match) == myidx(1))).t();
						}
						if(idx_match.n_cols>0){
							idx_match = idx_match(arma::find(
								direct_child_list.submat(2*idx_row,idx_match) == myidx(2))).t();
						}

						// check if search resulted in new index
						if(idx_match.n_cols>0){
							// make sure it is only one index
							assert(idx_match.n_cols==1);

							// get type of interaction
							const arma::uword myinteractiontype = search_list_shift_type(m);

							// get corresponding node index
							const arma::uword idx_matching_node = 
								direct_child_indexes(0,arma::as_scalar(idx_match));

							// check if this node number can exist
							assert(idx_matching_node<next_level_->num_nodes_);

							// store for direct interactions
							if(interaction_type==0){
								// find index of corresponding node
								next_level_->direct_interactions_(k)(
									next_level_->num_direct_(k)) = idx_matching_node;

								// increment storage counter
								next_level_->num_direct_(k)++;
								//next_level_->num_direct_found_(idx_matching_node)++; // <- this can not be done in parallel
								#ifndef NDEBUG
								num_direct_found[idx_matching_node]++;
								#endif
							}

							// store for approximate interactions
							else if(interaction_type==1){
								// check if node contains sources
								// this would mess up the checking algorithm with 
								// num_approx_found so only in non-debug mode
								#ifdef NDEBUG
								if((next_level_->num_sources_(idx_matching_node)>0) && 
									(next_level_->num_targets_(k)>0)){
								#endif

									// find index of corresponding node
									next_level_->approx_interactions_(k)(next_level_->num_approx_(k)) = idx_matching_node;
									next_level_->approx_type_(k)(next_level_->num_approx_(k)) = myinteractiontype;

									// increment storage counter
									next_level_->num_approx_(k)++;
									//next_level_->num_approx_found_(idx_matching_node)++; // <- this can not be done in parallel
									#ifndef NDEBUG
									num_approx_found[idx_matching_node]++; // atomic incrementation
									#endif

								#ifdef NDEBUG
								}
								#endif
							}
						}

					}

					// resize direct interaction list
					// (preserves elements) to save some memory
					next_level_->direct_interactions_(k).resize(
						next_level_->num_direct_(k));

					// resize approximate interaction list
					// (preserves elements) to save some memory
					next_level_->approx_interactions_(k).resize(
						next_level_->num_approx_(k));
					next_level_->approx_type_(k).resize(
						next_level_->num_approx_(k));
				}

			}
		//}
		});

		// copy from atomic into armadillo arrays
		#ifndef NDEBUG
		for (arma::uword i=0;i<next_level_->num_nodes_;i++){
			next_level_->num_direct_found_(i) = num_direct_found[i];
			next_level_->num_approx_found_(i) = num_approx_found[i];
		}
		#endif

		// sanity check
		assert(arma::as_scalar(arma::all(
			next_level_->num_direct_==next_level_->num_direct_found_,1)));
		assert(arma::as_scalar(arma::all(
			next_level_->num_approx_==next_level_->num_approx_found_,1)));

		// timer
		last_list_time_ = timer.toc();

		// call this method on next level
		// stop before leaf level
		if(!next_level_->is_leaf()){
			next_level_->setup_interaction_lists();
		}
	}

	// setup multipole to local lists
	void NodeLevel::setup_mp2lp_lists(){
		{
			// allocate multipole to localpole list as field
			arma::field<arma::Mat<arma::uword> > 
				fast_mp2lp_list_field(next_level_->num_nodes_,3);

			// setup multipole to localpole list
			// first column contains interactoin type
			// second column contains source node
			// third column continas target node
			for(arma::uword i=0;i<next_level_->num_nodes_;i++){
				// interaction type
				fast_mp2lp_list_field(i,0) = next_level_->approx_type_(i);

				// source node
				fast_mp2lp_list_field(i,1) = next_level_->approx_interactions_(i);

				// clean list
				next_level_->approx_interactions_(i).reset();

				// target node
				fast_mp2lp_list_field(i,2) = i*arma::Col<arma::uword>(
					next_level_->num_approx_(i),arma::fill::ones);
			}
			
			// clean
			next_level_->approx_interactions_.reset();

			// stick field array together into a matrix
			next_level_->fast_mp2lp_list_ = 
				cmn::Extra::field2mat(fast_mp2lp_list_field);
		}

		// find column to sort
		arma::uword m2l_sorting_column = stngs_->get_m2l_sorting();

		// override to 2 in case gpu compiled and enabled
		#ifdef ENABLE_CUDA_KERNELS
		if(stngs_->get_fmm_enable_gpu() && stngs_->get_use_m2l_batch_sorting()){
			m2l_sorting_column = 2;
		}
		#else
		// check sorting column against available CPU kernels
		if(m2l_sorting_column!=0 && m2l_sorting_column!=2)
			rat_throw_line("m2l sorting column must be set to 0 or 2");
		#endif

		// sort column 0: is interaction type, 1 is source multipole, 2 is target localpole
		const arma::Mat<arma::uword> sorting_index = 
			arma::sort_index(next_level_->fast_mp2lp_list_.col(m2l_sorting_column),"ascend");
		next_level_->fast_mp2lp_list_ = 
			next_level_->fast_mp2lp_list_.rows(sorting_index);

		// filter non-source nodes and non target nodes from list
		next_level_->fast_mp2lp_list_ = next_level_->fast_mp2lp_list_.rows(
			arma::find(next_level_->num_sources_.cols(next_level_->fast_mp2lp_list_.col(1))>0 
			&& next_level_->num_targets_.cols(next_level_->fast_mp2lp_list_.col(2))>0));

		// setup indexing arrays into this list thereby
		// making sections of the same interaction type
		const arma::Mat<arma::uword> mp2lp_sections = cmn::Extra::find_sections(
			(next_level_->fast_mp2lp_list_.col(m2l_sorting_column)).t());
		
		// define sections
		next_level_->fast_mp2lp_list_first_ = mp2lp_sections.row(0);
		next_level_->fast_mp2lp_list_last_ = mp2lp_sections.row(1);

		// perform a special sort
		#ifdef ENABLE_CUDA_KERNELS
		if(stngs_->get_fmm_enable_gpu() && stngs_->get_use_m2l_batch_sorting()){
			// requires m2l_sorting column to be 2
			assert(m2l_sorting_column==2);

			// call batch sorting algorithm
			mp2lp_batch_sorting(); 
		}
		#endif

		// call this method on next level
		// stop before leaf level
		if(!next_level_->is_leaf()){
			next_level_->setup_mp2lp_lists();
		}
	}

	// recursive call to number nodes
	void NodeLevel::setup_multipoles(){

		// call this method on next level
		if(!is_leaf()){
			next_level_->setup_multipoles();
		}

		// allocate indices to zeros
		multipole_index_.zeros(1,num_nodes_);
		localpole_index_.zeros(1,num_nodes_);
		
		// set multipole indices (deliberately start from 1, zero means it does not exist)
		multipole_index_.cols(source_list_) = 
			arma::regspace<arma::Row<arma::uword> >(1,num_source_nodes_);

		// set localpole indices (deliberately start from 1, zero means it does not exist)
		localpole_index_.cols(target_list_) = 
			arma::regspace<arma::Row<arma::uword> >(1,num_target_nodes_);
		
		// calculate multipole size
		// arma::uword multipole_size = Extra::polesize(stngs_->get_num_exp());
		// arma::uword number_dimensions = grid_->get_num_dim();

		// allocate multipole data
		//Mp_.set_size(multipole_size, number_dimensions*num_source_nodes_);
		//Lp_.set_size(multipole_size, number_dimensions*num_target_nodes_);
	}

	// run full multipole method
	void NodeLevel::run_mlfmm(const cmn::ShLogPr &lg){
		// header
		lg->msg(2,"%sRunning MLFMM%s\n",KBLU,KNRM);

		// create armadillo timer
		arma::wall_clock timer;

		// set timer
		timer.tic();
		
		// check for cancel
		if(lg->is_cancelled())return;

		// allocate output
		grid_->fmm_setup();

		// check for cancel
		if(lg->is_cancelled())return;
		
		// upward pass
		mlfmm_upward_pass(lg);

		// downward pass
		mlfmm_downward_pass(lg);

		// check for cancel
		if(lg->is_cancelled())return;

		// allocate output
		grid_->fmm_post();

		// get time
		const fltp fmm_time = timer.toc();

		// report statistics
		lg->msg("time used: %s%.2f [s]%s\n",KYEL,fmm_time,KNRM);
		lg->msg(-2,"\n");
	}

	// multipole method upward pass
	void NodeLevel::mlfmm_upward_pass(const cmn::ShLogPr &lg){
		// check for cancel
		if(lg->is_cancelled())return;

		// recursive call to next level
		// by doing this first an upward pass is created
		if(!is_leaf()){
			next_level_->mlfmm_upward_pass(lg);
			if(lg->is_cancelled())return;
		}

		// for leaf level 
		else{
			// source to target operation
			if(stngs_->get_enable_s2t()){
				// fire thread that runs fully independently from MLFMM
				if(stngs_->get_split_s2t()){
					// message
					lg->msg("level %s%02llu%s->%s%02llu%s - source 2 target...... %s(S2T) (thread)%s\n",
						KYEL,level_,KNRM,KYEL,level_,KNRM,KGRN,KNRM);
					
					// create thread
					thread_so2ta_exceptions_ = NULL;
					thread_so2ta_ = std::thread([&](){
						// run source to target
						try{
							source_to_target(lg);
						}catch(...){
							thread_so2ta_exceptions_ = std::current_exception();
						}
					});

					// check for cancel
					if(lg->is_cancelled())return;
				}

				// run on main thread
				else{
					// message
					lg->msg("level %s%02llu%s->%s%02llu%s - source 2 target...... %s(S2T)%s\n",
						KYEL,level_,KNRM,KYEL,level_,KNRM,KGRN,KNRM);

					// run source to target
					source_to_target(lg);

					// check for cancel
					if(lg->is_cancelled())return;
				}
			}

			// setup multipoles from sources
			// std::cout<<"S2M"<<std::endl;
			if(stngs_->get_enable_fmm()){
				// report in log
				lg->msg("level %s%02llu%s->%s%02llu%s - source 2 multipole... %s(S2M)%s\n",
						KYEL,level_,KNRM,KYEL,level_,KNRM,KGRN,KNRM);

				// create multipoles for this level
				allocate_multipoles();

				// calculate source to multipole
				source_to_multipole();

				// check for cancel
				if(lg->is_cancelled())return;
			}
		}

		// transform multipoles in this level to localpoles
		// there is no approximate interactions below level 2
		if(level_>1){
			if(stngs_->get_enable_fmm()){
				
				// allocate
				allocate_localpoles();

				// split thread here
				if(stngs_->get_split_m2l()){
					// report in log
					lg->msg("level %s%02llu%s->%s%02llu%s - multipole 2 localpole %s(M2L) (thread)%s\n",
						KYEL,level_,KNRM,KYEL,level_,KNRM,KGRN,KNRM);

					// create multipole to localpole thread
					thread_mp2lp_exceptions_ = NULL;
					thread_mp2lp_ = std::thread([&](){
						// run source to target
						try{
							// call multipole to localpole kernel
							multipole_to_localpole();
						}catch(...){
							thread_mp2lp_exceptions_ = std::current_exception();
						}
					});

					// check for cancel
					if(lg->is_cancelled())return;
				}else{
					// report in log
					lg->msg("level %s%02llu%s->%s%02llu%s - multipole 2 localpole %s(M2L)%s\n",
						KYEL,level_,KNRM,KYEL,level_,KNRM,KGRN,KNRM);

					// calculate multipole to localpole
					multipole_to_localpole();

					// check for cancel
					if(lg->is_cancelled())return;
				}
			}
		}

		// send multipoles to parent
		if(!is_root() && level_>2){
			// std::cout<<"M2M_"<<level_<<"p"<<std::endl;
			if(stngs_->get_enable_fmm()){
				// report in log
				lg->msg("level %s%02llu%s->%s%02llu%s - multipole 2 multipole %s(M2M)%s\n",
					KYEL,level_,KNRM,KYEL,level_-1,KNRM,KGRN,KNRM);

				// allocate multipoles for this level
				prev_level_->allocate_multipoles();

				// calculate multipole to multipole step
				multipole_to_multipole();

				// check for cancel
				if(lg->is_cancelled())return;
			}
		}
	}

	// multipole method downward pass
	void NodeLevel::mlfmm_downward_pass(const cmn::ShLogPr &lg){
		// check for cancel
		if(lg->is_cancelled()){
			// do so as well for next level
			if(next_level_!=NULL)next_level_->mlfmm_downward_pass(lg);

			// join dangling threads
			if(thread_mp2lp_.joinable()){
				thread_mp2lp_.join();
				if(thread_mp2lp_exceptions_)
					std::rethrow_exception(thread_mp2lp_exceptions_);
			}
			if(thread_so2ta_.joinable()){
				thread_so2ta_.join();
				if(thread_so2ta_exceptions_)
					std::rethrow_exception(thread_so2ta_exceptions_);
			}

			// deallocate
			deallocate_multipoles();
			deallocate_localpoles();

			// the the fast way out
			return;
		}

		// join multipole to localpole thread
		if(level_>1){
			if(stngs_->get_enable_fmm()){
				if(stngs_->get_split_m2l()){
					// report in log
					lg->msg("level %s%02llu%s->%s%02llu%s - multipole 2 localpole %s(M2L) (join)%s\n",
							KYEL,level_,KNRM,KYEL,level_,KNRM,KGRN,KNRM);

					// join multipole to localpole thread
					thread_mp2lp_.join();
					if(thread_mp2lp_exceptions_)
						std::rethrow_exception(thread_mp2lp_exceptions_);
				}
			}
		}

		// acquire localpoles from parent
		if(!is_root() && level_>2){
			// std::cout<<"L2L_"<<level_<<"p"<<std::endl;
			if(stngs_->get_enable_fmm()){
				// report in log
				lg->msg("level %s%02llu%s->%s%02llu%s - localpole 2 localpole %s(L2L)%s\n",
					KYEL,level_-1,KNRM,KYEL,level_,KNRM,KGRN,KNRM);

				// calculate
				localpole_to_localpole();

				// deallocate multipoles and localpoles of this level
				prev_level_->deallocate_multipoles();
				prev_level_->deallocate_localpoles();

				// check for cancel
				if(lg->is_cancelled())return;
			}
		}

		// recursive call to next level
		// by doing this last a downward pass is created
		if(!is_leaf()){
			next_level_->mlfmm_downward_pass(lg);
		}

		// for leaf node
		else{
			// join source to target
			// this joining happens before localpole
			// to target to avoid the requirement for a 
			// mutex lock inside the targets
			// std::cout<<"S2T_join"<<std::endl;
			if(stngs_->get_enable_s2t()){
				if(stngs_->get_split_s2t()){
					// report in log
					lg->msg("level %s%02llu%s->%s%02llu%s - source 2 target...... %s(S2T) (join)%s\n",
						KYEL,level_,KNRM,KYEL,level_,KNRM,KGRN,KNRM);

					// join source to target thread
					thread_so2ta_.join();
					if(thread_so2ta_exceptions_)
						std::rethrow_exception(thread_so2ta_exceptions_);

					// check for cancel
					if(lg->is_cancelled())return;
				}
			}

			// run localpole to target
			// std::cout<<"L2T"<<std::endl;
			if(stngs_->get_enable_fmm()){
				// report in log
				lg->msg("level %s%02llu%s->%s%02llu%s - localpole 2 target... %s(L2T)%s\n",
					KYEL,level_,KNRM,KYEL,level_,KNRM,KGRN,KNRM);

				// run localpole to target
				localpole_to_target();
				
				// deallocate poles
				deallocate_multipoles();
				deallocate_localpoles();

				// check for cancel
				if(lg->is_cancelled())return;
			}

			// once all the dust has settled down
			// check for not-a-numbers
			// assert(!field->has_nan());
		}
		
	}

	// set all multipoles and localpoles
	// stored in this level to zero
	void NodeLevel::allocate_multipoles(){
		// calculate size of multipoles
		arma::uword multipole_size = Extra::polesize(stngs_->get_num_exp());
		arma::uword number_dimensions = grid_->get_num_dim();

		// allocate multipoles to zeros
		Mp_.zeros(multipole_size, number_dimensions*num_source_nodes_); 
		//Lp_.zeros(multipole_size, number_dimensions*num_target_nodes_);
	}

	// set all multipoles and localpoles
	// stored in this level to zero
	void NodeLevel::deallocate_multipoles(){
		// deallocate multipoles
		if(!stngs_->get_persistent_memory()){
			Mp_.reset();
		}
	}

	// set all multipoles and localpoles
	// stored in this level to zero
	void NodeLevel::allocate_localpoles(){
		// calculate size of multipoles
		arma::uword multipole_size = Extra::polesize(stngs_->get_num_exp());
		arma::uword number_dimensions = grid_->get_num_dim();

		// allocate multipoles to zeros
		//Mp_.zeros(multipole_size, number_dimensions*num_source_nodes_); 
		Lp_.zeros(multipole_size, number_dimensions*num_target_nodes_);
	}

	// set all multipoles and localpoles
	// stored in this level to zero
	void NodeLevel::deallocate_localpoles(){
		// deallocate multipoles
		if(!stngs_->get_persistent_memory()){
			Lp_.reset();
		}
	}

	// recursive deallocation
	void NodeLevel::deallocate_recursively(){
		// deallocate both multipoles and localpoles
		deallocate_multipoles(); deallocate_localpoles();

		// call on next level
		if(!is_leaf())next_level_->deallocate_recursively();
	}

	// source to multipole step
	void NodeLevel::source_to_multipole(){
		// run for leaf level
		assert(is_leaf());
		assert(!Mp_.is_empty());
		assert(arma::all(arma::all(Mp_==0)));

		// get list
		//const arma::Mat<arma::uword> so2mp_list = source_to_multipole_list();

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// calculate in leaf
		grid_->source_to_multipole(Mp_);

		// get time
		last_So2Mp_time_ = timer.toc();
	}

	// multipole to multipole step
	void NodeLevel::multipole_to_multipole(){
		// make sure this is not called on root level
		// as it does not have a previous level
		assert(!is_root());

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// ensure previous level and its multipoles exist
		assert(prev_level_!=NULL);
		assert(!prev_level_->Mp_.is_empty());
		assert(arma::all(arma::all(prev_level_->Mp_==0)));

		// number of dimensions
		const arma::uword number_dimensions = grid_->get_num_dim();

		// get matrix for performing multipole to multipole operation
		const ShFmmMat_Mp2MpPr& M = grid_->get_multipole_to_multipole_matrix();

		// calculate type
		const arma::Row<arma::uword> type = cmn::Extra::modulus(morton_index_,8);

		// use an atomic bool to avoid race conditions on the columns of the localpoles
		std::vector<std::mutex> col_locks(prev_level_->Mp_.n_cols/number_dimensions);
		
		// // initialize each column lock to false
		// for(auto &lock : col_locks)lock.store(false, std::memory_order_relaxed);


		// walk over child box-types and perform multipole
		// to multipole translation for each type
		// for(arma::uword k=0;k<8;k++){
		cmn::parfor(0,8,stngs_->get_parallel_m2m(),[&](arma::uword k, int) { // second int is CPU number
			// find necessary indices
			const arma::Row<arma::uword> indices = 
				arma::find(type==k && num_sources_>0).t();

			// check if any nodes found that can be translated with this matrix
			if(!indices.is_empty()){
				// check indexes
				assert(arma::as_scalar(arma::all(indices>=0,1)));
				assert(arma::as_scalar(arma::all(indices<num_nodes_,1)));

				// multipole indices
				const arma::Row<arma::uword> multipole_indices = multipole_index_.cols(indices);

				// get indices of corresponding parent
				const arma::Row<arma::uword> parent_indices = parent_.cols(indices);

				// get multipole indices of corresponding parent
				const arma::Row<arma::uword> parent_multipole_indices = 
					prev_level_->multipole_index_.cols(parent_indices);

				// check that indices fall within range
				assert(arma::as_scalar(arma::all(multipole_indices>0,1)));
				assert(arma::as_scalar(arma::all(multipole_indices<=num_source_nodes_,1)));

				// check that parent indices fall within range
				assert(arma::as_scalar(arma::all(parent_multipole_indices>0,1)));
				assert(arma::as_scalar(arma::all(parent_multipole_indices<=prev_level_->num_source_nodes_,1)));
				
				// expand indices
				// note that here we shift multipole index by -1 to 
				// get back to original location
				const arma::Row<arma::uword> expanded_multipole_indices = 
					fmm::Extra::expand_indices(multipole_indices-1,number_dimensions);
				const arma::Row<arma::uword> expanded_parent_multipole_indices = 
					fmm::Extra::expand_indices(parent_multipole_indices-1,number_dimensions);

				// perform multipole translation (num_below_+1) times
				const arma::Mat<std::complex<fltp> > Mpcon = 
					M->apply(k,Mp_.cols(expanded_multipole_indices),num_below_);

				// add the contribution to the localpole 
				for(arma::uword i=0;i<parent_multipole_indices.n_elem;i++) {
					// get column index
					const arma::uword parent_multipole_index = parent_multipole_indices(i)-1;
					assert(parent_multipole_index<prev_level_->Mp_.n_cols/number_dimensions);
					assert(i<Mpcon.n_cols/number_dimensions);

					// update the multipole using a mutex lock
					{
						// acquire the lock for this column
						const std::lock_guard<std::mutex> lock_guard(col_locks[parent_multipole_index]);

						// update the column
						prev_level_->Mp_.cols(parent_multipole_index*number_dimensions,(parent_multipole_index+1)*number_dimensions-1) += 
							Mpcon.cols(i*number_dimensions,(i+1)*number_dimensions-1);
					}
				}
			}
		//}
		});

		// check for nans
		assert(!prev_level_->Mp_.has_nan());

		// get time
		last_Mp2Mp_time_ = timer.toc();
	}


	// multipole to localpole 
	// this method selects the most appropriate kernel
	void NodeLevel::multipole_to_localpole(){
		// check if cuda enabled
		#ifdef ENABLE_CUDA_KERNELS
			if(stngs_->get_fmm_enable_gpu()){
				// call method
				multipole_to_localpole_cuda();
				
				// do not continue
				return;
			}
		#endif

		// mutipole to localpole sorted by type
		if(stngs_->get_m2l_sorting()==0){
			multipole_to_localpole_by_type();
		}

		// mutipole to localpole sorted by target
		else if(stngs_->get_m2l_sorting()==2){	
			multipole_to_localpole_by_target();
		}

		// mutipole to localpole sorted by source
		else{
			// by source not available
			assert(0);
		}
	}



	// multipole to localpole step basic kernel
	void NodeLevel::multipole_to_localpole_by_type(){
		// no interactions for root level and level 1
		assert(level_>1);

		// check if allocated
		assert(!Mp_.is_empty());
		assert(!Lp_.is_empty());
		assert(arma::all(arma::all(Lp_==0)));

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// number of dimensions
		const arma::uword number_dimensions = grid_->get_num_dim();

		// get matrix from grid
		const ShFmmMat_Mp2LpPr &M = grid_->get_multipole_to_localpole_matrix();

		// get distance matrix for current level
		FmmMat_Mp2Lp_dist Mdist;
		Mdist.set_num_exp(stngs_->get_num_exp());
		Mdist.setup(num_below_);

		// use an atomic bool to avoid race conditions on the columns of the localpoles
		std::vector<std::mutex> col_locks(Lp_.n_cols/number_dimensions);

		// run over sections in fast interaction list
		// each section represents one type of transformation
		// for(arma::uword k=0;k<fast_mp2lp_list_first_.n_cols;k++){
		cmn::parfor(0,fast_mp2lp_list_first_.n_cols,stngs_->get_parallel_m2l(),
			[&](arma::uword k, int) { // second int is CPU number
			// get indexes
			const arma::uword first_index = fast_mp2lp_list_first_(k);
			const arma::uword last_index = fast_mp2lp_list_last_(k);
			
			// get my type
			const arma::uword mytype = fast_mp2lp_list_(first_index,0);

			// all types should be the same
			assert(arma::as_scalar(arma::all(fast_mp2lp_list_.submat(first_index,0,last_index,0)==mytype)));

			// get source and target nodes
			const arma::Row<arma::uword> source_node_indices = 
				(fast_mp2lp_list_.submat(arma::span(first_index,last_index),arma::span(1,1))).t();
			const arma::Row<arma::uword> target_node_indices = 
				(fast_mp2lp_list_.submat(arma::span(first_index,last_index),arma::span(2,2))).t();

			// check if in range
			assert(arma::as_scalar(arma::all(source_node_indices>=0,1)));
			assert(arma::as_scalar(arma::all(source_node_indices<num_nodes_,1)));
			assert(arma::as_scalar(arma::all(target_node_indices>=0,1)));
			assert(arma::as_scalar(arma::all(target_node_indices<num_nodes_,1)));

			// get multipole indices
			const arma::Row<arma::uword> source_multipole_indices = 
				multipole_index_.cols(source_node_indices);
			const arma::Row<arma::uword> target_localpole_indices = 
				localpole_index_.cols(target_node_indices);

			// check if in range
			assert(arma::as_scalar(arma::all(source_multipole_indices>0,1)));
			assert(arma::as_scalar(arma::all(source_multipole_indices<=num_source_nodes_,1)));
			assert(arma::as_scalar(arma::all(target_localpole_indices>0,1)));
			assert(arma::as_scalar(arma::all(target_localpole_indices<=num_target_nodes_,1)));

			// expand index arrays
			const arma::Row<arma::uword> source_multipole_indices_expanded = 
				fmm::Extra::expand_indices(source_multipole_indices-1,number_dimensions);
			const arma::Row<arma::uword> target_localpole_indices_expanded = 
				fmm::Extra::expand_indices(target_localpole_indices-1,number_dimensions);

			// perform multipole to localpole tranformation with matrix
			// const arma::Mat<std::complex<fltp> > Lpcon = M->apply(
			// 	mytype,Mp_.cols(source_multipole_indices_expanded),Mdist);
			const arma::Mat<std::complex<fltp> > Md = Mdist.get_matrix()%M->get_matrix(mytype);
			const arma::Mat<std::complex<fltp> > Lpcon = Md*Mp_.cols(source_multipole_indices_expanded); // this is a heavy gemm line!
			// we can investigate rotating the Mp before translation

			// add the contribution to the localpole 
			for(arma::uword i=0;i<target_localpole_indices.n_elem;i++) {
				// get column index
				const arma::uword target_idx = target_localpole_indices(i)-1;
				assert(target_idx<Lp_.n_cols/number_dimensions);
				assert(i<Lpcon.n_cols/number_dimensions);

				// update the localpole using a mutex lock
				{
					const std::lock_guard<std::mutex> lock_guard(col_locks[target_idx]);

					// update the column
					Lp_.cols(target_idx*number_dimensions,(target_idx+1)*number_dimensions-1) += 
						Lpcon.cols(i*number_dimensions,(i+1)*number_dimensions-1);
				}
			}
		});

		// check for nans
		assert(!Lp_.has_nan());

		// get time
		last_Mp2Lp_time_ = timer.toc();
	}

	// multipole to localpole
	// alternative kernel that avoids locking
	// note that the m2l lists are also organised differently
	// (see NodeLevel::setup_interaction_lists())
	void NodeLevel::multipole_to_localpole_by_target(){
		// no interactions for root level and level 1
		assert(level_>1);

		// check if allocated
		assert(!Mp_.is_empty());
		assert(!Lp_.is_empty());
		assert(arma::all(arma::all(Lp_==0)));
		
		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// get matrix from grid
		const ShFmmMat_Mp2LpPr& M = grid_->get_multipole_to_localpole_matrix();

		// get distance matrix for current level
		FmmMat_Mp2Lp_dist Mdist;
		Mdist.set_num_exp(stngs_->get_num_exp());
		Mdist.setup(num_below_);

		// number of dimensions
		const arma::uword number_dimensions = grid_->get_num_dim();

		// run over sections in fast interaction list
		// each section represents one type of transformation
		// for(arma::uword k=0;k<fast_mp2lp_list_first_.n_cols;k++){
		cmn::parfor(0,fast_mp2lp_list_first_.n_cols,stngs_->get_parallel_m2l(),
			[&](arma::uword k, int) { // second int is CPU number
			// get indexes
			const arma::uword first_index = fast_mp2lp_list_first_(k);
			const arma::uword last_index = fast_mp2lp_list_last_(k);

			// walk over target nodes
			for(arma::uword j=first_index;j<=last_index;j++){
				// get my type
				const arma::uword mytype = fast_mp2lp_list_(j,0);

				// get my source node
				const arma::uword source_node = fast_mp2lp_list_(j,1);

				// get my target node
				const arma::uword target_node = fast_mp2lp_list_(j,2);

				// check if in range
				assert(source_node<num_nodes_);
				assert(target_node<num_nodes_);

				// get my source multipole idx
				const arma::uword source_multipole_index = multipole_index_(source_node);

				// get my target multipole idx
				const arma::uword target_multipole_index = localpole_index_(target_node);

				// check that indexes are not zero
				assert(source_multipole_index>0);
				assert(target_multipole_index>0);

				// get columns by manually expanding indexes
				const arma::uword tar_first_col = number_dimensions*(target_multipole_index-1);
				const arma::uword tar_last_col = tar_first_col + number_dimensions - 1;
				const arma::uword src_first_col = number_dimensions*(source_multipole_index-1);
				const arma::uword src_last_col = src_first_col + number_dimensions - 1;
				
				// add the contribution to the localpole 
				Lp_.cols(tar_first_col,tar_last_col) += M->apply(
					mytype,Mp_.cols(src_first_col,src_last_col),Mdist);
			}
		});

		// check for nans
		assert(!Lp_.has_nan());

		// get time
		last_Mp2Lp_time_ = timer.toc();
	}

	// localpole to localpole step
	void NodeLevel::localpole_to_localpole(){
		// make sure this is not called on root level
		// as it does not have a previous level
		assert(!is_root());

		// check if previous level exists
		assert(prev_level_!=NULL);
		assert(!Lp_.is_empty());
		assert(!prev_level_->Lp_.is_empty());

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// number of dimensions
		arma::uword number_dimensions = grid_->get_num_dim();

		// get matrix from grid
		ShFmmMat_Lp2LpPr M = grid_->get_localpole_to_localpole_matrix();

		// calculate type
		const arma::Row<arma::uword> type = cmn::Extra::modulus(morton_index_,8);

		// walk over box-types and perform localpole
		// to localpole translation for each type
		// this can be parallel
		// for(arma::uword k=0;k<8;k++){
		cmn::parfor(0,8,stngs_->get_parallel_l2l(),[&](arma::uword k, int) { // second int is CPU number
			// find necessary indices
			const arma::Row<arma::uword> indices = 
				arma::find(type==k && num_targets_>0).t();

			// check if any nodes found that can be translated with this matrix
			if(!indices.is_empty()){
				// check indexes
				assert(arma::as_scalar(arma::all(indices>=0,1)));
				assert(arma::as_scalar(arma::all(indices<num_nodes_,1)));

				// localpole indices
				const arma::Row<arma::uword> localpole_indices = localpole_index_.cols(indices);

				// get indices of corresponding parent
				const arma::Row<arma::uword> parent_indices = parent_.cols(indices);

				// get multipole indices of corresponding parent
				const arma::Row<arma::uword> parent_localpole_indices = 
					prev_level_->localpole_index_.cols(parent_indices);
			
				// check that indices fall within range
				assert(arma::as_scalar(arma::all(localpole_indices>0,1)));
				assert(arma::as_scalar(arma::all(localpole_indices<=num_target_nodes_,1)));
			
				// check that parent indices fall within range
				assert(arma::as_scalar(arma::all(parent_localpole_indices>0,1)));
				assert(arma::as_scalar(arma::all(parent_localpole_indices<=prev_level_->num_target_nodes_,1)));
				
				// expand indices
				// note that here we shift multipole index by -1 to 
				// get back to original location
				const arma::Row<arma::uword> expanded_localpole_indices = 
					fmm::Extra::expand_indices(localpole_indices-1,number_dimensions);
				const arma::Row<arma::uword> expanded_parent_localpole_indices = 
					fmm::Extra::expand_indices(parent_localpole_indices-1,number_dimensions);

				// perform localpole translation (num_below_+1) times
				Lp_.cols(expanded_localpole_indices) += 
					M->apply(k,prev_level_->Lp_.cols(expanded_parent_localpole_indices),num_below_);
			}
		});

		// check for nans
		assert(!Lp_.has_nan());

		// get time
		last_Lp2Lp_time_ = timer.toc();
	}

	// localpole to target step for vector potential
	void NodeLevel::localpole_to_target(){

		// run for leaf level
		assert(is_leaf());
		assert(!Lp_.is_empty());
		
		// get list
		//const arma::Mat<arma::uword> lp2ta_list = localpole_to_target_list();

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// run localpole to target at leaf
		grid_->localpole_to_target(Lp_);

		// get time
		last_Lp2Ta_time_ = timer.toc();
	}

	// source to target step for vector potential
	void NodeLevel::source_to_target(const cmn::ShLogPr &lg){
		// make sure this is called on a leaf node
		assert(is_leaf());

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// run source to target
		grid_->source_to_target(s2t_list_tar_,s2t_list_src_,lg);

		// get time
		last_So2Ta_time_ = timer.toc();
	}

	// setup source to target list
	void NodeLevel::setup_s2t_lists(){
		// make sure this is called on a leaf node
		assert(is_leaf());

		// find target nodes that have direct interactions
		arma::Row<arma::uword> idx_targets = 
			arma::find(num_direct_(target_list_)>0).t();
		
		// walk over indexes
		arma::Row<arma::uword> ns(idx_targets.n_elem,arma::fill::zeros);
		for(arma::uword i=0;i<idx_targets.n_elem;i++){
			// find my node index
			const arma::uword target_node = target_list_(idx_targets(i));

			// count number of sources in direct interaction
			for(arma::uword j=0;j<num_direct_(target_node);j++)
				ns(i)+=num_sources_(direct_interactions_(target_node)(j));
		}
		
		// remove target nodes that have no direct interactions
		idx_targets = idx_targets.cols(arma::find(ns>0));

		// allocate list
		s2t_list_tar_.set_size(idx_targets.n_elem);
		s2t_list_src_.set_size(idx_targets.n_elem);

		// walk over nodes and generate index lists
		for(arma::uword i=0;i<idx_targets.n_elem;i++){
			// this target node
			const arma::uword target_idx = idx_targets(i);

			// find my node index
			const arma::uword target_node = target_list_(target_idx);

			// check if there is indeed direct interactions
			assert(num_direct_(target_node)>0);

			// allocate source index list
			s2t_list_src_(i).set_size(num_direct_(target_node));

			// counter for number of sources
			arma::uword idx = 0;

			// walk over neighbour list
			for(arma::uword j=0;j<num_direct_(target_node);j++){
				// get source node from neighbour list
				const arma::uword source_node = direct_interactions_(target_node)(j);

				// check if in range
				assert(source_node<num_nodes_);
			
				// check if this is a source node
				if(num_sources_(source_node)>0){
					//source_indices_field(j) = separate_index.cols(first_source,last_source);
					s2t_list_src_(i)(idx) = multipole_index_(source_node)-1;

					// increment storage index
					idx ++;
				}
			}
			
			// check if there is at least one source
			assert(idx>0);

			// resize
			s2t_list_src_(i).resize(idx);

			// set array of targets
			s2t_list_tar_(i) = localpole_index_(target_node)-1;
		}
	}

	// access source to target lists
	const arma::field<arma::Col<arma::uword> >& NodeLevel::get_s2t_lists_src() const{
		return s2t_list_src_;
	}

	// get source to target lists
	const arma::Col<arma::uword>& NodeLevel::get_s2t_list_tar() const{
		return s2t_list_tar_;
	}

	// set parent index
	void NodeLevel::set_parent(const arma::Mat<arma::uword> &parent_index){
		assert(parent_index.n_rows==1);
		parent_ = parent_index;
	}

	// get morton indices
	const arma::Row<arma::uword>& NodeLevel::get_morton() const{
		return morton_index_;
	}

	// get number of sources in each node
	const arma::Row<arma::uword>& NodeLevel::get_num_sources() const{
		return num_sources_;
	}

	// get number of targets in each node
	const arma::Row<arma::uword>& NodeLevel::get_num_targets() const{
		return num_targets_;
	}

	// get number of targets in each node
	const arma::Row<arma::uword>& NodeLevel::get_first_child() const{
		return first_child_;
	}

	// get number of targets in each node
	const arma::Row<arma::uword>& NodeLevel::get_last_child() const{
		return last_child_;
	}

	// get number of source nodesz
	arma::uword NodeLevel::get_num_source_nodes() const{
		return num_source_nodes_;
	}

	// get number of source nodesz
	arma::uword NodeLevel::get_num_target_nodes() const{
		return num_target_nodes_;
	}


	// get previous level pointer
	ShNodeLevelPr NodeLevel::get_next_level(){
		return next_level_;
	}


	// get location of all nodes within the grid
	const arma::Mat<arma::uword>& NodeLevel::get_grid_location() const{
		// ensure that grid locations have been created
		assert(!grid_location_.is_empty());

		// return grid locations
		return grid_location_;
	}

	// get all coordinates of the nodes in this level
	const arma::Mat<fltp>& NodeLevel::get_node_coords() const{
		// make sure coordinates 
		// have been created properly
		assert(!Rn_.is_empty());

		// return coordinates
		return Rn_;
	}

	// method for getting pointer to leaf level
	const ShNodeLevelPr& NodeLevel::get_leaf()const{
		// if leaf return pointer to this level
		if(is_leaf())return prev_level_->next_level_;

		// make sure next level exists (i.e. if the tree is setup)
		assert(next_level_!=NULL);

		// for non leaf levels forward request to next level
		return next_level_->get_leaf();
	}

	// check if level is leaf level
	bool NodeLevel::is_leaf() const{
		return next_level_==NULL;
	}

	// check if level is leaf level
	bool NodeLevel::is_root() const{
		return level_==0;
	}

	// get number of nodes in this level
	arma::uword NodeLevel::get_num_nodes() const{
		return num_nodes_;
	}

	// get number of nodes in this level
	arma::uword NodeLevel::get_level() const{
		return level_;
	}
			
	// get number of nodes in this level
	arma::uword NodeLevel::get_num_below() const{
		return num_below_;
	}

	// get number of m2l interactions
	arma::uword NodeLevel::get_mp2lp_list_size() const{
		return fast_mp2lp_list_.n_rows;
	}

	// get multipole indexing array
	const arma::Row<arma::uword>& NodeLevel::get_multipole_index() const{
		return multipole_index_;
	}

	// Methods specifically written for cuda
	#ifdef ENABLE_CUDA_KERNELS

	// sorting of the multipole to localpole list
	// for ensuring that the output localpole is 
	// only referenced by a single matrix multiplication
	void NodeLevel::mp2lp_batch_sorting(){
		// get numbers
		arma::uword num_interactions = next_level_->fast_mp2lp_list_.n_rows;
		arma::uword num_groups = next_level_->fast_mp2lp_list_first_.n_elem;

		// create batch list
		arma::Col<arma::uword> idx_batch(num_interactions);

		// walk over groups with same target
		// and divide them over separate batches
		for(arma::uword i=0;i<num_groups;i++){
			// get section with same target localpole
			arma::uword first = next_level_->fast_mp2lp_list_first_(i);
			arma::uword last = next_level_->fast_mp2lp_list_last_(i);

			// check if this section contains only one localpole
			assert(arma::as_scalar(arma::all(
				next_level_->fast_mp2lp_list_.submat(arma::span(first,last),arma::span(2,2))==
				next_level_->fast_mp2lp_list_(first,2))));

			// fill batch
			idx_batch.rows(first,last) = arma::regspace<arma::Col<arma::uword> >(0,last-first);
		}

		// sort by batch
		const arma::Mat<arma::uword> batch_sorting_index = 
			arma::sort_index(idx_batch,"ascend");

		// sort list
		next_level_->fast_mp2lp_list_ = next_level_->fast_mp2lp_list_.rows(batch_sorting_index);

		// sort batch idx
		idx_batch = idx_batch.rows(batch_sorting_index);

		// setup indexing arrays into this list thereby
		// making sections of the same interaction type
		const arma::Mat<arma::uword> batch_sections = 
			cmn::Extra::find_sections(idx_batch.t());

		// define sections
		next_level_->fast_mp2lp_list_first_ = batch_sections.row(0);
		next_level_->fast_mp2lp_list_last_ = batch_sections.row(1);
	}

	// multipole to localpole for cuda
	void NodeLevel::multipole_to_localpole_cuda(){

		// no interactions for root level and level 1
		assert(level_>1);

		// check no interaction case
		if(num_target_nodes_==0llu)return;
		if(num_source_nodes_==0llu)return;

		// check if allocated
		assert(!Mp_.is_empty());
		assert(!Lp_.is_empty());
		assert(arma::all(arma::all(Lp_==0)));

		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// number of dimensions
		const arma::uword num_dim = grid_->get_num_dim();
		assert(num_dim>0llu);

		// number of expansions
		const int num_exp = stngs_->get_num_exp();
		assert(num_exp>0);

		// polesize
		const int polesize = stngs_->get_polesize();
		assert(polesize>0);

		// get distance matrix for current level
		FmmMat_Mp2Lp_dist Mdist;
		Mdist.set_num_exp(num_exp);
		Mdist.setup(num_below_);

		// get interaction matrix list from grid
		const ShFmmMat_Mp2LpPr M = grid_->get_multipole_to_localpole_matrix();
		const arma::uword num_int_type = M->get_num_mat();
		assert(num_int_type!=0llu);

		// concatenate interaction matrices
		// arma::wall_clock timer2; timer2.tic();
		arma::Mat<std::complex<cufltp> > Mint(polesize,polesize*num_int_type);
		for(arma::uword i=0;i<num_int_type;i++){
			#ifdef RAT_CUDA_CONVERSION_NEEDED
				// create matrix
				arma::Mat<std::complex<fltp> > dMint = Mdist.apply(M->get_matrix(i));

				// check limits
				dMint = arma::Mat<std::complex<fltp> >(
					arma::clamp(arma::real(dMint), -std::numeric_limits<cufltp>::max(), std::numeric_limits<cufltp>::max()),
					arma::clamp(arma::imag(dMint), -std::numeric_limits<cufltp>::max(), std::numeric_limits<cufltp>::max()));
				
				// insert into list with conversion
				Mint.cols(i*polesize,(i+1)*polesize-1) = 
					arma::conv_to<arma::Mat<std::complex<cufltp> > >::from(dMint);
			#else
				// insert into list without conversion
				Mint.cols(i*polesize,(i+1)*polesize-1) = Mdist.apply(M->get_matrix(i));
			#endif
		}
		
		// create pointer to localpole memory
		std::complex<cufltp> *Lp_ptr, *Mp_ptr;
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			arma::Mat<std::complex<cufltp> > Lp_single(Lp_.n_rows, Lp_.n_cols, arma::fill::zeros);
			arma::Mat<std::complex<cufltp> > Mp_single = 
				arma::conv_to<arma::Mat<std::complex<cufltp> > >::from(Mp_);
			Lp_ptr = Lp_single.memptr(); 
			Mp_ptr = Mp_single.memptr();
		#else
			Lp_ptr = Lp_.memptr(); Mp_ptr = Mp_.memptr();
		#endif

		// check input
		assert(!Mint.has_nan()); assert(Mint.is_finite()); 
		assert(fast_mp2lp_list_.is_finite());

		// run CUDA kernel
		GpuKernels::mp2lp_kernel(
			num_dim, // number of dimensions
			polesize, // size of the multipoles and localpoles
			Lp_ptr, // matrix with target localpoles
			num_target_nodes_, // number of localpoles
			Mp_ptr, // matrix with target multipoles
			num_source_nodes_, // number of multipoles
			Mint.memptr(), // matrix with transformation matrices
			num_int_type, // number of interaction types
			fast_mp2lp_list_.colptr(0), // interaction type for this index
			fast_mp2lp_list_.colptr(1), // source node index list
			fast_mp2lp_list_.colptr(2), // target node index list
			multipole_index_.memptr(),	// index at which the multipoles are stored
			localpole_index_.memptr(), // index at which the localpoles are stored
			fast_mp2lp_list_first_.memptr(), // group start (see sort type)
			fast_mp2lp_list_last_.memptr(),	// group end
			fast_mp2lp_list_first_.n_elem,
			stngs_->get_gpu_devices());	// number of groups

		// convert cuda typed localpoles back
		#ifdef RAT_CUDA_CONVERSION_NEEDED
			Lp_ = arma::conv_to<arma::Mat<std::complex<fltp> > >::from(Lp_single);
		#endif

		// check for nans
		assert(!Lp_.has_nan());

		// free memory
		// GpuKernels::free_pinned(Mp_single_ptr);
		// GpuKernels::free_pinned(Lp_single_ptr);

		// get time
		last_Mp2Lp_time_ = timer.toc();

		// done
		return;
	}
	#endif


	// display function
	void NodeLevel::display(const cmn::ShLogPr &lg) const{
		// calculate size of multipoles
		const arma::uword polesize = Extra::polesize(stngs_->get_num_exp());
		const arma::uword num_dim = grid_->get_num_dim();
		
		// calculate memory needed to store multipoles and localpoles
		const fltp fltp_size = fltp(sizeof(fltp)); // [byte]
		const fltp uword_size = fltp(sizeof(arma::uword)); // [byte]
		const fltp mp_matrix_memory = (2*num_source_nodes_*num_dim*polesize*fltp_size)/1024/1024;
		const fltp lp_matrix_memory = (2*num_target_nodes_*num_dim*polesize*fltp_size)/1024/1024;
		const fltp interaction_list_memory = (fast_mp2lp_list_.n_elem*uword_size)/1024/1024;
		const fltp interaction_batch_memory = (2*fast_mp2lp_list_first_.n_elem*uword_size)/1024/1024;

		// header for all levels
		if(is_root()){
			lg->msg(2,"%sOct-Tree Statistics per Level:%s\n",KBLU,KNRM);
		}

		// header
		if(is_root()){
			lg->msg(2,"%sstats for level %s%02llu/%02llu%s - %s(tree root)%s\n",
				KBLU, KYEL,level_,num_below_,KNRM,KGRN,KNRM);
		}
		else if(is_leaf()){
			lg->msg(2,"%sstats for level %s%02llu/%02llu%s - %s(tree leaf)%s\n",
				KBLU, KYEL,level_,num_below_,KNRM,KGRN,KNRM);
		}
		else{
			lg->msg(2,"%sstats for level %s%02llu/%02llu%s - %s(normal level)%s\n",
				KBLU, KYEL,level_,num_below_,KNRM,KGRN,KNRM);
		}

		// display statistics on this level
		lg->msg("number of nodes: %s%llu%s\n",KYEL,num_nodes_,KNRM);
		lg->msg("number of source nodes: %s%llu%s\n",KYEL,num_source_nodes_,KNRM);
		lg->msg("number of target nodes: %s%llu%s\n",KYEL,num_target_nodes_,KNRM);
		lg->msg("mp %sDENSE_CX%s matrix: %s%llu%s X %s%llu%s, memory: %s%.2f%s [MB]\n",
			KGRN,KNRM,KYEL,polesize,KNRM,KYEL,num_source_nodes_*num_dim,KNRM,KYEL,mp_matrix_memory,KNRM);
		lg->msg("lp %sDENSE_CX%s matrix: %s%llu%s X %s%llu%s, memory: %s%.2f%s [MB]\n",
			KGRN,KNRM,KYEL,polesize,KNRM,KYEL,num_target_nodes_*num_dim,KNRM,KYEL,lp_matrix_memory,KNRM);
		lg->msg("number of mp2lp interactions %s%llu%s, memory: %s%.2f%s [MB]\n",
			KYEL,fast_mp2lp_list_.n_rows,KNRM,
			KYEL,interaction_list_memory,KNRM);
		lg->msg("number of mp2lp batches %s%llu%s, memory: %s%.2f%s [MB]\n",
			KYEL,fast_mp2lp_list_first_.n_elem,KNRM,
			KYEL,interaction_batch_memory,KNRM);
		
		// display timings
		lg->msg("tree structure time: %s%.2f [ms]%s\n",KYEL,1e3*last_structure_time_,KNRM);
		lg->msg("node setup time: %s%.2f [ms]%s\n",KYEL,1e3*last_node_time_,KNRM);
		if(!is_leaf())lg->msg("interaction list setup time: %s%.2f [ms]%s\n",KYEL,1e3*last_list_time_,KNRM);
		if(is_leaf())lg->msg("source to multipole time: %s%.2f [ms]%s\n",KYEL,1e3*last_So2Mp_time_,KNRM);
		if(level_>2)lg->msg("multipole to multipole time: %s%.2f [ms]%s\n",KYEL,1e3*last_Mp2Mp_time_,KNRM);
		if(level_>=2)lg->msg("multipole to localpole time: %s%.2f [ms]%s\n",KYEL,1e3*last_Mp2Lp_time_,KNRM);
		if(level_>2)lg->msg("localpole to localpole time: %s%.2f [ms]%s\n",KYEL,1e3*last_Lp2Lp_time_,KNRM);
		if(is_leaf())lg->msg("localpole to target time: %s%.2f [ms]%s\n",KYEL,1e3*last_Lp2Ta_time_,KNRM);
		if(is_leaf())lg->msg("source to target time: %s%.2f [ms]%s\n",KYEL,1e3*last_So2Ta_time_,KNRM);

		// return
		lg->msg(-2,"\n");

		// call on next level
		if(!is_leaf()){
			next_level_->display(lg);
		}

		// return
		else{
			lg->msg(-2,"\n");
		}
	}

}}
