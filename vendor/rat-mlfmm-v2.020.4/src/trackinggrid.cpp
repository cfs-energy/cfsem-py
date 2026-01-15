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
#include "trackinggrid.hh"

// code specific to Rat
namespace rat{namespace fmm{


	// constructor
	TrackingGrid::TrackingGrid(){
		
	}

	// factory
	ShTrackingGridPr TrackingGrid::create(){
		//return ShIListPr(new IList);
		return std::make_shared<TrackingGrid>();
	}


	// setup function
	void TrackingGrid::setup(const cmn::ShLogPr &lg){
		// normal setup
		Grid::setup(lg);

		// clear any remaining map 
		morton2node_.clear();

		const arma::Row<arma::uword> target_morton_indices = morton_index_.cols(arma::find(element_type_<0));

		// create a map to relate morton indices to their respective node
		for(arma::uword i=0;i<first_target_.n_cols;i++){
			// std::cout<<target_morton_indices(first_target_(i))<<std::endl;
			morton2node_[target_morton_indices(first_target_(i))] = i;
		}
	}

	// set grid
	void TrackingGrid::setup_direct(){
		// just get grid size
		const arma::Col<fltp>::fixed<3> Rmax = arma::max(tar_->get_target_coords(),1);
		const arma::Col<fltp>::fixed<3> Rmin = arma::min(tar_->get_target_coords(),1);
		grid_size_ = 1.001*arma::max(Rmax - Rmin);
		grid_position_ = (Rmax + Rmin)/2 - grid_size_/2;
	}

	// override localpole to target
	void TrackingGrid::localpole_to_target(
		const arma::Mat<std::complex<fltp> > &Lp){
		Lp_ = Lp;
	}

	// override source to target
	void TrackingGrid::source_to_target(
		const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list, 
		const rat::cmn::ShLogPr& /*lg*/){
		
		// store lists
		target_list_ = target_list; 
		source_list_ = source_list;

		// create map
		node2stlist_.clear();
		for(arma::uword i=0;i<target_list.n_elem;i++){
			node2stlist_[target_list(i)] = i;
		}
	}

 
	// check which target points are within the grid
	// this doesn't yet say whether the box has a localpole
	arma::Row<arma::uword> TrackingGrid::is_inside(const arma::Mat<fltp> &Rt)const{
		const arma::Mat<fltp> v = 
			(Rt.each_col() - grid_position_)/grid_size_;
		const arma::Row<arma::uword> idx_in_grid = 
			arma::all(v>RAT_CONST(0.0) && v<RAT_CONST(1.0),0);
		return idx_in_grid;
	}

	// calculate field at target position(s)
	void TrackingGrid::calc_field(
		const ShTargetsPr &tar,
		const rat::cmn::ShLogPr& lg){

		// early out
		if(tar->num_targets()==0)return;

		// calculate external
		src_->calc_external(tar,stngs_,lg);

		// when there are no sources no need to continue
		if(src_->num_sources()==0)return;

		// check for direct calculation
		if(stngs_->get_direct()==DirectMode::ALWAYS){
			src_->calc_direct(tar,stngs_,lg); return;
		}

		// check if setup
		if(Lp_.is_empty())rat_throw_line("tracking grid not setup");

		// get target coords
		arma::Mat<fltp> Rt = tar->get_target_coords();

		// number of target points
		const arma::uword num_targets = Rt.n_cols;

		// number of dimensions in sources
		const arma::uword num_dim = src_->get_num_dim();

		// set boxsize root level
		fltp boxsize = grid_size_;

		// allocate morton indices
		arma::Row<arma::uword> morton_index(num_targets, arma::fill::zeros);

		// weight array (determines order in which the grid 
		// is serialized with the Morton indices, also see grid)
		const arma::Col<arma::uword>::fixed<3> weight = stngs_->get_morton_weights(); // x,y,z

		// allocate node positions
		arma::Mat<fltp> Rn(3,num_targets);

		// assign morton indices (no sorting)
		for(arma::uword ilevel=0;ilevel<num_levels_;ilevel++){
			// calculate boxsize for this level
			boxsize /= 2;

			// find three dimensional index in grid (incl. conversion to integer values)
			const arma::Mat<arma::uword> grid_index = 
				arma::conv_to<arma::Mat<arma::uword> >::from(
				(Rt.each_col() - grid_position_)/boxsize);

			// get the modulus this is the position within the parent's box
			const arma::Mat<arma::uword> grid_index_mod = cmn::Extra::modulus(grid_index,2);

			// increment morton indices for next level
			morton_index = cmn::Extra::bitshift(morton_index,3);
			morton_index += arma::sum(grid_index_mod.each_col()%weight,0);

			// calculate position of the nodes
			Rn = (arma::conv_to<arma::Mat<fltp> >::from(grid_index)+0.5)*boxsize;
		}

		// add grid position
		Rn.each_col()+=grid_position_;

		// sort targets by morton indices
		const arma::Row<arma::uword> sort_index = arma::sort_index(morton_index).t();

		// sort indices
		morton_index = morton_index.cols(sort_index);
		Rt = Rt.cols(sort_index);
		Rn = Rn.cols(sort_index);

		// sort targets into buckets
		tar->sort_targets(sort_index);

		// get sections
		const arma::Mat<arma::uword> sections = cmn::Extra::find_sections(morton_index);
		arma::Row<arma::uword> first_target = sections.row(0);
		arma::Row<arma::uword> last_target = sections.row(1);
		arma::uword num_sections = sections.n_cols;

		// allocate node index
		// arma::Row<arma::uword> node_index(num_targets);
		arma::Row<arma::uword> section_node_index(num_sections);

		// get node indices
		for(arma::uword i=0;i<num_sections;i++){
			// try find node index
			auto it = morton2node_.find(morton_index(first_target(i)));
			
			// check if index was found
			if(it==morton2node_.end()){
				section_node_index(i) = num_target_nodes_; continue;
			}

			// get index from iterator
			section_node_index(i) = (*it).second;

			// check index
			if(section_node_index(i)>=num_target_nodes_)
				rat_throw_line("node index beyond number of target nodes");
		}

		// calculate relative positions
		const arma::Mat<fltp> dR = Rn - Rt;

		// setup localpole to target
		tar->setup_localpole_to_target(dR, num_dim,stngs_);

		// find nodes that are non existent
		const arma::Col<arma::uword> idx_outside = 
			arma::find(section_node_index==num_target_nodes_);

		// dump these nodes from the list
		section_node_index.shed_cols(idx_outside);
		first_target.shed_cols(idx_outside);
		last_target.shed_cols(idx_outside);
		num_sections -= idx_outside.n_elem;

		// check if any nodes
		if(num_sections>0){
			// localpole to single target
			// as we have not grouped them by sorting	
			tar->localpole_to_target(
				Lp_.cols(fmm::Extra::expand_indices(
					section_node_index,num_dim)), 
				first_target, last_target, num_dim, stngs_);

			// allocate source and target lists
			arma::field<arma::Col<arma::uword> > source_list(num_sections); 
			arma::Col<arma::uword> target_list(num_sections); arma::uword cnt = 0;

			// walk over node indices
			for(arma::uword i=0;i<section_node_index.n_elem;i++){
				// check if this node has sources associated to it
				auto it = node2stlist_.find(section_node_index(i));
				if(it==node2stlist_.end())continue;

				// add it to source to target lists
				target_list(cnt) = i;
				source_list(cnt) = source_list_((*it).second);
				cnt++;
			}

			// resize to real number on list
			if(cnt>0){
				target_list = target_list.rows(0,cnt-1);
				source_list = source_list.rows(0,cnt-1);

				// some checks
				if(target_list.n_elem!=source_list.n_elem)
					rat_throw_line("source list does not match target list number of elements");

				if(arma::any(target_list>=first_target.n_elem))
					rat_throw_line("target list beyond number of sections");

				// source to target trickery
				src_->source_to_target(tar, 
					target_list, source_list, 
					first_source_, last_source_, 
					first_target, last_target, stngs_,lg);	
			}
		}

		// sort targets into buckets
		tar->unsort_targets(sort_index);
	}

}}