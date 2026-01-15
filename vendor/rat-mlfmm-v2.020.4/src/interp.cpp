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
#include "interp.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// constructors
	Interp::Interp(){

	}

	// constructors
	Interp::Interp(const fmm::ShInterpPrList &ip){
		set_mesh(ip);
	}

	// factory
	ShInterpPr Interp::create(){
		return std::make_shared<Interp>();
	}

	// factory
	ShInterpPr Interp::create(const fmm::ShInterpPrList &ip){
		return std::make_shared<Interp>(ip);
	}

	// subdivide algorithm for tetrahedrons
	ShSourcesPr Interp::subdivide_tetrahedrons(const fltp grid_size){
		// ensure we're dealing with a tetrahedron
		assert(n_.n_rows == 4);

			// precomputed local (r,s,t) coordinates for order‑2 subdivision (10 unique points)
		const arma::Mat<fltp> Rq = arma::Mat<fltp>({
			{0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5},
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5}
		});

		// connectivity of the 8 subtetrahedra (matching Rq order: a=0,b=1,c=2,d=3,ab=4,ac=5,ad=6,bc=7,bd=8,cd=9)
		const arma::Mat<arma::uword> Tconn = arma::Mat<arma::uword>({
			{0, 4, 5, 6},  // subtet at corner a
			{4, 1, 7, 8},  // at corner b
			{5, 7, 2, 9},  // at corner c
			{6, 8, 9, 3},  // at corner d
			{4, 7, 5, 6},  // mid region facets
			{5, 7, 9, 6},
			{6, 8, 9, 7},
			{4, 7, 8, 6}
		}).t();

		// split elements by size
		const arma::Col<arma::uword> idx_div  = arma::find(2*element_radius_ > grid_size);
		const arma::Col<arma::uword> idx_keep = arma::find(2*element_radius_ <= grid_size);

		// allocate refined points & data
		arma::Mat<fltp> Rnnew(3, idx_div.n_elem * Rq.n_cols);
		arma::Mat<fltp> Mnew(M_.n_rows, idx_div.n_elem * Rq.n_cols);
		for(arma::uword i = 0; i < idx_div.n_elem; i++){
			const arma::Col<arma::uword>::fixed<4> cols = n_.col(idx_div(i));
			// use quad2cart: Rn_.cols(cols) is 3x4, Rq is 3x10 -> outputs 3x10
			Rnnew.cols(i*Rq.n_cols, (i+1)*Rq.n_cols-1) = cmn::Tetrahedron::quad2cart(Rn_.cols(cols), Rq);
			Mnew.cols(i*Rq.n_cols, (i+1)*Rq.n_cols-1) = cmn::Tetrahedron::quad2cart(M_.cols(cols), Rq);
		}

		// build connectivity with global offset
		arma::Mat<arma::uword> nnew(4, idx_div.n_elem * Tconn.n_cols);
		for(arma::uword i = 0; i < idx_div.n_elem; i++){
			nnew.cols(i*Tconn.n_cols, (i+1)*Tconn.n_cols-1) = Tconn + i*Rq.n_cols;
		}

		// assemble new interpolation object
		auto interp = Interp::create();
		interp->set_mesh(
			arma::join_horiz(Rnnew, Rn_),
			arma::join_horiz(nnew, n_.cols(idx_keep) + Rnnew.n_cols)
		);
		interp->set_interpolation_values(type_, arma::join_horiz(Mnew, M_));
		interp->cleanup_mesh();
		return interp;
	}

	// subdivide algorithm for hexahedrons
	ShSourcesPr Interp::subdivide_hexahedrons(const fltp grid_size){
		// check
		assert(n_.n_rows==8);

		// subdivide
		arma::Mat<fltp> Rq(3,27); arma::uword cnt1 = 0llu;
		for(arma::uword i=0;i<3;i++){
			for(arma::uword j=0;j<3;j++){
				for(arma::uword k=0;k<3;k++){
					Rq(0,cnt1) = -1.0 + static_cast<fltp>(i);
					Rq(1,cnt1) = -1.0 + static_cast<fltp>(j);
					Rq(2,cnt1) = -1.0 + static_cast<fltp>(k);
					cnt1++;
				}
			}
		}
		assert(cnt1==27);

		// connectivity matrix
		arma::Mat<arma::uword> N(8,8); arma::uword cnt2 = 0llu;
		for(arma::uword i=0;i<2;i++){
			for(arma::uword j=0;j<2;j++){
				for(arma::uword k=0;k<2;k++){
					N(0,cnt2) = i*3*3 + j*3 + k;			 // v0
					N(1,cnt2) = (i+1)*3*3 + j*3 + k;		 // v1
					N(2,cnt2) = (i+1)*3*3 + (j+1)*3 + k;	 // v2
					N(3,cnt2) = i*3*3 + (j+1)*3 + k;		 // v3
					N(4,cnt2) = i*3*3 + j*3 + (k+1);		 // v4
					N(5,cnt2) = (i+1)*3*3 + j*3 + (k+1);	 // v5
					N(6,cnt2) = (i+1)*3*3 + (j+1)*3 + (k+1); // v6
					N(7,cnt2) = i*3*3 + (j+1)*3 + (k+1);	 // v7
					cnt2++;
				}
			}
		}
		assert(cnt2==8);

		// find elements that are too large
		const arma::Col<arma::uword> idx_div = arma::find(2*element_radius_>grid_size);
		const arma::Col<arma::uword> idx_keep = arma::find(2*element_radius_<=grid_size);

		// walk over elements
		arma::Mat<fltp> Rnnew(3,idx_div.n_elem*Rq.n_cols);
		arma::Mat<fltp> Mnew(M_.n_rows,idx_div.n_elem*Rq.n_cols);
		for(arma::uword i=0;i<idx_div.n_elem;i++){
			Rnnew.cols(i*Rq.n_cols,(i+1)*Rq.n_cols-1) = cmn::Hexahedron::quad2cart(Rn_.cols(n_.col(idx_div(i))),Rq);
			Mnew.cols(i*Rq.n_cols,(i+1)*Rq.n_cols-1) = cmn::Hexahedron::quad2cart(M_.cols(n_.col(idx_div(i))),Rq);
		}

		// create new connectivity matrix
		arma::Mat<arma::uword> nnew(N.n_rows, idx_div.n_elem*N.n_cols);
		for(arma::uword i=0;i<idx_div.n_elem;i++){
			nnew.cols(i*N.n_cols,(i+1)*N.n_cols-1) = N + i*Rq.n_cols;
		}

		// store to self
		const ShInterpPr interp = Interp::create();
		interp->set_mesh(
			arma::join_horiz(Rnnew,Rn_),
			arma::join_horiz(nnew,n_.cols(idx_keep)+Rnnew.n_cols));
		interp->set_interpolation_values(type_, arma::join_horiz(Mnew,M_));
		interp->cleanup_mesh();

		// return the interpolation
		return interp;
	}

	// subdivision into smaller elements
	ShSourcesPr Interp::subdivide(const fltp grid_size){
		return n_.n_rows==4 ? subdivide_tetrahedrons(grid_size) : subdivide_hexahedrons(grid_size);
	}

	// cleanup mesh
	void Interp::cleanup_mesh(){
		// calculate node valence
		arma::Col<arma::uword> node_valence(Rn_.n_cols,arma::fill::zeros);
		for(arma::uword i=0;i<n_.n_cols;i++)node_valence.rows(n_.col(i))+=1;

		// create reindexing array
		const arma::Col<arma::uword> reindex1 = arma::join_vert(
			arma::Col<arma::uword>{0}, arma::cumsum(node_valence>0));

		// shed nodes that are not connected anymore
		const arma::Col<arma::uword> shed_nodes = arma::find(node_valence==0);
		Rn_.shed_cols(shed_nodes); M_.shed_cols(shed_nodes);

		// merge co-inciding nodes
		const arma::Row<arma::uword> reindex2 = cmn::Extra::combine_nodes(Rn_,M_);

		// update elements
		n_ = arma::reshape(reindex2(reindex1(arma::vectorise(n_))),n_.n_rows,n_.n_cols);

		// perform setup again
		setup();
	}

	char Interp::get_type()const{
		return type_;
	}


	// set mesh
	void Interp::set_mesh(
		const fmm::ShInterpPrList &ip){

		// number of input meshes
		const arma::uword num_meshes = ip.n_elem;

		// allocate mesh
		arma::field<arma::Mat<fltp> > Rnfld(1,num_meshes);
		arma::field<arma::Mat<arma::uword> > nfld(1,num_meshes);
		arma::field<arma::Mat<fltp> > M(1,num_meshes);

		// set type
		type_ = ip(0)->get_type();

		// gather
		arma::uword node_shift = 0;
		for(arma::uword i=0;i<num_meshes;i++){
			// copy data
			Rnfld(i) = ip(i)->Rn_;
			nfld(i) = ip(i)->n_ + node_shift;
			M(i) = ip(i)->M_;

			// account for node shifting
			node_shift += Rnfld(i).n_cols;

			if(type_!=ip(i)->get_type())
				rat_throw_line("types do not agree");
		}

		// combine and store in self
		Rn_ = cmn::Extra::field2mat(Rnfld);
		n_ = cmn::Extra::field2mat(nfld);
		M_ = cmn::Extra::field2mat(M);

		// calculate volume and centroids
		setup();
	}

	// set mesh
	void Interp::set_mesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n){

		// set data
		n_ = n; Rn_ = Rn;

		// call setup function
		setup();
	}

	// set values
	void Interp::set_interpolation_values(const char type, const arma::Mat<fltp> &M){
		type_ = type; M_ = M;
	}

	// setup function
	void Interp::setup(){
		// check user input
		if(Rn_.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(n_.n_rows!=8 && n_.n_rows!=4)rat_throw_line("element matrix must have four or eight rows");
		if(n_.max()>=Rn_.n_cols && n_.n_cols>0)rat_throw_line("element matrix contains index outside coordinate matrix");

		// get counters
		const arma::uword num_elements = n_.n_cols;

		// allocate element centroids
		Re_.set_size(3,n_.n_cols);

		// walk over elements
		const arma::Col<fltp> gp = n_.n_rows==8 ? cmn::Hexahedron::create_gauss_points(1) : cmn::Tetrahedron::create_gauss_points(1);
		arma::Col<fltp>::fixed<3> Rq = gp.rows(0,2);
		for(arma::uword i=0;i<num_elements;i++){
			// calculate centroid using quadrilateral coordinates
			Re_.col(i) = n_.n_rows==8 ? cmn::Hexahedron::quad2cart(Rn_.cols(n_.col(i)),Rq) : cmn::Tetrahedron::quad2cart(Rn_.cols(n_.col(i)),Rq);
		}

		// distance to center
		element_radius_.zeros(1,num_elements);
		for(arma::uword i=0;i<n_.n_rows;i++){
			arma::Mat<fltp> Rr = (Re_ - Rn_.cols(n_.row(i)));
			element_radius_ = arma::max(element_radius_,
				arma::sqrt(arma::sum(Rr%Rr,0)));
		}
	}

	// get node coordinates
	arma::Mat<fltp> Interp::get_node_coords() const{
		// check if node coordinates were set
		if(Rn_.is_empty())rat_throw_line("coordinates matrix not set");

		// return node coordinates
		return Rn_;
	}

	// get elements
	arma::Mat<arma::uword> Interp::get_elements() const{
		if(n_.is_empty())rat_throw_line("element matrix not set");
		return n_;
	}

	// get element size for determining grid refinement criterion
	fltp Interp::element_size() const{
		return 2*arma::max(element_radius_);
	}

	// all source types should have these methods
	// in order to commmunicate with MLFMM
	arma::Mat<fltp> Interp::get_source_coords() const{
		// check if coordinates were set
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

		// return element centroids
		return Re_;
	}

	// // get source coordinates at specified indices
	// arma::Mat<fltp> Interp::get_source_coords(
	// 	const arma::Row<arma::uword> &indices) const{
	// 	// check if coordinates were set
	// 	if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

	// 	// return element centroids at given indices
	// 	return Re_.cols(indices);
	// }

	// field calculation from specific sources
	void Interp::calc_direct(const ShTargetsPr &tar, const ShSettingsPr &/*stngs*/, const rat::cmn::ShLogPr& lg) const{
		// check if target has specified type
		if(tar->has(type_)){
			// allocate
			arma::Mat<fltp> fld(get_num_dim(),tar->num_targets(),arma::fill::zeros);

			// get targets
			const arma::Mat<fltp> Rt = tar->get_target_coords();

			// walk over all elements
			for(arma::uword i=0;i<n_.n_cols;i++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// calculate distance of 
				// all target points to this source
				const arma::Row<fltp> dist = cmn::Extra::vec_norm(Rt.each_col() - Re_.col(i));

				// find indexes of target points 
				// that are inside sphere
				const arma::Row<arma::uword> indices_near = 
					arma::find(dist<=num_dist_*element_radius_(i)).t();

				// for target points that are close
				// use gauss points to do the integration
				if(!indices_near.is_empty()){
					// my nodes
					const arma::Mat<fltp> myRn = Rn_.cols(n_.col(i));

					// get quadrilateral coordinates (iteratively)
					arma::Mat<fltp> Rqt = n_.n_rows==8 ? 
						cmn::Hexahedron::cart2quad(myRn, Rt.cols(indices_near), RAT_CONST(1e-4)) : 
						cmn::Tetrahedron::cart2quad(myRn, Rt.cols(indices_near), RAT_CONST(1e-4));

					// find indexes inside
					const arma::Row<arma::uword> idx_inside = n_.n_rows==8 ?
						arma::find(arma::all(Rqt<(RAT_CONST(1.0)+tol_) && Rqt>-(RAT_CONST(1.0)+tol_),0)).t().eval() :
						arma::find(cmn::Tetrahedron::is_inside(Rt.cols(indices_near), myRn, arma::Col<arma::uword>{0,1,2,3})).t().eval();

					// interpolate
					fld.cols(indices_near.cols(idx_inside)) = n_.n_rows==8 ? 
						cmn::Hexahedron::quad2cart(M_.cols(n_.col(i)),Rqt.cols(idx_inside)) : 
						cmn::Tetrahedron::quad2cart(M_.cols(n_.col(i)),Rqt.cols(idx_inside));
				}
			}

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field(type_,fld,true);
		}
	}

	void Interp::source_to_target(const ShTargetsPr &tar, 
		const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list, 
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target, 
		const ShSettingsPr &stngs,
		const rat::cmn::ShLogPr& lg) const{

		// get counters
		const arma::uword num_elements = n_.n_cols;
		const arma::uword num_nodes = Rn_.n_cols;

		// check input
		if(M_.n_cols!=num_nodes)rat_throw_line("interpolation matrix does not match number of nodes");
		if(Re_.n_cols!=num_elements)rat_throw_line("element centroids are not setup");

		// check if target has specified type
		if(tar->has(type_)){
			// get targets
			const arma::Mat<fltp> Rt = tar->get_target_coords();

			// allocate
			arma::Mat<fltp> fld(get_num_dim(),tar->num_targets(),arma::fill::zeros);

			// walk over source nodes
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
				// check cancel flag
				if(lg->is_cancelled())return;

				// get target index
				const arma::uword target_idx = target_list(i);

				// get location of the targets
				const arma::uword ft = first_target(target_idx);
				const arma::uword lt = last_target(target_idx);
				assert(ft<=lt); assert(lt<tar->num_targets());

				// my target positions
				const arma::Mat<fltp> myRt = Rt.cols(ft,lt);

				// walk over source elements
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// check cancel flag
					if(lg->is_cancelled())return;

					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source elements
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_elements);

					// walk over sources in list
					for(arma::uword n=0;n<(ls-fs+1);n++){
						// get source index
						const arma::uword mysource = fs+n;

						// calculate distance of 
						// all target points to this source
						const arma::Row<fltp> dist = cmn::Extra::vec_norm(myRt.each_col() - Re_.col(mysource));

						// find indexes of target points 
						// that are inside sphere
						const arma::Row<arma::uword> indices_near = 
							arma::find(dist<=num_dist_*element_radius_(mysource)).t();

						// for target points that are close
						// use gauss points to do the integration
						if(!indices_near.is_empty()){
							// my nodes
							const arma::Mat<fltp> myRn = Rn_.cols(n_.col(mysource));

							// get quadrilateral coordinates (iteratively)
							const arma::Mat<fltp> Rqt = n_.n_rows==8 ? 
								cmn::Hexahedron::cart2quad(myRn, myRt.cols(indices_near), RAT_CONST(1e-4)) : 
								cmn::Tetrahedron::cart2quad(myRn, myRt.cols(indices_near), RAT_CONST(1e-4));

							// find indexes inside
							const arma::Row<arma::uword> idx_inside = n_.n_rows==8 ? 
								arma::find(arma::all(Rqt<(1.0+tol_) && Rqt>-(RAT_CONST(1.0)+tol_),0)).t().eval() :
								arma::find(cmn::Tetrahedron::is_inside(myRt.cols(indices_near), myRn, arma::Col<arma::uword>{0,1,2,3})).t().eval();

							// if there is inside indices
							if(!idx_inside.is_empty()){
								// interpolate
								fld.cols(ft+indices_near.cols(idx_inside)) = n_.n_rows==8 ?
									cmn::Hexahedron::quad2cart(M_.cols(n_.col(mysource)),Rqt.cols(idx_inside)) : 
									cmn::Tetrahedron::quad2cart(M_.cols(n_.col(mysource)),Rqt.cols(idx_inside));
							}
						}

					}
				}
			});
			
			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field(type_,fld,true);
		}

	}

	// sort sources
	void Interp::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element node index matrix not set");
		if(M_.is_empty())rat_throw_line("value matrix not set");
		if(Re_.is_empty())rat_throw_line("element centroid matrix not calculated");
		if(element_radius_.is_empty())rat_throw_line("element radius vector not calculated");
		
		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_ = n_.cols(sort_idx);
		Re_ = Re_.cols(sort_idx);
		element_radius_ = element_radius_.cols(sort_idx);
	}

	// unsort sources
	void Interp::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element node index matrix not set");
		if(M_.is_empty())rat_throw_line("value matrix not set");
		if(Re_.is_empty())rat_throw_line("element centroid matrix not calculated");
		if(element_radius_.is_empty())rat_throw_line("element radius vector not calculated");
		
		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_.cols(sort_idx) = n_;
		Re_.cols(sort_idx) = Re_;
		element_radius_.cols(sort_idx) = element_radius_;
	}

	// get number of dimensions
	arma::uword Interp::get_num_dim() const{
		return M_.n_rows;
	}

	// getting basic information
	arma::uword Interp::num_sources() const{
		return n_.n_cols;
	}

	// basic build in mesh shape cylinder
	// for testing purposes
	// note that after calling this method
	void Interp::setup_cylinder(
		const fltp Rin, const fltp Rout, 
		const fltp height, const arma::uword nr, 
		const arma::uword nz, const arma::uword nl){

		// check user input
		if(Rout<=Rin)rat_throw_line("outer radius must be larger than inner radius");
		if(height<=0)rat_throw_line("height must be larger than zero");
		if(nr<=1)rat_throw_line("number of radial coordinates must be larger than one");
		if(nz<=1)rat_throw_line("number of axial coordinates must be larger than one");
		if(nl<=1)rat_throw_line("number of azymuthal coordinates must be larger than one");

		// create azymuthal coordinates of nodes
		arma::Row<fltp> theta = arma::linspace<arma::Row<fltp> >(0,-(RAT_CONST(1.0)-RAT_CONST(1.0)/nl)*2*arma::Datum<fltp>::pi,nl);

		// create radial coordinates of nodes
		arma::Row<fltp> rho = arma::linspace<arma::Row<fltp> >(Rin,Rout,nr);

		// create axial cooridnates of nodes
		arma::Row<fltp> z = arma::linspace<arma::Row<fltp> >(-height/2,height/2,nz);

		// create matrix to hold node coordinates
		arma::Mat<fltp> xn(nl,nr*nz), yn(nl,nr*nz), zn(nl,nr*nz);

		// build node coordinates in two dimensions (first block of matrix)
		for(arma::uword i=0;i<nr;i++){
			// generate circles with different radii
			xn.col(i) = rho(i)*arma::cos(theta).t();
			yn.col(i) = rho(i)*arma::sin(theta).t();
			zn.col(i).fill(z(0));
		}

		// extrude to other axial planes
		for(arma::uword j=1;j<nz;j++){
			// copy coordinates from ground plane
			xn.cols(j*nr,(j+1)*nr-1) = xn.cols(0,nr-1);
			yn.cols(j*nr,(j+1)*nr-1) = yn.cols(0,nr-1);
			zn.cols(j*nr,(j+1)*nr-1).fill(z(j));
		}

		// number of nodes
		const arma::uword num_nodes = nr*nl*nz;
		
		// create node coordinates
		Rn_.set_size(3,num_nodes);
		Rn_.row(0) = arma::reshape(xn,1,num_nodes);
		Rn_.row(1) = arma::reshape(yn,1,num_nodes);
		Rn_.row(2) = arma::reshape(zn,1,num_nodes);

		// create matrix of node indices
		arma::Mat<arma::uword> node_idx = arma::regspace<arma::Mat<arma::uword> >(0,num_nodes-1);
		node_idx.reshape(nl,nr*nz);

		// close mesh by setting last row to the first
		node_idx = arma::join_vert(node_idx,node_idx.row(0));

		// get definition of hexahedron element
		arma::Mat<arma::sword> M = 
			arma::conv_to<arma::Mat<arma::sword> >::from(
			(cmn::Hexahedron::get_corner_nodes()+1)/2);

		// calculate number of elements
		const arma::uword num_elements = nl*(nr-1)*(nz-1);

		// allocate elements
		n_.set_size(8,num_elements);

		// create elements between the nodes	
		for(arma::uword j=0;j<nz-1;j++){
			// walk over corner nodes
			for(arma::uword k=0;k<8;k++){
				// get matrix indexes
				arma::uword idx0 = M(k,0), idx1 = M(k,0)+nl+1-2;
				arma::uword idx2 = (j+M(k,2))*nr+M(k,1), idx3 = (j+M(k,2)+1)*nr+M(k,1)-2;
				arma::uword idx4 = j*nl*(nr-1), idx5 = (j+1)*nl*(nr-1)-1;

				// get node indexes for this corner
				n_.submat(arma::span(k,k),arma::span(idx4,idx5)) =
					arma::reshape(node_idx.submat(arma::span(idx0,idx1),
						arma::span(idx2,idx3)),1,nl*(nr-1));
			}
		}

		// calculate volume and element centroids
		setup();
	}

}}