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
#include "currentsurface.hh"

// rat-common headers
#include "rat/common/gauss.hh"
#include "rat/common/error.hh"
#include "rat/common/extra.hh"
#include "rat/common/elements.hh"

// rat-mlfmm headers
#include "chargedpolyhedron.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// default constructor
	// default is hex
	CurrentSurface::CurrentSurface(){
		
	}

	// constructor from group of meshes
	CurrentSurface::CurrentSurface(const ShCurrentSurfacePrList &meshes){
		set_mesh(meshes);
	}

	// constructor from group of meshes
	CurrentSurface::CurrentSurface(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Je) : CurrentSurface(){
		set_mesh(Rn,n,Je);
	}

	// factory
	ShCurrentSurfacePr CurrentSurface::create(){
		//return ShIListPr(new IList);
		return std::make_shared<CurrentSurface>();
	}

	ShCurrentSurfacePr CurrentSurface::create(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Je){
		return std::make_shared<CurrentSurface>(Rn,n,Je);
	}

	// factory
	ShCurrentSurfacePr CurrentSurface::create(const ShCurrentSurfacePrList &meshes){
		return std::make_shared<CurrentSurface>(meshes);
	}

	// constructor
	void CurrentSurface::set_mesh(const ShCurrentSurfacePrList &meshes){
		// number of input meshes
		const arma::uword num_meshes = meshes.n_elem;

		// allocate mesh
		arma::field<arma::Mat<fltp> > Rnfld(1,num_meshes);
		arma::field<arma::Mat<arma::uword> > nfld(1,num_meshes);
		arma::field<arma::Mat<fltp> > Jefld(1,num_meshes);

		// gather
		arma::uword node_shift = 0;
		for(arma::uword i=0;i<num_meshes;i++){
			// copy data
			Rnfld(i) = meshes(i)->Rn_;
			nfld(i) = meshes(i)->n_ + node_shift;
			Jefld(i) = meshes(i)->Je_;

			// account for node shifting
			node_shift += Rnfld(i).n_cols;
		}

		// combine and store in self
		Rn_ = cmn::Extra::field2mat(Rnfld);
		n_ = cmn::Extra::field2mat(nfld);
		Je_ = cmn::Extra::field2mat(Jefld);

		// set counters
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;

		// calculate volume and centroids
		calculate_element_areas();
	}

	// setting a hexagonal mesh with volume elements
	void CurrentSurface::set_mesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n){

		// check input
		if(Rn.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(n.n_rows!=3 && n.n_rows!=4)rat_throw_line("element node index matrix must have three or four rows");
		if(n.max()>=Rn.n_cols)rat_throw_line("element matrix contains index outside coordinate matrix");

		// set supplied values
		Rn_ = Rn; n_ = n;

		// get number of supplied nodes
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;	

		// calculate areas
		calculate_element_areas();
	}

	// setting a hexagonal mesh with volume elements
	void CurrentSurface::set_mesh(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Je){

		// check input
		if(Rn.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(n.n_rows!=3 && n.n_rows!=4)rat_throw_line("element node index matrix must have three or four rows");
		if(n.max()>=Rn.n_cols)rat_throw_line("element matrix contains index outside coordinate matrix");
		if(Je.n_rows!=3)rat_throw_line("current density matrix must have three rows");

		// set supplied values
		Rn_ = Rn; n_ = n; Je_ = Je;

		// get number of supplied nodes
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;	

		// calculate areas
		calculate_element_areas();
	}

	// set number of gauss points
	void CurrentSurface::set_num_gauss(const arma::sword num_gauss){
		num_gauss_ = num_gauss;
	}


	// calculate areas and centroids of the elements
	void CurrentSurface::calculate_element_areas(){
		// check if it is really quad
		if(n_.n_rows!=3 && n_.n_rows!=4)rat_throw_line("element node index matrix must have three or four rows");

		// calculate face normals
		Ne_ = n_.n_rows==3 ? 
			cmn::Triangle::calc_face_normals(Rn_,n_) :
			cmn::Quadrilateral::calc_face_normals(Rn_,n_);

		// calculate face area using two triangles
		Ae_ = n_.n_rows==3 ? 
			cmn::Triangle::calc_area(Rn_,n_) :
			cmn::Quadrilateral::calc_area(Rn_,n_);

		// allocate centroids
		Re_.set_size(3,num_elements_);

		// walk over elements and calculate centroids
		arma::Col<fltp>::fixed<2> Rq = {0,0};
		for(arma::uword i=0;i<num_elements_;i++){
			// calculate centroid using quadrilateral coordinates
			Re_.col(i) = n_.n_rows==3 ? 
				cmn::Triangle::quad2cart(Rn_.cols(n_.col(i)),Rq) : 
				cmn::Quadrilateral::quad2cart(Rn_.cols(n_.col(i)),Rq);
		}

		// distance to center
		element_radius_.zeros(1,num_elements_);
		for(arma::uword i=0;i<n_.n_rows;i++){
			arma::Mat<fltp> Rr = (Re_ - Rn_.cols(n_.row(i)));
			element_radius_ = arma::max(element_radius_,
				arma::sqrt(arma::sum(Rr%Rr,0)));
		}

		// check elements
		assert(arma::all(element_radius_>1e-9));
	}

	// subdivision into smaller elements
	ShSourcesPr CurrentSurface::subdivide(const fltp grid_size){
		// current element radius
		const fltp old_element_radius = element_radius_.max();

		// shape depenent parameters
		const arma::uword num_sub_nodes = n_.n_rows==3 ? 6llu : 9llu;
		const arma::uword num_sub_elements = 4llu;
		const arma::Mat<arma::uword> E = n_.n_rows==3 ? 
			cmn::Triangle::get_edges().eval() : 
			cmn::Quadrilateral::get_edges().eval();

		// element element subdivision table
		const arma::Mat<arma::uword> N = n_.n_rows==3 ? 
			arma::reshape(arma::Mat<arma::uword>{0,3,5, 3,1,4, 3,4,5, 5,4,2},3,4) : 
			arma::reshape(arma::Mat<arma::uword>{0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3},4,4);

		// find elements to subdivide
		const arma::Col<arma::uword> idx_sub = arma::find(2*element_radius_>grid_size);
		const arma::Col<arma::uword> idx_keep = arma::find(2*element_radius_<=grid_size);

		// allocate new nodes and elements
		arma::Mat<fltp> Rnnew(3,idx_sub.n_elem*num_sub_nodes);
		arma::Mat<arma::uword> nnew(n_.n_rows,idx_sub.n_elem*num_sub_elements);
		arma::Mat<fltp> Jenew(3,idx_sub.n_elem*num_sub_elements);

		// walk over elements
		for(arma::uword i=0;i<idx_sub.n_elem;i++){
			// get nodes for this elements
			const arma::Mat<fltp> Rn = Rn_.cols(n_.col(idx_sub(i)));

			// create new nodes on edges
			arma::Mat<fltp> Re(3,E.n_rows);
			for(arma::uword j=0;j<E.n_rows;j++)
				Re.col(j) = arma::mean(Rn.cols(E.row(j)),1);
		
			// create new node in middle
			const arma::Col<fltp>::fixed<3> Rm = arma::mean(Rn,1);

			// indexing
			const arma::uword idx1 = i*num_sub_nodes;
			const arma::uword idx2 = (i+1)*num_sub_nodes - 1;

			// combine and store nodes
			Rnnew.cols(idx1,idx2) = n_.n_rows==3 ? 
				arma::join_horiz(Rn,Re) : arma::join_horiz(Rn,Re,Rm);

			// create elements
			const arma::uword idx3 = i*num_sub_elements;
			const arma::uword idx4 = (i+1)*num_sub_elements - 1;
			nnew.cols(idx3,idx4) = N + i*num_sub_nodes;

			// currents
			if(!Je_.empty())
				Jenew.cols(idx3,idx4) = arma::repmat(Je_.col(idx_sub(i)),1,num_sub_elements);
		}

		// store to self
		const ShCurrentSurfacePr new_current_surface = CurrentSurface::create(
			arma::join_horiz(Rnnew,Rn_),
			arma::join_horiz(nnew,n_.cols(idx_keep)+Rnnew.n_cols),
			arma::join_horiz(Jenew,Je_.cols(idx_keep)));
		new_current_surface->cleanup_mesh();

		// element size should be halved
		if(new_current_surface->element_radius_.cols(0,nnew.n_cols-1).max()>RAT_CONST(0.51)*old_element_radius)
			rat_throw_line("element subdivision did not work");

		// recursive call
		if(new_current_surface->element_size()>grid_size)
			return new_current_surface->subdivide(grid_size);

		// done
		return new_current_surface;
	}

	// cleanup mesh
	void CurrentSurface::cleanup_mesh(){
		// calculate node valence
		arma::Col<arma::uword> node_valence(Rn_.n_cols,arma::fill::zeros);
		for(arma::uword i=0;i<n_.n_cols;i++)node_valence.rows(n_.col(i))+=1;

		// create reindexing array
		const arma::Col<arma::uword> reindex1 = arma::join_vert(
			arma::Col<arma::uword>{0}, arma::cumsum(node_valence>0));

		// shed nodes that are no longer connected
		Rn_.shed_cols(arma::find(node_valence==0));

		// merge co-inciding nodes
		const arma::Row<arma::uword> reindex2 = cmn::Extra::combine_nodes(Rn_);
		n_ = arma::reshape(reindex2(reindex1(arma::vectorise(n_))),n_.n_rows,n_.n_cols);

		// perform setup again
		num_nodes_ = Rn_.n_cols;
		num_elements_ = n_.n_cols;

		// calculate volume and centroids
		calculate_element_areas();

		// calculate field at element centroids
		// Rt_ = Re_; num_targets_ = Rt_.n_cols;
	}

	// set magnetisation nodes
	void CurrentSurface::set_magnetisation_nodes(const arma::Mat<fltp> &Mn){
		// calculate magnetisation at elements
		arma::Mat<fltp> Me(3,num_elements_);

		// calculate
		const arma::Col<fltp>::fixed<2> Rq = {0,0};
		for(arma::uword i=0;i<num_elements_;i++)
			Me.col(i) = n_.n_rows==3 ? 
				cmn::Triangle::quad2cart(Mn.cols(n_.col(i)),Rq) : 
				cmn::Quadrilateral::quad2cart(Mn.cols(n_.col(i)),Rq);

		// calculate current
		Je_ = cmn::Extra::cross(Me,Ne_);
	}

	// get element volumes
	arma::Row<fltp> CurrentSurface::get_area() const{
		// check if volumes were calculated
		if(Ae_.is_empty())rat_throw_line("element area vector was not calculated");

		// return element volumes
		return Ae_;
	}

	// get number of sources (each element has one source)
	arma::uword CurrentSurface::get_num_nodes() const{
		assert(num_nodes_>0);
		return num_nodes_;
	}

	// get number of sources (each element has one source)
	arma::uword CurrentSurface::get_num_elements() const{
		assert(num_elements_>0);
		return num_elements_;
	}

	// get node coordinates
	const arma::Mat<fltp>& CurrentSurface::get_node_coords() const{
		// check if node coordinates were set
		if(Rn_.is_empty())rat_throw_line("node coordinate matrix not set");

		// return node coordinates
		return Rn_;
	}

	// get element node indices
	const arma::Mat<arma::uword>& CurrentSurface::get_elements() const{
		// check if elements were set
		if(n_.is_empty())rat_throw_line("element node index matrix not set");

		// return 
		return n_;
	}

	// get element centroids
	arma::Mat<fltp> CurrentSurface::get_source_coords() const{
		// check if coordinates were set
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

		// return element centroids
		return Re_;
	}

	// // get element centroids of specific elements
	// arma::Mat<fltp> CurrentSurface::get_source_coords(
	// 	const arma::Row<arma::uword> &indices) const{
	// 	// check if coordinates were set
	// 	if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix not calculated");

	// 	// return element centroids
	// 	return Re_.cols(indices);
	// }

	// count number of sources stored
	arma::uword CurrentSurface::num_sources() const{
		// return number of elements
		assert(num_elements_>0);
		return num_elements_;
	}

	// get number of dimensions
	arma::uword CurrentSurface::get_num_dim() const{
		assert(num_dim_>0);
		return num_dim_;
	}

	// get element size for determining grid refinement criterion
	fltp CurrentSurface::element_size() const{
		return 2*arma::max(element_radius_);
	}

	// sorting function
	void CurrentSurface::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element matrix was not set");
		if(Je_.is_empty())rat_throw_line("current density matrix was not set");
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix was not set");
		if(Ae_.is_empty())rat_throw_line("element area vector was not set");
		if(Ne_.is_empty())rat_throw_line("face normal matrix was not set");
		if(element_radius_.is_empty())rat_throw_line("element radius vector was not set");

		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_ = n_.cols(sort_idx);
		Je_ = Je_.cols(sort_idx);
		Re_ = Re_.cols(sort_idx);
		Ae_ = Ae_.cols(sort_idx);
		Ne_ = Ne_.cols(sort_idx);
		element_radius_ = element_radius_.cols(sort_idx);
	}

	// unsorting function
	void CurrentSurface::unsort_sources(const arma::Row<arma::uword> &sort_idx){
		// check if sources were properly set
		if(n_.is_empty())rat_throw_line("element matrix was not set");
		if(Je_.is_empty())rat_throw_line("current density matrix was not set");
		if(Re_.is_empty())rat_throw_line("element centroid coordinate matrix was not set");
		if(Ae_.is_empty())rat_throw_line("element area vector was not set");
		if(Ne_.is_empty())rat_throw_line("face normal matrix was not set");
		if(element_radius_.is_empty())rat_throw_line("element radius vector was not set");

		// check if sort array right length
		assert(n_.n_cols == sort_idx.n_elem);

		// sort sources
		n_.cols(sort_idx) = n_;
		Je_.cols(sort_idx) = Je_;
		Re_.cols(sort_idx) = Re_;
		Ae_.cols(sort_idx) = Ae_;
		Ne_.cols(sort_idx) = Ne_;
		element_radius_.cols(sort_idx) = element_radius_;
	}


	// setup source to multipole matrices
	void CurrentSurface::setup_source_to_multipole(
		const arma::Mat<fltp> &dR,
		const ShSettingsPr &stngs){
		
		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_s2m()){
			dR_ = dR;
		}

		// maximize speed over memory efficiency
		else{
			// get number of expansions
			const int num_exp = stngs->get_num_exp();

			// create gauss points
			const arma::Mat<fltp> gp = n_.n_rows==3 ? 
				cmn::Triangle::create_gauss_points(num_gauss_) : 
				cmn::Quadrilateral::create_gauss_points(num_gauss_);
			const arma::Mat<fltp> Rqg = gp.rows(0,1);
			const arma::Row<fltp> wqg = gp.row(2);

			// position of multipole
			const arma::Mat<fltp> Rmp = Re_ + dR;

			// jacobian calculation
			const arma::Mat<fltp> dN = n_.n_rows==3 ? 
				cmn::Triangle::shape_function_derivative(Rqg) : 
				cmn::Quadrilateral::shape_function_derivative(Rqg);

			// at carthesian coordinates
			arma::Mat<fltp> dRgc(3,Rqg.n_cols*n_.n_cols);
			arma::Row<fltp> wg(Rqg.n_cols*n_.n_cols);
			for(arma::uword i=0;i<n_.n_cols;i++){
				// nodes
				const arma::Mat<fltp> Rn = Rn_.cols(n_.col(i));

				// calculate relative position with respect to multipole
				dRgc.cols(i*Rqg.n_cols, (i+1)*Rqg.n_cols-1) = Rmp.col(i) - (n_.n_rows==3 ? 
					cmn::Triangle::quad2cart(Rn,Rqg) : 
					cmn::Quadrilateral::quad2cart(Rn,Rqg)).eval().each_col();

				// calculate weight 
				const arma::Mat<fltp> J = n_.n_rows==3 ? 
					cmn::Triangle::shape_function_jacobian(Rn,dN) : 
					cmn::Quadrilateral::shape_function_jacobian(Rn,dN);
				const arma::Row<fltp> Jdet = n_.n_rows==3 ? 
					cmn::Triangle::jacobian2determinant(J) : 
					cmn::Quadrilateral::jacobian2determinant(J);

				// weight
				wg.cols(i*Rqg.n_cols, (i+1)*Rqg.n_cols-1) = Jdet%wqg;
			}

			// set number of expansions and setup matrix
			M_J_.set_num_exp(num_exp);
			M_J_.calc_matrix(-dRgc);
			M_J_.merge_matrix(wg,Rqg.n_cols);
		}
	}

	// get multipole contribution of the sources with indices
	// the contributions of the sources are already summed
	void CurrentSurface::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const ShSettingsPr &stngs) const{
		
		// check input
		assert(first_source.n_elem==last_source.n_elem);
		assert(!Mp.is_empty());

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_s2m()){
			// calculate effective current
			const arma::Mat<fltp> Ieff = Je_.each_row()%Ae_;

			// create gauss points
			const arma::Mat<fltp> gp = n_.n_rows==3 ? 
				cmn::Triangle::create_gauss_points(num_gauss_) : 
				cmn::Quadrilateral::create_gauss_points(num_gauss_);
			const arma::Mat<fltp> Rqg = gp.rows(0,1);
			const arma::Row<fltp> wqg = gp.row(2);

			// position of multipole
			const arma::Mat<fltp> Rmp = Re_ + dR_;

			// jacobian calculation
			const arma::Mat<fltp> dN = n_.n_rows==3 ? 
				cmn::Triangle::shape_function_derivative(Rqg) : 
				cmn::Quadrilateral::shape_function_derivative(Rqg);

			// walk over source elements
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// number of sources in this batch
				const arma::uword num_sources = last_source(i) - first_source(i) + 1;

				// allocate
				arma::Mat<fltp> dRgc(3,Rqg.n_cols*num_sources);
				arma::Row<fltp> wg(Rqg.n_cols*num_sources);

				// walk over sources
				for(arma::uword j=0;j<num_sources;j++){

					// get index
					const arma::uword mysource = first_source(i) + j;

					// nodes
					const arma::Mat<fltp> Rn = Rn_.cols(n_.col(mysource));

					// calculate relative position with respect to multipole
					dRgc.cols(j*Rqg.n_cols, (j+1)*Rqg.n_cols-1) = Rmp.col(mysource) - (n_.n_rows==3 ? 
						cmn::Triangle::quad2cart(Rn,Rqg) : 
						cmn::Quadrilateral::quad2cart(Rn,Rqg)).eval().each_col();

					// calculate weight 
					const arma::Mat<fltp> J = n_.n_rows==3 ? 
						cmn::Triangle::shape_function_jacobian(Rn,dN) : 
						cmn::Quadrilateral::shape_function_jacobian(Rn,dN);
					const arma::Row<fltp> Jdet = n_.n_rows==3 ? 
						cmn::Triangle::jacobian2determinant(J) : 
						cmn::Quadrilateral::jacobian2determinant(J);

					// weight
					wg.cols(j*Rqg.n_cols, (j+1)*Rqg.n_cols-1) = Jdet%wqg;
				}

				// calculate contribution of currents and return calculated multipole
				StMat_So2Mp_J M_J;
				M_J.set_num_exp(num_exp);
				M_J.calc_matrix(-dRgc);
				M_J.merge_matrix(wg,Rqg.n_cols);

				// apply matrix
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J.apply(
					Je_.cols(first_source(i),last_source(i)));
			});
		}

		// faster less memory efficient implementation
		else{
			// walk over source nodes
			cmn::parfor(0,first_source.n_elem,stngs->get_parallel_s2m(),[&](arma::uword i, int){
				// add child source contribution to this multipole
				Mp.cols(i*num_dim_, (i+1)*num_dim_-1) = M_J_.apply(
					Je_.cols(first_source(i),last_source(i)),
					first_source(i),last_source(i));
			});
		}
	}

	// source to target kernel
	void CurrentSurface::source_to_target(
		const ShTargetsPr &tar, const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, const arma::Row<arma::uword> &last_target, 
		const ShSettingsPr &stngs, 
		const rat::cmn::ShLogPr& lg) const{

		// create gauss points
		const arma::Mat<fltp> gp = n_.n_rows==3 ? 
			cmn::Triangle::create_gauss_points(num_gauss_) : 
			cmn::Quadrilateral::create_gauss_points(num_gauss_);
		const arma::Mat<fltp> Rqg = gp.rows(0,1);
		const arma::Row<fltp> wqg = gp.row(2);
		const arma::Mat<fltp> dN = n_.n_rows==3 ? 
			cmn::Triangle::shape_function_derivative(Rqg) : 
			cmn::Quadrilateral::shape_function_derivative(Rqg);

		// setup jacobian determinants, coordinates and weights for the gauss points here
		arma::Mat<fltp> Rs(3,num_elements_*Rqg.n_cols);
		arma::Mat<fltp> Js(3,num_elements_*Rqg.n_cols);
		cmn::parfor(0,num_elements_,stngs->get_parallel_s2t(),[&](arma::uword i, int){
			const arma::Mat<fltp> myRn = Rn_.cols(n_.col(i));
			Rs.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1) = n_.n_rows==3 ? cmn::Triangle::quad2cart(myRn,Rqg) : cmn::Quadrilateral::quad2cart(myRn,Rqg);
			const arma::Mat<fltp> J = n_.n_rows==3 ? cmn::Triangle::shape_function_jacobian(myRn,dN) : cmn::Quadrilateral::shape_function_jacobian(myRn,dN);
			const arma::Row<fltp> Jdet = n_.n_rows==3 ? cmn::Triangle::jacobian2determinant(J) : cmn::Quadrilateral::jacobian2determinant(J);
			Js.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1) = arma::repmat(Je_.col(i),1,Rqg.n_cols).eval().each_row()%(Jdet%wqg);
		});

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// allocate
			arma::Mat<fltp> A(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over source nodes
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
			//for(arma::uword i=0;i<target_list.n_elem;i++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// get target node
				const arma::uword target_idx = target_list(i);

				// get location of the targets
				const arma::uword ft = first_target(target_idx);
				const arma::uword lt = last_target(target_idx);
				assert(ft<=lt); assert(lt<tar->num_targets());

				// my target positions
				const arma::Mat<fltp> myRt = tar->get_target_coords(ft,lt);

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// check cancel flag
					if(lg->is_cancelled())return;

					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source elements
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_elements_);

					// walk over sources in list
					for(arma::uword n=0;n<(ls-fs+1);n++){
						// get source index
						const arma::uword mysource = fs+n;

						// calculate distance of 
						// all target points to this source
						const arma::Row<fltp> dist = cmn::Extra::vec_norm(myRt.each_col() - Re_.col(mysource));

						// find indexes of target points 
						// that are far away
						const arma::Row<arma::uword> indices_far = 
							arma::find(dist>num_dist_a_*element_radius_(mysource)).t();
						const arma::Row<arma::uword> indices_near = 
							arma::find(dist<=num_dist_a_*element_radius_(mysource)).t();

						// run normal calculation for far targets
						if(!indices_far.is_empty()){
							// create source points
							A.cols(ft+indices_far) += Savart::calc_I2A(
								Rs.cols(mysource*Rqg.n_cols,(mysource+1)*Rqg.n_cols-1),
								Js.cols(mysource*Rqg.n_cols,(mysource+1)*Rqg.n_cols-1),
								myRt.cols(indices_far),false);
						}

						// for target points that are close
						// use gauss points to do the integration
						if(!indices_near.is_empty()){
							// use charged polygon analytical equations
							const arma::Row<fltp> phi = ChargedPolyhedron::calc_scalar_potential(Rn_.cols(n_.col(mysource)), myRt.cols(indices_near));
							A.cols(ft+indices_near) += RAT_CONST(1e-7)*Je_.col(mysource)%arma::repmat(phi,3,1).eval().each_col();
						}

					}
				}
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			tar->add_field('A',A,true);
		}

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// allocate
			arma::Mat<fltp> H(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over source nodes
			cmn::parfor(0,target_list.n_elem,stngs->get_parallel_s2t(),[&](arma::uword i, int){
			//for(arma::uword i=0;i<target_list.n_elem;i++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// get target node
				const arma::uword target_idx = target_list(i);

				// get location of the targets
				const arma::uword ft = first_target(target_idx);
				const arma::uword lt = last_target(target_idx);
				assert(ft<=lt); assert(lt<tar->num_targets());

				// my target positions
				const arma::Mat<fltp> myRt = tar->get_target_coords(ft,lt);

				// walk over source nodes
				for(arma::uword j=0;j<source_list(i).n_elem;j++){
					// check cancel flag
					if(lg->is_cancelled())return;

					// get target node
					const arma::uword source_idx = source_list(i)(j);

					// get my source elements
					const arma::uword fs = first_source(source_idx);
					const arma::uword ls = last_source(source_idx);
					assert(fs<=ls); assert(ls<num_elements_);

					// walk over sources in list
					for(arma::uword n=0;n<(ls-fs+1);n++){
						// get source index
						const arma::uword mysource = fs+n;

						// calculate distance of 
						// all target points to this source
						const arma::Row<fltp> dist = cmn::Extra::vec_norm(
							myRt.each_col() - Re_.col(mysource));

						// find indexes of target points 
						// that are far away
						const arma::Row<arma::uword> indices_far = 
							arma::find(dist>num_dist_b_*element_radius_(mysource)).t();
						const arma::Row<arma::uword> indices_near = 
							arma::find(dist<=num_dist_b_*element_radius_(mysource)).t();

						// run normal calculation for far targets
						// use gauss points
						if(!indices_far.is_empty()){
							// create source points
							H.cols(ft+indices_far) += Savart::calc_I2H(
								Rs.cols(mysource*Rqg.n_cols,(mysource+1)*Rqg.n_cols-1),
								Js.cols(mysource*Rqg.n_cols,(mysource+1)*Rqg.n_cols-1),
								myRt.cols(indices_far),false);

							// H.cols(ft+indices_far) += Savart::calc_I2H(Re_.col(mysource),
							// 	Je_.col(mysource)*Ae_(mysource), myRt.cols(indices_far),false);
						}

						// for target points that are close
						// use analytical equation
						if(!indices_near.is_empty()){
							// use charged polygon analytical equations
							const arma::Mat<fltp> Hcharge = ChargedPolyhedron::calc_magnetic_field(
								Rn_.cols(n_.col(mysource)), myRt.cols(indices_near));
							H.cols(ft+indices_near) += cmn::Extra::cross(
								arma::repmat(Je_.col(mysource),1,Hcharge.n_cols), Hcharge)/(4*arma::Datum<fltp>::pi);
						}

					}
				}
			});

			// check cancel flag
			if(lg->is_cancelled())return;

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}
	}

	// direct calculation
	void CurrentSurface::calc_direct(
		const ShTargetsPr &tar, 
		const ShSettingsPr &stngs, 
		const cmn::ShLogPr &lg) const{

		// get targets
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// create gauss points
		const arma::Mat<fltp> gp = n_.n_rows==3 ? 
			cmn::Triangle::create_gauss_points(num_gauss_) : 
			cmn::Quadrilateral::create_gauss_points(num_gauss_);
		const arma::Mat<fltp> Rqg = gp.rows(0,1);
		const arma::Row<fltp> wqg = gp.row(2);
		const arma::Mat<fltp> dN = n_.n_rows==3 ? 
			cmn::Triangle::shape_function_derivative(Rqg) : 
			cmn::Quadrilateral::shape_function_derivative(Rqg);

		// setup jacobian determinants, coordinates and weights for the gauss points here
		arma::Mat<fltp> Rs(3,num_elements_*Rqg.n_cols);
		arma::Mat<fltp> Js(3,num_elements_*Rqg.n_cols);
		cmn::parfor(0,num_elements_,stngs->get_parallel_s2t(),[&](arma::uword i, int){
			const arma::Mat<fltp> myRn = Rn_.cols(n_.col(i));
			Rs.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1) = n_.n_rows==3 ? cmn::Triangle::quad2cart(myRn,Rqg) : cmn::Quadrilateral::quad2cart(myRn,Rqg);
			const arma::Mat<fltp> J = n_.n_rows==3 ? cmn::Triangle::shape_function_jacobian(myRn,dN) : cmn::Quadrilateral::shape_function_jacobian(myRn,dN);
			const arma::Row<fltp> Jdet = n_.n_rows==3 ? cmn::Triangle::jacobian2determinant(J) : cmn::Quadrilateral::jacobian2determinant(J);
			Js.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1) = arma::repmat(Je_.col(i),1,Rqg.n_cols).eval().each_row()%(Jdet%wqg);
		});

		// forward calculation of vector potential to extra
		if(tar->has('A')){
			// allocate
			arma::Mat<fltp> A(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over sources in list
			for(arma::uword n=0;n<num_elements_;n++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// calculate distance of 
				// all target points to this source
				const arma::Mat<fltp> dist = cmn::Extra::vec_norm(Rt.each_col() - Re_.col(n));

				// find indexes of target points 
				// that are far away
				const arma::Row<arma::uword> indices_far = 
					arma::find(dist>num_dist_a_*element_radius_(n)).t();
				const arma::Row<arma::uword> indices_near = 
					arma::find(dist<=num_dist_a_*element_radius_(n)).t();

				// run normal calculation for far targets
				if(!indices_far.is_empty()){
					// add vector potential
					A.cols(indices_far) += Savart::calc_I2A(
						Rs.cols(n*Rqg.n_cols,(n+1)*Rqg.n_cols-1),
						Js.cols(n*Rqg.n_cols,(n+1)*Rqg.n_cols-1),
						Rt.cols(indices_far),stngs->get_parallel_s2t());
				}

				// for target points that are close
				// use gauss points to do the integration
				if(!indices_near.is_empty()){
					// use charged polygon analytical equations
					const arma::Row<fltp> phi = ChargedPolyhedron::calc_scalar_potential(Rn_.cols(n_.col(n)), Rt.cols(indices_near));
					A.cols(indices_near) += RAT_CONST(1e-7)*Je_.col(n)%arma::repmat(phi,3,1).eval().each_col();
				}
			}

			// set field to targets
			tar->add_field('A',A,true);
		}

		// forward calculation of vector potential to extra
		if(tar->has('H') || tar->has('B')){
			// allocate
			arma::Mat<fltp> H(num_dim_,tar->num_targets(),arma::fill::zeros);

			// walk over sources in list
			for(arma::uword n=0;n<num_elements_;n++){
				// check cancel flag
				if(lg->is_cancelled())return;

				// calculate distance of 
				// all target points to this source
				const arma::Mat<fltp> dist = cmn::Extra::vec_norm(Rt.each_col() - Re_.col(n));

				// find indexes of target points 
				// that are far away
				const arma::Row<arma::uword> indices_far = 
					arma::find(dist>num_dist_b_*element_radius_(n)).t();
				const arma::Row<arma::uword> indices_near = 
					arma::find(dist<=num_dist_b_*element_radius_(n)).t();

				// run normal calculation for far targets
				if(!indices_far.is_empty()){
					// add magnetic field
					H.cols(indices_far) += Savart::calc_I2H(
						Rs.cols(n*Rqg.n_cols,(n+1)*Rqg.n_cols-1),
						Js.cols(n*Rqg.n_cols,(n+1)*Rqg.n_cols-1),
						Rt.cols(indices_far),stngs->get_parallel_s2t());
				}

				// for target points that are close
				// use gauss points to do the integration
				if(!indices_near.is_empty()){
					// use charged polygon analytical equations
					const arma::Mat<fltp> Hcharge = ChargedPolyhedron::calc_magnetic_field(
						Rn_.cols(n_.col(n)), Rt.cols(indices_near));
					H.cols(indices_near) += cmn::Extra::cross(
						arma::repmat(Je_.col(n),1,Hcharge.n_cols), Hcharge)/(4*arma::Datum<fltp>::pi);
				}
			}

			// set field to targets
			if(tar->has('H'))tar->add_field('H',H,true);
			if(tar->has('B'))tar->add_field('B',arma::Datum<fltp>::mu_0*H,true);
		}

	}


	// basic build in mesh shape cylinder
	// for testing purposes
	// note that after calling this method
	// it is still necessary to calculate element
	// volumes to complete setup.
	void CurrentSurface::setup_cylinder_shell(
		const fltp R, const fltp height, 
		const arma::uword nz, const arma::uword nl, const fltp J){

		// check user input
		if(height<=0)rat_throw_line("height must be larger than zero");
		if(nz<=1)rat_throw_line("number of axial coordinates must be larger than one");
		if(nl<=1)rat_throw_line("number of azymuthal coordinates must be larger than one");

		// create azymuthal coordinates of nodes
		arma::Row<fltp> theta = arma::linspace<arma::Row<fltp> >(0,-(RAT_CONST(1.0)-RAT_CONST(1.0)/(nl+1))*2*arma::Datum<fltp>::pi,nl);

		// create axial cooridnates of nodes
		arma::Row<fltp> z = arma::linspace<arma::Row<fltp> >(-height/2,height/2,nz);

		// create matrix to hold node coordinates
		arma::Mat<fltp> xn(nl,nz), yn(nl,nz), zn(nl,nz);

		// build node coordinates in two dimensions (first block of matrix)
		for(arma::uword i=0;i<nz;i++){
			// generate circles with different radii
			xn.col(i) = R*arma::cos(theta).t();
			yn.col(i) = R*arma::sin(theta).t();
			zn.col(i).fill(z(i));
		}

		// number of nodes
		num_nodes_ = nl*nz;
		
		// create node coordinates
		Rn_.set_size(3,num_nodes_);
		Rn_.row(0) = arma::reshape(xn,1,num_nodes_);
		Rn_.row(1) = arma::reshape(yn,1,num_nodes_);
		Rn_.row(2) = arma::reshape(zn,1,num_nodes_);

		// create matrix of node indices
		arma::Mat<arma::uword> node_idx = arma::regspace<arma::Mat<arma::uword> >(0,num_nodes_);
		node_idx.reshape(nl,nz);

		// close mesh by setting last row to the first
		node_idx = arma::join_vert(node_idx,node_idx.row(0));

		// get definition of quadrilateral element
		arma::Mat<arma::sword>::fixed<4,2> M = 
			arma::conv_to<arma::Mat<arma::sword> >::from(
			(cmn::Quadrilateral::get_corner_nodes()+1)/2);

		// calculate number of elements
		num_elements_ = nl*(nz-1);

		// allocate elements
		n_.set_size(4,num_elements_);

		// walk over corner nodes
		for(arma::uword k=0;k<4;k++){
			// get matrix indexes
			arma::uword idx0 = M(k,0), idx1 = M(k,0)+nl+1-2;
			arma::uword idx2 = M(k,1), idx3 = M(k,1)+nz-2;

			// get node indexes for this corner
			n_.row(k) = arma::reshape(
				node_idx.submat(arma::span(idx0,idx1),
				arma::span(idx2,idx3)),1,nl*(nz-1));
		}

		// finish setup by calculating volumes and centroids
		calculate_element_areas();
		
		// element orientation	
		arma::Mat<fltp> Ne = arma::join_vert(Re_.rows(0,1),arma::Row<fltp>(num_elements_,arma::fill::zeros));
		Ne = Ne.each_row()/cmn::Extra::vec_norm(Ne);
		arma::Mat<fltp> De(3,num_elements_,arma::fill::zeros); De.row(2).fill(1.0);
		arma::Mat<fltp> Le = cmn::Extra::cross(Ne,De);

		// set currents
		Je_ = -J*Le; 

		// check if volume is correct (within 0.1%)
		assert(((height*2*arma::Datum<fltp>::pi*R) - 
			arma::as_scalar(arma::sum(arma::sum(get_area()))))/
			(height*2*arma::Datum<fltp>::pi*R)<1e-3f);
	}

}}

