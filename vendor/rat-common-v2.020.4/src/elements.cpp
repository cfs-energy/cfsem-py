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
#include "elements.hh"

// common headers
#include "gauss.hh"
#include "extra.hh"
#include "parfor.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// QUADRILATERAL
	// quadrilateral corner node positions 
	// in quadrilateral coordinates
	// matrix columns are given as [xi,nu]
	Quadrilateral::CornerNodeMatrix Quadrilateral::get_corner_nodes(){
		// setup matrix
		arma::Mat<arma::sword>::fixed<4,2> M{
			-1,+1,+1,-1, 
			-1,-1,+1,+1};

		// return matrix
		return M;
	}

	// quadrilateral edges connectivity
	Quadrilateral::EdgeMatrix Quadrilateral::get_edges(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<4,2> M{
			0,1,2,3, 
			1,2,3,0};

		// return matrix
		return M;
	}

	// conversion to triangles
	Quadrilateral::TriangleConversionMatrix Quadrilateral::triangle_conversion_matrix(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<2,3> M{
			{0,1,3}, // first triangle
			{1,2,3}}; // second triangle
				
		// return matrix
		return M;
	}

	// create gauss points
	arma::Mat<fltp> Quadrilateral::create_gauss_points(const arma::sword num_gauss){
		// create gauss points in quad coords
		const arma::uword ng = std::abs(num_gauss);
		const cmn::Gauss gs(num_gauss);
		const arma::Row<fltp>& wg = gs.get_weights();
		const arma::Row<fltp>& xg = gs.get_abscissae();
		arma::Mat<fltp> Rq(3,ng*ng);
		for(arma::uword i=0;i<ng;i++){
			for(arma::uword j=0;j<ng;j++){
				const arma::uword idx = i*ng + j;
				Rq(0,idx) = xg(i);
				Rq(1,idx) = xg(j);
				Rq(2,idx) = wg(i)*wg(j);
			}
		}

		// check weights
		assert(std::abs(arma::accu(Rq.row(2)))-4.0<1e-9);

		// return gauss points and weights
		return Rq;
	}


	// quadrilateral shape function
	arma::Mat<fltp> Quadrilateral::shape_function(
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rq.n_rows==2);

		// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
		arma::Mat<arma::sword>::fixed<4,2> M = Quadrilateral::get_corner_nodes();

		// calculate shape functions
		arma::Mat<fltp> Nie(4,Rq.n_cols);

		// fill matrix
		for(arma::uword i=0;i<4;i++)
			Nie.row(i) = 
				(1+Rq.row(0)*fltp(M(i,0)))%
				(1+Rq.row(1)*fltp(M(i,1)))/4;

		// return 
		return Nie;
	}

	// quadrilateral shape function derivative in quadrilateral coordinates
	// format: [dNdnu;dNdxi]
	arma::Mat<fltp> Quadrilateral::shape_function_derivative(
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rq.n_rows==2);

		// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
		arma::Mat<arma::sword>::fixed<4,2> M = Quadrilateral::get_corner_nodes();

		// calculate shape functions
		arma::Mat<fltp> Nie(8,Rq.n_cols);

		// fill matrix
		for(arma::uword i=0;i<4;i++){
			Nie.row(0+i) = fltp(M(i,0))*(1+Rq.row(1)*fltp(M(i,1)))/4;
			Nie.row(4+i) = fltp(M(i,1))*(1+Rq.row(0)*fltp(M(i,0)))/4;
		}

		// return 
		return Nie;
	}

	// jacobian functions
	arma::Mat<fltp> Quadrilateral::shape_function_jacobian(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<fltp> &dN){

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==4);

		// need to work in plane
		const arma::Mat<fltp>::fixed<3,2> M = surface_to_volume_matrix(Rn);

		// calculate cartesian coordinates in plane
		const arma::Row<fltp> u = M.col(0).t()*Rn;
		const arma::Row<fltp> v = M.col(1).t()*Rn;

		// get shape derivative of shape function
		const arma::Mat<fltp> dNiedxi = dN.rows(0,3); 
		const arma::Mat<fltp> dNiednu = dN.rows(4,7); 

		// setup jacobian
		arma::Mat<fltp> J(4,dN.n_cols);
		J.row(0) = u*dNiedxi;
		J.row(1) = u*dNiednu;
		J.row(2) = v*dNiedxi;
		J.row(3) = v*dNiednu;

		// return jacobian
		return J;
	}

	// convert volume to in-plane
	arma::Mat<fltp>::fixed<3,2> Quadrilateral::surface_to_volume_matrix(const arma::Mat<fltp>&Rn){
		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==4);

		// need to work in plane
		arma::Col<fltp>::fixed<3> V1 = Rn.col(1) - Rn.col(0);
		arma::Col<fltp>::fixed<3> V2 = Rn.col(3) - Rn.col(0);
		V1.each_row()/=Extra::vec_norm(V1);
		V2.each_row()/=Extra::vec_norm(V2);
		arma::Col<fltp>::fixed<3> V3 = Extra::cross(Extra::cross(V1,V2),V1);
		V3.each_row()/=Extra::vec_norm(V3);

		// combine vectors into conversion matrix
		const arma::Mat<fltp>::fixed<3,2> M = arma::join_horiz(V1,V3);

		// return the martix
		return M;
	}

	// calculate determinant for the jacobian matrices
	arma::Row<fltp> Quadrilateral::jacobian2determinant(const arma::Mat<fltp>&J){
		// check input
		assert(J.n_rows==4);

		// calculate determinants
		const arma::Row<fltp> Jdet = J.row(0)%J.row(3) - J.row(2)%J.row(1);

		// return determinants
		return Jdet;
	}

	// invert each jacobian matrix
	arma::Mat<fltp> Quadrilateral::invert_jacobian(const arma::Mat<fltp>&J, const arma::Row<fltp>&Jdet){
		// check input
		assert(J.n_rows==4); assert(J.n_cols==Jdet.n_elem);

		// invert matrices (stored columnwise)
		arma::Mat<fltp> Jinv(4,J.n_cols);
		Jinv.row(0) = J.row(3);
		Jinv.row(1) = -J.row(1);
		Jinv.row(2) = -J.row(2);
		Jinv.row(3) = J.row(0);

		// divide by determinant
		Jinv.each_row()/=Jdet;
		
		// return inverted matrices
		return Jinv;
	}


	// quadrilateral coordinates to carthesian coordinates
	// Rn can also be a different quantity at the nodes for interpolation
	// Rq = [xi;nu], Rc = [x;y;z], Rn = [x;y;z]
	arma::Mat<fltp> Quadrilateral::quad2cart(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rn.n_cols==4); assert(Rq.n_rows==2);
		
		// get shape functions
		arma::Mat<fltp> Nie = Quadrilateral::shape_function(Rq);

		// get coordinates from matrix vector product
		arma::Mat<fltp> Rc = Rn*Nie;

		// return cartesian coords
		return Rc;
	}

	// nedelec 2D shape functions
	arma::Mat<fltp> Quadrilateral::raviart_thomas_hdiv_shape_function(const arma::Mat<fltp> &Rq){
		// check input
		assert(Rq.n_rows==2);

		// get coordinates
		const arma::Row<fltp> xi = Rq.row(0);
		const arma::Row<fltp> nu = Rq.row(1);

		// calculate shape function
		arma::Mat<fltp> N(8,Rq.n_cols,arma::fill::zeros);
		N.row(0*2 + 1) = RAT_CONST(0.5)*nu - RAT_CONST(0.5); // shape function for edge 01, runs from -1 to 0, exits
		N.row(1*2 + 0) = RAT_CONST(0.5) + RAT_CONST(0.5)*xi; // shape function for edge 12, runs from 0 to 1, exits
		N.row(2*2 + 1) = RAT_CONST(0.5) + RAT_CONST(0.5)*nu; // shape function for edge 23, runs from 0 to 1, exits
		N.row(3*2 + 0) = RAT_CONST(0.5)*xi - RAT_CONST(0.5); // shape function for edge 30, runs from -1 to 0, exits

		// return shape function
		// all vectors point outwards
		return -N/2;
	}

	// convert quadrilateral vectors to cartesian vectors 
	// using contravariant piola mapping. This preserves 
	// the tangential component of basis functions.
	// for more info see: https://defelement.com/ciarlet.html
	arma::Mat<fltp> Quadrilateral::quad2cart_contravariant_piola(
		const arma::Mat<fltp> &Vq, 
		const arma::Mat<fltp> &J, 
		const arma::Row<fltp> &Jdet){

		// check input
		assert(Vq.n_cols==J.n_cols);
		assert(J.n_cols==Jdet.n_cols);

		// allocate output
		arma::Mat<fltp> Vc(Vq.n_rows, Vq.n_cols);

		// walk over gauss points
		for(arma::uword j=0;j<Vq.n_cols;j++){
			const arma::Mat<fltp>::fixed<2,2> Ji = arma::reshape(J.col(j),2,2).t();
			Vc.col(j) = arma::vectorise(Ji*arma::reshape(Vq.col(j),2,Vq.n_rows/2));
		}

		// divide by determinant
		Vc.each_row()/=arma::abs(Jdet);

		// return carthesian vectors
		return Vc;
	}


	// convert quadrilateral vectors to cartesian vectors using
	// using covariant piola mapping. This preserves 
	// the normal component of basis functions on facets.
	// for more info see: https://defelement.com/ciarlet.html
	arma::Mat<fltp> Quadrilateral::quad2cart_covariant_piola(
		const arma::Mat<fltp> &Vq, 
		const arma::Mat<fltp> &Jinv){

		// allocate output
		arma::Mat<fltp> Vc(Vq.n_rows,Vq.n_cols);

		// walk over points
		for(arma::uword j=0;j<Vq.n_cols;j++){
			const arma::Mat<fltp>::fixed<2,2> Ji = arma::reshape(Jinv.col(j),2,2);
			Vc.col(j) = arma::vectorise(Ji*arma::reshape(Vq.col(j),2,Vq.n_rows/2));
		}

		// return carthesian vectors
		return Vc;
	}

	// function for checking if surface nodes are in plane
	bool Quadrilateral::check_nodes(
		const arma::Mat<fltp> &Rn,
		const fltp tol){

		// check input
		assert(Rn.n_cols==4); assert(Rn.n_rows==3); assert(tol>0);

		// get typical size
		fltp diag = std::sqrt(arma::as_scalar(arma::sum(
			(Rn.col(0) - Rn.col(2))%(Rn.col(0) - Rn.col(2)),0)));

		// calculate the plane in which the face is located
		arma::Col<fltp>::fixed<3> V1 = Rn.col(1)-Rn.col(0);
		arma::Col<fltp>::fixed<3> V2 = Rn.col(3)-Rn.col(1);

		// get face normal
		arma::Col<fltp>::fixed<3> N = Extra::cross(V1,V2);
		N = N/arma::as_scalar(Extra::vec_norm(N));

		// check if diagonals are in plane
		fltp eps1 = arma::as_scalar(Extra::dot(N,Rn.col(2)-Rn.col(0)));
		fltp eps2 = arma::as_scalar(Extra::dot(N,Rn.col(1)-Rn.col(3)));

		// check flatness (returns true if element valid)
		return (std::abs(eps1)/diag)<tol && (std::abs(eps2)/diag)<tol;
	}


	// carthesian coordinates to quadrilateral coordinates
	// this function can likely be improved by not using finite difference
	// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
	arma::Mat<fltp> Quadrilateral::cart2quad(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rc, 
		const fltp tol){
		
		// check nodes
		assert(Quadrilateral::check_nodes(Rn));

		// get typical size
		fltp diag = std::sqrt(arma::as_scalar(arma::sum(
			(Rn.col(0) - Rn.col(2))%(Rn.col(0) - Rn.col(2)),0)));

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==4);
		assert(Rc.n_rows==3);
		
		// calculate the plane in which the face is located
		// arma::Col<fltp>::fixed<3> R0 = Rn.col(0);
		arma::Col<fltp>::fixed<3> V1 = Rn.col(1)-Rn.col(0);
		arma::Col<fltp>::fixed<3> V2 = Rn.col(3)-Rn.col(0);
		V1 /= arma::as_scalar(Extra::vec_norm(V1)); 
		V2 /= arma::as_scalar(Extra::vec_norm(V2));
		arma::Mat<fltp>::fixed<2,3> M12 = arma::join_vert(V1.t(),V2.t());

		// initial guess
		arma::Mat<fltp> Rq(2,Rc.n_cols,arma::fill::zeros);

		// iterations
		// fltp delta = RAT_CONST(1e-2);
		for(arma::uword i=0;i<100;i++){
			// calculate difference
			const arma::Mat<fltp> Rc00 = M12*Quadrilateral::quad2cart(Rn,Rq);
			const arma::Mat<fltp> dRc = Rc00 - M12*Rc;
			
			// check for convergence
			const fltp conv = arma::max(arma::max(arma::abs(dRc)))/diag;
			if(conv<tol)break;

			// calculate jacobian
			const arma::Mat<fltp> J = shape_function_jacobian(Rn,shape_function_derivative(Rq));
			
			// invert jacobian
			const arma::Row<fltp> Jdet = jacobian2determinant(J);
			const arma::Mat<fltp> Jinv = invert_jacobian(J,Jdet);

			// check
			assert(Jinv.is_finite());

			// shift coordinates
			Rq -= arma::join_vert(
				Jinv.row(0)%dRc.row(0) + Jinv.row(1)%dRc.row(1), 
				Jinv.row(2)%dRc.row(0) + Jinv.row(3)%dRc.row(1));

			// check if fail elements are likely non-planar
			// if(i>100)rat_throw_line("cart2quad does not converge");
		}

		// check
		assert(arma::all(arma::all(arma::abs(M12*Quadrilateral::quad2cart(Rn,Rq) - M12*Rc)<tol)));

		// return coords
		return Rq;
	}

	// calculate sinfgle element area
	fltp Quadrilateral::calc_area(
		const arma::Mat<fltp> &Rn){

		// check input
		assert(Rn.n_cols==4);
		assert(Rn.n_rows==3);

		// extract vectors that span the face from opposing nodes
		const arma::Col<fltp>::fixed<3> V0 = Rn.col(1) - Rn.col(0);
		const arma::Col<fltp>::fixed<3> V1 = Rn.col(3) - Rn.col(0);
		const arma::Col<fltp>::fixed<3> V2 = Rn.col(1) - Rn.col(2);
		const arma::Col<fltp>::fixed<3> V3 = Rn.col(3) - Rn.col(2);

		const arma::Col<fltp>::fixed<3> V01 = Extra::cross(V0,V1);
		const arma::Col<fltp>::fixed<3> V23 = Extra::cross(V2,V3);

		// calculate face area using two triangles
		const fltp A = arma::as_scalar(Extra::vec_norm(V01)/2 + Extra::vec_norm(V23)/2);

		// return element volumes
		return A;
	}

	// calculate element areas for both 2 and 3D meshes
	arma::Row<fltp> Quadrilateral::calc_area(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n){

		// check input
		assert(Rn.n_rows==3 || Rn.n_rows==2);
		assert(n.n_rows==4);

		// extract vectors that span the face from opposing nodes
		const arma::Mat<fltp> V0 = Rn.cols(n.row(1)) - Rn.cols(n.row(0));
		const arma::Mat<fltp> V1 = Rn.cols(n.row(3)) - Rn.cols(n.row(0));
		const arma::Mat<fltp> V2 = Rn.cols(n.row(1)) - Rn.cols(n.row(2));
		const arma::Mat<fltp> V3 = Rn.cols(n.row(3)) - Rn.cols(n.row(2));

		// cross product
		const arma::Mat<fltp> V01 = Rn.n_rows==3 ? Extra::cross(V0,V1) : Extra::cross2D(V0,V1);
		const arma::Mat<fltp> V23 = Rn.n_rows==3 ? Extra::cross(V2,V3) : Extra::cross2D(V2,V3);

		// calculate face area using two triangles
		// note that the V01 and V23 are vectors for 3 dimensions and
		// scalars for 2 dimensions. In the latter case the vecnorm
		// function is not really needed.
		const arma::Row<fltp> A = arma::vecnorm(V01)/2 + arma::vecnorm(V23)/2;

		// return element volumes
		return A;
	}

	// invert elements without reversing them
	arma::Mat<arma::uword> Quadrilateral::invert_elements(const arma::Mat<arma::uword>&n){
		assert(n.n_rows==4);
		return n.rows(arma::Col<arma::uword>::fixed<4>{1,0,3,2});
	}

	// reverse elements without inverting them
	arma::Mat<arma::uword> Quadrilateral::reverse_elements(const arma::Mat<arma::uword>&n){
		return n.rows(arma::Col<arma::uword>::fixed<4>{2,3,0,1});
	}

	// setup a grid around a singular point
	// grid containes nodes in quadrilateral coordinates (Rqgrd)
	// and weights (wgrd)
	void Quadrilateral::setup_source_grid(
		arma::Mat<fltp> &Rqgrd, arma::Row<fltp> &wgrd,
		const arma::Col<fltp>::fixed<2> &Rqs,
		const arma::Row<fltp> &xg, const arma::Row<fltp> &wg){
			
		// check gauss weights
		assert(arma::all(arma::abs(Rqs)<=RAT_CONST(1.0)));
		assert(arma::as_scalar(arma::sum(wg))-RAT_CONST(2.0)<RAT_CONST(1e-5));
		assert(Rqs.n_rows==2); assert(Rqs.n_cols==1);
		assert(xg.n_elem==wg.n_elem);

		// scale gauss points
		arma::Row<fltp> sxg = (xg+1)/2;

		// get lengths of each side
		arma::Col<fltp>::fixed<2> a = Rqs + RAT_CONST(1.0);
		arma::Col<fltp>::fixed<2> b = RAT_CONST(1.0) - Rqs;

		// split face depending on target position
		arma::field<arma::Row<fltp> > x(2), w(2);
		for(arma::uword j=0;j<2;j++){
			if(a(j)>1e-9 && b(j)>RAT_CONST(1e-9)){
				x(j) = arma::join_horiz(a(j)*sxg - RAT_CONST(1.0), b(j)*sxg + Rqs(j));
				w(j) = arma::join_horiz(wg*a(j)/2, wg*b(j)/2);
			}
			else if(a(j)>1e-9 && b(j)<1e-9){
				x(j) = a(j)*sxg - RAT_CONST(1.0); w(j) = wg*a(j)/2;
			}
			else{
				x(j) = b(j)*sxg - RAT_CONST(1.0); w(j) = wg*b(j)/2;
			}
			assert(arma::all(w(j)>0));
		}

		// number of elements in grid
		arma::uword Ngrd = x(0).n_elem * x(1).n_elem;

		// allocate coordinates and weights
		Rqgrd.set_size(2,Ngrd); wgrd.zeros(Ngrd);

		// setup quadrilateral grid coordinates
		for(arma::uword k=0;k<x(0).n_elem;k++){
			for(arma::uword l=0;l<x(1).n_elem;l++){
				// index
				arma::uword idx = k*x(1).n_elem + l;

				// set grid coordinate
				Rqgrd.at(0,idx) = x(0).at(k); 
				Rqgrd.at(1,idx) = x(1).at(l); 
				wgrd.at(idx) = w(0).at(k)*w(1)(l)/4;
			}
		}	

		// sanity check
		assert(arma::all(wgrd>0));
		assert(std::abs(arma::as_scalar(arma::sum(wgrd))-RAT_CONST(1.0))<RAT_CONST(1e-5));

		// check
		assert(arma::all(arma::all(Rqgrd<RAT_CONST(1.0) && Rqgrd>-RAT_CONST(1.0))));
	}


	// function for calculating face normals
	arma::Mat<fltp> Quadrilateral::calc_face_normals(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n){

		// extract vectors that span the face from opposing nodes
		const arma::Mat<fltp> V0 = Rn.cols(n.row(1)) - Rn.cols(n.row(0));
		const arma::Mat<fltp> V1 = Rn.cols(n.row(2)) - Rn.cols(n.row(0));
		const arma::Mat<fltp> V2 = Rn.cols(n.row(1)) - Rn.cols(n.row(2));
		const arma::Mat<fltp> V3 = Rn.cols(n.row(3)) - Rn.cols(n.row(2));

		// cross product
		arma::Mat<fltp> N01 = Extra::cross(V0,V1);
		arma::Mat<fltp> N23 = -Extra::cross(V2,V3);

		// check direction
		// assert((N01-N23).is_zero(1e-8));

		// face normal
		arma::Mat<fltp> Nf = (N01 + N23)/2;

		// normalize
		Nf.each_row() /= Extra::vec_norm(Nf);

		// return face normals
		return Nf; 
	}

	// calculate face normals at nodes
	arma::Mat<fltp> Quadrilateral::calc_node_normals(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n){

		// calculate face normals at elements
		arma::Mat<rat::fltp> Nf = rat::cmn::Quadrilateral::calc_face_normals(Rn,n);

		// interpolate at nodes and average
		arma::Mat<rat::fltp> Nn(3,Rn.n_cols,arma::fill::zeros);
		
		// add to each connected node
		for(arma::uword i=0;i<n.n_rows;i++)Nn.cols(n.row(i)) += Nf;

		// re-normalize
		Nn.each_row() /= rat::cmn::Extra::vec_norm(Nn);

		// return node normals
		return Nn;
	}

	// calculate face centroids
	arma::Mat<fltp> Quadrilateral::calc_face_centroids(
		const arma::Mat<fltp> &Rn, const arma::Mat<arma::uword> &n){
		// check input
		assert(n.n_rows==4);
		assert(n.max()<Rn.n_cols);

		// counters
		const arma::uword num_elements = n.n_cols;

		// allocate
		arma::Mat<fltp> Rf(3,num_elements);

		// walk over elements
		for(arma::uword j=0;j<num_elements;j++)
			Rf.col(j) = arma::mean(Rn.cols(n.col(j)),1);

		// return centroids
		return Rf;
	}

	// convert quadrilateral mesh to triangular mesh
	arma::Mat<arma::uword> Quadrilateral::quad2tri(const arma::Mat<arma::uword> &n){
		// conversion to triangles
		arma::Mat<arma::uword>::fixed<2,3> Mq2t = rat::cmn::Quadrilateral::triangle_conversion_matrix();

		// create triangles
		const arma::Mat<arma::uword> t = arma::reshape(n.rows(arma::vectorise(Mq2t.t())),3,n.n_cols*Mq2t.n_rows);

		// return triangular mesh
		return t;
	}

	// convert quadrilateral mesh to triangular mesh
	// but always keep shorter diagonal
	arma::Mat<arma::uword> Quadrilateral::quad2tri(const arma::Mat<arma::uword> &n, const arma::Mat<fltp>& Rn){
		// calculate length of diagonal
		const arma::Row<fltp> ell1 = Extra::vec_norm(Rn.cols(n.row(0)) - Rn.cols(n.row(2)));
		const arma::Row<fltp> ell2 = Extra::vec_norm(Rn.cols(n.row(1)) - Rn.cols(n.row(3)));

		// copy quad mesh and shift cases with diagonal from 1-3
		arma::Mat<arma::uword> nshifted = n;
		const arma::Col<arma::uword> idx = arma::find(ell2>ell1);
		nshifted.cols(idx) = arma::shift(nshifted.cols(idx),1);

		// return triangular mesh
		return quad2tri(nshifted);
	}

	// find edges
	arma::Mat<arma::uword> Quadrilateral::find_edges(
		const arma::Mat<fltp> &Rn, const arma::Mat<arma::uword> &n, 
		const fltp angle_treshold){

		// get face normals
		const arma::Mat<fltp> Nf = Quadrilateral::calc_face_normals(Rn,n);

		// find all unique edge elements
		arma::Mat<arma::uword>::fixed<4,2> Me = Quadrilateral::get_edges();
		arma::Mat<arma::uword> edge_nodes(2,n.n_cols*Me.n_rows);
		arma::Row<arma::uword> edge_faces(n.n_cols*Me.n_rows);
		arma::Row<arma::uword> edge_index(n.n_cols*Me.n_rows);
		for(arma::uword i=0;i<Me.n_rows;i++){
			edge_nodes.cols(i*n.n_cols,(i+1)*n.n_cols-1) = n.rows(Me.row(i));
			edge_faces.cols(i*n.n_cols,(i+1)*n.n_cols-1) = arma::regspace<arma::Row<arma::uword> >(0,n.n_cols-1);
			edge_index.cols(i*n.n_cols,(i+1)*n.n_cols-1).fill(i);
		}

		// sort edges to ensure same direction
		edge_nodes = arma::sort(edge_nodes,"ascend",0);

		// sort columns
		for(arma::uword i=0;i<Me.n_cols;i++){
			const arma::Col<arma::uword> sorting_index = 
				arma::stable_sort_index(edge_nodes.row(i));
			edge_nodes = edge_nodes.cols(sorting_index);
			edge_faces = edge_faces.cols(sorting_index);
			edge_index = edge_index.cols(sorting_index);
		}

		// index
		const arma::Col<arma::uword> duplicate_indices = arma::find(
			arma::all(edge_nodes.head_cols(edge_nodes.n_cols-1) == 
			edge_nodes.tail_cols(edge_nodes.n_cols-1),0));

		// get adjacent face normals
		const arma::Mat<rat::fltp> N1 = Nf.cols(edge_faces(duplicate_indices));
		const arma::Mat<rat::fltp> N2 = Nf.cols(edge_faces(duplicate_indices+1));

		// calculate angles
		const arma::Row<rat::fltp> angle = arma::acos(
			arma::clamp(rat::cmn::Extra::dot(N1,N2)/
			(rat::cmn::Extra::vec_norm(N1)%
				rat::cmn::Extra::vec_norm(N2)),
			RAT_CONST(-1.0),RAT_CONST(1.0)) );

		// // find edges with angle above treshold
		// // rat::fltp edge_tresh = RAT_CONST(20.0)*2*arma::Datum<rat::fltp>::pi/RAT_CONST(360.0);
		// const arma::Col<arma::uword> edge_indices = 
		// 	duplicate_indices(arma::find(angle>angle_treshold));

		// find edges that have no duplicate
		arma::Row<arma::uword> idx_true_edge(edge_nodes.n_cols,arma::fill::ones);
		idx_true_edge(duplicate_indices).fill(0); idx_true_edge(duplicate_indices+1).fill(0);

		// find edges with angle above treshold
		// rat::fltp edge_tresh = RAT_CONST(20.0)*2*arma::Datum<rat::fltp>::pi/RAT_CONST(360.0);
		const arma::Col<arma::uword> edge_indices = 
			arma::join_vert(duplicate_indices(arma::find(angle>=angle_treshold)),arma::find(idx_true_edge));


		// return edges
		return edge_nodes.cols(edge_indices);
	}

	// split mesh on edges
	arma::Row<arma::uword> Quadrilateral::separate_smooth_surfaces(
		arma::Mat<fltp> &Rn, arma::Mat<arma::uword> &n, 
		const fltp angle_treshold){

		// get face normals
		const arma::Mat<fltp> Nf = Quadrilateral::calc_face_normals(Rn,n);

		// find all unique edge elements
		arma::Mat<arma::uword>::fixed<4,2> Me = Quadrilateral::get_edges();
		arma::Mat<arma::uword> edge_nodes(2,n.n_cols*Me.n_rows);
		arma::Row<arma::uword> edge_faces(n.n_cols*Me.n_rows);
		arma::Row<arma::uword> edge_index(n.n_cols*Me.n_rows);
		for(arma::uword i=0;i<Me.n_rows;i++){
			edge_nodes.cols(i*n.n_cols,(i+1)*n.n_cols-1) = n.rows(Me.row(i));
			edge_faces.cols(i*n.n_cols,(i+1)*n.n_cols-1) = arma::regspace<arma::Row<arma::uword> >(0,n.n_cols-1);
			edge_index.cols(i*n.n_cols,(i+1)*n.n_cols-1).fill(i);
		}

		// sort edges to ensure same direction
		edge_nodes = arma::sort(edge_nodes,"ascend",0);

		// sort columns
		for(arma::uword i=0;i<Me.n_cols;i++){
			const arma::Col<arma::uword> sorting_index = 
				arma::stable_sort_index(edge_nodes.row(i));
			edge_nodes = edge_nodes.cols(sorting_index);
			edge_faces = edge_faces.cols(sorting_index);
			edge_index = edge_index.cols(sorting_index);
		}

		// index
		const arma::Col<arma::uword> duplicate_indices = arma::find(
			arma::all(edge_nodes.head_cols(edge_nodes.n_cols-1) == 
			edge_nodes.tail_cols(edge_nodes.n_cols-1),0));

		// get adjacent face normals
		const arma::Mat<rat::fltp> N1 = Nf.cols(edge_faces(duplicate_indices));
		const arma::Mat<rat::fltp> N2 = Nf.cols(edge_faces(duplicate_indices+1));

		// calculate angles
		// arma::Row<rat::fltp> angle = arma::acos(
		// 	arma::clamp(rat::cmn::Extra::dot(N1,N2)/
		// 	(rat::cmn::Extra::vec_norm(N1)%
		// 		rat::cmn::Extra::vec_norm(N2)),
		// 	RAT_CONST(-1.0),RAT_CONST(1.0)) );

		const arma::Row<rat::fltp> x = 
			arma::clamp(rat::cmn::Extra::dot(N1,N2)/
			(rat::cmn::Extra::vec_norm(N1)%
				rat::cmn::Extra::vec_norm(N2)),
			RAT_CONST(-1.0),RAT_CONST(1.0));
		const arma::Row<rat::fltp> angle = (-RAT_CONST(0.69813170079773212)*x%x-RAT_CONST(0.87266462599716477))%x + RAT_CONST(1.5707963267948966);

		// find edges with angle above treshold
		// rat::fltp edge_tresh = RAT_CONST(20.0)*2*arma::Datum<rat::fltp>::pi/RAT_CONST(360.0);
		const arma::Col<arma::uword> edge_indices = 
			duplicate_indices(arma::find(angle<angle_treshold));

		// get the adjacent faces on the dges
		const arma::Row<arma::uword> face1 = edge_faces.cols(edge_indices);
		const arma::Row<arma::uword> face2 = edge_faces.cols(edge_indices+1);
		const arma::Row<arma::uword> edge1 = edge_index.cols(edge_indices);
		const arma::Row<arma::uword> edge2 = edge_index.cols(edge_indices+1);

		// keep track of the original node index
		// arma::Mat<arma::uword> original_index(4,Rn.n_cols);
		// original_index.each_row() = arma::regspace<arma::Row<arma::uword> >(0,Rn.n_cols-1);
		arma::Row<arma::uword> original_index = arma::vectorise(n).t();

		// split all quads
		Rn = Rn.cols(arma::vectorise(n));
		n = arma::reshape(arma::regspace<arma::Col<arma::uword> >(0,n.n_elem-1),4,n.n_cols);

		// // merge nodes
		// for(arma::uword i=0;i<face1.n_elem;i++){
		// 	for(arma::uword j=0;j<Me.n_cols;j++){
		// 		n(Me(edge1(i),j),face1(i)) = n(Me(edge2(i),1-j),face2(i));
		// 	}
		// }

		// merge nodes
		for(;;){
			bool change = false;
			for(arma::uword i=0;i<face1.n_elem;i++){
				for(arma::uword j=0;j<Me.n_cols;j++){
					if(n(Me(edge1(i),j),face1(i)) != n(Me(edge2(i),1-j),face2(i))){
						// find smallest index
						const arma::uword idx = std::min(n(Me(edge1(i),j),face1(i)), n(Me(edge2(i),1-j),face2(i)));
							
						// set index to both nodes
						n(Me(edge1(i),j),face1(i)) = idx; n(Me(edge2(i),1-j),face2(i)) = idx; 

						// flag change
						change = true;
					}
				}
			}
			if(!change)break;
		}

		// count node references
		arma::Row<arma::uword> cnt(Rn.n_cols,arma::fill::zeros);
		for(arma::uword i=0;i<n.n_elem;i++)cnt(n(i))++;
		const arma::Row<arma::uword> active_nodes = 
			arma::find(cnt>0).t();

		// re-index
		arma::Row<arma::uword> re_index(Rn.n_cols, arma::fill::zeros);
		re_index.cols(active_nodes) = arma::regspace<arma::Row<arma::uword> >(0,active_nodes.n_elem-1);

		// remove unused nodes
		n = arma::reshape(re_index(arma::vectorise(n)),4,n.n_cols); 
		Rn = Rn.cols(active_nodes);

		original_index = original_index.cols(active_nodes);

		// return the original index
		return original_index;
	}

	// extrapolate values from gauss points to nodes
	// using the discrete local smoothing technique for a quadrilateral element
	arma::Mat<fltp> Quadrilateral::gauss2nodes(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<fltp> &Rqg,
		const arma::Row<fltp> &wg,
		const arma::Mat<fltp> &Vg){

		// check input dimensions: for quadrilaterals, reference coordinates are 2D.
		assert(Rqg.n_rows == 2);
		assert(Rqg.n_cols == Vg.n_cols);

		// compute the derivatives of the shape functions at the gauss points.
		const arma::Mat<fltp> dN = shape_function_derivative(Rqg);
		// compute the Jacobian matrix at each Gauss point from nodal coordinates.
		const arma::Mat<fltp> J = shape_function_jacobian(Rn, dN);
		// compute determinants of the Jacobian (e.g., area scaling).
		const arma::Row<fltp> Jdet = jacobian2determinant(J);

		// evaluate the shape functions at the gauss points.
		const arma::Mat<fltp> Ng = shape_function(Rqg);

		// create conversion matrices.
		// here M will be a 4x4 matrix (4 nodes per quadrilateral).
		arma::Mat<fltp>::fixed<4, 4> M(arma::fill::zeros);
		// P is a 4 x numGaussPts matrix.
		arma::Mat<fltp> P(4, Rqg.n_cols);
		for (arma::uword i = 0; i < 4; i++) {
			for (arma::uword k = 0; k < Rqg.n_cols; k++) {
				for (arma::uword j = 0; j < 4; j++) {
					M(i, j) += Ng(i, k) * Ng(j, k) * Jdet(k) * wg(k);
				}
				// The comment below reflects that in standard quadrilateral coordinates,
				// the edge length of the reference element is 2 (from -1 to 1).
				P(i, k) = Ng(i, k) * Jdet(k) * wg(k); // the 2 is needed here because in quadrilateral coordinates we have edges with length 2
			}
		}

		// regularize the conversion matrix M to avoid issues with nearly singular matrices.
		const fltp condM = arma::cond(M);
		const fltp eps_factor = (condM > static_cast<fltp>(1e8)) ? static_cast<fltp>(1e-3) : static_cast<fltp>(1e-6);
		const fltp epsilon = eps_factor * M.diag().max();
		M.diag() += epsilon;

		// allocate the matrix for nodal values.
		arma::Mat<fltp> Vn(Vg.n_rows, 4, arma::fill::zeros);
		for (arma::uword m = 0; m < Vg.n_rows; m++) {
			arma::Col<fltp> x;
			// Solve the linear system (M * x = P * Vg.row(m).t()) for each quantity.
			if (arma::solve(x, M, P * Vg.row(m).t(), arma::solve_opts::force_approx))
				Vn.row(m) = x.t();
		}

		// return the extrapolated values at the element nodes.
		return Vn;
	}

	// convert gauss to nodes for the entire mesh
	arma::Mat<fltp> Quadrilateral::gauss2nodes(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Vg,
		const arma::Mat<fltp> &Rqg,
		const arma::Row<fltp> &wg, 
		const arma::uword num_nodes){

		// check input dimensions: for quadrilaterals, reference coordinates are 2D.
		assert(Rqg.n_rows == 2);
		assert(n.n_rows == 4);
		assert(Vg.n_cols % Rqg.n_cols == 0);
		assert(n.max() < num_nodes);

		// allocate output. Each element contributes 4 node values.
		arma::Mat<fltp> Vnn(Vg.n_rows, 4 * n.n_cols);

		// loop (in parallel) over elements and apply local smoothing.
		parfor(0, n.n_cols, true, [&](arma::uword i, int) {
			Vnn.cols(i * 4, (i + 1) * 4 - 1) =
				gauss2nodes(Rn.cols(n.col(i)), Rqg, wg, Vg.cols(i * Rqg.n_cols, (i + 1) * Rqg.n_cols - 1));
		});

		// calculate the area of each quadrilateral element.
		const arma::Row<fltp> Ae = calc_area(Rn, n);

		// average the contributions at each global node.
		arma::Mat<fltp> Vn(Vg.n_rows, num_nodes, arma::fill::zeros);
		arma::Row<fltp> cnt(1, num_nodes, arma::fill::zeros);
		for (arma::uword i = 0; i < n.n_cols; i++) {
			// Add the weighted nodal contributions for element i.
			Vn.cols(n.col(i)) += Vnn.cols(i * 4, (i + 1) * 4 - 1) * Ae(i);
			// Count the total weight (here the element area) at each node.
			cnt(n.col(i)) += Ae(i);
		}

		// perform the weighted average and return values at nodes.
		return Vn.each_row() / cnt;
	}


	// HEXAHEDRON
	// hexahedron corner node positions 
	// in quadrilateral coordinates
	// matrix columns are given as [xi,nu,mu]
	Hexahedron::CornerNodeMatrix Hexahedron::get_corner_nodes(){
		// setup matrix
		arma::Mat<arma::sword>::fixed<8,3> M{ 
			-1,+1,+1,-1,-1,+1,+1,-1,
			-1,-1,+1,+1,-1,-1,+1,+1,
			-1,-1,-1,-1,+1,+1,+1,+1};

		// return matrix
		return M;
	}

	// matrix for converting a hexahedron into five tetrahedrons
	Hexahedron::TetrahedronConversionMatrix Hexahedron::tetrahedron_conversion_matrix(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<5,4> M{ 
			{0,4,5,7}, // first thetahedron
			{0,1,2,5}, // second thetahedron
			{0,2,3,7}, // third thetahedron
			{2,5,6,7}, // fourth thetahedron
			{0,5,2,7}}; // fifth thetahedron (the central larger one)

		// return matrix
		return M;
	}

	// matrix for converting a hexahedron 
	// plus central node into twelve tetrahedrons
	Hexahedron::SpecialTetrahedronConversionMatrix Hexahedron::tetrahedron_conversion_matrix_special(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<12,4> M{ 
			{8,0,1,2},
			{8,0,3,2},
			{8,4,5,6},
			{8,4,7,6},
			{8,2,1,5},
			{8,2,6,5},
			{8,0,1,5},
			{8,0,4,5},
			{8,0,3,7},
			{8,0,4,7},
			{8,3,2,6},
			{8,3,7,6}};

		// return matrix
		return M;
	}

	// hexahedron face matrix
	Hexahedron::FaceMatrix Hexahedron::get_faces(){
		// setup matrix
		// faces is counter clockwise order (as seen from outside)
		arma::Mat<arma::uword>::fixed<6,4> M{
			{0,1,5,4},
			{1,2,6,5},
			{2,3,7,6},
			{3,0,4,7},
			{0,3,2,1},
			{4,5,6,7}};

		// return matrix
		return M;
	}

	// hexahedron edges
	Hexahedron::EdgeMatrix Hexahedron::get_edges(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<12,2> M{
			0,1,2,3, 0,1,2,3, 4,5,6,7,
			1,2,3,0, 4,5,6,7, 5,6,7,4};

		// return matrix
		return M;
	}

	// matrix containing the edge indices in the element
	// but sorted per face, every edge is referenced twice
	Hexahedron::Face2EdgeMatrix Hexahedron::get_face2edge(){
		return arma::reshape(arma::Col<arma::uword>{
			0,5,8,4, 1,6,9,5, 2,7,10,6, 3,7,11,4, 3,2,1,0, 8,9,10,11},4,6).t();
	}

	// hexahedron edges connected to each face (column)
	Hexahedron::FaceEdgeMatrix Hexahedron::get_face_edges(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<4,6> M{
			0,5,8,4,
			1,6,9,5,
			2,7,10,6,
			3,4,11,7,
			3,2,1,0,
			8,9,10,11};

		// return matrix
		return M;
	}

	// hexahedron edges connected to each face
	// negative means that the edge is connected in reverse
	Hexahedron::FaceEdgeReverseMatrix Hexahedron::get_face_edges_reverse(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<4,6> M{
			1,1,0,0,
			1,1,0,0,
			1,1,0,0,
			1,1,0,0,
			0,0,0,0,
			1,1,1,1};

		// return matrix
		return M;
	}

	// dimensions in quadrilateral coordinates in which the face is located
	Hexahedron::FaceNormalMatrix Hexahedron::get_facenormal(){
		// setup matrix
		// faces is counter clockwise order (as seen from outside)
		arma::Mat<arma::sword>::fixed<6,3> M{
			{0,-1,0},
			{+1,0,0},
			{0,+1,0},
			{-1,0,0},
			{0,0,-1},
			{0,0,+1}};

		// return matrix
		return M;
	}

	// dimension in which the facenormal is pointing
	Hexahedron::FaceNormalDirVector Hexahedron::get_facenormal_dir(){
		return arma::Col<arma::uword>::fixed<6>{1,0,1,0,2,2};
	}

	// dimensions in quadrilateral coordinates in which the face is located
	Hexahedron::FaceDimMatrix Hexahedron::get_facedim(){
		// setup matrix
		// dimensions in which the faces are spanned
		arma::Mat<arma::uword>::fixed<6,3> M{
			0,1,0,1,0,0,
			2,2,2,2,1,1,
			1,0,1,0,2,2}; // fixed

		// return maxtrix
		return M;
	}

	// hexahedron face center node positions 
	// in quadrilateral coordinates
	// matrix columns are given as [xi,nu,mu]
	Hexahedron::FaceNodeMatrix Hexahedron::get_face_nodes(){
		// setup matrix
		arma::Mat<arma::sword>::fixed<6,3> M{
			0,1,0,-1,0,0,
			-1,0,1,0,0,0,
			0,0,0,0,-1,1};

		// return matrix
		return M;
	}

	// dimensions in quadrilateral coordinates in which the face is located
	Hexahedron::FaceDirVector Hexahedron::get_facedir(){
		return arma::Col<arma::sword>::fixed<6>{-1,+1,+1,-1,-1,+1}; 
	}

	// refine mesh
	void Hexahedron::refine_mesh_to_tetrahedron(arma::Mat<fltp>&Rn, arma::Mat<arma::uword>&n){
		// check input
		assert(Rn.n_rows==3); assert(n.n_rows==8);

		// check for inverted elements
		{
			const arma::Col<arma::uword> idx = arma::find(is_clockwise(Rn,n)==0);
			n.cols(idx) = invert_elements(n.cols(idx));
		}

		// get points
		const arma::Mat<fltp> R0 = Rn.cols(n.row(0));
		const arma::Mat<fltp> R1 = Rn.cols(n.row(1));
		const arma::Mat<fltp> R2 = Rn.cols(n.row(2));
		const arma::Mat<fltp> R3 = Rn.cols(n.row(3));
		const arma::Mat<fltp> R4 = Rn.cols(n.row(4));
		const arma::Mat<fltp> R5 = Rn.cols(n.row(5));
		const arma::Mat<fltp> R6 = Rn.cols(n.row(6));
		const arma::Mat<fltp> R7 = Rn.cols(n.row(7));

		// center point
		const arma::Mat<fltp> Rm = (R0 + R1 + R2 + R3 + R4 + R5 + R6 + R7)/8;

		// create points on face centers
		const arma::Mat<fltp> Rf0 = (R0 + R1 + R5 + R4)/4;
		const arma::Mat<fltp> Rf1 = (R1 + R2 + R6 + R5)/4;
		const arma::Mat<fltp> Rf2 = (R2 + R3 + R7 + R6)/4;
		const arma::Mat<fltp> Rf3 = (R3 + R0 + R4 + R7)/4;
		const arma::Mat<fltp> Rf4 = (R0 + R3 + R2 + R1)/4;
		const arma::Mat<fltp> Rf5 = (R4 + R5 + R6 + R7)/4;

		// form new tetrahedrons at corners
		const arma::Mat<fltp> Rt00 = arma::reshape(arma::join_vert(Rf0,R0,R1,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt01 = arma::reshape(arma::join_vert(Rf0,R1,R5,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt02 = arma::reshape(arma::join_vert(Rf0,R5,R4,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt03 = arma::reshape(arma::join_vert(Rf0,R4,R0,Rm),3,4*n.n_cols);

		const arma::Mat<fltp> Rt10 = arma::reshape(arma::join_vert(Rf1,R1,R2,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt11 = arma::reshape(arma::join_vert(Rf1,R2,R6,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt12 = arma::reshape(arma::join_vert(Rf1,R6,R5,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt13 = arma::reshape(arma::join_vert(Rf1,R5,R1,Rm),3,4*n.n_cols);

		const arma::Mat<fltp> Rt20 = arma::reshape(arma::join_vert(Rf2,R2,R3,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt21 = arma::reshape(arma::join_vert(Rf2,R3,R7,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt22 = arma::reshape(arma::join_vert(Rf2,R7,R6,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt23 = arma::reshape(arma::join_vert(Rf2,R6,R2,Rm),3,4*n.n_cols);

		const arma::Mat<fltp> Rt30 = arma::reshape(arma::join_vert(Rf3,R3,R0,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt31 = arma::reshape(arma::join_vert(Rf3,R0,R4,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt32 = arma::reshape(arma::join_vert(Rf3,R4,R7,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt33 = arma::reshape(arma::join_vert(Rf3,R7,R3,Rm),3,4*n.n_cols);

		const arma::Mat<fltp> Rt40 = arma::reshape(arma::join_vert(Rf4,R0,R3,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt41 = arma::reshape(arma::join_vert(Rf4,R3,R2,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt42 = arma::reshape(arma::join_vert(Rf4,R2,R1,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt43 = arma::reshape(arma::join_vert(Rf4,R1,R0,Rm),3,4*n.n_cols);

		const arma::Mat<fltp> Rt50 = arma::reshape(arma::join_vert(Rf5,R4,R5,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt51 = arma::reshape(arma::join_vert(Rf5,R5,R6,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt52 = arma::reshape(arma::join_vert(Rf5,R6,R7,Rm),3,4*n.n_cols);
		const arma::Mat<fltp> Rt53 = arma::reshape(arma::join_vert(Rf5,R7,R4,Rm),3,4*n.n_cols);

		// combine
		Rn = arma::join_horiz(
			arma::join_horiz(arma::join_horiz(Rt00,Rt01,Rt02,Rt03), arma::join_horiz(Rt10,Rt11,Rt12,Rt13), arma::join_horiz(Rt20,Rt21,Rt22,Rt23)),
			arma::join_horiz(arma::join_horiz(Rt30,Rt31,Rt32,Rt33), arma::join_horiz(Rt40,Rt41,Rt42,Rt43), arma::join_horiz(Rt50,Rt51,Rt52,Rt53)));

		// combine nodes in a mesh
		n = Extra::combine_nodes(Rn);
		n = arma::reshape(n,4,n.n_elem/4);

		// check for inverted elements
		{
			const arma::Col<arma::uword> idx = arma::find(Tetrahedron::is_clockwise(Rn,n)==0);
			n.cols(idx) = Tetrahedron::invert_elements(n.cols(idx));
		}

		// done
		return;
	}

	// create gauss points
	arma::Mat<fltp> Hexahedron::create_gauss_points(const arma::sword num_gauss){
		// create gauss points in quad coords
		const arma::uword ng = std::abs(num_gauss);
		const cmn::Gauss gs(num_gauss);
		const arma::Row<fltp>& wg = gs.get_weights();
		const arma::Row<fltp>& xg = gs.get_abscissae();

		// fill in coordinae and weights
		arma::Mat<fltp> Rq(4,ng*ng*ng);
		for(arma::uword i=0;i<ng;i++){
			for(arma::uword j=0;j<ng;j++){
				for(arma::uword k=0;k<ng;k++){
					const arma::uword idx = i*ng*ng + j*ng + k;
					Rq(0,idx) = xg(i);
					Rq(1,idx) = xg(j);
					Rq(2,idx) = xg(k);
					Rq(3,idx) = wg(i)*wg(j)*wg(k);
				}
			}
		}

		// check weights
		assert(std::abs(arma::accu(Rq.row(3))-8.0)<RAT_CONST(1e-4));

		// return gauss points and weights
		return Rq;
	}

	// calculate element skewedness as a measure of element quality
	arma::Row<fltp> Hexahedron::calc_skewedness(
		const arma::Mat<fltp>&Rn, 
		const arma::Mat<arma::uword>&n){

		// check input
		assert(n.n_rows==8);

		// allocate output
		arma::Row<fltp> skewedness(n.n_cols,arma::fill::ones);

		// get list of faces for each element
		arma::Mat<arma::uword>::fixed<6,4> M = cmn::Hexahedron::get_faces();

		// walk over edges
		for(arma::uword i=0;i<M.n_rows;i++){
			// get face
			const arma::Mat<arma::uword> f = n.rows(M.row(i));

			// walk over corners
			for(arma::uword j=0;j<f.n_rows;j++){
				// shifted indices
				const arma::uword idx1 = j;
				const arma::uword idx2 = (j+1)%f.n_rows;
				const arma::uword idx3 = (j+2)%f.n_rows;

				// get edge vectors
				const arma::Mat<fltp> V1 = Rn.cols(f.row(idx2)) - Rn.cols(f.row(idx1));
				const arma::Mat<fltp> V2 = Rn.cols(f.row(idx3)) - Rn.cols(f.row(idx2));

				// calculate skew and take minimum value for this element
				skewedness = arma::min(skewedness, arma::sin(arma::acos(cmn::Extra::dot(V1,V2)/(cmn::Extra::vec_norm(V1)%cmn::Extra::vec_norm(V2)))));
			}
		}
		
		// return the skewedness
		return skewedness;
	}

	// calculate element skewedness as a measure of element quality
	arma::Row<fltp> Hexahedron::calc_aspect_ratio(
		const arma::Mat<fltp>&Rn, 
		const arma::Mat<arma::uword>&n){

		// check input
		assert(n.n_rows==8);

		// allocate output
		arma::Row<fltp> shortest_edge(n.n_cols,arma::fill::value(1e99));
		arma::Row<fltp> longest_edge(n.n_cols,arma::fill::zeros);

		// get list of faces for each element
		arma::Mat<arma::uword>::fixed<12,2> E = cmn::Hexahedron::get_edges();

		// walk over edges
		for(arma::uword i=0;i<E.n_rows;i++){
			// calculate edge lengths
			const arma::Row<fltp> ell = cmn::Extra::vec_norm(Rn.cols(n.row(E(i,1))) - Rn.cols(n.row(E(i,0))));

			// find shortest and longest edges
			shortest_edge = arma::min(shortest_edge, ell);
			longest_edge = arma::max(longest_edge, ell);
		}
		
		// return the aspect ratio
		return longest_edge/shortest_edge;
	}

	// hexahedron shape function
	arma::Mat<fltp> Hexahedron::shape_function(
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rq.n_rows==3);

		// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
		arma::Mat<arma::sword>::fixed<8,3> M = Hexahedron::get_corner_nodes();

		// calculate shape functions
		arma::Mat<fltp> Nie(8,Rq.n_cols);

		// fill matrix
		for(arma::uword i=0;i<8;i++){
			Nie.row(i) = 
				(RAT_CONST(1.0)+Rq.row(0)*fltp(M(i,0)))%
				(RAT_CONST(1.0)+Rq.row(1)*fltp(M(i,1)))%
				(RAT_CONST(1.0)+Rq.row(2)*fltp(M(i,2)))/8;
		}

		// return 
		return Nie;
	}

	// partial derivatives of hexahedron shape function in quadrilateral coordinates
	// format: [dNdnu;dNdmu;dNdxi]
	arma::Mat<fltp> Hexahedron::shape_function_derivative(
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rq.n_rows==3);

		// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
		const arma::Mat<arma::sword>::fixed<8,3> M = Hexahedron::get_corner_nodes();

		// calculate shape functions
		arma::Mat<fltp> dNie(24,Rq.n_cols);

		// get components
		const arma::Row<fltp> xi = Rq.row(0);
		const arma::Row<fltp> nu = Rq.row(1);
		const arma::Row<fltp> mu = Rq.row(2);

		// fill matrix
		for(arma::uword i=0;i<8;i++){
			dNie.row( 0+i) = fltp(M(i,0)) * (RAT_CONST(1.0)+nu*fltp(M(i,1))) % (RAT_CONST(1.0)+mu*fltp(M(i,2))) / 8;
			dNie.row( 8+i) = fltp(M(i,1)) * (RAT_CONST(1.0)+xi*fltp(M(i,0))) % (RAT_CONST(1.0)+mu*fltp(M(i,2))) / 8;
			dNie.row(16+i) = fltp(M(i,2)) * (RAT_CONST(1.0)+xi*fltp(M(i,0))) % (RAT_CONST(1.0)+nu*fltp(M(i,1))) / 8;
		}

		// return 
		return dNie;
	}

	// jacobian of the shape function given an quadrilateral coordinates
	//     [xn*dndxi,yn*dndxi,zn*dndxi]
	// J = |xn*dndnu,yn*dndnu,zn*dndnu|
	//     [xn*dndmu,yn*dndmu,zn*dndmu]
	// output jacobian stored column wise
	arma::Mat<fltp> Hexahedron::shape_function_jacobian(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<fltp> &dN){

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==8);

		// get shape derivative of shape function
		const arma::Mat<fltp> dNiedxi = dN.rows(0,7); 
		const arma::Mat<fltp> dNiednu = dN.rows(8,15); 
		const arma::Mat<fltp> dNiedmu = dN.rows(16,23);

		// get node coordinate components
		const arma::Row<fltp> xn = Rn.row(0);
		const arma::Row<fltp> yn = Rn.row(1);
		const arma::Row<fltp> zn = Rn.row(2);

		// setup jacobian
		arma::Mat<fltp> J(9,dN.n_cols);
		J.row(0) = xn*dNiedxi; J.row(1) = xn*dNiednu; J.row(2) = xn*dNiedmu;
		J.row(3) = yn*dNiedxi; J.row(4) = yn*dNiednu; J.row(5) = yn*dNiedmu;
		J.row(6) = zn*dNiedxi; J.row(7) = zn*dNiednu; J.row(8) = zn*dNiedmu;

		// return jacobian
		return J;
	}

	// convert quadrilateral vectors to cartesian vectors 
	// using contravariant piola mapping. This preserves 
	// the tangential component of basis functions.
	// for more info see: https://defelement.com/ciarlet.html
	arma::Mat<fltp> Hexahedron::quad2cart_contravariant_piola(
		const arma::Mat<fltp> &Vq, 
		const arma::Mat<fltp> &J, 
		const arma::Row<fltp> &Jdet){

		// allocate output
		arma::Mat<fltp> Vc(Vq.n_rows, Vq.n_cols);

		// walk over gauss points
		for(arma::uword j=0;j<Vq.n_cols;j++){
			const arma::Mat<fltp>::fixed<3,3> Ji = arma::reshape(J.col(j),3,3).t();
			Vc.col(j) = arma::vectorise(Ji*arma::reshape(Vq.col(j),3,Vq.n_rows/3));
		}

		// divide by determinant
		Vc.each_row()/=arma::abs(Jdet);

		// return carthesian vectors
		return Vc;
	}

	// convert quadrilateral vectors to cartesian vectors using
	// using covariant piola mapping. This preserves 
	// the normal component of basis functions on facets.
	// for more info see: https://defelement.com/ciarlet.html
	arma::Mat<fltp> Hexahedron::quad2cart_covariant_piola(
		const arma::Mat<fltp> &Vq, 
		const arma::Mat<fltp> &Jinv){

		// allocate output
		arma::Mat<fltp> Vc(Vq.n_rows,Vq.n_cols);

		// walk over points
		for(arma::uword j=0;j<Vq.n_cols;j++){
			const arma::Mat<fltp>::fixed<3,3> Ji = arma::reshape(Jinv.col(j),3,3);
			Vc.col(j) = arma::vectorise(Ji*arma::reshape(Vq.col(j),3,Vq.n_rows/3));
		}

		// return carthesian vectors
		return Vc;
	}



	// shape function for Hdiv elements (Raviart Thomas, Nedelec)
	// divergence conforming facet element
	// https://defelement.com/elements/examples/hexahedron-Qdiv-1.html
	// input coordinates are given as [xi;nu;mu]
	arma::Mat<fltp> Hexahedron::raviart_thomas_hdiv_shape_function(const arma::Mat<fltp> &Rq){
		// check input
		assert(Rq.n_rows==3);

		// get coordinates
		const arma::Row<fltp> xi = Rq.row(0);
		const arma::Row<fltp> nu = Rq.row(1);
		const arma::Row<fltp> mu = Rq.row(2);

		// calculate shape function
		arma::Mat<fltp> N(18,Rq.n_cols,arma::fill::zeros);
		N.row(0*3 + 1) = RAT_CONST(0.5)*nu - RAT_CONST(0.5); // shape function for face 0154, runs from -1 to 0, exits
		N.row(1*3 + 0) = RAT_CONST(0.5) + RAT_CONST(0.5)*xi; // shape function for face 1265, runs from 0 to 1, exits
		N.row(2*3 + 1) = RAT_CONST(0.5) + RAT_CONST(0.5)*nu; // shape function for face 2376, runs from 0 to 1, exits
		N.row(3*3 + 0) = RAT_CONST(0.5)*xi - RAT_CONST(0.5); // shape function for face 3047, runs from -1 to 0, exits
		N.row(4*3 + 2) = RAT_CONST(0.5)*mu - RAT_CONST(0.5); // shape function for face 0321, runs from -1 to 0, exits
		N.row(5*3 + 2) = RAT_CONST(0.5) + RAT_CONST(0.5)*mu; // shape function for face 4567, runs from 0 to 1, exits

		// return shape function
		// all vectors point inwards
		return -N/4;
	}

	// // divergence shape function for Hdiv elements (Raviart Thomas, Nedelec)
	// arma::Mat<fltp> Hexahedron::raviart_thomas_hdiv_shape_function_div(const arma::Mat<fltp> &Rq){
	// 	// check input
	// 	assert(Rq.n_rows==3);

	// 	// calculate shape function
	// 	arma::Mat<fltp> N(6,Rq.n_cols,arma::fill::value(0.5));

	// 	// all vectors point inwards
	// 	return -N/4;
	// }

	// // shape function for Hcurl elements
	// // curl conforming edge element
	// // https://defelement.com/elements/examples/hexahedron-Qcurl-1.html
	// // input coordinates are given as [xi;nu;mu]
	// arma::Mat<fltp> Hexahedron::nedelec_hcurl_shape_function(const arma::Mat<fltp> &Rq){
	// 	// check input
	// 	assert(Rq.n_rows==3);

	// 	// get coordinates
	// 	const arma::Row<fltp> xi = RAT_CONST(0.5)*Rq.row(0) + RAT_CONST(0.5);
	// 	const arma::Row<fltp> nu = RAT_CONST(0.5)*Rq.row(1) + RAT_CONST(0.5);
	// 	const arma::Row<fltp> mu = RAT_CONST(0.5)*Rq.row(2) + RAT_CONST(0.5);

	// 	// calculate shape function
	// 	arma::Mat<fltp> N(36,Rq.n_cols,arma::fill::zeros);

	// 	N.row( 0*3 + 0) = nu%mu - nu - mu + RAT_CONST(1.0);
	// 	N.row( 1*3 + 1) = xi%(RAT_CONST(1.0) - mu);
	// 	N.row( 2*3 + 0) = -(nu%(RAT_CONST(1.0) - mu));
	// 	N.row( 3*3 + 1) = -(xi%mu - xi - mu + RAT_CONST(1.0));

	// 	N.row( 4*3 + 2) = xi%nu - xi - nu + RAT_CONST(1.0);
	// 	N.row( 5*3 + 2) = xi%(RAT_CONST(1.0) - nu);
	// 	N.row( 6*3 + 2) = xi%nu;
	// 	N.row( 7*3 + 2) = nu%(RAT_CONST(1.0) - xi);

	// 	N.row( 8*3 + 0) = mu%(RAT_CONST(1.0) - nu);
	// 	N.row( 9*3 + 1) = xi%mu;
	// 	N.row(10*3 + 0) = -(nu%mu);
	// 	N.row(11*3 + 1) = -(mu%(RAT_CONST(1.0) - xi));

	// 	// return shape function
	// 	return -N;
	// }

	// // compute the curl of the H(curl) shape functions on the reference element
	// arma::Mat<fltp> Hexahedron::nedelec_hcurl_shape_function_curl(const arma::Mat<fltp> &Rq) {
	// 	// get coordinates
	// 	const arma::Row<fltp> xi = RAT_CONST(0.5)*Rq.row(0) + RAT_CONST(0.5);
	// 	const arma::Row<fltp> nu = RAT_CONST(0.5)*Rq.row(1) + RAT_CONST(0.5);
	// 	const arma::Row<fltp> mu = RAT_CONST(0.5)*Rq.row(2) + RAT_CONST(0.5);

	// 	// calculate shape function
	// 	arma::Mat<fltp> N(36,Rq.n_cols,arma::fill::zeros);

	// 	// Nx = nu%mu - nu - mu + RAT_CONST(1.0);
	// 	N.row( 0*3 + 1) = nu - 1.0; // dNxdmu
	// 	N.row( 0*3 + 2) = -(mu - 1.0); // -dNxdnu

	// 	// Ny = xi%(RAT_CONST(1.0) - mu);
	// 	N.row( 1*3 + 0) = xi; // -dNydmu
	// 	N.row( 1*3 + 2) = 1.0 - mu; // dNydxi
			
	// 	// Nx = -(nu%(RAT_CONST(1.0) - mu));
	// 	N.row( 2*3 + 1) = nu; // dNxdmu
	// 	N.row( 2*3 + 2) = -(mu-1.0); // -dNxdnu
			
	// 	// Ny = -(xi%mu - xi - mu + RAT_CONST(1.0));
	// 	N.row( 3*3 + 0) = -(xi-1.0);// -dNydmu 
	// 	N.row( 3*3 + 2) = -mu+1.0;// dNydxi

	// 	// Nz = xi%nu - xi - nu + RAT_CONST(1.0);
	// 	N.row( 4*3 + 0) = xi - 1.0; // dNzdnu
	// 	N.row( 4*3 + 1) = -(nu - 1.0); // -dNzdxi

	// 	// Nz = xi%(RAT_CONST(1.0) - nu);
	// 	N.row( 5*3 + 0) = xi; // dNzdnu
	// 	N.row( 5*3 + 1) = -(1.0 - nu); // -dNzdxi

	// 	// Nz = xi%nu;
	// 	N.row( 6*3 + 0) = xi; // dNzdnu
	// 	N.row( 6*3 + 1) = -nu; // -dNzdxi

	// 	// Nz = nu%(RAT_CONST(1.0) - xi);
	// 	N.row( 7*3 + 0) = 1.0-xi; // dNzdnu
	// 	N.row( 7*3 + 1) = -(-nu); // -dNzdxi

	// 	// Nx = mu%(RAT_CONST(1.0) - nu);
	// 	N.row( 8*3 + 1) = 1.0-nu; // dNxdmu
	// 	N.row( 8*3 + 2) = -(-mu); // -dNxdnu

	// 	// Ny = xi%mu;
	// 	N.row( 9*3 + 0) = -xi; // -dNydmu 
	// 	N.row( 9*3 + 2) = mu; // dNydxi

	// 	// Nx = -(nu%mu);
	// 	N.row( 10*3 + 1) = -nu; // dNxdmu
	// 	N.row( 10*3 + 2) = -(-mu); // -dNxdnu

	// 	// Ny = -(mu%(RAT_CONST(1.0) - xi));
	// 	N.row( 11*3 + 0) = -(1.0-xi); // -dNydmu 
	// 	N.row( 11*3 + 2) = mu; // dNydxi

	// 	// return shape function
	// 	return -N;
	// }

	// transpose matrix
	arma::Mat<fltp> Hexahedron::jacobian_t(const arma::Mat<fltp> &J){
		// get rows
		arma::Mat<fltp> Jt(9,J.n_cols);
		for(arma::uword i=0;i<3;i++)
			for(arma::uword j=0;j<3;j++)
				Jt.row(i*3 + j) = J.row(j*3 + i);
		return Jt;
	}

	// calculate determinant of the jacobian
	arma::Row<fltp> Hexahedron::jacobian2determinant(const arma::Mat<fltp> &J){
		// check input
		assert(J.n_rows==9);

		// get rows
		const arma::Row<fltp> J11 = J.row(0);
		const arma::Row<fltp> J21 = J.row(1);
		const arma::Row<fltp> J31 = J.row(2);
		
		const arma::Row<fltp> J12 = J.row(3);
		const arma::Row<fltp> J22 = J.row(4);
		const arma::Row<fltp> J32 = J.row(5);

		const arma::Row<fltp> J13 = J.row(6);
		const arma::Row<fltp> J23 = J.row(7);
		const arma::Row<fltp> J33 = J.row(8);

		// calculate determinants
		const arma::Row<fltp> Jdet = 
			J11%J22%J33 + J21%J32%J13 + J31%J12%J23 - 
			J31%J22%J13 - J11%J32%J23 - J21%J12%J33;

		// return determinants
		return Jdet;
	}

	// method for inverting jacobian matrix
	arma::Mat<fltp> Hexahedron::invert_jacobian(const arma::Mat<fltp> &J,const arma::Row<fltp> &Jdet){ 
		// check input
		assert(J.n_rows==9); assert(J.n_cols==Jdet.n_elem);

		// get rows
		const arma::Row<fltp> J11 = J.row(0);
		const arma::Row<fltp> J21 = J.row(1);
		const arma::Row<fltp> J31 = J.row(2);
		const arma::Row<fltp> J12 = J.row(3);
		const arma::Row<fltp> J22 = J.row(4);
		const arma::Row<fltp> J32 = J.row(5);
		const arma::Row<fltp> J13 = J.row(6);
		const arma::Row<fltp> J23 = J.row(7);
		const arma::Row<fltp> J33 = J.row(8);

		// invert matrices (stored columnwise)
		arma::Mat<fltp> Jinv(9,J.n_cols);
		Jinv.row(0) = J22%J33-J32%J23;
		Jinv.row(1) = J31%J23-J21%J33;
		Jinv.row(2) = J21%J32-J31%J22;
		Jinv.row(3) = J13%J32-J12%J33;
		Jinv.row(4) = J11%J33-J13%J31;
		Jinv.row(5) = J12%J31-J32%J11;
		Jinv.row(6) = J12%J23-J22%J13;
		Jinv.row(7) = J21%J13-J11%J23;
		Jinv.row(8) = J11%J22-J21%J12;

		// divide by determinant
		Jinv.each_row()/=Jdet;

		// return inverted matrices
		return Jinv;
	}

	// convert derivative of shape function into carthesian coordinates
	// using the inverted jacobian
	arma::Mat<fltp> Hexahedron::derivative2cart(
		const arma::Mat<fltp>&Jinv,
		const arma::Row<fltp>&/*Jdet*/,
		const arma::Mat<fltp>&dN){

		// calculate cartesian derivatives
		arma::Mat<fltp> dNcart(24,Jinv.n_cols);
		for(arma::uword i=0;i<Jinv.n_cols;i++){
			// get jacobian matrix and make square
			const arma::Mat<fltp>::fixed<3,3> A = arma::reshape(Jinv.col(i),3,3);

			// multiply inverted jacobian with the shape function derivative vectors
			dNcart.col(i) = arma::reshape((A*arma::reshape(dN.col(i),8,3).t()).t(),24,1);
		}

		// return carthesian derivatives
		return dNcart;
	}

	// quadrilateral coordinates to carthesian coordinates
	// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
	arma::Mat<fltp> Hexahedron::shape_function_derivative_cart(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<fltp> &Rq){

		// get shape derivative of shape function
		const arma::Mat<fltp> dN = shape_function_derivative(Rq);
		const arma::Mat<fltp> J = shape_function_jacobian(Rn,dN);
		const arma::Row<fltp> Jdet = jacobian2determinant(J);
		const arma::Mat<fltp> Jinv = invert_jacobian(J,Jdet);

		// use ufnction to convert derivatives to cartesian coordinates
		return derivative2cart(Jinv,Jdet,dN);
	}

	// quadrilateral coordinates to carthesian coordinates
	// Rn can also be a different quantity at the nodes for interpolation
	// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
	arma::Mat<fltp> Hexahedron::quad2cart(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rn.n_cols==8); assert(Rq.n_rows==3);
		
		// get shape functions
		const arma::Mat<fltp> Nie = Hexahedron::shape_function(Rq);
		//std::cout<<Nie<<std::endl;

		// get coordinates from matrix vector product
		const arma::Mat<fltp> Rc = Rn*Nie;

		// return cartesian coords
		return Rc;
	}

	// carthesian derivative calculated at quadrilateral coordinates
	// phi is the quantity at the nodes
	// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]; 
	// dphi = [dphidx;dphidy;dphidz]
	arma::Mat<fltp> Hexahedron::quad2cart_derivative(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq,
		const arma::Mat<fltp> &phi){

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==8);
		assert(Rq.n_rows==3); assert(phi.n_cols==8);

		// get carthesian derivative of shape functions
		const arma::Mat<fltp> dNie = Hexahedron::shape_function_derivative_cart(Rn,Rq);

		// get coordinates from matrix vector product
		const arma::Mat<fltp> dphi = arma::reshape(phi*arma::reshape(dNie,8,3*Rq.n_cols),3*phi.n_rows,Rq.n_cols);

		// return cartesian coords
		return dphi;
	}

	// calculate curl
	arma::Mat<fltp> Hexahedron::quad2cart_curl(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq,
		const arma::Mat<fltp> &V){

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==8);
		assert(Rq.n_rows==3); assert(V.n_cols==8);
		assert(V.n_rows==3);

		// get derivative of V at Rq
		arma::Mat<fltp> dV = Hexahedron::quad2cart_derivative(Rn,Rq,V);

		// calculate cross product
		// dV = [dVx/dx (0), dVy/dx (1), dVz/dx (2),
		// 		 dVx/dy (3), dVy/dy (4), dVz/dy (5),
		// 		 dVx/dz (6), dVy/dz (7), dVz/dz (8)]
		arma::Mat<fltp> curl(3,Rq.n_cols);
		curl.row(0) = dV.row(5) - dV.row(7);
		curl.row(1) = dV.row(6) - dV.row(2);
		curl.row(2) = dV.row(1) - dV.row(3);

		// output calculated curl
		return curl;
	}


	// // curl conforming edge element H(curl)
	// // https://defelement.com/elements/examples/hexahedron-Qcurl-1.html
	// arma::Mat<fltp> Hexahedron::nedelec_shape_function(
	// 	const arma::Mat<fltp> &Rq){

	// 	// allocate edge contribution matrix
	// 	arma::Mat<fltp> Nie(12,Rq.n_cols);

	// 	// calculate shape functions
	// 	// bot plate
	// 	Nie.row(0) = Rq.row(1)%Rq.row(2) - Rq.row(1) - Rq.row(2) + 1.0; // in x -> curl in zy etc
	// 	Nie.row(1) = Rq.row(0)%(RAT_CONST(1.0) - Rq.row(2)); // in y
	// 	Nie.row(2) = -Rq.row(1)%(RAT_CONST(1.0) - Rq.row(2)); // in x
	// 	Nie.row(3) = -Rq.row(0)%Rq.row(2) - Rq.row(0) - Rq.row(2) + 1.0; // in y
		
	// 	// pillars
	// 	Nie.row(4) = Rq.row(0)%Rq.row(1) - Rq.row(0) - Rq.row(1) + 1.0; // in z
	// 	Nie.row(5) = Rq.row(0)%(RAT_CONST(1.0) - Rq.row(1)); // in z
	// 	Nie.row(6) = Rq.row(0)%Rq.row(1); // in z
	// 	Nie.row(7) = Rq.row(1)%(RAT_CONST(1.0) - Rq.row(0)); // in z

	// 	// top plate
	// 	Nie.row(8) = Rq.row(2)%(RAT_CONST(1.0) - Rq.row(1)); // in x
	// 	Nie.row(9) = Rq.row(0)%Rq.row(1); // in y
	// 	Nie.row(10) = -Rq.row(1)%Rq.row(2); // in x
	// 	Nie.row(11) = -Rq.row(2)%(RAT_CONST(1.0) - Rq.row(0)); // in y

	// 	// return values
	// 	return Nie;
	// }

	// carthesian coordinates to quadrilateral coordinates
	// this function can likely be improved by not using finite difference
	// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
	arma::Mat<fltp> Hexahedron::cart2quad(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rc, 
		const fltp tol){
		
		// get typical size
		const fltp diag = arma::as_scalar(Extra::vec_norm(Rn.col(0) - Rn.col(6)));

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==8);
		assert(Rc.n_rows==3);
		
		// initial guess
		arma::Mat<fltp> Rq(3,Rc.n_cols,arma::fill::zeros);

		// iterations
		// const fltp delta = RAT_CONST(1e-2);
		for(arma::uword i=0;i<100;i++){
			// calculate difference
			const arma::Mat<fltp> Rc00 = Hexahedron::quad2cart(Rn,Rq);
			const arma::Mat<fltp> dRc = Rc00 - Rc;

			// check for convergence
			// const fltp conv = arma::max(arma::max(arma::abs(dRc)));
			const fltp conv = arma::max(Extra::vec_norm(dRc));
			if(conv<tol*diag)break;

			// calculate jacobian
			const arma::Mat<fltp> J = shape_function_jacobian(Rn,shape_function_derivative(Rq));

			// invert jacobian
			const arma::Row<fltp> Jdet = jacobian2determinant(J);
			const arma::Mat<fltp> Jinv = invert_jacobian(J,Jdet);

			// check
			assert(Jinv.is_finite());

			// shift coordinates
			Rq -= arma::join_vert(
				Jinv.row(0)%dRc.row(0) + Jinv.row(1)%dRc.row(1) + Jinv.row(2)%dRc.row(2), 
				Jinv.row(3)%dRc.row(0) + Jinv.row(4)%dRc.row(1) + Jinv.row(5)%dRc.row(2),
				Jinv.row(6)%dRc.row(0) + Jinv.row(7)%dRc.row(1) + Jinv.row(8)%dRc.row(2));

			// check if fail elements are likely non-planar
			// if(i>100)rat_throw_line("cart2quad does not converge");
		}

		// check (this check sometimes fails need to check why)
		// assert(arma::all(arma::all(arma::abs(Hexahedron::quad2cart(Rn,Rq)-Rc)<tol*diag)));

		// return carthesian coordinates
		return Rq;
	}

	// extrapolate values from gauss points to nodes
	// using the discrete local smoothing technique
	arma::Mat<fltp> Hexahedron::gauss2nodes(
		const arma::Mat<fltp> &Rn, // Nodal coordinates of the hexahedral element.
		const arma::Mat<fltp> &Rqg, // Gauss point coordinates in the reference element.
		const arma::Row<fltp> &wg, // Gauss quadrature weights.
		const arma::Mat<fltp> &Vg){ //The values at Gauss points that need to be extrapolated to nodes.

		// check input
		assert(Rqg.n_rows==3);
		assert(Rqg.n_cols==Vg.n_cols);

		// jacobian
		const arma::Mat<fltp> dN = shape_function_derivative(Rqg);
		const arma::Mat<fltp> J = shape_function_jacobian(Rn,dN);
		const arma::Row<fltp> Jdet = jacobian2determinant(J);

		// calculate shape function at gauss points
		const arma::Mat<fltp> Ng = shape_function(Rqg);

		// create conversion matrices
		arma::Mat<fltp>::fixed<8,8> M(arma::fill::zeros);
		arma::Mat<fltp> P(8,Rqg.n_cols);
		for(arma::uword i=0;i<8;i++){
			for(arma::uword k=0;k<Rqg.n_cols;k++){
				for(arma::uword j=0;j<8;j++){
					M(i,j) += Ng(i,k)*Ng(j,k)*Jdet(k)*wg(k);
				}
				P(i,k) = Ng(i,k)*Jdet(k)*wg(k); // the 2 is needed here because in quadrilateral coordinates we have edges with length 2
			}
		}

		// regularize
		const fltp condM = arma::cond(M);
		const fltp eps_factor = (condM > RAT_CONST(1e8)) ? RAT_CONST(1e-3) : RAT_CONST(1e-6); 
		const fltp epsilon = eps_factor*M.diag().max();
		M.diag()+=epsilon;

		// walk over values
		arma::Mat<fltp> Vn(Vg.n_rows,8,arma::fill::zeros);
		for(arma::uword m=0;m<Vg.n_rows;m++){
			// solve without throwing errors
			arma::Col<fltp> x; if(arma::solve(x,M,P*Vg.row(m).t(),arma::solve_opts::force_approx))Vn.row(m) = x.t();
		}

		// return values at nodes
		return Vn;
	}

	// extrapolate values from gauss points for entire mesh
	arma::Mat<fltp> Hexahedron::gauss2nodes(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n, 
		const arma::Mat<fltp> &Vg, 
		const arma::Mat<fltp> &Rqg, 
		const arma::Row<fltp> &wg,
		const arma::uword num_nodes){

		// check input
		assert(Rqg.n_rows==3);
		assert(n.n_rows==8);
		assert(Vg.n_cols%Rqg.n_cols==0);
		assert(n.max()<num_nodes);

		// allocate output
		arma::Mat<fltp> Vnn(Vg.n_rows,8*n.n_cols);

		// walk over elements and perform local smoothing (in parallel)
		parfor(0,n.n_cols,true,[&](arma::uword i, int){
			Vnn.cols(i*8,(i+1)*8-1) = gauss2nodes(Rn.cols(n.col(i)),Rqg,wg,Vg.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1));
		});

		// calculate volume
		const arma::Row<fltp> Ve = calc_volume(Rn,n);

		// average at nodes
		arma::Mat<fltp> Vn(Vg.n_rows,num_nodes,arma::fill::zeros);
		arma::Row<fltp> cnt(1,num_nodes);
		for(arma::uword i=0;i<n.n_cols;i++){
			Vn.cols(n.col(i)) += Vnn.cols(i*8,(i+1)*8-1)*Ve(i); 
			cnt(n.col(i)) += Ve(i);
		}

		// return values at nodes
		return Vn.each_row()/cnt;
	}

	// function for determining if points are inside a hexahedron
	arma::Row<arma::uword> Hexahedron::is_inside(
		const arma::Mat<fltp> &R,
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n){

		// check input
		assert(R.n_rows==3); 
		assert(Rn.n_rows==3);
		assert(n.n_rows==8);
		assert(n.n_elem>0);
		assert(arma::max(arma::max(n))<Rn.n_cols);	

		// get number of elements
		const arma::uword num_elem = n.n_cols;

		// get hexahedron to tetrahedron matrix
		const arma::Mat<arma::uword>::fixed<5,4> M = 
			Hexahedron::tetrahedron_conversion_matrix();

		// make node connectivity matrix for tetrahedrons
		arma::Mat<arma::uword> nt(4*5,num_elem);
		for(int i=0;i<5;i++){
			nt.rows(i*4,(i+1)*4-1) = n.rows(M.row(i));
		}
		nt.reshape(4,5*num_elem);
		
		// check if inside tetrahedrons
		const arma::Mat<arma::uword> isinside = 
			Tetrahedron::is_inside(R,Rn,nt);

		// reshape
		return isinside;
	}

	// setup a grid around a singular point
	// grid containes nodes in quadrilateral coordinates (Rqgrd)
	// and weights (wgrd)
	void Hexahedron::setup_source_grid(
		arma::Mat<fltp> &Rqgrd, arma::Row<fltp> &wgrd,
		const arma::Col<fltp>::fixed<3> &Rqs,
		const arma::Row<fltp> &xg, const arma::Row<fltp> &wg){
		
		// check gauss weights
		assert(arma::all(arma::abs(Rqs)<=1.0));
		assert(arma::as_scalar(arma::sum(wg))-2.0<1e-5);

		// scale gauss points
		arma::Row<fltp> sxg = (xg+1)/2;

		// get side lengths
		arma::Col<fltp>::fixed<3> a = Rqs+1.0;
		arma::Col<fltp>::fixed<3> b = 1.0-Rqs;

		// split face depending on target position
		arma::field<arma::Row<fltp> > x(3), w(3);
		for(arma::uword j=0;j<3;j++){
			if(a(j)>1e-6f && b(j)>1e-6f){
				x(j) = arma::join_horiz(a(j)*sxg - 1.0f, b(j)*sxg + Rqs(j));
				w(j) = arma::join_horiz(wg*a(j)/2, wg*b(j)/2);
			}
			else if(a(j)>1e-6f && b(j)<1e-6f){
				x(j) = a(j)*sxg - 1.0; w(j) = wg*a(j)/2;
			}
			else{
				x(j) = b(j)*sxg - 1.0; w(j) = wg*b(j)/2;
			}
			assert(arma::all(w(j)>0));
		}

		// number of elements in grid
		arma::uword Ngrd = x(0).n_elem * x(1).n_elem * x(2).n_elem;

		// allocate coordinates
		Rqgrd.set_size(3,Ngrd); wgrd.zeros(Ngrd);

		// setup grid
		for(arma::uword k=0;k<x(0).n_elem;k++){
			for(arma::uword l=0;l<x(1).n_elem;l++){
				for(arma::uword m=0;m<x(2).n_elem;m++){
					// index
					arma::uword idx = k*x(1).n_elem*x(2).n_elem + l*x(2).n_elem + m;

					// set grid coordinate
					Rqgrd.at(0,idx) = x(0).at(k); 
					Rqgrd.at(1,idx) = x(1).at(l); 
					Rqgrd.at(2,idx) = x(2).at(m);
					wgrd.at(idx) = w(0).at(k)*w(1).at(l)*w(2).at(m)/8;
				}
			}
		}	

		// sanity check
		assert(arma::all(wgrd>0));
		assert(std::abs(arma::as_scalar(arma::sum(wgrd))-1.0)<1e-5);

		// check
		assert(arma::all(arma::all(Rqgrd<1.0 && Rqgrd>-1.0)));
	}

	// check if polygon is clockwise
	arma::Row<arma::uword> Hexahedron::is_clockwise(
		const arma::Mat<fltp> &R, const arma::Mat<arma::uword> &n){
		// // select edges
		// const arma::Mat<fltp> V1 = R.cols(n.row(1)) - R.cols(n.row(0));
		// const arma::Mat<fltp> V2 = R.cols(n.row(2)) - R.cols(n.row(1));

		// // select point from other plane 
		// const arma::Mat<fltp> V3 = R.cols(n.row(5)) - R.cols(n.row(1));

		// // cross product
		// const arma::Row<fltp> d = Extra::dot(V3,Extra::cross(V1,V2));

		// // return orientation
		// return d>RAT_CONST(0.0);
		
		// chek for inverted hexahedrons using the jacobian matrix
		arma::Row<arma::uword> is_inverted(n.n_cols);
		const arma::Col<fltp> Rq = {0,0,0};
		const arma::Col<fltp> dN = shape_function_derivative(Rq);
		for(arma::uword i=0;i<n.n_cols;i++){
			const arma::Mat<fltp> J = shape_function_jacobian(R.cols(n.col(i)),dN);
			const arma::Mat<fltp> Jdet = jacobian2determinant(J);
			is_inverted(i) = arma::as_scalar(Jdet)>0;
		}
		return is_inverted;
	}

	// invert elements without reversing them
	arma::Mat<arma::uword> Hexahedron::invert_elements(const arma::Mat<arma::uword>&n){
		assert(n.n_rows==8);
		return n.rows(arma::Col<arma::uword>::fixed<8>{3,2,1,0, 7,6,5,4});
	}

	// reverse elements without inverting them
	arma::Mat<arma::uword> Hexahedron::reverse_elements(const arma::Mat<arma::uword>&n){
		assert(n.n_rows==8);
		return n.rows(arma::Col<arma::uword>::fixed<8>{7,6,5,4, 3,2,1,0});
	}


	// volume of a single hexahedron
	fltp Hexahedron::calc_volume(
		const arma::Mat<fltp> &Rn){

		// check input
		assert(Rn.n_cols==8);
		assert(Rn.n_rows==3);

		// get tetrahedron conversion matrix
		const TetrahedronConversionMatrix M = tetrahedron_conversion_matrix();

		// allocate volumes and centers of mass
		fltp Ve = RAT_CONST(0.0);

		// walk over tetrahedrons and add their volumes
		for(arma::uword i=0;i<M.n_rows;i++)
			Ve += arma::as_scalar(Tetrahedron::calc_volume(Rn, M.row(i).t()));

		// return element volumes
		return Ve;
	}

	// volume hexahedrons in a mesh
	arma::Row<fltp> Hexahedron::calc_volume(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n){

		// check input
		assert(Rn.n_rows==3);
		assert(n.n_rows==8);

		// get tetrahedron conversion matrix
		const TetrahedronConversionMatrix M = tetrahedron_conversion_matrix();

		// allocate volumes and centers of mass
		arma::Row<fltp> Ve(n.n_cols,arma::fill::zeros);

		// walk over tetrahedrons and add their volumes
		for(arma::uword i=0;i<M.n_rows;i++)
			Ve += Tetrahedron::calc_volume(Rn, n.rows(M.row(i)));

		// return element volumes
		return Ve;
	}

	// extract the surface elements
	arma::Mat<arma::uword> Hexahedron::extract_surface(
		const arma::Mat<arma::uword> &n){

		assert(n.n_rows==8);

		// get counters
		const arma::uword num_elements = n.n_cols;

		// get list of faces for each element
		arma::Mat<arma::uword>::fixed<6,4> M = cmn::Hexahedron::get_faces();

		// create list of all faces present in mesh
		arma::Mat<arma::uword> S(4,num_elements*6);
		for(arma::uword i=0;i<6;i++){
			S.cols(i*num_elements,(i+1)*num_elements-1) = n.rows(M.row(i));
		}

		// sort each column in S
		arma::Mat<arma::uword> Ss(4,num_elements*6);
		for(arma::uword i=0;i<num_elements*6;i++){
			Ss.col(i) = arma::sort(S.col(i));
		}

		// sort S by rows
		for(arma::uword i=0;i<4;i++){
			arma::Col<arma::uword> idx = arma::stable_sort_index(Ss.row(3-i));
			Ss = Ss.cols(idx); S = S.cols(idx);
		}

		// find duplicates and mark
		arma::Row<arma::uword> duplicates = 
			arma::all(Ss.cols(0,num_elements*6-2)==Ss.cols(1,num_elements*6-1),0);

		// extend duplicate list to contain first and second
		arma::Row<arma::uword> extended(num_elements*6,arma::fill::zeros);
		extended.head(num_elements*6-1) += duplicates;
		extended.tail(num_elements*6-1) += duplicates;

		// return surface 
		return S.cols(arma::find(extended==0));
	}

	// // subdivide to remove singularity from equation
	// // should work for both two and three dimensions
	// // the singularity is assumed to be between a and b
	// // output is from 0 to a+b
	// arma::Mat<fltp> Hexahedron::subdivide(
	// 	const arma::Col<fltp> &a, 
	// 	const arma::Col<fltp> &b){

	// 	// check
	// 	assert(a.n_elem==b.n_elem);
	// 	assert(arma::all(a>=0));
	// 	assert(arma::all(b>=0));

	// 	// calculate total width
	// 	arma::Col<fltp> w = a+b;

	// 	// get distance
	// 	arma::Col<fltp> dist(a.n_elem);
	// 	for(arma::uword j=0;j<dist.n_elem;j++){
	// 		if(a(j)<b(j))dist(j)=a(j); else dist(j)=b(j);
	// 	}

	// 	// sort by distance
	// 	arma::Col<arma::uword> idx = arma::sort_index(dist,"ascend");

	// 	// first dimension 
	// 	arma::Mat<fltp> division(dist.n_elem,2,arma::fill::zeros);
	// 	if(a(idx(0))<b(idx(0))){
	// 		arma::Row<fltp> val = {a(idx(0)),a(idx(0)),(w(idx(0))-2*a(idx(0)))/2,(w(idx(0))-2*a(idx(0)))/2};
	// 		division.row(idx(0)) = val;
	// 	}else{
	// 		arma::Row<fltp> val = {(w(idx(0))-2*b(idx(0)))/2,(w(idx(0))-2*b(idx(0)))/2,b(idx(0)),b(idx(0))};
	// 		division.row(idx(0)) = val;
	// 	}
		
	// 	// other dimensions
	// 	for(arma::uword j=1;j<dist.n_elem;j++){
	// 		if(dist(idx(j))<w(idx(j))/100){
	// 			arma::Row<fltp> val = {w(idx(j))/2,w(idx(j))/2,0,0};
	// 			division.row(idx(j)) = val;
	// 		}else{
	// 			arma::Row<fltp> val = {a(idx(j))-dist(idx(0)),dist(idx(0)),dist(idx(0)),b(idx(j))-dist(idx(0))};
	// 			division.row(idx(j)) = val;
	// 		}
	// 	}

	// 	//std::cout<<division<<std::endl;

	// 	// shift all non-zero values to the left
	// 	arma::uword cntmax = 0;
	// 	for(arma::uword j=0;j<division.n_rows;j++){
	// 		arma::uword cnt = 0;
	// 		for(arma::uword i=0;i<division.n_cols;i++){
	// 			if(division(j,i)>1e-9){
	// 				division(j,cnt) = division(j,i);
	// 				cnt++;
	// 			}
	// 		}
	// 		cntmax = std::max(cnt,cntmax);
	// 	}

	// 	// remove trailing zeros
	// 	division.resize(division.n_rows,cntmax);

	// 	// check
	// 	assert(arma::as_scalar(arma::all((arma::sum(division,1) - w)<1e-7,0)));
	// 	//assert(arma::all(arma::all(division>1e-9)));

	// 	// calculate node positions
	// 	// arma::Mat<fltp> sxg = arma::cumsum(division,1)-division/2;

	// 	// divide again
	// 	arma::Mat<fltp> seconddivision = arma::join_vert(division/2,division/2);
	// 	seconddivision.reshape(division.n_rows,division.n_cols*2);

	// 	// return gauss points
	// 	return seconddivision;
	// }


	// create hexahedral mesh in a sphere centered on the origin
	void Hexahedron::create_sphere(
		arma::Mat<fltp>&Rn, 
		arma::Mat<arma::uword>&n, 
		arma::Mat<arma::uword>&s, 
		const fltp sphere_radius, 
		const fltp element_size,
		const fltp core_size){

		// core radius
		const fltp box_side = core_size*2*sphere_radius/std::sqrt(RAT_CONST(3.0));

		// number of elemnets
		const arma::uword num_elem_core = std::max(1,int(box_side/element_size));
		const arma::uword num_node_core = num_elem_core + 1;

		// create nodes
		arma::Mat<fltp> Rncore(3,num_node_core*num_node_core*num_node_core);

		// indexes
		// arma::Cube<arma::uword> n(num_node_core,num_node_core,num_node_core);

		// create a mesh of a cube
		arma::uword cnt = 0;
		for(arma::uword i=0;i<num_node_core;i++){
			for(arma::uword j=0;j<num_node_core;j++){
				for(arma::uword k=0;k<num_node_core;k++){
					Rncore(0,cnt) = i*box_side/num_elem_core - box_side/2;
					Rncore(1,cnt) = j*box_side/num_elem_core - box_side/2;
					Rncore(2,cnt) = k*box_side/num_elem_core - box_side/2;
					cnt++;
				}
			}
		}

		// create elements
		arma::Mat<arma::uword> ncore(8,num_elem_core*num_elem_core*num_elem_core);
		cnt = 0;
		for(arma::uword i=0;i<num_elem_core;i++){
			for(arma::uword j=0;j<num_elem_core;j++){
				for(arma::uword k=0;k<num_elem_core;k++){
					ncore(0,cnt) = i*num_node_core*num_node_core + j*num_node_core + k;
					ncore(1,cnt) = (i+1)*num_node_core*num_node_core + j*num_node_core + k;
					ncore(2,cnt) = (i+1)*num_node_core*num_node_core + (j+1)*num_node_core + k;
					ncore(3,cnt) = i*num_node_core*num_node_core + (j+1)*num_node_core + k;
					ncore(4,cnt) = i*num_node_core*num_node_core + j*num_node_core + k + 1;
					ncore(5,cnt) = (i+1)*num_node_core*num_node_core + j*num_node_core + k + 1;
					ncore(6,cnt) = (i+1)*num_node_core*num_node_core + (j+1)*num_node_core + k + 1;
					ncore(7,cnt) = i*num_node_core*num_node_core + (j+1)*num_node_core + k + 1;
					cnt++;
				}
			}
		}

		// number of elements in shell
		const arma::uword num_elem_shell = std::max(1,int((sphere_radius - box_side/2)/element_size));
		const arma::uword num_node_shell = num_elem_shell+1;
 
		// shell coordinates
		arma::Mat<fltp> Rnshell(3,num_node_core*num_node_core*num_node_shell);

		// create a mesh of a cube
		cnt = 0;
		for(arma::uword i=0;i<num_node_core;i++){
			for(arma::uword j=0;j<num_node_core;j++){
				// core surface
				arma::Col<fltp>::fixed<3> R0{
					i*box_side/num_elem_core - box_side/2,
					j*box_side/num_elem_core - box_side/2,
					box_side/2};

				// coordinate projected on sphere surface
				arma::Col<fltp>::fixed<3> R1 = 
					sphere_radius*(R0.each_row()/cmn::Extra::vec_norm(R0));

				// walk over layers
				for(arma::uword k=0;k<num_node_shell;k++){
					Rnshell.col(cnt) = R0 + (float(k)/(num_node_shell-1))*(R1-R0);
					cnt++;
				}
			}
		}

		// shell volume mesh
		arma::Mat<arma::uword> nshell(8,num_elem_core*num_elem_core*num_elem_shell);

		// create elements
		cnt = 0;
		for(arma::uword i=0;i<num_elem_core;i++){
			for(arma::uword j=0;j<num_elem_core;j++){
				for(arma::uword k=0;k<num_elem_shell;k++){
					nshell(0,cnt) = i*num_node_core*num_node_shell + j*num_node_shell + k;
					nshell(1,cnt) = (i+1)*num_node_core*num_node_shell + j*num_node_shell + k;
					nshell(2,cnt) = (i+1)*num_node_core*num_node_shell + (j+1)*num_node_shell + k;
					nshell(3,cnt) = i*num_node_core*num_node_shell + (j+1)*num_node_shell + k;
					nshell(4,cnt) = i*num_node_core*num_node_shell + j*num_node_shell + k + 1;
					nshell(5,cnt) = (i+1)*num_node_core*num_node_shell + j*num_node_shell + k + 1;
					nshell(6,cnt) = (i+1)*num_node_core*num_node_shell + (j+1)*num_node_shell + k + 1;
					nshell(7,cnt) = i*num_node_core*num_node_shell + (j+1)*num_node_shell + k + 1;
					cnt++;
				}
			}
		}

		// surface mesh
		arma::Mat<arma::uword> sshell(4,num_elem_core*num_elem_core);

		// create surface mesh
		cnt = 0;
		for(arma::uword i=0;i<num_elem_core;i++){
			for(arma::uword j=0;j<num_elem_core;j++){
				sshell(3,cnt) = i*num_node_core*num_node_shell + j*num_node_shell + num_node_shell-1;
				sshell(2,cnt) = (i+1)*num_node_core*num_node_shell + j*num_node_shell + num_node_shell-1;
				sshell(1,cnt) = (i+1)*num_node_core*num_node_shell + (j+1)*num_node_shell + num_node_shell-1;
				sshell(0,cnt) = i*num_node_core*num_node_shell + (j+1)*num_node_shell + num_node_shell-1;
				cnt++;
			}
		}

		// // check mesh
		// for(arma::uword i=0;i<sshell.n_cols;i++){
			
		// 	// rat_throw_line("mesh is not planar");
		// 	// get typical size
		// 	fltp diag = std::sqrt(arma::as_scalar(arma::sum(
		// 		(Rnshell.col(nshell(0,i)) - Rnshell.col(nshell(2,i)))%
		// 		(Rnshell.col(nshell(0,i)) - Rnshell.col(nshell(2,i))),0)));

		// 	// calculate the plane in which the face is located
		// 	arma::Col<fltp>::fixed<3> V1 = Rnshell.col(nshell(1,i))-Rnshell.col(nshell(0,i));
		// 	arma::Col<fltp>::fixed<3> V2 = Rnshell.col(nshell(3,i))-Rnshell.col(nshell(1,i));

		// 	// get face normal
		// 	arma::Col<fltp>::fixed<3> N = cmn::Extra::cross(V1,V2);
		// 	N = N/arma::as_scalar(cmn::Extra::vec_norm(N));

		// 	// check if diagonals are in plane
		// 	fltp eps1 = arma::as_scalar(cmn::Extra::dot(N,Rnshell.col(nshell(2,i))-Rnshell.col(nshell(0,i))))/diag;
		// 	fltp eps2 = arma::as_scalar(cmn::Extra::dot(N,Rnshell.col(nshell(1,i))-Rnshell.col(nshell(3,i))))/diag;


		// 	std::cout<<"sh "<<i<<" "<<eps1<<" "<<eps2<<std::endl;
		// 	std::cout<<cmn::Quadrilateral::check_nodes(Rnshell.cols(nshell.col(i)))<<std::endl;
		// }

		// create all parts
		arma::field<arma::Mat<fltp> > Rnfld(1,7);
		arma::field<arma::Mat<arma::uword> > nfld(1,7);
		arma::field<arma::Mat<arma::uword> > sfld(1,6);
		cnt = 0;
		for(arma::uword i=0;i<4;i++){
			Rnfld(i) = Rnshell; nfld(i) = nshell + cnt; sfld(i) = sshell + cnt;
			Rnfld(i) = Extra::create_rotation_matrix({1,0,0},i*arma::Datum<fltp>::pi/2)*Rnfld(i);
			// TransRotate::create({1,0,0},i*arma::Datum<fltp>::pi/2)->apply_coords(Rnfld(i),stngs.time);


			cnt += Rnfld(i).n_cols;
		}
		for(arma::uword i=0;i<2;i++){
			Rnfld(4+i) = Rnshell; nfld(4+i) = nshell + cnt; sfld(4+i) = sshell + cnt;
			// TransRotate::create({0,1,0},(float(i)-RAT_CONST(0.5))*arma::Datum<fltp>::pi)->apply_coords(Rnfld(4+i),stngs.time);
			Rnfld(4+i) = Extra::create_rotation_matrix({0,1,0},(float(i)-RAT_CONST(0.5))*arma::Datum<fltp>::pi)*Rnfld(4+i);
			cnt += Rnfld(4+i).n_cols;
		}
		Rnfld(6) = Rncore; nfld(6) = ncore + cnt; 
		cnt += Rnfld(6).n_cols;

		// combine
		Rn = cmn::Extra::field2mat(Rnfld);
		n = cmn::Extra::field2mat(nfld);
		s = cmn::Extra::field2mat(sfld);

		// combine nodes in a mesh
		const arma::Row<arma::uword> idx = cmn::Extra::combine_nodes(Rn);

		// remove duplicate points
		n = arma::reshape(idx(arma::vectorise(n)),n.n_rows,n.n_cols);
		s = arma::reshape(idx(arma::vectorise(s)),s.n_rows,s.n_cols);

		// check output
		assert(arma::max(arma::max(n))<Rn.n_cols); 
		assert(arma::max(arma::max(s))<Rn.n_cols);

		// invert hexahedrons
		{
			arma::Mat<arma::uword> n0 = n.rows(0,3);
			arma::Mat<arma::uword> n1 = n.rows(4,7);
			n.rows(0,3) = n1; n.rows(4,7) = n0;
		}

		// ensure clockwise mesh
		const arma::Col<arma::uword> index = arma::find(cmn::Hexahedron::is_clockwise(Rn,n));
		n.cols(index) = cmn::Hexahedron::invert_elements(n.cols(index)); //arma::join_vert(n.cols(index).eval().rows(4,7), n.cols(index).eval().rows(0,3));

		// done
		return;
	}


	// TETRAHEDRON
	// hexahedron face matrix
	Tetrahedron::FaceMatrix Tetrahedron::get_faces(){
		// setup matrix
		// faces is counter clockwise order (as seen from outside)
		arma::Mat<arma::uword>::fixed<4,3> M{
			{0,2,1},
			{0,1,3},
			{0,3,2},
			{1,2,3}};

		// return matrix
		return M;
	}

	Tetrahedron::EdgeMatrix Tetrahedron::get_edges(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<6,2> M{
			2,1,1,0,0,0,
			3,3,2,3,2,1};
		return M;
	}

	// tetrahedron corner node positions 
	// in quadrilateral coordinates
	// matrix columns are given as [dz1,dz2,dz3,dz4]
	Tetrahedron::CornerNodeMatrix Tetrahedron::get_corner_nodes(){
		// setup matrix
		arma::Mat<arma::sword>::fixed<4,3> M{
			0,1,0,0,
			0,0,1,0,
			0,0,0,1};

		// return matrix
		return M;
	}

	// vector wise determinants from matrix
	// |x1,y1,z1,1;x2,y2,z2,1;x3,y3,z3,1;x4,y4,z4,1| = 
	// -x3 y2 z1 + x4 y2 z1 + x2 y3 z1 - x4 y3 z1 - x2 y4 z1 + x3 y4 z1 + 
	//  x3 y1 z2 - x4 y1 z2 - x1 y3 z2 + x4 y3 z2 + x1 y4 z2 - x3 y4 z2 - 
	//  x2 y1 z3 + x4 y1 z3 + x1 y2 z3 - x4 y2 z3 - x1 y4 z3 + x2 y4 z3 + 
	//  x2 y1 z4 - x3 y1 z4 - x1 y2 z4 + x3 y2 z4 + x1 y3 z4 - x2 y3 z4
	arma::Row<fltp> Tetrahedron::special_determinant(
		const arma::Mat<fltp> &R0, 
		const arma::Mat<fltp> &R1, 
		const arma::Mat<fltp> &R2,
		const arma::Mat<fltp> &R3){

		// calculate determinants
		arma::Row<fltp> det = 
		   -R2.row(0)%R1.row(1)%R0.row(2) + R3.row(0)%R1.row(1)%R0.row(2) + 
			R1.row(0)%R2.row(1)%R0.row(2) - R3.row(0)%R2.row(1)%R0.row(2) - 
			R1.row(0)%R3.row(1)%R0.row(2) + R2.row(0)%R3.row(1)%R0.row(2) + 
			R2.row(0)%R0.row(1)%R1.row(2) - R3.row(0)%R0.row(1)%R1.row(2) - 
			R0.row(0)%R2.row(1)%R1.row(2) + R3.row(0)%R2.row(1)%R1.row(2) + 
			R0.row(0)%R3.row(1)%R1.row(2) - R2.row(0)%R3.row(1)%R1.row(2) - 
			R1.row(0)%R0.row(1)%R2.row(2) + R3.row(0)%R0.row(1)%R2.row(2) + 
			R0.row(0)%R1.row(1)%R2.row(2) - R3.row(0)%R1.row(1)%R2.row(2) - 
			R0.row(0)%R3.row(1)%R2.row(2) + R1.row(0)%R3.row(1)%R2.row(2) + 
			R1.row(0)%R0.row(1)%R3.row(2) - R2.row(0)%R0.row(1)%R3.row(2) - 
			R0.row(0)%R1.row(1)%R3.row(2) + R2.row(0)%R1.row(1)%R3.row(2) + 
			R0.row(0)%R2.row(1)%R3.row(2) - R1.row(0)%R2.row(1)%R3.row(2);

		// return determinants
		return det;
	}

	// calculate element skewedness as a measure of element quality
	arma::Row<fltp> Tetrahedron::calc_skewedness(
		const arma::Mat<fltp>&Rn, 
		const arma::Mat<arma::uword>&n){

		// check input
		assert(n.n_rows==4);

		// allocate output
		arma::Row<fltp> skewedness(n.n_cols,arma::fill::ones);

		// get list of faces for each element
		arma::Mat<arma::uword>::fixed<4,3> M = get_faces();

		// walk over edges
		for(arma::uword i=0;i<M.n_rows;i++){
			// get face
			const arma::Mat<arma::uword> f = n.rows(M.row(i));

			// walk over corners
			for(arma::uword j=0;j<f.n_rows;j++){
				// shifted indices
				const arma::uword idx1 = j;
				const arma::uword idx2 = (j+1)%f.n_rows;
				const arma::uword idx3 = (j+2)%f.n_rows;

				// get edge vectors
				const arma::Mat<fltp> V1 = Rn.cols(f.row(idx2)) - Rn.cols(f.row(idx1));
				const arma::Mat<fltp> V2 = Rn.cols(f.row(idx3)) - Rn.cols(f.row(idx2));

				// calculate skew and take minimum value for this element
				skewedness = arma::max(arma::abs(arma::Datum<fltp>::pi/3 - arma::acos(cmn::Extra::dot(V1,V2)/(cmn::Extra::vec_norm(V1)%cmn::Extra::vec_norm(V2)))));
			}
		}
		
		// return the skewedness
		return skewedness;
	}

	// calculate element skewedness as a measure of element quality
	arma::Row<fltp> Tetrahedron::calc_aspect_ratio(
		const arma::Mat<fltp>&Rn, 
		const arma::Mat<arma::uword>&n){

		// check input
		assert(n.n_rows==4);

		// allocate output
		arma::Row<fltp> shortest_edge(n.n_cols,arma::fill::value(1e99));
		arma::Row<fltp> longest_edge(n.n_cols,arma::fill::zeros);

		// get list of faces for each element
		const EdgeMatrix E = get_edges();

		// walk over edges
		for(arma::uword i=0;i<E.n_rows;i++){
			// calculate edge lengths
			const arma::Row<fltp> ell = cmn::Extra::vec_norm(Rn.cols(n.row(E(i,1))) - Rn.cols(n.row(E(i,0))));

			// find shortest and longest edges
			shortest_edge = arma::min(shortest_edge, ell);
			longest_edge = arma::max(longest_edge, ell);
		}
		
		// return the aspect ratio
		return longest_edge/shortest_edge;
	}

	// extrapolate values from gauss points to nodes
	// using the discrete local smoothing technique
	arma::Mat<fltp> Tetrahedron::gauss2nodes(
		const arma::Mat<fltp> &Rn, // Nodal coordinates of the hexahedral element.
		const arma::Mat<fltp> &Rqg, // Gauss point coordinates in the reference element.
		const arma::Row<fltp> &wg, // Gauss quadrature weights.
		const arma::Mat<fltp> &Vg){ //The values at Gauss points that need to be extrapolated to nodes.

		// check input
		assert(Rqg.n_rows==3);
		assert(Rqg.n_cols==Vg.n_cols);

		// jacobian
		const arma::Mat<fltp> dN = shape_function_derivative(Rqg);
		const arma::Mat<fltp> J = shape_function_jacobian(Rn,dN);
		const arma::Row<fltp> Jdet = jacobian2determinant(J);

		// calculate shape function at gauss points
		const arma::Mat<fltp> Ng = shape_function(Rqg);

		// create conversion matrices
		arma::Mat<fltp>::fixed<4,4> M(arma::fill::zeros);
		arma::Mat<fltp> P(4,Rqg.n_cols);
		for(arma::uword i=0;i<4;i++){
			for(arma::uword k=0;k<Rqg.n_cols;k++){
				for(arma::uword j=0;j<4;j++){
					M(i,j) += Ng(i,k)*Ng(j,k)*Jdet(k)*wg(k);
				}
				P(i,k) = Ng(i,k)*Jdet(k)*wg(k); // the 2 is needed here because in quadrilateral coordinates we have edges with length 2
			}
		}

		// regularize
		const fltp condM = arma::cond(M);
		const fltp eps_factor = (condM > RAT_CONST(1e8)) ? RAT_CONST(1e-3) : RAT_CONST(1e-6); 
		const fltp epsilon = eps_factor*M.diag().max();
		M.diag()+=epsilon;

		// walk over values
		arma::Mat<fltp> Vn(Vg.n_rows,4,arma::fill::zeros);
		for(arma::uword m=0;m<Vg.n_rows;m++){
			// solve without throwing errors
			arma::Col<fltp> x; if(arma::solve(x,M,P*Vg.row(m).t(),arma::solve_opts::force_approx))Vn.row(m) = x.t();
		}

		// return values at nodes
		return Vn;
	}

	// extrapolate values from gauss points for entire mesh
	arma::Mat<fltp> Tetrahedron::gauss2nodes(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n, 
		const arma::Mat<fltp> &Vg, 
		const arma::Mat<fltp> &Rqg, 
		const arma::Row<fltp> &wg,
		const arma::uword num_nodes){

		// check input
		assert(Rqg.n_rows==3);
		assert(n.n_rows==4);
		assert(Vg.n_cols%Rqg.n_cols==0);
		assert(n.max()<num_nodes);
		assert(Rqg.n_cols*n.n_cols==Vg.n_cols);

		// allocate output
		arma::Mat<fltp> Vnn(Vg.n_rows,4*n.n_cols);

		// walk over elements and perform local smoothing (in parallel)
		parfor(0,n.n_cols,true,[&](arma::uword i, int){
			Vnn.cols(i*4,(i+1)*4-1) = gauss2nodes(Rn.cols(n.col(i)),Rqg,wg,Vg.cols(i*Rqg.n_cols,(i+1)*Rqg.n_cols-1));
		});

		// average at nodes
		arma::Mat<fltp> Vn(Vg.n_rows,num_nodes,arma::fill::zeros);
		arma::Row<fltp> cnt(1,num_nodes);
		for(arma::uword i=0;i<n.n_cols;i++){
			Vn.cols(n.col(i)) += Vnn.cols(i*4,(i+1)*4-1); 
			cnt(n.col(i)) += 1;
		}

		// return values at nodes
		return Vn.each_row()/cnt;
	}


	// refine mesh
	void Tetrahedron::refine_mesh(arma::Mat<fltp>&Rn, arma::Mat<arma::uword>&n, const arma::uword num_refine){

		// walk over steps
		for(arma::uword i=0;i<num_refine;i++){
			// create midpoints on edges
			const arma::Mat<fltp> R0 = Rn.cols(n.row(0));
			const arma::Mat<fltp> R1 = Rn.cols(n.row(1));
			const arma::Mat<fltp> R2 = Rn.cols(n.row(2));
			const arma::Mat<fltp> R3 = Rn.cols(n.row(3));
			const arma::Mat<fltp> R01 = (R0 + R1)/2;
			const arma::Mat<fltp> R02 = (R0 + R2)/2;
			const arma::Mat<fltp> R03 = (R0 + R3)/2;
			const arma::Mat<fltp> R12 = (R1 + R2)/2;
			const arma::Mat<fltp> R13 = (R1 + R3)/2;
			const arma::Mat<fltp> R23 = (R2 + R3)/2;

			// form new tetrahedrons at corners
			const arma::Mat<fltp> Rc1 = arma::reshape(arma::join_vert(R0,R01,R02,R03),3,4*n.n_cols);
			const arma::Mat<fltp> Rc2 = arma::reshape(arma::join_vert(R01,R1,R12,R13),3,4*n.n_cols);
			const arma::Mat<fltp> Rc3 = arma::reshape(arma::join_vert(R02,R12,R2,R23),3,4*n.n_cols);
			const arma::Mat<fltp> Rc4 = arma::reshape(arma::join_vert(R03,R13,R23,R3),3,4*n.n_cols);

			// form new tetrahedrons in mid section
			const arma::Mat<fltp> Rm1 = arma::reshape(arma::join_vert(R01,R12,R02,R13),3,4*n.n_cols);
			const arma::Mat<fltp> Rm2 = arma::reshape(arma::join_vert(R01,R13,R02,R03),3,4*n.n_cols);
			const arma::Mat<fltp> Rm3 = arma::reshape(arma::join_vert(R02,R12,R23,R13),3,4*n.n_cols);
			const arma::Mat<fltp> Rm4 = arma::reshape(arma::join_vert(R02,R13,R23,R03),3,4*n.n_cols);

			// combine
			Rn = arma::join_horiz(arma::join_horiz(Rc1,Rc2,Rc3,Rc4),arma::join_horiz(Rm1,Rm2,Rm3,Rm4));

			// combine nodes in a mesh
			n = Extra::combine_nodes(Rn);
			n = arma::reshape(n,4,n.n_elem/4);
		}

		// done
		return;
	}

	// create unit icosahedron
	void Tetrahedron::create_icosahedron(arma::Mat<fltp>&Rn, arma::Mat<arma::uword>&n){
		// create 12 vertices of a icosahedron
		const fltp t = (1.0 + std::sqrt(5.0)) / 2.0; // golden ratio
		Rn.set_size(3,13);
		Rn.col(0) = arma::Col<fltp>::fixed<3>{-1,t,0};
		Rn.col(1) = arma::Col<fltp>::fixed<3>{ 1,t,0};
		Rn.col(2) = arma::Col<fltp>::fixed<3>{-1,-t,0};
		Rn.col(3) = arma::Col<fltp>::fixed<3>{1,-t,0};
		Rn.col(4) = arma::Col<fltp>::fixed<3>{ 0,-1,t};
		Rn.col(5) = arma::Col<fltp>::fixed<3>{ 0,1,t};
		Rn.col(6) = arma::Col<fltp>::fixed<3>{ 0,-1,-t};
		Rn.col(7) = arma::Col<fltp>::fixed<3>{ 0,1,-t};
		Rn.col(8) = arma::Col<fltp>::fixed<3>{ t,0,-1};
		Rn.col(9) = arma::Col<fltp>::fixed<3>{ t,0,1};
		Rn.col(10) = arma::Col<fltp>::fixed<3>{-t,0,-1};
		Rn.col(11) = arma::Col<fltp>::fixed<3>{-t,0,1};
		Rn.col(12) = arma::Col<fltp>::fixed<3>{0,0,0}; // this last point is the origin

		// create tetrahedrons
		n.set_size(4,20);
		n.col(0) = arma::Col<arma::uword>::fixed<4>{12, 0, 11, 5};
		n.col(1) = arma::Col<arma::uword>::fixed<4>{12, 0, 5, 1};
		n.col(2) = arma::Col<arma::uword>::fixed<4>{12, 0, 1, 7};
		n.col(3) = arma::Col<arma::uword>::fixed<4>{12, 0, 7, 10};
		n.col(4) = arma::Col<arma::uword>::fixed<4>{12, 0, 10, 11};
		
		n.col(5) = arma::Col<arma::uword>::fixed<4>{12, 1, 5, 9};
		n.col(6) = arma::Col<arma::uword>::fixed<4>{12, 5, 11, 4};
		n.col(7) = arma::Col<arma::uword>::fixed<4>{12, 11, 10, 2};
		n.col(8) = arma::Col<arma::uword>::fixed<4>{12, 10, 7, 6};
		n.col(9) = arma::Col<arma::uword>::fixed<4>{12, 7, 1, 8};

		n.col(10) = arma::Col<arma::uword>::fixed<4>{12, 3, 9, 4};
		n.col(11) = arma::Col<arma::uword>::fixed<4>{12, 3, 4, 2};
		n.col(12) = arma::Col<arma::uword>::fixed<4>{12, 3, 2, 6};
		n.col(13) = arma::Col<arma::uword>::fixed<4>{12, 3, 6, 8};
		n.col(14) = arma::Col<arma::uword>::fixed<4>{12, 3, 8, 9};

		n.col(15) = arma::Col<arma::uword>::fixed<4>{12, 4, 9, 5};
		n.col(16) = arma::Col<arma::uword>::fixed<4>{12, 2, 4, 11};
		n.col(17) = arma::Col<arma::uword>::fixed<4>{12, 6, 2, 10};
		n.col(18) = arma::Col<arma::uword>::fixed<4>{12, 8, 6, 7};
		n.col(19) = arma::Col<arma::uword>::fixed<4>{12, 9, 8, 1};

		// project to surface of unit sphere
		Rn.cols(0,11) = Rn.cols(0,11).eval().each_row()/cmn::Extra::vec_norm(Rn.cols(0,11));

		// doen
		return;
	}

	// extract the surface elements
	arma::Mat<arma::uword> Tetrahedron::extract_surface(const arma::Mat<arma::uword> &n){

		// must be tetrahedral mesh
		assert(n.n_rows==4);

		// get face matrix for tetrahedrons
		const arma::Mat<arma::uword>::fixed<4,3> F = get_faces();

		// face element matrices
		arma::Mat<arma::uword> f(F.n_cols, F.n_rows*n.n_cols);

		// walk over tetrahedrons
		for(arma::uword i=0;i<n.n_cols;i++){
			// get this tetrahedron
			const arma::Col<arma::uword> t = n.col(i);

			// walk over faces
			for(arma::uword j=0;j<F.n_rows;j++){
				// get faces from this hexahedron
				f.col(i*F.n_rows + j) = t.rows(arma::vectorise(F.row(j)));
			}
		}

		// create new matrix with sorted faces
		arma::Mat<arma::uword> fs(f.n_rows,f.n_cols);
		for(arma::uword i=0;i<f.n_cols;i++)
			fs.col(i) = arma::sort(f.col(i));

		// origin
		arma::Col<arma::uword> origin = arma::regspace<arma::Col<arma::uword> >(0,f.n_cols-1);

		// sort faces
		for(arma::uword i=0;i<fs.n_rows;i++){
			const arma::Col<arma::uword> sort_index = i==0 ? arma::sort_index(fs.row(i)).eval() : arma::stable_sort_index(fs.row(i)).eval();
			origin = origin(sort_index); fs = fs.cols(sort_index);
		}

		// find duplicate faces
		arma::Col<arma::uword> is_duplicate = arma::all(fs.head_cols(fs.n_cols-1)==fs.tail_cols(fs.n_cols-1)).t();
		arma::Col<arma::uword> is_surface(fs.n_cols,arma::fill::ones); 
		is_surface.rows(arma::find(is_duplicate)).fill(0); 
		is_surface.rows(arma::find(is_duplicate)+1).fill(0);

		// find surface mesh
		const arma::Mat<arma::uword> s = f.cols(origin(arma::find(is_surface)));

		// return surface
		return s;
	}

	// extract the surface elements
	// includes check for orientation of faces
	arma::Mat<arma::uword> Tetrahedron::extract_surface(const arma::Mat<fltp>&Rn, const arma::Mat<arma::uword> &n){
		// must be tetrahedral mesh
		assert(n.n_rows==4);

		// get face matrix for tetrahedrons
		const arma::Mat<arma::uword>::fixed<4,3> F = get_faces();

		// face element matrices
		arma::Mat<arma::uword> f(F.n_cols, F.n_rows*n.n_cols);

		// outward vector
		arma::Mat<fltp> V(3,F.n_rows*n.n_cols);

		// walk over tetrahedrons
		for(arma::uword i=0;i<n.n_cols;i++){
			// get this tetrahedron
			const arma::Col<arma::uword> t = n.col(i);

			// tetrahedron center point
			const arma::Col<fltp>::fixed<3> Rc = arma::mean(Rn.cols(n.col(i)),1);

			// walk over faces
			for(arma::uword j=0;j<F.n_rows;j++){
				// get faces from this hexahedron
				f.col(i*F.n_rows + j) = t.rows(arma::vectorise(F.row(j)));

				// face center point
				const arma::Col<fltp>::fixed<3> Rf = arma::mean(Rn.cols(f.col(i*F.n_rows + j)),1);

				// outward vector
				V.col(i*F.n_rows + j) = Rf - Rc;
			}
		}

		// orient faces
		const arma::Mat<fltp> Nf = Triangle::calc_face_normals(Rn,f);

		// check orientation and invert if needed
		const arma::Col<arma::uword> idx = arma::find(Extra::dot(Nf,V)<0);
		f.cols(idx) = Triangle::invert_elements(f.cols(idx));

		// create new matrix with sorted faces
		arma::Mat<arma::uword> fs(f.n_rows,f.n_cols);
		for(arma::uword i=0;i<f.n_cols;i++)
			fs.col(i) = arma::sort(f.col(i));

		// origin
		arma::Col<arma::uword> origin = arma::regspace<arma::Col<arma::uword> >(0,f.n_cols-1);

		// sort faces
		for(arma::uword i=0;i<fs.n_rows;i++){
			const arma::Col<arma::uword> sort_index = i==0 ? arma::sort_index(fs.row(i)).eval() : arma::stable_sort_index(fs.row(i)).eval();
			origin = origin(sort_index); fs = fs.cols(sort_index);
		}

		// find duplicate faces
		arma::Col<arma::uword> is_duplicate = arma::all(fs.head_cols(fs.n_cols-1)==fs.tail_cols(fs.n_cols-1)).t();
		arma::Col<arma::uword> is_surface(fs.n_cols,arma::fill::ones); 
		is_surface.rows(arma::find(is_duplicate)).fill(0); 
		is_surface.rows(arma::find(is_duplicate)+1).fill(0);

		// find surface mesh
		const arma::Mat<arma::uword> s = f.cols(origin(arma::find(is_surface)));

		// return surface
		return s;
	}

	// create tetrahedron mesh on a unit sphere using icosahedron
	// the sphere is centered on the origin
	void Tetrahedron::create_sphere(
		arma::Mat<fltp>&Rn, 
		arma::Mat<arma::uword>&n, 
		arma::Mat<arma::uword>&s, 
		const fltp sphere_radius, 
		const fltp element_size){

		// edge length of icosahedron
		const fltp ell_edge = 2*sphere_radius*std::sin(arma::Datum<fltp>::pi/5);

		// calculate number of refinements needed
		const arma::uword num_refine = arma::uword(std::log2(ell_edge/element_size));

		// create icosahedron
		create_icosahedron(Rn,n);

		// get face matrix for tetrahedrons
		// const arma::Mat<arma::uword>::fixed<4,3> F = get_faces();

		// refine
		for(arma::uword i=0;i<num_refine;i++){
			// refine once
			refine_mesh(Rn,n);

			// get surface
			s = extract_surface(n);

			// find unique surface nodes
			const arma::Col<arma::uword> idx = arma::unique(arma::vectorise(s));

			// project surface nodes back to sphere
			Rn.cols(idx) = Rn.cols(idx).eval().each_row()/cmn::Extra::vec_norm(Rn.cols(idx));
		}

		// scale sphere
		Rn*=sphere_radius;

		// done
		return;
	}


	// function for determining if points are inside a thetahedron
	arma::Row<arma::uword> Tetrahedron::is_inside(
		const arma::Mat<fltp> &R,
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n,
		const fltp tolerance){

		// check input
		assert(R.n_rows==3); 
		assert(Rn.n_rows==3);
		assert(n.n_rows==4);
		assert(n.n_elem>0);
		assert(arma::max(arma::max(n))<Rn.n_cols);

		// get number of elements
		arma::uword num_elem = n.n_cols;
		arma::uword num_points = R.n_cols;

		// allocate output (outside is nan)
		arma::Row<arma::uword> isinside(1,num_points,arma::fill::zeros);

		// get four corners
		arma::Mat<fltp> R0 = Rn.cols(n.row(0)); 
		arma::Mat<fltp> R1 = Rn.cols(n.row(1));
		arma::Mat<fltp> R2 = Rn.cols(n.row(2)); 
		arma::Mat<fltp> R3 = Rn.cols(n.row(3));

		// walk over target points
		for(arma::uword k=0;k<num_points;k++){
		//parfor(0,num_points,true,[&](int k, int){
			// allocate determinant
			arma::Mat<fltp> determinants(5,num_elem);
			determinants.row(0) = Tetrahedron::special_determinant(R0,R1,R2,R3);

			// create array from point
			arma::Mat<fltp> Rp(3,num_elem);
			Rp.row(0).fill(R(0,k)); Rp.row(1).fill(R(1,k)); Rp.row(2).fill(R(2,k));

			// calculate determinants
			determinants.row(1) = Tetrahedron::special_determinant(Rp,R1,R2,R3);
			determinants.row(2) = Tetrahedron::special_determinant(R0,Rp,R2,R3);
			determinants.row(3) = Tetrahedron::special_determinant(R0,R1,Rp,R3);
			determinants.row(4) = Tetrahedron::special_determinant(R0,R1,R2,Rp);

			// get index
			// arma::Row<arma::uword> idx = 
			// 	arma::find(arma::all(determinants>-1e-12,0) || 
			// 	arma::all(determinants<1e-12,0)).t();

			// check that point is not in more than one tetrahedron
			// this can happen now that we have set 1e-12 tolerance
			// assert(idx.n_elem<=1);
			
			// store
			// if(!idx.is_empty())isinside(k) = true;

			// check if inside
			isinside(k) = arma::as_scalar(arma::any(
				arma::all(determinants>-tolerance,0) || 
				arma::all(determinants<tolerance,0),1));

		//});
		}

		// return array of 
		return isinside;
	}

	// volume of a tetrahedron from its four corners
	// https://en.wikipedia.org/wiki/Tetrahedron#Volume
	arma::Row<fltp> Tetrahedron::calc_volume(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n){

		// check that input is in three dimensional space
		assert(n.n_rows==4); assert(Rn.n_rows==3);
		assert(arma::max(arma::max(n))<Rn.n_cols);

		// get points
		const arma::Mat<fltp> R0 = Rn.cols(n.row(0)); const arma::Mat<fltp> R1 = Rn.cols(n.row(1));
		const arma::Mat<fltp> R2 = Rn.cols(n.row(2)); const arma::Mat<fltp> R3 = Rn.cols(n.row(3));

		// calculate
		const arma::Row<fltp> V = arma::abs(Extra::dot((R0-R3),Extra::cross((R1-R3),(R2-R3))))/6;
		
		// return volume array
		return V;
	}

	// shape function
	arma::Mat<fltp> Tetrahedron::shape_function(const arma::Mat<fltp> &Rq){
		// check input
		assert(Rq.n_rows==3);

		// calculate shape functions
		const arma::Mat<fltp> Nie = arma::join_vert(RAT_CONST(1.0) - arma::sum(Rq,0), Rq);

		// return 
		return Nie;
	}

	// partial derivatives of hexahedron shape function in quadrilateral coordinates
	// format: [dNdnu;dNdmu;dNdxi]
	arma::Mat<fltp> Tetrahedron::shape_function_derivative(
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rq.n_rows==3);

		// calculate shape functions
		arma::Mat<fltp> dNie(12,Rq.n_cols);
		dNie.each_col() = arma::Col<fltp>::fixed<12>{
			-RAT_CONST(1.0), RAT_CONST(1.0),RAT_CONST(0.0),RAT_CONST(0.0), 
			-RAT_CONST(1.0), RAT_CONST(0.0),RAT_CONST(1.0),RAT_CONST(0.0), 
			-RAT_CONST(1.0), RAT_CONST(0.0),RAT_CONST(0.0),RAT_CONST(1.0)};

		// return 
		return dNie;
	}

	// shape function for Hdiv elements (Raviart Thomas, Nedelec)
	// divergence conforming facet element
	// https://defelement.org/elements/examples/tetrahedron-raviart-thomas-lagrange-0.html
	// input coordinates are given as [xi;nu;mu]
	arma::Mat<fltp> Tetrahedron::raviart_thomas_hdiv_shape_function(const arma::Mat<fltp> &Rq){
		// check input
		assert(Rq.n_rows==3);

		// get quadrilateral coordinates
		const arma::Row<fltp> xi = Rq.row(0);
		const arma::Row<fltp> nu = Rq.row(1);
		const arma::Row<fltp> mu = Rq.row(2);

		// calculate shape function
		arma::Mat<fltp> N(12,Rq.n_cols,arma::fill::zeros);

		// shape function for face 1
		N.row(0*3 + 0) = xi;
		N.row(0*3 + 1) = nu;
		N.row(0*3 + 2) = mu - 1.0;

		// shape function for face 2
		N.row(1*3 + 0) = xi;
		N.row(1*3 + 1) = nu - 1.0;
		N.row(1*3 + 2) = mu;

		// shape function for face 3 
		N.row(2*3 + 0) = xi - 1.0;
		N.row(2*3 + 1) = nu;
		N.row(2*3 + 2) = mu;

		// shape function for face 4
		N.row(3*3 + 0) = xi;
		N.row(3*3 + 1) = nu;
		N.row(3*3 + 2) = mu;

		// return shape function
		return 2*N;
	}

	// // shape function for Hcurl elements
	// // curl conforming edge element
	// // https://defelement.org/elements/examples/tetrahedron-nedelec1-lagrange-0.html
	// // input coordinates are given as [xi;nu;mu]
	// arma::Mat<fltp> Tetrahedron::nedelec_hcurl_shape_function(const arma::Mat<fltp> &Rq){
	// 	// check input
	// 	assert(Rq.n_rows==3);

	// 	// get quadrilateral coordinates
	// 	const arma::Row<fltp> xi = Rq.row(0);
	// 	const arma::Row<fltp> nu = Rq.row(1);
	// 	const arma::Row<fltp> mu = Rq.row(2);

	// 	// calculate shape function
	// 	arma::Mat<fltp> N(18,Rq.n_cols,arma::fill::zeros);

	// 	N.row( 0*3 + 1) = -mu;
	// 	N.row( 0*3 + 2) = nu;
		
	// 	N.row( 1*3 + 0) = -mu;
	// 	N.row( 1*3 + 2) = xi;

	// 	N.row( 2*3 + 0) = -nu;
	// 	N.row( 2*3 + 1) = xi;

	// 	N.row( 3*3 + 0) = mu;
	// 	N.row( 3*3 + 1) = mu;
	// 	N.row( 3*3 + 2) = -xi-nu+1.0;

	// 	N.row( 4*3 + 0) = nu;
	// 	N.row( 4*3 + 1) = -xi-mu+1.0;
	// 	N.row( 4*3 + 2) = nu;

	// 	N.row( 5*3 + 0) = -nu-mu+1.0;
	// 	N.row( 5*3 + 1) = xi;
	// 	N.row( 5*3 + 2) = xi;

	// 	// return he shape function
	// 	return 2*N;
	// }

	// convert derivative of shape function into carthesian coordinates
	// using the inverted jacobian
	arma::Mat<fltp> Tetrahedron::derivative2cart(
		const arma::Mat<fltp>&Jinv,
		const arma::Row<fltp>&/*Jdet*/,
		const arma::Mat<fltp>&dN){

		// calculate cartesian derivatives
		arma::Mat<fltp> dNcart(12,Jinv.n_cols);
		for(arma::uword i=0;i<Jinv.n_cols;i++){
			// get jacobian matrix and make square
			const arma::Mat<fltp>::fixed<3,3> A = arma::reshape(Jinv.col(i),3,3);

			// multiply inverted jacobian with the shape function derivative vectors
			dNcart.col(i) = arma::reshape((A*arma::reshape(dN.col(i),4,3).t()).t(),12,1);
		}

		// return carthesian derivatives
		return dNcart;
	}

	// quadrilateral coordinates to carthesian coordinates
	// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
	arma::Mat<fltp> Tetrahedron::shape_function_derivative_cart(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<fltp> &Rq){

		// get shape derivative of shape function
		const arma::Mat<fltp> dN = shape_function_derivative(Rq);

		const arma::Mat<fltp> J = shape_function_jacobian(Rn,dN);

		const arma::Row<fltp> Jdet = jacobian2determinant(J);

		const arma::Mat<fltp> Jinv = invert_jacobian(J,Jdet);

		// use ufnction to convert derivatives to cartesian coordinates
		const arma::Mat<fltp> dNcart = derivative2cart(Jinv,Jdet,dN);

		// return derivative
		return dNcart;
	}

	// carthesian derivative calculated at quadrilateral coordinates
	// phi is the quantity at the nodes
	// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]; 
	// dphi = [dphidx;dphidy;dphidz]
	arma::Mat<fltp> Tetrahedron::quad2cart_derivative(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq,
		const arma::Mat<fltp> &phi){

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==4);
		assert(Rq.n_rows==3); assert(phi.n_cols==4);

		// get carthesian derivative of shape functions
		const arma::Mat<fltp> dNie = shape_function_derivative_cart(Rn,Rq);

		// get coordinates from matrix vector product
		const arma::Mat<fltp> dphi = arma::reshape(phi*arma::reshape(dNie,4,3*Rq.n_cols),3*phi.n_rows,Rq.n_cols);

		// return cartesian coords
		return dphi;
	}

	// quadrilateral coordinates to carthesian coordinates
	arma::Mat<fltp> Tetrahedron::quad2cart(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rn.n_cols==4); assert(Rq.n_rows==3);
		
		// get shape functions
		const arma::Mat<fltp> Nie = shape_function(Rq);

		// get coordinates from matrix vector product
		const arma::Mat<fltp> Rc = Rn*Nie;

		// return cartesian coords
		return Rc;
	}

	// carthesian coordinates to quadrilateral coordinates
	// this function can likely be improved by not using finite difference
	// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
	arma::Mat<fltp> Tetrahedron::cart2quad(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rc, 
		const fltp tol){
		
		// get typical size
		const fltp diag = arma::as_scalar(cmn::Extra::vec_norm(Rn.col(1) - Rn.col(0)));

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==4);
		assert(Rc.n_rows==3);
		
		// initial guess
		arma::Mat<fltp> Rq(3,Rc.n_cols,arma::fill::zeros);

		// iterations
		// const fltp delta = RAT_CONST(1e-2);
		for(arma::uword i=0;i<100;i++){
			// calculate difference
			const arma::Mat<fltp> Rc00 = quad2cart(Rn,Rq);
			const arma::Mat<fltp> dRc = Rc00 - Rc;

			// check for convergence
			const fltp conv = arma::max(arma::max(arma::abs(dRc)));
			if(conv<tol*diag)break;

			// calculate jacobian
			const arma::Mat<fltp> J = shape_function_jacobian(Rn,shape_function_derivative(Rq));

			// invert jacobian
			const arma::Row<fltp> Jdet = jacobian2determinant(J);
			const arma::Mat<fltp> Jinv = invert_jacobian(J,Jdet);

			// shift coordinates
			Rq -= arma::join_vert(
				Jinv.row(0)%dRc.row(0) + Jinv.row(1)%dRc.row(1) + Jinv.row(2)%dRc.row(2), 
				Jinv.row(3)%dRc.row(0) + Jinv.row(4)%dRc.row(1) + Jinv.row(5)%dRc.row(2),
				Jinv.row(6)%dRc.row(0) + Jinv.row(7)%dRc.row(1) + Jinv.row(8)%dRc.row(2));

			// // calculate change
			// for(arma::uword j=0;j<Rc.n_cols;j++){
			// 	// assemble matrix
			// 	// arma::Mat<fltp>::fixed<3,3> A;
			// 	// A.col(0) = dRc0.col(j); A.col(1) = dRc1.col(j); A.col(2) = dRc2.col(j);
			// 	arma::Mat<fltp>::fixed<3,3> A = -arma::reshape(J.col(j),3,3).t();

			// 	// update solution
			// 	Rq.col(j) += arma::solve(A,dRc.col(j), arma::solve_opts::fast);
			// }
		}

		// return carthesian coordinates
		return Rq;
	}

	// calculate curl
	arma::Mat<fltp> Tetrahedron::quad2cart_curl(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq,
		const arma::Mat<fltp> &V){

		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==4);
		assert(Rq.n_rows==3); assert(V.n_cols==4);
		assert(V.n_rows==3);

		// get derivative of V at Rq
		arma::Mat<fltp> dV = quad2cart_derivative(Rn,Rq,V);

		// calculate cross product
		// dV = [dVx/dx (0), dVy/dx (1), dVz/dx (2),
		// 		 dVx/dy (3), dVy/dy (4), dVz/dy (5),
		// 		 dVx/dz (6), dVy/dz (7), dVz/dz (8)]
		arma::Mat<fltp> curl(3,Rq.n_cols);
		curl.row(0) = dV.row(5) - dV.row(7);
		curl.row(1) = dV.row(6) - dV.row(2);
		curl.row(2) = dV.row(1) - dV.row(3);

		// output calculated curl
		return curl;
	}

	arma::Mat<fltp> Tetrahedron::shape_function_jacobian(
		const arma::Mat<fltp>&Rn,
		const arma::Mat<fltp>&dN){
		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==4);

		// get shape derivative of shape function
		const arma::Mat<fltp> dNiedxi = dN.rows(0,3); 
		const arma::Mat<fltp> dNiednu = dN.rows(4,7); 
		const arma::Mat<fltp> dNiedmu = dN.rows(8,11);

		// get node coordinate components
		const arma::Row<fltp> xn = Rn.row(0);
		const arma::Row<fltp> yn = Rn.row(1);
		const arma::Row<fltp> zn = Rn.row(2);

		// setup jacobian
		arma::Mat<fltp> J(9,dN.n_cols);
		J.row(0) = xn*dNiedxi; J.row(1) = xn*dNiednu; J.row(2) = xn*dNiedmu;
		J.row(3) = yn*dNiedxi; J.row(4) = yn*dNiednu; J.row(5) = yn*dNiedmu;
		J.row(6) = zn*dNiedxi; J.row(7) = zn*dNiednu; J.row(8) = zn*dNiedmu;

		// return jacobian
		return J;
	}

	// method for inverting jacobian matrix
	arma::Mat<fltp> Tetrahedron::invert_jacobian(
		const arma::Mat<fltp> &J,
		const arma::Row<fltp> &Jdet){ 

		// same as hexahedron
		return Hexahedron::invert_jacobian(J,Jdet);
	}

	// convert quadrilateral vectors to cartesian vectors 
	// using contravariant piola mapping. This preserves 
	// the tangential component of basis functions.
	// for more info see: https://defelement.com/ciarlet.html
	arma::Mat<fltp> Tetrahedron::quad2cart_contravariant_piola(
		const arma::Mat<fltp> &Vq, 
		const arma::Mat<fltp> &J, 
		const arma::Row<fltp> &Jdet){

		// return carthesian vectors
		return Hexahedron::quad2cart_contravariant_piola(Vq,J,Jdet);
	}

	// convert quadrilateral vectors to cartesian vectors using
	// using covariant piola mapping. This preserves 
	// the normal component of basis functions on facets.
	// for more info see: https://defelement.com/ciarlet.html
	arma::Mat<fltp> Tetrahedron::quad2cart_covariant_piola(
		const arma::Mat<fltp> &Vq, 
		const arma::Mat<fltp> &Jinv){

		// same as hexahedron
		return Hexahedron::quad2cart_covariant_piola(Vq,Jinv);
	}

	// create gauss points
	arma::Mat<fltp> Tetrahedron::create_gauss_points(const arma::sword num_gauss){
		// create gauss points
		arma::Mat<fltp> Rqg;

		// KEAST0, the Keast Rule, order 1, degree of precision 1. 
		if(std::abs(num_gauss)==1){
			Rqg.set_size(4,1);
			Rqg.col(0) = arma::Col<fltp>::fixed<4>{1.0/4,1.0/4,1.0/4, 1.0/6};
		}

		// KEAST1, the Keast Rule, order 4, precision 2. 
		else if(std::abs(num_gauss)==2){
			Rqg.set_size(4,4);
			Rqg.col(0) = arma::Col<fltp>{0.5854101966249685,0.1381966011250105,0.1381966011250105, 0.2500000000000000/6};
			Rqg.col(1) = arma::Col<fltp>{0.1381966011250105,0.1381966011250105,0.1381966011250105, 0.2500000000000000/6};
			Rqg.col(2) = arma::Col<fltp>{0.1381966011250105,0.1381966011250105,0.5854101966249685, 0.2500000000000000/6};
			Rqg.col(3) = arma::Col<fltp>{0.1381966011250105,0.5854101966249685,0.1381966011250105, 0.2500000000000000/6};
		}

		// KEAST2, the Keast Rule, order 5, degree of precision 3. 
		else if(std::abs(num_gauss)==3){
			Rqg.set_size(4,5);
			Rqg.col(0) = arma::Col<fltp>{0.2500000000000000,0.2500000000000000,0.2500000000000000, -0.8000000000000000/6};
			Rqg.col(1) = arma::Col<fltp>{0.5000000000000000,0.1666666666666667,0.1666666666666667, 0.4500000000000000/6};
			Rqg.col(2) = arma::Col<fltp>{0.1666666666666667,0.1666666666666667,0.1666666666666667, 0.4500000000000000/6};
			Rqg.col(3) = arma::Col<fltp>{0.1666666666666667,0.1666666666666667,0.5000000000000000, 0.4500000000000000/6};
			Rqg.col(4) = arma::Col<fltp>{0.1666666666666667,0.5000000000000000,0.1666666666666667, 0.4500000000000000/6};
		}

		// KEAST4, the Keast Rule, order 11, degree of precision 4. 
		else if(std::abs(num_gauss)==4){
			Rqg.set_size(4,11);
			Rqg.row(0) = arma::Row<fltp>{0.2500000000000000, 0.7857142857142857, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.1005964238332008, 0.3994035761667992, 0.3994035761667992, 0.3994035761667992, 0.1005964238332008, 0.1005964238332008};
			Rqg.row(1) = arma::Row<fltp>{0.2500000000000000, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.7857142857142857, 0.3994035761667992, 0.1005964238332008, 0.3994035761667992, 0.1005964238332008, 0.3994035761667992, 0.1005964238332008};
			Rqg.row(2) = arma::Row<fltp>{0.2500000000000000, 0.0714285714285714, 0.0714285714285714, 0.7857142857142857, 0.0714285714285714, 0.3994035761667992, 0.3994035761667992, 0.1005964238332008, 0.1005964238332008, 0.1005964238332008, 0.3994035761667992};
			Rqg.row(3) = (RAT_CONST(1.0)/6)*arma::Row<fltp>{-0.0789333333333333, 0.0457333333333333, 0.0457333333333333, 0.0457333333333333, 0.0457333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333};
		}

		// KEAST6, the Keast Rule, order 15, degree of precision 5. 
		else if(std::abs(num_gauss)==5){
			Rqg.set_size(4,15);
			Rqg.row(0) = arma::Row<fltp>{0.2500000000000000, 0.0000000000000000, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.7272727272727273, 0.0909090909090909, 0.0909090909090909, 0.0909090909090909, 0.4334498464263357, 0.0665501535736643, 0.0665501535736643, 0.0665501535736643, 0.4334498464263357, 0.4334498464263357};
			Rqg.row(1) = arma::Row<fltp>{0.2500000000000000, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0000000000000000, 0.0909090909090909, 0.0909090909090909, 0.0909090909090909, 0.7272727272727273, 0.0665501535736643, 0.4334498464263357, 0.0665501535736643, 0.4334498464263357, 0.0665501535736643, 0.4334498464263357};
			Rqg.row(2) = arma::Row<fltp>{0.2500000000000000, 0.3333333333333333, 0.3333333333333333, 0.0000000000000000, 0.3333333333333333, 0.0909090909090909, 0.0909090909090909, 0.7272727272727273, 0.0909090909090909, 0.0665501535736643, 0.0665501535736643, 0.4334498464263357, 0.4334498464263357, 0.4334498464263357, 0.0665501535736643};
			Rqg.row(3) = (RAT_CONST(1.0)/6)*arma::Row<fltp>{0.1817020685825351, 0.0361607142857143, 0.0361607142857143, 0.0361607142857143, 0.0361607142857143, 0.0698714945161738, 0.0698714945161738, 0.0698714945161738, 0.0698714945161738, 0.0656948493683187, 0.0656948493683187, 0.0656948493683187, 0.0656948493683187, 0.0656948493683187, 0.0656948493683187};
		}

		// KEAST7, the Keast Rule, order 24, degree of precision 6. 
		else if(std::abs(num_gauss)==6){
			Rqg.set_size(4,24);
			Rqg.row(0) = arma::Row<fltp>{3.561913862225449e-01,2.146028712591517e-01,2.146028712591517e-01,2.146028712591517e-01,8.779781243961660e-01,4.067395853461130e-02,4.067395853461130e-02,4.067395853461130e-02,3.298632957317310e-02,3.223378901422757e-01,3.223378901422757e-01,3.223378901422757e-01,2.696723314583159e-01,6.366100187501750e-02,6.366100187501750e-02,6.030056647916491e-01,6.366100187501750e-02,6.366100187501750e-02,6.366100187501750e-02,2.696723314583159e-01,6.030056647916491e-01,6.366100187501750e-02,2.696723314583159e-01,6.030056647916491e-01};
			Rqg.row(1) = arma::Row<fltp>{2.146028712591517e-01,2.146028712591517e-01,2.146028712591517e-01,3.561913862225449e-01,4.067395853461130e-02,4.067395853461130e-02,4.067395853461130e-02,8.779781243961660e-01,3.223378901422757e-01,3.223378901422757e-01,3.223378901422757e-01,3.298632957317310e-02,6.366100187501750e-02,2.696723314583159e-01,6.366100187501750e-02,6.366100187501750e-02,6.030056647916491e-01,6.366100187501750e-02,2.696723314583159e-01,6.030056647916491e-01,6.366100187501750e-02,6.030056647916491e-01,6.366100187501750e-02,2.696723314583159e-01};
			Rqg.row(2) = arma::Row<fltp>{2.146028712591517e-01,2.146028712591517e-01,3.561913862225449e-01,2.146028712591517e-01,4.067395853461130e-02,4.067395853461130e-02,8.779781243961660e-01,4.067395853461130e-02,3.223378901422757e-01,3.223378901422757e-01,3.298632957317310e-02,3.223378901422757e-01,6.366100187501750e-02,6.366100187501750e-02,2.696723314583159e-01,6.366100187501750e-02,6.366100187501750e-02,6.030056647916491e-01,6.030056647916491e-01,6.366100187501750e-02,2.696723314583159e-01,2.696723314583159e-01,6.030056647916491e-01,6.366100187501750e-02};
			Rqg.row(3) = (RAT_CONST(1.0)/6)*arma::Row<fltp>{0.0399227502581679,0.0399227502581679,0.0399227502581679,0.0399227502581679,0.0100772110553207,0.0100772110553207,0.0100772110553207,0.0100772110553207,0.0553571815436544,0.0553571815436544,0.0553571815436544,0.0553571815436544,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857};
		}

		// KEAST9, the Keast Rule, order 45, degree of precision 8. 
		else if(std::abs(num_gauss)>6){
			Rqg.set_size(4,45);
			Rqg.row(0) = arma::Row<fltp>{2.500000000000000e-01,6.175871903000830e-01,1.274709365666390e-01,1.274709365666390e-01,1.274709365666390e-01,9.037635088221031e-01,3.207883039263230e-02,3.207883039263230e-02,3.207883039263230e-02,4.502229043567190e-01,4.977709564328100e-02,4.977709564328100e-02,4.977709564328100e-02,4.502229043567190e-01,4.502229043567190e-01,3.162695526014501e-01,1.837304473985499e-01,1.837304473985499e-01,1.837304473985499e-01,3.162695526014501e-01,3.162695526014501e-01,2.291778784481710e-02,2.319010893971509e-01,2.319010893971509e-01,5.132800333608811e-01,2.319010893971509e-01,2.319010893971509e-01,2.319010893971509e-01,2.291778784481710e-02,5.132800333608811e-01,2.319010893971509e-01,2.291778784481710e-02,5.132800333608811e-01,7.303134278075384e-01,3.797004847182860e-02,3.797004847182860e-02,1.937464752488044e-01,3.797004847182860e-02,3.797004847182860e-02,3.797004847182860e-02,7.303134278075384e-01,1.937464752488044e-01,3.797004847182860e-02,7.303134278075384e-01,1.937464752488044e-01};
			Rqg.row(1) = arma::Row<fltp>{2.500000000000000e-01,1.274709365666390e-01,1.274709365666390e-01,1.274709365666390e-01,6.175871903000830e-01,3.207883039263230e-02,3.207883039263230e-02,3.207883039263230e-02,9.037635088221031e-01,4.977709564328100e-02,4.502229043567190e-01,4.977709564328100e-02,4.502229043567190e-01,4.977709564328100e-02,4.502229043567190e-01,1.837304473985499e-01,3.162695526014501e-01,1.837304473985499e-01,3.162695526014501e-01,1.837304473985499e-01,3.162695526014501e-01,2.319010893971509e-01,2.291778784481710e-02,2.319010893971509e-01,2.319010893971509e-01,5.132800333608811e-01,2.319010893971509e-01,2.291778784481710e-02,5.132800333608811e-01,2.319010893971509e-01,5.132800333608811e-01,2.319010893971509e-01,2.291778784481710e-02,3.797004847182860e-02,7.303134278075384e-01,3.797004847182860e-02,3.797004847182860e-02,1.937464752488044e-01,3.797004847182860e-02,7.303134278075384e-01,1.937464752488044e-01,3.797004847182860e-02,1.937464752488044e-01,3.797004847182860e-02,7.303134278075384e-01};
			Rqg.row(2) = arma::Row<fltp>{2.500000000000000e-01,1.274709365666390e-01,1.274709365666390e-01,6.175871903000830e-01,1.274709365666390e-01,3.207883039263230e-02,3.207883039263230e-02,9.037635088221031e-01,3.207883039263230e-02,4.977709564328100e-02,4.977709564328100e-02,4.502229043567190e-01,4.502229043567190e-01,4.502229043567190e-01,4.977709564328100e-02,1.837304473985499e-01,1.837304473985499e-01,3.162695526014501e-01,3.162695526014501e-01,3.162695526014501e-01,1.837304473985499e-01,2.319010893971509e-01,2.319010893971509e-01,2.291778784481710e-02,2.319010893971509e-01,2.319010893971509e-01,5.132800333608811e-01,5.132800333608811e-01,2.319010893971509e-01,2.291778784481710e-02,2.291778784481710e-02,5.132800333608811e-01,2.319010893971509e-01,3.797004847182860e-02,3.797004847182860e-02,7.303134278075384e-01,3.797004847182860e-02,3.797004847182860e-02,1.937464752488044e-01,1.937464752488044e-01,3.797004847182860e-02,7.303134278075384e-01,7.303134278075384e-01,1.937464752488044e-01,3.797004847182860e-02};
			Rqg.row(3) = (RAT_CONST(1.0)/6)*arma::Row<fltp>{-2.359620398477557e-01,2.448789635605620e-02,2.448789635605620e-02,2.448789635605620e-02,2.448789635605620e-02,3.948520639826100e-03,3.948520639826100e-03,3.948520639826100e-03,3.948520639826100e-03,2.630555295073710e-02,2.630555295073710e-02,2.630555295073710e-02,2.630555295073710e-02,2.630555295073710e-02,2.630555295073710e-02,8.298038305505891e-02,8.298038305505891e-02,8.298038305505891e-02,8.298038305505891e-02,8.298038305505891e-02,8.298038305505891e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,2.544262454810230e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02,1.343243843768520e-02};
		}

		// not available
		else{
			rat_throw_line("table does not exist");
		}

		// return gauss points and weights
		return Rqg;
	}

	// calculate determinant of the jacobian
	arma::Row<fltp> Tetrahedron::jacobian2determinant(const arma::Mat<fltp> &J){
		// same as hexahedron
		return Hexahedron::jacobian2determinant(J);
	}

	// check if faces are correct
	arma::Row<arma::uword> Tetrahedron::is_clockwise(
		const arma::Mat<fltp> &R, 
		const arma::Mat<arma::uword> &n){
		assert(R.n_rows==3); assert(n.n_rows==4);
		const arma::Col<fltp>::fixed<3> Rq{0.25,0.25,0.25}; // at midpoint
		const arma::Mat<fltp> dN = shape_function_derivative(Rq);
		arma::Row<fltp> Jdet(n.n_cols);
		for(arma::uword i=0;i<n.n_cols;i++){
			const arma::Mat<fltp>::fixed<3,4> Rn = R.cols(n.col(i));
			const arma::Mat<fltp> J = shape_function_jacobian(Rn,dN);
			Jdet(i) = arma::as_scalar(jacobian2determinant(J));
		}
		return Jdet>RAT_CONST(0.0);
	}

	// invert the elements
	arma::Mat<arma::uword> Tetrahedron::invert_elements(const arma::Mat<arma::uword>&n){
		assert(n.n_rows==4);
		return n.rows(arma::Col<arma::uword>{0,3,2,1});
	}


	// LINE 
	// corner nodes in quadrilateral coordinates
	Line::CornerNodeMatrix Line::get_corner_nodes(){
		// setup matrix
		arma::Mat<arma::sword>::fixed<2,1> M = {-1,+1}; 

		// return matrix
		return M;
	}

	// create gauss points
	arma::Mat<fltp> Line::create_gauss_points(const arma::sword num_gauss){
		// create gauss points in quad coords
		const cmn::Gauss gs(num_gauss);
		const arma::Row<fltp>& wg = gs.get_weights();
		const arma::Row<fltp>& xg = gs.get_abscissae();
		const arma::Mat<fltp> Rq = arma::join_vert(xg,wg);

		// check weights
		assert(std::abs(arma::accu(Rq.row(1))-2.0)<10*gs.get_tol());

		// return gauss points and weights
		return Rq;
	}

	// quadrilateral shape function
	arma::Mat<fltp> Line::shape_function(
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rq.n_rows==1);

		// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
		arma::Mat<arma::sword>::fixed<2,1> M = Line::get_corner_nodes();

		// calculate shape functions
		arma::Mat<fltp> Nie(2,Rq.n_cols);

		// fill matrix
		for(arma::uword i=0;i<2;i++){
			Nie.row(i) = (1.0+Rq.row(0)*fltp(M(i,0)))/2;
		}

		// return 
		return Nie;
	}

	// quadrilateral shape function derivative in quadrilateral coordinates
	// format: [dNdnu;dNdxi]
	arma::Mat<fltp> Line::shape_function_derivative(
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rq.n_rows==1);

		// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
		arma::Mat<arma::sword>::fixed<2,1> M = Line::get_corner_nodes();

		// calculate shape functions
		arma::Mat<fltp> Nie(2,Rq.n_cols);

		// fill matrix
		for(arma::uword i=0;i<2;i++){
			Nie.row(i) = fltp(M(i,0))*(1 + Rq.row(0))/2;
		}

		// return 
		return Nie;
	}

	// matrix for converting from line coordiantes to 3D coordinates
	arma::Col<fltp>::fixed<3> Line::surface_to_volume_matrix(const arma::Mat<fltp>&Rn){
		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==2);

		// get edge
		const arma::Col<fltp>::fixed<3> V1 = Extra::normalize(Rn.col(1) - Rn.col(0));

		return V1;
	}

	// jacobian functions
	arma::Row<fltp> Line::shape_function_jacobian(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<fltp> &dN){
		
		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==2);

		// get edge
		const arma::Col<fltp>::fixed<3> M = surface_to_volume_matrix(Rn);

		// calculate cartesian coordinates along line
		const arma::Row<fltp> u = M.t()*Rn;

		// setup jacobian
		const arma::Row<fltp> J = u*dN;

		// return jacobian
		return J;
	}

	// nedelec shape functions
	arma::Mat<fltp> Line::raviart_thomas_hdiv_shape_function(const arma::Mat<fltp> &Rq){
		// check input
		assert(Rq.n_rows==1);

		// get coordinates
		const arma::Row<fltp>& xi = Rq;

		// calculate shape function
		arma::Mat<fltp> N(2,Rq.n_cols,arma::fill::zeros);
		N.row(0) = RAT_CONST(0.5)*xi - RAT_CONST(0.5); // shape function for node 0, runs from 0 to 1
		N.row(1) = RAT_CONST(0.5) + RAT_CONST(0.5)*xi; // shape function for node 1, runs from -1 to 0

		// return shape function
		// all vectors point inwards
		return -N;
	}

	// convert quadrilateral vectors to cartesian vectors 
	// using contravariant piola mapping. This preserves 
	// the tangential component of basis functions.
	// for more info see: https://defelement.com/ciarlet.html
	arma::Mat<fltp> Line::quad2cart_contravariant_piola(
		const arma::Mat<fltp> &Vq, 
		const arma::Row<fltp> &J, 
		const arma::Row<fltp> &Jdet){

		// check input
		assert(Vq.n_cols==J.n_cols);
		assert(J.n_cols==Jdet.n_cols);

		// walk over gauss points
		const arma::Mat<fltp> Vc = Vq.each_row()%(J/Jdet);

		// return carthesian vectors
		return Vc;
	}


	// convert quadrilateral vectors to cartesian vectors using
	// using covariant piola mapping. This preserves 
	// the normal component of basis functions on facets.
	// for more info see: https://defelement.com/ciarlet.html
	arma::Mat<fltp> Line::quad2cart_covariant_piola(
		const arma::Mat<fltp> &Vq, 
		const arma::Row<fltp> &Jinv){

		// walk over gauss points
		const arma::Mat<fltp> Vc = Vq.each_row()%Jinv;

		// return carthesian vectors
		return Vc;
	}

	// calculate determinant for the jacobian matrices
	arma::Row<fltp> Line::jacobian2determinant(const arma::Row<fltp>&J){
		return J;
	}

	// invert each jacobian matrix
	arma::Row<fltp> Line::invert_jacobian(const arma::Row<fltp>&J, const arma::Row<fltp>&/*Jdet*/){
		// check input
		assert(arma::all(J!=0));

		// invert matrices (stored columnwise)
		const arma::Row<fltp> Jinv = RAT_CONST(1.0)/J;

		// return inverted matrices
		return Jinv;
	}

	// quadrilateral coordinates to carthesian coordinates
	arma::Mat<fltp> Line::quad2cart(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rn.n_cols==2); assert(Rq.n_rows==1);
		
		// get shape functions
		arma::Mat<fltp> Nie = Line::shape_function(Rq);

		// get coordinates from matrix vector product
		arma::Mat<fltp> Rc = Rn*Nie;

		// return cartesian coords
		return Rc;
	}

	// volume hexahedrons in a mesh
	arma::Row<fltp> Line::calc_length(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n){
		// check input
		assert(n.n_rows==2);
		assert(Rn.n_rows==3 || Rn.n_rows==2);

		// difference between connected nodes
		return Extra::vec_norm(Rn.cols(n.row(1))-Rn.cols(n.row(0)));
	}

	// Extrapolate values from Gauss points to nodes
	// using the discrete local smoothing technique for a line element.
	arma::Mat<fltp> Line::gauss2nodes(
		const arma::Mat<fltp> &Rn,    // Nodal coordinates of the line element.
		const arma::Mat<fltp> &Rqg,   // Gauss point coordinates in the reference element (1 x numGaussPts).
		const arma::Row<fltp> &wg,    // Gauss quadrature weights.
		const arma::Mat<fltp> &Vg)    // Values at Gauss points that need to be extrapolated.
	{
		// For a line element, the reference coordinate is 1D.
		assert(Rqg.n_rows == 1);
		// Number of Gauss points should match the number of columns in Vg.
		assert(Rqg.n_cols == Vg.n_cols);

		// Compute derivative of shape functions at Gauss points.
		const arma::Mat<fltp> dN = shape_function_derivative(Rqg);
		// Compute the Jacobian matrix at Gauss points.
		const arma::Mat<fltp> J = shape_function_jacobian(Rn, dN);
		// Compute the Jacobian determinants (length scaling factors).
		const arma::Row<fltp> Jdet = jacobian2determinant(J);

		// Evaluate the shape functions at the Gauss points.
		const arma::Mat<fltp> N = shape_function(Rqg);

		// Create the conversion matrices.
		// For a line element, we have 2 nodes.
		arma::Mat<fltp>::fixed<2, 2> M(arma::fill::zeros);
		// P has size 2 x (number of Gauss points).
		arma::Mat<fltp> P(2, Rqg.n_cols);
		for (arma::uword i = 0; i < 2; i++) {
			for (arma::uword k = 0; k < Rqg.n_cols; k++) {
				for (arma::uword j = 0; j < 2; j++) {
					M(i, j) += N(i, k) * N(j, k) * Jdet(k) * wg(k);
				}
				// Note: In standard 1D reference coordinates (typically [-1, 1]) the length is 2.
				// Adjust the factor below if needed for your specific integration scheme.
				P(i, k) = N(i, k) * Jdet(k) * wg(k);  
			}
		}

		// Regularize the conversion matrix M to avoid nearly singular systems.
		const fltp condM = arma::cond(M);
		const fltp eps_factor = (condM > static_cast<fltp>(1e8)) ? static_cast<fltp>(1e-3) : static_cast<fltp>(1e-6);
		const fltp epsilon = eps_factor * M.diag().max();
		M.diag() += epsilon;

		// Solve for the nodal values.
		arma::Mat<fltp> Vn(Vg.n_rows, 2, arma::fill::zeros);
		for (arma::uword m = 0; m < Vg.n_rows; m++) {
			arma::Col<fltp> x;
			if (arma::solve(x, M, P * Vg.row(m).t(), arma::solve_opts::force_approx))
				Vn.row(m) = x.t();
		}

		// Return extrapolated nodal values for this line element.
		return Vn;
	}

	// gauss to nodes for mesh of lines
	arma::Mat<fltp> Line::gauss2nodes(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n,
		const arma::Mat<fltp> &Vg,
		const arma::Mat<fltp> &Rqg,
		const arma::Row<fltp> &wg,
		const arma::uword num_nodes){
		// For a line element, ensure the reference dimension is 1 and each element has 2 nodes.
		assert(Rqg.n_rows == 1);
		assert(n.n_rows == 2);
		assert(Vg.n_cols % Rqg.n_cols == 0);
		assert(n.max() < num_nodes);

		// Allocate a matrix to hold the nodal values computed per element.
		// Each element contributes 2 sets of nodal values.
		arma::Mat<fltp> Vnn(Vg.n_rows, 2 * n.n_cols);

		// Loop over each element (the loop can be parallelized).
		parfor(0, n.n_cols, true, [&](arma::uword i, int) {
			Vnn.cols(i * 2, (i + 1) * 2 - 1) =
				gauss2nodes(Rn.cols(n.col(i)), Rqg, wg, Vg.cols(i * Rqg.n_cols, (i + 1) * Rqg.n_cols - 1));
		});

		// Calculate the "volume" of each element.
		// For a line element, this is the length.
		const arma::Row<fltp> elle = calc_length(Rn, n);

		// Allocate output for nodal values over the full mesh.
		arma::Mat<fltp> Vn(Vg.n_rows, num_nodes, arma::fill::zeros);
		// A counter for the total weight (length) accumulated at each node.
		arma::Row<fltp> cnt(1, num_nodes, arma::fill::zeros);

		// Loop over each element to assemble the contributions.
		for (arma::uword i = 0; i < n.n_cols; i++) {
			Vn.cols(n.col(i)) += Vnn.cols(i * 2, (i + 1) * 2 - 1) * elle(i);
			cnt(n.col(i)) += elle(i);
		}

		// Compute the weighted average for each node and return the result.
		return Vn.each_row() / cnt;
	}




	// TRIANGLE
	// triangle edges connectivity
	Triangle::EdgeMatrix Triangle::get_edges(){
		// setup matrix
		arma::Mat<arma::uword>::fixed<3,2> M; 
		M.col(0) = arma::Col<arma::uword>::fixed<3>{0,1,2};
		M.col(1) = arma::Col<arma::uword>::fixed<3>{1,2,0};

		// return matrix
		return M;
	}

	// triangle corner node positions 
	Triangle::CornerNodeMatrix Triangle::get_corner_nodes(){
		// setup matrix
		arma::Mat<arma::sword>::fixed<3,2> M; 
		M.col(0) = arma::Col<arma::sword>::fixed<3>{0,1,0};
		M.col(1) = arma::Col<arma::sword>::fixed<3>{0,0,1};

		// return matrix
		return M;
	}

	// function for calculating face normals
	arma::Mat<fltp> Triangle::calc_face_normals(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n){

		// check input
		assert(Rn.n_rows==3); assert(n.n_rows==3);

		// extract vectors that span the face from opposing nodes
		const arma::Mat<fltp> V0 = Rn.cols(n.row(1)) - Rn.cols(n.row(0));
		const arma::Mat<fltp> V1 = Rn.cols(n.row(2)) - Rn.cols(n.row(0));

		// cross product
		arma::Mat<fltp> Nf = Extra::cross(V0,V1);

		// check vector length
		assert(arma::all(Extra::vec_norm(Nf)>0));

		// normalize
		Nf.each_row() /= Extra::vec_norm(Nf);

		// return face normals
		return Nf; 
	}

	// calculate element areas
	arma::Row<fltp> Triangle::calc_area(
		const arma::Mat<fltp> &Rn,
		const arma::Mat<arma::uword> &n){

		// check mesh
		assert(n.n_rows==3 || n.n_rows==2);
		assert(n.max()<Rn.n_cols);

		// extract vectors that span the face from opposing nodes
		const arma::Mat<fltp> V0 = Rn.cols(n.row(1)) - Rn.cols(n.row(0));
		const arma::Mat<fltp> V1 = Rn.cols(n.row(2)) - Rn.cols(n.row(0));

		// cross product
		const arma::Mat<fltp> V = Rn.n_rows==3 ? Extra::cross(V0,V1) : Extra::cross2D(V0,V1);
		
		// calculate face area using two triangles
		const arma::Row<fltp> A = Extra::vec_norm(V)/2;

		// return element volumes
		return A;
	}

	// calculate face normals at nodes
	arma::Mat<fltp> Triangle::calc_node_normals(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<arma::uword> &n){

		// check mesh
		assert(n.max()<Rn.n_cols);
		assert(n.n_rows==3);

		// calculate face normals at elements
		const arma::Mat<fltp> Nf = calc_face_normals(Rn,n);
		const arma::Row<fltp> A = calc_area(Rn,n);

		// check face normals
		assert(!Nf.has_nan());

		// interpolate at nodes and average
		arma::Mat<rat::fltp> Nn(3,Rn.n_cols,arma::fill::zeros);
		
		// add to each connected node
		for(arma::uword i=0;i<n.n_rows;i++)Nn.cols(n.row(i)) += Nf.each_row()%A;

		// re-normalize
		Nn.each_row() /= rat::cmn::Extra::vec_norm(Nn);

		// return node normals
		return Nn;
	}


	// find edges
	arma::Mat<arma::uword> Triangle::find_edges(
		const arma::Mat<fltp> &Rn, const arma::Mat<arma::uword> &n, 
		const fltp angle_treshold){

		// check input
		assert(Rn.n_rows==3); assert(n.n_rows==3);

		// get face normals
		const arma::Mat<fltp> Nf = Triangle::calc_face_normals(Rn,n);

		// find all unique edge elements
		arma::Mat<arma::uword>::fixed<3,2> Me = Triangle::get_edges();
		arma::Mat<arma::uword> edge_nodes(2,n.n_cols*Me.n_rows);
		arma::Row<arma::uword> edge_faces(n.n_cols*Me.n_rows);
		arma::Row<arma::uword> edge_index(n.n_cols*Me.n_rows);
		for(arma::uword i=0;i<Me.n_rows;i++){
			edge_nodes.cols(i*n.n_cols,(i+1)*n.n_cols-1) = n.rows(Me.row(i));
			edge_faces.cols(i*n.n_cols,(i+1)*n.n_cols-1) = arma::regspace<arma::Row<arma::uword> >(0,n.n_cols-1);
			edge_index.cols(i*n.n_cols,(i+1)*n.n_cols-1).fill(i);
		}

		// sort edges to ensure same direction
		edge_nodes = arma::sort(edge_nodes,"ascend",0);

		// sort columns
		for(arma::uword i=0;i<Me.n_cols;i++){
			const arma::Col<arma::uword> sorting_index = 
				arma::stable_sort_index(edge_nodes.row(i));
			edge_nodes = edge_nodes.cols(sorting_index);
			edge_faces = edge_faces.cols(sorting_index);
			edge_index = edge_index.cols(sorting_index);
		}

		const arma::Row<arma::uword> is_duplicate = 
			arma::all(edge_nodes.head_cols(edge_nodes.n_cols-1) == 
			edge_nodes.tail_cols(edge_nodes.n_cols-1),0);

		// index
		const arma::Col<arma::uword> duplicate_indices = arma::find(is_duplicate);

		// find edges that have no duplicate
		arma::Row<arma::uword> idx_true_edge(edge_nodes.n_cols,arma::fill::ones);
		idx_true_edge(duplicate_indices).fill(0); idx_true_edge(duplicate_indices+1).fill(0);

		// allocate edges
		arma::Col<arma::uword> edge_indices;

		// check if angle set
		if(angle_treshold!=RAT_CONST(0.0)){
			// get adjacent face normals
			const arma::Mat<rat::fltp> N1 = Nf.cols(edge_faces(duplicate_indices));
			const arma::Mat<rat::fltp> N2 = Nf.cols(edge_faces(duplicate_indices+1));

			// calculate angles
			arma::Row<rat::fltp> angle = arma::acos(
				arma::clamp(rat::cmn::Extra::dot(N1,N2)/
				(rat::cmn::Extra::vec_norm(N1)%
					rat::cmn::Extra::vec_norm(N2)),
				RAT_CONST(-1.0),RAT_CONST(1.0)) );
			
			// find edges with angle above treshold
			// combine angled edges and true edges
			edge_indices = arma::join_vert(duplicate_indices(
				arma::find(angle>=angle_treshold)),arma::find(idx_true_edge));
		}
		
		// only true edges
		else{
			edge_indices = arma::find(idx_true_edge);
		}	

		// return edges
		return edge_nodes.cols(edge_indices);
	}

	// split mesh on edges
	arma::Row<arma::uword> Triangle::separate_smooth_surfaces(
		arma::Mat<fltp> &Rn, arma::Mat<arma::uword> &n, const fltp angle_treshold){

		// check input
		assert(n.n_rows==3); assert(Rn.n_rows==3); assert(angle_treshold>0);

		// get face normals
		const arma::Mat<fltp> Nf = Triangle::calc_face_normals(Rn,n);

		// find all unique edge elements
		arma::Mat<arma::uword>::fixed<3,2> Me = Triangle::get_edges();
		arma::Mat<arma::uword> edge_nodes(2,n.n_cols*Me.n_rows);
		arma::Row<arma::uword> edge_faces(n.n_cols*Me.n_rows);
		arma::Row<arma::uword> edge_index(n.n_cols*Me.n_rows);
		for(arma::uword i=0;i<Me.n_rows;i++){
			edge_nodes.cols(i*n.n_cols,(i+1)*n.n_cols-1) = n.rows(Me.row(i));
			edge_faces.cols(i*n.n_cols,(i+1)*n.n_cols-1) = arma::regspace<arma::Row<arma::uword> >(0,n.n_cols-1);
			edge_index.cols(i*n.n_cols,(i+1)*n.n_cols-1).fill(i);
		}

		// sort edges to ensure same direction
		edge_nodes = arma::sort(edge_nodes,"ascend",0);

		// sort columns
		for(arma::uword i=0;i<Me.n_cols;i++){
			const arma::Col<arma::uword> sorting_index = 
				arma::stable_sort_index(edge_nodes.row(i));
			edge_nodes = edge_nodes.cols(sorting_index);
			edge_faces = edge_faces.cols(sorting_index);
			edge_index = edge_index.cols(sorting_index);
		}

		// index
		const arma::Col<arma::uword> duplicate_indices = arma::find(
			arma::all(edge_nodes.head_cols(edge_nodes.n_cols-1) == 
			edge_nodes.tail_cols(edge_nodes.n_cols-1),0));

		// get adjacent face normals
		const arma::Mat<rat::fltp> N1 = Nf.cols(edge_faces(duplicate_indices));
		const arma::Mat<rat::fltp> N2 = Nf.cols(edge_faces(duplicate_indices+1));

		// calculate angles
		arma::Row<rat::fltp> angle = arma::acos(
			arma::clamp(rat::cmn::Extra::dot(N1,N2)/
			(rat::cmn::Extra::vec_norm(N1)%
				rat::cmn::Extra::vec_norm(N2)),
			RAT_CONST(-1.0),RAT_CONST(1.0)) );

		// const arma::Row<rat::fltp> x = 
		// 	arma::clamp(rat::cmn::Extra::dot(N1,N2)/
		// 	(rat::cmn::Extra::vec_norm(N1)%
		// 		rat::cmn::Extra::vec_norm(N2)),
		// 	RAT_CONST(-1.0),RAT_CONST(1.0));
		// const arma::Row<rat::fltp> angle = (-RAT_CONST(0.69813170079773212)*x%x-RAT_CONST(0.87266462599716477))%x + RAT_CONST(1.5707963267948966);

		// find edges with angle above treshold
		// rat::fltp edge_tresh = RAT_CONST(20.0)*2*arma::Datum<rat::fltp>::pi/RAT_CONST(360.0);
		const arma::Col<arma::uword> edge_indices = 
			duplicate_indices(arma::find(angle<angle_treshold));

		// get the adjacent faces on the dges
		const arma::Row<arma::uword> face1 = edge_faces.cols(edge_indices);
		const arma::Row<arma::uword> face2 = edge_faces.cols(edge_indices+1);
		const arma::Row<arma::uword> edge1 = edge_index.cols(edge_indices);
		const arma::Row<arma::uword> edge2 = edge_index.cols(edge_indices+1);

		// keep track of the original node index
		// arma::Mat<arma::uword> original_index(4,Rn.n_cols);
		// original_index.each_row() = arma::regspace<arma::Row<arma::uword> >(0,Rn.n_cols-1);
		arma::Row<arma::uword> original_index = arma::vectorise(n).t();

		// split all triangles
		Rn = Rn.cols(arma::vectorise(n));
		n = arma::reshape(arma::regspace<arma::Col<arma::uword> >(0,n.n_elem-1),3,n.n_cols);

		// merge nodes
		for(;;){
			bool change = false;
			for(arma::uword i=0;i<face1.n_elem;i++){
				for(arma::uword j=0;j<Me.n_cols;j++){
					if(n(Me(edge1(i),j),face1(i)) != n(Me(edge2(i),1-j),face2(i))){
						// find smallest index
						const arma::uword idx = std::min(n(Me(edge1(i),j),face1(i)), n(Me(edge2(i),1-j),face2(i)));
							
						// set index to both nodes
						n(Me(edge1(i),j),face1(i)) = idx; n(Me(edge2(i),1-j),face2(i)) = idx; 

						// flag change
						change = true;
					}
				}
			}
			if(!change)break;
		}

		// count node references
		arma::Row<arma::uword> cnt(Rn.n_cols,arma::fill::zeros);
		for(arma::uword i=0;i<n.n_elem;i++)cnt(n(i))++;
		const arma::Row<arma::uword> active_nodes = 
			arma::find(cnt>0).t();

		// re-index
		arma::Row<arma::uword> re_index(Rn.n_cols, arma::fill::zeros);
		re_index.cols(active_nodes) = arma::regspace<arma::Row<arma::uword> >(0,active_nodes.n_elem-1);

		// remove unused nodes
		n = arma::reshape(re_index(arma::vectorise(n)),3,n.n_cols); 
		Rn = Rn.cols(active_nodes);

		original_index = original_index.cols(active_nodes);

		// return the original index
		return original_index;
	}

	// shape function
	arma::Mat<fltp> Triangle::shape_function(const arma::Mat<fltp> &Rq){
		// check input
		assert(Rq.n_rows==2);

		// calculate shape functions
		const arma::Mat<fltp> Nie = arma::join_vert(RAT_CONST(1.0) - arma::sum(Rq,0), Rq);

		// return 
		return Nie;
	}


	// quadrilateral coordinates to carthesian coordinates
	arma::Mat<fltp> Triangle::quad2cart(
		const arma::Mat<fltp> &Rn, 
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rn.n_cols==3); assert(Rq.n_rows==2);
		
		// get shape functions
		const arma::Mat<fltp> Nie = Triangle::shape_function(Rq);

		// get coordinates from matrix vector product
		const arma::Mat<fltp> Rc = Rn*Nie;

		// return cartesian coords
		return Rc;
	}

	// partial derivatives of hexahedron shape function in quadrilateral coordinates
	// format: [dNdnu;dNdmu;dNdxi]
	arma::Mat<fltp> Triangle::shape_function_derivative(
		const arma::Mat<fltp> &Rq){
		
		// check input
		assert(Rq.n_rows==2);

		// calculate shape functions
		arma::Mat<fltp> dNie(6,Rq.n_cols);
		dNie.each_col() = arma::Col<fltp>::fixed<6>{
			-RAT_CONST(1.0),RAT_CONST(1.0),RAT_CONST(0.0), 
			-RAT_CONST(1.0),RAT_CONST(0.0),RAT_CONST(1.0)};

		// return 
		return dNie;
	}

	// convert volume to in-plane
	arma::Mat<fltp>::fixed<3,2> Triangle::surface_to_volume_matrix(const arma::Mat<fltp>&Rn){
		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==3);

		// need to work in plane
		arma::Col<fltp>::fixed<3> V1 = Rn.col(1) - Rn.col(0);
		arma::Col<fltp>::fixed<3> V2 = Rn.col(2) - Rn.col(0);
		V1.each_row()/=Extra::vec_norm(V1);
		V2.each_row()/=Extra::vec_norm(V2);
		arma::Col<fltp>::fixed<3> V3 = Extra::cross(Extra::cross(V1,V2),V1);
		V3.each_row()/=Extra::vec_norm(V3);

		// combine vectors into conversion matrix
		const arma::Mat<fltp>::fixed<3,2> M = arma::join_horiz(V1,V3);

		// return the martix
		return M;
	}

	// jacobian 
	arma::Mat<fltp> Triangle::shape_function_jacobian(
		const arma::Mat<fltp>&Rn,
		const arma::Mat<fltp>&dN){
		// check input
		assert(Rn.n_rows==3); assert(Rn.n_cols==3);

		// create conversion matrix
		const arma::Mat<fltp>::fixed<3,2> M = surface_to_volume_matrix(Rn);

		// calculate cartesian coordinates in plane
		const arma::Row<fltp> u = M.col(0).t()*Rn;
		const arma::Row<fltp> v = M.col(1).t()*Rn; 

		// get shape derivative of shape function
		const arma::Mat<fltp> dNiedxi = dN.rows(0,2); 
		const arma::Mat<fltp> dNiednu = dN.rows(3,5); 

		// setup jacobian
		arma::Mat<fltp> J(4,dN.n_cols);
		J.row(0) = u*dNiedxi;
		J.row(1) = u*dNiednu;
		J.row(2) = v*dNiedxi;
		J.row(3) = v*dNiednu;

		// return jacobian
		return J;
	}

	// method for inverting jacobian matrix
	arma::Mat<fltp> Triangle::invert_jacobian(
		const arma::Mat<fltp> &J,
		const arma::Row<fltp> &Jdet){ 

		// same as hexahedron
		return Quadrilateral::invert_jacobian(J,Jdet);
	}

	// create gauss points
	arma::Mat<fltp> Triangle::create_gauss_points(const arma::sword num_gauss){
		// create gauss points
		arma::Mat<fltp> Rqg;

		// CENTROID, the centroid rule, order 1, degree of precision 1. 
		if(std::abs(num_gauss)<=1){
			Rqg.set_size(3,1);
			Rqg.col(0) = arma::Col<fltp>::fixed<3>{1.0/3,1.0/3, 1.0/2};
		}

		// STRANG1, order 3, degree of precision 2. 
		else if(std::abs(num_gauss)<=2){
			Rqg.set_size(3, 3);
			Rqg.col(0) = arma::Col<fltp>::fixed<3>{2.0/3, 1.0/6, 1.0/6};
			Rqg.col(1) = arma::Col<fltp>::fixed<3>{1.0/6, 2.0/3, 1.0/6};
			Rqg.col(2) = arma::Col<fltp>::fixed<3>{1.0/6, 1.0/6, 1.0/6};
		}

		// STRANG3, order 4, degree of precision 3. 
		else if(num_gauss==3){
			Rqg.set_size(3,4);
			Rqg.col(0) = arma::Col<fltp>::fixed<3>{1.0/3,1.0/3, -27.0/96};
			Rqg.col(1) = arma::Col<fltp>::fixed<3>{1.0/5,1.0/5, 25.0/96};
			Rqg.col(2) = arma::Col<fltp>::fixed<3>{3.0/5,1.0/5, 25.0/96};
			Rqg.col(3) = arma::Col<fltp>::fixed<3>{1.0/5,3.0/5, 25.0/96};
		}

		// STRANG5, order 6, degree of precision 4. 
		else if(num_gauss==4){
			Rqg.set_size(3,6);
			Rqg.col(0) = arma::Col<fltp>::fixed<3>{0.816847572980459,0.091576213509771, 0.109951743655322};
			Rqg.col(1) = arma::Col<fltp>::fixed<3>{0.091576213509771,0.816847572980459, 0.109951743655322};
			Rqg.col(2) = arma::Col<fltp>::fixed<3>{0.091576213509771,0.091576213509771, 0.109951743655322};
			Rqg.col(3) = arma::Col<fltp>::fixed<3>{0.108103018168070,0.445948490915965, 0.223381589678011};
			Rqg.col(4) = arma::Col<fltp>::fixed<3>{0.445948490915965,0.108103018168070, 0.223381589678011};
			Rqg.col(5) = arma::Col<fltp>::fixed<3>{0.445948490915965,0.445948490915965, 0.223381589678011};
			Rqg.row(2)*=0.5;
		}

		// STRANG7, order 7, degree of precision 5. 
		else if(num_gauss==5){
			Rqg.set_size(3,7);
			Rqg.col(0) = arma::Col<fltp>::fixed<3>{0.33333333333333333,0.33333333333333333,0.22500000000000000};
			Rqg.col(1) = arma::Col<fltp>::fixed<3>{0.79742698535308720,0.10128650732345633,0.12593918054482717};
			Rqg.col(2) = arma::Col<fltp>::fixed<3>{0.10128650732345633,0.79742698535308720,0.12593918054482717};
			Rqg.col(3) = arma::Col<fltp>::fixed<3>{0.10128650732345633,0.10128650732345633,0.12593918054482717};
			Rqg.col(4) = arma::Col<fltp>::fixed<3>{0.05971587178976981,0.47014206410511505,0.13239415278850616};
			Rqg.col(5) = arma::Col<fltp>::fixed<3>{0.47014206410511505,0.05971587178976981,0.13239415278850616};
			Rqg.col(6) = arma::Col<fltp>::fixed<3>{0.47014206410511505,0.47014206410511505,0.13239415278850616};
			Rqg.row(2)*=0.5;
		}

		// STRANG8, order 9, degree of precision 6. 
		else if(num_gauss==6){
			Rqg.set_size(3,9);
			Rqg.col(0) = arma::Col<fltp>::fixed<3>{0.124949503233232,0.437525248383384,0.205950504760887};
			Rqg.col(1) = arma::Col<fltp>::fixed<3>{0.437525248383384,0.124949503233232,0.205950504760887};
			Rqg.col(2) = arma::Col<fltp>::fixed<3>{0.437525248383384,0.437525248383384,0.205950504760887};
			Rqg.col(3) = arma::Col<fltp>::fixed<3>{0.797112651860071,0.165409927389841,0.063691414286223};
			Rqg.col(4) = arma::Col<fltp>::fixed<3>{0.797112651860071,0.037477420750088,0.063691414286223};
			Rqg.col(5) = arma::Col<fltp>::fixed<3>{0.165409927389841,0.797112651860071,0.063691414286223};
			Rqg.col(6) = arma::Col<fltp>::fixed<3>{0.165409927389841,0.037477420750088,0.063691414286223};
			Rqg.col(7) = arma::Col<fltp>::fixed<3>{0.037477420750088,0.797112651860071,0.063691414286223};
			Rqg.col(8) = arma::Col<fltp>::fixed<3>{0.037477420750088,0.165409927389841,0.063691414286223};
			Rqg.row(2)*=0.5;
		}

		// STRANG10, order 13, degree of precision 7. 
		else if(num_gauss==7){
			Rqg.set_size(3,13);
			Rqg.col(0) = arma::Col<fltp>::fixed<3>{0.333333333333333,0.333333333333333,-0.149570044467670};
			Rqg.col(1) = arma::Col<fltp>::fixed<3>{0.479308067841923,0.260345966079038,0.175615257433204};
			Rqg.col(2) = arma::Col<fltp>::fixed<3>{0.260345966079038,0.479308067841923,0.175615257433204};
			Rqg.col(3) = arma::Col<fltp>::fixed<3>{0.260345966079038,0.260345966079038,0.175615257433204};
			Rqg.col(4) = arma::Col<fltp>::fixed<3>{0.869739794195568,0.065130102902216,0.053347235608839};
			Rqg.col(5) = arma::Col<fltp>::fixed<3>{0.065130102902216,0.869739794195568,0.053347235608839};
			Rqg.col(6) = arma::Col<fltp>::fixed<3>{0.065130102902216,0.065130102902216,0.053347235608839};
			Rqg.col(7) = arma::Col<fltp>::fixed<3>{0.638444188569809,0.312865496004875,0.077113760890257};
			Rqg.col(8) = arma::Col<fltp>::fixed<3>{0.638444188569809,0.048690315425316,0.077113760890257};
			Rqg.col(9) = arma::Col<fltp>::fixed<3>{0.312865496004875,0.638444188569809,0.077113760890257};
			Rqg.col(10) = arma::Col<fltp>::fixed<3>{0.312865496004875,0.048690315425316,0.077113760890257};
			Rqg.col(11) = arma::Col<fltp>::fixed<3>{0.048690315425316,0.638444188569809,0.077113760890257};
			Rqg.col(12) = arma::Col<fltp>::fixed<3>{0.048690315425316,0.312865496004875,0.077113760890257};
			Rqg.row(2)*=0.5;
		}

		// TOMS706_37, order 37, degree of precision 13, a rule from ACM TOMS algorithm #706. 
		else if(num_gauss>7){
			Rqg.set_size(3,37);
			Rqg.col(0) = arma::Col<fltp>::fixed<3>{0.333333333333333333333333333333,0.333333333333333333333333333333,0.051739766065744133555179145422};
			Rqg.col(1) = arma::Col<fltp>::fixed<3>{0.950275662924105565450352089520,0.024862168537947217274823955239,0.008007799555564801597804123460};
			Rqg.col(2) = arma::Col<fltp>::fixed<3>{0.024862168537947217274823955239,0.950275662924105565450352089520,0.008007799555564801597804123460};
			Rqg.col(3) = arma::Col<fltp>::fixed<3>{0.024862168537947217274823955239,0.024862168537947217274823955239,0.008007799555564801597804123460};
			Rqg.col(4) = arma::Col<fltp>::fixed<3>{0.171614914923835347556304795551,0.414192542538082326221847602214,0.046868898981821644823226732071};
			Rqg.col(5) = arma::Col<fltp>::fixed<3>{0.414192542538082326221847602214,0.171614914923835347556304795551,0.046868898981821644823226732071};
			Rqg.col(6) = arma::Col<fltp>::fixed<3>{0.414192542538082326221847602214,0.414192542538082326221847602214,0.046868898981821644823226732071};
			Rqg.col(7) = arma::Col<fltp>::fixed<3>{0.539412243677190440263092985511,0.230293878161404779868453507244,0.046590940183976487960361770070};
			Rqg.col(8) = arma::Col<fltp>::fixed<3>{0.230293878161404779868453507244,0.539412243677190440263092985511,0.046590940183976487960361770070};
			Rqg.col(9) = arma::Col<fltp>::fixed<3>{0.230293878161404779868453507244,0.230293878161404779868453507244,0.046590940183976487960361770070};
			Rqg.col(10) = arma::Col<fltp>::fixed<3>{0.772160036676532561750285570113,0.113919981661733719124857214943,0.031016943313796381407646220131};
			Rqg.col(11) = arma::Col<fltp>::fixed<3>{0.113919981661733719124857214943,0.772160036676532561750285570113,0.031016943313796381407646220131};
			Rqg.col(12) = arma::Col<fltp>::fixed<3>{0.113919981661733719124857214943,0.113919981661733719124857214943,0.031016943313796381407646220131};
			Rqg.col(13) = arma::Col<fltp>::fixed<3>{0.009085399949835353883572964740,0.495457300025082323058213517632,0.010791612736631273623178240136};
			Rqg.col(14) = arma::Col<fltp>::fixed<3>{0.495457300025082323058213517632,0.009085399949835353883572964740,0.010791612736631273623178240136};
			Rqg.col(15) = arma::Col<fltp>::fixed<3>{0.495457300025082323058213517632,0.495457300025082323058213517632,0.010791612736631273623178240136};
			Rqg.col(16) = arma::Col<fltp>::fixed<3>{0.062277290305886993497083640527,0.468861354847056503251458179727,0.032195534242431618819414482205};
			Rqg.col(17) = arma::Col<fltp>::fixed<3>{0.468861354847056503251458179727,0.062277290305886993497083640527,0.032195534242431618819414482205};
			Rqg.col(18) = arma::Col<fltp>::fixed<3>{0.468861354847056503251458179727,0.468861354847056503251458179727,0.032195534242431618819414482205};
			Rqg.col(19) = arma::Col<fltp>::fixed<3>{0.022076289653624405142446876931,0.851306504174348550389457672223,0.015445834210701583817692900053};
			Rqg.col(20) = arma::Col<fltp>::fixed<3>{0.022076289653624405142446876931,0.126617206172027096933163647918,0.015445834210701583817692900053};
			Rqg.col(21) = arma::Col<fltp>::fixed<3>{0.851306504174348550389457672223,0.022076289653624405142446876931,0.015445834210701583817692900053};
			Rqg.col(22) = arma::Col<fltp>::fixed<3>{0.851306504174348550389457672223,0.126617206172027096933163647918,0.015445834210701583817692900053};
			Rqg.col(23) = arma::Col<fltp>::fixed<3>{0.126617206172027096933163647918,0.022076289653624405142446876931,0.015445834210701583817692900053};
			Rqg.col(24) = arma::Col<fltp>::fixed<3>{0.126617206172027096933163647918,0.851306504174348550389457672223,0.015445834210701583817692900053};
			Rqg.col(25) = arma::Col<fltp>::fixed<3>{0.018620522802520968955913511549,0.689441970728591295496647976487,0.017822989923178661888748319485};
			Rqg.col(26) = arma::Col<fltp>::fixed<3>{0.018620522802520968955913511549,0.291937506468887771754472382212,0.017822989923178661888748319485};
			Rqg.col(27) = arma::Col<fltp>::fixed<3>{0.689441970728591295496647976487,0.018620522802520968955913511549,0.017822989923178661888748319485};
			Rqg.col(28) = arma::Col<fltp>::fixed<3>{0.689441970728591295496647976487,0.291937506468887771754472382212,0.017822989923178661888748319485};
			Rqg.col(29) = arma::Col<fltp>::fixed<3>{0.291937506468887771754472382212,0.018620522802520968955913511549,0.017822989923178661888748319485};
			Rqg.col(30) = arma::Col<fltp>::fixed<3>{0.291937506468887771754472382212,0.689441970728591295496647976487,0.017822989923178661888748319485};
			Rqg.col(31) = arma::Col<fltp>::fixed<3>{0.096506481292159228736516560903,0.635867859433872768286976979827,0.037038683681384627918546472190};
			Rqg.col(32) = arma::Col<fltp>::fixed<3>{0.096506481292159228736516560903,0.267625659273967961282458816185,0.037038683681384627918546472190};
			Rqg.col(33) = arma::Col<fltp>::fixed<3>{0.635867859433872768286976979827,0.096506481292159228736516560903,0.037038683681384627918546472190};
			Rqg.col(34) = arma::Col<fltp>::fixed<3>{0.635867859433872768286976979827,0.267625659273967961282458816185,0.037038683681384627918546472190};
			Rqg.col(35) = arma::Col<fltp>::fixed<3>{0.267625659273967961282458816185,0.096506481292159228736516560903,0.037038683681384627918546472190};
			Rqg.col(36) = arma::Col<fltp>::fixed<3>{0.267625659273967961282458816185,0.635867859433872768286976979827,0.037038683681384627918546472190};
			Rqg.row(2)*=0.5;
		}

		// // quadrature from https://zhilin.math.ncsu.edu/TEACHING/MA587/Gaussian_Quadrature2D.pdf -> GQUTM
		// else if(std::abs(num_gauss)<=5){
		// 	Rqg.set_size(3,14);
		// 	Rqg.row(0) = arma::Row<fltp>::fixed<14>{6.943184420297371e-02,6.943184420297371e-02,6.943184420297371e-02,6.943184420297371e-02,6.943184420297371e-02,3.300094782075720e-01,3.300094782075720e-01,3.300094782075720e-01,3.300094782075720e-01,6.699905217924280e-01,6.699905217924280e-01,6.699905217924280e-01,9.305681557970260e-01,9.305681557970260e-01};
		// 	Rqg.row(1) = arma::Row<fltp>::fixed<14>{4.365302387072518e-02,2.147428814693420e-01,4.652840778985130e-01,7.158252743276839e-01,8.869151319263010e-01,4.651867752656094e-02,2.211032225007380e-01,4.488872992916900e-01,6.234718442658670e-01,3.719261778493340e-02,1.650047391037860e-01,2.928168604226380e-01,1.467267513102734e-02,5.475916907194637e-02};
		// 	Rqg.row(2) = arma::Row<fltp>::fixed<14>{1.917346464706755e-02,3.873334126144628e-02,4.603770904527855e-02,3.873334126144628e-02,1.917346464706755e-02,3.799714764789616e-02,7.123562049953998e-02,7.123562049953998e-02,3.799714764789616e-02,2.989084475992800e-02,4.782535161588505e-02,2.989084475992800e-02,6.038050853208200e-03,6.038050853208200e-03};
		// }

		// else{ // this is for 9
		// 	Rqg.set_size(3,44);
		// 	Rqg.row(0) = arma::Row<fltp>::fixed<44>{1.985507175123191e-02,1.985507175123191e-02,1.985507175123191e-02,1.985507175123191e-02,1.985507175123191e-02,1.985507175123191e-02,1.985507175123191e-02,1.985507175123191e-02,1.985507175123191e-02,1.016667612931870e-01,1.016667612931870e-01,1.016667612931870e-01,1.016667612931870e-01,1.016667612931870e-01,1.016667612931870e-01,1.016667612931870e-01,1.016667612931870e-01,2.372337950418360e-01,2.372337950418360e-01,2.372337950418360e-01,2.372337950418360e-01,2.372337950418360e-01,2.372337950418360e-01,2.372337950418360e-01,4.082826787521750e-01,4.082826787521750e-01,4.082826787521750e-01,4.082826787521750e-01,4.082826787521750e-01,4.082826787521750e-01,5.917173212478249e-01,5.917173212478249e-01,5.917173212478249e-01,5.917173212478249e-01,5.917173212478249e-01,7.627662049581641e-01,7.627662049581641e-01,7.627662049581641e-01,7.627662049581641e-01,8.983332387068130e-01,8.983332387068130e-01,8.983332387068130e-01,9.801449282487680e-01,9.801449282487680e-01};
		// 	Rqg.row(1) = arma::Row<fltp>::fixed<44>{1.560378988162790e-02,8.035663927218221e-02,1.894760146773020e-01,3.311647899161120e-01,4.900724641243840e-01,6.489801383326560e-01,7.906689135714660e-01,8.997882889765860e-01,9.645411383671399e-01,1.783647091104033e-02,9.133063094134081e-02,2.131150034306400e-01,3.667739011113350e-01,5.315593375954780e-01,6.852182352761730e-01,8.070026077654729e-01,8.804967677957730e-01,1.940938228235618e-02,9.857563833019303e-02,2.266006195206780e-01,3.813831024790820e-01,5.361655854374870e-01,6.641905666279710e-01,7.433568226758080e-01,1.997947907913758e-02,1.002341371520440e-01,2.252611078301700e-01,3.664562134176550e-01,4.914831840957800e-01,5.717378421686869e-01,1.915257191055202e-02,9.421749319819557e-02,2.041413393760880e-01,3.140651855539790e-01,3.891301068416230e-01,1.647157989702492e-02,7.828940091495819e-02,1.589443941268770e-01,2.207622151448110e-01,1.145801331145764e-02,5.083338064659329e-02,9.020874798172894e-02,4.195870365439417e-03,1.565920138579250e-02};
		// 	Rqg.row(2) = arma::Row<fltp>::fixed<44>{2.015983497663207e-03,4.480916044841641e-03,6.464359484621604e-03,7.747662769908149e-03,8.191474625434276e-03,7.747662769908149e-03,6.464359484621604e-03,4.480916044841641e-03,2.015983497663207e-03,5.055663745070170e-03,1.110639128725685e-02,1.566747257514398e-02,1.811354111938598e-02,1.811354111938598e-02,1.566747257514398e-02,1.110639128725685e-02,5.055663745070170e-03,7.745946956361961e-03,1.673231410555364e-02,2.284153446586376e-02,2.500282281756943e-02,2.284153446586376e-02,1.673231410555364e-02,7.745946956361961e-03,9.191827856850984e-03,1.935542449754594e-02,2.510431683577024e-02,2.510431683577024e-02,1.935542449754594e-02,9.191827856850984e-03,8.770885597453929e-03,1.771853503082167e-02,2.105991205229386e-02,1.771853503082167e-02,8.770885597453929e-03,6.471997505236908e-03,1.213345702759751e-02,1.213345702759751e-02,6.471997505236908e-03,3.140105492486528e-03,5.024168787978471e-03,3.140105492486528e-03,5.024749628293684e-04,5.024749628293684e-04};
		// }

		else{
			rat_throw_line("this number of gauss points was not implemented");
		}

		assert(std::abs(arma::accu(Rqg.row(2))-0.5)<1e-12);

		// return gauss points and weights
		return Rqg;
	}

	// create gauss points by subdividing the triangle into 3 quadrilaterals
	// this subdivision is known as catmull clark
	arma::Mat<fltp> Triangle::create_gauss_points_catmull_clark(const arma::sword num_gauss){
		// get corner points in quadrilateral coordinates
		const arma::Mat<fltp>::fixed<3,3> P = arma::join_vert(
			arma::conv_to<arma::Mat<fltp> >::from(get_corner_nodes()).t(),
			arma::Row<fltp>::fixed<3>(arma::fill::zeros));

		// get center point and midpoint on each edge
		const arma::Col<fltp>::fixed<3> Pmid = arma::mean(P,1);
		const arma::Col<fltp>::fixed<3> P01 = (P.col(0) + P.col(1))/2;
		const arma::Col<fltp>::fixed<3> P12 = (P.col(1) + P.col(2))/2;
		const arma::Col<fltp>::fixed<3> P20 = (P.col(2) + P.col(0))/2;

		// catmull clark subdivision into 3 quadrilaterals
		const arma::Mat<fltp>::fixed<3,4> Q1 = arma::join_horiz(P.col(0),P01,Pmid,P20);
		const arma::Mat<fltp>::fixed<3,4> Q2 = arma::join_horiz(P.col(1),P12,Pmid,P01);
		const arma::Mat<fltp>::fixed<3,4> Q3 = arma::join_horiz(P.col(2),P20,Pmid,P12);
		
		// create gauss points for quadrilateral
		const arma::Mat<fltp> gpq = Quadrilateral::create_gauss_points(num_gauss);
		const arma::Mat<fltp> Rq = gpq.rows(0,1);
		const arma::Row<fltp> wq = gpq.row(2);
		const arma::Mat<fltp> dN = Quadrilateral::shape_function_derivative(Rq);

		// map to triangle
		const arma::Mat<fltp> gp1 = Quadrilateral::quad2cart(Q1,Rq);
		const arma::Mat<fltp> gp2 = Quadrilateral::quad2cart(Q2,Rq);
		const arma::Mat<fltp> gp3 = Quadrilateral::quad2cart(Q3,Rq);

		// jacobian 
		const arma::Row<fltp> w1 = wq%Quadrilateral::jacobian2determinant(Quadrilateral::shape_function_jacobian(Q1,dN));
		const arma::Row<fltp> w2 = wq%Quadrilateral::jacobian2determinant(Quadrilateral::shape_function_jacobian(Q2,dN));
		const arma::Row<fltp> w3 = wq%Quadrilateral::jacobian2determinant(Quadrilateral::shape_function_jacobian(Q3,dN));

		// join
		const arma::Mat<fltp> gp = arma::join_vert(arma::join_horiz(gp1,gp2,gp3).rows(0,1),arma::join_horiz(w1,w2,w3));

		// return gauss points and weights
		return gp;
	}


	// calculate determinant for the jacobian matrices
	arma::Row<fltp> Triangle::jacobian2determinant(const arma::Mat<fltp>&J){
		// same as quad
		return Quadrilateral::jacobian2determinant(J);
	}

	arma::Mat<arma::uword> Triangle::invert_elements(const arma::Mat<arma::uword>&n){
		return arma::flipud(n);
	}

}}