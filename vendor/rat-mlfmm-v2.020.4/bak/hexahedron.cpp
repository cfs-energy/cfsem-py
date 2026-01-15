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
#include "hexahedron.hh"


// hexahedron corner node positions 
// in quadrilateral coordinates
// matrix columns are given as [xi,nu,mu]
arma::Mat<arma::sword>::fixed<8,3> Hexahedron::get_corner_nodes(){
	// setup matrix
	arma::Mat<arma::sword>::fixed<8,3> M; 
	M.col(0) = arma::Col<arma::sword>{-1,+1,+1,-1,-1,+1,+1,-1};
	M.col(1) = arma::Col<arma::sword>{-1,-1,+1,+1,-1,-1,+1,+1};
	M.col(2) = arma::Col<arma::sword>{-1,-1,-1,-1,+1,+1,+1,+1};

	// return matrix
	return M;
}

// matrix for converting a hexahedron into five tetrahedrons
arma::Mat<arma::uword>::fixed<5,4> Hexahedron::tetrahedron_conversion_matrix(){
	// setup matrix
	arma::Mat<arma::uword>::fixed<5,4> M; 
	M.row(0) = arma::Row<arma::uword>{0,4,5,7}; // first thetahedron
	M.row(1) = arma::Row<arma::uword>{0,1,2,5}; // second thetahedron
	M.row(2) = arma::Row<arma::uword>{0,2,3,7}; // third thetahedron
	M.row(3) = arma::Row<arma::uword>{2,5,6,7}; // fourth thetahedron
	M.row(4) = arma::Row<arma::uword>{0,5,2,7}; // fifth thetahedron (the central larger one)

	// return matrix
	return M;
}

// matrix for converting a hexahedron 
// plus central node into twelve tetrahedrons
arma::Mat<arma::uword>::fixed<12,4> Hexahedron::tetrahedron_conversion_matrix_special(){
	// setup matrix
	arma::Mat<arma::uword>::fixed<12,4> M; 
	M.row(0) = arma::Row<arma::uword>{8,0,1,2}; 
	M.row(1) = arma::Row<arma::uword>{8,0,3,2}; 
	M.row(2) = arma::Row<arma::uword>{8,4,5,6}; 
	M.row(3) = arma::Row<arma::uword>{8,4,7,6}; 
	M.row(4) = arma::Row<arma::uword>{8,2,1,5}; 
	M.row(5) = arma::Row<arma::uword>{8,2,6,5}; 
	M.row(6) = arma::Row<arma::uword>{8,0,1,5};
	M.row(7) = arma::Row<arma::uword>{8,0,4,5};
	M.row(8) = arma::Row<arma::uword>{8,0,3,7}; 
	M.row(9) = arma::Row<arma::uword>{8,0,4,7}; 
	M.row(10)= arma::Row<arma::uword>{8,3,2,6}; 
	M.row(11)= arma::Row<arma::uword>{8,3,7,6}; 

	// return matrix
	return M;
}

// hexahedron face matrix
arma::Mat<arma::uword>::fixed<6,4> Hexahedron::get_faces(){
	// setup matrix
	arma::Mat<arma::uword>::fixed<6,4> M; 

	// faces is counter clockwise order (as seen from outside)
	M.row(0) = arma::Row<arma::uword>{0,1,5,4}; 
	M.row(1) = arma::Row<arma::uword>{1,2,6,5}; 
	M.row(2) = arma::Row<arma::uword>{2,3,7,6}; 
	M.row(3) = arma::Row<arma::uword>{3,0,4,7}; 
	M.row(4) = arma::Row<arma::uword>{0,3,2,1}; 
	M.row(5) = arma::Row<arma::uword>{4,5,6,7}; 

	// return matrix
	return M;
}

// hexahedron edges
arma::Mat<arma::uword>::fixed<12,2> Hexahedron::get_edges(){
	// setup matrix
	arma::Mat<arma::uword>::fixed<12,2> M; 
	M.col(0) = arma::Col<arma::uword>{0,1,2,3,0,1,2,3,4,5,6,7};
	M.col(1) = arma::Col<arma::uword>{1,2,3,0,4,5,6,7,5,6,7,5};

	// return matrix
	return M;
}

// dimensions in quadrilateral coordinates in which the face is located
arma::Mat<arma::sword>::fixed<6,3> Hexahedron::get_facenormal(){
	// setup matrix
	arma::Mat<arma::sword>::fixed<6,3> M; 

	// faces is counter clockwise order (as seen from outside)
	M.row(0) = arma::Row<arma::sword>{0,-1,0}; 
	M.row(1) = arma::Row<arma::sword>{+1,0,0}; 
	M.row(2) = arma::Row<arma::sword>{0,+1,0}; 
	M.row(3) = arma::Row<arma::sword>{-1,0,0}; 
	M.row(4) = arma::Row<arma::sword>{0,0,-1}; 
	M.row(5) = arma::Row<arma::sword>{0,0,+1}; 

	// return matrix
	return M;
}

// dimensions in quadrilateral coordinates in which the face is located
arma::Mat<arma::uword>::fixed<6,3> Hexahedron::get_facedim(){
	// setup matrix
	arma::Mat<arma::uword>::fixed<6,3> M; 

	// dimensions in which the faces are spanned
	M.col(0) = arma::Col<arma::uword>{0,1,0,1,0,0};
	M.col(1) = arma::Col<arma::uword>{2,2,2,2,1,1};
	M.col(2) = arma::Col<arma::uword>{1,0,1,0,2,2}; // fixed

	// return maxtrix
	return M;
}


// dimensions in quadrilateral coordinates in which the face is located
arma::Col<arma::sword>::fixed<6> Hexahedron::get_facedir(){
	// setup matrix
	arma::Col<arma::sword>::fixed<6> M = {-1,+1,+1,-1,-1,+1}; 

	// return matrix
	return M;
}



// hexahedron shape function
arma::Mat<double> Hexahedron::shape_function(
	const arma::Mat<double> &Rq){
	
	// check input
	assert(Rq.n_rows==3);

	// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
	arma::Mat<arma::sword>::fixed<8,3> M = Hexahedron::get_corner_nodes();

	// calculate shape functions
	arma::Mat<double> Nie(8,Rq.n_cols);

	// fill matrix
	for(arma::uword i=0;i<8;i++){
		Nie.row(i) = 
			(1+Rq.row(0)*M(i,0))%
			(1+Rq.row(1)*M(i,1))%
			(1+Rq.row(2)*M(i,2))/8;
	}

	// return 
	return Nie;
}

// partial derivatives of hexahedron shape function
// format: [dNdnu;dNdmu;dNdxi]
arma::Mat<double> Hexahedron::shape_function_derivative(
	const arma::Mat<double> &Rq){
	
	// check input
	assert(Rq.n_rows==3);

	// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
	arma::Mat<arma::sword>::fixed<8,3> M = Hexahedron::get_corner_nodes();

	// calculate shape functions
	arma::Mat<double> dNie(24,Rq.n_cols);

	// fill matrix
	for(arma::uword i=0;i<8;i++){
		dNie.row( 0+i) = M(i,0)*(1+Rq.row(1)*M(i,1))%(1+Rq.row(2)*M(i,2))/8;
		dNie.row( 8+i) = M(i,1)*(1+Rq.row(0)*M(i,0))%(1+Rq.row(2)*M(i,2))/8;
		dNie.row(16+i) = M(i,2)*(1+Rq.row(0)*M(i,0))%(1+Rq.row(1)*M(i,1))/8;
	}

	// return 
	return dNie;
}

// quadrilateral coordinates to carthesian coordinates
// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
arma::Mat<double> Hexahedron::shape_function_derivative_cart(
	const arma::Mat<double> &Rn,
	const arma::Mat<double> &Rq){
	
	// check input
	assert(Rn.n_rows==3); assert(Rn.n_cols==8);
	assert(Rq.n_rows==3);

	// get shape derivative of shape function
	arma::Mat<double> dNie = Hexahedron::shape_function_derivative(Rq);
	arma::Mat<double> dNiedxi = dNie.rows(0,7); 
	arma::Mat<double> dNiednu = dNie.rows(8,15); 
	arma::Mat<double> dNiedmu = dNie.rows(16,23);

	// setup jacobian
	arma::Row<double> J11 = Rn.row(0)*dNiedxi;
	arma::Row<double> J21 = Rn.row(0)*dNiednu;
	arma::Row<double> J31 = Rn.row(0)*dNiedmu;
	arma::Row<double> J12 = Rn.row(1)*dNiedxi;
	arma::Row<double> J22 = Rn.row(1)*dNiednu;
	arma::Row<double> J32 = Rn.row(1)*dNiedmu;
	arma::Row<double> J13 = Rn.row(2)*dNiedxi;
	arma::Row<double> J23 = Rn.row(2)*dNiednu;
	arma::Row<double> J33 = Rn.row(2)*dNiedmu;
   
   	// calculate determinants
   	arma::Row<double> Jdet= J11%J22%J33 + J21%J32%J13 + 
   		J31%J12%J23 - J31%J22%J13 - J11%J32%J23 - J21%J12%J33;

   	// invert matrices (stored columnwise
   	arma::Mat<double> Jinv(9,Rq.n_cols);
   	Jinv.row(0) = J22%J33-J32%J23;
   	Jinv.row(1) = J31%J23-J21%J33;
   	Jinv.row(2) = J21%J32-J31%J22;
   	Jinv.row(3) = J32%J13-J12%J33;
   	Jinv.row(4) = J11%J33-J31%J13;
   	Jinv.row(5) = J31%J12-J11%J32;
   	Jinv.row(6) = J12%J23-J22%J13;
	Jinv.row(7) = J21%J13-J11%J23;
   	Jinv.row(8) = J11%J22-J21%J12;
	
	// calculate cartesian derivatives
	arma::Mat<double> dNcart(24,Rq.n_cols);
	for(arma::uword i=0;i<Rq.n_cols;i++){
		arma::Mat<double>::fixed<3,3> A = arma::reshape(Jinv.col(i),3,3);
		dNcart.col(i) = arma::reshape((A*arma::reshape(dNie.col(i),8,3).t()).t(),24,1)/Jdet(i);
	}

	// return derivatives
	return dNcart;
}



// quadrilateral coordinates to carthesian coordinates
// Rn can also be a different quantity at the nodes for interpolation
// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
arma::Mat<double> Hexahedron::quad2cart(
	const arma::Mat<double> &Rn, 
	const arma::Mat<double> &Rq){
	
	// check input
	assert(Rn.n_cols==8); assert(Rq.n_rows==3);
	
	// get shape functions
	arma::Mat<double> Nie = Hexahedron::shape_function(Rq);
	//std::cout<<Nie<<std::endl;

	// get coordinates from matrix vector product
	arma::Mat<double> Rc = Rn*Nie;

	// return cartesian coords
	return Rc;
}

// carthesian derivative calculated at quadrilateral coordinates
// phi is the quantity at the nodes
// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]; 
// dphi = [dphidx;dphidy;dphidz]
arma::Mat<double> Hexahedron::quad2cart_derivative(
	const arma::Mat<double> &Rn, 
	const arma::Mat<double> &Rq,
	const arma::Mat<double> &phi){
	
	// check input
	assert(Rn.n_rows==3); assert(Rn.n_cols==8);
	assert(Rq.n_rows==3); assert(phi.n_cols==8);
	
	// get carthesian derivative of shape functions
	arma::Mat<double> dNie = Hexahedron::shape_function_derivative_cart(Rn,Rq);
	
	// get coordinates from matrix vector product
	arma::Mat<double> dphi = arma::reshape(phi*arma::reshape(dNie,8,3*Rq.n_cols),3*phi.n_rows,Rq.n_cols);

	// return cartesian coords
	return dphi;
}

// calculate curl
arma::Mat<double> Hexahedron::quad2cart_curl(
	const arma::Mat<double> &Rn, 
	const arma::Mat<double> &Rq,
	const arma::Mat<double> &V){

	// check input
	assert(Rn.n_rows==3); assert(Rn.n_cols==8);
	assert(Rq.n_rows==3); assert(V.n_cols==8);
	assert(V.n_rows==3);

	// get derivative of V at Rq
	arma::Mat<double> dV = Hexahedron::quad2cart_derivative(Rn,Rq,V);

	// calculate cross product
	// dV = [dVx/dx (0), dVy/dx (1), dVz/dx (2),
	// 		 dVx/dy (3), dVy/dy (4), dVz/dy (5),
	// 		 dVx/dz (6), dVy/dz (7), dVz/dz (8)]
	arma::Mat<double> curl(3,Rq.n_cols);
	curl.row(0) = dV.row(5) - dV.row(7);
	curl.row(1) = dV.row(6) - dV.row(2);
	curl.row(2) = dV.row(1) - dV.row(3);

	// output calculated curl
	return curl;
}



// carthesian coordinates to quadrilateral coordinates
// this function can likely be improved by not using finite difference
// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
arma::Mat<double> Hexahedron::cart2quad(
	const arma::Mat<double> &Rn, 
	const arma::Mat<double> &Rc, 
	const double tol){
	
	// get typical size
	double diag = std::sqrt(arma::as_scalar(arma::sum(
		(Rn.col(0) - Rn.col(6))%(Rn.col(0) - Rn.col(6)),0)));

	// check input
	assert(Rn.n_rows==3); assert(Rn.n_cols==8);
	assert(Rc.n_rows==3);
	
	// initial guess
	arma::Mat<double> Rq(3,Rc.n_cols,arma::fill::zeros);

	// iterations
	double delta = 1e-4;
	for(arma::uword i=0;;i++){
		// calculate difference
		arma::Mat<double> Rc00 = Hexahedron::quad2cart(Rn,Rq);
		arma::Mat<double> dRc = Rc00 - Rc;

		// check for convergence
		double conv = arma::max(arma::max(arma::abs(dRc)));
		if(conv<tol*diag)break;

		// quadrilateral finite difference coords
		arma::Mat<double> Rq0 = Rq; Rq0.row(0) += delta;
		arma::Mat<double> Rq1 = Rq; Rq1.row(1) += delta;
		arma::Mat<double> Rq2 = Rq; Rq2.row(2) += delta;
		
		// calculate carthesian coordinates
		arma::Mat<double> Rc0 = Hexahedron::quad2cart(Rn,Rq0);
		arma::Mat<double> Rc1 = Hexahedron::quad2cart(Rn,Rq1);
		arma::Mat<double> Rc2 = Hexahedron::quad2cart(Rn,Rq2);

		// finite difference
		arma::Mat<double> dRc0 = (Rc00-Rc0)/delta;
		arma::Mat<double> dRc1 = (Rc00-Rc1)/delta;
		arma::Mat<double> dRc2 = (Rc00-Rc2)/delta;

		// calculate change
		for(arma::uword i=0;i<Rc.n_cols;i++){
			// assemble matrix
			arma::Mat<double>::fixed<3,3> A;
			A.col(0) = dRc0.col(i);	A.col(1) = dRc1.col(i);	A.col(2) = dRc2.col(i);

			// update solution
			Rq.col(i) += arma::solve(A,dRc.col(i), arma::solve_opts::fast);
		}

		// check
		assert(i<100);
	}

	// check
	assert(arma::all(arma::all(arma::abs(Hexahedron::quad2cart(Rn,Rq)-Rc)<tol*diag)));

	// return carthesian coordinates
	return Rq;
}

// function for determining if points are inside a hexahedron
arma::Row<arma::uword> Hexahedron::is_inside(
	const arma::Mat<double> &R,
	const arma::Mat<double> &Rn,
	const arma::Mat<arma::uword> &n){

	// check input
	assert(R.n_rows==3); 
	assert(Rn.n_rows==3);
	assert(n.n_rows==8);
	assert(n.n_elem>0);
	assert(arma::max(arma::max(n))<Rn.n_cols);	

	// get number of elements
	arma::uword num_elem = n.n_cols;

	// get hexahedron to tetrahedron matrix
	arma::Mat<arma::uword>::fixed<5,4> M = 
		Hexahedron::tetrahedron_conversion_matrix();

	// make node connectivity matrix for tetrahedrons
	arma::Mat<arma::uword> nt(4*5,num_elem);
	for(int i=0;i<5;i++){
		nt.rows(i*4,(i+1)*4-1) = n.rows(M.row(i));
	}
	nt.reshape(4,5*num_elem);
	
	// check if inside tetrahedrons
	arma::Mat<arma::uword> isinside = 
		Tetrahedron::is_inside(R,Rn,nt);

	// reshape
	return isinside;
}

// setup a grid around a singular point
// grid containes nodes in quadrilateral coordinates (Rqgrd)
// and weights (wgrd)
void Hexahedron::setup_source_grid(
	arma::Mat<double> &Rqgrd, arma::Row<double> &wgrd,
	const arma::Col<double>::fixed<3> &Rqs,
	const arma::Row<double> &xg, const arma::Row<double> &wg){
	
	// check gauss weights
	assert(arma::all(arma::abs(Rqs)<=1.0));
	assert(arma::as_scalar(arma::sum(wg))-2.0<1e-5);

	// scale gauss points
	arma::Row<double> sxg = (xg+1)/2;

	// get side lengths
	arma::Col<double>::fixed<3> a = Rqs+1.0;
	arma::Col<double>::fixed<3> b = 1.0-Rqs;

	// split face depending on target position
	arma::field<arma::Row<double> > x(3), w(3);
	for(arma::uword j=0;j<3;j++){
		if(a(j)>1e-9 && b(j)>1e-9){
			x(j) = arma::join_horiz(a(j)*sxg - 1.0, b(j)*sxg + Rqs(j));
			w(j) = arma::join_horiz(wg*a(j)/2, wg*b(j)/2);
		}
		else if(a(j)>1e-9 && b(j)<1e-9){
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
	const arma::Mat<double> &R, const arma::Mat<arma::uword> &n){
	// select edges
	arma::Mat<double> V1 = R.cols(n.row(1)) - R.cols(n.row(0));
	arma::Mat<double> V2 = R.cols(n.row(2)) - R.cols(n.row(1));

	// select point from other plane 
	arma::Mat<double> V3 = R.cols(n.row(5)) - R.cols(n.row(1));

	// cross product
	arma::Row<double> d = Extra::dot(V3,Extra::cross(V1,V2));

	// return orientation
	return d>0;
}


// volume hexahedrons in a mesh
arma::Row<double> Hexahedron::calc_volume(
	const arma::Mat<double> &Rn,
	const arma::Mat<arma::uword> &n){

	// get tetrahedron conversion matrix
	arma::Mat<arma::uword>::fixed<5,4> M = Hexahedron::tetrahedron_conversion_matrix();

	// allocate volumes and centers of mass
	arma::Row<double> Ve(n.n_cols,arma::fill::zeros);

	// walk over tetrahedrons
	for(arma::uword i=0;i<M.n_rows;i++){
		// calc their volumes
		Ve += Tetrahedron::calc_volume(Rn, n.rows(M.row(i)));
	}

	// return element volumes
	return Ve;
}

// // subdivide to remove singularity from equation
// // should work for both two and three dimensions
// // the singularity is assumed to be between a and b
// // output is from 0 to a+b
// arma::Mat<double> Hexahedron::subdivide(
// 	const arma::Col<double> &a, 
// 	const arma::Col<double> &b){

// 	// check
// 	assert(a.n_elem==b.n_elem);
// 	assert(arma::all(a>=0));
// 	assert(arma::all(b>=0));

// 	// calculate total width
// 	arma::Col<double> w = a+b;

// 	// get distance
// 	arma::Col<double> dist(a.n_elem);
// 	for(arma::uword j=0;j<dist.n_elem;j++){
// 		if(a(j)<b(j))dist(j)=a(j); else dist(j)=b(j);
// 	}

// 	// sort by distance
// 	arma::Col<arma::uword> idx = arma::sort_index(dist,"ascend");

// 	// first dimension 
// 	arma::Mat<double> division(dist.n_elem,2,arma::fill::zeros);
// 	if(a(idx(0))<b(idx(0))){
// 		arma::Row<double> val = {a(idx(0)),a(idx(0)),(w(idx(0))-2*a(idx(0)))/2,(w(idx(0))-2*a(idx(0)))/2};
// 		division.row(idx(0)) = val;
// 	}else{
// 		arma::Row<double> val = {(w(idx(0))-2*b(idx(0)))/2,(w(idx(0))-2*b(idx(0)))/2,b(idx(0)),b(idx(0))};
// 		division.row(idx(0)) = val;
// 	}
	
// 	// other dimensions
// 	for(arma::uword j=1;j<dist.n_elem;j++){
// 		if(dist(idx(j))<w(idx(j))/100){
// 			arma::Row<double> val = {w(idx(j))/2,w(idx(j))/2,0,0};
// 			division.row(idx(j)) = val;
// 		}else{
// 			arma::Row<double> val = {a(idx(j))-dist(idx(0)),dist(idx(0)),dist(idx(0)),b(idx(j))-dist(idx(0))};
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
// 	// arma::Mat<double> sxg = arma::cumsum(division,1)-division/2;

// 	// divide again
// 	arma::Mat<double> seconddivision = arma::join_vert(division/2,division/2);
// 	seconddivision.reshape(division.n_rows,division.n_cols*2);

// 	// return gauss points
// 	return seconddivision;
// }




