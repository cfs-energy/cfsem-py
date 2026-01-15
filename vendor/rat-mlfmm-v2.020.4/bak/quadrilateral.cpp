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
#include "quadrilateral.hh"


// quadrilateral corner node positions 
// in quadrilateral coordinates
// matrix columns are given as [xi,nu]
arma::Mat<arma::sword>::fixed<4,2> Quadrilateral::get_corner_nodes(){
	// setup matrix
	arma::Mat<arma::sword>::fixed<4,2> M; 
	M.col(0) = arma::Col<arma::sword>{-1,+1,+1,-1};
	M.col(1) = arma::Col<arma::sword>{-1,-1,+1,+1};

	// return matrix
	return M;
}

// quadrilateral edges connectivity
arma::Mat<arma::uword>::fixed<4,2> Quadrilateral::get_edges(){
	// setup matrix
	arma::Mat<arma::uword>::fixed<4,2> M; 
	M.col(0) = arma::Col<arma::uword>{0,1,2,3};
	M.col(1) = arma::Col<arma::uword>{1,2,3,0};

	// return matrix
	return M;
}


// quadrilateral shape function
arma::Mat<double> Quadrilateral::shape_function(
	const arma::Mat<double> &Rq){
	
	// check input
	assert(Rq.n_rows==2);

	// hexahedron nodes in quadrilateral coordinates [xi,nu,mu]
	arma::Mat<arma::sword>::fixed<4,2> M = Quadrilateral::get_corner_nodes();

	// calculate shape functions
	arma::Mat<double> Nie(4,Rq.n_cols);

	// fill matrix
	for(arma::uword i=0;i<4;i++){
		Nie.row(i) = 
			(1+Rq.row(0)*M(i,0))%
			(1+Rq.row(1)*M(i,1))/4;
	}

	// return 
	return Nie;
}

// quadrilateral coordinates to carthesian coordinates
// Rn can also be a different quantity at the nodes for interpolation
// Rq = [xi;nu], Rc = [x;y;z], Rn = [x;y;z]
arma::Mat<double> Quadrilateral::quad2cart(
	const arma::Mat<double> &Rn, 
	const arma::Mat<double> &Rq){
	
	// check input
	assert(Rn.n_cols==4); assert(Rq.n_rows==2);
	
	// get shape functions
	arma::Mat<double> Nie = Quadrilateral::shape_function(Rq);
	//std::cout<<Nie<<std::endl;

	// get coordinates from matrix vector product
	arma::Mat<double> Rc = Rn*Nie;

	// return cartesian coords
	return Rc;
}


// function for checking if surface nodes are in plane
bool Quadrilateral::check_nodes(
	const arma::Mat<double> &Rn){
	// get typical size
	double diag = std::sqrt(arma::as_scalar(arma::sum(
		(Rn.col(0) - Rn.col(2))%(Rn.col(0) - Rn.col(2)),0)));

	// calculate the plane in which the face is located
	arma::Col<double>::fixed<3> V1 = Rn.col(1)-Rn.col(0);
	arma::Col<double>::fixed<3> V2 = Rn.col(3)-Rn.col(1);

	// get face normal
	arma::Col<double>::fixed<3> N = Extra::cross(V1,V2);
	N = N/arma::as_scalar(Extra::vec_norm(N));

	// check if diagonals are in plane
	double eps1 = arma::as_scalar(Extra::dot(N,Rn.col(2)-Rn.col(0)));
	double eps2 = arma::as_scalar(Extra::dot(N,Rn.col(1)-Rn.col(3)));

	// check flatness (returns true if element valid)
	return (eps1/diag)<1e-5 && (eps2/diag)<1e-5;
}


// carthesian coordinates to quadrilateral coordinates
// this function can likely be improved by not using finite difference
// Rq = [xi;nu;mu], Rc = [x;y;z], Rn = [x;y;z]
arma::Mat<double> Quadrilateral::cart2quad(
	const arma::Mat<double> &Rn, 
	const arma::Mat<double> &Rc, 
	const double tol){
	
	// check nodes
	assert(Quadrilateral::check_nodes(Rn));

	// get typical size
	double diag = std::sqrt(arma::as_scalar(arma::sum(
		(Rn.col(0) - Rn.col(2))%(Rn.col(0) - Rn.col(2)),0)));

	// check input
	assert(Rn.n_rows==3); assert(Rn.n_cols==4);
	assert(Rc.n_rows==3);
	
	// calculate the plane in which the face is located
	// arma::Col<double>::fixed<3> R0 = Rn.col(0);
	arma::Col<double>::fixed<3> V1 = Rn.col(1)-Rn.col(0);
	arma::Col<double>::fixed<3> V2 = Rn.col(3)-Rn.col(0);
	V1 = V1/arma::as_scalar(Extra::vec_norm(V1)); 
	V2 = V2/arma::as_scalar(Extra::vec_norm(V2));
	arma::Mat<double>::fixed<2,3> M12 = arma::join_vert(V1.t(),V2.t());

	// initial guess
	arma::Mat<double> Rq(2,Rc.n_cols,arma::fill::zeros);
	
	// iterations
	double delta = 1e-4;
	for(arma::uword i=0;;i++){
		// calculate difference
		arma::Mat<double> Rc00 = M12*Quadrilateral::quad2cart(Rn,Rq.rows(0,1));
		arma::Mat<double> dRc = Rc00 - M12*Rc;
		
		// check for convergence
		double conv = arma::max(arma::max(arma::abs(dRc)))/diag;
		if(conv<tol)break;

		// quadrilateral finite difference coords
		arma::Mat<double> Rq0 = Rq; Rq0.row(0) += delta;
		arma::Mat<double> Rq1 = Rq; Rq1.row(1) += delta;

		// calculate carthesian coordinates
		arma::Mat<double> Rc0 = M12*Quadrilateral::quad2cart(Rn,Rq0.rows(0,1)); 
		arma::Mat<double> Rc1 = M12*Quadrilateral::quad2cart(Rn,Rq1.rows(0,1));

		// finite difference
		arma::Mat<double> dRc0 = (Rc00-Rc0)/delta;
		arma::Mat<double> dRc1 = (Rc00-Rc1)/delta;
		
		// calculate change
		for(arma::uword i=0;i<Rc.n_cols;i++){
			// assemble matrix
			arma::Mat<double>::fixed<2,2> A;
			A.col(0) = dRc0.col(i);	
			A.col(1) = dRc1.col(i);	
			
			// update solution
			Rq.col(i) += arma::solve(A,dRc.col(i));
		}

		// check
		assert(i<100);
	}

	// return coords
	return Rq;
}

// setup a grid around a singular point
// grid containes nodes in quadrilateral coordinates (Rqgrd)
// and weights (wgrd)
void Quadrilateral::setup_source_grid(
	arma::Mat<double> &Rqgrd, arma::Row<double> &wgrd,
	const arma::Col<double>::fixed<2> &Rqs,
	const arma::Row<double> &xg, const arma::Row<double> &wg){
		
	// check gauss weights
	assert(arma::all(arma::abs(Rqs)<=1.0));
	assert(arma::as_scalar(arma::sum(wg))-2.0<1e-5);
	assert(Rqs.n_rows==2); assert(Rqs.n_cols==1);
	assert(xg.n_elem==wg.n_elem);

	// scale gauss points
	arma::Row<double> sxg = (xg+1)/2;

	// get lengths of each side
	arma::Col<double>::fixed<2> a = Rqs + 1.0;
	arma::Col<double>::fixed<2> b = 1.0 - Rqs;

	// split face depending on target position
	arma::field<arma::Row<double> > x(2), w(2);
	for(arma::uword j=0;j<2;j++){
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
	assert(std::abs(arma::as_scalar(arma::sum(wgrd))-1.0)<1e-5);

	// check
	assert(arma::all(arma::all(Rqgrd<1.0 && Rqgrd>-1.0)));
}
