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

#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>
#include <functional>

#include "common/extra.hh"
#include "gmres.hh"

// main
int main(){
	// settings
	arma::uword N = 100;
	arma::uword num_iter_max = 150;
	arma::uword num_restart = 32;
	double tol = 1e-6;

	// system of equations
	arma::Mat<double> A(N,N,arma::fill::randu);
	A.diag(-1).fill(2.0);
	A.diag(0).fill(5.0);
	A.diag(+1).fill(2.0);
	arma::Col<double> b(N,arma::fill::ones);

	// output vector (and initial guess)
	arma::Col<double> x(N,arma::fill::zeros);

	// setup preconditioner
	arma::Mat<double> M(N,N,arma::fill::zeros);
	for(int i=-2;i<=2;i++)M.diag(i) = A.diag(i);

	// decomposition
	arma::Mat<double> L, U, P;
	arma::lu(L, U, P, M);

	// generate solver
	GMRES* mygmres = new GMRES;

	// system function
	mygmres->set_systemfun(
		[A](const arma::Col<double> &x){
	 	return A*x;
	});

	// preconditioner
	mygmres->set_precfun(
		[P,L,U](const arma::Col<double> &r){	
		//return arma::solve(M,x);
		arma::Col<double> out = arma::solve(arma::trimatu(U),arma::solve(arma::trimatl(L),P*r,
			arma::solve_opts::fast),arma::solve_opts::fast);
		return out;
	});

	// solver settings
	mygmres->set_num_iter_max(num_iter_max);
	mygmres->set_num_restart(num_restart);
	mygmres->set_tol(tol);

	// solve system
	mygmres->solve(x,b);

	// output 
	mygmres->display();
	
	// check result
	assert(arma::all((A*x-b)/arma::norm(b)<tol));

	// delete the solver
	delete mygmres;

	// return
	return 0;
}