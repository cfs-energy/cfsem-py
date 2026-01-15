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

// general headers
#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>
#include <functional>

// specific headers
#include "error.hh"
#include "extra.hh"
#include "gmres.hh"
#include "log.hh"

// main
int main(){
	// settings
	const arma::uword N = 100;
	const arma::uword num_iter_max = 150;
	const arma::uword num_restart = 32;
	const rat::fltp tol = RAT_CONST(1e-6);

	// create log
	rat::cmn::ShLogPr lg = rat::cmn::Log::create();

	// system of equations
	arma::Mat<rat::fltp> A(N,N,arma::fill::randu);
	A.diag(-1).fill(RAT_CONST(2.0));
	A.diag(0).fill(RAT_CONST(5.0));
	A.diag(+1).fill(RAT_CONST(2.0));
	const arma::Col<rat::fltp> b(N,arma::fill::ones);

	// output vector (and initial guess)
	arma::Col<rat::fltp> x(N,arma::fill::zeros);

	// setup preconditioner
	arma::Mat<rat::fltp> M(N,N,arma::fill::zeros);
	for(int i=-2;i<=2;i++)M.diag(i) = A.diag(i);

	// decomposition
	arma::Mat<rat::fltp> L, U, P;
	arma::lu(L, U, P, M);

	// generate solver
	rat::cmn::ShGMRESPr mygmres = rat::cmn::GMRES::create();

	// system function
	mygmres->set_systemfun(
		[A](const arma::Col<rat::fltp> &x){
	 	return A*x;
	});

	// preconditioner
	mygmres->set_precfun(
		[P,L,U](const arma::Col<rat::fltp> &r){	
		//return arma::solve(M,x);
		arma::Col<rat::fltp> out = arma::solve(arma::trimatu(U),arma::solve(arma::trimatl(L),P*r,
			arma::solve_opts::fast),arma::solve_opts::fast);
		return out;
	});

	// solver settings
	mygmres->set_num_iter_max(num_iter_max);
	mygmres->set_num_restart(num_restart);
	mygmres->set_tol(tol);

	// solve system
	mygmres->solve(x,b,lg);

	// check result
	if(arma::any((A*x-b)/arma::norm(b)>tol))rat_throw_line("tolerance not achieved");

	// return
	return 0;
}