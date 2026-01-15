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

// include header
#include "steepest.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	Steepest::Steepest(){
		
	}

	// factory
	ShSteepestPr Steepest::create(){
		return std::make_shared<Steepest>();
	}

	// system function
	void Steepest::set_systemfun(StSysFun systemfun){
		systemfun_ = systemfun;
	}

	// gradient function using finite difference
	void Steepest::set_finite_difference(){
		// note capture by reference 
		// always updated values will be used
		gradientfun_ = [&](const arma::Col<fltp> &x){
			arma::Col<fltp> G = approximate_gradient(x, delta_, use_parallel_, use_central_diff_, systemfun_);
			return G;
		};
	}

	// external gradient function
	void Steepest::set_gradientfun(StGradFun gradientfun){
		gradientfun_ = gradientfun;
	}

	// set initial values
	void Steepest::set_initial(const arma::Col<fltp> &x0){
		x0_ = x0;
	}

	// set stepsize for finite difference
	void Steepest::set_delta(const fltp delta){
		if(delta<=0)rat_throw_line("finite difference stepsize must be larger than zero");
		delta_ = delta;
	}		

	// set stepsize for finite difference
	void Steepest::set_gamma(const fltp gamma){
		if(gamma<=0)rat_throw_line("line search damping factor must be larger than zero");
		gamma_ = gamma;
	}	

	// set stepsize for finite difference
	void Steepest::set_tolx(const fltp tolx){
		if(tolx<=0)rat_throw_line("tolerance must be larger than zero");
		tolx_ = tolx;
	}	

	// set parallel evaluation of finite difference
	void Steepest::set_use_parallel(const bool use_parallel){
		use_parallel_ = use_parallel;
	}

	// set central difference
	void Steepest::set_use_central_diff(const bool use_central_diff){
		use_central_diff_ = use_central_diff;
	}

	// set central difference
	void Steepest::set_use_linesearch(const bool use_linesearch){
		use_linesearch_ = use_linesearch;
	}

	// set maximum number of iterations
	void Steepest::set_num_iter_max(const arma::uword num_iter_max){
		num_iter_max_ = num_iter_max;
	}

	// run optimisation
	void Steepest::optimise(ShLogPr lg){
		// print header to log
		lg->msg(2,"%s%sSTEEPEST-DESCENT OPTIMIZER%s\n",KGRN,KBLD,KNRM);

		// display settings
		lg->msg(2,"%ssettings%s\n",KBLU,KNRM);
		lg->msg("number of variables: %s%04llu%s\n",KYEL,x0_.n_elem,KNRM);
		lg->msg("variable tolerance: %s%8.2e%s\n",KYEL,tolx_,KNRM);
		lg->msg("max number of iter: %s%04llu%s\n",KYEL,num_iter_max_,KNRM);
		lg->msg("finite diff paralel: %s%llu%s\n",KYEL,use_parallel_,KNRM);
		lg->msg("use linesearch: %s%llu%s\n",KYEL,use_linesearch_,KNRM);
		lg->msg("central difference: %s%llu%s\n",KYEL,use_central_diff_,KNRM);
		lg->msg(-2,"\n");

		lg->msg(2,"%srunning optimization%s\n",KBLU,KNRM);

		// set initial value 
		x_ = x0_;

		// header
		lg->msg("%s%4s %8s %8s%s\n",KBLD,"iter", "fval", "stepnorm",KNRM);

		// iterate
		for(arma::uword i=0;i<num_iter_max_;i++){
			// calculate gradient
			arma::Col<fltp> dfdx = gradientfun_(x_);
			fltp fk = systemfun_(x_);

			// calculate step norm
			fltp stepnorm = arma::norm(dfdx);

			// display
			// if(display_){
			// 	reportfun_(i,fk,stepnorm);
			// }
			lg->msg("%04llu %8.2e %8.2e\n", i, fk, stepnorm);

			// check tolerance
			if(stepnorm<tolx_)break;
			
			// line search 
			if(use_linesearch_){
				arma::Col<fltp> xx = x_;
				fltp alpha = 1;
				fltp c = 1e-4f;
				x_ = xx - alpha*dfdx;
				fltp fk1 = systemfun_(x_);
				while(fk1>fk - c*alpha*arma::as_scalar(dfdx.t()*dfdx)){
					alpha*=gamma_;
					x_ = xx - alpha*dfdx;
					fk1 = systemfun_(x_);
				}
			}

			// without linesearch
			else{
				x_ -= gamma_*dfdx;
			}
		}

		// calculate final function value
		fval_ = systemfun_(x_);

		// done
		lg->msg(-2,"\n");
		lg->msg(-2);
	}

	// gradient approximation (central difference)
	arma::Col<fltp> Steepest::approximate_gradient(
		const arma::Col<fltp> &x, const fltp dx, 
		const bool use_parallel, const bool central_diff, 
		StSysFun sysfn){

		// allocate df
		arma::Col<fltp> df(x.n_elem);

		// for central finite difference
		if(central_diff){
			// walk over df
			parfor(0,x.n_elem,use_parallel,[&](arma::uword i, arma::uword) {
				arma::Col<fltp> dxe(x.n_elem,arma::fill::zeros); dxe(i) += dx;
				df(i) = (sysfn(x+dxe/2) - sysfn(x-dxe/2))/dx;
			});
		}

		// for anti-symmetric finite difference
		else{
			// calculate center
			const fltp f0 = sysfn(x);

			// walk over df
			parfor(0,x.n_elem,use_parallel,[&](arma::uword i, arma::uword) {
				arma::Col<fltp> dxe(x.n_elem,arma::fill::zeros); dxe(i) += dx;
				df(i) = (sysfn(x+dxe)-f0)/dx;
			});
		}

		// return gradient
		return df;
	}

	// get result vector
	arma::Col<fltp> Steepest::get_result() const{
		return x_;
	}

	// // display function
	// void Steepest::default_displayfun(
	// 	const arma::uword n, const fltp fval, const fltp stepnorm){
	// 	// display header
	// 	if(n==0){
	// 		std::printf("\nSteepest Descent Optimiser\n");
	// 		std::printf("%10s %10s %10s\n", "iter", "fval", "stepnorm");
	// 		for(arma::uword n = 0;n<37;n++)std::printf("-");
	// 		std::printf("\n");
	// 	}

	// 	// display iteration
	// 	std::printf("%10llu %10.4g %10.4g\n", n, fval, stepnorm);
	// }

}}
