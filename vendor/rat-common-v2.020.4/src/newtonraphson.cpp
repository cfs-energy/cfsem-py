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

// this code was translated to C++ and adapted from:
// https://ch.mathworks.com/matlabcentral/fileexchange/43097-newton-raphson-solver

// include header file
#include "newtonraphson.hh"

// common headers
#include "error.hh"
#include "parfor.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	NewtonRaphson::NewtonRaphson(){

	}

	// factory
	ShNewtonRaphsonPr NewtonRaphson::create(){
		return std::make_shared<NewtonRaphson>();
	}

	// setting of system function
	void NewtonRaphson::set_systemfun(NRSysFun fn){
		systemfun_ = fn;
	}

	// set jacobian function externally
	void NewtonRaphson::set_jacfun(NRJacFun fn){
		jacfun_ = fn;
	}

	// set jacobian function
	void NewtonRaphson::set_finite_difference(){
		// note capture by reference 
		// always updated values will be used
		jacfun_ = [&](const arma::Col<fltp> &x){
			const arma::Mat<fltp> J = approximate_jacobian(x, delta_, use_parallel_, use_central_diff_, systemfun_);
			return J;
		};
	}

	// set tolerance
	void NewtonRaphson::set_tolx(const fltp tolx){
		if(tolx<=0)rat_throw_line("variabel tolerance must be larger than zero");
		tolx_ = tolx;
	}	

	// set tolerance
	void NewtonRaphson::set_tolfun(const fltp tolfun){
		if(tolfun<=0)rat_throw_line("function tolerance must be larger than zero");
		tolfun_ = tolfun;
	}		
			
	// set stepsize for finite difference
	void NewtonRaphson::set_delta(const fltp delta){
		if(delta<=0)rat_throw_line("finite difference stepsize must be larger than zero");
		delta_ = delta;
	}		

	// set maximum number of iterations
	void NewtonRaphson::set_num_iter_max(const arma::uword num_iter_max){
		if(num_iter_max<=0)rat_throw_line("number of iterations must be larger than zero");
		num_iter_max_ = num_iter_max;
	}

	// set parallel evaluation of finite difference
	void NewtonRaphson::set_use_parallel(const bool use_parallel){
		use_parallel_ = use_parallel;
	}

	// set central difference
	void NewtonRaphson::set_use_central_diff(const bool use_central_diff){
		use_central_diff_ = use_central_diff;
	}

	// set initial values (also defines number of equations)
	void NewtonRaphson::set_initial(const arma::Col<fltp> &x0){
		x0_ = x0;
	}

	// solver functions
	void NewtonRaphson::solve(ShLogPr lg){
		// get number of equations
		const arma::uword num_equations = x0_.n_elem;
		
		// print header to log
		lg->msg(2,"%s%sNEWTON-RAPHSON SOLVER%s\n",KGRN,KBLD,KNRM);

		// display settings
		lg->msg(2,"%ssettings%s\n",KBLU,KNRM);
		lg->msg("number of equations: %s%04llu%s\n",KYEL,num_equations,KNRM);
		lg->msg("variable tolerance: %s%8.2e%s\n",KYEL,tolx_,KNRM);
		lg->msg("function tolerance: %s%8.2e%s\n",KYEL,tolfun_,KNRM);
		lg->msg("max number of iter: %s%04llu%s\n",KYEL,num_iter_max_,KNRM);
		lg->msg("finite diff stepsize: %s%8.2e%s\n",KYEL,delta_,KNRM);
		lg->msg("finite diff paralel: %s%llu%s\n",KYEL,use_parallel_,KNRM);
		lg->msg("central difference: %s%llu%s\n",KYEL,use_central_diff_,KNRM);
		lg->msg(-2,"\n");

		// set exitflag
		flag_ = 0;

		// x scaling value, remove zeros
		arma::Col<fltp> typx = arma::clamp(arma::abs(x0_), 1.0f, arma::Datum<fltp>::inf); 

		// initial weight
		arma::Col<fltp> weight(num_equations,arma::fill::ones);

		// check initial guess
		x_ = x0_;

		// evaluate initial guess
		J_ = jacfun_(x_);
		F_ = systemfun_(x_);

		// check 
		if(F_.n_elem!=num_equations)rat_throw_line("system function output verctor does not match number of equations");
		if(J_.n_cols!=num_equations)rat_throw_line("jacobian matrix number of columns is unequal to the number of equations");
		if(J_.n_rows!=num_equations)rat_throw_line("jacobian matrix number of rows is unequal to the number of equations");

		// scale jacobian
		//arma::Mat<fltp> Jstar = J/J0;
		arma::Mat<fltp> Jstar = (J_.each_col()/weight).each_row()%typx.t();

		// check for nans and infinites
		if(!Jstar.is_finite()){
			flag_ = -1; // matrix may be singular
		}

		// calculate reciprocal condition
		rc_ = 1.0f/arma::cond(Jstar);

		// check initial guess
		resnorm_ = arma::norm(F_); // calculate norm of the residuals

		// dummy values
		dx_.zeros(num_equations);
		fltp convergence = arma::Datum<fltp>::inf; 
		
		// solver
		num_iter_ = 0; // start counter
		lambda_ = 1.0f; // backtracking

		// allocate
		fltp lambda_min=0, fold=0, slope=0, lambda2=0, f2=0, resnorm0; 
		arma::Col<fltp> xold;

		// header for output table
		lg->msg(2,"%srunning solver%s\n",KBLU,KNRM);
		lg->msg("%s%4s %8s %8s %8s %8s %8s%s\n",KBLD,"iter","resnorm","stepnorm","lambda","rcond","conv",KNRM);

		// iterations
		while((resnorm_>tolfun_ || lambda_<1.0f) && flag_>=0 && num_iter_<=num_iter_max_){
			// newton calculation
			if(lambda_==1){
				// increment counter
				num_iter_++;

				// calculate newton step
				arma::Col<fltp> dx_star = -arma::solve(Jstar,F_);
				
				// scale step with typical value for x
				dx_ = dx_star%typx; 
				
				// gradient of resnorm
				arma::Row<fltp> g = F_.t()*Jstar; 
		
				// slope of gradient
				slope = arma::as_scalar(g*dx_star); 
				
				// objective function
				fold = arma::as_scalar(F_.t()*F_); 

				// initial value
				xold = x_;

				// calculate lambda
				lambda_min = tolx_/arma::max(arma::abs(dx_)/arma::clamp(arma::abs(xold),0.0f,1.0f));
			}

			// check lambda
			if(lambda_<lambda_min){
				flag_ = 2; // x is too close to XOLD
				break;
			}

			// update x along dx scaled with lambda
			x_ = xold + dx_*lambda_;

			// evaluate next residuals
			J_ = jacfun_(x_);
			F_ = systemfun_(x_);

			// check 
			if(F_.n_elem!=num_equations)rat_throw_line("system function output verctor does not match number of equations");
			if(J_.n_cols!=num_equations)rat_throw_line("jacobian matrix number of columns is unequal to the number of equations");
			if(J_.n_rows!=num_equations)rat_throw_line("jacobian matrix number of rows is unequal to the number of equations");

			// scale next Jacobian
			//Jstar = J/J0; 
			Jstar = (J_.each_col()/weight).each_row()%typx.t();

			// next objective function
			fltp f = arma::as_scalar(F_.t()*F_); 
			
			// check for convergence
			// save previous lambda
			fltp lambda1 = lambda_; 

			// stepsize update
			if(f>(fold+alpha_*lambda_*slope)){
				if(lambda_==1){
					// calculate lambda
					lambda_ = -slope/2/(f-fold-slope); 
				}else{
					fltp A = 1.0f/(lambda1 - lambda2);
					arma::Mat<fltp> B(2,2);
					B.row(0) = arma::Row<fltp>{1.0f/(lambda1*lambda1),-1.0f/(lambda2*lambda2)};
					B.row(1) = arma::Row<fltp>{-lambda2/(lambda1*lambda1),lambda1/(lambda2*lambda2)};
					arma::Col<fltp> C = {f-fold-lambda1*slope,f2-fold-lambda2*slope};
					arma::Mat<fltp> coeff = A*B*C;
					fltp a = coeff(0); // not fully sure if correctly translated
					fltp b = coeff(1);
					if(a==0){
						lambda_ = -slope/2/b;
					}else{
						fltp discriminant = b*b - 3.0*a*slope;
						if(discriminant<0){
							lambda_ = max_lambda_*lambda1;
						}else if(b<=0){
							lambda_ = (-b+std::sqrt(discriminant))/3.0/a;
						}else{
							lambda_ = -slope/(b+std::sqrt(discriminant));
						}
					}

					// minimum step length
					lambda_ = std::min(lambda_,max_lambda_*lambda1); 
				}
			}else{
				lambda_ = 1.0;
			}

			// check for nans and infinites
			if(!Jstar.is_finite()){
				flag_ = -1; // matrix may be singular
				
				break;
			}

			if(lambda_<1.0f){
				lambda2 = lambda1;
				f2 = f; // save 2nd most previous value
				lambda_ = std::max(lambda_,min_lambda_*lambda1); // minimum step length
				continue;
			}

			// display
			resnorm0 = resnorm_; // old resnorm
			resnorm_ = arma::norm(F_); // calculate new resnorm
			convergence = std::log(resnorm0/resnorm_); // calculate convergence rate
			stepnorm_ = arma::norm(dx_); // norm of the step
			rc_ = 1/arma::cond(Jstar); // reciprocal condition

			// print line into table
			lg->msg("%04llu %08.2e %08.2e %8.4f %8.4f %8.4f\n", num_iter_, resnorm_, stepnorm_, lambda1, rc_, convergence);
		}

		// analyse output
		if(num_iter_>=num_iter_max_){
			flag_ = 0;
			lg->msg("%s= max number of iterations exceeded%s\n",KRED,KNRM);
		}else if(flag_==2){
			lg->msg("%s= x is too close to previous value%s\n",KRED,KNRM);
		}else if(flag_==-1){
			lg->msg("%s= jacobian matrix may be singular%s\n",KRED,KNRM);
		}else{
			lg->msg("%s= solution found%s\n",KCYN,KNRM);
		}

		// done
		lg->msg(-2,"\n");
		lg->msg(-2);
	}

	// get result vector
	arma::Col<fltp> NewtonRaphson::get_result() const{
		return x_;
	}

	// jacobian approximation at x
	// this can be used as a replacement for the 
	// jacobian function. It uses a center 
	// difference approximation and should only 
	// be used for small systems (not super efficient)
	arma::Mat<fltp> NewtonRaphson::approximate_jacobian(
		const arma::Col<fltp> &x, const fltp dx, 
		const bool use_parallel, const bool central_diff, NRSysFun sysfn){

		// allocate jacbian matrix
		arma::Mat<fltp> J(x.n_elem,x.n_elem);

		// for central finite difference
		if(central_diff){
			// walk over degrees of freedom	
			parfor(0,x.n_elem,use_parallel,[&](arma::uword i, int) {
				arma::Col<fltp> delta(x.n_elem,arma::fill::zeros); 
				delta(i)+=dx;
				arma::Col<fltp> dF = sysfn(x+delta)-sysfn(x-delta);
				assert(dF.n_elem==x.n_elem);
				J.col(i) = dF/(2*dx);
			});
		}

		// for anti-symmetric finite difference
		else{
			// calculate center
			const arma::Col<fltp> f0 = sysfn(x);

			// walk over degrees of freedom	
			parfor(0,x.n_elem,use_parallel,[&](arma::uword i, int) {
				arma::Col<fltp> delta(x.n_elem,arma::fill::zeros); 
				delta(i)+=dx;
				arma::Col<fltp> dF = sysfn(x+delta)-f0;
				assert(dF.n_elem==x.n_elem);
				J.col(i) = dF/dx;
			});
		}

		// return jacobian matrix
		return J;
	}

}}