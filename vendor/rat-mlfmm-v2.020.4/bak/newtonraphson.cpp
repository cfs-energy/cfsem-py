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

// this file was evolved from:
// http://math.nist.gov/iml++/

// include header file
#include "newtonraphson.hh"

// constructor
NewtonRaphson::NewtonRaphson(){
	// set report function
	reportfun_ = NewtonRaphson::default_displayfun;
}

// factory
ShNewtonRaphsonPr NewtonRaphson::create(){
	return std::make_shared<NewtonRaphson>();
}

// setting of system function
void NewtonRaphson::set_systemfun(NRSysFun fn){
	systemfun_ = fn;
}

// set jacobian function
void NewtonRaphson::set_jacfun(){
	// note capture by reference 
	// always updated values will be used
	jacfun_ = [&](const arma::Col<double> &x){
		arma::Mat<double> J = approximate_jacobian(x, delta_, use_parallel_, systemfun_);
		return J;
	};
}

// set jacobian function externally
void NewtonRaphson::set_jacfun(NRJacFun fn){
	jacfun_ = fn;
}

// set report function
void NewtonRaphson::set_reportfun(NRReportFun fn){
	reportfun_ = fn;
}


void NewtonRaphson::set_display(const bool display){
	display_ = display;
}

// set tolerance
void NewtonRaphson::set_tolx(const double tolx){
	assert(tolx>0);
	tolx_ = tolx;
}	

// set tolerance
void NewtonRaphson::set_tolfun(const double tolfun){
	assert(tolfun>0);
	tolfun_ = tolfun;
}		
		
// set stepsize for finite difference
void NewtonRaphson::set_delta(const double delta){
	assert(delta>0);
	delta_ = delta;
}		

// set maximum number of iterations
void NewtonRaphson::set_num_iter_max(const arma::uword num_iter_max){
	assert(num_iter_max>0);
	num_iter_max_ = num_iter_max;
}

// set parallel evaluation of finite difference
void NewtonRaphson::set_use_parallel(const bool use_parallel){
	use_parallel_ = use_parallel;
}

// set initial values
void NewtonRaphson::set_initial(const arma::Col<double> &x0){
	x0_ = x0;
}

// solver functions
void NewtonRaphson::solve(){
	// set exitflag
	flag_ = 0;

	// get number of equations
	arma::uword num_equations = x0_.n_elem;
	
	// x scaling value, remove zeros
	arma::Col<double> typx = arma::clamp(arma::abs(x0_), 1.0, arma::datum::inf); 

	// initial weight
	arma::Col<double> weight(num_equations,arma::fill::ones);

	// check initial guess
	x_ = x0_;

	// evaluate initial guess
	J_ = jacfun_(x_);
	F_ = systemfun_(x_);

	// scale jacobian
	//arma::Mat<double> Jstar = J/J0;
	arma::Mat<double> Jstar = (J_.each_col()/weight).each_row()%typx.t();

	// check for nans and infinites
	if(!Jstar.is_finite()){
    	flag_ = -1; // matrix may be singular
	}

	// calculate reciprocal condition
	rc_ = 1/arma::cond(Jstar);

	// check initial guess
	resnorm_ = arma::norm(F_); // calculate norm of the residuals

	// dummy values
	dx_.zeros(num_equations);
	double convergence = arma::datum::inf; 
	
	// solver
	num_iter_ = 0; // start counter
	lambda_ = 1.0; // backtracking

	// allocate
	double lambda_min=0, fold=0, slope=0, lambda2=0, f2=0, resnorm0; 
	arma::Col<double> xold;

	// iterations
	while((resnorm_>tolfun_ || lambda_<1.0) && flag_>=0 && num_iter_<=num_iter_max_){
		// newton calculation
		if(lambda_==1){
			// increment counter
			num_iter_++;

			// solve for dx
			//arma::Col<double> dx_ = 


			// calculate newton step
			arma::Col<double> dx_star = -arma::solve(Jstar,F_);
			
			// scale step with typical value for x
			dx_ = dx_star%typx; 
			
			// gradient of resnorm
			arma::Row<double> g = F_.t()*Jstar; 
	
			// slope of gradient
			slope = arma::as_scalar(g*dx_star); 
			
			// objective function
			fold = arma::as_scalar(F_.t()*F_); 

			// initial value
			xold = x_;

			// calculate lambda
			lambda_min = tolx_/arma::max(arma::abs(dx_)/arma::clamp(arma::abs(xold),0.0,1.0));
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

		// scale next Jacobian
		//Jstar = J/J0; 
		Jstar = (J_.each_col()/weight).each_row()%typx.t();

		// next objective function
		double f = arma::as_scalar(F_.t()*F_); 
		
		// check for convergence
		// save previous lambda
		double lambda1 = lambda_; 

		// stepsize update
		if(f>(fold+alpha_*lambda_*slope)){
			if(lambda_==1){
				// calculate lambda
			    lambda_ = -slope/2/(f-fold-slope); 
			}else{
				double A = 1.0/(lambda1 - lambda2);
				arma::Mat<double> B(2,2);
				B.row(0) = arma::Row<double>{1.0/(lambda1*lambda1),-1.0/(lambda2*lambda2)};
				B.row(1) = arma::Row<double>{-lambda2/(lambda1*lambda1),lambda1/(lambda2*lambda2)};
				arma::Col<double> C = {f-fold-lambda1*slope,f2-fold-lambda2*slope};
				arma::Mat<double> coeff = A*B*C;
				double a = coeff(0); // not fully sure if correctly translated
				double b = coeff(1);
				if(a==0){
				    lambda_ = -slope/2/b;
				}else{
				    double discriminant = b*b - 3.0*a*slope;
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

		if(lambda_<1.0){
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
    	if(display_==true){
    		reportfun_(num_iter_, resnorm_, stepnorm_, lambda1, rc_, convergence);
    	}
	}

	// analyse output
	if(num_iter_>=num_iter_max_){
		flag_ = 0;
		message_ = "Number of iterations exceeded num_iter_max.";
	}else if(flag_==2){
    	message_ = "Number of iterations exceeded num_iter_max.";
	}else if(flag_==-1){
    	message_ = "Matrix may be singular. Step was NaN or Inf.";
	}else{
    	message_ = "Normal exit.";
    }

}

// get result vector
arma::Col<double> NewtonRaphson::get_result() const{
	return x_;
}

// get message
std::string NewtonRaphson::get_message() const{
	return message_;
}

// jacobian approximation at x
// this can be used as a replacement for the 
// jacobian function. It uses a center 
// difference approximation and should only 
// be used for small systems (not super efficient)
arma::Mat<double> NewtonRaphson::approximate_jacobian(
	const arma::Col<double> &x, const double dx, 
	const bool use_parallel, NRSysFun sysfn){

	// allocate jacbian matrix
	arma::Mat<double> J(x.n_elem,x.n_elem);

	// walk over degrees of freedom	
	//for(arma::uword i=0;i<x.n_elem;i++){
	parfor(0,x.n_elem,use_parallel,[&](int i, int) {
		arma::Col<double> delta(x.n_elem,arma::fill::zeros); 
		delta(i)+=dx;
		arma::Col<double> dF = sysfn(x+delta)-sysfn(x-delta);
		assert(dF.n_elem==x.n_elem);
		J.col(i) = dF/(2*dx);
	});

	// return jacobian matrix
	return J;
}

// display function
void NewtonRaphson::default_displayfun(
	const arma::uword n, const double r, const double s, 
	const double l, const double rc, const double c){
	// display header
	//if(n==1)std::printf("Niter, resnorm, stepnorm, lambda, rcond, convergence\n");
	if(n==1){
		std::printf("\nNewton Raphson Solver\n");
		std::printf("%10s %10s %10s %10s %10s %12s\n", 
			"iter", "resnorm", "stepnorm", "lambda", "rcond", "convergence");
		for(arma::uword n = 0;n<67;n++)std::printf("-");
		std::printf("\n");
	}

	// display iteration
	std::printf("%10llu %10.4g %10.4g %10.4g %10.4g %12.4g\n", n, r, s, l, rc, c);
}











