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
#include "gmres.hh"

// constructor
GMRES::GMRES(){

}

// factory
ShGMRESPr GMRES::create(){
	return std::make_shared<GMRES>();
}

// setting of system function
void GMRES::set_systemfun(
	std::function<arma::Col<double>(const arma::Col<double>) > fn){
	// set system function
	systemfun_ = fn;
}

// setting of preconditioner function
void GMRES::set_precfun(
	std::function<arma::Col<double>(const arma::Col<double>) > fn){
	// set system function
	precfun_ = fn;
}

// setting of preconditioner function
void GMRES::set_reportfun(
	std::function<void(int,double) > fn){
	// set system function
	reportfun_ = fn;
}

// set tolerance
void GMRES::set_tol(const double tol){
	assert(tol>0);
	tol_ = tol;
}		
		
// set maximum number of iterations
void GMRES::set_num_iter_max(const arma::uword num_iter_max){
	assert(num_iter_max>0);
	num_iter_max_ = num_iter_max;
}
		
// set number of iterations before restart
void GMRES::set_num_restart(const arma::uword num_restart){
	assert(num_restart>0);
	num_restart_ = num_restart;
}

// report function
void GMRES::iter_report() const{
	// check if report function was set
	if(reportfun_!=nullptr){
		// forward iteration and residual to report function
		reportfun_(num_iter_,relres_);
	}
}


// solve system (x is output but also contains initial guess)
void GMRES::solve(arma::Col<double> &x, const arma::Col<double> &b){
	
	// some checking
	assert(x.n_rows==b.n_rows);
	assert(!x.has_nan());
	assert(!b.has_nan());
	assert(systemfun_!=nullptr);
	assert(precfun_!=nullptr);

	// set output flag to the default 1
	flag_ = 1;

	// allocate
	H_.zeros(num_restart_+1,num_restart_);
	s_.set_size(num_restart_+1);
	cs_.set_size(num_restart_+1);
	sn_.set_size(num_restart_+1);
	v_.set_size(num_restart_+1);

	// calculate the norm of b to check with the result
	normb_ = arma::norm(precfun_(b));
	if(normb_ == 0.0)normb_ = 1;

	// check if initial solution acceptable
	arma::Col<double> r = precfun_(b - systemfun_(x));
	double beta = arma::norm(r);
	relres_ = beta / normb_;
	num_iter_ = 0;
	iter_report();
	if (relres_ <= tol_) {
		// tol = relres_;
		flag_ = 0;
		return;
	}

	// iterations
	arma::uword j = 1;
	while (j <= num_iter_max_) {
		v_(0) = r * (1.0 / beta);    // ??? r / beta
		//s = 0.0;
		s_.zeros();
		s_(0) = beta;
		
		// inner loop
		for (arma::uword i = 0; i < num_restart_ && j <= num_iter_max_; i++, j++) {
			arma::Col<double> w = precfun_(systemfun_(v_(i)));
			for (arma::uword k = 0; k <= i; k++) {
				H_(k, i) = arma::dot(w, v_(k));
				w -= H_(k, i) * v_(k);
			}
			H_(i+1, i) = arma::norm(w);
			v_(i+1) = w * (1.0 / H_(i+1, i)); // ??? w / H(i+1, i)

			for (arma::uword k = 0; k < i; k++){
				apply_plane_rot(H_(k,i), H_(k+1,i), cs_(k), sn_(k));
			}

			gen_plane_rot(H_(i,i), H_(i+1,i), cs_(i), sn_(i));
			apply_plane_rot(H_(i,i), H_(i+1,i), cs_(i), sn_(i));
			apply_plane_rot(s_(i), s_(i+1), cs_(i), sn_(i));
			relres_ = std::abs(s_(i+1)) / normb_;
			num_iter_ = j;
			iter_report();
			if (relres_ < tol_) {
				update(x, i);
				// tol = relres_;
				
				flag_ = 0;
				return;
			}
		}
		update(x, num_restart_ - 1);
		r = precfun_(b - systemfun_(x));
		beta = arma::norm(r);
		relres_ = beta / normb_;
		num_iter_ = j;
		iter_report();
		if (relres_ < tol_) {
			// tol = relres_;
			
			flag_ = 0;
			return;
		}
	}
	
	num_iter_ = j;
	return;
}


// backsolve function: H_*y = x
// probably better blas version available (need to check)
void GMRES::update(arma::Col<double> &x, const int k){
	// create solution vector
	arma::Col<double> y(s_);

	// Backsolve:  
	for (int i = k; i >= 0; i--) {
		y(i) /= H_(i,i);
		for (int j = i - 1; j >= 0; j--){
			y(j) -= H_(j,i) * y(i);
		}
	}

	for (int j = 0; j <= k; j++){
	 	x += v_(j) * y(j);
	}
}


// plane rotation functions
void GMRES::gen_plane_rot(double &dx, double &dy, double &cs, double &sn){
	if (dy == 0.0) {
		cs = 1.0;
		sn = 0.0;
	} else if (abs(dy) > abs(dx)) {
		double temp = dx / dy;
		sn = 1.0 / sqrt( 1.0 + temp*temp );
		cs = temp * sn;
	} else {
		double temp = dy / dx;
		cs = 1.0 / sqrt( 1.0 + temp*temp );
		sn = temp * cs;
	}
}
void GMRES::apply_plane_rot(double &dx, double &dy, double &cs, double &sn){
	double temp  =  cs * dx + sn * dy;
	dy = -sn * dx + cs * dy;
	dx = temp;
}

// display function
void GMRES::display() const{
	std::printf("GMRES:\n");
	std::printf("  output flag = %i\n",flag_);
	std::printf("  iterations performed = %llu\n",num_iter_);
	std::printf("  tolerance achieved = %6.3e\n\n",relres_);
}
