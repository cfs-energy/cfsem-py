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

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	GMRES::GMRES(){

	}

	// factory
	ShGMRESPr GMRES::create(){
		return std::make_shared<GMRES>();
	}

	// setting of system function
	void GMRES::set_systemfun(GmresSysFun fn){
		systemfun_ = fn;
	}

	// setting of preconditioner function
	void GMRES::set_precfun(GmresPrecFun fn){
		precfun_ = fn;
	}

	// set tolerance
	void GMRES::set_tol(const fltp tol){
		if(tol<=0)rat_throw_line("tolerance must be larger than zero");
		tol_ = tol;
	}		
			
	// set maximum number of iterations
	void GMRES::set_num_iter_max(const arma::uword num_iter_max){
		if(num_iter_max<=0)rat_throw_line("maximum number of iterations must be larger than zero");
		num_iter_max_ = num_iter_max;
	}
			
	// set number of iterations before restart
	void GMRES::set_num_restart(const arma::uword num_restart){
		if(num_restart<=0)rat_throw_line("number of iterations before restart must be larger than zero");
		num_restart_ = num_restart;
	}

	// solve system (x is output but also contains initial guess)
	void GMRES::solve(arma::Col<fltp> &x, const arma::Col<fltp> &b, ShLogPr lg){
		// print header to log
		lg->msg(2,"%s%sGMRES NEWTON-KRYLOV SOLVER%s\n",KGRN,KBLD,KNRM);

		// some checking
		if(precfun_==NULL)rat_throw_line("preconditioner not set");
		if(systemfun_==NULL)rat_throw_line("system function not set");
		if(x.n_rows!=b.n_rows)rat_throw_line("solution vector and right hand side are not equal in length");
		if(!x.is_finite())rat_throw_line("initial guess is not finite");
		if(!b.is_finite())rat_throw_line("right hand side is not finite");

		// counters
		const arma::uword num_equations = b.n_rows;

		// display settings
		lg->msg(2,"%ssettings%s\n",KBLU,KNRM);
		lg->msg("number of equations: %s%04llu%s\n",KYEL,num_equations,KNRM);
		lg->msg("solution tolerance: %s%8.2e%s\n",KYEL,tol_,KNRM);
		lg->msg("max number of iter: %s%04llu%s\n",KYEL,num_iter_max_,KNRM);
		lg->msg("num iter for restart: %s%04llu%s\n",KYEL,num_restart_,KNRM);
		lg->msg(-2,"\n");

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

		// header
		lg->msg(2,"%srunning solver%s\n",KBLU,KNRM);
		
		// check if initial solution acceptable
		lg->msg("check initial\n");
		arma::Col<fltp> r = precfun_(b - systemfun_(x));
		if(r.n_elem!=num_equations)rat_throw_line("precfun output inconsistent with number of equations");
		if(!r.is_finite())rat_throw_line("precfun output is not finite");
		fltp beta = arma::norm(r);
		relres_ = beta / normb_;
		num_iter_ = 0;
		if (relres_ <= tol_) {
			// tol = relres_;
			flag_ = 0;
			return;
		}

		// table header
		lg->msg("%s%4s %8s%s\n",KBLD,"iter","resnorm",KNRM);

		// iterations
		arma::uword j = 1;
		while (j <= num_iter_max_){
			v_(0) = r * (1.0f / beta);    // ??? r / beta
			//s = 0.0;
			s_.zeros();
			s_(0) = beta;
			
			// inner loop
			for (arma::uword i = 0; i < num_restart_ && j <= num_iter_max_; i++, j++){
				// run system function
				arma::Col<fltp> w = precfun_(systemfun_(v_(i)));
				if(w.n_elem!=num_equations)rat_throw_line("precfun output inconsistent with number of equations");
				if(!w.is_finite())rat_throw_line("precfun output is not finite");
				
				// process
				for (arma::uword k = 0; k <= i; k++) {
					H_(k, i) = arma::dot(w, v_(k));
					w -= H_(k, i) * v_(k);
				}
				H_(i+1, i) = arma::norm(w);
				v_(i+1) = w * (1.0 / H_(i+1, i)); // ??? w / H(i+1, i)

				// apply rotations
				for (arma::uword k = 0; k < i; k++){
					apply_plane_rot(H_(k,i), H_(k+1,i), cs_(k), sn_(k));
				}
				gen_plane_rot(H_(i,i), H_(i+1,i), cs_(i), sn_(i));
				apply_plane_rot(H_(i,i), H_(i+1,i), cs_(i), sn_(i));
				apply_plane_rot(s_(i), s_(i+1), cs_(i), sn_(i));
				relres_ = std::abs(s_(i+1)) / normb_;
				num_iter_ = j;

				// report
				lg->msg("%04llu %08.2e\n",num_iter_,relres_);

				// check tolerance
				if (relres_ < tol_) {
					update(x, int(i));
					// tol = relres_;
					
					flag_ = 0;
					return;
				}
			}
			update(x, int(num_restart_) - 1);
			r = precfun_(b - systemfun_(x));
			beta = arma::norm(r);
			relres_ = beta / normb_;
			num_iter_ = j;
			
			// report
			lg->msg("<< restart >>\n");

			// check tolerance
			if (relres_ < tol_){
				flag_ = 0;
				return;
			}
		}
		
		num_iter_ = j;

		// solver done
		lg->msg(-2,"\n");
		lg->msg(-2);
		return;
	}


	// backsolve function: H_*y = x
	// probably better blas version available (need to check)
	void GMRES::update(arma::Col<fltp> &x, const int k){
		// create solution vector
		arma::Col<fltp> y(s_);

		// Backsolve:  
		for(int i=k;i>=0;i--){
			y(i)/=H_(i,i);
			for(int j = i-1;j>=0;j--){
				y(j)-=H_(j,i)*y(i);
			}
		}

		for(int j=0;j<=k;j++){
		 	x+=v_(j)*y(j);
		}
	}


	// plane rotation functions
	void GMRES::gen_plane_rot(fltp &dx, fltp &dy, fltp &cs, fltp &sn){
		if(dy == 0.0f){
			cs = 1.0f;
			sn = 0.0f;
		}else if(abs(dy) > abs(dx)){
			fltp temp = dx / dy;
			sn = 1.0f / sqrt( 1.0f + temp*temp );
			cs = temp * sn;
		}else{
			fltp temp = dy / dx;
			cs = 1.0f / sqrt( 1.0f + temp*temp );
			sn = temp * cs;
		}
	}
	void GMRES::apply_plane_rot(fltp &dx, fltp &dy, fltp &cs, fltp &sn){
		fltp temp  =  cs * dx + sn * dy;
		dy = -sn * dx + cs * dy;
		dx = temp;
	}

    fltp GMRES::get_residual(){
        return relres_;
    }

    arma::uword GMRES::get_num_iter(){
        return num_iter_;
    }

}}
