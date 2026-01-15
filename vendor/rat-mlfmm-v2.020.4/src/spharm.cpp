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
#include "spharm.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// construction without calculation
	Spharm::Spharm(const int num_exp){
		num_exp_ = num_exp;
	}

	// regular constructor
	Spharm::Spharm(const int num_exp, const arma::Mat<fltp> &sphcoord){
		// check supplied coordinates
		assert(sphcoord.n_rows==3);
		
		// set nmax
		num_exp_ = num_exp;

		// calculate
		calculate(sphcoord);
	}

	// method for calculating the normal spherical harmonics
	void Spharm::calculate(const arma::Mat<fltp> &sphcoord){

		// check input
		assert(arma::all(sphcoord.row(1)>=0));
		assert(arma::all(sphcoord.row(1)<=arma::Datum<fltp>::pi));

		// set number of harmonics
		num_harm_ = int(sphcoord.n_cols);

		// create factorial list
		const arma::Col<fltp> facs = Extra::factorial(2*num_exp_);

		// check if factorial list has sufficient length
		assert(2*num_exp_+1==static_cast<int>(facs.n_elem));

		// allocate output
		Y_.set_size(Extra::polesize(num_exp_),num_harm_);

		// allocate output
		arma::Mat<fltp> Pr(Extra::polesize(num_exp_),num_harm_); // legendre polynomial

		// set initial values
		Pr.row(0).fill(RAT_CONST(1.0));
		Y_.row(0).fill(RAT_CONST(1.0));
		
		// get mu
		arma::Row<fltp> mu = arma::cos(sphcoord.row(1)); // cos(theta)

		// // calculation variables
		fltp dbfac = RAT_CONST(1.0);
		arma::Row<fltp> sqr = -arma::sqrt(RAT_CONST(1.0)-(mu%mu));
		arma::Row<fltp> sqcr(num_harm_,arma::fill::ones);

		// precalculate sin and cosines
		arma::Mat<std::complex<fltp> > sincos_phi_m(num_exp_+1, num_harm_);
		for(int m=0;m<=num_exp_;m++)
			sincos_phi_m.row(m) = arma::Row<std::complex<fltp> >(
				arma::cos(sphcoord.row(2)*m), arma::sin(sphcoord.row(2)*m));

		// iterate over levels
		for(int n=1; n<=num_exp_; n++){
			// this part calculates the Legendre polynomials
			// core (eq. 2.1.54, p48)
			for(int m=0; m < n-1; m++){
				Pr.row(Extra::nm2fidx(n,m)) = 
					(mu*(fltp(2*n-1)/(n-m)))%Pr.row(Extra::nm2fidx(n-1, m))
					-(fltp(n+m-1)/(n-m)) * Pr.row(Extra::nm2fidx(n-2, m));
			}

			// inner diagonal (eq. 2.1.52, p48)
			Pr.row(Extra::nm2fidx(n,n-1)) = fltp(2*n-1)*mu%Pr.row(Extra::nm2fidx(n-1,n-1));

			// diagonal (eq. 2.1.51, p47)
			dbfac *= fltp(2*n - 1);
			sqcr %= sqr;
			Pr.row(Extra::nm2fidx(n,n)) = dbfac*sqcr;

			// spherical harmonics calculation (eq. 2.1.59, p49)
			for(int m=0; m<=n; m++){
				const fltp sqfac = std::sqrt(facs(n-m)/facs(n+m)); // counter-acted by Anm?

				// insert row
				Y_.row(Extra::nm2fidx(n,m)) = sqfac*Pr.row(Extra::nm2fidx(n,m))%sincos_phi_m.row(m);
			}

			// mirror
			for(int m=1; m<=n; m++){
				Y_.row(Extra::nm2fidx(n,-m)) = arma::conj(Y_.row(Extra::nm2fidx(n,m)));
			}
		}
	}

	// subtraction
	Spharm Spharm::operator-(const Spharm &otherharmonic){
		// check sizes
		assert(Y_.n_cols==otherharmonic.Y_.n_cols);
		assert(Y_.n_rows==otherharmonic.Y_.n_rows);

		// create output harmonic
		Spharm out(num_exp_);
		out.num_harm_ = num_harm_;

		// perform subtraction
		out.Y_ = Y_ - otherharmonic.Y_;

		// return new harmonic
		return out;
	}

	// division by scalar
	Spharm Spharm::operator/(const fltp &div){
			
		// create output harmonic
		Spharm out(num_exp_);
		out.num_harm_ = num_harm_;

		// perform subtraction
		out.Y_ = Y_/div;

		// return new harmonic
		return out;
	}

	// division by vector
	void Spharm::divide(const arma::Row<fltp> &div){
		// check sizes
		assert(Y_.n_cols==div.n_elem);

		// perform subtraction
		Y_.each_row() /= arma::conv_to<arma::Row<std::complex<fltp> > >::from(div);
	}

	// get data from harmonic
	std::complex<fltp> Spharm::get_nm(const int i, const int n, const int m) const{
		// check whether position is valid
		assert(n>=0); assert(n<=num_exp_);
		assert(m>=-n); assert(m<=n);

		// get value from stored data
		std::complex<fltp> Ynm = Y_(Extra::nm2fidx(n,m),i);

		// return value
		return Ynm;
	}

	// get data from harmonic
	arma::Row<std::complex<fltp> > Spharm::get_nm(const int n, const int m) const{
		// check whether position is valid
		assert(n>=0); assert(n<=num_exp_);
		assert(m>=-n); assert(m<=n);

		// get value from stored data
		arma::Row<std::complex<fltp> > Ynm = Y_.row(Extra::nm2fidx(n,m));

		// return value
		return Ynm;
	}


	// get data from harmonic
	const arma::Mat<std::complex<fltp> >& Spharm::get_all() const{
		return Y_;
	}

	// display stored values in the terminal
	void Spharm::display(const cmn::ShLogPr &lg) const{
		for (int i=0;i<num_harm_;i++){
			display(i,lg);
		}
	}

	// display stored values in the terminal
	void Spharm::display(const int i, const cmn::ShLogPr &lg) const{
		// check if length of Y matches length of num_exp
		assert(Extra::polesize(num_exp_)==static_cast<int>(Y_.n_rows));

		// header displaying which harmonic
		if(num_harm_>1){
			int boxwidth = 5+(2*num_exp_+1)*(6+1);
			for (int j=0;j<boxwidth;j++){lg->msg(0,"-");}; lg->msg(0,"\n");
			lg->msg("Spherical harmonic with index %03d/%03d\n",i,num_harm_);
		}

		// display harmonic
		Extra::display_pole(num_exp_,Y_.col(i),lg);
	}

}}
