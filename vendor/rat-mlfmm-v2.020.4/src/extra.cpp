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
#include "extra.hh"

// code specific to Rat
namespace rat{namespace fmm{
		
	// create list fo nm-indices
	arma::Mat<int> Extra::nmlist(const int nmax){
		// allocate list
		arma::Mat<int> nmlist(polesize(nmax),2);

		// walk over nm and make list
		arma::uword idx = 0;
		for(int n=0;n<=nmax;n++){
			for(int m=-n;m<=n;m++){
				nmlist.row(idx++) = arma::Row<int>::fixed<2>{n,m};
			}
		}

		// return the list
		return nmlist;
	}

	// create multi-dimensional index lists
	arma::Row<arma::uword> Extra::expand_indices(
		const arma::Row<arma::uword> &indices, 
		const arma::uword num_dim){ 

		// compute
		const arma::Row<arma::uword> dim_indices = arma::vectorise(
			arma::repmat(num_dim*indices,num_dim,1).eval().each_col() + 
			arma::regspace<arma::Col<arma::uword> >(0,num_dim-1)).t();

		// return output
		return dim_indices;
	}

	// harmonics object indexing
	int Extra::nm2fidx(const int n, const int m){
		// check input
		assert(n>=0); assert(m>=-n); assert(m<=n);

		// return index into harmonic
		return n*(n+1)+m;
	}

	// harmonics object indexing
	int Extra::hfnm2fidx(const int n, const int m){
		// check input
		assert(n>=0); assert(m>=0); assert(m<=n);

		// return index into harmonic
		return n*(n+1)/2+m;
	}

	// size of harmonics object
	int Extra::polesize(const int nmax){
		return nm2fidx(nmax,nmax)+1;
	}

	// size of harmonics object
	arma::uword Extra::polesize(const arma::uword nmax){
		return nm2fidx(int(nmax),int(nmax))+1;
	}

	// size of fltp harmonics object
	int Extra::dbpolesize(const int nmax){
		return nm2fidx(2*nmax,2*nmax)+1;
	}

	// size of fltp harmonics object
	int Extra::hfpolesize(const int nmax){
		return hfnm2fidx(nmax,nmax)+1;
	}

	// Calculate list of factorials
	arma::Col<fltp> Extra::factorial(const int nfacs){

		// it is not possible to store 171! in a fltp
		#ifdef RAT_DOUBLE_PRECISION
		assert(nfacs<170); // for double
		#else
		assert(nfacs<33); // for fltp
		#endif
		assert(nfacs>=1);

		// allocate
		arma::Col<fltp> facs(nfacs+1);

		// fill ouput array recursively
		facs(0)=RAT_CONST(1.0); facs(1)=RAT_CONST(1.0);
		for(int i=2; i<=nfacs; i++){
			facs(i) = facs(i-1)*i;
		}

		// return output
		return facs;
	}

	// function to calculate Anm
	fltp Extra::Anm(const arma::Col<fltp> &facs, const int n, const int m){
		return std::pow(-RAT_CONST(1.0),n)/std::sqrt(facs(n-m)*facs(n+m));
	}

	// functions for calculating i^n
	std::complex<fltp> Extra::ipown(const int n){
		// allocate output
		std::complex<fltp> icx(0,RAT_CONST(1.0));
		//in.real(Extra::round(std::sinf(arma::Datum<fltp>::pi*fltp(n)/2+arma::Datum<fltp>::pi/2)));
		//in.imag(Extra::round(std::sinf(arma::Datum<fltp>::pi*fltp(n)/2)));
		return std::pow(icx,n);
	}

	// convert carthesian to spherical coordinates
	// in essence [x;y;z] to [rho;theta;phi] where
	// rho is distance to center, theta is angle with z-axis 
	// and phi is angle in xy-plane
	arma::Mat<fltp> Extra::cart2sph(
		const arma::Mat<fltp> &cartcoord){
		// check input
		assert(cartcoord.is_finite());

		// allocate output
		arma::Mat<fltp> sphcoord(3,cartcoord.n_cols);

		// calculate rho using vectorised pythagoras
		sphcoord.row(0) = arma::sqrt(arma::sum(cartcoord%cartcoord,0)); // rho
		arma::Mat<fltp> mu = cartcoord.row(2)/sphcoord.row(0); 

		// deal with zero rho
		mu(find(sphcoord.row(0)==0)).fill(0);

		// make sure that mu lies in the interval of -1 to 1
		mu.elem( arma::find(mu < -RAT_CONST(1.0)) ).fill(-RAT_CONST(1.0)); 
		mu.elem( arma::find(mu > RAT_CONST(1.0)) ).fill(RAT_CONST(1.0)); 

		// calculate theta
		sphcoord.row(1) = arma::acos(mu);

		// calculate phi
		sphcoord.row(2) = arma::atan2(cartcoord.row(1),cartcoord.row(0));

		// assert that there is no nans
		assert(sphcoord.is_finite());

		// return matrix
		return sphcoord;
	}

		// sparse matrix self multiplication
	// for complex sparse matrices
	arma::SpMat<std::complex<fltp> > Extra::matrix_self_multiplication(
		const arma::SpMat<std::complex<fltp> > &M, const arma::uword N){
		
		// make sure that N is larger than zero
		//assert(N>=0);

		// copy matrix
		arma::SpMat<std::complex<fltp> > Mnew = M;
		
		// multiply N-times
		for(arma::uword i=0;i<N;i++){
			Mnew *= Mnew; // check this
		}

		// output new matrix
		return Mnew;
	}

	// dense matrix self multiplication
	// for complex sparse matrices
	arma::Mat<std::complex<fltp> > Extra::matrix_self_multiplication(
		const arma::Mat<std::complex<fltp> > &M, const arma::uword N){
		
		// make sure that N is larger than zero
		//assert(N>=0);

		// copy matrix
		arma::Mat<std::complex<fltp> > Mnew = M;
		
		// multiply N-times
		for(arma::uword i=0;i<N;i++){
			Mnew *= Mnew; // check this
		}

		// output new matrix
		return Mnew;
	}


	// self inductance of wire
	// input is wire diameter in [m] and length in [m] output is inductance in [H] 
	// permeability is always 1.0 except for magnetic materials
	// equation source:
	// Transmission Line Design Handbook by Brian C Wadell, Artech House 1991 page 382 paragraph 6.2.1.1
	// Inductance Calculations, F. W. Grover, Dover Publications, 2004 .
	// http://www.g3ynh.info/zdocs/magnetics/appendix/Grover46/Grover_errata.pdf
	arma::Row<fltp> Extra::calc_wire_inductance(const arma::Row<fltp> &d, const arma::Row<fltp> &l, const fltp mu){
		// convert to [cm]
		const arma::Row<fltp> dcm = RAT_CONST(1e2)*d;
		const arma::Row<fltp> lcm = RAT_CONST(1e2)*l;

		// calculate in [nH]
		const arma::Row<fltp> L = 2*lcm%(
			arma::log(((2*lcm)/dcm)%(RAT_CONST(1.0)+arma::sqrt(RAT_CONST(1.0)+arma::square(dcm/(2*lcm))))) - 
			arma::sqrt(1.0+arma::square(dcm/(2*lcm))) + mu/4.0f + dcm/(2*lcm));

		// convert to [H] and return
		return RAT_CONST(1e-9)*L;
	}

	// self inductance of ribbon
	// input is width in [m], length in [m] and thickness in [m] output is inductance in [H]
	// equation source:
	// % Radio Engineers Handbook, McGraw-Hill, New York, 1945.
	arma::Row<fltp> Extra::calc_ribbon_inductance(const arma::Row<fltp> &w, const arma::Row<fltp> &l, const arma::Row<fltp> &t){
		// translate to cm
		const arma::Row<fltp> wcm = RAT_CONST(1e2)*w; 
		const arma::Row<fltp> lcm = RAT_CONST(1e2)*l;
		const arma::Row<fltp> tcm = RAT_CONST(1e2)*t;

		// calculate in [muH]
		const arma::Row<fltp> Lmh = RAT_CONST(2e-3)*lcm%(arma::log((RAT_CONST(2.0)*lcm)/(wcm+tcm))+RAT_CONST(0.5)+RAT_CONST(0.2235)*((wcm+tcm)/lcm));

		// convert to [H] and return
		return RAT_CONST(1e-6)*Lmh;
	}

	// display complex pole
	void Extra::display_pole(const int num_exp, const arma::Col<std::complex<fltp> > &M, const rat::cmn::ShLogPr &lg, const arma::uword num_digits){
		lg->msg("Real Values\n");
		display_pole(num_exp, arma::real(M),lg,num_digits);
		lg->msg("Imaginary Values\n");
		display_pole(num_exp, arma::imag(M),lg,num_digits);
	}

	// display pole
	void Extra::display_pole(const int num_exp, const arma::Col<fltp> &M, const rat::cmn::ShLogPr &lg, const arma::uword num_digits){
		// strings
		const std::string mstr = "m=%+0" + std::to_string(5+num_digits) + "d";
		const std::string nstr = "n=%03d";
		const std::string vstr = "%+1." + std::to_string(num_digits) + "e";

		// make table header
		lg->msg("      ");
		for(int m=-num_exp;m<=num_exp;m++){
			lg->msg(0,mstr.c_str(),m);
			lg->msg(0," ");
		}

		// next line
		lg->msg(0,"\n");

		// walk over n
		for(int n=0;n<=num_exp;n++){
			// line header
			lg->msg(nstr.c_str(),n);
			lg->msg(0," ");

			// leading whitespace
			for(int m=-num_exp;m<-n;m++){
				for(arma::uword i=0;i<7+num_digits;i++)
					lg->msg(0,".");
				lg->msg(0," ");
			}

			// walk over m
			for(int m=-n;m<=n;m++){
				lg->msg(0,vstr.c_str(),M(rat::fmm::Extra::nm2fidx(n,m)));
			}

			// trailing whitespace
			for(int m=n;m<num_exp;m++){
				for(arma::uword i=0;i<7+num_digits;i++)
					lg->msg(0,".");
				lg->msg(0," ");
			}

			// next line
			lg->msg(0,"\n");
		}

		// next line
		lg->msg("\n");
	}

	// display complex pole
	void Extra::display_pole(const int nmax, const int mmax, const arma::Col<std::complex<fltp> > &M, const rat::cmn::ShLogPr &lg, const arma::uword num_digits){
		lg->msg("Real Values\n");
		display_pole(nmax,mmax, arma::real(M),lg,num_digits);
		lg->msg("Imaginary Values\n");
		display_pole(nmax,mmax, arma::imag(M),lg,num_digits);
	}

	// display pole
	void Extra::display_pole(const int nmax, const int mmax, const arma::Col<fltp> &M, const rat::cmn::ShLogPr &lg, const arma::uword num_digits){
		
		// strings
		const std::string mstr = "m=%+0" + std::to_string(5+num_digits) + "d";
		const std::string nstr = "n=%03d";
		const std::string vstr = "%+1." + std::to_string(num_digits) + "e";

		// make table header
		lg->msg("      ");
		for(int m=-mmax;m<=mmax;m++){
			lg->msg(0,mstr.c_str(),m);
			lg->msg(0," ");
		}
		// next line
		lg->msg(0,"\n");

		// walk over n
		for(int n=0;n<=nmax;n++){
			// line header
			lg->msg(nstr.c_str(),n);
			lg->msg(0," ");

			// leading whitespace
			for(int m=-mmax;m<-std::min(n,mmax);m++){
				for(arma::uword i=0;i<7+num_digits;i++)
					lg->msg(0,".");
				lg->msg(0," ");
			}

			// walk over m
			for(int m=-std::min(n,mmax);m<=std::min(n,mmax);m++){
				if(n==0 && m==0)lg->msg(0,"%s",KYEL);
				lg->msg(0,vstr.c_str(),M(rat::fmm::Extra::nm2fidx(n,m)));
				lg->msg(0," ");
				if(n==0 && m==0)lg->msg(0,"%s",KNRM);
			}

			// trailing whitespace
			for(int m=std::min(n,mmax);m<mmax;m++){
				for(arma::uword i=0;i<7+num_digits;i++)
					lg->msg(0,".");
				lg->msg(0," ");
			}

			// next line
			lg->msg(0,"\n");
		}

		// next line
		lg->msg("\n");
	}


}}