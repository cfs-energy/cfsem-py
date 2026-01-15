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
#include "stmat.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// default constructor
	StMat_So2Mp_J::StMat_So2Mp_J(){

	}

	// constructor
	StMat_So2Mp_J::StMat_So2Mp_J(const int num_exp, const arma::Mat<fltp> &dR){
		num_exp_ = num_exp;	
		calc_matrix(dR);
	}

	// factory
	ShStMat_So2Mp_JPr StMat_So2Mp_J::create(){
		//return ShIListPr(new IList);
		return std::make_shared<StMat_So2Mp_J>();
	}


	// set number of expansions
	void StMat_So2Mp_J::set_num_exp(const int num_exp){
		assert(num_exp>0);
		num_exp_ = num_exp;
	}

	// set number of expansions
	void StMat_So2Mp_J::set_num_exp(const arma::uword num_exp){
		assert(num_exp>0);
		num_exp_ = static_cast<int>(num_exp);
	}

	// matrix setup
	void StMat_So2Mp_J::calc_matrix(const arma::Mat<fltp> &dR){
		
		// check input
		assert(num_exp_>0);
		assert(dR.n_rows==3);

		// set number of sources
		num_sources_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		const arma::Mat<fltp> dRsph = Extra::cart2sph(dR);

		// calculate harmonic
		Spharm L(num_exp_,dRsph);

		// set matrix array
		M_.set_size(Extra::polesize(num_exp_),num_sources_);

		// walk over elements of harmonic
		for(int n=0;n<=num_exp_;n++){
			// calculate rho^n
			const arma::Mat<fltp> rhon = arma::pow(dRsph.row(0),n);
			for(int m=-n;m<=n;m++){
				// calculate and store in matrix
				M_.row(Extra::nm2fidx(n,m)) = 
				rhon%L.get_nm(n,-m);
			}
		}

		// check for nans
		assert(M_.is_finite());
	}

	// merge matrix typically when using gauss points
	void StMat_So2Mp_J::merge_matrix(
		const arma::Row<fltp>&weights, 
		const arma::uword num_stride){
		assert(M_.is_finite());
		assert(weights.is_finite());
		assert(weights.n_cols==M_.n_cols);

		// apply weights
		for(arma::uword i=0;i<M_.n_rows;i++)
			for(arma::uword j=0;j<M_.n_cols;j++)
				M_(i,j)*=weights(j);

		// add values
		M_ = arma::reshape(
			arma::sum(arma::reshape(M_.st(),num_stride,M_.n_rows*M_.n_cols/num_stride),0),
			M_.n_cols/num_stride,M_.n_rows).st();

		// update number of sources
		num_sources_/=num_stride;

		// check
		assert(M_.is_finite());
	}

	// apply to point sources
	arma::Mat<std::complex<fltp> > StMat_So2Mp_J::apply(
		const arma::Mat<fltp> &Ieff) const{

		// check input
		//assert(Ieff.n_rows==3);
		assert(Ieff.n_cols==num_sources_);
		assert(!M_.is_empty());

	 	// calculate multipole
	 	return M_*(Ieff.t());
	}

	// apply to point sources
	arma::Mat<std::complex<fltp> > StMat_So2Mp_J::apply(
		const arma::Mat<fltp> &Ieff, 
		const arma::Mat<arma::uword> &indices) const{
		
		// check input
		// assert(Ieff.n_rows==3);
		assert(!M_.is_empty());

	 	// calculate multipole
	 	return M_.cols(indices)*(Ieff.t());
	}

	// apply to point sources
	arma::Mat<std::complex<fltp> > StMat_So2Mp_J::apply(
		const arma::Mat<fltp> &Ieff, 
		const arma::uword first_index,
		const arma::uword last_index) const{

		// check input
		// assert(Ieff.n_rows==3);
		assert(!M_.is_empty());

	 	// calculate multipole
	 	return M_.cols(first_index,last_index)*(Ieff.t());
	}

	// check if it was setup
	bool StMat_So2Mp_J::is_empty() const{
		return M_.is_empty();
	}	

	// display the values stored in the matrix
	void StMat_So2Mp_J::display() const{
		// header
		printf("Source to Multipole matrix for currents\n");
			
		// display matrix
		cmn::Extra::display_mat(M_);
	}



	// default constructor
	StMat_So2Mp_M::StMat_So2Mp_M(){

	}

	// constructor
	StMat_So2Mp_M::StMat_So2Mp_M(const int num_exp, const arma::Mat<fltp> &dR){
		num_exp_ = num_exp;	
		calc_matrix(dR);
	}

	// factory
	ShStMat_So2Mp_MPr StMat_So2Mp_M::create(){
		//return ShIListPr(new IList);
		return std::make_shared<StMat_So2Mp_M>();
	}

	// set number of expansions
	void StMat_So2Mp_M::set_num_exp(const int num_exp){
		assert(num_exp>0);
		num_exp_ = num_exp;
	}

	// set number of expansions
	void StMat_So2Mp_M::set_num_exp(const arma::uword num_exp){
		assert(num_exp>0);
		num_exp_ = static_cast<int>(num_exp);
	}

	// matrix setup
	void StMat_So2Mp_M::calc_matrix(const arma::Mat<fltp> &dR){
		// check input
		assert(dR.n_rows==3);
		assert(num_exp_>0);
		assert(dR.is_finite());

		// set number of matrices
		num_sources_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		const arma::Mat<fltp> dRsph = Extra::cart2sph(dR);

		// calculate harmonic
		Spharm L(num_exp_,dRsph);

		// calculate factorials
		const arma::Col<fltp> facs = Extra::factorial(2*num_exp_);
		
		// allocate matrix with zeros
		M_.zeros(Extra::polesize(num_exp_),3*num_sources_);

		// allocate indexlist
		arma::Mat<int> jknm(3*Extra::polesize(num_exp_-1),4);

		// get indices
		// walk over target pole
		arma::uword p = 0; 
		for (int j=1;j<=num_exp_;j++){
			for (int k=-j;k<=j;k++){
				// walk over source pole
				int n=j-1;
				for (int m=std::max(-n,k-j+n);m<=std::min(n,k+j-n);m++){
					// set indices to list
					jknm(p,0) = j; jknm(p,1) = k; 
					jknm(p,2) = n; jknm(p,3) = m;
					p++;

					// for debugging
					//std::cout<<"j="<<j<<", k="<<k<<", n="<<n<<", m="<<m<<std::endl;
					//std::cout<<"j-n="<<j-n<<", k-m="<<k-m<<std::endl;
				}
			}
		}
		
		// sanity check
		//std::cout<<jknm.n_rows<<std::endl;
		assert(p==jknm.n_rows);

		// run over indices and calculate in parallel
		cmn::parfor(0, jknm.n_rows, use_parallel_,[&](arma::uword i, int) { // second int is CPU number
			// get indices
			const int j = jknm(i,0); const int k = jknm(i,1); 
			const int n = jknm(i,2); const int m = jknm(i,3);

			// calculate values for matrix
			// translation factor
			const arma::Mat<std::complex<fltp> > cst = 
				Extra::Anm(facs,n,m)*
				Extra::Anm(facs,j-n,k-m)*
				Extra::ipown(std::abs(k)-std::abs(m)-std::abs(k-m))*
				arma::pow(dRsph.row(0),n)%
				L.get_nm(n,-m)/
				Extra::Anm(facs,j,k);

			// insert into matrix (note: j-n = 1)
			const arma::uword tid = Extra::nm2fidx(j,k);
			const arma::uword sid = Extra::nm2fidx(1,k-m)-1;
			assert(sid<3);
			M_.submat(arma::span(tid,tid),
				arma::span(sid*num_sources_,
				(sid+1)*num_sources_-1)) = cst;
		});

		// calculate mini multipoles for each source
		Mconv_ = get_mini_multipole_matrix();
	}

	// get conversion matrix
	arma::Mat<std::complex<fltp> > StMat_So2Mp_M::get_matrix(const arma::uword k) const{
		return M_.cols(arma::Row<arma::uword>::fixed<3>{k,k+num_sources_,k+2*num_sources_});
	}

	// apply to point sources
	arma::Mat<std::complex<fltp> > StMat_So2Mp_M::apply(
		const arma::Mat<fltp> &Meff) const{

		// check sizes
		assert(Meff.n_cols==num_sources_);
		assert(!M_.is_empty());
		assert(!Mconv_.is_empty());

	 	// calculate multipole
	 	return M_*arma::reshape((Mconv_*Meff).st(),3*Meff.n_cols,3);
	}

	// apply to point sources with indices
	arma::Mat<std::complex<fltp> > StMat_So2Mp_M::apply(
		const arma::Mat<fltp> &Meff,
		const arma::Mat<arma::uword> &indices) const{

		// check sizes
		assert(Meff.n_cols==indices.n_cols);
		assert(!M_.is_empty());
		assert(Meff.n_rows==3);
		assert(!Mconv_.is_empty());

	 	// calculate multipole
	 	return M_.cols(arma::join_horiz(arma::join_horiz(
	 		indices,indices+num_sources_),indices+2*num_sources_))*
	 		arma::reshape((Mconv_*Meff).st(),3*Meff.n_cols,3);
	}

	// method for calculating multipoles at element coordinates
	// these multipoles only contain n=1 and m=-1,0,+1
	arma::Mat<std::complex<fltp> >::fixed<9,3> StMat_So2Mp_M::get_mini_multipole_matrix(){
		// create complex 0 and 1
		const std::complex<fltp> icmplx(0,RAT_CONST(1.0));
		const std::complex<fltp> rcmplx(RAT_CONST(1.0),0);
		const fltp sqrt2 = std::sqrt(RAT_CONST(2.0));
		
		// setup matrix
		arma::Mat<std::complex<fltp> >::fixed<9,3> Mconv;

		// x multipole, n = 1, m = -1
		Mconv.row(0) = arma::Row<std::complex<fltp> >::fixed<3>{0,0,+0.5f*sqrt2*icmplx};

		// x multipole, n = 1, m = 0
		Mconv.row(1) = arma::Row<std::complex<fltp> >::fixed<3>{0,RAT_CONST(1.0)*rcmplx,0};

		// x multipole, n = 1, m = +1
		Mconv.row(2) = arma::Row<std::complex<fltp> >::fixed<3>{0,0,-0.5f*sqrt2*icmplx};

		// y multipole, n = 1, m = -1
		Mconv.row(3) = arma::Row<std::complex<fltp> >::fixed<3>{0,0,-0.5f*sqrt2*rcmplx};

		// y multipole, n = 1, m = 0
		Mconv.row(4) = arma::Row<std::complex<fltp> >::fixed<3>{-RAT_CONST(1.0)*rcmplx,0,0};

		// y multipole, n = 1, m = +1
		Mconv.row(5) = arma::Row<std::complex<fltp> >::fixed<3>{0,0,-0.5f*sqrt2*rcmplx};

		// z multipole, n = 1, m = -1
		Mconv.row(6) = arma::Row<std::complex<fltp> >::fixed<3>{-0.5f*sqrt2*icmplx,0.5f*sqrt2*rcmplx,0};

		// z multipole, n = 1, m = 0
		Mconv.row(7) = arma::Row<std::complex<fltp> >::fixed<3>{0,0,0};

		// z multipole, n = 1, m = +1
		Mconv.row(8) = arma::Row<std::complex<fltp> >::fixed<3>{+0.5f*sqrt2*icmplx,0.5f*sqrt2*rcmplx,0};

		// return matrix
		return Mconv;
	}

	// check if it was setup
	bool StMat_So2Mp_M::is_empty() const{
		return M_.is_empty();
	}	

	// direct setting of the matrix
	void StMat_So2Mp_M::set_M(const arma::uword num_sources, const arma::Mat<std::complex<fltp> > &M, const int num_exp){
		assert(Extra::polesize(num_exp)==static_cast<int>(M.n_rows));
		assert(3*num_sources==M.n_cols);
		M_ = M; num_exp_ = num_exp; num_sources_ = num_sources;
	}

	// direct access to matrix
	arma::Mat<std::complex<fltp> > StMat_So2Mp_M::get_M() const{
		return M_;
	}

	// set parallelism
	void StMat_So2Mp_M::set_use_parallel(const bool use_parallel){
		use_parallel_ = use_parallel;
	}

	// display the values stored in the matrix
	void StMat_So2Mp_M::display() const{
		// header
		printf("Source to Multipole matrix for magnetic moments\n");
			
		// display matrix
		cmn::Extra::display_mat(M_);
	}



	// default constructor
	StMat_Mp2Ta::StMat_Mp2Ta(){

	}

	// constructor
	StMat_Mp2Ta::StMat_Mp2Ta(const int num_exp, const arma::Mat<fltp> &dR){
		num_exp_ = num_exp;	
		calc_matrix(dR);
	}

	// factory
	ShStMat_Mp2TaPr StMat_Mp2Ta::create(){
		//return ShIListPr(new IList);
		return std::make_shared<StMat_Mp2Ta>();
	}

	// set number of expansions
	void StMat_Mp2Ta::set_num_exp(const int num_exp){
		assert(num_exp>0);
		num_exp_ = num_exp;
	}

	// set number of expansions
	void StMat_Mp2Ta::set_num_exp(const arma::uword num_exp){
		assert(num_exp>0);
		num_exp_ = static_cast<int>(num_exp);
	}

	// calculate vector potential from multipole
	// required for intermediate checking and barnes hut algorithm
	// also known as treecode with O(NlogN) complexity
	void StMat_Mp2Ta::calc_matrix(const arma::Mat<fltp> &dR){
		// check input
		assert(num_exp_>0);
		assert(dR.n_rows==3);

		// set number of targets
		num_targets_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		const arma::Mat<fltp> dRsph = Extra::cart2sph(dR);

		// calculate harmonic
		Spharm L(num_exp_,dRsph);
		
		// set matrix array
		M_.set_size(num_targets_,Extra::polesize(num_exp_));

		// for multipole
		for(int n=0;n<=num_exp_;n++){
			const arma::Mat<fltp> rhon = arma::pow(dRsph.row(0),n+1);
			for(int m=-n;m<=n;m++){
				M_.col(Extra::nm2fidx(n,m)) = L.get_nm(n,m).st()/rhon.t();
			}
		}

		// check for nans
		assert(M_.is_finite());
	}

	// apply to multipole
	arma::Mat<fltp> StMat_Mp2Ta::apply(const arma::Mat<std::complex<fltp> > &Mp) const{
		// check if it is a single multipole
		assert(Extra::polesize(num_exp_)==static_cast<int>(Mp.n_rows)); assert(3==Mp.n_cols);

		// get real part and return transpose
		const arma::Mat<fltp> A = arma::real(M_*Mp);

		// return A
		return A.t();
	}

	// check if it was setup
	bool StMat_Mp2Ta::is_empty() const{
		return M_.is_empty();
	}	

	// display the values stored in the matrix
	void StMat_Mp2Ta::display() const{
		// header
		printf("Multipole to Target matrix\n");
		
		// display matrix
		cmn::Extra::display_mat(M_);
	}

	// default constructor
	StMat_Lp2Ta::StMat_Lp2Ta(){

	}

	// constructor
	StMat_Lp2Ta::StMat_Lp2Ta(const int num_exp, const arma::Mat<fltp> &dR){
		num_exp_ = num_exp;	
		calc_matrix(dR);
	}

	// factory
	ShStMat_Lp2TaPr StMat_Lp2Ta::create(){
		//return ShIListPr(new IList);
		return std::make_shared<StMat_Lp2Ta>();
	}

	// set number of expansions
	void StMat_Lp2Ta::set_num_exp(const int num_exp){
		assert(num_exp>0);
		num_exp_ = num_exp;
	}

	// set number of expansions
	void StMat_Lp2Ta::set_num_exp(const arma::uword num_exp){
		assert(num_exp>0);
		num_exp_ = static_cast<int>(num_exp);
	}

	// calculate vector potential from multipole
	// required for intermediate checking and barnes hut algorithm
	// also known as treecode with O(NlogN) complexity
	void StMat_Lp2Ta::calc_matrix(const arma::Mat<fltp> &dR){
		// check input
		assert(num_exp_>0);
		assert(dR.n_rows==3);
		assert(dR.is_finite());

		// set number of targets
		num_targets_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		const arma::Mat<fltp> dRsph = Extra::cart2sph(dR);

		// calculate harmonic
		Spharm L(num_exp_,dRsph);

		// set matrix array
		M_.set_size(num_targets_,Extra::polesize(num_exp_));
		
		// walk over localpole
		for(int j=0; j<=num_exp_;j++){
			const arma::Mat<fltp> rhoj = arma::pow(dRsph.row(0),j);
			for(int k=-j; k<=j; k++){
				// calculate and insert into matrix
				M_.col(Extra::nm2fidx(j,k)) = L.get_nm(j,k).st()%rhoj.t();
			}
		}

		// check for nans
		assert(!M_.has_nan());
	}

	// merge matrix typically when using gauss points
	void StMat_Lp2Ta::merge_matrix(
		const arma::Row<fltp>&weights, 
		const arma::uword num_stride){
		
		assert(M_.is_finite());
		assert(weights.is_finite());
		assert(weights.n_elem==M_.n_rows);

		// apply weights
		for(arma::uword i=0;i<M_.n_rows;i++)
			for(arma::uword j=0;j<M_.n_cols;j++)
				M_(i,j)*=weights(i);

		// add values
		M_ = arma::reshape(
			arma::sum(arma::reshape(M_,num_stride,M_.n_elem/num_stride),0),
			M_.n_rows/num_stride,M_.n_cols);

		// update number of sources
		num_targets_/=num_stride;

		// check
		assert(M_.is_finite());
	}

	// apply to localpole
	arma::Mat<fltp> StMat_Lp2Ta::apply(
		const arma::Mat<std::complex<fltp> > &Lp) const{
		// check if it is a single multipole
		assert(Extra::polesize(num_exp_)==static_cast<int>(Lp.n_rows)); 

		// get real part and return transpose
		arma::Mat<fltp> A = arma::real(M_*Lp);

		// return A
		return A.t();
	}

	// apply to localpole
	arma::Mat<fltp> StMat_Lp2Ta::apply(
		const arma::Mat<std::complex<fltp> > &Lp, 
		const arma::Mat<arma::uword> &indices) const{

		// check if it is a single multipole
		assert(Extra::polesize(num_exp_)==static_cast<int>(Lp.n_rows));
		assert(arma::as_scalar(arma::max(indices,1))<num_targets_);
		assert(!M_.is_empty());

		// get real part and return transpose
		arma::Mat<fltp> A = arma::real(M_.rows(indices)*Lp);

		// return A
		return A.t();
	}

	// apply to localpole
	arma::Mat<fltp> StMat_Lp2Ta::apply(
		const arma::Mat<std::complex<fltp> > &Lp, 
		const arma::uword first_target,
		const arma::uword last_target) const{

		// check if it is a single multipole
		assert(Extra::polesize(num_exp_)==static_cast<int>(Lp.n_rows));
		assert(first_target<=last_target);
		assert(last_target<num_targets_);
		assert(!M_.is_empty());

		// get real part and return transpose
		arma::Mat<fltp> A = arma::real(M_.rows(first_target,last_target)*Lp);

		// return A
		return A.t();
	}

	// check if it was setup
	bool StMat_Lp2Ta::is_empty() const{
		return M_.is_empty();
	}	

	// display the values stored in the matrix
	void StMat_Lp2Ta::display() const{
		// header
		printf("Multipole to Target matrix\n");
		
		// display matrix
		cmn::Extra::display_mat(M_);
	}


	// default constructor
	StMat_Lp2Ta_Curl::StMat_Lp2Ta_Curl(){

	}

	// constructor
	StMat_Lp2Ta_Curl::StMat_Lp2Ta_Curl(const int num_exp, const arma::Mat<fltp> &dR){
		num_exp_ = num_exp;	
		calc_matrix(dR);
	}

	// factory
	ShStMat_Lp2Ta_CurlPr StMat_Lp2Ta_Curl::create(){
		//return ShIListPr(new IList);
		return std::make_shared<StMat_Lp2Ta_Curl>();
	}


	// set number of expansions
	void StMat_Lp2Ta_Curl::set_num_exp(const int num_exp){
		assert(num_exp>0);
		num_exp_ = num_exp;
	}

	// set number of expansions
	void StMat_Lp2Ta_Curl::set_num_exp(const arma::uword num_exp){
		assert(num_exp>0);
		num_exp_ = static_cast<int>(num_exp);
	}

	// setup matrix for localpole to target calculation for magnetic field
	void StMat_Lp2Ta_Curl::calc_matrix(const arma::Mat<fltp> &dR){
		// check input
		assert(dR.n_rows==3);
		assert(num_exp_>0);
		assert(dR.is_finite());

		// set number of matrices
		num_targets_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		const arma::Mat<fltp> dRsph = Extra::cart2sph(-dR);

		// calculate harmonic
		const Spharm L(num_exp_,dRsph);

		// calculate factorials
		const arma::Col<fltp> facs = Extra::factorial(2*num_exp_);

		// allocate matrix with zeros
		M_.zeros(3*num_targets_,Extra::polesize(num_exp_));

		// keep track of rho^(n-1)
		arma::Row<fltp> rhonminj(num_targets_,arma::fill::ones);

		// calculate values for matrix
		// walk over source pole
		for (int n=1;n<=num_exp_;n++){
			// walk over target pole
			for (int k=-1;k<=1;k++){
				// walk over source pole
				for (int m=std::max(-n,k-n+1);m<=std::min(n,k+n-1);m++){
					// common factor
					const arma::Row<std::complex<fltp> > cst = 
						(rhonminj%L.get_nm(n-1,m-k))*
						(Extra::Anm(facs,n-1,m-k)*
						Extra::Anm(facs,1,k)*
						Extra::ipown(std::abs(m)-std::abs(m-k)-std::abs(k))/
						std::pow(-1,n+1)/
						Extra::Anm(facs,n,m));

					// insert into matrix (note: j-n = 1)
					const arma::uword tid = Extra::nm2fidx(1,k)-1;
					const arma::uword sid = Extra::nm2fidx(n,m);
					assert(tid<3);
					M_.submat(arma::span(tid*num_targets_,
						(tid+1)*num_targets_-1),
						arma::span(sid,sid)) = cst.st();
				}
			}

			// calculate rho^(n-1)
			rhonminj %= dRsph.row(0);
		}

		// setup mini matrix
		// from armadillo docs: the sparse matrix class is not intended for 
		// small matrices (eg. ≤ 100x100), due to the overhead of the compressed storage format
		// Mconv_ = arma::SpMat<std::complex<fltp> >(
		// 	StMat_Lp2Ta_Curl::get_mini_multipole_matrix().st());
		Mconv_ = StMat_Lp2Ta_Curl::get_mini_multipole_matrix().st();
	}


	// // calculate magnetic field from localpole
	// // (this part of the code can probably be rewritten translating the
	// // localpole to the target position then extracting n=1 row for the field
	// // this means that So2Mp_M and Lp2Ta_H should be symmetric)
	// void StMat_Lp2Ta_Curl::calc_matrix2(const arma::Mat<fltp> &dR){

	// 	// check input
	// 	assert(num_exp_>0);
	// 	assert(dR.n_rows==3);
	// 	assert(dR.is_finite());

	// 	// set number of targets
	// 	num_targets_ = dR.n_cols;

	// 	// convert to spherical coordinates (rho,theta,phi)
	// 	arma::Mat<fltp> dRsph = Extra::cart2sph(dR);
		
	// 	// deal with elements at position of harmonic
	// 	arma::Mat<fltp> rho = dRsph.row(0);
	// 	rho(find(rho<1e-12)).fill(1e-12); dRsph.row(0) = rho;

	// 	// calculate harmonic and special harmonic functions
	// 	Spharm L(num_exp_,dRsph);

	// 	// calculate derivative in theta direction
	// 	fltp dtheta = arma::Datum<fltp>::pi/720;
	// 	arma::Mat<fltp> dRsph_1 = dRsph; dRsph_1.row(1) -= dtheta/2;
	// 	arma::Mat<fltp> dRsph_2 = dRsph; dRsph_2.row(1) += dtheta/2;
	// 	dRsph_1.row(1) = arma::clamp(dRsph_1.row(1),0,arma::Datum<fltp>::pi);
	// 	dRsph_2.row(1) = arma::clamp(dRsph_2.row(1),0,arma::Datum<fltp>::pi);
	// 	Spharm L1(num_exp_,dRsph_1);
	// 	Spharm L2(num_exp_,dRsph_2);
	// 	Spharm Ldtheta = L2-L1; Ldtheta.divide(dRsph_2.row(1) - dRsph_1.row(1));

	// 	// calculate derivative in phi direction
	// 	fltp dphi = arma::Datum<fltp>::pi/720;
	// 	arma::Mat<fltp> dRsph_3 = dRsph; dRsph_3.row(2) -= dphi/2;
	// 	arma::Mat<fltp> dRsph_4 = dRsph; dRsph_4.row(2) += dphi/2;
	// 	Spharm L3(num_exp_,dRsph_3);
	// 	Spharm L4(num_exp_,dRsph_4);
	// 	Spharm Ldphi = (L4-L3)/dphi;

	// 	// calculate size of pole
	// 	int N = Extra::polesize(num_exp_);

	// 	// allocate matrix
	// 	M_.set_size(3*num_targets_,N);
			
	// 	// pre-calculate required cos and sines for coordinate transformation
	// 	arma::Mat<fltp> sin_theta = arma::sin(dRsph.row(1));
	// 	arma::Mat<fltp> cos_theta = arma::cos(dRsph.row(1));
	// 	arma::Mat<fltp> sin_phi = arma::sin(dRsph.row(2));
	// 	arma::Mat<fltp> cos_phi = arma::cos(dRsph.row(2));

	// 	// walk over localpole
	// 	int p = 0;
	// 	arma::Mat<arma::sword> jk(N,2);
	// 	for(int j=0;j<=num_exp_;j++){
	// 		for(int k=-j;k<=j;k++){
	// 			jk(p,0) = j; jk(p,1) = k;
	// 			p++;
	// 		}
	// 	}
	// 	// sanity check
	// 	assert(p==N);

	// 	// run over indices and calculate in parallel
	// 	cmn::parfor(0, N, use_parallel_,[&](int i, int) { // second int is CPU number
	// 		// get indices
	// 		int j = jk(i,0); int k = jk(i,1);

	// 		// calculate rho^(j-1)
	// 		arma::Mat<fltp> rhojmo = arma::pow(dRsph.row(0),j-1); // rho = dRsph.row(0)
		
	// 		// get gradient of Legendre function in spherical coordinates
	// 		arma::Mat<std::complex<fltp> > dLdrho = fltp(j)*L.get_nm(j,k)%rhojmo;
	// 		arma::Mat<std::complex<fltp> > dLdtheta = Ldtheta.get_nm(j,k)%rhojmo;
	// 		arma::Mat<std::complex<fltp> > dLdphi = Ldphi.get_nm(j,k)%rhojmo;

	// 		// convert partial derivatives to carthesian coordinates 
	// 		// (division by rho is already done in rhojmo)
	// 		// later move outside for loop
	// 		// https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
	// 		// http://mathworld.wolfram.com/SphericalCoordinates.html
	// 		arma::Mat<std::complex<fltp> > dLdx = 
	// 			cos_phi%sin_theta%dLdrho + 
	// 			cos_phi%cos_theta%dLdtheta - 
	// 			sin_phi/sin_theta%dLdphi;
	// 		arma::Mat<std::complex<fltp> > dLdy = 
	// 			sin_phi%sin_theta%dLdrho + 
	// 			sin_phi%cos_theta%dLdtheta + 
	// 			cos_phi/sin_theta%dLdphi;
	// 		arma::Mat<std::complex<fltp> > dLdz = 
	// 			cos_theta%dLdrho - 
	// 			sin_theta%dLdtheta; 

	// 		// insert three components into matrix
	// 		M_.col(Extra::nm2fidx(j,k)) = 
	// 			arma::join_horiz(dLdx,
	// 			arma::join_horiz(dLdy,
	// 							dLdz)).st();
	// 	});

	// 	// check for nans
	// 	assert(!M_.has_nan());
	// }

	// get localpole to target matrix
	arma::Mat<std::complex<fltp> > StMat_Lp2Ta_Curl::get_matrix() const{
		return M_;
	}

	// get localpole to target matrix
	arma::Mat<std::complex<fltp> > StMat_Lp2Ta_Curl::get_matrix(const arma::uword k) const{
		return M_.rows(arma::Row<arma::uword>::fixed<3>{k,k+num_targets_,k+2*num_targets_});
	}

	// apply to localpole
	arma::Mat<fltp> StMat_Lp2Ta_Curl::apply(
		const arma::Mat<std::complex<fltp> > &Lp) const{

		// check if it is a single multipole
		assert(!M_.is_empty());
		assert(Extra::polesize(num_exp_)==static_cast<int>(Lp.n_rows)); 
		assert(3==Lp.n_cols); // assumed dimension of three
		
		//M -> 3*Nt by Polesize
		//Lp -> polesize by 3
		//M*Lp -> 3*Nt by 3(mini mp)

		// calculate mini localpoles at target locations [x1,y1,z1;x2,y2,z2;x3,y3,z3]
		const arma::Mat<std::complex<fltp> > Mlp = M_*Lp;
		assert(Mlp.n_elem==9*num_targets_);

		// conversion matrix
		//arma::Mat<std::complex<fltp> >::fixed<3,9> Mconv = StMat_So2Mp_M::get_mini_multipole_matrix().st();
		// arma::SpMat<std::complex<fltp> > Mconv(StMat_Lp2Ta_Curl::get_mini_multipole_matrix().st());

		// calculate field [x1,x2,x3,y1,y2,y3,z1,z2,z3]
		arma::Mat<fltp> curl = arma::real(Mconv_*arma::reshape(Mlp,num_targets_,9).st()); // /(4*arma::Datum<fltp>::pi)

		assert(curl.n_rows == 3);
		assert(curl.n_cols == num_targets_);

		// convert
		return curl;
	}

	// apply to localpole
	arma::Mat<fltp> StMat_Lp2Ta_Curl::apply(
		const arma::Mat<std::complex<fltp> > &Lp,
		const arma::uword first_index,
		const arma::uword last_index) const{

		// check if it is a single multipole
		assert(!M_.is_empty()); assert(static_cast<int>(M_.n_cols)==Extra::polesize(num_exp_));
		assert(Extra::polesize(num_exp_)==static_cast<int>(Lp.n_rows)); 
		assert(3==Lp.n_cols); // assumed dimension of three

		// conversion matrix
		// arma::Mat<std::complex<fltp> >::fixed<3,9> Mconv = StMat_So2Mp_M::get_mini_multipole_matrix().st();

		// number of targets
		const arma::uword num_targets = last_index-first_index+1;

		// create output matrix
		arma::Mat<std::complex<fltp> > Mlp(num_targets*3,3);
		for(arma::uword i=0;i<3;i++)
			Mlp.rows(i*num_targets,(i+1)*num_targets-1) = 
				M_.rows(i*num_targets_+first_index,
				i*num_targets_+last_index)*Lp;

		// calculate mini localpoles at target locations
		// const arma::uword idx1 = num_dim*first_index;
		// const arma::uword idx2 = num_dim*(last_index+1)-1;
		// const arma::Mat<std::complex<fltp> > Mlp = M_.rows(idx1,idx2)*Lp;
		// assert(Mlp.n_rows==num_dim); assert(Mlp.n_cols==num_dim*num_targets);
		// assert(Mlp.n_elem==num_dim*num_dim*num_targets);

		// calculate field
		const arma::Mat<fltp> curl = arma::real(Mconv_*arma::reshape(Mlp,num_targets,9).st()); // /(4*arma::Datum<fltp>::pi)
		assert(curl.n_rows == 3); assert(curl.n_cols == num_targets);

		// std::cout<<Mconv.n_rows<<" "<<Mconv.n_cols<<std::endl;
		// std::cout<<Mlp.n_rows<<" "<<Mlp.n_cols<<std::endl;
		// std::cout<<curl.n_rows<<" "<<curl.n_cols<<std::endl;

		// convert
		return curl;
	}

	// method for calculating multipoles at element coordinates
	// these multipoles only contain n=1 and m=-1,0,+1
	arma::Mat<std::complex<fltp> >::fixed<9,3> StMat_Lp2Ta_Curl::get_mini_multipole_matrix(){
		
		// create complex 0 and 1
		const std::complex<fltp> icmplx(0,RAT_CONST(1.0));
		const std::complex<fltp> rcmplx(RAT_CONST(1.0),0);
		const fltp sqrt2 = std::sqrt(RAT_CONST(2.0));
		
		// setup matrix
		arma::Mat<std::complex<fltp> >::fixed<9,3> Mconv;

		// x multipole, n = 1, m = -1
		Mconv.row(0) = arma::Row<std::complex<fltp> >::fixed<3>{
			RAT_CONST(0.0),RAT_CONST(0.0),-RAT_CONST(0.5)*sqrt2*icmplx};

		// x multipole, n = 1, m = 0
		Mconv.row(1) = arma::Row<std::complex<fltp> >::fixed<3>{
			RAT_CONST(0.0),RAT_CONST(1.0)*rcmplx,RAT_CONST(0.0)};

		// x multipole, n = 1, m = +1
		Mconv.row(2) = arma::Row<std::complex<fltp> >::fixed<3>{
			RAT_CONST(0.0),RAT_CONST(0.0),+RAT_CONST(0.5)*sqrt2*icmplx};

		// y multipole, n = 1, m = -1
		Mconv.row(3) = arma::Row<std::complex<fltp> >::fixed<3>{
			RAT_CONST(0.0),RAT_CONST(0.0),-RAT_CONST(0.5)*sqrt2*rcmplx};

		// y multipole, n = 1, m = 0
		Mconv.row(4) = arma::Row<std::complex<fltp> >::fixed<3>{
			-RAT_CONST(1.0)*rcmplx,RAT_CONST(0.0),RAT_CONST(0.0)};

		// y multipole, n = 1, m = +1
		Mconv.row(5) = arma::Row<std::complex<fltp> >::fixed<3>{
			RAT_CONST(0.0),RAT_CONST(0.0),-RAT_CONST(0.5)*sqrt2*rcmplx};

		// z multipole, n = 1, m = -1
		Mconv.row(6) = arma::Row<std::complex<fltp> >::fixed<3>{
			+RAT_CONST(0.5)*sqrt2*icmplx,RAT_CONST(0.5)*sqrt2*rcmplx,RAT_CONST(0.0)};

		// z multipole, n = 1, m = 0
		Mconv.row(7) = arma::Row<std::complex<fltp> >::fixed<3>{
			RAT_CONST(0.0),RAT_CONST(0.0),RAT_CONST(0.0)};

		// z multipole, n = 1, m = +1
		Mconv.row(8) = arma::Row<std::complex<fltp> >::fixed<3>{
			-RAT_CONST(0.5)*sqrt2*icmplx,RAT_CONST(0.5)*sqrt2*rcmplx,RAT_CONST(0.0)};

		// return matrix
		return Mconv;
	}


	// calculate multipole
	 	//return M_*arma::reshape((Mconv*Meff).st(),3*Meff.n_cols,3);

	// // helper function for calculating magnetic field
	// arma::Mat<fltp> StMat_Lp2Ta_Curl::apply_helper(
	// 	const arma::Mat<fltp> &dLLp, const arma::uword N) const{

	// 	// reshape result
	// 	// [xx,yx,zx,xy,yy,zy,xz,yz,zz]
	// 	arma::Mat<fltp> dLLpt = arma::reshape(dLLp,N,9);

	// 	// get output
	// 	//arma::Mat<fltp> dAxdx = dLLpt.col(0);
	// 	arma::Mat<fltp> dAxdy = dLLpt.col(1);
	// 	arma::Mat<fltp> dAxdz = dLLpt.col(2);

	// 	arma::Mat<fltp> dAydx = dLLpt.col(3);
	// 	//arma::Mat<fltp> dAydy = dLLpt.col(4);
	// 	arma::Mat<fltp> dAydz = dLLpt.col(5);

	// 	arma::Mat<fltp> dAzdx = dLLpt.col(6);
	// 	arma::Mat<fltp> dAzdy = dLLpt.col(7);
	// 	//arma::Mat<fltp> dAzdz = dLLpt.col(8);
		
	// 	// cross product
	// 	arma::Mat<fltp> H(N,3);
	// 	H.col(0) = dAzdy - dAydz;
	// 	H.col(1) = dAxdz - dAzdx;
	// 	H.col(2) = dAydx - dAxdy;

	// 	// check for nans
	// 	assert(H.is_finite());
	// 	assert(!H.has_nan());

	// 	// return H in [A/m]
	// 	return H.t()/(4*arma::Datum<fltp>::pi);
	// }

	// // matrix for calculating cross product
	// arma::Mat<fltp>::fixed<3,9> StMat_Lp2Ta_Curl::get_cross_product_matrix(){
	// 	// create matrix
	// 	arma::Mat<fltp>::fixed<3,9> Mcr;

	// 	// x-direction
	// 	Mcr.row(0) = arma::Row<fltp>::fixed<9>{0,0,0 ,0,0,-1, 0,1,0};

	// 	// y-direction
	// 	Mcr.row(1) = arma::Row<fltp>::fixed<9>{0,0,1 ,0,0,0, -1,0,0};

	// 	// z-direction
	// 	Mcr.row(2) = arma::Row<fltp>::fixed<9>{0,-1,0 ,1,0,0, 0,0,0};

	// 	// return matrix
	// 	return Mcr;
	// }




	// check if it was setup
	bool StMat_Lp2Ta_Curl::is_empty() const{
		return M_.is_empty();
	}	

	// direct setting of the matrix
	void StMat_Lp2Ta_Curl::set_M(const arma::uword num_targets, const arma::Mat<std::complex<fltp> > &M, const int num_exp){
		assert(Extra::polesize(num_exp)==static_cast<int>(M.n_cols));
		assert(3*num_targets==M.n_rows);
		M_ = M; num_exp_ = num_exp; num_targets_ = num_targets;
	}

	// direct access to matrix
	arma::Mat<std::complex<fltp> > StMat_Lp2Ta_Curl::get_M() const{
		return M_;
	}

	// // set parallelism
	// void StMat_Lp2Ta_Curl::set_use_parallel(const bool use_parallel){
	// 	use_parallel_ = use_parallel;
	// }

	// display the values stored in the matrix
	void StMat_Lp2Ta_Curl::display() const{
		// header
		printf("Multipole to Target matrix\n");
		printf("number of rows: %llu\n",M_.n_rows);
		printf("number of cols: %llu\n",M_.n_cols);

		// display matrix
		cmn::Extra::display_mat(M_);
	}










	// default constructor
	StMat_Lp2Ta_Grad::StMat_Lp2Ta_Grad(){

	}

	// constructor
	StMat_Lp2Ta_Grad::StMat_Lp2Ta_Grad(const int num_exp, const arma::Mat<fltp> &dR){
		num_exp_ = num_exp;	
		calc_matrix(dR);
	}

	// factory
	ShStMat_Lp2Ta_GradPr StMat_Lp2Ta_Grad::create(){
		//return ShIListPr(new IList);
		return std::make_shared<StMat_Lp2Ta_Grad>();
	}


	// set number of expansions
	void StMat_Lp2Ta_Grad::set_num_exp(const int num_exp){
		assert(num_exp>0);
		num_exp_ = num_exp;
	}

	// set number of expansions
	void StMat_Lp2Ta_Grad::set_num_exp(const arma::uword num_exp){
		assert(num_exp>0);
		num_exp_ = static_cast<int>(num_exp);
	}

	// setup matrix for localpole to target calculation for magnetic field
	void StMat_Lp2Ta_Grad::calc_matrix(const arma::Mat<fltp> &dR){
		// check input
		assert(dR.n_rows==3);
		assert(num_exp_>0);
		assert(dR.is_finite());

		// set number of matrices
		num_targets_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		const arma::Mat<fltp> dRsph = Extra::cart2sph(-dR);

		// calculate harmonic
		Spharm L(num_exp_,dRsph);

		// calculate factorials
		const arma::Col<fltp> facs = Extra::factorial(2*num_exp_);
		
		// allocate matrix with zeros
		M_.zeros(3*num_targets_,Extra::polesize(num_exp_));

		// calculate values for matrix
		// walk over target pole
		for (int k=-1;k<=1;k++){
			// walk over source pole
			for (int n=1;n<=num_exp_;n++){
				// calculate rho^(n-j)
				const arma::Mat<fltp> rhonminj = arma::pow(dRsph.row(0),n-1);
				for (int m=std::max(-n,k-n+1);m<=std::min(n,k+n-1);m++){
					// common factor
					const arma::Row<std::complex<fltp> > cst = 
						Extra::Anm(facs,n-1,m-k)*
						Extra::Anm(facs,1,k)*
						rhonminj*
						Extra::ipown(std::abs(m)-std::abs(m-k)-std::abs(k))%
						L.get_nm(n-1,m-k)/
						std::pow(-1,n+1)/
						Extra::Anm(facs,n,m);

					// insert into matrix (note: j-n = 1)
					const arma::uword tid = Extra::nm2fidx(1,k)-1;
					const arma::uword sid = Extra::nm2fidx(n,m);
					assert(tid<3);
					M_.submat(arma::span(tid*num_targets_,
						(tid+1)*num_targets_-1),
						arma::span(sid,sid)) = cst.st();
				}
			}
		}

		// setup mini matrix
		// Mconv_ = arma::SpMat<std::complex<fltp> >(StMat_Lp2Ta_Grad::get_mini_multipole_matrix().st());
		Mconv_ = StMat_Lp2Ta_Grad::get_mini_multipole_matrix().st();
	}

	// get localpole to target matrix
	arma::Mat<std::complex<fltp> > StMat_Lp2Ta_Grad::get_matrix() const{
		return M_;
	}

	// get localpole to target matrix
	arma::Mat<std::complex<fltp> > StMat_Lp2Ta_Grad::get_matrix(const arma::uword k) const{
		return M_.rows(arma::Row<arma::uword>::fixed<3>{k,k+num_targets_,k+2*num_targets_});
	}

	// apply to localpole
	arma::Mat<fltp> StMat_Lp2Ta_Grad::apply(
		const arma::Mat<std::complex<fltp> > &Lp) const{

		// check if it is a single multipole
		assert(Extra::polesize(num_exp_)==static_cast<int>(Lp.n_rows)); 
		assert(1==Lp.n_cols); // assumed dimension of three
		assert(!Mconv_.is_empty()); assert(!M_.is_empty());

		// calculate mini localpoles at target locations [m=-1,m=0,m=1]
		arma::Mat<std::complex<fltp> > Mlp = M_*Lp;
		assert(Mlp.n_elem==3*num_targets_);

		// conversion matrix
		//arma::Mat<std::complex<fltp> >::fixed<3,9> Mconv = StMat_So2Mp_M::get_mini_multipole_matrix().st();
		

		// calculate field [x1,x2,x3,y1,y2,y3,z1,z2,z3]
		arma::Mat<fltp> grad = arma::real(Mconv_*arma::reshape(Mlp,num_targets_,3).st());

		assert(grad.n_rows == 3);
		assert(grad.n_cols == num_targets_);

		// convert
		return grad;
	}

	// apply to localpole
	arma::Mat<fltp> StMat_Lp2Ta_Grad::apply(
		const arma::Mat<std::complex<fltp> > &Lp,
		const arma::uword first_index,
		const arma::uword last_index) const{

		// check if it is a single multipole
		assert(Extra::polesize(num_exp_)==static_cast<int>(Lp.n_rows)); 
		assert(1==Lp.n_cols); // assumed dimension of three
		assert(!Mconv_.is_empty()); assert(!M_.is_empty());

		// conversion matrix
		//arma::Mat<std::complex<fltp> >::fixed<3,9> Mconv = StMat_So2Mp_M::get_mini_multipole_matrix().st();
		// const arma::SpMat<std::complex<fltp> > Mconv(StMat_Lp2Ta_Grad::get_mini_multipole_matrix().st());

		// number of targets
		const arma::uword num_targets = last_index-first_index+1;

		// calculate mini localpoles at target locations
		const arma::Mat<std::complex<fltp> > Mlp = (M_.rows(first_index,last_index)*Lp).st();
		assert(Mlp.n_elem==3*num_targets);

		// calculate field
		const arma::Mat<fltp> grad = arma::real(Mconv_*arma::reshape(Mlp,num_targets,3).st());
		assert(grad.n_rows == 3);
		assert(grad.n_cols == num_targets);

		// convert
		return grad;
	}

	// method for calculating multipoles at element coordinates
	// these multipoles only contain n=1 and m=-1,0,+1
	arma::Mat<std::complex<fltp> >::fixed<3,3> StMat_Lp2Ta_Grad::get_mini_multipole_matrix(){
		// create complex 0 and 1
		const std::complex<fltp> icmplx(0,RAT_CONST(1.0));
		const std::complex<fltp> rcmplx(RAT_CONST(1.0),0);
		
		// setup matrix
		arma::Mat<std::complex<fltp> >::fixed<3,3> Mconv;

		// n = 1, m = -1
		Mconv.row(0) = arma::Row<std::complex<fltp> >::fixed<3>{0.5f*std::sqrt(RAT_CONST(2.0))*rcmplx,-0.5f*std::sqrt(RAT_CONST(2.0))*icmplx,0};

		// n = 1, m = 0
		Mconv.row(1) = arma::Row<std::complex<fltp> >::fixed<3>{0,0,-RAT_CONST(1.0)*rcmplx};

		// n = 1, m = +1
		Mconv.row(2) = arma::Row<std::complex<fltp> >::fixed<3>{0.5f*std::sqrt(RAT_CONST(2.0))*rcmplx,+0.5f*std::sqrt(RAT_CONST(2.0))*icmplx,0};

		// return matrix
		return Mconv;
	}

	// check if it was setup
	bool StMat_Lp2Ta_Grad::is_empty() const{
		return M_.is_empty();
	}	

	// direct setting of the matrix
	void StMat_Lp2Ta_Grad::set_M(const arma::uword num_targets, const arma::Mat<std::complex<fltp> > &M, const int num_exp){
		assert(Extra::polesize(num_exp)==static_cast<int>(M.n_cols));
		assert(3*num_targets==M.n_rows);
		M_ = M; num_exp_ = num_exp; num_targets_ = num_targets;
	}

	// direct access to matrix
	arma::Mat<std::complex<fltp> > StMat_Lp2Ta_Grad::get_M() const{
		return M_;
	}

	// set parallelism
	void StMat_Lp2Ta_Grad::set_use_parallel(const bool use_parallel){
		use_parallel_ = use_parallel;
	}

	// display the values stored in the matrix
	void StMat_Lp2Ta_Grad::display() const{
		// header
		printf("Multipole to Target matrix\n");
		printf("number of rows: %llu\n",M_.n_rows);
		printf("number of cols: %llu\n",M_.n_cols);

		// display matrix
		cmn::Extra::display_mat(M_);
	}

}}