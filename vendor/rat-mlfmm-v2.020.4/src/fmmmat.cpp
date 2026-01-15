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
#include "fmmmat.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// default constructor
	FmmMat_Mp2Mp::FmmMat_Mp2Mp(){

	}

	// constructor from coordinates
	FmmMat_Mp2Mp::FmmMat_Mp2Mp(const int num_exp, const arma::Mat<fltp> &dR){
		// check input
		assert(dR.n_rows==3);

		// set number of expansions
		num_exp_ = num_exp;	

		// calculate matrix from coordinates
		calc_matrix(dR);
	}

	// factory
	ShFmmMat_Mp2MpPr FmmMat_Mp2Mp::create(){
		//return ShIListPr(new IList);
		return std::make_shared<FmmMat_Mp2Mp>();
	}

	// set number of expansions
	// determines accuracy of calculation
	void FmmMat_Mp2Mp::set_num_exp(const int num_exp){
		num_exp_=num_exp;
	}

	// for unsigned input
	void FmmMat_Mp2Mp::set_num_exp(const arma::uword num_exp){
		num_exp_=static_cast<int>(num_exp);
	}

	// Assemble multipole-to-multipole matrix
	void FmmMat_Mp2Mp::calc_matrix(const arma::Mat<fltp> &dR){
		// check input
		assert(dR.n_rows==3);
		assert(num_exp_>0);
		assert(dR.is_finite());

		// set number of matrices
		num_mat_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		arma::Mat<fltp> dRsph = Extra::cart2sph(dR);

		// calculate harmonic
		Spharm L(num_exp_,dRsph);

		// calculate factorials
		const arma::Col<fltp> facs = Extra::factorial(2*num_exp_);

		// calculate number of non-zero elements in Mp2Mp matrix
		int Nnz = num_nz(num_exp_);
		
		// allocate vectors for fast matrix assembly
		arma::Mat<arma::uword> loc(2,Nnz);
		arma::Mat<std::complex<fltp> > val(Nnz,num_mat_);

		// calculate values for matrix
		// walk over target pole
		int idx = 0;
		for (int j=0;j<=num_exp_;j++){
			for (int k=-j;k<=j;k++){
				// walk over source pole
				for (int n=0;n<=j;n++){
					for (int m=std::max(-n,k-j+n);m<=std::min(n,k+j-n);m++){
						// common factor
						arma::Mat<std::complex<fltp> > cst = 
							Extra::Anm(facs,n,m)*
							Extra::Anm(facs,j-n,k-m)*
							Extra::ipown(std::abs(k)-std::abs(m)-std::abs(k-m))*
							arma::pow(dRsph.row(0),n)%
							L.get_nm(n,-m)/
							Extra::Anm(facs,j,k);

						// add translated multipole
						loc(0,idx) = Extra::nm2fidx(j,k);
						loc(1,idx) = Extra::nm2fidx(j-n,k-m);
						val.row(idx) = cst;
						idx += 1;
					}
				}
			}
		}

		// check matrix filling
		assert(idx==Nnz);

		// set matrix array
		M_.set_size(1,num_mat_);

		// insert into sparse matrix
		int N = Extra::polesize(num_exp_);
		for(arma::uword i=0;i<num_mat_;i++){
			M_(i) = arma::SpMat<std::complex<fltp> >(loc,val.col(i).st(),N,N,true,true);
			//assert(M_(i).n_nonzero==Nnz);

			// check for nans
			assert(M_(i).is_finite());
		}


	}

	// get matrix with index k
	arma::SpMat<std::complex<fltp> > FmmMat_Mp2Mp::get_matrix(const arma::uword k) const{
		return M_(k);
	}

	// apply specific matrix N times
	arma::Mat<std::complex<fltp> > FmmMat_Mp2Mp::apply(
		const arma::uword k, const arma::Mat<std::complex<fltp> > &Mp) const{
		// forward the request with N=0
		return apply(k,Mp,0);
	}

	// apply calculated transformation matrix to multipole
	// the distance over which the matrix is appplied is 
	// multiplied by 2^N
	arma::Mat<std::complex<fltp> > FmmMat_Mp2Mp::apply(
		const arma::uword k, 
		const arma::Mat<std::complex<fltp> > &Mp, 
		const arma::uword N) const{

		// check input
		assert(Extra::polesize(num_exp_)==static_cast<int>(Mp.n_rows));
		assert(k<num_mat_);

		// calculate
		return Extra::matrix_self_multiplication(M_(k),N)*Mp;
	}

	// display the values stored in the matrix
	void FmmMat_Mp2Mp::display() const{
		for (arma::uword k=0;k<num_mat_;k++){
			display(k);
		}
	}

	// display the values stored in the matrix
	void FmmMat_Mp2Mp::display(const arma::uword k) const{
		// multipole matrix
		std::printf("Multipole to Multipole matrix\n");

		// header displaying which harmonic
		if(num_mat_>1){
			int boxwidth = Extra::polesize(num_exp_)*(6+1)-1;
			for (int j=0;j<boxwidth;j++){std::printf("-");}; std::printf("\n");
			std::printf("Matrix with index %03llu/%03llu\n",k,num_mat_);
		}

		// plot sparse matrix
		cmn::Extra::display_spmat(M_(k));
	}

	// number of non-zeros in each matrix
	arma::uword FmmMat_Mp2Mp::get_num_mat() const{
		return num_mat_;
	}

	// number of non-zeros in each matrix
	int FmmMat_Mp2Mp::get_num_exp() const{
		return num_exp_;
	}

	// number of non-zeros in each matrix
	int FmmMat_Mp2Mp::num_nz(const int num_exp){
		int Nnz = 0;
		for (int n=0;n<=num_exp;n++){
			Nnz += (1+2*n)*Extra::polesize(num_exp-n);
		}
		return Nnz;
	}

	// display in log
	void FmmMat_Mp2Mp::display(cmn::ShLogPr &lg) const{
		// get statistics
		int mat_size = Extra::polesize(num_exp_);
		int num_nz = FmmMat_Mp2Mp::num_nz(num_exp_);

		// estimated memory (note that armadillo internally 
		// uses a special storage format of which we can not 
		// be sure how much memory it uses)
		fltp mp2mp_memory = (4*num_mat_*num_nz*64.0/8)/1024/1024; 
		fltp mp2mp_sparsity = static_cast<fltp>(num_nz)/(mat_size*mat_size);

		// display statistics
		lg->msg(2,"multipole to multipole %s(Mp2Mp)%s\n",KGRN,KNRM);
		lg->msg("number of matrices: %s%llu%s\n",KYEL,num_mat_,KNRM);
		lg->msg("matrix type: %sSPARSE_CX%s\n",KGRN,KNRM);
		lg->msg("matrix size: %s%llu%s X %s%llu%s\n",KYEL,mat_size,KNRM,KYEL,mat_size,KNRM);
		lg->msg("number of nonzeros: %s%llu%s\n",KYEL,num_nz,KNRM);
		lg->msg("sparsity level: %s%.2f%s [pct]\n",KYEL,100*mp2mp_sparsity,KNRM);
		lg->msg(-2,"estimated memory: %s%.2f%s [MB]\n",KYEL,mp2mp_memory,KNRM);
		lg->newl();
	}


	// create distance matrix
	// scales distance with 2^N
	// constructor
	FmmMat_Mp2Lp_dist::FmmMat_Mp2Lp_dist(){
		
	}

	// setup distance matrix
	void FmmMat_Mp2Lp_dist::setup(
		const arma::uword N){

		// calculate size of multipoles
		arma::uword mypolesize = Extra::polesize(num_exp_);

		// allocate matrix
		M_.set_size(mypolesize,mypolesize);

		// case where N equals zero
		if(N==0){
			M_.fill(1);
		}

		// case where N is larger than zero
		else{
			// factor
			// arma::uword fac = std::pow(2,N);

			// walk over target
			for (int j=0; j<=num_exp_;j++){
				for (int k=-j;k<=j;k++){
					// walk over source
					for (int n=0;n<=num_exp_;n++){
						for (int m=-n;m<=n;m++){
							// fill matrix
							//M_(Extra::nm2fidx(j,k),Extra::nm2fidx(n,m)) = 
							//	1/std::pow(fac,j+n+1);
							M_(Extra::nm2fidx(j,k),Extra::nm2fidx(n,m)) = 
								1.0/std::pow(2,N*(j+n+1));

						}
					}
				}
			}
		}

		// check if matrix is finite
		assert(M_.is_finite());
	}

	// get stored matrix
	const arma::Mat<fltp>& FmmMat_Mp2Lp_dist::get_matrix() const{
		// ensure that matrix is setup
		assert(!M_.is_empty());

		// return matrix
		return M_;
	}

	// set number of expansions
	void FmmMat_Mp2Lp_dist::set_num_exp(const int num_exp){
		assert(num_exp>0);
		num_exp_=num_exp;
	}

	// apply distance matrix to Mp2Lp matrix
	arma::Mat<std::complex<fltp> > FmmMat_Mp2Lp_dist::apply(
		const arma::Mat<std::complex<fltp> >& M) const{
		// ensure that matrix is setup
		assert(!M_.is_empty());

		// calculate size of multipoles
		const arma::uword polesize = Extra::polesize(num_exp_);

		// return element wise product
		return arma::repmat(M_,1,M.n_cols/polesize)%M;
	}

	// default constructor
	FmmMat_Mp2Lp::FmmMat_Mp2Lp(){

	}

	// constructor
	FmmMat_Mp2Lp::FmmMat_Mp2Lp(const int num_exp, const arma::Mat<fltp> &dR){
		assert(dR.n_rows==3);
		num_exp_ = num_exp;	
		calc_matrix(dR);
	}

	// factory
	ShFmmMat_Mp2LpPr FmmMat_Mp2Lp::create(){
		//return ShIListPr(new IList);
		return std::make_shared<FmmMat_Mp2Lp>();
	}

	// set number of expansions
	// determines accuracy of calculation
	void FmmMat_Mp2Lp::set_num_exp(const int num_exp){
		num_exp_=num_exp;
	}

	// for unsigned input
	void FmmMat_Mp2Lp::set_num_exp(const arma::uword num_exp){
		num_exp_=static_cast<int>(num_exp);
	}

	// Assemble multipole-to-localpole matrix
	void FmmMat_Mp2Lp::calc_matrix(const arma::Mat<fltp> &dR){
		// check input
		assert(num_exp_>0);
		assert(dR.n_rows==3);
		assert(dR.is_finite());

		// set number of matrices
		num_mat_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		arma::Mat<fltp> dRsph = Extra::cart2sph(dR);

		// calculate harmonic (note the 2*num_exp here)
		Spharm L(2*num_exp_,dRsph);
		
		// calculate factorials (note the 2*2*num_exp here)
		const arma::Col<fltp> facs = Extra::factorial(2*2*num_exp_); // longer than normal

		// set matrix array
		int N = Extra::polesize(num_exp_);
		
		// allocate matrices
		M_.set_size(1,num_mat_);
		for (arma::uword i=0;i<num_mat_;i++){
			M_(i).set_size(N,N);
		}

		// allocate indexlist
		arma::Mat<int> jknm(N*N,4);

		// index list creation
		// walk over target
		int p = 0; 
		for (int j=0; j<=num_exp_;j++){
			for (int k=-j;k<=j;k++){
				// walk over source
				for (int n=0;n<=num_exp_;n++){
					for (int m=-n;m<=n;m++){
						// set indices to list
						jknm(p,0) = j; jknm(p,1) = k; 
						jknm(p,2) = n; jknm(p,3) = m;
						p++;
					}
				}
			}
		}
		
		// sanity check
		assert(p==N*N);

		// run over indices and calculate in parallel
		cmn::parfor(0, N*N, true,[&](arma::uword i, int) { // second int is CPU number
			// get indices
			int j = jknm(i,0); int k = jknm(i,1); int n = jknm(i,2); int m = jknm(i,3);

			// constant prefactor
			arma::Mat<std::complex<fltp> > cst = 
				Extra::Anm(facs,n,m)*
				Extra::Anm(facs,j,k)*
				Extra::ipown(std::abs(k-m)-std::abs(k)-std::abs(m))*
				L.get_nm(j+n,m-k)/
				std::pow(-1,n)/
				Extra::Anm(facs,j+n,m-k)/
				arma::pow(dRsph.row(0),j+n+1); // rho = dRsph.row(0)

			// add calculated values to the different matrices
			for (arma::uword q=0;q<num_mat_;q++){
				M_(q)(Extra::nm2fidx(j,k),Extra::nm2fidx(n,m)) = cst(q);
			}
		});

		// check for nans
		#ifndef NDEBUG
		for(arma::uword i=0;i<M_.n_elem;i++)
			assert(M_(i).is_finite());
		#endif
	}


	// multipole to localpole translation without matrix
	arma::Mat<std::complex<fltp> > FmmMat_Mp2Lp::calc_direct(
		const arma::Mat<std::complex<fltp> > &Mp, const arma::Mat<fltp> &dR, 
		const int num_exp, const arma::uword num_dim){
		// check input
		assert(dR.n_rows==3);
		assert(dR.is_finite());

		// set number of matrices
		arma::uword num_translations = dR.n_cols;
		int polesize = Extra::polesize(num_exp);

		// convert to spherical coordinates (rho,theta,phi)
		arma::Mat<fltp> dRsph = Extra::cart2sph(dR);

		// calculate harmonic (note the 2*num_exp here)
		Spharm L(2*num_exp,dRsph);
		
		// calculate factorials (note the 2*2*num_exp here)
		const arma::Col<fltp> facs = Extra::factorial(2*2*num_exp); // longer than normal

		// allocate indexlist
		arma::Mat<int> jk(polesize,2);

		// index list creation
		// walk over target
		int p = 0; 
		for (int j=0; j<=num_exp;j++){
			for (int k=-j;k<=j;k++){
				// set indices to list
				jk(p,0) = j; jk(p,1) = k; 
				p++;
			}
		}

		// sanity check
		assert(p==polesize);

		// reshape multipoles
		arma::Mat<std::complex<fltp> > Mpr = arma::reshape(Mp,num_dim*polesize,num_translations);

		// allocate output
		arma::Mat<std::complex<fltp> > Lp(num_dim*polesize,num_translations,arma::fill::zeros);

		// run over indices and calculate in parallel
		cmn::parfor(0, polesize, false,[&](arma::uword i, int) { // second int is CPU number
			// get indices
			int j = jk(i,0); int k = jk(i,1);

			// walk over source
			for (int n=0;n<=num_exp;n++){
				for (int m=-n;m<=n;m++){
					// constant prefactor
					arma::Row<std::complex<fltp> > cst = 
						Extra::Anm(facs,n,m)*
						Extra::Anm(facs,j,k)*
						Extra::ipown(std::abs(k-m)-std::abs(k)-std::abs(m))*
						L.get_nm(j+n,m-k)/
						std::pow(-1,n)/
						Extra::Anm(facs,j+n,m-k)/
						arma::pow(dRsph.row(0),j+n+1); // rho = dRsph.row(0)

					// add calculated values to the different matrices
					for(arma::uword q=0;q<num_dim;q++){
						// get indexes
						const arma::uword tidx = Extra::nm2fidx(j,k) + q*polesize;
						const arma::uword sidx = Extra::nm2fidx(n,m) + q*polesize;

						// perform translation row-by-row
						Lp.row(tidx) += cst%Mpr.row(sidx);
					}
				}
			}
		});

		// reshape localpole
		Lp.reshape(polesize,num_dim*num_translations);

		// return localpole
		return Lp;
	}



	// get entire stored matrix
	const arma::field<arma::Mat<std::complex<fltp> > >& FmmMat_Mp2Lp::get_matrix() const{
		return M_;
	}


	// get stored matrix at index k
	const arma::Mat<std::complex<fltp> >& FmmMat_Mp2Lp::get_matrix(
		const arma::uword k) const{
		assert(!M_.is_empty());
		return M_(k);
	}

	// apply calculated transformation matrix to multipole
	arma::Mat<std::complex<fltp> > FmmMat_Mp2Lp::apply(
		const arma::uword k, const arma::Mat<std::complex<fltp> > &Mp) const{

		// check input
		assert(Extra::polesize(num_exp_)==static_cast<int>(Mp.n_rows));
		assert(k<num_mat_);

		// calculate matrix vector product(s)
		return get_matrix(k)*Mp;
	}

	// apply calculated transformation matrix 
	// to multipole with distance scaling matrix
	arma::Mat<std::complex<fltp> > FmmMat_Mp2Lp::apply(
		const arma::uword k, const arma::Mat<std::complex<fltp> > &Mp,
		const FmmMat_Mp2Lp_dist &Mdist) const{

		// check input
		assert(Extra::polesize(num_exp_)==static_cast<int>(Mp.n_rows));
		assert(k<num_mat_);

		// calculate matrix vector product(s)
		return Mdist.apply(get_matrix(k))*Mp;
	}


	// display the values stored in the matrix
	void FmmMat_Mp2Lp::display() const{
		for (arma::uword k=0;k<num_mat_;k++){
			display(k);
		}
	}

	// display the values stored in the matrix
	void FmmMat_Mp2Lp::display(const arma::uword k) const{
		// multipole matrix
		printf("Multipole to Localpole matrix\n");

		// header displaying which harmonic
		if(num_mat_>1){
			int boxwidth = Extra::polesize(num_exp_)*(6+1)-1;
			for (int j=0;j<boxwidth;j++){printf("-");}; printf("\n");
			printf("Matrix with index %03llu/%03llu\n",k,num_mat_);
		}   

		// plot sparse matrix
		cmn::Extra::display_mat(get_matrix(k));
	}

	// number of non-zeros in each matrix
	arma::uword FmmMat_Mp2Lp::get_num_mat() const{
		return num_mat_;
	}

	// number of non-zeros in each matrix
	int FmmMat_Mp2Lp::get_num_exp() const{
		return num_exp_;
	}

	// number of non-zeros in each matrix
	// (its actually a dense matrix)
	arma::uword FmmMat_Mp2Lp::num_nz(const int num_exp){
		return Extra::polesize(num_exp)*Extra::polesize(num_exp);
	}

	// display in log
	void FmmMat_Mp2Lp::display(cmn::ShLogPr &lg) const{
		// get statistics;
		arma::uword mat_size = Extra::polesize(num_exp_);

		// estimated memory
		const fltp mp2lp_memory = 2*(num_mat_*mat_size*mat_size*64.0/8)/1024/1024; 
		
		// display statistics
		lg->msg(2,"multipole to multipole %s(Mp2Lp)%s\n",KGRN,KNRM);
		lg->msg("number of matrices: %s%llu%s\n",KYEL,num_mat_,KNRM);
		lg->msg("matrix type: %sDENSE_CX%s\n",KGRN,KNRM);
		lg->msg("matrix size: %s%llu%s X %s%llu%s\n",KYEL,mat_size,KNRM,KYEL,mat_size,KNRM);
		lg->msg(-2,"estimated memory: %s%.2f%s [MB]\n",KYEL,mp2lp_memory,KNRM);
		lg->newl();
	}


	// default constructor
	FmmMat_Lp2Lp::FmmMat_Lp2Lp(){

	}

	// constructor
	FmmMat_Lp2Lp::FmmMat_Lp2Lp(const int num_exp, const arma::Mat<fltp> &dR){
		assert(dR.n_rows==3);
		set_num_exp(num_exp);
		calc_matrix(dR);
	}

	// factory
	ShFmmMat_Lp2LpPr FmmMat_Lp2Lp::create(){
		return std::make_shared<FmmMat_Lp2Lp>();
	}

	// factory
	ShFmmMat_Lp2LpPr FmmMat_Lp2Lp::create(const int num_exp, const arma::Mat<fltp> &dR){
		return std::make_shared<FmmMat_Lp2Lp>(num_exp,dR);
	}

	// set number of expansions
	// determines accuracy of calculation
	void FmmMat_Lp2Lp::set_num_exp(const int num_exp){
		num_exp_=num_exp;
	}

	// for unsigned input
	void FmmMat_Lp2Lp::set_num_exp(const arma::uword num_exp){
		num_exp_=static_cast<int>(num_exp);
	}

	// Assemble localpole-to-localpole matrix
	void FmmMat_Lp2Lp::calc_matrix(const arma::Mat<fltp> &dR){
		// check input
		assert(num_exp_>0);
		assert(dR.n_rows==3);
		assert(dR.is_finite());

		// set number of matrices
		num_mat_ = dR.n_cols;

		// convert to spherical coordinates (rho,theta,phi)
		arma::Mat<fltp> dRsph = Extra::cart2sph(dR);

		// calculate harmonic
		Spharm L(num_exp_,dRsph);
		
		// calculate factorials
		const arma::Col<fltp> facs = Extra::factorial(2*num_exp_);

		// calculate number of non-zero elements in Lp2Lp matrix
		int Nnz = num_nz(num_exp_);

		// allocate vectors for fast matrix assembly
		arma::umat loc(2,Nnz);
		arma::Mat<std::complex<fltp> > val(Nnz,num_mat_);

		// calculate values for matrix
		// walk over target pole
		int idx = 0;
		// iterate over target pole
		for (int j=0;j<=num_exp_;j++){
			for (int k=-j;k<=j;k++){
				// walk over source pole
				for (int n=j;n<=num_exp_;n++){
					// calculate rho^(n-j)
					arma::Mat<fltp> rhonminj = arma::pow(dRsph.row(0),n-j);
					for (int m=std::max(-n,k-n+j);m<=std::min(n,k+n-j);m++){
						// common factor
						arma::Row<std::complex<fltp> > cst = 
							Extra::Anm(facs,n-j,m-k)*
							Extra::Anm(facs,j,k)*
							rhonminj*
							Extra::ipown(std::abs(m)-std::abs(m-k)-std::abs(k))%
							L.get_nm(n-j,m-k)/
							std::pow(-1,n+j)/
							Extra::Anm(facs,n,m);

						// add translated multipole
						loc(0,idx) = Extra::nm2fidx(j,k);
						loc(1,idx) = Extra::nm2fidx(n,m);
						val.row(idx) = cst;
						idx += 1;
					}
				}
			}
		}

		// check matrix filling
		assert(idx==Nnz);

		// set matrix array
		M_.set_size(1,num_mat_);

		// insert columns into sparse matrix
		int N = Extra::polesize(num_exp_);
		for (arma::uword i=0;i<num_mat_;i++){
			M_(i) = arma::SpMat<std::complex<fltp> >(loc,val.col(i),N,N,true,true);
			//assert(M_(i).n_nonzero==Nnz);

			// check for nans
			assert(M_(i).is_finite());
		}

	}

	// get matrix with index k
	const arma::SpMat<std::complex<fltp> >& FmmMat_Lp2Lp::get_matrix(const arma::uword k) const{
		return M_(k);
	}


	// apply calculated transformation matrix to multipole
	arma::Mat<std::complex<fltp> > FmmMat_Lp2Lp::apply(
		const arma::uword k, const arma::Mat<std::complex<fltp> > &Lp) const{
		// forward request with N=0
		return apply(k,Lp,0);
	}


	// apply calculated transformation matrix to multipole
	arma::Mat<std::complex<fltp> > FmmMat_Lp2Lp::apply(
		const arma::uword k, const arma::Mat<std::complex<fltp> > &Lp,
		const arma::uword N) const{

		// check input
		assert(Extra::polesize(num_exp_)==static_cast<int>(Lp.n_rows));
		assert(k<num_mat_);

		// calculate
		return Extra::matrix_self_multiplication(M_(k),N)*Lp;
	}


	// number of non-zeros in each matrix
	arma::uword FmmMat_Lp2Lp::get_num_mat() const{
		return num_mat_;
	}

	// number of non-zeros in each matrix
	int FmmMat_Lp2Lp::get_num_exp() const{
		return num_exp_;
	}

	// display the values stored in the matrix
	void FmmMat_Lp2Lp::display() const{
		for(arma::uword k=0;k<num_mat_;k++){
			display(k);
		}
	}

	// display the values stored in the matrix
	void FmmMat_Lp2Lp::display(const arma::uword k) const{
		// multipole matrix
		std::printf("Localpole to Localpole matrix\n");

		// header displaying which harmonic
		if(num_mat_>1){
			int boxwidth = Extra::polesize(num_exp_)*(6+1)-1;
			for (int j=0;j<boxwidth;j++){printf("-");}; printf("\n");
			std::printf("Matrix with index %03llu/%03llu\n",k,num_mat_);
		}

		// plot sparse matrix
		cmn::Extra::display_spmat(M_(k));
	}

	// display matrix in log
	void FmmMat_Lp2Lp::display(cmn::ShLogPr &lg) const{
		// get statistics
		arma::uword mat_size = Extra::polesize(num_exp_);
		arma::uword num_nz = FmmMat_Lp2Lp::num_nz(num_exp_);

		// estimated memory (note that armadillo internally 
		// uses a special storage format of which we can not 
		// be sure how much memory it uses)
		fltp lp2lp_memory = (4*num_mat_*num_nz*64.0/8)/1024/1024; 
		fltp lp2lp_sparsity = static_cast<fltp>(num_nz)/(mat_size*mat_size);

		// display statistics
		lg->msg(2,"localpole to localpole %s(Lp2Lp)%s\n",KGRN,KNRM);
		lg->msg("number of matrices: %s%llu%s\n",KYEL,num_mat_,KNRM);
		lg->msg("matrix type: %sSPARSE_CX%s\n",KGRN,KNRM);
		lg->msg("matrix size: %s%llu%s X %s%llu%s\n",KYEL,mat_size,KNRM,KYEL,mat_size,KNRM);
		lg->msg("number of nonzeros: %s%llu%s\n",KYEL,num_nz,KNRM);
		lg->msg("sparsity level: %s%.2f%s [pct]\n",KYEL,100*lp2lp_sparsity,KNRM);
		lg->msg(-2,"estimated memory: %s%.2f%s [MB]\n",KYEL,lp2lp_memory,KNRM);
		lg->newl();
	}

	// number of non-zeros in each matrix
	int FmmMat_Lp2Lp::num_nz(const int num_exp){
		int Nnz = 0;
		for (int n=0;n<=num_exp;n++){
			Nnz += (1+2*n)*Extra::polesize(num_exp-n);
		}
		return Nnz;
	}

}}