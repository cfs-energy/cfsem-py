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
namespace rat{namespace cmn{

	// rounding function for C++97
	fltp Extra::round(const fltp d){
		return std::floor(d + 0.5);
	}

	// square function
	fltp Extra::square(const fltp x){
		return x*x;
	}

	// sign function
	fltp Extra::sign(const fltp s){
		fltp ss;
		if(s>0)ss=1.0f;
		else if(s<0)ss=-1.0f;
		else ss=0.0f;
		return ss;
	}

	// lowest common multiplier
	arma::uword Extra::lcm(const arma::Row<arma::uword>& vals, const arma::uword max_iter){
		// take lowest value
		const arma::uword idx = vals.index_min();
		arma::uword cm = vals(idx);
		for(arma::uword i=0;i<max_iter;i++)
			if(arma::all((cm - arma::floor(cm/vals)%vals)==0))return cm; else cm += vals(idx);
		return 0;
	}

	// greatest common divisor recursive calculation
	arma::sword Extra::gcd(const arma::sword a, const arma::sword b){
		const arma::sword aa = std::abs(a);
		const arma::sword ab = std::abs(b);
	
		if(ab==0)return aa;
		return gcd(ab, aa%ab);
	}

	
	// custom cross product that allows 3XN vectors as input (similar to Matlab)
	// this is not possible in the armadillo library
	// https://en.wikipedia.org/wiki/Cross_product
	arma::Mat<fltp> Extra::cross(
		const arma::Mat<fltp> &R1, 
		const arma::Mat<fltp> &R2){
		assert(R1.n_rows==3); assert(R2.n_rows==3); assert(R1.n_cols==R2.n_cols);

		// allocate output
		arma::Mat<fltp> M(R1.n_rows, R1.n_cols);

		// calculate cross product
		M.row(0) = R1.row(1)%R2.row(2) - R1.row(2)%R2.row(1);
		M.row(1) = R1.row(2)%R2.row(0) - R1.row(0)%R2.row(2);
		M.row(2) = R1.row(0)%R2.row(1) - R1.row(1)%R2.row(0);

		// return output matrix
		return M;
	}

	// cross product in 2D returns out of plane component
	arma::Row<fltp> Extra::cross2D(
		const arma::Mat<fltp> &R1,
		const arma::Mat<fltp> &R2){
		assert(R1.n_rows==2); assert(R2.n_rows==2); assert(R2.n_cols==R2.n_cols);
		return R1.row(0)%R2.row(1) - R1.row(1)%R2.row(0);
	}

	// custom dot product that allows 3XN vectors as input (similar to Matlab)
	// this is not possible in the armadillo library
	arma::Row<fltp> Extra::dot(
		const arma::Mat<fltp> &R1, 
		const arma::Mat<fltp> &R2){
		assert(R1.n_rows==2 || R1.n_rows==3);
		assert(R2.n_rows==R1.n_rows); assert(R1.n_cols==R2.n_cols);
		return arma::sum(R1%R2,0);
	}

	// Norm of a vector or 3XN array of vectors
	arma::Row<fltp> Extra::vec_norm(
		const arma::Mat<fltp> &R){
		assert(R.n_rows==3 || R.n_rows==2);
		return arma::vecnorm(R); // armadillo 12.4 has cought up and implemented this for us
	}

	// normalize without checking
	arma::Mat<fltp> Extra::normalize(const arma::Mat<fltp> &V){
		return (V.each_row()/vec_norm(V)).eval();
	}

	// normalize a vector
	arma::Mat<fltp> Extra::normalize(const arma::Mat<fltp> &V, const bool check_zeros){
		const arma::Row<fltp> Vnorm = vec_norm(V);
		arma::Mat<fltp> Vn = V.each_row()/Vnorm;
		if(!check_zeros)return Vn;
		
		// find zeros and fix norm
		Vn.cols(arma::find(Vnorm==RAT_CONST(0.0))).fill(0.0);
		return Vn;
	}

	// angle between vectors in 3XN matrices
	// returns angle in radians
	arma::Row<fltp> Extra::vec_angle(
		const arma::Mat<fltp> &R1, 
		const arma::Mat<fltp> &R2){
		assert((R1.n_rows==3 && R2.n_rows==3) || (R1.n_rows==2 && R2.n_rows==2));
		return arma::acos(arma::clamp(Extra::dot(R1,R2)/(vec_norm(R1)%vec_norm(R2)),-RAT_CONST(1.0),RAT_CONST(1.0)));
	}

	// Norm of a vector or 3XN array of vectors
	arma::Col<fltp>::fixed<3> Extra::unit_vec(const char direction){
		arma::Col<fltp>::fixed<3> vector(arma::fill::zeros);
		if(direction=='x' || direction=='X')vector(0) = RAT_CONST(1.0);
		else if(direction=='y' || direction=='Y')vector(1) = RAT_CONST(1.0);
		else if(direction=='z' || direction=='Z')vector(2) = RAT_CONST(1.0);
		else rat_throw_line("direction not recognized");
		return vector;
	}

	arma::Col<fltp>::fixed<3> Extra::null_vec(){
		return {RAT_CONST(0.0),RAT_CONST(0.0),RAT_CONST(0.0)};
	}

	// convert index to x,y or z char
	char Extra::idx2xyz(const arma::uword idx, const bool capitalize){
		if(capitalize){
			if(idx==0)return 'X';
			if(idx==1)return 'Y';
			if(idx==2)return 'Z';
		}else{
			if(idx==0)return 'x';
			if(idx==1)return 'y';
			if(idx==2)return 'z';
		}
		rat_throw_line("index not recognized");
	}

	// convert x,y or z char to index
	arma::uword Extra::xyz2idx(const char xyz){
		if(xyz=='x' || xyz=='X')return 0;
		if(xyz=='y' || xyz=='Y')return 1;
		if(xyz=='z' || xyz=='Z')return 2;
		rat_throw_line("direction not recognized");
	}

	// check if char is x,y or z
	bool Extra::is_xyz(const char xyz){
		return xyz=='x' || xyz=='y' || xyz=='z' || 
			xyz=='X' || xyz=='Y' || xyz=='Z';
	}

	// display complex armadillo matrix
	void Extra::display_mat(const arma::Mat<std::complex<fltp> > &M){
		// walk over matrix
		std::printf("Real part\n");
		Extra::display_mat(arma::real(M));

		// walk over matrix
		std::printf("Complex part\n");
		Extra::display_mat(arma::imag(M));

		// whiteline
		std::printf("\n");
	}

	// display real armadillo matrix
	void Extra::display_mat(const arma::Mat<fltp> &M){
		// walk over matrix
		for (int i=0;i<static_cast<int>(M.n_rows);i++){
			for (int j=0;j<static_cast<int>(M.n_cols);j++){
				// check for zero
				fltp val = M(i,j);
				std::printf("%+01.0e ",val);
			}
			std::printf("\n");
		}

		// whiteline
		std::printf("\n");
	}

	// display complex sparse armadillo matrix
	void Extra::display_spmat(const arma::SpMat<std::complex<fltp> > &M){
		// walk over matrix
		std::printf("Real part\n");
		Extra::display_spmat(arma::real(M));

		// walk over matrix
		std::printf("Complex part\n");
		Extra::display_spmat(arma::imag(M));

		// whiteline
		std::printf("\n");
	}

	// display real sparse armadillo matrix
	void Extra::display_spmat(const arma::SpMat<fltp> &M){
		// walk over matrix
		for (int i=0;i<static_cast<int>(M.n_rows);i++){
			for (int j=0;j<static_cast<int>(M.n_cols);j++){
				// check for zero
				fltp val = M(i,j);
				if (val==0.0){
					std::printf("...... ");
				}else{
					std::printf("%+01.0e ",val);
				}
			}
			std::printf("\n");
		}

		// whiteline
		std::printf("\n");
	}

	// custom function for finding sections in array
	// for example find_sections([1,1,1,2,2,2,3,3],"first") should give [0,3,6]
	// for example find_sections([1,1,1,2,2,2,3,3],"last") should give [2,5,7]
	arma::Mat<arma::uword> Extra::find_sections(const arma::Row<arma::uword> &M){
		// make sure it is a row vector
		assert(M.n_rows==1);

		// check if matrix is empty
		if(!M.is_empty()){
			// find indices at changing parts
			const arma::Row<arma::uword> idx = arma::find(M.tail_cols(M.n_cols-1)!=M.head_cols(M.n_cols-1)).t(); 
		
			// make first and last indexes
			const arma::Row<arma::uword> idx1 = arma::join_horiz(arma::Mat<arma::uword>(1,1,arma::fill::zeros),idx+1);
			const arma::Row<arma::uword> idx2 = arma::join_horiz(idx,arma::Mat<arma::uword>(1,1,arma::fill::value(M.n_cols-1)));
		
			// return
			return arma::join_vert(idx1,idx2);
		}

		// if matrix is empty return empty indices
		else return arma::Mat<arma::uword>(2,0);
	}

	// set sections
	// takes output from find_sections and creates origin array
	arma::Row<arma::uword> Extra::set_sections(const arma::Mat<arma::uword> &indices){
		// check input
		assert(indices.n_rows==2);

		// get last section
		arma::uword num_elem = indices(1,indices.n_cols-1)+1;
		
		// allocate output
		arma::Row<arma::uword> origin(num_elem);

		// fill output
		for(arma::uword i=0;i<indices.n_cols;i++){
			origin.cols(indices(0,i),indices(1,i)).fill(i);
		}
		
		// return output
		return origin;
	}

	// custom bitshift operation as it is not available in armadillo
	arma::Mat<arma::uword> Extra::bitshift(const arma::Mat<arma::uword> &M, const int nshift){
		// // allocate
		// arma::Mat<arma::uword> bs;

		// // use different function for each option
		// if (nshift>=0){
		// 	bs = arma::floor(M*std::pow(2,nshift));
		// }else{
		// 	bs = arma::floor(M/(1.0/std::pow(2,nshift)));
		// }

		// bitshift without pow
		// arma::Mat<arma::uword> bs = M;
		// if(nshift>=0)for(int i=0;i<nshift;i++)bs*=2;
		// else if(nshift<0)for(int i=0;i<-nshift;i++)bs/=2;

		// bitshift with operator
		// allocate
		arma::Mat<arma::uword> bs(M.n_rows, M.n_cols);

		// positive shift
		if(nshift>0)for(arma::uword i=0;i<M.n_elem;i++)bs.at(i) = M.at(i)<<nshift;

		// negative shift
		else if(nshift<0)for(arma::uword i=0;i<M.n_elem;i++)bs.at(i) = M.at(i)>>(-nshift);

		// no shift
		else bs = M;

		// output result
		return bs;
	}

	// function for generating coordinates arround a center point given by xyz coords
	arma::Mat<fltp> Extra::random_coordinates(
		const fltp xc, 
		const fltp yc, 
		const fltp zc, 
		const fltp size,
		const arma::uword N){

		// create center point
		arma::Col<fltp>::fixed<3> Rc = {xc,yc,zc};

		// call overloaded function
		return Extra::random_coordinates(Rc,size,N);
	}

	// function for generating coordinates arround a center point given by vector
	arma::Mat<fltp> Extra::random_coordinates(
		arma::Col<fltp>::fixed<3> Rc, 
		const fltp size,
		const arma::uword N){

		// create coordinates
		arma::Mat<fltp> R = size*(arma::Mat<fltp>(3,N,arma::fill::randu)-0.5);
		R = R.each_col() + Rc;

		// return output
		return R; 
	}

	// calculate modulus
	arma::Mat<arma::uword> Extra::modulus(
		const arma::Mat<arma::uword> &A, 
		const arma::uword i){
		
		return A - arma::floor(A/i)*i;
	}

	// calculate modulus
	arma::Mat<fltp> Extra::modulus(
		const arma::Mat<fltp> &A, const fltp v){
		
		return A - arma::floor(A/v)*v;
	}

	// reverse field array helper function (static)
	void Extra::reverse_field(arma::field<arma::Mat<fltp> > &A){
		// swap fields
		for(arma::uword i=0;i<static_cast<arma::uword>(std::floor(A.n_elem/2));i++){
			std::swap(A(i), A(A.n_elem-1-static_cast<arma::sword>(i)));
		}

		// swap columns
		for(arma::uword i=0;i<A.n_elem;i++){
			A(i) = arma::fliplr(A(i));
		}
	}

	// build rotation matrix
	arma::Mat<fltp>::fixed<3,3> Extra::create_rotation_matrix(
		const fltp phi, const fltp theta, const fltp psi){

		// setup rotation matrix
		arma::Mat<fltp>::fixed<3,3> Ax,Ay,Az,A;

		// rotation arround x-axis
		Ax.row(0) = arma::Row<fltp>{1.0f,0.0f,0.0f}; 
		Ax.row(1) = arma::Row<fltp>{0.0f,std::cos(phi),-std::sin(phi)}; 
		Ax.row(2) = arma::Row<fltp>{0.0f,std::sin(phi),std::cos(phi)}; 
		
		// rotation arround y-axis
		Ay.row(0) = arma::Row<fltp>{std::cos(theta),0.0f,std::sin(theta)}; 
		Ay.row(1) = arma::Row<fltp>{0.0f,1.0f,0.0f}; 
		Ay.row(2) = arma::Row<fltp>{-std::sin(theta),0.0f,std::cos(theta)};
		
		// rotation arround z-axis
		Az.row(0) = arma::Row<fltp>{std::cos(psi),-std::sin(psi),0.0f}; 
		Az.row(1) = arma::Row<fltp>{std::sin(psi),std::cos(psi),0.0f}; 
		Az.row(2) = arma::Row<fltp>{0.0f,0.0f,1.0f};
		
		// combined rotation matrix
		A = Ax*Ay*Az;

		// return matrix
		return A;
	}

	// build rotation matrix from two vectors
	// the matrix rotates v1 onto v2
	arma::Mat<fltp>::fixed<3,3> Extra::create_rotation_matrix(
		const arma::Col<fltp>::fixed<3> &V1,
		const arma::Col<fltp>::fixed<3> &V2){

		// unit matrix
		const arma::Mat<fltp>::fixed<3,3> I = arma::diagmat(arma::Col<fltp>::fixed<3>(arma::fill::ones));

		// chekc if vectors in same direction
		if(std::abs(arma::as_scalar(cmn::Extra::dot(V1,V2)/(cmn::Extra::vec_norm(V1)%cmn::Extra::vec_norm(V2)))-RAT_CONST(1.0))<RAT_CONST(1e-9))return I;

		// normalize
		const arma::Col<fltp>::fixed<3> V1nrm = V1.each_row()/Extra::vec_norm(V1);
		const arma::Col<fltp>::fixed<3> V2nrm = V2.each_row()/Extra::vec_norm(V2);

		// calculate cross adn dot products
		const arma::Col<fltp>::fixed<3> v = Extra::cross(V1nrm,V2nrm); 
		const fltp s = arma::as_scalar(Extra::vec_norm(v)); // sin of angle
		const fltp c = arma::as_scalar(Extra::dot(V1nrm,V2nrm)); // cos of angle

		// zero degree case
		if(std::abs(c - RAT_CONST(1.0)) < arma::Datum<fltp>::eps)return I;

		// 180 deg case
		if(c<-RAT_CONST(1.0)+arma::Datum<fltp>::eps){
			arma::Col<fltp>::fixed<3> orth = Extra::unit_vec('x');
			if(std::abs(arma::as_scalar(Extra::dot(orth, V1))) > RAT_CONST(0.9))orth = Extra::unit_vec('y');
			return create_rotation_matrix(Extra::cross(V1,orth), arma::Datum<fltp>::pi);
		}

		// matrices
		const arma::Mat<fltp>::fixed<3,3> vx{0,v(2),-v(1), -v(2),0,v(0), v(1),-v(0),0};
		
		// calculate rotation matrix
		const arma::Mat<fltp>::fixed<3,3> M = I + vx + (vx*vx)*(1.0-c)/(s*s);

		// check and return
		assert(is_pure_rotation_matrix(M));
		assert(M.is_finite());
		return M;
	}

	// build rotation matrix for rotation arround axis with vector input
	arma::Mat<fltp>::fixed<3,3> Extra::create_rotation_matrix(
		const arma::Col<fltp>::fixed<3> &V, const fltp alpha){
		return Extra::create_rotation_matrix(V(0),V(1),V(2),alpha);
	}


	// build rotation matrix for rotation arround axis
	// https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis%E2%80%93angle
	arma::Mat<fltp>::fixed<3,3> Extra::create_rotation_matrix(
		const fltp ux, const fltp uy, const fltp uz, const fltp alpha){

		// early out
		if(alpha==RAT_CONST(0.0))return arma::diagmat(arma::Col<fltp>(3,arma::fill::ones));

		// normalize vector
		const fltp ell_vec = std::sqrt(ux*ux + uy*uy + uz*uz);
		const fltp uxn = ux/ell_vec;
		const fltp uyn = uy/ell_vec;
		const fltp uzn = uz/ell_vec;

		// setup rotation matrix
		arma::Mat<fltp>::fixed<3,3> A;

		// setup matrix
		// https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis%E2%80%93angle
		A.row(0) = arma::Row<fltp>::fixed<3>{
			std::cos(alpha)+uxn*uxn*(1-std::cos(alpha)),
			uxn*uyn*(1-std::cos(alpha))-uzn*std::sin(alpha),
			uxn*uzn*(1-std::cos(alpha))+uyn*std::sin(alpha)};
		A.row(1) = arma::Row<fltp>::fixed<3>{
			uyn*uxn*(1-std::cos(alpha))+uzn*std::sin(alpha),
			std::cos(alpha)+uyn*uyn*(1-std::cos(alpha)),
			uyn*uzn*(1-std::cos(alpha))-uxn*std::sin(alpha)};
		A.row(2) = arma::Row<fltp>::fixed<3>{
			uzn*uxn*(1-std::cos(alpha))-uyn*std::sin(alpha),
			uzn*uyn*(1-std::cos(alpha))+uxn*std::sin(alpha),
			std::cos(alpha)+uzn*uzn*(1-std::cos(alpha))};

		// return matrix
		return A;
	}

	// check if something is a rotation matrix
	bool Extra::is_pure_rotation_matrix(const arma::Mat<fltp>::fixed<3,3> &M, const rat::fltp tol){
		const arma::Mat<fltp>::fixed<3,3> I = M.t()*M;
		if(!((I - arma::diagmat(arma::Col<fltp>::fixed<3>(arma::fill::ones))).is_zero(tol)))return false;
		if(std::abs(arma::det(M)-RAT_CONST(1.0))>tol)return false;
		return true;
	}

	// orthogonalize
	arma::Mat<fltp>::fixed<3,3> Extra::gram_schmidt_orthogonalization(arma::Mat<fltp>::fixed<3,3>& V){
		arma::Mat<fltp>::fixed<3,3> U(arma::fill::zeros);
		U.col(0) = Extra::normalize(V.col(0));
		for(arma::uword i=1;i<V.n_cols;i++){
			U.col(i) = V.col(i);
			for(arma::uword j=0;j<i-1;j++){
				U.col(i) = U.col(i) - arma::as_scalar(Extra::dot(U.col(j),U.col(i)))*U.col(j);
			}
			U.col(i) = Extra::normalize(U.col(i));
		}
		return U;
	}

	// orthogonalize using modified Gram-Schmidt
	arma::Mat<fltp>::fixed<3,3> Extra::modified_gram_schmidt_orthogonalization(arma::Mat<fltp>::fixed<3,3>& V){
		arma::Mat<fltp>::fixed<3,3> U(arma::fill::zeros);
		arma::Mat<fltp>::fixed<3,3> Q(arma::fill::zeros);
		
		for(arma::uword i = 0; i < V.n_cols; i++){
			Q.col(i) = V.col(i);
			for(arma::uword j = 0; j < i; j++){
				Q.col(i) = Q.col(i) - arma::as_scalar(Extra::dot(U.col(j), Q.col(i))) * U.col(j);
			}
			U.col(i) = Extra::normalize(Q.col(i));
		}
		return U;
	}

	// get vector and angle from rotation matrix
	arma::Col<fltp>::fixed<4> Extra::rotmatrix2angle(
		const arma::Mat<fltp>::fixed<3,3> &M, const rat::fltp tol){

		// check if matrix is special
		if(M.is_symmetric(tol)){
			// check for identity matrix
			if(((M - arma::diagmat(arma::Col<fltp>::fixed<3>(arma::fill::ones))).is_zero(tol))){
				// return with arbitrary vector and zero rotation
				return arma::Col<fltp>::fixed<4>{0,1,0,0};
			}

			// symmetric but not identity matrix
			else{
				// helpers
				rat::fltp x,y,z;
				const rat::fltp hsqrt2 = std::sqrt(RAT_CONST(2.0))/2;
				const rat::fltp xx = (M(0,0)+1)/2, yy = (M(1,1)+1)/2, zz = (M(2,2)+1)/2;
				const rat::fltp xy = (M(0,1)+M(1,0))/4, xz = (M(0,2)+M(2,0))/4, yz = (M(1,2)+M(2,1))/4;

				// m[0][0] is the largest diagonal term
				if ((xx > yy) && (xx > zz)){
					if(xx<tol){x = RAT_CONST(0.0); y = hsqrt2; z = hsqrt2;}
					else{x = std::sqrt(xx); y = xy/x; z = xz/x;}
				}

				// m[1][1] is the largest diagonal term
				else if (yy > zz) {
					if(yy<tol){x = hsqrt2; y = RAT_CONST(0.0); z = hsqrt2;}
					else{y = std::sqrt(yy); x = xy/y; z = yz/y;}
				}

				// m[2][2] is the largest diagonal term
				else{
					if(zz<tol){x = hsqrt2; y = hsqrt2; z = RAT_CONST(0.0);}
					else{z = std::sqrt(zz); x = xz/z; y = yz/z;}
				}

				// store in vector and return
				return arma::Col<fltp>::fixed<4>{x,y,z,arma::Datum<fltp>::pi};
			}
		}

		// regular method for extracting angle
		else{
			// rotation angle
			const fltp angle = std::acos((M(0,0) + M(1,1) + M(2,2) - RAT_CONST(1.0))/2);
			
			// helper variable
			const fltp v = std::sqrt(std::pow(M(2,1) - M(1,2),2) + std::pow(M(0,2) - M(2,0),2) + std::pow(M(1,0) - M(0,1),2));
			
			// rotation vector
			const fltp x = (M(2,1) - M(1,2))/v, y = (M(0,2) - M(2,0))/v, z = (M(1,0) - M(0,1))/v;

			// store in vector and return
			return arma::Col<fltp>::fixed<4>{x,y,z,angle};
		}
	}


	// interpolation for one scalar
	fltp Extra::interp1(
		const arma::Mat<fltp> &X, 
		const arma::Mat<fltp> &Y, 
		const fltp XI,
		const char *type,
		const bool extrap){

		// check input
		assert(X.is_sorted());

		// convert to array
		arma::Col<fltp> xi(1),yi(1); xi(0) = XI;

		// perform linear interpolation
		Extra::interp1(
			arma::vectorise(X),
			arma::vectorise(Y),
			xi,yi,type,extrap);

		// return value
		return yi(0);
	}


	// // get sort index (as defined in armadillo) for matrix for each row
	// // this can be implemented by using multiple sort_index_safe
	// arma::Row<arma::uword> Extra::matrix_sort_index_by_row(
	// 	arma::Mat<arma::uword> M){

	// 	// increment
	// 	M += 1;

	// 	// walk over rows of M
	// 	arma::uword mult_val = 1;
	// 	for(arma::uword i=0;i<M.n_rows;i++){
	// 		arma::uword myrow = M.n_rows-i-1;
	// 		arma::uword maxval = arma::as_scalar(
	// 			arma::max(M.row(myrow),1));
	// 		M.row(myrow) *= mult_val;
	// 		mult_val *= (maxval+1);
	// 	}

	// 	// sort normally
	// 	return arma::sort_index(arma::sum(M,0)).t();
	// }

	// interpolation and or extrapolation
	// only linear, x does not need to be linearly increasing
	void Extra::interp1(
		const arma::Col<fltp> &x,
		const arma::Col<fltp> &y,
		const fltp &xi, fltp &yi,
		const char *type,
		const bool extrap){
		
		// convert to arrays
		arma::Col<fltp> yii;
		Extra::interp1(x,y,arma::Col<fltp>{xi},yii,type,extrap);
		yi = arma::as_scalar(yii);
	}


	// interpolation and or extrapolation
	// only linear, x does not need to be linearly increasing
	void Extra::interp1(
		const arma::Col<fltp> &x,
		const arma::Col<fltp> &y,
		const arma::Col<fltp> &xi,
		arma::Col<fltp> &yi,
		const char *type,
		const bool extrap){

		// check input
		assert(xi.is_finite()); assert(x.is_finite()); assert(y.is_finite());

		// end index
		arma::uword end = x.n_elem-1;

		// normal interpolation
		arma::interp1(x,y,xi,yi,type);

		// if extrapolation
		if(extrap==true){
			// extrapolation beyond end
			const arma::Col<arma::uword> idx1 = arma::find(xi>x(x.n_elem-1));
			yi(idx1) = y(end) + (xi(idx1)-x(end))*((y(end)-y(end-1))/(x(end)-x(end-1)));

			// extrapolation before start
			const arma::Col<arma::uword> idx2 = arma::find(xi<x(0));
			yi(idx2) = y(0) - (x(0)-xi(idx2))*((y(1)-y(0))/(x(1)-x(0)));
		}
	}

	// interpolation assuming X is monotonically increasing
	// but not XI like armadillo 
	void Extra::lininterp1f(
		const arma::Col<fltp> &x,
		const arma::Col<fltp> &y,
		const arma::Col<fltp> &xi,
		arma::Col<fltp> &yi,
		const bool extrap){

		// check input
		assert(x.n_elem==y.n_elem);
		assert(!x.is_empty()); assert(!y.is_empty());
		assert(x.is_sorted("strictascend"));

		// allocate output
		yi.set_size(xi.n_elem);

		// interpolation
		{
			// find indexes in range
			const arma::Col<arma::uword> idx_mid = arma::find(xi>x.front() && xi<x.back());

			// check if any exist
			if(!idx_mid.is_empty()){
				// get range for X
				const fltp xrange = x.back() - x.front();

				// find position relative to start and end
				const arma::Col<fltp> xrel = (xi(idx_mid) - x.front())/xrange;
				
				// convert this to an index
				arma::Col<arma::uword> xidx = arma::conv_to<arma::Col<arma::uword> >::from(xrel*static_cast<fltp>(x.n_elem-1));

				// // keep in bounds
				// xidx(arma::find(xidx==x.n_elem)) = xidx(arma::find(xidx==x.n_elem))-1;

				// get interpolation coordinates
				const arma::Col<fltp> x1 = x(xidx);
				const arma::Col<fltp> x2 = x(xidx+1);
				const arma::Col<fltp> y1 = y(xidx);
				const arma::Col<fltp> y2 = y(xidx+1);

				// linear interpolation
				yi(idx_mid) = (xi(idx_mid)-x1)%((y2-y1)/(x2-x1)) + y1;
			}
		}

		// extrapolation
		if(extrap){
			// check if extrapolation possible
			assert(x.n_elem>=2); assert(y.n_elem>=2);

			// find indexes beyond range
			const arma::Col<arma::uword> idx_beyond = arma::find(xi>=x.back());

			// check if any exist
			if(!idx_beyond.is_empty()){
				// get last two points
				const fltp x1 = x(x.n_elem-2);
				const fltp x2 = x(x.n_elem-1);
				const fltp y1 = y(y.n_elem-2);
				const fltp y2 = y(y.n_elem-1);

				// calculate slope
				const fltp b = (y2-y1)/(x2-x1);

				// extrapolate
				yi(idx_beyond) = y2 + b*(xi(idx_beyond)-x2);
			}

			// find indexes before range
			const arma::Col<arma::uword> idx_before = arma::find(xi<=x.front());

			// check if any exist
			if(!idx_before.is_empty()){
				// get last two points
				const fltp x1 = x(0);
				const fltp x2 = x(1);
				const fltp y1 = y(0);
				const fltp y2 = y(1);

				// calculate slope
				const fltp b = (y2-y1)/(x2-x1);

				// extrapolate
				yi(idx_before) = y1 - b*(x1-xi(idx_before));
			}
		}

		// extrapolation with fixed value
		else{
			yi(arma::find(xi<x.front() || xi>x.back())).fill(0.0);
		}
	}

	// interp2 overload for single value return
	fltp Extra::interp2(const arma::Col<fltp>& x,
		const arma::Col<fltp>& y,
		const arma::Mat<fltp>& v,
		const fltp xi,
		const fltp yi,
		const bool extrap) {

		// convert to arrays
		arma::Col<fltp> voo;
		Extra::interp2(x, y, v, arma::Col<fltp>{xi}, arma::Col<fltp>{yi}, voo, extrap);
		return arma::as_scalar(voo);
	}

	// interp2 overload for single value reference
	void Extra::interp2(const arma::Col<fltp>& x,
		const arma::Col<fltp>& y,
		const arma::Mat<fltp>& v,
		const fltp& xi,
		const fltp& yi,
		fltp& vo,
		const bool extrap) {

		// convert to arrays
		arma::Col<fltp> voo;
		Extra::interp2(x, y, v, arma::Col<fltp>{xi}, arma::Col<fltp>{yi}, voo, extrap);
		vo = arma::as_scalar(voo);
	}

	// interp2
	// https://en.wikipedia.org/wiki/Bilinear_interpolation
	// decided not to wrap to armadillo's interp2 because of the
	// constraint that XI,YI must be monotonically increasing
	void Extra::interp2(const arma::Col<fltp>& x,
		const arma::Col<fltp>& y,
		const arma::Mat<fltp>& v,
		const arma::Col<fltp>& xi,
		const arma::Col<fltp>& yi,
		arma::Col<fltp>& vo,
		const bool extrap) {

		// check input
		assert(xi.n_elem == yi.n_elem);
		assert(x.n_elem == v.n_cols);
		assert(y.n_elem == v.n_rows);
		assert(x.is_sorted());
		assert(y.is_sorted());

		// allocate
		vo.set_size(xi.n_elem);

		// find indices in range
		arma::Col<arma::uword> idx_mid = arma::find((xi >= x.front() && xi < x.back() && yi >= y.front() && yi < y.back()));

		if(!idx_mid.is_empty()) {

			// get grid indices for the valid points
			arma::Col<arma::uword> xidx(idx_mid.n_elem);
			arma::Col<arma::uword> yidx(idx_mid.n_elem);
			arma::Col<fltp> xmid = xi(idx_mid);
			arma::Col<fltp> ymid = yi(idx_mid);
			for(arma::uword i = 0; i < idx_mid.n_elem; i++) {
				xidx.row(i) = arma::find(x > xmid(i), 1, "first") - 1;
				yidx.row(i) = arma::find(y > ymid(i), 1, "first") - 1;
			}

			// slope
			arma::Col<fltp> xd = (xi(idx_mid) - x(xidx)) / (x(xidx + 1) - x(xidx));
			arma::Col<fltp> yd = (yi(idx_mid) - y(yidx)) / (y(yidx + 1) - y(yidx));

			// bounding plane

			// v01 --- v11
			//  |       |
			// v00 --- v10

			arma::Col<fltp> v00(idx_mid.n_elem);
			arma::Col<fltp> v10(idx_mid.n_elem);
			arma::Col<fltp> v01(idx_mid.n_elem);
			arma::Col<fltp> v11(idx_mid.n_elem);

			// fill manually
			for(arma::uword i = 0; i < idx_mid.n_elem; i++) {
				v00.row(i) = v(yidx(i), xidx(i));
				v10.row(i) = v(yidx(i), xidx(i) + 1);
				v01.row(i) = v(yidx(i) + 1, xidx(i));
				v11.row(i) = v(yidx(i) + 1, xidx(i) + 1);
			}

			// bounding line
			arma::Col<fltp> v0 = v00 % (1 - yd) + v10 % yd;
			arma::Col<fltp> v1 = v01 % (1 - yd) + v11 % yd;

			// points
			vo(idx_mid) = v0 % (1 - xd) + v1 % xd;
		}

		// extrapolation
		if(extrap) {
			// @hey, not implemented
			rat_throw_line("interp2 extrapolation not implemented");

		} else {
			arma::Col<arma::uword> idx_mid = arma::find(xi < x.front() || xi > x.back() || yi < y.front() || yi > y.back());
			vo(idx_mid).fill(0.0);
		}
	}

	// assumes the input arrays are linearly and monotonically spaced
	// does not check!
	void Extra::lininterp2f(
		const arma::Col<fltp>& x,
		const arma::Col<fltp>& y,
		const arma::Mat<fltp>& v,
		const arma::Col<fltp>& xi,
		const arma::Col<fltp>& yi,
		arma::Col<fltp>& vi,
		const bool extrap) {
		// check input
		assert(x.n_elem == v.n_cols);
		assert(y.n_elem == v.n_rows);
		assert(xi.n_elem == yi.n_elem);
		assert(!x.is_empty());
		assert(!y.is_empty());
		assert(!v.is_empty());
		assert(x.is_sorted("strictascend"));
		assert(y.is_sorted("strictascend"));

		// allocate

		arma::Col<fltp> vi0, vi1;
		vi0.set_size(xi.n_elem);
		vi1.set_size(xi.n_elem);
		vi.set_size(xi.n_elem);

		// interpolation
		{

			// find indices in range
			// checks if within bounds
			arma::Col<arma::uword> idx_mid =
				arma::find(xi >= x.front() && xi < x.back() && yi >= y.front() && yi < y.back());

			// check if exists
			if(!idx_mid.is_empty()) {

				// get range
				const fltp xrange = x.back() - x.front();
				const fltp yrange = y.back() - y.front();

				// find relative positions
				// since the x vector is monotonic and linear we can do this
				// this is 0->1
				// in 2D these relative positions are different
				// thus we get different xidx and yidx here
				const arma::Col<fltp> xrel = (xi(idx_mid) - x.front()) / xrange;
				const arma::Col<fltp> yrel = (yi(idx_mid) - y.front()) / yrange;

				// convert to indices
				arma::Col<arma::uword> xidx =
					arma::conv_to<arma::Col<arma::uword>>::from(xrel * static_cast<fltp>(x.n_elem - 1));
				arma::Col<arma::uword> yidx =
					arma::conv_to<arma::Col<arma::uword>>::from(yrel * static_cast<fltp>(y.n_elem - 1));

				assert(xidx.n_elem == yidx.n_elem);

				// C01 --- C11
				//  |       |
				// C00 --- C10
				// get bounding values
				// get interpolation coordinates
				const arma::Col<fltp> x0 = x(xidx);
				const arma::Col<fltp> x1 = x(xidx + 1);
				const arma::Col<fltp> y0 = y(yidx);
				const arma::Col<fltp> y1 = y(yidx + 1);

				// note the input matrices are transpose
				// to access vectors
				arma::Col<fltp> v00, v01, v10, v11;
				v00.set_size(xidx.n_elem);
				v01.set_size(xidx.n_elem);
				v10.set_size(xidx.n_elem);
				v11.set_size(xidx.n_elem);

				// fill vectors manually
				for(int i = 0; i < static_cast<int>(xidx.n_elem); i++) {
					v00(i) = v(yidx(i), xidx(i));
					v01(i) = v(yidx(i) + 1, xidx(i));
					v10(i) = v(yidx(i), xidx(i) + 1);
					v11(i) = v(yidx(i) + 1, xidx(i) + 1);
				}

				assert(v00.n_elem == v01.n_elem);
				assert(v10.n_elem == v11.n_elem);

				// linear interpolation of slice
				vi0 = v00 + (xi(idx_mid) - x0) % ((v10 - v00) / (x1 - x0));
				vi1 = v01 + (xi(idx_mid) - x0) % ((v11 - v01) / (x1 - x0));

				// linearly interpolate in y to points
				vi(idx_mid) = vi0 + (yi(idx_mid) - y0) % ((vi1 - vi0) / (y1 - y0));
			}

			// extrapolations
			if(extrap) {
				// @hey, not implemented
				assert(0);
			} else {
				vi(arma::find(xi < x.front() || xi > x.back() || yi < y.front() || yi > y.back()))
					.fill(0.0);
			}
		}
	}

	// scalar output
	fltp Extra::interp3(const arma::Col<fltp>& x,
		const arma::Col<fltp>& y,
		const arma::Col<fltp>& z,
		const arma::Cube<fltp>& v,
		fltp xi,
		fltp yi,
		fltp zi,
		const bool extrap) {

		// convert to arrays
		arma::Col<fltp> voo;
		Extra::interp3(x, y, z, v, arma::Col<fltp>{xi}, arma::Col<fltp>{yi}, arma::Col<fltp>{zi}, voo, extrap);
		return arma::as_scalar(voo);
	}

	// interp3 void overload for single input
	void Extra::interp3(const arma::Col<fltp>& x,
		const arma::Col<fltp>& y,
		const arma::Col<fltp>& z,
		const arma::Cube<fltp>& v,
		const fltp& xi,
		const fltp& yi,
		const fltp& zi,
		fltp& vo,
		const bool extrap) {

		// convert to arrays
		arma::Col<fltp> voo;
		Extra::interp3(x, y, z, v, arma::Col<fltp>{xi}, arma::Col<fltp>{yi}, arma::Col<fltp>{zi}, voo, extrap);
		vo = arma::as_scalar(voo);
	}

	// trilinear interpolation
	// wiki https://en.wikipedia.org/wiki/Trilinear_interpolation
	// unoptimized but does not require linearly spaced inputs
	void Extra::interp3(const arma::Col<fltp>& x,
		const arma::Col<fltp>& y,
		const arma::Col<fltp>& z,
		const arma::Cube<fltp>& v,
		const arma::Col<fltp>& xi,
		const arma::Col<fltp>& yi,
		const arma::Col<fltp>& zi,
		arma::Col<fltp>& vo,
		const bool extrap) {

		// check input
		assert(xi.n_elem == yi.n_elem && xi.n_elem == zi.n_elem);
		assert(x.n_elem == v.n_cols);
		assert(y.n_elem == v.n_rows);
		assert(z.n_elem == v.n_slices);
		assert(x.is_sorted());
		assert(y.is_sorted());
		assert(z.is_sorted());

		// allocate
		vo.set_size(xi.n_elem);

		// find indices in range
		arma::Col<arma::uword> idx_mid = arma::find(
			(xi >= x.front() && xi < x.back() && yi >= y.front() && yi < y.back() && zi >= z.front() && zi < z.back()));

		// points in range
		if(!idx_mid.is_empty()) {

			// get grid indices for the valid points
			arma::Col<arma::uword> xidx(idx_mid.n_elem);
			arma::Col<arma::uword> yidx(idx_mid.n_elem);
			arma::Col<arma::uword> zidx(idx_mid.n_elem);
			arma::Col<fltp> xmid = xi(idx_mid);
			arma::Col<fltp> ymid = yi(idx_mid);
			arma::Col<fltp> zmid = zi(idx_mid);

			for(arma::uword i = 0; i < idx_mid.n_elem; i++) {
				xidx.row(i) = arma::find(x > xmid(i), 1, "first") - 1;
				yidx.row(i) = arma::find(y > ymid(i), 1, "first") - 1;
				zidx.row(i) = arma::find(z > zmid(i), 1, "first") - 1;
			}


			// slope
			arma::Col<fltp> xd = (xi(idx_mid) - x(xidx)) / (x(xidx + 1) - x(xidx));
			arma::Col<fltp> yd = (yi(idx_mid) - y(yidx)) / (y(yidx + 1) - y(yidx));
			arma::Col<fltp> zd = (zi(idx_mid) - z(zidx)) / (z(zidx + 1) - z(zidx));

			// get bounding cubes
			/*
		 C011    C111
		  \       \
		 |  C001 --- C101
		 C010 |   C110 |
		  \   |      \ |
		   C000 --- C100
		   */
			arma::Col<fltp> v000(idx_mid.n_elem);
			arma::Col<fltp> v100(idx_mid.n_elem);
			arma::Col<fltp> v001(idx_mid.n_elem);
			arma::Col<fltp> v101(idx_mid.n_elem);
			arma::Col<fltp> v010(idx_mid.n_elem);
			arma::Col<fltp> v110(idx_mid.n_elem);
			arma::Col<fltp> v011(idx_mid.n_elem);
			arma::Col<fltp> v111(idx_mid.n_elem);

			// fill manually
			for(arma::uword i = 0; i < idx_mid.n_elem; i++) {
				v000.row(i) = v(yidx(i), xidx(i), zidx(i));
				v100.row(i) = v(yidx(i), xidx(i) + 1, zidx(i));
				v001.row(i) = v(yidx(i), xidx(i), zidx(i) + 1);
				v101.row(i) = v(yidx(i), xidx(i) + 1, zidx(i) + 1);
				v010.row(i) = v(yidx(i) + 1, xidx(i), zidx(i));
				v110.row(i) = v(yidx(i) + 1, xidx(i) + 1, zidx(i));
				v011.row(i) = v(yidx(i) + 1, xidx(i), zidx(i) + 1);
				v111.row(i) = v(yidx(i) + 1, xidx(i) + 1, zidx(i) + 1);
			}

			// bounding plane
			arma::Col<fltp> v00 = v000 % (1 - xd) + v100 % xd;
			arma::Col<fltp> v01 = v001 % (1 - xd) + v101 % xd;
			arma::Col<fltp> v10 = v010 % (1 - xd) + v110 % xd;
			arma::Col<fltp> v11 = v011 % (1 - xd) + v111 % xd;

			// bounding line
			arma::Col<fltp> v0 = v00 % (1 - yd) + v10 % yd;
			arma::Col<fltp> v1 = v01 % (1 - yd) + v11 % yd;

			// points
			vo(idx_mid) = v0 % (1 - zd) + v1 % zd;
		}

		// extrap
		if(extrap) {
			// @hey, not implemented
			rat_throw_line("interp3 extrapolation not implemented");

		} else {
			arma::Col<arma::uword> idx_mid = arma::find(
				xi < x.front() || xi > x.back() || yi < y.front() || yi > y.back() || zi < z.front() || zi > z.back());
			vo(idx_mid).fill(0.0);
		}
	}

	// assumes the input arrays are linearly and monotonically spaced
	// does not check!
	void Extra::lininterp3f(
		const arma::Col<rat::fltp>& x,
		const arma::Col<rat::fltp>& y,
		const arma::Col<rat::fltp>& z,
		const arma::Cube<rat::fltp>& v,
		const arma::Col<rat::fltp>& xi,
		const arma::Col<rat::fltp>& yi,
		const arma::Col<rat::fltp>& zi,
		arma::Col<rat::fltp>& vi,
		const bool extrap) {

		// check input
		assert(x.n_elem == v.n_cols);
		assert(y.n_elem == v.n_rows);
		assert(z.n_elem == v.n_slices);
		assert(xi.n_elem == yi.n_elem);
		assert(xi.n_elem == zi.n_elem);
		assert(!x.is_empty());
		assert(!y.is_empty());
		assert(!z.is_empty());
		assert(!v.is_empty());
		assert(x.is_sorted("strictascend"));
		assert(y.is_sorted("strictascend"));
		assert(z.is_sorted("strictascend"));

		// allocate
		vi.set_size(xi.n_elem);

		// interpolation
		{

			// find indices in range
			// checks if within bounds
			arma::Col<arma::uword> idx_mid = arma::find(xi >= x.front() && xi < x.back() &&
				yi >= y.front() && yi < y.back() && zi >= z.front() && zi < z.back());

			// check if exists
			if(!idx_mid.is_empty()) {

				// get range
				const fltp xrange = x.back() - x.front();
				const fltp yrange = y.back() - y.front();
				const fltp zrange = z.back() - z.front();

				// find relative positions
				// since the x vector is monotonic and linear we can do this
				// this is 0->1
				// in 3D these relative positions are different
				// thus we get different xidx and yidx here
				const arma::Col<fltp> xrel = (xi(idx_mid) - x.front()) / xrange;
				const arma::Col<fltp> yrel = (yi(idx_mid) - y.front()) / yrange;
				const arma::Col<fltp> zrel = (zi(idx_mid) - z.front()) / zrange;

				// convert to indices
				arma::Col<arma::uword> xidx =
					arma::conv_to<arma::Col<arma::uword>>::from(xrel * static_cast<fltp>(x.n_elem - 1));

				arma::Col<arma::uword> yidx =
					arma::conv_to<arma::Col<arma::uword>>::from(yrel * static_cast<fltp>(y.n_elem - 1));

				arma::Col<arma::uword> zidx =
					arma::conv_to<arma::Col<arma::uword>>::from(zrel * static_cast<fltp>(z.n_elem - 1));

				assert(xidx.n_elem == yidx.n_elem);
				assert(xidx.n_elem == zidx.n_elem);

				// get bounding cubes
				/*
		 C011    C111
		  \       \
		 |  C001 --- C101
		 C010 |   C110 |
		  \   |      \ |
		   C000 --- C100
		   */

				// note the input cube (x,y) is transposed
				// to follow armadillo interp2 

				// to access vectors from cube
				// cube boundaries
				arma::Col<fltp> v000, v100, v001, v101, v010, v110, v011, v111;
				v000.set_size(xidx.n_elem);
				v100.set_size(xidx.n_elem);
				v001.set_size(xidx.n_elem);
				v101.set_size(xidx.n_elem);
				v010.set_size(xidx.n_elem);
				v110.set_size(xidx.n_elem);
				v011.set_size(xidx.n_elem);
				v111.set_size(xidx.n_elem);

				// fill vectors manual
				assert(xidx.n_elem == idx_mid.n_elem);
				// note v010 is vxyz but the matrix is stored (y,x,z)
				for(int i = 0; i < static_cast<int>(idx_mid.n_elem); i++) {
					v000(i) = v(yidx(i), xidx(i), zidx(i));
					v100(i) = v(yidx(i), xidx(i) + 1, zidx(i));
					v001(i) = v(yidx(i), xidx(i), zidx(i) + 1);
					v101(i) = v(yidx(i), xidx(i) + 1, zidx(i) + 1);
					v010(i) = v(yidx(i) + 1, xidx(i), zidx(i));
					v110(i) = v(yidx(i) + 1, xidx(i) + 1, zidx(i));
					v011(i) = v(yidx(i) + 1, xidx(i), zidx(i) + 1);
					v111(i) = v(yidx(i) + 1, xidx(i) + 1, zidx(i) + 1);
				}

				assert(v000.n_elem == v100.n_elem);
				assert(v001.n_elem == v101.n_elem);
				assert(v010.n_elem == v110.n_elem);
				assert(v011.n_elem == v111.n_elem);

				// midpoints
				// get bounding cube values
				const arma::Col<fltp> x0 = x(xidx);
				const arma::Col<fltp> x1 = x(xidx + 1);
				const arma::Col<fltp> y0 = y(yidx);
				const arma::Col<fltp> y1 = y(yidx + 1);
				const arma::Col<fltp> z0 = z(zidx);
				const arma::Col<fltp> z1 = z(zidx + 1);
				arma::Col<fltp> xd = (xi(idx_mid) - x0) / (x1 - x0);
				arma::Col<fltp> yd = (yi(idx_mid) - y0) / (y1 - y0);
				arma::Col<fltp> zd = (zi(idx_mid) - z0) / (z1 - z0);

				// interpolate bounding planes in x
				arma::Col<fltp> v00 = v000 + ((v100 - v000) % xd);
				arma::Col<fltp> v01 = v001 + ((v101 - v001) % xd);
				arma::Col<fltp> v10 = v010 + ((v110 - v010) % xd);
				arma::Col<fltp> v11 = v011 + ((v111 - v011) % xd);

				// interpolate bounding line in y
				arma::Col<fltp> v0 = v00 + ((v10 - v00) % yd);
				arma::Col<fltp> v1 = v01 + ((v11 - v01) % yd);

				// finally the point
				vi(idx_mid) = v0 + ((v1 - v0) % zd);
			}

			// get bounds lower and set to lowest
			if(extrap) {
				// @hey, not implemented
				assert(0);
			} else {
				arma::Col<arma::uword> idx_mid = arma::find(xi < x.front() && xi > x.back() &&
					yi < y.front() && yi > y.back() && zi < z.front() && zi > z.back());
				vi(idx_mid).fill(0.0);
			}
		}
	}

	// unwrap angles
	arma::Row<fltp> Extra::unwrap_angles(
		const arma::Row<fltp>& alpha, 
		const rat::fltp angle_tol){

		// allocate
		arma::Row<fltp> unwrapped_alpha = alpha;

		// unwrap the angle
		for(arma::uword i=1;i<alpha.n_elem;i++) {
			while((unwrapped_alpha(i)-unwrapped_alpha(i-1))>arma::Datum<rat::fltp>::pi-angle_tol)
				unwrapped_alpha(i) -= arma::Datum<rat::fltp>::tau;
			while((unwrapped_alpha(i)-unwrapped_alpha(i-1))<-arma::Datum<rat::fltp>::pi+angle_tol)
				unwrapped_alpha(i) += arma::Datum<rat::fltp>::tau;
		}

		// check output
		if(!unwrapped_alpha.is_finite())rat_throw_line("angles are not finite");

		// return unwrapped alpha
		return unwrapped_alpha;
	}


	// Compute the running average
	// not the most efficient algorithm but it works
	arma::Row<fltp> Extra::smooth_core(
		const arma::Row<fltp>&ell, 
		const arma::Row<fltp>&alpha, 
		const fltp window_length, 
		const bool use_parallel){

		// check input
		assert(ell.n_elem==alpha.n_elem);

		// allocate
		arma::Row<rat::fltp> smoothed_alpha(alpha.n_elem);

		// calculate cumulative integrals
		const arma::Row<rat::fltp> cumtrapz_alpha = rat::cmn::Extra::cumtrapz(ell, alpha, 1);

		// walk over elements in alpha
		rat::cmn::parfor(0,alpha.n_elem,use_parallel,[&](arma::uword i, int /*cpu*/) {
			// get window start and end positions
			const rat::fltp ell1 = std::max(ell.front(), ell(i) - window_length/2);
			const rat::fltp ell2 = std::min(ell.back(), ell(i) + window_length/2);

			// get integrated values at these positions
			const rat::fltp v1 = rat::cmn::Extra::interp1(ell, cumtrapz_alpha, ell1, "linear", true);
			const rat::fltp v2 = rat::cmn::Extra::interp1(ell, cumtrapz_alpha, ell2, "linear", true);

			// calculate the average over the window
			smoothed_alpha(i) = (v2 - v1)/(ell2 - ell1);
		});

		// check output
		if(!smoothed_alpha.is_finite())rat_throw_line("angles are not finite");

		// return smoothed alpha
		return smoothed_alpha;
	}

	// multipass smoothing
	arma::Row<fltp> Extra::smooth(
		const arma::Row<fltp>&ell, 
		const arma::Row<fltp>&alpha, 
		const fltp window_length, 
		const arma::uword num_passes, 
		const bool use_parallel){

		// without window no smoothing
		if(window_length<=RAT_CONST(0.0))return alpha;

		// create copy
		arma::Row<fltp> smoothed_alpha = alpha;

		// walk over passes
		for(arma::uword j=0;j<num_passes;j++)
			smoothed_alpha = smooth_core(ell, smoothed_alpha, window_length, use_parallel);

		// return smoothed alpha
		return smoothed_alpha;
	}

	// find unique rows and sort
	arma::Mat<arma::uword> Extra::unique_rows(const arma::Mat<arma::uword> &M){
		// case of empty
		if(M.empty())return M;

		// copy M
		arma::Mat<arma::uword> Mout = M;

		// sort rows
		for(arma::uword i=0;i<Mout.n_cols;i++){
			const arma::Col<arma::uword> sort_index = arma::stable_sort_index(Mout.col(i));
			Mout = Mout.rows(sort_index);
		} 

		// find unique 
		arma::Col<arma::uword> is_unique(Mout.n_rows);
		is_unique(0) = true;
		for(arma::uword i=1;i<Mout.n_rows;i++)
			is_unique(i) = arma::any(Mout.row(i)!=Mout.row(i-1));

		// select only unique rows
		Mout = Mout.rows(arma::find(is_unique));

		// return 
		return Mout;
	}

	// parallel implementation of sort
	arma::Row<arma::uword> Extra::parsort(const arma::Row<arma::uword> &vals){
		// for very short arrays
		if(vals.n_elem<60000)return arma::sort(vals);

		// get number of CPU's
		const arma::uword num_splits = 6;
		arma::uword num_subdivide = arma::uword(std::pow(2llu,num_splits));

		// split array
		arma::field<arma::Row<arma::uword> > split_vals(num_subdivide);

		// get number of elements
		const arma::uword num_elements = vals.n_elem;

		// get number of elements in each
		const arma::uword num_split = num_elements/num_subdivide+1;

		// fill arrays
		for(arma::uword i=0;i<num_subdivide;i++){
			split_vals(i) = vals.cols(i*num_split,std::min((i+1)*num_split-1,num_elements-1));
		}

		// sort split arrays in parallel using standard sorting
		parfor(0,num_subdivide,true,[&](arma::uword i, int) {
			split_vals(i) = arma::sort(split_vals(i));
		});

		// now perform merge
		for(arma::uword k=num_subdivide;k>1;k/=2){
			//for(arma::uword j=0;j<k/2;j++){
			parfor(0,k/2,true,[&](arma::uword j, int) {
				// which arrays
				const arma::uword a = j; 
				const arma::uword b = j+k/2;

				// number of merged values
				const arma::uword num_merged = split_vals(a).n_elem + split_vals(b).n_elem;

				// create new merged array
				arma::Row<arma::uword> merged_vals(num_merged);

				// walk over array and merge
				arma::uword idxa = 0, idxb = 0;
				for(arma::uword k=0;k<num_merged;k++){
					if(idxa>=split_vals(a).n_elem){
						merged_vals(k) = split_vals(b)(idxb); idxb++;
					}else if(idxb>=split_vals(b).n_elem){
						merged_vals(k) = split_vals(a)(idxa); idxa++;
					}else if(split_vals(a)(idxa)<split_vals(b)(idxb)){
						merged_vals(k) = split_vals(a)(idxa); idxa++;
					}else{
						merged_vals(k) = split_vals(b)(idxb); idxb++;
					}
				}

				// store merged vals
				split_vals(a) = merged_vals;
				split_vals(b).reset();
			});
		}

		// check output
		assert(split_vals(0).is_sorted());
		assert(split_vals(0).n_elem==num_elements);

		// return merged array
		return split_vals(0);
	}

	// make a new directory if it doesn't already exist
	// the directory will have default privileges
	void Extra::create_directory(const boost::filesystem::path &dirname){
		boost::filesystem::create_directory(dirname);
	}

	// cumulative trapzoidal integration
	arma::Mat<fltp> Extra::cumtrapz(const arma::Row<fltp> &t, const arma::Mat<fltp> &v, const arma::uword dim){
		// check input
		assert(!t.is_empty()); assert(!v.is_empty());
		assert(t.is_sorted("ascend"));

		// check single value
		if(t.n_elem<2)return arma::Row<fltp>(dim==0 ? v.n_cols : v.n_rows,arma::fill::zeros);

		// allocate output
		const arma::uword num_elements = t.n_elem;
		arma::Mat<fltp> vi(v.n_rows, v.n_cols);
		if(dim==0){
			// check input
			assert(v.n_rows==t.n_elem);

			// integrate
			vi.row(0).fill(0);
			const arma::Mat<fltp> mid = (v.head_rows(num_elements-1) + v.tail_rows(num_elements-1))/2;
			const arma::Col<fltp> dt = arma::diff(t,1,1).t();
			vi.rows(1,num_elements-1) = arma::cumsum(mid.each_col()%dt,0);
		}else if(dim==1){
			// check input
			assert(v.n_cols==t.n_elem);

			// integrate
			vi.col(0).fill(0);
			const arma::Mat<fltp> mid = (v.head_cols(num_elements-1) + v.tail_cols(num_elements-1))/2;
			const arma::Row<fltp> dt = arma::diff(t,1,1);
			vi.cols(1,num_elements-1) = arma::cumsum(mid.each_row()%dt,1);
		}else{
			rat_throw_line("dimension does not exist");
		}

		// return integrated values
		return vi;
	}

	// random integer numbers larger than zeros
	// with the poisson distribution
	arma::Row<arma::uword> Extra::poisson_random(
		const arma::uword num_values, const rat::fltp lambda){
		const rat::fltp target=std::exp(-lambda);
		arma::Row<arma::uword> k(num_values,arma::fill::zeros);
		arma::Row<rat::fltp> p(num_values,arma::fill::randu);
		arma::Col<arma::uword> idx(num_values,arma::fill::ones);
		while(!idx.is_empty()){
			// find indexes that have not yet reached the target value
			idx = arma::find(p>target);

			// multiply them with a flat random number
			p(idx)%=arma::Row<rat::fltp>(idx.n_elem, arma::fill::randu);

			// increment the poisson number
			k(idx)+=1;
		}
		return k;
	}

	// // combine nodes in a mesh
	// arma::Row<arma::uword> Extra::combine_nodes(arma::Mat<fltp> &Rn, const fltp gridsize){
	// 	// check if there are any nodes to combine
	// 	if(Rn.empty())return{};

	// 	// snap to grid
	// 	arma::Mat<arma::sword> grd_idx = arma::conv_to<arma::Mat<arma::sword> >::from(Rn/gridsize);

	// 	// keep track of index
	// 	arma::Row<arma::uword> original_idx = arma::regspace<arma::Row<arma::uword> >(0,Rn.n_cols-1);

	// 	// merge duplicate points
	// 	for(arma::uword i=0;i<Rn.n_rows;i++){
	// 		// create sorting indices
	// 		const arma::Col<arma::uword> sort_idx = arma::stable_sort_index(grd_idx.row(i));

	// 		// sort nodes
	// 		Rn = Rn.cols(sort_idx); grd_idx = grd_idx.cols(sort_idx);

	// 		// sort indexes
	// 		original_idx = original_idx.cols(sort_idx);
	// 	}

	// 	// invert indexes
	// 	arma::Row<arma::uword> idx2(Rn.n_cols); idx2.cols(original_idx) = arma::regspace<arma::Row<arma::uword> >(0,Rn.n_cols-1);

	// 	// find duplicate points
	// 	const arma::Row<arma::uword> idx = arma::join_horiz(arma::Row<arma::uword>{0}, 
	// 		arma::cumsum(arma::any(grd_idx.tail_cols(grd_idx.n_cols-1)!=grd_idx.head_cols(grd_idx.n_cols-1),0)));

	// 	// set sections
	// 	const arma::Mat<arma::uword> sect = cmn::Extra::find_sections(idx);

	// 	// remove duplicate points
	// 	Rn = Rn.cols(sect.row(0));

	// 	// combine indexing schemes
	// 	return idx.cols(idx2);
	// }

	// combine nodes in a mesh
	arma::Row<arma::uword> Extra::combine_nodes(arma::Mat<fltp> &Rn, const fltp delta){
		arma::Mat<fltp> data(0,Rn.n_cols);
		return combine_nodes(Rn,data,delta);
	}

	// combine nodes in a mesh
	arma::Row<arma::uword> Extra::combine_nodes(arma::Mat<fltp> &Rn, arma::Mat<fltp> &data, const fltp delta){
		// check if there are any nodes to combine
		if(Rn.empty())return arma::Row<arma::uword>{};

		// get number of original nodes
		const arma::uword num_nodes = Rn.n_cols;

		// linearize coordinates
		arma::Row<fltp> spatial_hash = arma::sum(Rn,0);

		// keep track of index
		arma::Row<arma::uword> original_idx = arma::regspace<arma::Row<arma::uword> >(0,Rn.n_cols-1);

		// create sorting indices
		const arma::Col<arma::uword> sort_idx = arma::stable_sort_index(spatial_hash);

		// sort nodes and data
		Rn = Rn.cols(sort_idx);
		if(!data.empty())
			data = data.cols(sort_idx);

		// sort indexes
		original_idx = original_idx.cols(sort_idx);
		
		// invert indexes
		arma::Row<arma::uword> idx2(Rn.n_cols); idx2.cols(original_idx) = arma::regspace<arma::Row<arma::uword> >(0,Rn.n_cols-1);

		// sort spatial ash
		spatial_hash = spatial_hash.cols(sort_idx);

		// find duplicates
		arma::Row<arma::uword> combine_with(Rn.n_cols,arma::fill::zeros);
	
		// find duplicate points
		for(arma::uword i=0;i<num_nodes-1;i++){
			// check if already combined
			if(combine_with(i)>0)continue;

			// backwards
			for(arma::uword j=i+1;j<num_nodes;j++){
				if((spatial_hash(j) - spatial_hash(i))>3*delta/std::sqrt(RAT_CONST(3.0)))break;
				if(combine_with(j)>0)continue;
				if(arma::as_scalar(cmn::Extra::vec_norm(Rn.col(i) - Rn.col(j)))<delta)combine_with(j) = i + 1; // offset of 1 to keep 0 as uncombined
			}
		}

		// find indices of nodes to keep
		arma::Col<arma::uword> keep_nodes = arma::find(combine_with==0);

		// assign future regularly spaced indexes of these nodes 
		arma::Row<arma::uword> idx(Rn.n_cols,arma::fill::zeros);
		idx.cols(keep_nodes) = arma::regspace<arma::Row<arma::uword> >(0,keep_nodes.n_elem-1);

		// re-index nodes that are to be combined with future nodes
		idx.cols(arma::find(combine_with>0)) = idx.cols(combine_with.cols(arma::find(combine_with>0))-1);

		// take nodes that are not duplicates
		Rn = Rn.cols(keep_nodes);

		// take non-duplicate data
		if(!data.empty())
			data = data.cols(keep_nodes);

		// combine indexing schemes
		return idx.cols(idx2);
	}

	// conncomp algorithm
	// expects NX2 matrix of indices
	arma::Row<arma::uword> Extra::conncomp(
		const arma::Mat<arma::uword> &n12,
		const arma::uword num_nodes){

		// check dimensions
		assert(n12.n_cols==2);

		// find unique and stort
		const arma::Mat<arma::uword> n12_unique = unique_rows(arma::join_vert(n12,arma::fliplr(n12)));
	
		// get row and column indices
		arma::Col<arma::uword> idx_row = n12_unique.col(0);
		arma::Col<arma::uword> idx_col = n12_unique.col(1);


		// check input
		if(arma::any(arma::vectorise(n12_unique)>=num_nodes))
			rat_throw_line("indices exceed specified number of nodes");

		// compress column indices
		// count number of entries in each column
		arma::Col<arma::uword> ncol(num_nodes, arma::fill::zeros);
		for(arma::uword i=0;i<idx_col.n_elem;i++)ncol(idx_col(i))++;

		// create compression array
		const arma::Col<arma::uword> idx_col_c = arma::join_vert(
			arma::Col<arma::uword>{0},arma::cumsum(ncol));

		// allocate output list
		arma::Row<arma::uword> graph_indices(num_nodes, arma::fill::zeros);

		// allocate search list
		std::list<arma::uword> connect_list;

		// walk over indices
		for(arma::uword id=1;;id++){
			// find root index from where the search is performed
			arma::Col<arma::uword> root_indices = arma::find(graph_indices==0,1,"first");
			
			// if no root found then search is complete
			if(root_indices.empty())break;

			// convert to scalar now that we know length is 1
			assert(root_indices.n_elem==1);
			arma::uword root_index = arma::as_scalar(root_indices);

			// add to list
			connect_list.push_back(root_index);
			
			// mark root as explored
			graph_indices(root_index) = id;

			// now keep searching
			while(!connect_list.empty()){
				// current search node
				arma::uword idx_current = connect_list.back();

				// remove from list
				connect_list.pop_back();

				// walk over column
				for(arma::uword j=idx_col_c(idx_current);j<idx_col_c(idx_current+1);j++){
					// get target
					const arma::uword row = n12_unique(j,0);

					// not visited before?
					if(graph_indices(row)==0){
						// then add to list to explore
						connect_list.push_back(row);

						// mark visited so we don't come back
						graph_indices(row) = id;
					}
				}
			}
		}

		// return indices
		return graph_indices - 1;
	}

	// calculate curvature
	arma::Mat<fltp> Extra::calc_curvature(const arma::Mat<fltp> &R){
		// less than three points can not calculate curvature
		if(R.n_cols<3)return arma::Mat<fltp>(3,R.n_cols,arma::fill::zeros);

		// calculate triangle sides
		const arma::Row<fltp> a = cmn::Extra::vec_norm(R.cols(0,R.n_cols-3) - R.cols(1,R.n_cols-2));
		const arma::Row<fltp> b = cmn::Extra::vec_norm(R.cols(1,R.n_cols-2) - R.cols(2,R.n_cols-1));
		const arma::Row<fltp> c = cmn::Extra::vec_norm(R.cols(2,R.n_cols-1) - R.cols(0,R.n_cols-3));
		const arma::Row<fltp> abc = a%b%c;

		// Heron's formula for area of triangle
		const arma::Row<fltp> k = arma::sqrt((a+(b+c))%(c-(a-b))%(c+(a-b))%(a+(b-c)))/4;
		
		// allocate curvature
		arma::Row<fltp> curvature(R.n_cols);
		curvature.cols(1,R.n_cols-2) = (4*k)/abc;

		// avoid division by zero
		arma::Col<arma::uword> id = arma::find(abc==RAT_CONST(0.0));
		curvature(id+1).fill(RAT_CONST(0.0));

		// repeat first and last
		curvature(0) = curvature(1); curvature(R.n_cols-1) = curvature(R.n_cols-2);

		// direction of curvature
		const arma::Mat<fltp> dR1 = arma::diff(R.cols(1,R.n_cols-1),1,1);
		const arma::Mat<fltp> dR2 = arma::diff(R.cols(0,R.n_cols-2),1,1);
		arma::Mat<fltp> N(3,R.n_cols);
		N.cols(1,R.n_cols-2) = cmn::Extra::cross((dR1+dR2)/2,cmn::Extra::cross(dR1,dR2));
		N.col(0) = N.col(1); N.col(R.n_cols-1) = N.col(R.n_cols-2);

		// normalize normal vector
		N.each_row()/=cmn::Extra::vec_norm(N);

		// add direction and return
		return N.each_row()%curvature;
	}

	// version tag
	std::string Extra::create_version_tag(const int major_version, const int minor_version, const int patch_version){
		std::ostringstream version_stream;
		version_stream<<major_version<<".";
		version_stream.width(3); version_stream.fill('0');
		version_stream<<minor_version;
		version_stream.width(1);
		version_stream<<"."<<patch_version;
		return version_stream.str();
	}

	// armadillo arma::find_unique but with tolerance
	arma::Col<arma::uword> Extra::find_unique(const arma::Mat<fltp> &vals, const bool ascending_indices, const fltp tol){
		// vectorise values
		arma::Col<fltp> vv = arma::vectorise(vals);
		
		// sort values while keeping sort index
		const arma::Col<arma::uword> sidx = arma::sort_index(vv);
		vv = vv(sidx);
			
		// check uniqueness by comparing adjacent elements
		const arma::Col<arma::uword> is_unique = arma::join_vert(arma::Col<arma::uword>{1},(arma::abs(arma::diff(vv,1,0))>tol));

		// find unique elements
		arma::Col<arma::uword> id = arma::find(is_unique);

		// unsort id's to their original element
		id = sidx(id);

		// sort indices to be ascending
		if(ascending_indices)id = arma::sort(id);

		// return indices
		return id;
	}


	// search functions
	arma::Col<arma::uword> Extra::ismember(const arma::Row<arma::uword>&A, const arma::Row<arma::uword>&B){
		// get sorting indices
		const arma::Col<arma::uword> asidx = arma::sort_index(A);
		const arma::Col<arma::uword> bsidx = arma::sort_index(B);
		
		// allocate output
		arma::Col<arma::uword> is(A.n_elem);

		// walk over a
		arma::uword j=0;
		for(arma::uword i=0;i<A.n_elem;i++){
			const arma::uword Ach = A(asidx(i));
			// walk over b
			for(;j<B.n_elem;){
				const arma::uword Bch = B(bsidx(j));
				if(Ach==Bch){
					is(asidx(i)) = true;
					break;
				}else if(Bch>Ach){
					is(asidx(i)) = false;
					break;
				}else{
					j++;
				}
			}
		}

		// return boolean array
		return is;
	}

}}
