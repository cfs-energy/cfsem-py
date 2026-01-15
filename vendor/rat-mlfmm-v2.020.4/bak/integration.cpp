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
#include "integration.hh"

// analytical solution
arma::Mat<double> Integration::calc_analytic(
	const arma::Mat<double> &Rt,
	const arma::Mat<double> &Rnh){
	// get face matrix
	arma::Mat<arma::uword>::fixed<6,4> M = Hexahedron::get_faces();

	// get quadrilateral edge matrix
	arma::Mat<arma::uword>::fixed<4,2> Me = Quadrilateral::get_edges();

	// create field contribution vector
	arma::Mat<double> H(9,Rt.n_cols,arma::fill::zeros);

	// walk over faces
	for(arma::uword j=0;j<M.n_rows;j++){
		// get face nodes
		arma::Mat<double>::fixed<3,4> Rf = Rnh.cols(M.row(j));

		// extract vectors that span the face from opposing nodes
		arma::Col<double>::fixed<3> V01 = Extra::cross(Rf.col(1) - Rf.col(0), Rf.col(3) - Rf.col(0));

		// calculate face normal unit vector
		arma::Col<double>::fixed<3> zeta = V01.each_row()/Extra::vec_norm(V01);

		// magnetisation along face direction
		double Mzetax = arma::as_scalar(Extra::dot(arma::Col<double>::fixed<3>{1.0,0,0},zeta));
		double Mzetay = arma::as_scalar(Extra::dot(arma::Col<double>::fixed<3>{0,1.0,0},zeta));
		double Mzetaz = arma::as_scalar(Extra::dot(arma::Col<double>::fixed<3>{0,0,1.0},zeta));

		// walk over edges
		for(arma::uword k=0;k<Me.n_rows;k++){
			// std::cout<<i<<" "<<j<<" "<<k<<std::endl;
			// get face nodes
			arma::Mat<double>::fixed<3,2> Re = Rf.cols(Me.row(k));

			// edge unit vector
			arma::Col<double>::fixed<3> Vg = Re.col(1)-Re.col(0);
			arma::Col<double>::fixed<3> xi = Vg.each_row()/Extra::vec_norm(Vg);

			// plane unit vector
			// arma::Col<double>::fixed<3> nu = Extra::cross(xi,zeta);
			arma::Col<double>::fixed<3> nu = Extra::cross(zeta,xi);

			// walk over target points
			for(arma::uword l=0;l<Rt.n_cols;l++){
				// relative position of start and end point
				arma::Col<double>::fixed<3> R0 = Rt.col(l) - Re.col(0);
				arma::Col<double>::fixed<3> R1 = Rt.col(l) - Re.col(1);
				
				// distance between points
				double r0 = arma::as_scalar(Extra::vec_norm(R0));
				double r1 = arma::as_scalar(Extra::vec_norm(R1));

				// in local coordinates
				double rxi0 = arma::as_scalar(Extra::dot(xi,R0));
				double rxi1 = arma::as_scalar(Extra::dot(xi,R1));
				double rnu = arma::as_scalar(Extra::dot(nu,R0)); 
				double rzeta = arma::as_scalar(Extra::dot(zeta,R0)); 

				// std::cout<<xi0<<" "<<xi1<<std::endl;
				// std::cout<<nu0<<" "<<nu1<<std::endl;
				// std::cout<<zeta0<<" "<<zeta1<<std::endl<<std::endl;

				// Following equations are adapted from
				// A.G.A.M. Armstrong et. al., "New developments in the magnet design computer program gfun", 
				// Presented at the 5th International Conference on Magnet Technology, Frascati, Rome, 1975.
				
				// calculate q using pytagoras
				const double q = std::sqrt(rzeta*rzeta + rnu*rnu);

				// orthogonal case
				// the target point lies on the line
				if(std::abs(q)<1e-10)continue;
				
				// calculate field in local coordinates
				// note that these are not valid on the surface
				// const double fac_xi = rnu*(
				// 	1.0/(r1+std::abs(rzeta)) - 
				// 	1.0/(r0+std::abs(rzeta)));

				const double fac_xi = 0;

				// double fac_nu = 
				// 	(std::log(r1+rxi1) - rxi1/(r1+std::abs(rzeta))) - 
				// 	(std::log(r0+rxi0) - rxi0/(r0+std::abs(rzeta)));

				// double fac_nu = 
				// 	(std::log(r1-rxi1) - rxi1/(r1+std::abs(rzeta))) - 
				// 	(std::log(r0-rxi0) - rxi0/(r0+std::abs(rzeta)));

				// const double fac_nu = 
				// 	(std::log(r1-rxi1)) - 
				// 	(std::log(r0-rxi0));
				
				const double fac_nu = 
					std::log((r1-rxi1)/(r0-rxi0));

				// const double fac_nu = 
				// 	(-std::log(r1+rxi1) - rxi1/(r1+std::abs(rzeta))) - 
				// 	(-std::log(r0+rxi0) - rxi0/(r0+std::abs(rzeta)));

				// double fac_nu = 
				// 	(std::log(r1-rxi1) + rxi1/(r1+std::abs(rzeta))) - 
				// 	(std::log(r0-rxi0) + rxi0/(r0+std::abs(rzeta)));

				// const double fac_zeta = 
				// 	(std::abs(rzeta)/rzeta)*std::asin((rxi1*rnu)/(q*(r1+std::abs(rzeta)))) -
				// 	(std::abs(rzeta)/rzeta)*std::asin((rxi0*rnu)/(q*(r0+std::abs(rzeta)))); 
				
				const double fac_zeta = 
					Extra::sign(rzeta)*std::asin((rxi1*rnu)/(q*(r1+std::abs(rzeta)))) -
					Extra::sign(rzeta)*std::asin((rxi0*rnu)/(q*(r0+std::abs(rzeta)))); 

				// check for nans
				// these can occur when q = 0
				assert(!std::isnan(fac_xi));
				assert(!std::isnan(fac_nu));
				assert(!std::isnan(fac_zeta));

				// field in cartesian coordinates
				double fac = -1.0/(4.0*arma::datum::pi);
				H.submat(0,l,2,l) += fac*(xi*fac_xi*Mzetax + nu*fac_nu*Mzetax + zeta*fac_zeta*Mzetax);
				H.submat(3,l,5,l) += fac*(xi*fac_xi*Mzetay + nu*fac_nu*Mzetay + zeta*fac_zeta*Mzetay);
				H.submat(6,l,8,l) += fac*(xi*fac_xi*Mzetaz + nu*fac_nu*Mzetaz + zeta*fac_zeta*Mzetaz);
			}
		}	
	}

	// return field contribution
	return H;
}



// analytical solution
arma::Mat<double> Integration::calc_analytic_tet(
	const arma::Mat<double> &Rt,
	const arma::Mat<double> &Rnh){

	// get tetrahedron conversion matrix
	arma::Mat<arma::uword>::fixed<5,4> Mt = Hexahedron::tetrahedron_conversion_matrix();

	// get face matrix
	arma::Mat<arma::uword>::fixed<4,3> M = Tetrahedron::get_faces();
	// M = arma::fliplr(M); // clockwise instead of counter-clockwise

	// get quadrilateral edge matrix
	arma::Mat<arma::uword>::fixed<3,2> Me = Triangle::get_edges();

	// create field contribution vector
	arma::Mat<double> H(9,Rt.n_cols,arma::fill::zeros);

	// walk over tetrahedrons
	for(arma::uword i=0;i<Mt.n_rows;i++){
		// get nodes of tetrahedron
		arma::Mat<double>::fixed<3,4> Rnt = Rnh.cols(Mt.row(i));

		// walk over faces
		for(arma::uword j=0;j<M.n_rows;j++){
			// get face nodes
			arma::Mat<double>::fixed<3,3> Rf = Rnt.cols(M.row(j));

			// extract vectors that span the face from opposing nodes
			arma::Col<double>::fixed<3> V01 = Extra::cross(Rf.col(1) - Rf.col(0), Rf.col(2) - Rf.col(0));

			// calculate face normal unit vector
			arma::Col<double>::fixed<3> zeta = V01.each_row()/Extra::vec_norm(V01);

			// magnetisation along face direction
			double Mzetax = arma::as_scalar(Extra::dot(arma::Col<double>::fixed<3>{1.0,0,0},zeta));
			double Mzetay = arma::as_scalar(Extra::dot(arma::Col<double>::fixed<3>{0,1.0,0},zeta));
			double Mzetaz = arma::as_scalar(Extra::dot(arma::Col<double>::fixed<3>{0,0,1.0},zeta));

			// walk over edges
			for(arma::uword k=0;k<Me.n_rows;k++){
				// std::cout<<i<<" "<<j<<" "<<k<<std::endl;
				// get face nodes
				arma::Mat<double>::fixed<3,2> Re = Rf.cols(Me.row(k));

				// edge unit vector
				arma::Col<double>::fixed<3> Vg = Re.col(1)-Re.col(0);
				arma::Col<double>::fixed<3> xi = Vg.each_row()/Extra::vec_norm(Vg);

				// plane unit vector
				// arma::Col<double>::fixed<3> nu = Extra::cross(xi,zeta);
				arma::Col<double>::fixed<3> nu = Extra::cross(zeta,xi);

				// walk over target points
				for(arma::uword l=0;l<Rt.n_cols;l++){
					// relative position of start and end point
					arma::Col<double>::fixed<3> R0 = Rt.col(l) - Re.col(0);
					arma::Col<double>::fixed<3> R1 = Rt.col(l) - Re.col(1);
					
					// distance between points
					double r0 = arma::as_scalar(Extra::vec_norm(R0));
					double r1 = arma::as_scalar(Extra::vec_norm(R1));

					// in local coordinates
					double rxi0 = arma::as_scalar(Extra::dot(xi,R0));
					double rxi1 = arma::as_scalar(Extra::dot(xi,R1));
					double rnu = arma::as_scalar(Extra::dot(nu,R0)); 
					double rzeta = arma::as_scalar(Extra::dot(zeta,R0)); 
		
					// std::cout<<xi0<<" "<<xi1<<std::endl;
					// std::cout<<nu0<<" "<<nu1<<std::endl;
					// std::cout<<zeta0<<" "<<zeta1<<std::endl<<std::endl;

					// Following equations are adapted from
					// A.G.A.M. Armstrong et. al., "New developments in the magnet design computer program gfun", 
					// Presented at the 5th International Conference on Magnet Technology, Frascati, Rome, 1975.
					
					// calculate q using pytagoras
					double q = std::sqrt(rzeta*rzeta + rnu*rnu);

					// calculate field in local coordinates
					// note that these are not valid on the surface
					double fac_xi = rnu*(
						1.0/(r1+std::abs(rzeta)) - 
						1.0/(r0+std::abs(rzeta)));

					// double fac_nu = 
					// 	(std::log(r1+rxi1) - rxi1/(r1+std::abs(rzeta))) - 
					// 	(std::log(r0+rxi0) - rxi0/(r0+std::abs(rzeta)));

					double fac_nu = 
						(std::log(r1-rxi1) - rxi1/(r1+std::abs(rzeta))) - 
						(std::log(r0-rxi0) - rxi0/(r0+std::abs(rzeta)));

					// double fac_nu = 
					// 	(std::log(r1-rxi1) + rxi1/(r1+std::abs(rzeta))) - 
					// 	(std::log(r0-rxi0) + rxi0/(r0+std::abs(rzeta)));

					double fac_zeta = 
						(std::abs(rzeta)/rzeta)*std::asin((rxi1*rnu)/(q*(r1+std::abs(rzeta)))) -
						(std::abs(rzeta)/rzeta)*std::asin((rxi0*rnu)/(q*(r0+std::abs(rzeta)))); 

					// field in cartesian coordinates
					double fac = -1.0/(4.0*arma::datum::pi);
					H.submat(0,l,2,l) += fac*(xi*fac_xi*Mzetax + nu*fac_nu*Mzetax + zeta*fac_zeta*Mzetax);
					H.submat(3,l,5,l) += fac*(xi*fac_xi*Mzetay + nu*fac_nu*Mzetay + zeta*fac_zeta*Mzetay);
					H.submat(6,l,8,l) += fac*(xi*fac_xi*Mzetaz + nu*fac_nu*Mzetaz + zeta*fac_zeta*Mzetaz);
				}
			}	
		}

	}	

	// return field contribution
	return H;
}


// numerical integration using surface integrals
arma::Mat<double> Integration::calc_numeric_surface(
	const arma::Mat<double> &Rt,
	const arma::Mat<double> &Rnh,
	const arma::sword num_gauss,
	const bool is_self_field){

	// make gauss point calculator
	// create gauss point calculator
	Gauss gp(num_gauss);

	// extract abscissae and weights
	arma::Row<double> xg = gp.get_abscissae(); 
	arma::Row<double> wg = gp.get_weights();

	// setup grid
	arma::Col<double> Rqts = arma::Col<double>{1.0,1.0};
	arma::Mat<double> Rqgrds; arma::Row<double> wgrds;
	Quadrilateral::setup_source_grid(Rqgrds, wgrds, Rqts, xg, wg);
	assert(std::abs(arma::sum(wgrds)-1.0)<1e-6);

	// get face matrix
	arma::Mat<arma::uword>::fixed<6,4> M = Hexahedron::get_faces();

	// create field contribution vector
	arma::Mat<double> H(9,Rt.n_cols,arma::fill::zeros);

	// walk over faces
	for(arma::uword j=0;j<M.n_rows;j++){
		// get face nodes
		arma::Mat<double> Rf = Rnh.cols(M.row(j));
		//std::cout<<i<<" "<<j<<std::endl;

		// extract vectors that span the face from opposing nodes
		arma::Mat<double> V0 = Rf.col(1) - Rf.col(0);
		arma::Mat<double> V1 = Rf.col(3) - Rf.col(0);
		arma::Mat<double> V2 = Rf.col(1) - Rf.col(2);
		arma::Mat<double> V3 = Rf.col(3) - Rf.col(2);

		arma::Mat<double> V01 = Extra::cross(V0,V1);
		arma::Mat<double> V23 = Extra::cross(V2,V3);

		// calculate face area using two triangles
		double A = arma::as_scalar(Extra::vec_norm(V01)/2 + Extra::vec_norm(V23)/2);

		// calculate face normal
		arma::Col<double>::fixed<3> N = V01.each_row()/Extra::vec_norm(V01);

		// calculate surface currents for unit magnetic moment in x,y and z directions
		arma::Col<double>::fixed<3> Jx = Extra::cross(arma::Col<double>{1.0,0,0},N);
		arma::Col<double>::fixed<3> Jy = Extra::cross(arma::Col<double>{0,1.0,0},N);
		arma::Col<double>::fixed<3> Jz = Extra::cross(arma::Col<double>{0,0,1.0},N);

		// calculate points in carthesian coordinates
		arma::Mat<double> Rc = Quadrilateral::quad2cart(Rf,Rqgrds);

		// calculate weighted current for each gauss node for unit magnetic moment in each direction
		arma::Mat<double> Ixa(3,Rc.n_cols); Ixa.each_col() = Jx; Ixa.each_row()%=A*wgrds;
		arma::Mat<double> Iya(3,Rc.n_cols); Iya.each_col() = Jy; Iya.each_row()%=A*wgrds;
		arma::Mat<double> Iza(3,Rc.n_cols); Iza.each_col() = Jz; Iza.each_row()%=A*wgrds;

		// integrate
		H.rows(0,2) += Savart::calc_I2H(Rc, Ixa, Rt, false);
		H.rows(3,5) += Savart::calc_I2H(Rc, Iya, Rt, false);
		H.rows(6,8) += Savart::calc_I2H(Rc, Iza, Rt, false);			
	}

	// calculate H instead of B/mu0
	if(is_self_field){
		H.row(0) -= 1.0;
		H.row(4) -= 1.0;
		H.row(8) -= 1.0;
	}

	// display
	// if(i==0)std::cout<<val_.col(i)<<std::endl;

	// return field contribution
	return H;
}

// numerical integration using surface integrals
arma::Mat<double> Integration::calc_numeric_volume(
	const arma::Mat<double> &Rt,
	const arma::Mat<double> &Rnh,
	const double Ve,
	const arma::sword num_gauss){
			
	// create gauss point calculator
	Gauss gp(num_gauss);

	// extract abscissae and weights
	arma::Row<double> xg = gp.get_abscissae(); 
	arma::Row<double> wg = gp.get_weights();
	
	// create gauss point coordinates 
	// no singularity present
	arma::Col<double> Rqt = {1.0,1.0,1.0};
	arma::Mat<double> Rqgrd; arma::Row<double> wgrd;
	Hexahedron::setup_source_grid(Rqgrd, wgrd, Rqt, xg, wg);
	assert(std::abs(arma::sum(wgrd)-1.0)<1e-6);

	// calculate source points in carthesian coordinates
	arma::Mat<double> Rs = Hexahedron::quad2cart(Rnh,Rqgrd);

	// create unit magnetisation vectors
	arma::Mat<double> Mx(3,Rqgrd.n_cols,arma::fill::zeros); Mx.row(0) = wgrd*Ve; 
	arma::Mat<double> My(3,Rqgrd.n_cols,arma::fill::zeros); My.row(1) = wgrd*Ve;
	arma::Mat<double> Mz(3,Rqgrd.n_cols,arma::fill::zeros); Mz.row(2) = wgrd*Ve;

	// create field contribution vector
	arma::Mat<double> H(9,Rt.n_cols,arma::fill::zeros);

	// walk over target points
	for(arma::uword i=0;i<Rt.n_cols;i++){
		// integrate
		H.submat(0,i,2,i) += Savart::calc_M2H(Rs, Mx, Rt.col(i), false);
		H.submat(3,i,5,i) += Savart::calc_M2H(Rs, My, Rt.col(i), false);
		H.submat(6,i,8,i) += Savart::calc_M2H(Rs, Mz, Rt.col(i), false);
	}

	// if(i==0)std::cout<<val_.col(i)<<std::endl;

	// return field contribution
	return H;
}

// numerical integration using surface integrals
arma::Mat<double> Integration::calc_savart(
	const arma::Mat<double> &Rt,
	const arma::Mat<double> &Rnh,
	const double Ve,
	const bool is_self_field){

	// calculate center of mass
	arma::Col<double> Re = Hexahedron::quad2cart(Rnh,arma::Col<double>::fixed<3>{0,0,0});

	// allocate field contribution
	arma::Mat<double> H(9,Rt.n_cols);

	// self field (using sphere approximation)
	// see: http://farside.ph.utexas.edu/teaching/jk1/lectures/node61.html
	if(is_self_field){
		H.zeros();
		H.row(0) = -1.0/3;
		H.row(4) = -1.0/3;
		H.row(8) = -1.0/3;
	}

	// mutual field
	else{
		// savart 
		for(arma::uword i=0;i<Rt.n_cols;i++){
			H.submat(0,i,2,i) = Savart::calc_M2H(Re,arma::Col<double>{Ve,0,0},Rt,false);
			H.submat(3,i,5,i) = Savart::calc_M2H(Re,arma::Col<double>{0,Ve,0},Rt,false);
			H.submat(6,i,8,i) = Savart::calc_M2H(Re,arma::Col<double>{0,0,Ve},Rt,false);
		}
	}

	// return field contribution
	return H;
}