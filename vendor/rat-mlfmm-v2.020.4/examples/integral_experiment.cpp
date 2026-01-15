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

// general headers
#include <armadillo>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

// specific headers
#include "rat/common/error.hh"
#include "rat/common/elements.hh"
#include "rat/common/extra.hh"

// header files for models
#include "brick8node.hh"
#include "savart.hh"

// main
int main(){
	// settings
	const rat::fltp D = RAT_CONST(0.01); // box side length
	const rat::fltp tol = RAT_CONST(1e-4);
	const arma::uword N = 100;
	const rat::fltp dxtrap = 0.005;
	const arma::sword num_gauss = 20;
	const rat::fltp radius = D*4;
	const rat::fltp I = 100;

	// // define box corners
	// const arma::Col<rat::fltp>::fixed<3> R0 = {-D/2,-D/2,-D/2}; 
	// const arma::Col<rat::fltp>::fixed<3> R1 = {+D/2,-D/2,-D/2-dxtrap}; 
	// const arma::Col<rat::fltp>::fixed<3> R2 = {+D/2,+D/2,-D/2-dxtrap}; 
	// const arma::Col<rat::fltp>::fixed<3> R3 = {-D/2,+D/2,-D/2};
	// const arma::Col<rat::fltp>::fixed<3> R4 = {-D/2,-D/2,+D/2}; 
	// const arma::Col<rat::fltp>::fixed<3> R5 = {+D/2,-D/2,+D/2+dxtrap}; 
	// const arma::Col<rat::fltp>::fixed<3> R6 = {+D/2,+D/2,+D/2+dxtrap}; 
	// const arma::Col<rat::fltp>::fixed<3> R7 = {-D/2,+D/2,+D/2}; 

	// graded element
	const arma::Col<rat::fltp>::fixed<3> R0 = {-D/2,-D/2,-D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R1 = {+D/2,-D/2,-D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R2 = {+D/2,+D/2,-D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R3 = {-D/2,+D/2,-D/2};
	const arma::Col<rat::fltp>::fixed<3> R4 = {-D/2-dxtrap,-D/2-dxtrap,+D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R5 = {+D/2+dxtrap,-D/2-dxtrap,+D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R6 = {+D/2+dxtrap,+D/2+dxtrap,+D/2}; 
	const arma::Col<rat::fltp>::fixed<3> R7 = {-D/2-dxtrap,+D/2+dxtrap,+D/2}; 

	// assemble matrix with nodes
	arma::Mat<rat::fltp> Rn(3,8);
	Rn.col(0) = R0; Rn.col(1) = R1; Rn.col(2) = R2; Rn.col(3) = R3;
	Rn.col(4) = R4; Rn.col(5) = R5; Rn.col(6) = R6; Rn.col(7) = R7;

	// element
	const arma::Col<arma::uword> nh = {0,1,2,3,4,5,6,7};

	// element current density
	const arma::Col<rat::fltp>::fixed<3> Je{0,0,I/(D*D)}; // from plane 1 and to plane 2

	// create targets on a plane
	const arma::Col<rat::fltp> xa = arma::linspace<arma::Col<rat::fltp> >(-3*D,3*D,100);
	const arma::Col<rat::fltp> za = arma::linspace<arma::Col<rat::fltp> >(-3*D,3*D,101);
	arma::Mat<rat::fltp> Rt(3,xa.n_elem*za.n_elem); arma::uword cnt = 0;
	for(arma::uword i=0;i<za.n_elem;i++){
		for(arma::uword j=0;j<xa.n_elem;j++){
			Rt.col(cnt++) = arma::Col<rat::fltp>::fixed<3>{xa(j),RAT_CONST(0.0),za(i)};
		}
	}

	// create line currents from face to face
	const arma::Mat<rat::fltp> gp = rat::cmn::Quadrilateral::create_gauss_points(num_gauss);
	const arma::Mat<rat::fltp> Rq = gp.rows(0,2);
	const arma::Row<rat::fltp> wq = gp.row(2);
	const arma::Mat<rat::fltp> Rc1 = rat::cmn::Quadrilateral::quad2cart(Rn.cols(0,3),Rq);
	const arma::Mat<rat::fltp> Rc2 = rat::cmn::Quadrilateral::quad2cart(Rn.cols(4,7),Rq);

	// create line elements
	const arma::Mat<rat::fltp> Rs = (Rc1 + Rc2)/2;
	const arma::Mat<rat::fltp> dRs = Rc2 - Rc1;
	const arma::Row<rat::fltp> Is = wq*I/4;
	const arma::Row<rat::fltp> epss = wq%rat::cmn::Extra::vec_norm(dRs)/4;

	// calculate field from gauss points
	std::cout<<"magnetic field:"<<std::endl;
	arma::wall_clock timer1; timer1.tic();
	const arma::Mat<rat::fltp> H1 = rat::fmm::Savart::calc_I2H_vl(Rs,dRs,Is,epss,Rt,true);
	const rat::fltp time1 = timer1.toc();
	std::cout<<"time1: "<<time1<<std::endl;

	// Part 2: Integration using surface integrals
	std::cout<<"magnetic field:"<<std::endl;
	arma::wall_clock timer2; timer2.tic();
	const arma::Mat<rat::fltp> H2 = rat::fmm::Brick8Node::calc_magnetic_field(Rn,Rt,I,3llu,1.1);
	const rat::fltp time2 = timer2.toc();
	std::cout<<"time2: "<<time2<<std::endl;

	// calculate field from gauss points
	std::cout<<"vector potential:"<<std::endl;
	arma::wall_clock timer3; timer3.tic();
	const arma::Mat<rat::fltp> A1 = rat::fmm::Savart::calc_I2A_vl(Rs,dRs,Is,epss,Rt,true);
	const rat::fltp time3 = timer3.toc();
	std::cout<<"time3: "<<time3<<std::endl;

	// Part 2: Integration using surface integrals
	std::cout<<"vector potential:"<<std::endl;
	arma::wall_clock timer4; timer4.tic();
	const arma::Mat<rat::fltp> A2 = rat::fmm::Brick8Node::calc_vector_potential(Rn,Rt,I,3llu,1.1);
	const rat::fltp time4 = timer4.toc();
	std::cout<<"time4: "<<time4<<std::endl;

	// create table
	const arma::Mat<rat::fltp> M = arma::join_horiz(Rt.t(),H1.t(),H2.t(),arma::join_horiz(A1.t(),A2.t()));

	M.save("data.csv",arma::csv_ascii);

	// matlab script for plotting data
	// data = csvread('data.csv');mu0 = 4*pi*1e-7; f = figure; subplot(1,3,1); imagesc(mu0*reshape(vecnorm(data(:,4:6)'),100,101)'); colorbar; xlabel('x [m]'); ylabel('z [m]'); subplot(1,3,2); imagesc(mu0*reshape(vecnorm(data(:,7:9)'),100,101)'); xlabel('x [m]'); ylabel('z [m]'); colorbar; subplot(1,3,3); imagesc(mu0*reshape(vecnorm(data(:,7:9)')-vecnorm(data(:,4:6)'),100,101)'); xlabel('x [m]'); ylabel('z [m]'); colorbar;
	// data = csvread('data.csv'); f = figure; subplot(1,3,1); imagesc(reshape(vecnorm(data(:,10:12)'),100,101)'); colorbar; xlabel('x [m]'); ylabel('z [m]'); subplot(1,3,2); imagesc(reshape(vecnorm(data(:,13:15)'),100,101)'); xlabel('x [m]'); ylabel('z [m]'); colorbar; subplot(1,3,3); imagesc(reshape(vecnorm(data(:,13:15)')-vecnorm(data(:,10:12)'),100,101)'); xlabel('x [m]'); ylabel('z [m]'); colorbar;

	// return
	return 0;
}