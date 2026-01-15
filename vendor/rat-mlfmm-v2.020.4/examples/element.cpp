
// general headers
#include <armadillo>

// specific headers
#include "currentsources.hh"
#include "mgntargets.hh"
#include "mlfmm.hh"


int main(){
	arma::uword nx = 1000;
	arma::uword ny = 1001;
	arma::Col<rat::fltp> Rs{0,0,0};
	arma::Col<rat::fltp> dRs{0.01,0,0};
	arma::Row<rat::fltp> Is{1000.0};
	arma::Row<rat::fltp> epss{0.002};

	// create grid
	arma::Row<rat::fltp> xa = arma::linspace<arma::Row<rat::fltp> >(-0.05,0.05,nx);
	arma::Col<rat::fltp> ya = arma::linspace<arma::Col<rat::fltp> >(-0.05,0.05,ny);

	// create arrays
	arma::Mat<rat::fltp> xm(ny,nx); xm.each_row() = xa;
	arma::Mat<rat::fltp> ym(ny,nx); ym.each_col() = ya;
	arma::Mat<rat::fltp> zm(ny,nx,arma::fill::zeros);

	// fix this shit
	arma::Mat<rat::fltp> Rt = arma::join_vert(
		arma::reshape(xm, 1, nx*ny),
		arma::reshape(ym, 1, nx*ny),
		arma::reshape(zm, 1, nx*ny));

	// setup source and target objects
	rat::fmm::ShCurrentSourcesPr src = rat::fmm::CurrentSources::create(Rs,dRs,Is,epss);
	src->set_van_Lanen(true);
	// src->set_so2ta_enable_gpu(true);
	rat::fmm::ShMgnTargetsPr tar = rat::fmm::MgnTargets::create(Rt); tar->set_field_type('H',3);

	// run MLFMM on sources and targets
	rat::fmm::ShMlfmmPr mlfmm = rat::fmm::Mlfmm::create(src,tar);
	mlfmm->calculate_direct();

	// get resulting field in 3 by N matrix with rows Bx, By and Bz
	const arma::Mat<rat::fltp> B = tar->get_field('B');

	// std::cout<<arma::reshape(B.row(0),ny,nx)<<std::endl;
	// std::cout<<arma::reshape(B.row(1),ny,nx)<<std::endl;
	arma::Mat<rat::fltp> Bmz = arma::reshape(B.row(2),ny,nx);
	// Bmz.save("bfield.csv",arma::csv_ascii);

	return 0;
}