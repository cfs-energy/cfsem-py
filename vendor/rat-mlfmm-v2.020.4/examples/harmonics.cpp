
// general headers
#include <armadillo>

// specific headers
#include "currentsources.hh"
#include "mgntargets.hh"
#include "mlfmm.hh"


#include "spharm.hh"

int main(){

	rat::fltp x = 0.01;
	rat::fltp y = 0.005;
	rat::fltp z = -0.007;
	int num_exp = 5;

	arma::Col<rat::fltp> facs = rat::fmm::Extra::factorial(2*num_exp);

	rat::fmm::Spharm harmonics(num_exp,rat::fmm::Extra::cart2sph(arma::Col<rat::fltp>{x,y,z}));
	harmonics.display();

	// position in spherical coordinates
	rat::fltp rho = std::sqrtf(x*x+y*y+z*z);
	rat::fltp mu = std::max(-1.0f,std::min(1.0f,z/rho));
	rat::fltp phi = std::atan2(y,x);

	// storing legendre polynomials
	rat::fltp Pn = 1.0f;
	rat::fltp fact = 1.0f;
	rat::fltp s = std::sqrtf(1.0f - mu*mu);

	// walk over m
	for(int m=0;m<=num_exp;m++){
		// update legendre polynomials
		// P^(m)_(n+1), P^(m)_(n), P^(m)_(n-1), respectively
		rat::fltp Pnm = Pn; 
		rat::fltp Pnm1 = Pnm; 

		// rat::fltp facnm = factorial(n-m);
		rat::fltp sqfac = std::sqrtf(1.0/facs(2*m));

		// calculate rho^n
		rat::fltp rhom = std::pow(rho, rat::fltp(m));

		// calculate sin and cos of angle
		rat::fltp cosphim = std::cos(phi*m);
		rat::fltp sinphim = std::sinf(phi*m);

		// calculate real and complex parts of Ynm
		rat::fltp Ynmre = sqfac*Pnm*cosphim;
		rat::fltp Ynmim = sqfac*Pnm*sinphim; // note that this is for Y_{n, -m}

		// std::cout<<sqfac<<std::endl;

		// store in multipole for diagonal
		// so2mp_core(localMp, myIeff, Ynmre, Ynmim, rho, phi, m, m, polesize);
		

		// second
		Pnm = mu*(2*m+1)*Pnm;

		// walk over n
		for(int n=m+1;n<=num_exp;n++){
			// rat::fltp facnm = factorial(n-m);
			sqfac = std::sqrtf(facs(n-m)/facs(n+m));

			// calculate rho^n
			rat::fltp rhon = std::pow(rho, rat::fltp(n));

			// calculate real and complex parts of Ynm
			Ynmre = sqfac*Pnm*cosphim;
			Ynmim = sqfac*Pnm*sinphim; // note that this is for Y_{n, -m}

			std::cout<<n<<" "<<m<<" "<<Ynmre<<" "<<Ynmim<<std::endl;

			// shift Legendre polynomials eq. 2.1.54, p48
			rat::fltp Pnm2 = Pnm1; Pnm1 = Pnm;
			Pnm = (mu*(2*n+1)*Pnm1-(n+m)*Pnm2)/(n-m+1);
			//Pnm = (mu*(2*n-1)*Pnm1 - (n+m-1)*Pnm2)/(n-m);
		}

		// update legendre polynomial
		Pn *= -fact*s;

		// update factor
		fact += 2;
	}


	return 0;
}