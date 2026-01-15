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
#include "soleno.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// constructor
	Soleno::Soleno(){

	}

	// factory
	ShSolenoPr Soleno::create(){
		//return UnNodeLevelPr(new NodeLevel);
		return std::make_shared<Soleno>();
	}

	// sign function
	int Soleno::sign(fltp a){
		int b;
		if(a>0)b=1;
		if(a<0)b=-1;
		if(a==0)b=0;
		return b;
	}

	// set parallel switch
	void Soleno::set_use_parallel(const bool use_parallel){
		use_parallel_ = use_parallel;
	}

	// calculation function
	// from Fawzi, "The accurate computation of self and mutual inductance of circular coils", 
	// IEEE Transactions on Power Apparatus and Systems, Vol. PAS-97, no. 2, March/April 1978
	// Appendix B : Modified Bulirsch algorithm for complete elliptic integrals,
	// this calculate the field [Brh,Bzh] for a current loop at[Rm,Zmh]
	void Soleno::rekenb(fltp *Brh, fltp *Bzh,
		const fltp Zmh, const fltp Rm, const fltp Rp){

		const fltp Hulp = std::sqrt(Rp * Rp + Zmh * Zmh);
		fltp Kc = std::sqrt(Rm * Rm + Zmh * Zmh) / Hulp;
		if(Kc <  RAT_CONST(1e-12))Kc =  RAT_CONST(1e-12);
		double Mj = 1.0; // must be double because float is not accurate enough
		fltp Nj = Kc;
		fltp Aj = -1.0;
		fltp Bj = 1.0;
		fltp Pj = Rm / Rp;
		fltp Cj = Pj + 1.0;
		if(std::abs(Rm / Rp) <  RAT_CONST(1e-12))Pj = 1.0;
		fltp Dj = Cj * Soleno::sign(Pj);
		Pj = std::abs(Pj);
		#ifndef NDEBUG
		int Tel = 1;
		#endif
		while(std::abs(Mj - Nj) > (RAT_CONST(1e-12) * Mj)){
			const fltp Xmn = Mj * Nj;
			const fltp Xmp = Xmn / Pj;
			const fltp D0 = Dj;
			Dj = 2 * (Dj + Xmp * Cj);
			Cj = Cj + D0 / Pj;
			const fltp B0 = Bj;
			Bj = 2 * (Bj + Nj * Aj);
			Aj = Aj + B0 / Mj;
			Pj = Pj + Xmp;
			Mj = Mj + Nj;
			Nj = 2 * std::sqrt(Xmn);
			#ifndef NDEBUG
			Tel++;
			#endif
			assert(Tel<50);
		}
		
		*Brh = (Aj + Bj / Mj) / (Mj * Hulp);
		*Bzh = (Cj + Dj / Mj) / ((Mj + Pj) * Hulp) * Zmh;
	}

	// Original Soleno Function (used by improved version)
	void Soleno::calc_B_original(
		fltp *Brt, fltp *Bzt, const fltp R, 
		const fltp Z, const fltp Rin, const fltp Rout, 
		const fltp Zlow, const fltp Zhigh, const fltp It, 
		const arma::uword Nlayers){
			
		fltp Brd = 0;
		fltp Bzd = 0;
		fltp Br1,Br2,Bz1,Bz2;
		
		if (Rin != Rout && Zlow != Zhigh){
			fltp Factor = It / (Zhigh - Zlow) / Nlayers;
			if (Factor != 0){
				fltp Zm1 = Z - Zlow;
				fltp Zm2 = Z - Zhigh;
				fltp dR = (Rout - Rin) / Nlayers;
				fltp dH = dR / 2;
				fltp dE = dH;
				for (arma::uword tt = 1; tt<=Nlayers; tt++){
					fltp Rs = Rin - dH + dR * tt;
					if (std::abs(R - Rs) >= dE){
						// extern
						fltp Rp = Rs + R;
						fltp Rm = Rs - R;
						Soleno::rekenb(&Br1,&Bz1,Zm1,Rm,Rp);
						Soleno::rekenb(&Br2,&Bz2,Zm2,Rm,Rp);
						Brd = Brd + (Br2 - Br1) * Factor * Rs;
						Bzd = Bzd + (Bz1 - Bz2) * Factor;
					}else{
						// intern
						// RIntern = true;
						fltp Rp = Rs + Rs - dE;
						fltp Rm = dE;
						fltp Weeg = (Rs - R + dE) / dE / 2;
						Soleno::rekenb(&Br1,&Bz1,Zm1,Rm,Rp);
						Soleno::rekenb(&Br2,&Bz2,Zm2,Rm,Rp);
						Brd = Brd + (Br2 - Br1) * Weeg * Factor * Rs;
						Bzd = Bzd + (Bz1 - Bz2) * Weeg * Factor;
						Weeg = 1 - Weeg;
						Rp = Rs + Rs + dE;
						Rm = -dE;
						Soleno::rekenb(&Br1,&Bz1,Zm1,Rm,Rp);
						Soleno::rekenb(&Br2,&Bz2,Zm2,Rm,Rp);
						Brd = Brd + (Br2 - Br1) * Weeg * Factor * Rs;
						Bzd = Bzd + (Bz1 - Bz2) * Weeg * Factor;
					}
				}
			}
		}
		
		*Brt = Brd * arma::Datum<fltp>::pi * RAT_CONST(1e-7);
		*Bzt = Bzd * arma::Datum<fltp>::pi * RAT_CONST(1e-7);
	}

	// Improved version of Soleno
	// better field in the radial direction
	// this method is called "Richardson Extrapolation"
	void Soleno::calc_B_extended(
		fltp *Brt, fltp *Bzt, const fltp R, 
		const fltp Z, const fltp Rin, const fltp Rout, 
		const fltp Zlow, const fltp Zhigh, const fltp I, 
		const arma::uword Nlayers){
		
		// set layers if not set remotely 5 is default value
		if (R >= Rin && R <= Rout){
			// internal
			fltp A = (R - Rin) / (Rout - Rin);
			int NL1 = int(Nlayers * A) + 1;
			int NL2 = int(Nlayers * (1 - A)) + 1;
			
			// deel 1
			Soleno::calc_B_original(Brt, Bzt, R, Z, Rin, R, Zlow, Zhigh, I * A, NL1);
			fltp Br2 = *Brt;
			fltp Bz2 = *Bzt;
			Soleno::calc_B_original(Brt, Bzt, R, Z, Rin, R, Zlow, Zhigh, I * A, NL1 * 2);
			fltp BZtemp = *Bzt + (*Bzt - Bz2) / 3; //Extrapolatie naar 1/Nl/Nl=0
			fltp BRtemp = *Brt + (*Brt - Br2) / 3;
			
			// deel 2
			Soleno::calc_B_original(Brt, Bzt, R, Z, R, Rout, Zlow, Zhigh, I * (1 - A), NL2);
			Bz2 = *Bzt;
			Br2 = *Brt;
			Soleno::calc_B_original(Brt, Bzt, R, Z, R, Rout, Zlow, Zhigh, I * (1 - A), NL2 * 2);
			
			*Bzt = BZtemp + *Bzt + (*Bzt - Bz2) / 3; //Extrapolatie naar 1/Nl/Nl=0
			*Brt = BRtemp + *Brt + (*Brt - Br2) / 3;
		}else{
			// external
			Soleno::calc_B_original(Brt, Bzt, R, Z, Rin, Rout, Zlow, Zhigh, I, Nlayers);
			fltp Bz2 = *Bzt;
			fltp Br2 = *Brt;
			Soleno::calc_B_original(Brt, Bzt, R, Z, Rin, Rout, Zlow, Zhigh, I, Nlayers * 2);
			*Bzt = *Bzt + (*Bzt - Bz2) / 3; //Extrapolatie naar 1/Nl/Nl=0
			*Brt = *Brt + (*Brt - Br2) / 3;
		}
	}

	// set single solenoid
	void Soleno::set_solenoid(
		const fltp Rin, const fltp Rout, 
		const fltp zlow, const fltp zhigh, 
		const fltp current, const fltp num_turns,
		const arma::uword num_layers){

		// check input
		assert(Rout>Rin);
		assert(zhigh>zlow);
		assert(num_layers>0);

		// convert to arrays and set to self
		Rin_ = arma::Row<fltp>{Rin};
		Rout_ = arma::Row<fltp>{Rout};
		zlow_ = arma::Row<fltp>{zlow};
		zhigh_ = arma::Row<fltp>{zhigh};
		current_ = arma::Row<fltp>{current};
		num_turns_ = arma::Row<fltp>{num_turns};
		num_layers_ = arma::Row<arma::uword>{num_layers};

		// set number of coils
		num_solenoids_ = 1;
	}

	// set multiple solenoids
	void Soleno::set_solenoid(
		const arma::Row<fltp> &Rin, const arma::Row<fltp> &Rout, 
		const arma::Row<fltp> &zlow, const arma::Row<fltp> &zhigh, 
		const arma::Row<fltp> &current, const arma::Row<fltp> &num_turns,
		const arma::Row<arma::uword> &num_layers){

		// check input
		assert(arma::all(Rout>Rin));
		assert(arma::all(zhigh>zlow));
		assert(arma::all(num_layers>0));

		// set solenoids
		Rin_ = Rin; Rout_ = Rout; zlow_ = zlow; 
		zhigh_ = zhigh; current_ = current; num_turns_ = num_turns;
		num_layers_ = num_layers;
		
		// set number of coils
		num_solenoids_ = Rin.n_elem;
	}

	// add single solenoid
	void Soleno::add_solenoid(
		const fltp Rin, const fltp Rout, 
		const fltp zlow, const fltp zhigh, 
		const fltp current, const fltp num_turns,
		const arma::uword num_layers){

		// check input
		assert(Rout>Rin);
		assert(zhigh>zlow);
		assert(num_layers>0);

		// increment number of coils
		num_solenoids_++;

		// resize storage
		Rin_.resize(num_solenoids_); Rout_.resize(num_solenoids_);
		zlow_.resize(num_solenoids_); zhigh_.resize(num_solenoids_);
		current_.resize(num_solenoids_); num_turns_.resize(num_solenoids_);
		num_layers_.resize(num_solenoids_);

		// add new solenoid
		Rin_(num_solenoids_-1) = Rin; Rout_(num_solenoids_-1) = Rout;
		zlow_(num_solenoids_-1) = zlow; zhigh_(num_solenoids_-1) = zhigh;
		current_(num_solenoids_-1) = current; num_turns_(num_solenoids_-1) = num_turns;
		num_layers_(num_solenoids_-1) = num_layers;
	}

	// add multiple solenoids
	void Soleno::add_solenoid(
		const arma::Row<fltp> &Rin, const arma::Row<fltp> &Rout, 
		const arma::Row<fltp> &zlow, const arma::Row<fltp> &zhigh, 
		const arma::Row<fltp> &current, const arma::Row<fltp> &num_turns,
		const arma::Row<arma::uword> &num_layers){

		// check input
		assert(arma::all(Rout>Rin));
		assert(arma::all(zhigh>zlow));
		assert(arma::all(num_layers>0));

		// increment number of coils
		num_solenoids_+=Rin.n_elem;

		// add solenoids
		Rin_ = arma::join_horiz(Rin_,Rin); Rout_ = arma::join_horiz(Rout_,Rout);
		zlow_ = arma::join_horiz(zlow_,zlow); zhigh_ = arma::join_horiz(zhigh_,zhigh);
		current_ = arma::join_horiz(current_,current); num_turns_ = num_turns;
		num_layers_ = arma::join_horiz(num_layers_,num_layers);
	}

	// calculation of flux density at specified target points
	arma::Mat<fltp> Soleno::calc_B(const arma::Mat<fltp> &Rt) const{
		// check input
		assert(Rin_.n_elem==Rout_.n_elem);
		assert(Rin_.n_elem==zlow_.n_elem);
		assert(Rin_.n_elem==zhigh_.n_elem);
		assert(Rin_.n_elem==current_.n_elem);
		assert(Rin_.n_elem==num_layers_.n_elem);
		assert(arma::all(num_layers_>0));

		// split coordinates
		arma::Row<fltp> x = Rt.row(0);
		arma::Row<fltp> y = Rt.row(1);
		arma::Row<fltp> z = Rt.row(2);
		const arma::uword num_targets = z.n_elem;

		// convert to polar coordinates
		arma::Row<fltp> rho = arma::hypot(x,y);

		// create output matrices
		arma::Row<fltp> Brt(num_targets);
		arma::Row<fltp> Bzt(num_targets);

		// walk over target points
		cmn::parfor(0,num_targets,use_parallel_,[&](arma::uword i, int){
			// allocate
			fltp Brt_local = RAT_CONST(0.0);
			fltp Bzt_local = RAT_CONST(0.0);

			// walk over solenoids
			for(arma::uword j=0;j<num_solenoids_;j++){
				// allocate output
				fltp Br = 0.0, Bz= 0.0;

				// calculate with original soleno
				calc_B_extended(&Br,&Bz,rho(i),z(i),
					Rin_(j),Rout_(j),zlow_(j),zhigh_(j),
					num_turns_(j)*current_(j),num_layers_(j));

				// add fields
				Brt_local += Br; Bzt_local += Bz;
			}

			// set to global arrays
			Brt(i) = Brt_local; Bzt(i) = Bzt_local;
		});

		// check output before returning
		assert(Brt.is_finite());
		assert(Bzt.is_finite());

		// convert magnetic flux density to cartesian
		arma::Mat<fltp> Bt(3,num_targets);
		Bt.row(0) = (x/rho)%Brt; 
		Bt.row(1) = (y/rho)%Brt;
		Bt.row(2) = Bzt;
		
		// x and y components are zero on the z-axis
		Bt(arma::Col<arma::uword>{0,1}, arma::find(rho==0)).fill(RAT_CONST(0.0));

		// check output before returning
		assert(Brt.is_finite());
		assert(Bzt.is_finite());
		
		// return target field
		return Bt;
	}

	// this function computes the azimuthal vector potential (Aphi) at a target point.
	fltp Soleno::calc_A_core(
		const fltp R, const fltp Z,
		 const fltp Rin, const fltp Rout,
		 const fltp Zlow, const fltp Zhigh,
		 const fltp I, const arma::uword Nlayers){

		// differential radial element.
		fltp Dr = (Rout - Rin) / Nlayers;
		fltp sum = 0.0;

		// vertical coordinates
		const fltp dZ = RAT_CONST(1e-7); // ugly finite difference distance
		const fltp S1 = std::abs(Zhigh - Z-dZ/2);
		const fltp S2 = std::abs(Zlow - Z-dZ/2);
		const fltp S3 = std::abs(Zlow - Z+dZ/2);
		const fltp S4 = std::abs(Zhigh - Z+dZ/2);

		// Loop over radial layers
		for (arma::uword k=1;k<=Nlayers;k++){
			// Midpoint radius of the kth annulus.
			const fltp Rs = Rin + (static_cast<fltp>(k) - 0.5) * Dr;

			// use the mutual inductance kernel.
			const fltp C1 = Sol_CI(Rs, R, S1);
			const fltp C2 = Sol_CI(Rs, R, S2);
			const fltp C3 = Sol_CI(Rs, R, S3);
			const fltp C4 = Sol_CI(Rs, R, S4);

			// accumulate the weighted kernel.
			sum += std::pow(Rs,1.5)*std::sqrt(R) * (C1-C2+C3-C4) / dZ;
		}

		// scale solution
		sum *= arma::Datum<fltp>::mu_0*I/Nlayers/(Zhigh-Zlow);

		// return vector potential
		return sum;
	}

	// calculation of flux density at specified target points
	arma::Mat<fltp> Soleno::calc_A(const arma::Mat<fltp> &Rt) const{
		// check input
		assert(Rin_.n_elem==Rout_.n_elem);
		assert(Rin_.n_elem==zlow_.n_elem);
		assert(Rin_.n_elem==zhigh_.n_elem);
		assert(Rin_.n_elem==current_.n_elem);
		assert(Rin_.n_elem==num_layers_.n_elem);
		assert(arma::all(num_layers_>0));

		// split coordinates
		arma::Row<fltp> x = Rt.row(0);
		arma::Row<fltp> y = Rt.row(1);
		arma::Row<fltp> z = Rt.row(2);

		// convert to polar coordinates
		arma::Row<fltp> rho = arma::sqrt(x%x + y%y);
		const arma::uword num_targets = z.n_elem;

		// create output matrices
		arma::Row<fltp> Aphit(num_targets);

		// walk over target points
		cmn::parfor(0,num_targets,use_parallel_,[&](arma::uword i, int){
			// allocate
			fltp Aphilocal = RAT_CONST(0.0);

			// walk over solenoids
			for(arma::uword j=0;j<num_solenoids_;j++){
				// calculate vector potential for two different number of layers
				const fltp Aphi2 = calc_A_core(rho(i),z(i),
					Rin_(j),Rout_(j),zlow_(j),zhigh_(j),
					num_turns_(j)*current_(j),num_layers_(j)); // for number of layers
				const fltp Aphi1 = calc_A_core(rho(i),z(i),
					Rin_(j),Rout_(j),zlow_(j),zhigh_(j),
					num_turns_(j)*current_(j),num_layers_(j)*2); // for twice number of layers

				// richardson extrapolation and store
				Aphilocal += Aphi1 + (Aphi1 - Aphi2)/3; 
			}

			// add to target
			Aphit(i) = Aphilocal;
		});

		// check output before returning
		assert(Aphit.is_finite());

		// For each point (x, y, z):
		arma::Mat<fltp> At(3,num_targets); 
		At.row(0) = (y/rho)%Aphit;
		At.row(1) = (-x/rho)%Aphit;
		At.row(2).zeros();

		// fix nan
		At.cols(arma::find(rho==0)).fill(0);

		// check output before returning
		assert(Aphit.is_finite());

		// return target field
		return At;
	}

	// Function for calculating elliptical integrals
	fltp Soleno::Sol_CI(const fltp R1, const fltp R2, const fltp ZZ){
		int Imax = 25;
		int Itel = 0;
		
		fltp Ci = 0.0;
		fltp Rm = R1 - R2;
		fltp Rp = R1 + R2;
		fltp Zkw = ZZ * ZZ;
		fltp Rkw = 4 * R1 * R2;
		fltp Kkw = (Rm * Rm + Zkw) / (Rp * Rp + Zkw);
		fltp tol = 10*arma::Datum<fltp>::eps;
		if(Kkw < tol){
			Ci = RAT_CONST(1.0) / (6 * std::atan(1.0));
		}else{
			fltp Kc = std::sqrt(Rkw / (Rp * Rp + Zkw));
			fltp Alfa1 = 1;
			fltp Beta1 = std::sqrt(Kkw);
			fltp Q1 = std::abs(Rm / Rp);
			bool R1eqR2 = (Q1 <= RAT_CONST(1e-10));
			fltp A1 = RAT_CONST(1.0) / (3 * Kc);
			fltp B1 = -Kkw * A1;
			fltp C1 = Kc * Zkw / Rkw;
			fltp D1 = 0.0;
			A1 = A1 - C1;
			if(R1eqR2){
				A1 = A1 + C1;
				B1 = -B1 - B1;
				C1 = 0.0;
			}
			while(std::abs(Alfa1 - Beta1) > Alfa1 * tol && Itel < Imax){
				Itel = Itel + 1;
				fltp AlfBe1 = Alfa1 * Beta1;
				if(R1eqR2==false){
					fltp C2 = C1 + D1 / Q1;
					fltp D2 = AlfBe1 * C1 / Q1 + D1;
					D2 = D2 + D2;
					fltp Q2 = Q1 + AlfBe1 / Q1;
					C1 = C2;
					D1 = D2;
					Q1 = Q2;
				}
				fltp Alfa2 = Alfa1 + Beta1;
				fltp Beta2 = 2 * std::sqrt(AlfBe1);
				fltp A2 = B1 / Alfa1 + A1;
				fltp B2 = B1 + Beta1 * A1;
				B2 = B2 + B2;
				Alfa1 = Alfa2;
				Beta1 = Beta2;
				A1 = A2;
				B1 = B2;
			}
			Ci = (A1 + B1 / Alfa1) / (Alfa1 + Alfa1) + (C1 + D1 / Alfa1) / (Alfa1 + Q1);
			assert(Itel < Imax);
		}
		return Ci;
	}

	// Main mutual inductance function returns mutual inductance between coil I and J
	fltp Soleno::CalcMutSub(
		const fltp zlow1, const fltp zlow2, const fltp zhigh1, const fltp zhigh2, 
		const fltp Rin1, const fltp Rin2, const fltp Rout1, const fltp Rout2,
		const fltp Nw1, const fltp Nw2, 
		const arma::uword NL1, const arma::uword NL2){
		
		fltp S1 = std::abs(zhigh1 - zlow2);
		fltp S2 = std::abs(zlow1 - zlow2);
		fltp S3 = std::abs(zlow1- zhigh2);
		fltp S4 = std::abs(zhigh1 - zhigh2);
		bool S3neS1 = (S3 != S1);
		bool S4neS2 = (S4 != S2);
		bool RdivNE = (Rin1 != Rin2) || (Rout1 != Rout2) || (NL1 != NL2);
		fltp Factor = 0.0000008 * arma::Datum<fltp>::pi * arma::Datum<fltp>::pi * 
			Nw1 / NL1 / (zhigh1 - zlow1) * Nw2 / NL2 / (zhigh2 - zlow2);
		fltp DrI = (Rout1 - Rin1) / NL1;
		fltp DrJ = (Rout2 - Rin2) / NL2;
		fltp Som = 0.0;
		
		for(arma::uword K=1;K<=NL1;K++){
			fltp R1 = Rin1 + (static_cast<fltp>(K) -  RAT_CONST(0.5)) * DrI;
			for(arma::uword L=1;L<=NL2;L++){
				fltp R2 = Rin2 + (static_cast<fltp>(L) -  RAT_CONST(0.5)) * DrJ;
				if(RdivNE || (L <= K)){
					fltp C1 = Sol_CI(R1, R2, S1);
					fltp C2 = Sol_CI(R1, R2, S2);
					fltp C3 = C1;
					fltp C4 = C2;
					if(S3neS1)C3 = Sol_CI(R1, R2, S3);
					if(S4neS2)C4 = Sol_CI(R1, R2, S4);
					fltp HH = R1 * R2;
					HH = sqrt(HH * HH * HH);
					if((RdivNE==false) && (L < K))HH = 2 * HH;
					Som = Som + (C1 - C2 + C3 - C4) * HH;
				}
			}
		}
		return Som * Factor;
	}

	// Mutual inductance routine
	arma::Mat<fltp> Soleno::calc_M() const{
		
		// Allocate output
		arma::Mat<fltp> M(num_solenoids_,num_solenoids_,arma::fill::zeros);
			
		// create indexes
		arma::Row<arma::uword> idx1(num_solenoids_*num_solenoids_);
		arma::Row<arma::uword> idx2(num_solenoids_*num_solenoids_);

		// walk over source
		for(arma::uword i=0; i<num_solenoids_; i++){
			// walk over target
			for(arma::uword j=0; j<num_solenoids_; j++){
				idx1(i*num_solenoids_ + j) = i;
				idx2(i*num_solenoids_ + j) = j;	
			}
		}

		// get triangle
		const arma::Row<arma::uword> id = arma::find(idx1>=idx2).t();
		idx1 = idx1.cols(id); idx2 = idx2.cols(id);

		// walk over coil combinations
		cmn::parfor(0,idx1.n_elem,use_parallel_,[&](arma::uword k, int){
			// get indexes
			arma::uword i = idx1(k), j = idx2(k);

			// allocate helper variable
			fltp Hulp, Hulp2;

			// calculate
			Hulp2 = CalcMutSub(zlow_(i), zlow_(j), zhigh_(i), zhigh_(j),
				Rin_(i), Rin_(j), Rout_(i),Rout_(j), num_turns_(i), num_turns_(j), 
				num_layers_(i), num_layers_(j)); // for number of layers
			Hulp = CalcMutSub(zlow_(i), zlow_(j), zhigh_(i), zhigh_(j),
				Rin_(i), Rin_(j), Rout_(i),Rout_(j), num_turns_(i), num_turns_(j), 
				2*num_layers_(i), 2*num_layers_(j)); // for twice number of layers

			// extrapolation
			Hulp = Hulp+(Hulp - Hulp2)/3.0; //extrapolatie naar 1/NLI/NLJ=0

			// store to matrix
			M(i,j) = Hulp; 
			if(j!=i)M(j,i) = Hulp;
		});

		// return inductance matrix
		return M;
	}

	// calculate field on targets
	void Soleno::calc_field(const ShTargetsPr &tar){
		// no field needed
		if(!tar->has('H') && !tar->has('B'))return;

		// get target coordinates
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// calculate field at target points
		const arma::Mat<fltp> B = calc_B(Rt);

		// set field to targets
		if(tar->has('H'))tar->add_field('H',B/arma::Datum<fltp>::mu_0,false);
		if(tar->has('B'))tar->add_field('B',B,false);
	}


}}
