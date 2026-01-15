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
#include "solenosources.hh"

// fmm headers
#include "soleno.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// default constructor
	// default is hex
	SolenoSources::SolenoSources(){
		
	}

	SolenoSources::SolenoSources(
		const arma::Col<fltp>::fixed<3>& axis,
		const arma::Col<fltp>::fixed<3>& offset,
		const fltp inner_radius, 
		const fltp outer_radius, 
		const fltp zlower, 
		const fltp zupper, 
		const fltp current, 
		const fltp num_turns, 
		const arma::uword num_layers){
		
		// position and rotation
		axis_ = axis;
		offset_ = offset;

		// soleno geometry
		inner_radius_ = inner_radius;
		outer_radius_ = outer_radius;
		zlower_ = zlower;
		zupper_ = zupper;
		current_ = current;
		num_turns_ = num_turns;
		num_layers_ = num_layers;
	}

	// factory
	ShSolenoSourcesPr SolenoSources::create(){
		return std::make_shared<SolenoSources>();
	}

	ShSolenoSourcesPr SolenoSources::create(
		const arma::Col<fltp>::fixed<3>& axis,
		const arma::Col<fltp>::fixed<3>& offset,
		const fltp inner_radius, 
		const fltp outer_radius, 
		const fltp zlower, 
		const fltp zupper, 
		const fltp current, 
		const fltp num_turns, 
		const arma::uword num_layers){
		return std::make_shared<SolenoSources>(
			axis,offset,inner_radius,outer_radius,
			zlower,zupper,current,num_turns,num_layers);
	}

	// direct calculation
	void SolenoSources::calc_external(
		const ShTargetsPr &tar, 
		const ShSettingsPr &stngs, 
		const cmn::ShLogPr &lg) const{

		// return
		if(lg->is_cancelled())return;

		// get target coordinates
		const arma::Mat<fltp> Rt = tar->get_target_coords();

		// coordinate transformation onto soleno coordinate system
		const arma::Mat<fltp>::fixed<3,3> Mrot = cmn::Extra::create_rotation_matrix(axis_,{0,0,1});

		// create soleno object
		const ShSolenoPr soleno = Soleno::create();
		soleno->set_use_parallel(stngs->get_parallel_s2t());
		soleno->set_solenoid(inner_radius_,outer_radius_,zlower_,zupper_,current_,num_turns_,num_layers_);

		// run the field calculation with soleno
		if(tar->has('B') || tar->has('H')){
			const arma::Mat<fltp> B = arma::inv(Mrot)*soleno->calc_B(Mrot*(Rt.each_col()-offset_));
			if(tar->has('H'))tar->add_field('H',B/arma::Datum<fltp>::mu_0,false);
			if(tar->has('B'))tar->add_field('B',B,false);
		}

		if(tar->has('A')){
			const arma::Mat<fltp> A = arma::inv(Mrot)*soleno->calc_A(Mrot*(Rt.each_col()-offset_));
			tar->add_field('A',A,false);
		}
	}

}}