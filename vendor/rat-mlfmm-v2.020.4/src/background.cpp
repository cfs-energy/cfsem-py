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
#include "background.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// constructor
	Background::Background(){

	}

	// factory
	ShBackgroundPr Background::create(){
		return std::make_shared<Background>();
	}

	// helper for setting magnetic field
	void Background::set_magnetic(const arma::Col<fltp>::fixed<3> &Hbg){
		// create matrix
		// there is actually freedom of choice here
		// this is just one of infinite options
		arma::Mat<fltp>::fixed<3,3> Mgrad(arma::fill::zeros);
		
		const arma::Col<fltp>::fixed<3> Bbg = arma::Datum<fltp>::mu_0*Hbg;

		Mgrad(0,1) = -Bbg(2)/2;
		Mgrad(0,2) = Bbg(1)/2;

		Mgrad(1,2) = -Bbg(0)/2;
		Mgrad(1,0) = Bbg(2)/2;

		Mgrad(2,0) = -Bbg(1)/2;
		Mgrad(2,1) = Bbg(0)/2;

		// set fields
		add_field('H',Hbg,arma::Mat<fltp>::fixed<3,3>(arma::fill::zeros));
		add_field('B',arma::Datum<fltp>::mu_0*Hbg,arma::Mat<fltp>::fixed<3,3>(arma::fill::zeros));
		add_field('A',cmn::Extra::null_vec(),Mgrad);
	}

	// has field
	bool Background::has_field()const{
		// check if field is non-zero
		for(arma::uword i=0;i<bgfld_.n_elem;i++)
			if(arma::any(bgfld_(i)!=RAT_CONST(0.0)))
				return true;

		// no field found
		return false;
	}

	// add field
	void Background::add_field(
		const char type, 
		const arma::Col<fltp>::fixed<3>&bgfld, 
		const arma::Mat<fltp>::fixed<3,3>&bggrad){

		// counter
		const arma::uword num_field = field_type_.n_elem;

		// allocate
		arma::field<char> new_field_type(field_type_.n_elem+1);
		arma::field<arma::Col<fltp>::fixed<3> > new_bgfld(field_type_.n_elem+1);
		arma::field<arma::Mat<fltp>::fixed<3,3> > new_bggrad(field_type_.n_elem+1);

		// copy old
		for(arma::uword i=0;i<num_field;i++){
			new_field_type(i) = field_type_(i);
			new_bgfld(i) = bgfld_(i);
			new_bggrad(i) = bggrad_(i);
		}

		// add new 
		new_field_type(num_field) = type;
		new_bgfld(num_field) = bgfld;
		new_bggrad(num_field) = bggrad;

		// set to self
		field_type_ = new_field_type;
		bgfld_ = new_bgfld; 
		bggrad_ = new_bggrad;
	}

	// set field to targets
	void Background::calc_field(const ShTargetsPr &tar) const{
		// walk over field types
		for(arma::uword i=0;i<field_type_.n_elem;i++){
			const char mystr = field_type_(i);
			if(tar->has(mystr)){
				// get elements
				const arma::uword num_dim = bgfld_(i).n_rows;
				const arma::uword num_targets = tar->num_targets();

				// get coordinates
				const arma::Mat<fltp> Rt = tar->get_target_coords();

				// create field matrix
				arma::Mat<fltp> fld(num_dim,num_targets);

				// set field
				fld.each_col() = bgfld_(i);

				// set field gradient
				fld += bggrad_(i)*Rt;

				// add field to targets
				tar->add_field(mystr,fld,false);
			}
		}

	}

}}