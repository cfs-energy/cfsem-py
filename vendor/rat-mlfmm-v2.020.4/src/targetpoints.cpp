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
#include "targetpoints.hh"

// code specific to Rat
namespace rat{namespace fmm{

	// default constructor
	TargetPoints::TargetPoints(){

	}	

	// default constructor
	TargetPoints::TargetPoints(const arma::Mat<fltp> &Rt){
		// set coordinates using appropriate method
		set_target_coords(Rt);
	}

	// factory
	ShTargetPointsPr TargetPoints::create(){
		return std::make_shared<TargetPoints>();
	}

	// factory
	ShTargetPointsPr TargetPoints::create(const arma::Mat<fltp> &Rt){
		return std::make_shared<TargetPoints>(Rt);
	}

	// clear field type
	void TargetPoints::clear_field_type(){
		field_type_.clear(); M_.clear();
	}

	// set field type and number of dimensions
	void TargetPoints::set_field_type(
		const std::string &field_type, 
		const arma::Row<arma::uword> &num_dim){

		// check if all values set
		if(field_type.length()!=num_dim.n_elem)rat_throw_line("field type string must equal length of dimensions vector");

		// reset field type
		clear_field_type();

		// walk over field types
		arma::uword i=0;
		for(auto it = field_type.begin();it!=field_type.end();it++,i++){
			// add field type
			add_field_type((*it),num_dim(i));	
		}

	}
	
	// settings
	void TargetPoints::set_field_type(
		const char field_type, 
		const arma::uword num_dim){

		// reset field type
		clear_field_type();

		// add field type
		add_field_type(field_type,num_dim);		
	}

	// set field tyep
	void TargetPoints::set_field_type(
		const std::unordered_map<char, arma::uword>&field_type){
		field_type_ = field_type;
	}

	// add field type
	void TargetPoints::add_field_type(
		const char field_type,
		const arma::uword num_dim){

		// add to list
		field_type_[field_type] = num_dim;
	}

	// check if calculated field stored
	bool TargetPoints::has_field() const{
		return !M_.empty();
	}

	// get number of dimensions
	arma::uword TargetPoints::get_target_num_dim(const char type) const{
		// return number of dimensions
		assert(has(type));
		return field_type_.at(type);
	}

	// check what field types requested
	bool TargetPoints::has(const char type) const{
		return field_type_.find(type)!=field_type_.end();
	}

	// set coordinates
	void TargetPoints::set_target_coords(
		const arma::Mat<fltp> &Rt){

		// check input
		if(Rt.n_rows!=3)rat_throw_line("target coordinate matrix must have three rows");
		if(!Rt.is_finite())rat_throw_line("target coordinate matrix must be finite");
		
		// set coordinates
		Rt_ = Rt;

		// set number of points
		num_targets_ = Rt.n_cols;
	}


	// allocate field
	void TargetPoints::allocate(){
		// check if all important stuff is set
		assert(num_targets()!=0);
		assert(!field_type_.empty());

		// walk over field types
		for(auto it=field_type_.begin();it!=field_type_.end();it++){
			// get values
			const char type = (*it).first; const arma::uword num_dim = (*it).second;

			// create key matrix pair with empty matrix
			M_.emplace(type, arma::Mat<fltp>());

			// set matrix size
			M_[type].zeros(num_dim, num_targets());

			// check if pair exists
			assert(M_.at(type).n_rows==num_dim);
			assert(M_.at(type).n_cols==num_targets());
		}

		// create a lock
		if(lock_==NULL)
			lock_ = std::unique_ptr<std::mutex>(new std::mutex);
	}

	// add field values
	void TargetPoints::add_field(
		const char type, 
		const arma::Mat<fltp> &Madd,
		const bool with_lock){

		if(!has(type))return;

		// check setup
		assert(!M_.empty());
		assert(Madd.n_rows==field_type_[type]);
		assert(M_.at(type).n_rows==field_type_[type]);
		assert(Madd.n_cols==num_targets_);

		// add field to matrix with locking
		assert(lock_!=NULL);
		if(with_lock){
			const std::lock_guard<std::mutex> lock_guard(*lock_);
			M_.at(type) += Madd;
		}

		// without locking
		else{
			M_.at(type) += Madd;
		}
	}

	// add field values
	void TargetPoints::add_field(
		const char type, 
		const arma::Mat<fltp> &Madd,
		const arma::Row<arma::uword> &ft,
		const arma::Row<arma::uword> &lt,
		const arma::Row<arma::uword> &ti,
		const bool with_lock){

		if(!has(type))return;

		// add field to matrix with locking
		assert(lock_!=NULL);
		if(with_lock){
			lock_->lock();
			for(arma::uword i=0;i<ft.n_elem;i++){
				M_.at(type).cols(ft(i),lt(i)) += Madd.cols(ti(i),ti(i+1)-1);
			}
			lock_->unlock();
		}

		// without locking
		else{
			for(arma::uword i=0;i<ft.n_elem;i++){
				M_.at(type).cols(ft(i),lt(i)) += Madd.cols(ti(i),ti(i+1)-1);
			}
		}
	}

	// get field from internal storage
	arma::Mat<fltp> TargetPoints::get_field(
		const char type) const{

		// make sure that type is only one character
		assert(!M_.empty());
		
		// check if this type exists
		if(!has(type))rat_throw_line("targets do not have this type: " + std::string(1,type));
		if(!has_field())rat_throw_line("targets did not have field calculated on them.");

		// add field to matrix
		return M_.at(type);
	}

	// get target point coordinates
	arma::Mat<fltp> TargetPoints::get_target_coords() const{
		assert(!Rt_.is_empty());
	 	return Rt_;
	}

	// // get target point coordinates
	// arma::Mat<fltp> TargetPoints::get_target_coords(
	// 	const char type) const{
	// 	assert(!Rt_.is_empty());
	// 	if(has(type))return Rt_; else return arma::Mat<fltp>{};
	// }

	// get target point coordinates
	arma::Mat<fltp> TargetPoints::get_target_coords(
		const arma::uword ft, const arma::uword lt) const{
		// check indexes
		assert(lt>=ft);	assert(lt<num_targets()); assert(ft<num_targets());
		assert(!Rt_.is_empty()); assert(Rt_.n_rows==3);

		// get targets
		return Rt_.cols(ft,lt);
	}

	// // get target point coordinates
	// arma::Mat<fltp> TargetPoints::get_target_coords(
	// 	const char type, const arma::uword ft, const arma::uword lt) const{
	// 	assert(lt>=ft);	
	// 	if(has(type))return Rt_.cols(ft,lt); else return arma::Mat<fltp>{};
	// }

	// // get target point coordinates
	// arma::Mat<fltp> TargetPoints::get_target_coords(
	// 	const arma::Row<arma::uword> &indices) const{
	// 	// get targets
	// 	return Rt_.cols(indices);
	// }

	// number of points stored
	arma::uword TargetPoints::num_targets() const{
		return num_targets_;
	}

	// set plane coordinates
	void TargetPoints::set_xy_plane(
		const fltp ellx, const fltp elly, const fltp xoff, const fltp yoff, 
		const fltp zoff, const arma::uword nx, const arma::uword ny){

		// check user input
		if(ellx<=0)rat_throw_line("length in x must be larger than zero");
		if(elly<=0)rat_throw_line("length in y must be larger than zero");
		if(nx<=1)rat_throw_line("number of coordinates in x must be larger than one");
		if(ny<=1)rat_throw_line("number of coordinates in y must be larger than one");
		
		// allocate coordinates
		arma::Mat<fltp> x(ny,nx); arma::Mat<fltp> y(ny,nx); 
		arma::Mat<fltp> z(ny,nx,arma::fill::zeros);

		// set coordinates to matrix
		x.each_row() = arma::linspace<arma::Row<fltp> >(-ellx/2,ellx/2,nx) + xoff;
		y.each_col() = arma::linspace<arma::Col<fltp> >(-elly/2,elly/2,ny) + yoff;
		z += zoff;
		
		// store coordinates
		arma::Mat<fltp> Rt(3,nx*ny);
		Rt.row(0) = arma::reshape(x,1,nx*ny);
		Rt.row(1) = arma::reshape(y,1,nx*ny);
		Rt.row(2) = arma::reshape(z,1,nx*ny);

		// set coordinates
		set_target_coords(Rt);
	}

	// set plane coordinates
	void TargetPoints::set_xz_plane(const fltp ellx, const fltp ellz, 
		const fltp xoff, const fltp yoff, const fltp zoff, 
		const arma::uword nx, const arma::uword nz){

		// check user input
		if(ellx<=0)rat_throw_line("length in x must be larger than zero");
		if(ellz<=0)rat_throw_line("length in z must be larger than zero");
		if(nx<=1)rat_throw_line("number of coordinates in x must be larger than one");
		if(nz<=1)rat_throw_line("number of coordinates in z must be larger than one");
		
		// allocate coordinates
		arma::Mat<fltp> x(nz,nx); arma::Mat<fltp> z(nz,nx); 
		arma::Mat<fltp> y(nz,nx,arma::fill::zeros);

		// set coordinates to matrix
		x.each_row() = arma::linspace<arma::Row<fltp> >(-ellx/2,ellx/2,nx) + xoff;
		y += yoff;
		z.each_col() = arma::linspace<arma::Col<fltp> >(-ellz/2,ellz/2,nz) + zoff;
		
		// store coordinates
		arma::Mat<fltp> Rt(3,nx*nz);
		Rt.row(0) = arma::reshape(x,1,nx*nz);
		Rt.row(1) = arma::reshape(y,1,nx*nz);
		Rt.row(2) = arma::reshape(z,1,nx*nz);

		// set coordinates
		set_target_coords(Rt);
	}

	// check for nans
	bool TargetPoints::has_nan() const{
		bool has_nan = false;
		for(auto it=M_.begin();it!=M_.end();it++)has_nan = has_nan || (*it).second.has_nan();
		return has_nan;
	}

	// sort function
	void TargetPoints::sort_targets(
		const arma::Row<arma::uword> &sort_idx){
		// check if targets set
		if(Rt_.is_empty())rat_throw_line("target coordinates are not set");

		// check if sort index is correct size
		assert(Rt_.n_cols==sort_idx.n_elem);

		// sort targets
		Rt_ = Rt_.cols(sort_idx);

		// if field is calculated
		for(auto it=M_.begin();it!=M_.end();it++)
			(*it).second = (*it).second.cols(sort_idx);
	}

	// unsort function
	void TargetPoints::unsort_targets(
		const arma::Row<arma::uword> &sort_idx){
		// check if targets set
		if(Rt_.is_empty())rat_throw_line("target coordinates are not set");

		// check if sort index is correct size
		assert(Rt_.n_cols==sort_idx.n_elem);

		// sort targets
		Rt_.cols(sort_idx) = Rt_;

		// if field is calculated
		for(auto it=M_.begin();it!=M_.end();it++)
			(*it).second.cols(sort_idx) = (*it).second;
	}

	// localpole to target setup function empty implementation
	void TargetPoints::setup_localpole_to_target(
		const arma::Mat<fltp> &/*dR*/, const arma::uword /*num_dim*/, const ShSettingsPr &/*stngs*/){

	}

	// localpole to target empty implementation
	void TargetPoints::localpole_to_target(
		const arma::Mat<std::complex<fltp> > &/*Lp*/, 
		const arma::Row<arma::uword> &/*first_target*/, 
		const arma::Row<arma::uword> &/*last_target*/, 
		const arma::uword /*num_dim*/, 
		const ShSettingsPr &/*stngs*/){
		rat_throw_line("the targets have no localpole to target implementation");
	}

}}