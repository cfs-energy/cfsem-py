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
#include "ilist.hh"

// code specific to Rat
namespace rat{namespace fmm{
	// constructor
	IList::IList(){

	}

	// constructor
	IList::IList(const ShSettingsPr &stngs){
		stngs_ = stngs;
	}


	// factory
	ShIListPr IList::create(){
		//return ShIListPr(new IList);
		return std::make_shared<IList>();
	}

	// factory
	ShIListPr IList::create(const ShSettingsPr &stngs){
		//return ShIListPr(new IList);
		return std::make_shared<IList>(stngs);
	}

	// settings
	void IList::set_settings(const ShSettingsPr &stngs){
		assert(stngs!=NULL);
		stngs_ = stngs;
	}

	// setup all interaction lists 
	void IList::setup(){
		// create armadillo timer
		arma::wall_clock timer;
		
		// set timer
		timer.tic();

		// make sure input settings are assigned
		assert(stngs_!=NULL);

		// setup small interaction list
		if(stngs_->get_large_ilist()){
			setup_large();
		}

		// setup large interaction list
		else{
			setup_small();
		}

		// get time
		last_setup_time_ = timer.toc();
		life_setup_time_ += last_setup_time_;
	}

	// setup all interaction lists 
	// using 3X3 box of direct interactions
	void IList::setup_small(){
		// make sure input settings are assigned
		assert(stngs_!=NULL);

		// allocate
		shift_type_.set_size(189,8);
		approx_nshift_.set_size(316,3);
		direct_nshift_.set_size(27,3);
		type_nshift_.set_size(8,3);
		
		// allocate counter
		arma::uword p;

		// definition of morton indexes
		//weight_ = {1,2,4}; // must contain a 1,2 and 4
		
		// make box-type shift matrix
		//std::cout<<"setting up shift list"<<std::endl;
		// this is how morton indices are defined
		// type_nshift_(0,0) = 0; type_nshift_(0,1) = 0; type_nshift_(0,2) = 0;
		// type_nshift_(1,0) = 0; type_nshift_(1,1) = 1; type_nshift_(1,2) = 0;
		// type_nshift_(2,0) = 1; type_nshift_(2,1) = 0; type_nshift_(2,2) = 0;
		// type_nshift_(3,0) = 1; type_nshift_(3,1) = 1; type_nshift_(3,2) = 0;
		// type_nshift_(4,0) = 0; type_nshift_(4,1) = 0; type_nshift_(4,2) = 1;
		// type_nshift_(5,0) = 0; type_nshift_(5,1) = 1; type_nshift_(5,2) = 1;
		// type_nshift_(6,0) = 1; type_nshift_(6,1) = 0; type_nshift_(6,2) = 1;
		// type_nshift_(7,0) = 1; type_nshift_(7,1) = 1; type_nshift_(7,2) = 1;

		// make box-type shift matrix from weights
		arma::Col<arma::uword>::fixed<8> box_type = 
			arma::regspace<arma::Col<arma::uword>::fixed<8> >(0,7);
		arma::Mat<arma::uword> A(8,3);
		A.col(0) = arma::floor(box_type/stngs_->get_morton_weights(0));
		A.col(1) = arma::floor(box_type/stngs_->get_morton_weights(1));
		A.col(2) = arma::floor(box_type/stngs_->get_morton_weights(2));
		type_nshift_ = arma::conv_to<arma::Mat<arma::sword> >::from(
			cmn::Extra::modulus(A,2));

		// make simple direct list
		// includes self
		//std::cout<<"setting up direct list"<<std::endl;
	 	p=0;
		for(arma::sword nx=-1;nx<=1;nx++){
		    for(arma::sword ny=-1;ny<=1;ny++){
		        for(arma::sword nz=-1;nz<=1;nz++){
		           // direct interaction list
		           direct_nshift_(p,0)=nx;
		           direct_nshift_(p,1)=ny;
		           direct_nshift_(p,2)=nz;

		           // increment counter
		           p++;
		        }
		    }
		}

		// sanity check
		assert(p==27);

		// make simple approx list
		// does not neighbours
		//std::cout<<"setting up approx list"<<std::endl;
	 	p=0;
		for(arma::sword nx=-3;nx<=3;nx++){
		    for(arma::sword ny=-3;ny<=3;ny++){
		        for(arma::sword nz=-3;nz<=3;nz++){
		        	if (nx<-1 || nx>1 || ny<-1 || ny>1 || nz<-1 || nz>1){
						// set values
						approx_nshift_(p,0) = nx;
						approx_nshift_(p,1) = ny;
						approx_nshift_(p,2) = nz;

						// increment counter
						p++;
			        }
		        }
		    }
		}

		// sanity check
		assert(p==316);

		// make index list for approximate interactions of each box type
		//std::cout<<"setting up approx index list"<<std::endl;
		for(arma::uword i=0;i<8;i++){
			// set counter to zero
			p = 0;

			// get coordinates
			const arma::Row<arma::sword> coord_min = -2-type_nshift_.row(i);
			const arma::Row<arma::sword> coord_max = 3-type_nshift_.row(i);

			// walk over coordinates
			for(arma::sword nx=coord_min(0);nx<=coord_max(0);nx++){
				for(arma::sword ny=coord_min(1);ny<=coord_max(1);ny++){
					for(arma::sword nz=coord_min(2);nz<=coord_max(2);nz++){
						if (nx<-1 || nx>1 || ny<-1 || ny>1 || nz<-1 || nz>1){
							
							// create required shift vector
							arma::Row<arma::sword>::fixed<3> nshift_req;
							nshift_req(0) = nx; nshift_req(1) = ny; nshift_req(2) = nz; 

							// find this vector in the approximate list
							arma::Mat<arma::sword> check = approx_nshift_.each_row()-nshift_req;
							shift_type_(p,i) = arma::as_scalar(arma::find(arma::all(check==0,1)));

							// sanity check
							assert(arma::all((approx_nshift_.row(shift_type_(p,i))-nshift_req)==0));

							// increment counter
							p++;
						}
					}
				}
			}

			// sanity check
			assert(p==189);
		}

		// check if all approximate entries are used
		assert(arma::all((arma::unique(shift_type_)-arma::regspace(0,315))==0));


	}

	// setup all interaction lists 
	// using 5X5 box of direct interactions
	void IList::setup_large(){
		// make sure input settings are assigned
		assert(stngs_!=NULL);

		// allocate
		shift_type_.set_size(651,8);
		direct_nshift_.set_size(93,3);
		type_nshift_.set_size(8,3);
		
		// allocate counter
		arma::uword p;

		// definition of morton indexes
		//weight_ = {1,2,4}; // must contain a 1,2 and 4
		
		// make box-type shift matrix
		//std::cout<<"setting up shift list"<<std::endl;
		// this is how morton indices are defined
		// type_nshift_(0,0) = 0; type_nshift_(0,1) = 0; type_nshift_(0,2) = 0;
		// type_nshift_(1,0) = 0; type_nshift_(1,1) = 1; type_nshift_(1,2) = 0;
		// type_nshift_(2,0) = 1; type_nshift_(2,1) = 0; type_nshift_(2,2) = 0;
		// type_nshift_(3,0) = 1; type_nshift_(3,1) = 1; type_nshift_(3,2) = 0;
		// type_nshift_(4,0) = 0; type_nshift_(4,1) = 0; type_nshift_(4,2) = 1;
		// type_nshift_(5,0) = 0; type_nshift_(5,1) = 1; type_nshift_(5,2) = 1;
		// type_nshift_(6,0) = 1; type_nshift_(6,1) = 0; type_nshift_(6,2) = 1;
		// type_nshift_(7,0) = 1; type_nshift_(7,1) = 1; type_nshift_(7,2) = 1;

		// make box-type shift matrix from weights
		arma::Col<arma::uword>::fixed<8> box_type = 
			arma::regspace<arma::Col<arma::uword>::fixed<8> >(0,7);
		arma::Mat<arma::uword> A(8,3);
		A.col(0) = arma::floor(box_type/stngs_->get_morton_weights(0));
		A.col(1) = arma::floor(box_type/stngs_->get_morton_weights(1));
		A.col(2) = arma::floor(box_type/stngs_->get_morton_weights(2));
		type_nshift_ = arma::conv_to<arma::Mat<arma::sword> >::from(
			cmn::Extra::modulus(A,2));

		// make simple direct list
		// includes self
		//std::cout<<"setting up direct list"<<std::endl;
	 	p=0;
		for(arma::sword nx=-2;nx<=2;nx++){
		    for(arma::sword ny=-2;ny<=2;ny++){
		        for(arma::sword nz=-2;nz<=2;nz++){
		        	if(std::abs(nx)+std::abs(ny)+std::abs(nz)<5){
						// direct interaction list
						direct_nshift_(p,0)=nx;
						direct_nshift_(p,1)=ny;
						direct_nshift_(p,2)=nz;

						// increment counter
						p++;
		       		}
		        }
		    }
		}
		
		// sanity check
		assert(p==93);

		// allocate
		arma::Mat<arma::sword> approx_nshift_exp(8*651,3);

		// make simple approx list
		// does not neighbours
		//std::cout<<"setting up approx list"<<std::endl;
	 	
	 	// create list
	 	for(arma::uword i=0;i<8;i++){
	 		// set p to zero
	 		p=0;

	 		// walk over list
			for(arma::sword x=-2;x<=2;x++){
				for(arma::sword y=-2;y<=2;y++){
					for(arma::sword z=-2;z<=2;z++){
						if(std::abs(x)+std::abs(y)+std::abs(z)<5){
							for(arma::sword xt=0;xt<2;xt++){
								for(arma::sword yt=0;yt<2;yt++){
									for(arma::sword zt=0;zt<2;zt++){
										// calculate relative position
										arma::sword xr = x*2 + xt - type_nshift_(i,0);
										arma::sword yr = y*2 + yt - type_nshift_(i,1);
										arma::sword zr = z*2 + zt - type_nshift_(i,2);

										// exclude direct
										if(arma::all(
											xr!=direct_nshift_.col(0) || 
											yr!=direct_nshift_.col(1) || 
											zr!=direct_nshift_.col(2))){

											// set values
											approx_nshift_exp(p+i*651,0) = xr;
											approx_nshift_exp(p+i*651,1) = yr;
											approx_nshift_exp(p+i*651,2) = zr;

											// increment counter
											p++;
										}
									}
								}
							}
						}
					}
				}
			}

			// sanity check
			assert(p==651);
		}

		// sort list
		for(arma::uword i=0;i<3;i++){
			arma::Col<arma::uword> idx = arma::stable_sort_index(approx_nshift_exp.col(i));
			approx_nshift_exp = approx_nshift_exp.rows(idx);
		}

		// remove duplicates
		arma::Col<arma::uword> mark(approx_nshift_exp.n_rows,arma::fill::zeros);
		for(arma::uword i=0;i<approx_nshift_exp.n_rows-1;i++){
			if(arma::all(approx_nshift_exp.row(i)==approx_nshift_exp.row(i+1))){
				mark(i+1)=true;
			}
		}

		// get final list
		approx_nshift_ = approx_nshift_exp.rows(arma::find(mark==0));

		// check length
		assert(approx_nshift_.n_rows==982);

		// now get indexes into array
		// create list
	 	for(arma::uword i=0;i<8;i++){
	 		// set p to zero
	 		p=0;

	 		// walk over list
			for(arma::sword x=-2;x<=2;x++){
				for(arma::sword y=-2;y<=2;y++){
					for(arma::sword z=-2;z<=2;z++){
						if(std::abs(x)+std::abs(y)+std::abs(z)<5){
							for(arma::sword xt=0;xt<2;xt++){
								for(arma::sword yt=0;yt<2;yt++){
									for(arma::sword zt=0;zt<2;zt++){
										// calculate relative position
										arma::sword xr = x*2 + xt - type_nshift_(i,0);
										arma::sword yr = y*2 + yt - type_nshift_(i,1);
										arma::sword zr = z*2 + zt - type_nshift_(i,2);

										// exclude direct
										if(arma::all(
											xr!=direct_nshift_.col(0) || 
											yr!=direct_nshift_.col(1) || 
											zr!=direct_nshift_.col(2))){

											// create required shift vector
											arma::Row<arma::sword>::fixed<3> nshift_req;
											nshift_req(0) = xr; nshift_req(1) = yr; nshift_req(2) = zr; 

											// find this vector in the approximate list
											arma::Mat<arma::sword> check = approx_nshift_.each_row()-nshift_req;
											shift_type_(p,i) = arma::as_scalar(arma::find(arma::all(check==0,1)));

											// sanity check
											assert(arma::all((approx_nshift_.row(shift_type_(p,i))-nshift_req)==0));

											// increment counter
											p++;
										}
									}
								}
							}
						}
					}
				}
			}

			// sanity check
			assert(p==651);
		}
		
		// check if all approximate entries are used
		assert(arma::all((arma::unique(shift_type_)-arma::regspace(0,981))==0));
	}


	// get direct nshift
	arma::Mat<arma::sword> IList::get_direct_nshift() const{
		return direct_nshift_;
	}

	arma::Mat<arma::uword> IList::get_direct_shift_type() const{
		return arma::regspace<arma::Col<
			arma::uword> >(0,direct_nshift_.n_rows);
	}

	// get approximate nshift for all types
	arma::Mat<arma::sword> IList::get_approx_nshift() const{
		return approx_nshift_;
	}

	// get shift type for all boxes stored columnwise in matrix
	arma::Mat<arma::uword> IList::get_approx_shift_type() const{
		return shift_type_;
	}

	// get shift type for one specific box
	arma::Mat<arma::uword> IList::get_approx_shift_type(
		const arma::uword type) const{
		return shift_type_.col(type);
	}

	// get approximate nshift
	arma::Mat<arma::sword> IList::get_approx_nshift(const arma::uword type) const{
		return approx_nshift_.rows(shift_type_.col(type));
	}

	// get type nshift unsigned
	arma::Mat<arma::uword> IList::get_type_nshift() const{
		return arma::conv_to<arma::Mat<arma::uword> >::from(type_nshift_);
	}

	// get type nshift unsigned
	// arma::Col<arma::uword>::fixed<3> IList::get_weight() const{
	// 	return weight_;
	// }

	// number of shifts
	arma::uword IList::get_num_type() const{
		return shift_type_.n_rows;
	}

	// number of approximates
	arma::uword IList::get_num_approx() const{
		return approx_nshift_.n_rows;
	}

	// number of direct
	arma::uword IList::get_num_direct() const{
		return direct_nshift_.n_rows;
	}

	// display stored information
	void IList::display(const cmn::ShLogPr &lg) const{
		// header
		lg->msg(2,"%sInteraction List(s)%s\n",KBLU,KNRM);

		// setup small interaction list
		if(stngs_->get_large_ilist()){
			lg->msg("using large interaction list (accurate)\n");
		}

		// setup large interaction list
		else{
			lg->msg("using normal interaction list (fast)\n");
		}

		// quantities
		lg->msg("number of direct: %s%llu%s\n",KYEL,get_num_direct(),KNRM);
		lg->msg("number of approximate per box: %s%llu%s\n",KYEL,get_num_type(),KNRM);
		lg->msg("number of all possible approximates: %s%llu%s\n",KYEL,get_num_approx(),KNRM);

		// display time
		lg->msg("setup time: %s%.2f / %.2f [ms]%s\n",KYEL,1e3*last_setup_time_,1e3*life_setup_time_,KNRM);

		// footer
		lg->msg(-2,"\n");
	}

}}
