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

// check if cuda available
#ifdef ENABLE_CUDA_KERNELS

// include header file
#include "gpucurrentsources.hh"

// code specific to Rat
namespace rat{namespace fmm{
	
	// constructor
	GpuCurrentSources::GpuCurrentSources(){

	}

	// constructor with input
	GpuCurrentSources::GpuCurrentSources(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &dRs, 
		const arma::Row<fltp> &Is,
		const arma::Row<fltp> &epss,
		const arma::Mat<fltp> &Rt,
		const bool calc_A,
		const bool calc_H){
		
		// check matrix size
		if(Rs.n_cols!=dRs.n_cols)rat_throw_line("coordinate and direction have different number of columns");
		if(Rs.n_cols!=Is.n_cols)rat_throw_line("coordinate and current have different number of columns");
		if(Rs.n_cols!=epss.n_cols)rat_throw_line("coordinate and softening have different number of columns");

		// set sources 
		set_source_coords(Rs); 
		set_direction(dRs);
		set_currents(Is); 
		set_softening(epss);
		
		// set targets
		set_target_coords(Rt);

		// set field type
		calc_A_ = calc_A;
		calc_H_ = calc_H;
	}


	// factory
	ShGpuCurrentSourcesPr GpuCurrentSources::create(){
		return std::make_shared<GpuCurrentSources>();
	}

	// factory with input
	ShGpuCurrentSourcesPr GpuCurrentSources::create(
		const arma::Mat<fltp> &Rs, 
		const arma::Mat<fltp> &dRs, 
		const arma::Row<fltp> &Is,
		const arma::Row<fltp> &epss,
		const arma::Mat<fltp> &Rt,
		const bool calc_A,
		const bool calc_H){

		//return ShIListPr(new IList);
		return std::make_shared<GpuCurrentSources>(Rs,dRs,Is,epss,Rt,calc_A,calc_H);
	}

	// destructor
	GpuCurrentSources::~GpuCurrentSources(){
		if(Rs_!=NULL)fmm::GpuKernels::free_cuda_managed(Rs_);
		if(dRs_!=NULL)fmm::GpuKernels::free_cuda_managed(dRs_);
		if(dRmp_!=NULL)fmm::GpuKernels::free_cuda_managed(dRmp_);
		if(Rt_!=NULL)fmm::GpuKernels::free_cuda_managed(Rt_);
		if(At_!=NULL)fmm::GpuKernels::free_cuda_managed(At_);
		if(Ht_!=NULL)fmm::GpuKernels::free_cuda_managed(Ht_);
	}

	// set field type and number of dimensions
	void GpuCurrentSources::set_field_type(
		const std::string &field_type, 
		const arma::Row<arma::uword> &num_dim){

		// check if all values set
		if(field_type.length()!=num_dim.n_elem)rat_throw_line("field type string must equal length of dimensions vector");

		// reset
		calc_A_ = false; calc_H_ = false;

		// walk over field types
		arma::uword i=0;
		for(auto it = field_type.begin();it!=field_type.end();it++,i++){
			if((*it)=='A')calc_A_=true; else if((*it)=='H' || (*it)=='B')calc_H_=true;
			else rat_throw_line("gpu current sources do not have this field type");
		}
	}

	// settings
	void GpuCurrentSources::set_field_type(
		const char field_type, 
		const arma::uword /*num_dim*/){

		// reset
		calc_A_ = false; calc_H_ = false;

		if(field_type=='A')calc_A_=true; 
		else if(field_type=='H' || field_type=='B')calc_H_=true;
		else rat_throw_line("gpu current sources do not have this field type");	
	}

	// get number of dimensions
	arma::uword GpuCurrentSources::get_num_dim() const{
		return num_dim_;
	}

	// get field function
	arma::Mat<fltp> GpuCurrentSources::get_field(const char type) const{
		
		// create matrices
		arma::Mat<cufltp> MAt(At_,4,num_targets_,false,true);
		arma::Mat<cufltp> MHt(Ht_,4,num_targets_,false,true);

		// get required field
		arma::Mat<fltp> M;
		if(type=='A')M = arma::conv_to<arma::Mat<fltp> >::from(MAt);
		else if(type=='H')M = arma::conv_to<arma::Mat<fltp> >::from(MHt);
		else if(type=='B')M = arma::conv_to<arma::Mat<fltp> >::from(MHt)*arma::Datum<fltp>::mu_0;
		else rat_throw_line("field not recognized");

		// remove last row and return
		return M.rows(0,2);
	}

	// check what field types requested
	bool GpuCurrentSources::has(const char type) const{
		if(type=='A')return calc_A_;
		else if(type=='H' || type=='B')return calc_H_; 
		else return false;
	}

	// set coordinate vectors
	void GpuCurrentSources::set_source_coords(const arma::Mat<fltp> &Rs){
		// check user input
		if(Rs.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(!Rs.is_finite())rat_throw_line("coordinate matrix must be finite");

		// free
		if(Rs_!=NULL)fmm::GpuKernels::free_cuda_managed(Rs_);

		// number of sources
		num_sources_ = Rs.n_cols;

		// create cuda managed memory
		// this memory is accessable both from CPU and GPU
		fmm::GpuKernels::create_cuda_managed(reinterpret_cast<void**>(&Rs_),4*num_sources_*sizeof(cufltp));

		// wrap to armadillo vectors
		// these vectors do not destroy the 
		// cuda memory when they are destroyed
		arma::Mat<cufltp> MRs(Rs_,4,num_sources_, false, true);

		// fill current with zeros and assign values
		MRs.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(Rs);
		MRs.row(3).zeros(); 
	}


	// set coordinate vectors
	void GpuCurrentSources::set_target_coords(const arma::Mat<fltp> &Rt){
		// check user input
		if(Rt.n_rows!=3)rat_throw_line("coordinate matrix must have three rows");
		if(!Rt.is_finite())rat_throw_line("coordinate matrix must be finite");

		// free
		if(Rt_!=NULL)fmm::GpuKernels::free_cuda_managed(Rt_);

		// number of sources
		num_targets_ = Rt.n_cols;

		// create cuda managed memory
		// this memory is accessable both from CPU and GPU
		fmm::GpuKernels::create_cuda_managed(reinterpret_cast<void**>(&Rt_),4*num_targets_*sizeof(cufltp));

		// wrap to armadillo vectors
		// these vectors do not destroy the 
		// cuda memory when they are destroyed
		arma::Mat<cufltp> MRt(Rt_,4,num_targets_, false, true);

		// fill current with zeros and assign values
		MRt.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(Rt);
		MRt.row(3).zeros();
	}


	// set direction vector
	void GpuCurrentSources::set_direction(const arma::Mat<fltp> &dRs){
		// check user input
		if(dRs.n_rows!=3)rat_throw_line("direction matrix must have three rows");
		if(!dRs.is_finite())rat_throw_line("direction matrix must be finite");
		if(dRs.n_cols!=num_sources_)rat_throw_line("number of direction vectors does not match number of sources");

		// free
		if(dRs_!=NULL)fmm::GpuKernels::free_cuda_managed(dRs_);

		// create cuda managed memory
		// this memory is accessable both from CPU and GPU
		fmm::GpuKernels::create_cuda_managed(reinterpret_cast<void**>(&dRs_),4*num_sources_*sizeof(cufltp));

		// wrap to armadillo vectors
		// these vectors do not destroy the 
		// cuda memory when they are destroyed
		arma::Mat<cufltp> MdRs(dRs_,4,num_sources_, false, true);

		// fill softening with zeros and assign values
		MdRs.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(dRs);
		MdRs.row(3).zeros();
	}


	// get typical element size (to limit grid)
	fltp GpuCurrentSources::element_size() const{
		if(use_van_Lanen_){
			return arma::max(cmn::Extra::vec_norm(arma::conv_to<arma::Mat<fltp> >::from(arma::Mat<cufltp>(dRs_,4,num_sources_,false,true).rows(0,2))));
		}else{
			return 0;
		}
	}

	// set currents
	void GpuCurrentSources::set_currents(const arma::Row<fltp> &Is){
		// check user input
		if(!Is.is_finite())rat_throw_line("current vector must be finite");
		if(Is.n_elem!=num_sources_)rat_throw_line("number of currents must match number of sources");
			
		// create armadillo matrix
		arma::Mat<cufltp> MdRs(dRs_,4,num_sources_,false,true);

		// set currents to Rs
		MdRs.row(3) = arma::conv_to<arma::Row<cufltp> >::from(Is);
	}

	// setting the softening factor
	void GpuCurrentSources::set_softening(const arma::Row<fltp> &epss){
		// check user input
		if(!epss.is_finite())rat_throw_line("softening vector must be finite");
		if(epss.n_elem!=num_sources_)rat_throw_line("number of softening factors must match number of sources");
		
		// create armadillo matrix
		arma::Mat<cufltp> MRs(Rs_,4,num_sources_,false,true);

		// set currents to Rs
		MRs.row(3) = arma::conv_to<arma::Row<cufltp> >::from(epss);
	}

	// sorting function
	void GpuCurrentSources::sort_sources(const arma::Row<arma::uword> &sort_idx){
		// cast to armadillo
		arma::Mat<cufltp> MRs(Rs_,4,num_sources_, false, true);
		arma::Mat<cufltp> MdRs(dRs_,4,num_sources_, false, true);

		// sort sources
		MRs = MRs.cols(sort_idx);
		MdRs = MdRs.cols(sort_idx);
	}

	// sorting function
	void GpuCurrentSources::sort_targets(const arma::Row<arma::uword> &sort_idx){
		// cast to armadillo
		arma::Mat<cufltp> MRt(Rt_,4,num_targets_, false, true);
		MRt = MRt.cols(sort_idx);

		// sort targets
		if(At_!=NULL){
			if(calc_A_){
				arma::Mat<cufltp> MAt(At_,4,num_targets_, false, true);
				MAt = MAt.cols(sort_idx);
			}
		}

		if(Ht_!=NULL){
			if(calc_H_){
				arma::Mat<cufltp> MHt(Ht_,4,num_targets_, false, true);
				MHt = MHt.cols(sort_idx);
			}
		}
	}

	// unsorting
	void GpuCurrentSources::unsort_sources(
		const arma::Row<arma::uword> &sort_idx){
		// cast to armadillo
		arma::Mat<cufltp> MRs(Rs_,4,num_sources_, false, true);
		arma::Mat<cufltp> MdRs(dRs_,4,num_sources_, false, true);

		// sort sources
		MRs.cols(sort_idx) = MRs;
		MdRs.cols(sort_idx) = MdRs;
	}

	void GpuCurrentSources::unsort_targets(
		const arma::Row<arma::uword> &sort_idx){
		// cast to armadillo
		arma::Mat<cufltp> MRt(Rt_,4,num_targets_, false, true);
		MRt.cols(sort_idx) = MRt;

		// sort targets
		if(At_!=NULL){
			if(calc_A_){
				arma::Mat<cufltp> MAt(At_,4,num_targets_, false, true);
				MAt.cols(sort_idx) = MAt;
			}
		}
		
		if(Ht_!=NULL){
			if(calc_H_){
				arma::Mat<cufltp> MHt(Ht_,4,num_targets_, false, true);
				MHt.cols(sort_idx) = MHt;
			}
		}
	}

	// setting of calculated field
	void GpuCurrentSources::add_field(
		const char type, 
		const arma::Mat<fltp> &Madd, 
		const bool with_lock){
		assert(lock_!=NULL);
		assert(Madd.n_cols==num_targets_);
		if(with_lock)lock_->lock();
		if(type=='A'){
			arma::Mat<cufltp> MAt(At_,4,num_targets_,false,true);
			MAt.rows(0,2) += arma::conv_to<arma::Mat<cufltp> >::from(Madd);
		}else if(type=='H'){
			arma::Mat<cufltp> MHt(Ht_,4,num_targets_,false,true);
			MHt.rows(0,2) += arma::conv_to<arma::Mat<cufltp> >::from(Madd);
		}
		if(with_lock)lock_->unlock();
	}
	
	// setting of calculated field for specific targets
	void GpuCurrentSources::add_field(
		const char type, 
		const arma::Mat<fltp> &Madd,
		const arma::Row<arma::uword> &ft, 
		const arma::Row<arma::uword> &lt, 
		const arma::Row<arma::uword> &ti, 
		const bool with_lock){
		if(with_lock)lock_->lock();
		if(type=='A'){
			arma::Mat<cufltp> MAt(At_,4,num_targets_,false,true);
			for(arma::uword i=0;i<ft.n_elem;i++){
				MAt.rows(0,2).cols(ft(i),lt(i)) += arma::conv_to<arma::Mat<cufltp> >::from(Madd.cols(ti(i),ti(i+1)-1));
			}
		}else if(type=='H'){
			arma::Mat<cufltp> MHt(Ht_,4,num_targets_,false,true);
			for(arma::uword i=0;i<ft.n_elem;i++){
				MHt.rows(0,2).cols(ft(i),lt(i)) += arma::conv_to<arma::Mat<cufltp> >::from(Madd.cols(ti(i),ti(i+1)-1));
			}
		}
		if(with_lock)lock_->unlock();
	}




	// count number of sources stored
	arma::uword GpuCurrentSources::num_sources() const{
		return num_sources_;
	}

	// get number of targets
	arma::uword GpuCurrentSources::num_targets() const{
		return num_targets_;
	}

	// method for getting all coordinates
	arma::Mat<fltp> GpuCurrentSources::get_source_coords() const{
		const arma::Mat<fltp> Rs = arma::conv_to<arma::Mat<fltp> >::from(
			arma::Mat<cufltp>(Rs_,4,num_sources_,false,true).rows(0,2));
		return Rs;
	}

	// method for getting all coordinates
	arma::Mat<fltp> GpuCurrentSources::get_target_coords() const{
		const arma::Mat<fltp> Rt = arma::conv_to<arma::Mat<fltp> >::from(
			arma::Mat<cufltp>(Rt_,4,num_targets_,false,true).rows(0,2));
		return Rt;
	}


	// get specified target coords
	arma::Mat<fltp> GpuCurrentSources::get_target_coords(const arma::uword ft, const arma::uword lt) const{
		return arma::conv_to<arma::Mat<fltp> >::from(
			arma::Mat<cufltp>(Rt_,4,num_targets_,false,true).cols(ft,lt).rows(0,2));
	}

	// method for getting all coordinates
	arma::Mat<fltp> GpuCurrentSources::get_source_direction() const{
		// return coordinates
		const arma::Mat<fltp> dRs = arma::conv_to<arma::Mat<fltp> >::from(
			arma::Mat<cufltp>(dRs_,4,num_sources_,false,true).rows(0,2));
		return dRs;
	}


	// set van Lanen kernel
	void GpuCurrentSources::set_van_Lanen(const bool use_van_Lanen){
		use_van_Lanen_ = use_van_Lanen;
	}

	// get van Lanen
	bool GpuCurrentSources::get_van_Lanen() const{
		return use_van_Lanen_;
	}

	// allocate field
	void GpuCurrentSources::allocate(){
		// allocate
		if(calc_A_){
			if(At_!=NULL)fmm::GpuKernels::free_cuda_managed(At_);
			fmm::GpuKernels::create_cuda_managed(reinterpret_cast<void**>(&At_),4*num_targets_*sizeof(cufltp));
			arma::Mat<cufltp> MAt(At_,4,num_targets_, false, true);
			MAt.zeros();
		}

		if(calc_H_){
			if(Ht_!=NULL)fmm::GpuKernels::free_cuda_managed(Ht_);
			fmm::GpuKernels::create_cuda_managed(reinterpret_cast<void**>(&Ht_),4*num_targets_*sizeof(cufltp));
			arma::Mat<cufltp> MHt(Ht_,4,num_targets_, false, true);
			MHt.zeros();
		}

		// create a lock
		if(lock_==NULL)lock_ = std::unique_ptr<std::mutex>(new std::mutex);
	}

	// setup source to multipole matrices
	void GpuCurrentSources::setup_source_to_multipole(
		const arma::Mat<fltp> &dR, 
		const ShSettingsPr &/*stngs*/){

		// check relative position
		if(dR.n_cols!=num_sources_)rat_throw_line("relative position vector does not match number of sources");

		// free
		if(dRmp_!=NULL)fmm::GpuKernels::free_cuda_managed(dRmp_);
		
		// create cuda managed memory
		// this memory is accessable both from CPU and GPU
		fmm::GpuKernels::create_cuda_managed(reinterpret_cast<void**>(&dRmp_),4*num_sources_*sizeof(cufltp));

		// wrap to armadillo vectors
		// these vectors do not destroy the 
		// cuda memory when they are destroyed
		arma::Mat<cufltp> MdRmp(dRmp_,4,num_sources_, false, true);
		arma::Mat<cufltp> MRs(Rs_,4,num_sources_, false, true);

		// fill current with zeros and assign values
		MdRmp.rows(0,2) = arma::conv_to<arma::Mat<cufltp> >::from(dR);
		MdRmp.row(3) = MRs.row(3); // copy eps
	}

	// get multipole contribution of the sources with indices
	// the contributions of the sources are already summed
	void GpuCurrentSources::source_to_multipole(
		arma::Mat<std::complex<fltp> > &Mp,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source,
		const ShSettingsPr &stngs) const{

		// check input
		assert(first_source.n_elem==last_source.n_elem);
		assert(!Mp.is_empty());

		// check input source indexing
		assert(first_source(0)==0);
		assert(arma::all(arma::abs(
			first_source.tail_cols(first_source.n_elem-1)-
			last_source.head_cols(last_source.n_elem-1)-1)==0));
		assert(last_source(last_source.n_elem-1)==num_sources_-1);

		// get number of expansions
		const int num_exp = stngs->get_num_exp();

		// get counters
		const arma::uword num_nodes = first_source.n_elem;

		// allocate output multipoles
		arma::Mat<std::complex<cufltp> > Mp_half(Extra::hfpolesize(num_exp), Mp.n_cols);

		// ofload to GPU kernel
		const bool is_managed = true; // using cuda managed memory
		GpuKernels::so2mp_kernel(
			Mp_half.memptr(), num_exp, dRmp_, dRs_, num_sources_,
			first_source.memptr(), last_source.memptr(), num_nodes,
			stngs->get_gpu_devices(), is_managed);

		// convert back and create m<0 half from complex conjugate of m>0 half
		for(int n=0;n<=num_exp;n++){
			for(int m=0;m<=n;m++){
				// check if cuda has different precision from rat
				#ifdef RAT_CUDA_CONVERSION_NEEDED
					Mp.row(Extra::nm2fidx(n,m)) = 
						arma::conv_to<arma::Row<std::complex<fltp> > >::from(
						Mp_half.row(Extra::hfnm2fidx(n,m)));		
				#else
					Mp.row(Extra::nm2fidx(n,m)) = Mp_half.row(Extra::hfnm2fidx(n,m));
				#endif
				if(m!=0)Mp.row(Extra::nm2fidx(n,-m)) = arma::conj(Mp.row(Extra::nm2fidx(n,m)));
			}
		}
	}

	


	// direct calculation of vector potential or magnetic 
	// field for all sources at all target points
	void GpuCurrentSources::calc_direct(const ShTargetsPr &tar, const ShSettingsPr &stngs, const cmn::ShLogPr &/*lg*/)const{
		// get targets
		const ShGpuCurrentSourcesPr gpu_tar = std::dynamic_pointer_cast<GpuCurrentSources>(tar);
		if(gpu_tar==NULL)rat_throw_line("gpu sources must also be used as targets");

		// get gpu devices
		std::set<int> gpu_devices = stngs->get_gpu_devices();
		if(gpu_devices.empty())rat_throw_line("No GPU devices set for calculation");

		// ofload to GPU kernel
		const bool is_managed = true; // using cuda managed memory
		GpuKernels::direct_kernel(
			At_, Ht_, Rt_, static_cast<long unsigned int>(num_targets_), 
			Rs_, dRs_, static_cast<long unsigned int>(num_sources_), 
			calc_A_, calc_H_, use_van_Lanen_, 
			gpu_devices, is_managed);
	}


	// source to target kernel
	void GpuCurrentSources::source_to_target(
		const ShTargetsPr &tar, 
		const arma::Col<arma::uword> &target_list, 
		const arma::field<arma::Col<arma::uword> > &source_list,
		const arma::Row<arma::uword> &first_source, 
		const arma::Row<arma::uword> &last_source, 
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const ShSettingsPr &stngs,
		const cmn::ShLogPr &/*lg*/)const{

		// do not run gpu if there is no source-target interactions
		if(source_list.is_empty() || target_list.is_empty())return;

		// get targets
		const ShGpuCurrentSourcesPr gpu_tar = std::dynamic_pointer_cast<GpuCurrentSources>(tar);
		if(gpu_tar==NULL)rat_throw_line("gpu sources must also be used as targets");

		// get counters
		const arma::uword num_target_nodes = first_target.n_elem;
		const arma::uword num_source_nodes = first_source.n_elem;
		const arma::uword num_target_list = target_list.n_elem;
		assert(source_list.n_elem==num_target_list);
		assert(last_source.n_elem==num_source_nodes);
		assert(last_target.n_elem==num_target_nodes);

		// create indexes for source and target lists
		arma::Col<arma::uword> source_list_idx(num_target_list + 1);
		source_list_idx(0) = 0;
		for(arma::uword i=0;i<num_target_list;i++)
			source_list_idx(i+1) = source_list(i).n_elem;

		// accumulate sizes to create indexes
		source_list_idx = arma::cumsum(source_list_idx);

		// nnumber of source nodes
		const arma::uword num_source_list = source_list_idx(num_target_list);

		// combine source list
		arma::Col<arma::uword> source_list_combined(num_source_list);

		// fill source list
		for(arma::uword i=0;i<num_target_list;i++){
			source_list_combined.rows(source_list_idx(i),source_list_idx(i+1)-1) = source_list(i); 
		}

		// convert indices to 32 bit integers
		const arma::Row<long unsigned int> target_list_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(target_list);
		const arma::Row<long unsigned int> source_list_idx_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(source_list_idx);
		const arma::Row<long unsigned int> source_list_combined_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(source_list_combined.t());
		const arma::Row<long unsigned int> first_source_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(first_source);
		const arma::Row<long unsigned int> last_source_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(last_source);
		const arma::Row<long unsigned int> first_target_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(first_target);
		const arma::Row<long unsigned int> last_target_single = 
			arma::conv_to<arma::Row<long unsigned int> >::from(last_target);

		// check sizes
		assert(first_target_single.n_elem==num_target_nodes); 
		assert(last_target_single.n_elem==num_target_nodes);
		assert(first_source_single.n_elem==num_source_nodes); 
		assert(last_source_single.n_elem==num_source_nodes);
		assert(target_list_single.n_elem==num_target_list); 
		assert(source_list_idx_single.n_elem==num_target_list+1);
		assert(source_list_combined_single.n_elem==num_source_list);

		// get gpu devices
		std::set<int> gpu_devices = stngs->get_gpu_devices();
		if(gpu_devices.empty())rat_throw_line("No GPU devices set for calculation");

		// ofload to GPU kernel
		const bool is_managed = true; // using cuda managed memory
		GpuKernels::so2ta_kernel(
			At_, Ht_, Rt_,  static_cast<long unsigned int>(num_targets_), 
			Rs_, dRs_, static_cast<long unsigned int>(num_sources_),
			first_target_single.memptr(), last_target_single.memptr(), static_cast<long unsigned int>(num_target_nodes),
			first_source_single.memptr(), last_source_single.memptr(), static_cast<long unsigned int>(num_source_nodes),
			target_list_single.memptr(), source_list_idx_single.memptr(), static_cast<long unsigned int>(num_target_list),
			source_list_combined_single.memptr(), static_cast<long unsigned int>(num_source_list), 
			calc_A_, calc_H_, use_van_Lanen_, gpu_devices, is_managed);
	}


	// setup localpole to target matrix
	void GpuCurrentSources::setup_localpole_to_target(
		const arma::Mat<fltp> &dR, 
		const arma::uword /*num_dim*/,
		const ShSettingsPr &stngs){

		// check input
		assert(dR.is_finite());

		// memory efficient implementation (default)
		if(stngs->get_memory_efficient_l2t()){
			dRlp_ = dR;
		}

		// maximize computation speed
		else{
			// get number of expansions
			const int num_exp = stngs->get_num_exp();

			// matrix for vector potential
			if(has('A')){
				// calculate matrix for all target points
				M_A_.set_num_exp(num_exp);
				M_A_.calc_matrix(-dR);
			}

			// matrix for magnetic field
			if(has('H') || has('B')){
				// calculate matrix for all target points
				M_H_.set_num_exp(num_exp);
				M_H_.calc_matrix(-dR);
			}
		}
	}

	// add field contribution of supplied localpole to targets with indices
	void GpuCurrentSources::localpole_to_target(
		const arma::Mat<std::complex<fltp> > &Lp,
		const arma::Row<arma::uword> &first_target, 
		const arma::Row<arma::uword> &last_target,
		const arma::uword num_dim, 
		const ShSettingsPr &stngs){

		// memory efficient implementation (default)
		// note that this method is called many times in parallel
		// do not re-use the class property of M_A and M_H
		if(stngs->get_memory_efficient_l2t()){
			// check if dR was set
			assert(!dRlp_.is_empty());

			// number of expansions
			const int num_exp = stngs->get_num_exp();

			// vector potential
			if(has('A')){
				// create temporary storage for vector potential
				arma::Mat<fltp> A(num_dim,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate matrix for these target points
					StMat_Lp2Ta M_A;
					M_A.set_num_exp(num_exp);
					M_A.calc_matrix(-dRlp_.cols(first_target(i),last_target(i)));

					// calculate and add vector potential
					A.cols(first_target(i),last_target(i)) += 
						RAT_CONST(1e-7)*M_A.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1));
				});

				// add to self
				add_field('A',A,true);
			}

			// magnetic field
			if(has('H') || has('B')){
				// create temporary storage for magnetic field
				arma::Mat<fltp> H(num_dim,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate matrix for these target points
					StMat_Lp2Ta_Curl M_H;
					M_H.set_num_exp(num_exp);
					M_H.calc_matrix(-dRlp_.cols(first_target(i),last_target(i)));

					// calculate and add vector potential
					H.cols(first_target(i),last_target(i)) += 
						M_H.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1))/(4*arma::Datum<fltp>::pi);
				});

				// add to self
				if(has('H'))add_field('H',H,true);
				if(has('B'))add_field('B',arma::Datum<fltp>::mu_0*H,true);
			}
		}

		// maximize computation speed using pre-calculated matrix
		else{
			// vector potential
			if(has('A')){
				// check if localpole to target matrix was set
				assert(!M_A_.is_empty());

				// create temporary storage for vector potential
				arma::Mat<fltp> A(num_dim,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate and add vector potential
					A.cols(first_target(i),last_target(i)) += 
						RAT_CONST(1e-7)*M_A_.apply(Lp.cols(i*num_dim, (i+1)*num_dim-1),
						first_target(i),last_target(i));
				});

				// add to self
				add_field('A',A,true);
			}

			// magnetic field
			if(has('H') || has('B')){
				// check if localpole to target matrix was set
				assert(!M_H_.is_empty());

				// create temporary storage for magnetic field
				arma::Mat<fltp> H(num_dim,num_targets_,arma::fill::zeros);

				// walk over target nodes
				cmn::parfor(0,first_target.n_elem,stngs->get_parallel_l2t(),[&](arma::uword i, int){
					// calculate and add vector potential
					H.cols(first_target(i),last_target(i)) += M_H_.apply(
						Lp.cols(i*num_dim, (i+1)*num_dim-1),first_target(i),
						last_target(i))/(4*arma::Datum<fltp>::pi);
				});

				// add to self
				if(has('H'))add_field('H',H,true);
				if(has('B'))add_field('B',arma::Datum<fltp>::mu_0*H,true);
			}
		}
	}
}}

#endif