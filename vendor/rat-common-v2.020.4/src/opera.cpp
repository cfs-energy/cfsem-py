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
#include "opera.hh"

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	Opera::Opera(const boost::filesystem::path &fname){
		// add extension
		boost::filesystem::path fname_ext = fname;
		fname_ext.replace_extension("cond");

		// open file for writing
		fid_.open(fname_ext.string(),std::fstream::binary);
		
		// ensure file is open
		if(!fid_.is_open())rat_throw_line("could not open file for writing");

		// set precision
		fid_ << std::fixed;
		fid_ << std::setprecision(12);

		// conductor file
		fid_ << "CONDUCTOR\n";

		// set first drive
		drive_idx_ = 0;
	}

	// destructor
	Opera::~Opera(){
		// write footer
		fid_ << "QUIT\n";

		// close file
		if(fid_.is_open())fid_.close();
	}

	// factory
	ShOperaPr Opera::create(const boost::filesystem::path &fname){
		//return ShCalcFreeCADPr(new CalcFreeCAD);
		return std::make_shared<Opera>(fname);
	}

	// write to output file
	void Opera::write_edges(
		const arma::field<arma::Mat<fltp> > &x,
		const arma::field<arma::Mat<fltp> > &y,
		const arma::field<arma::Mat<fltp> > &z,
		const fltp J){

		// walk over coil sections
		for(arma::uword j=0;j<x.n_elem;j++){
			// walk over edges
			for(arma::uword k=0;k<x(j).n_cols-1;k++){
				fid_ << "DEFINE BR8\n";

				// settings
				fid_ << "0.0 0.0 0.0 0.0 0.0 0.0\n";
				fid_ << "0.0 0.0 0.0\n";
				fid_ << "0.0 0.0 0.0\n";

				// nodes in first face
				for(arma::uword l=0;l<4;l++)
					fid_ << x(j)(l,k) << " " << y(j)(l,k) << " " << z(j)(l,k) << "\n";

				// nodes in second face
				for(arma::uword l=0;l<4;l++)
					fid_ << x(j)(l,k+1) << " " << y(j)(l,k+1) << " " << z(j)(l,k+1) << "\n";

				// current 
				fid_ << J << " 1 'drive " << drive_idx_ << "'\n";

				// mirror settings
				fid_ << "0 0 0\n"; // no mirror
				fid_ << "1e-6\n"; // tolerance
			}
		}

		// increment drive
		drive_idx_++;
	}


}}