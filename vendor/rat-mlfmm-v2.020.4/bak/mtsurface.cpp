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
#include "mtsurface.hh"


// constructor
MTSurface::MTSurface(){
	
}

// factory
ShMTSurfacePr MTSurface::create(){
	return std::make_shared<MTSurface>();
}

// get target point coordinates
arma::Mat<double> MTSurface::get_target_coords() const{
 	return Rn_;
}

// get target point coordinates
arma::Mat<double> MTSurface::get_target_coords(
	const arma::Row<arma::uword> &indices) const{
 	return Rn_.cols(indices);
}

// number of points stored
arma::uword MTSurface::num_targets() const{
	return num_nodes_;
}

// write to gmsh file
void MTSurface::export_gmsh(ShGmshFilePr gmsh){
	// call parent class
	Surface::export_gmsh(gmsh);

	// check if field calculated
	if(has_field()){
		arma::Mat<double> B = get_field("B",3);
		arma::Row<double> Bmag = arma::sqrt(arma::sum(B%B,0));

		// data
		gmsh->write_nodedata(B,"Mgn. Flux Density (B)");
		gmsh->write_nodedata(Bmag,"Mgn. Flux Density norm (|B|)");
		
		arma::Mat<double> H = get_field("H",3);
		arma::Row<double> Hmag = arma::sqrt(arma::sum(H%H,0));

		// data
		gmsh->write_nodedata(H,"Mgn. Field (H)");
		gmsh->write_nodedata(Hmag,"Mgn. Field norm (|H|)");	
	}
}