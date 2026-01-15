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
#include "marchingcubes.hh"

// rat common headers
#include "elements.hh"
#include "extra.hh"

// using tables and concept from Paul Bourke
// http://www.paulbourke.net/geometry/polygonise/

// code specific to Rat
namespace rat{namespace cmn{

	// constructor
	MarchingCubes::MarchingCubes(){
		
	}

	// factory
	ShMarchingCubesPr MarchingCubes::create(){
		return std::make_shared<MarchingCubes>();
	}

	// Linearly interpolate the position where an isosurface cuts
	// an edge between two vertices, each with their own scalar value
	arma::Col<fltp> MarchingCubes::vertex_interp(
		const fltp level,
		const arma::Col<fltp> &p1,
		const arma::Col<fltp> &p2,
		const fltp valp1,
		const fltp valp2,
		const fltp tol) const{

		// handle exeptions
		if (std::abs(level-valp1) < tol)return p1;
		if (std::abs(level-valp2) < tol)return p2;
		if (std::abs(valp1-valp2) < tol)return p1;

		// calculate coordinate
		const fltp mu = (level - valp1)/(valp2 - valp1);
		return p1 + mu*(p2 - p1);
	}

	// polygonise a cube
	arma::Mat<fltp> MarchingCubes::polygonise(
		const arma::Mat<fltp> &R, 
		const arma::Row<fltp> &val, 
		const fltp level) const{

		// check if tables are set
		assert(is_setup_);
		assert(R.n_cols==num_nodes_input_);
		assert(val.n_elem==num_nodes_input_);
		assert(!val.has_nan());
		assert(val.is_finite());

		// number of dimensions
		const arma::uword num_dim = R.n_rows;

		// Determine the index into the edge table which
		// tells us which vertices are inside of the surface
		int cube_index = 0;
		int power = 1;
		for(arma::uword i=0;i<num_nodes_input_;i++){
			if(val(i)<level)cube_index |= power;
			power*=2;
		}

		// element is entirely in/out of the surface
		if(tri_table_(0,cube_index) == -1)return arma::Mat<fltp>(num_dim,0);

		// tolerance
		const fltp tol = tolerance_*(arma::max(val) - arma::min(val));

		// number of edges
		const arma::uword num_edges = edge_list_.n_rows;

		// allocate vertex list
		arma::Mat<fltp> vertex_list(num_dim,num_edges); 

		// walk over edges
		power = 1;
		for(arma::uword i=0;i<num_edges;i++){

			// check if edge is active
			if(edge_table_(cube_index) & power){
				// get node indeces of edge
				const arma::uword e1 = edge_list_(i,0); 
				const arma::uword e2 = edge_list_(i,1);

				// check if they are not the same
				assert(e1!=e2);
				// assert(e1<num_edges);
				// assert(e2<num_edges);

				// place vertex
				vertex_list.col(i) = vertex_interp(level,R.col(e1),R.col(e2),val(e1),val(e2),tol);
			}

			// increment power
			power *= 2;
		}

		// extend vertex list to also include original nodes
		vertex_list = arma::join_horiz(vertex_list, R);

		// allocate output
		arma::Mat<fltp> elements(num_nodes_output_*num_dim, max_num_output_elements_);

		// Create the triangle
		arma::uword num_output_elements = 0;

		// walk over elements
		for(arma::uword i=0;tri_table_(i,cube_index)!=-1;i+=num_nodes_output_){
			// walk over nodes inside this element
			for(arma::uword j=0;j<num_nodes_output_;j++){
				// store point in element matrix
				elements.submat(j*num_dim,num_output_elements,
					(j+1)*num_dim-1,num_output_elements) = 
					vertex_list.col(tri_table_(i+j,cube_index));
			}

			// increment counter
			num_output_elements++;
		}

		// if there were no good triangles return
		if(num_output_elements==0)return arma::Mat<fltp>(R.n_rows,0);

		// return triangles
		return arma::reshape(elements.cols(0,num_output_elements-1),R.n_rows,num_nodes_output_*num_output_elements);
	}

	// perform marching cubes on a grid
	arma::Mat<fltp> MarchingCubes::polygonise(
		const fltp xmin, const fltp xmax, const arma::uword num_x, 
		const fltp ymin, const fltp ymax, const arma::uword num_y, 
		const fltp zmin, const fltp zmax, const arma::uword num_z, 
		const arma::Row<fltp> &values, const fltp level){

		// check
		assert(num_x>=2); assert(num_y>=2); assert(num_z>=2);

		// check
		if(!is_setup_)rat_throw_line("tables not setup");
		if(num_nodes_input_!=8)rat_throw_line("number of input nodes must be eight for a grid");

		// calculate number of cells
		const arma::uword num_cell_x = num_x - 1;
		const arma::uword num_cell_y = num_y - 1;
		const arma::uword num_cell_z = num_z - 1;
		// const arma::uword num_cell = num_cell_x*num_cell_y*num_cell_z;

		// calculate grid spacing
		const fltp dx = (xmax-xmin)/num_cell_x;
		const fltp dy = (ymax-ymin)/num_cell_y;
		const fltp dz = (zmax-zmin)/num_cell_z;

		// allocate triangles
		std::list<arma::Mat<fltp> > tri_list;

		// walk over cells in the grid
		arma::uword num_tri = 0;
		for(arma::uword i=0;i<num_cell_x;i++){
			for(arma::uword j=0;j<num_cell_y;j++){
				for(arma::uword k=0;k<num_cell_z;k++){
					// create node coordinates
					const arma::Mat<fltp>::fixed<3,8> R{
						xmin+i*dx,     ymin+j*dy,     zmin+k*dz,
						xmin+(i+1)*dx, ymin+j*dy,     zmin+k*dz,
						xmin+(i+1)*dx, ymin+(j+1)*dy, zmin+k*dz,
						xmin+i*dx,     ymin+(j+1)*dy, zmin+k*dz,
						xmin+i*dx,     ymin+j*dy,     zmin+(k+1)*dz,
						xmin+(i+1)*dx, ymin+j*dy,     zmin+(k+1)*dz,
						xmin+(i+1)*dx, ymin+(j+1)*dy, zmin+(k+1)*dz,
						xmin+i*dx,     ymin+(j+1)*dy, zmin+(k+1)*dz};

					// create value array
					const arma::Row<fltp>::fixed<8> V = {
						values(k*num_x*num_y + j*num_x + i),
						values(k*num_x*num_y + j*num_x + i+1),
						values(k*num_x*num_y + (j+1)*num_x + i+1),
						values(k*num_x*num_y + (j+1)*num_x + i),
						values((k+1)*num_x*num_y + j*num_x + i),
						values((k+1)*num_x*num_y + j*num_x + i+1),
						values((k+1)*num_x*num_y + (j+1)*num_x + i+1),
						values((k+1)*num_x*num_y + (j+1)*num_x + i)};

					// check volume
					assert(arma::as_scalar(cmn::Hexahedron::calc_volume(R,arma::Col<arma::uword>{0,1,2,3,4,5,6,7}))!=0);

					// build triangles
					const arma::Mat<fltp> M = polygonise(R,V,level);

					// add to list
					if(!M.empty()){
						tri_list.push_back(M);
						num_tri+=M.n_cols;
					}
				}
			}
		}

		// combine
		arma::Mat<fltp> tri(num_nodes_output_,num_tri);
		arma::uword idx = 0;
		for(auto it=tri_list.begin();it!=tri_list.end();it++){
			const arma::uword num_tri_add = (*it).n_cols;
			tri.cols(idx,idx+num_tri_add-1) = (*it);
			idx+=num_tri_add;
		}

		// sanity check
		assert(idx==num_tri);

		// convert to matrix and return
		return tri;
	}


	// perform marching cubes on a grid
	arma::Mat<fltp> MarchingCubes::polygonise(
		const fltp xmin, const fltp xmax, const arma::uword num_x, 
		const fltp ymin, const fltp ymax, const arma::uword num_y, 
		const arma::Row<fltp> &values, const fltp level){

		// check
		assert(num_x>=2); assert(num_y>=2);

		// check
		if(!is_setup_)rat_throw_line("tables not setup");
		if(num_nodes_input_!=4)rat_throw_line("number of input nodes must be four for a grid");

		// calculate number of cells
		const arma::uword num_cell_x = num_x - 1;
		const arma::uword num_cell_y = num_y - 1;

		// calculate grid spacing
		const fltp dx = (xmax-xmin)/num_cell_x;
		const fltp dy = (ymax-ymin)/num_cell_y;

		// allocate triangles
		std::list<arma::Mat<fltp> > tri_list;

		// walk over cells in the grid
		arma::uword num_tri = 0;
		for(arma::uword i=0;i<num_cell_x;i++){
			for(arma::uword j=0;j<num_cell_y;j++){
				// create node coordinates
				const arma::Mat<fltp>::fixed<2,4> R{
					xmin+i*dx,     ymin+j*dy,
					xmin+(i+1)*dx, ymin+j*dy,   
					xmin+(i+1)*dx, ymin+(j+1)*dy,
					xmin+i*dx,     ymin+(j+1)*dy};

				// create value array
				const arma::Row<fltp>::fixed<4> V = {
					values(j*num_x + i),
					values(j*num_x + i+1),
					values((j+1)*num_x + i+1),
					values((j+1)*num_x + i)};

				// build edges
				const arma::Mat<fltp> M = polygonise(R,V,level);

				// add to list
				if(!M.empty()){
					tri_list.push_back(M);
					num_tri+=M.n_cols;
				}
			}
		}

		// combine
		arma::Mat<fltp> tri(num_nodes_output_,num_tri);
		arma::uword idx = 0;
		for(auto it=tri_list.begin();it!=tri_list.end();it++){
			const arma::uword num_tri_add = (*it).n_cols;
			tri.cols(idx,idx+num_tri_add-1) = (*it);
			idx+=num_tri_add;
		}

		// sanity check
		assert(idx==num_tri);

		// convert to matrix and return
		return tri;
	}

	// perform marching cubes on a mesh
	arma::Mat<fltp> MarchingCubes::polygonise(
		const arma::Mat<fltp> &Rn, 
		const arma::Row<fltp> &vals,
		const arma::Mat<arma::uword> &n,
		const fltp level){
	
		// check
		if(!is_setup_)rat_throw_line("tables not setup");
		if(n.n_rows!=num_nodes_input_)rat_throw_line("number of input nodes must be equal to input mesh");
	
		// allocate triangles
		std::list<arma::Mat<fltp> > tri_list;
		arma::uword num_tri = 0;
	
		// walk over elements
		for(arma::uword i=0;i<n.n_cols;i++){
			// build triangles
			const arma::Mat<fltp> M = polygonise(Rn.cols(n.col(i)),vals.cols(n.col(i)),level);
	
			// add to list
			if(!M.empty()){
				tri_list.push_back(M);
				num_tri+=M.n_cols;
			}
		}

		// combine
		arma::Mat<fltp> tri(Rn.n_rows,num_tri);
		arma::uword idx = 0;
		for(auto it=tri_list.begin();it!=tri_list.end();it++){
			const arma::uword num_tri_add = (*it).n_cols;
			tri.cols(idx,idx+num_tri_add-1) = (*it);
			idx+=num_tri_add;
		}

		// sanity check
		assert(idx==num_tri);

		// convert to matrix and return
		return tri;
	}


	// hexahedron to iso-triangles
	void MarchingCubes::setup_hex2isotri(){
		edge_table_ = arma::Row<int>::fixed<256>{
			0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
			0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
			0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
			0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
			0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
			0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
			0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
			0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
			0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
			0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
			0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
			0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
			0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
			0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
			0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
			0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
			0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
			0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
			0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
			0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
			0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
			0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
			0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
			0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
			0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
			0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
			0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
			0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
			0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
			0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
			0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
			0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };

		// table of edges forming triangles
		tri_table_ = arma::Mat<int>::fixed<16,256>{
			-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1,
			3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1,
			3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1,
			3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1,
			9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1,
			9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1,
			2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1,
			8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1,
			9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1,
			4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1,
			3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1,
			1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1,
			4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1,
			4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1,
			9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1,
			5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1,
			2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1,
			9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1,
			0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1,
			2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1,
			10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1,
			4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1,
			5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1,
			5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1,
			9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1,
			0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1,
			1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1,
			10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1,
			8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1,
			2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1,
			7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1,
			9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1,
			2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1,
			11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1,
			9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1,
			5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1,
			11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1,
			11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1,
			1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1,
			9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1,
			5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1,
			2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1,
			0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1,
			5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1,
			6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1,
			3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1,
			6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1,
			5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1,
			1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1,
			10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1,
			6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1,
			8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1,
			7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1,
			3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1,
			5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1,
			0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1,
			9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1,
			8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1,
			5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1,
			0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1,
			6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1,
			10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1,
			10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1,
			8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1,
			1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1,
			3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1,
			0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1,
			10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1,
			3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1,
			6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1,
			9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1,
			8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1,
			3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1,
			6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1,
			0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1,
			10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1,
			10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1,
			2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1,
			7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1,
			7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1,
			2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1,
			1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1,
			11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1,
			8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1,
			0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1,
			7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1,
			10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1,
			2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1,
			6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1,
			7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1,
			2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1,
			1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1,
			10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1,
			10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1,
			0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1,
			7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1,
			6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1,
			8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1,
			9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1,
			6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1,
			4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1,
			10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1,
			8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1,
			0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1,
			1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1,
			8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1,
			10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1,
			4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1,
			10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1,
			5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1,
			11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1,
			9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1,
			6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1,
			7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1,
			3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1,
			7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1,
			9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1,
			3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1,
			6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1,
			9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1,
			1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1,
			4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1,
			7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1,
			6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1,
			3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1,
			0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1,
			6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1,
			0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1,
			11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1,
			6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1,
			5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1,
			9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1,
			1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1,
			1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1,
			10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1,
			0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1,
			5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1,
			10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1,
			11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1,
			9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1,
			7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1,
			2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1,
			8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1,
			9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1,
			9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1,
			1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1,
			9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1,
			9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1,
			5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1,
			0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1,
			10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1,
			2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1,
			0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1,
			0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1,
			9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1,
			5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1,
			3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1,
			5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1,
			8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1,
			0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1,
			9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1,
			0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1,
			1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1,
			3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1,
			4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1,
			9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1,
			11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1,
			11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1,
			2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1,
			9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1,
			3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1,
			1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1,
			4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1,
			4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1,
			0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1,
			3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1,
			3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1,
			0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1,
			9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1,
			1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<12,2>{
			0,1,2,3, 4,5,6,7, 0,1,2,3,
			1,2,3,0, 5,6,7,4, 4,5,6,7};

		// number of input nodes
		num_nodes_input_ = 8;
		num_nodes_output_ = 3;
		max_num_output_elements_ = 5;

		// flag setup
		is_setup_ = true;
	}

	// tetrahedron to iso-triangles
	void MarchingCubes::setup_tet2isotri(){
		// binary numbers for edges
		int e0 = 1<<0; int e1 = 1<<1; int e2 = 1<<2; 
		int e3 = 1<<3; int e4 = 1<<4; int e5 = 1<<5;

		// tells which edges need an extra node
		edge_table_ = arma::Row<int>::fixed<16>{
			0, // no intersection
			e3 | e4 | e5, // corner around 0
			e1 | e2 | e5,
			e1 | e4 | e3 | e2, 
			e0 | e2 | e4,
			e0 | e2 | e3 | e5,
			e0 | e1 | e4 | e5,
			e0 | e1 | e3,
			e0 | e1 | e3,
			e0 | e1 | e4 | e5, 
			e0 | e2 | e3 | e5,
			e0 | e2 | e4,
			e1 | e2 | e3 | e4,
			e1 | e2 | e5,
			e3 | e4 | e5,
			0};

		// table of nodes forming triangles based on cube index
		const int X = -1;
		tri_table_ = arma::Mat<int>::fixed<7,16>{
			X,X,X, X,X,X, X, // 0 -> nothing
			3,5,4, X,X,X, X, // 1 -> nodes  0    -> edges 345
			2,5,1, X,X,X, X, // 2 -> nodes   1   -> edges 125
			3,1,4, 4,1,2, X, // 3 -> nodes  01   -> edges 1234
			4,2,0, X,X,X, X, // 4 -> nodes    2  -> edges 024
			3,5,2, 3,2,0, X, // 5 -> nodes  0 2  -> edges 0235
			4,5,1, 4,1,0, X, // 6 -> nodes   12  -> edges 0145
			3,1,0, X,X,X, X, // 7 -> nodes  012  -> edges 310
			3,0,1, X,X,X, X, // 8 -> nodes     3 -> edges 034
			1,5,4, 1,4,0, X, // 9 -> nodes  0  3 -> edges 0145
			2,5,3, 2,3,0, X, // 10 -> nodes  1 3 -> edges 0235
			4,0,2, X,X,X, X, // 11 -> nodes 01 3 -> edges 024
			3,4,2, 3,2,1, X, // 12 -> nodes   23 -> edges 1234
			5,2,1, X,X,X, X, // 13 -> nodes 0 23 -> edges 125
			4,5,3, X,X,X, X, // 14 -> nodes  123 -> edges 345
			X,X,X, X,X,X, X, // 15 -> nodes 0123 -> no edges
		};

		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<6,2>{
			2,1,1,0,0,0,
			3,3,2,3,2,1};

		// number of input nodes
		num_nodes_input_ = 4;
		num_nodes_output_ = 3;
		max_num_output_elements_ = 2;

		// flag setup
		is_setup_ = true;
	}


	// qiadrilateral to triangles
	void MarchingCubes::setup_quad2isoline(){
		// binary numbers for edges
		int e1 = 1<<0; int e2 = 1<<1; 
		int e3 = 1<<2; int e4 = 1<<3;
		edge_table_ = arma::Row<int>::fixed<16>{
			0, //1
			e1 | e4, //2
			e1 | e2, //3
			e2 | e4, //6
			e2 | e3, //4
			e1 | e2 | e3 | e4, //14
			e1 | e3, //7
			e3 | e4, //9
			e3 | e4, //5
			e1 | e3, //13
			e1 | e2 | e3 | e4, //15
			e2 | e3, //11
			e2 | e4, //8
			e1 | e2, //12
			e1 | e4, //10
			0}; //16

		// table of edges forming line sections
		const int X = -1;
		tri_table_ = arma::Mat<int>::fixed<5,16>{
			 X, X,  X, X,  X, //0
			 3, 0,  X, X,  X, //1
			 0, 1,  X, X,  X, //2
			 3, 1,  X, X,  X, //3
			 1, 2,  X, X,  X, //4
			 3, 0,  1, 2,  X, //5
			 0, 2,  X, X,  X, //6
			 3, 2,  X, X,  X, //7
			 2, 3,  X, X,  X, //8
			 2, 0,  X, X,  X, //9
			 0, 1,  2, 3,  X, //10
			 2, 1,  X, X,  X, //11
			 1, 3,  X, X,  X, //12
			 1, 0,  X, X,  X, //13
			 0, 3,  X, X,  X, //14
			 X, X,  X, X,  X}; //15

		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<4,2>{
			0,1,2,3, 1,2,3,0};

		// number of input nodes
		num_nodes_input_ = 4;
		num_nodes_output_ = 2;
		max_num_output_elements_ = 2;

		// flag setup
		is_setup_ = true;
	}

	// qiadrilateral to triangles
	void MarchingCubes::setup_quad2tri(){
		// binary numbers for edges
		int e1 = 1<<0; int e2 = 1<<1; 
		int e3 = 1<<2; int e4 = 1<<3;
		edge_table_ = arma::Row<int>::fixed<16>{
			0, //1
			e1 | e4, //2
			e1 | e2, //3
			e2 | e4, //6
			e2 | e3, //4
			e1 | e2 | e3 | e4, //14
			e1 | e3, //7
			e3 | e4, //9
			e3 | e4, //5
			e1 | e3, //13
			e1 | e2 | e3 | e4, //15
			e2 | e3, //11
			e2 | e4, //8
			e1 | e2, //12
			e1 | e4, //10
			0}; //16

		// table of edges forming triangles
		const int X = -1;
		tri_table_ = arma::Mat<int>::fixed<10,16>{
			 X, X, X,  X, X, X,  X, X, X,  X, //1
			 4, 0, 3,  X, X, X,  X, X, X,  X, //2
			 0, 5, 1,  X, X, X,  X, X, X,  X, //3
			 4, 5, 3,  5, 1, 3,  X, X, X,  X, //6
			 1, 6, 2,  X, X, X,  X, X, X,  X, //4
			 4, 0, 3,  1, 6, 2,  X, X, X,  X, //14 ambiguous
			 0, 5, 2,  5, 6, 2,  X, X, X,  X, //7
			 4, 5, 3,  5, 2, 3,  5, 6, 2,  X, //9
			 2, 7, 3,  X, X, X,  X, X, X,  X, //5
			 4, 0, 7,  0, 2, 7,  X, X, X,  X, //13
			 0, 5, 1,  2, 7, 3,  X, X, X,  X, //15 ambiguous
			 4, 5, 1,  4, 1, 2,  4, 2, 7,  X, //11
			 1, 6, 7,  7, 3, 1,  X, X, X,  X, //8
			 4, 0, 7,  0, 1, 7,  1, 6, 7,  X, //12
			 0, 5, 6,  0, 6, 3,  3, 6, 7,  X, //10
			 4, 5, 7,  5, 6, 7,  X, X, X,  X}; //16

		// flip
		for(arma::uword i=0;i<(tri_table_.n_rows-1)/3;i++)
			tri_table_.rows(i*3,(i+1)*3-1) = 
				arma::flipud(tri_table_.rows(i*3,(i+1)*3-1));
		
		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<4,2>{
			0,1,2,3, 1,2,3,0};

		// number of input nodes
		num_nodes_input_ = 4;
		num_nodes_output_ = 3;
		max_num_output_elements_ = 3;

		// flag setup
		is_setup_ = true;
	}

	// qiadrilateral to triangles
	void MarchingCubes::setup_tri2isoline(){
		// binary numbers for edges
		int e1 = 1<<0; int e2 = 1<<1; int e3 = 1<<2;
		edge_table_ = arma::Row<int>::fixed<8>{
			0,
			e1 | e3,
			e1 | e2,
			e2 | e3,
			e2 | e3,
			e1 | e2,
			e1 | e3,
			0};

		// table of edges forming triangles
		const int X = -1;
		tri_table_ = arma::Mat<int>::fixed<3,8>{
			 X, X, X,
			 2, 0, X,
			 0, 1, X,
			 2, 1, X,
			 1, 2, X,
			 1, 0, X,
			 0, 2, X,
			 X, X, X};

		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<3,2>{
			0,1,2, 1,2,0};

		// number of input nodes
		num_nodes_input_ = 3;
		num_nodes_output_ = 2;
		max_num_output_elements_ = 1;

		// flag setup
		is_setup_ = true;
	}

	// triangle to triangle
	void MarchingCubes::setup_tri2tri(){
		// binary numbers for edges
		int e1 = 1<<0; int e2 = 1<<1; int e3 = 1<<2;
		edge_table_ = arma::Row<int>::fixed<8>{
			0,
			e1 | e3,
			e1 | e2,
			e2 | e3,
			e2 | e3,
			e1 | e2,
			e1 | e3,
			0};

		// table of edges forming triangles
		const int X = -1;
		tri_table_ = arma::Mat<int>::fixed<7,8>{
			 X, X, X,  X, X, X,  X,
			 0, 2, 3,  X, X, X,  X,
			 0, 4, 1,  X, X, X,  X,
			 3, 1, 2,  3, 4, 1,  X,
			 1, 5, 2,  X, X, X,  X,
			 3, 0, 1,  3, 1, 5,  X,
			 0, 4, 2,  4, 5 ,2,  X,
			 3, 4, 5,  X, X, X,  X};

		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<3,2>{
			0,1,2, 1,2,0};

		// number of input nodes
		num_nodes_input_ = 3;
		num_nodes_output_ = 3;
		max_num_output_elements_ = 2;

		// flag setup
		is_setup_ = true;
	}

	// line to line
	void MarchingCubes::setup_line2line(){
		// binary numbers for edges
		int e1 = 1<<0;
		edge_table_ = arma::Row<int>::fixed<4>{
			0,
			e1,
			e1,
			0};

		// table of edges forming lines
		const int X = -1;
		tri_table_ = arma::Mat<int>::fixed<3,4>{
			 X, X, X,
			 1, 0, X,
			 0, 2, X,
			 1, 2, X};

		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<1,2>{
			0, 1};

		// number of input nodes
		num_nodes_input_ = 2;
		num_nodes_output_ = 2;
		max_num_output_elements_ = 1;

		// flag setup
		is_setup_ = true;
	}

	// line to point at iso value
	void MarchingCubes::setup_line2isopoint(){
		// binary numbers for edges
		int e1 = 1<<0;
		edge_table_ = arma::Row<int>::fixed<4>{
			0,
			e1,
			e1,
			0};

		// table of edges forming lines
		const int X = -1;
		tri_table_ = arma::Mat<int>::fixed<2,4>{
			 X, X,
			 2, X,
			 2, X,
			 X, X};

		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<1,2>{};

		// number of input nodes
		num_nodes_input_ = 2;
		num_nodes_output_ = 1;
		max_num_output_elements_ = 1;

		// flag setup
		is_setup_ = true;
	}

	// qiadrilateral to triangles
	void MarchingCubes::setup_point2point(){
		// binary numbers for edges
		edge_table_ = arma::Row<int>::fixed<1>{0};

		// table of edges forming triangles
		const int X = -1;
		tri_table_ = arma::Mat<int>::fixed<2,2>{
			 X, X,
			 0, X};

		// build edge list
		edge_list_ = arma::Mat<arma::uword>::fixed<0,0>{}; // there is no edges

		// number of input nodes
		num_nodes_input_ = 1;
		num_nodes_output_ = 1;
		max_num_output_elements_ = 1;

		// flag setup
		is_setup_ = true;
	}


	// static helper function for combining line elements
	// combines output of for example tri2isoline and quad2isoline
	// in which the line segments are defined separately
	arma::field<arma::Mat<fltp> > MarchingCubes::combine_line_segments(const arma::Mat<fltp> &R){
		// check if values exist
		if(R.empty())return{};

		// copy nodes
		arma::Mat<fltp> Rs = R;

		// get tolerance
		const fltp tol = RAT_CONST(1e-7)*arma::as_scalar(cmn::Extra::vec_norm(arma::max(Rs,1) - arma::min(Rs,1)));

		// drop zero length elements
		arma::Row<arma::uword> idx_remove(Rs.n_cols,arma::fill::zeros);
		for(arma::uword i=0;i<Rs.n_cols/2;i++){
			const fltp ell = arma::as_scalar(cmn::Extra::vec_norm(Rs.col(2*i+1) - Rs.col(2*i)));
			if(ell<tol){idx_remove(2*i) = 1; idx_remove(2*i+1) = 1;}
		}
		Rs.shed_cols(arma::find(idx_remove));

		// create elements
		arma::Mat<arma::uword> s = arma::reshape(
			arma::regspace<arma::Col<arma::uword> >(0,Rs.n_cols-1),2,Rs.n_cols/2);

		// combine nodes 
		const arma::Row<arma::uword> idx = cmn::Extra::combine_nodes(Rs,tol);

		// re-index elements
		s = arma::reshape(idx(arma::vectorise(s)),s.n_rows,s.n_cols);

		// remove elements connected to only one node
		s.shed_cols(arma::find(s.row(0)==s.row(1)));



		// determine graph connectivity
		const arma::Row<arma::uword> nc12 = Extra::conncomp(s.t(),Rs.n_cols);
		const arma::uword num_separate_graphs = arma::max(nc12) + 1;
		const arma::Row<arma::uword> sc12 = nc12.cols(s.row(0));

		// walk over graphs
		arma::field<arma::Mat<arma::uword> > ssub(num_separate_graphs);
		for(arma::uword i=0;i<num_separate_graphs;i++){
			// extract this component
			ssub(i) = s.cols(arma::find(sc12==i));


			// this part of the code needs further improvement!
			// always from low to high index
			for(arma::uword j=0;j<ssub(i).n_cols;j++)
				ssub(i).col(j) = arma::sort(ssub(i).col(j));

			// sort edge matrix
			arma::Row<arma::uword> sidx;
			sidx = arma::sort_index(ssub(i).row(1)).t(); ssub(i) = ssub(i).cols(sidx); 
			sidx = arma::stable_sort_index(ssub(i).row(0)).t(); ssub(i) = ssub(i).cols(sidx); 
			
			// walk over s and put edges in the correct order
			for(arma::uword j=0;j<ssub(i).n_cols-1;j++){
				// find next section
				const arma::Row<arma::uword> idx2 = arma::find(ssub(i).submat(0,j+1,0,ssub(i).n_cols-1)==ssub(i)(1,j),1,"first").t();

				// check if exists
				if(!idx2.is_empty()){
					ssub(i).swap_cols(j+1+arma::as_scalar(idx2),j+1);
					continue;
				}

				// find next section
				const arma::Row<arma::uword> idx3 = arma::find(ssub(i).submat(1,j+1,1,ssub(i).n_cols-1)==ssub(i)(1,j),1,"first").t();

				// check if exists
				if(!idx3.is_empty()){
					ssub(i).swap_cols(j+1+arma::as_scalar(idx3),j+1);
					ssub(i).col(j+1) = arma::flipud(ssub(i).col(j+1));
					continue;
				}
			}
		}

		// combine parts again
		arma::uword cnt = 0;
		for(arma::uword i=0;i<num_separate_graphs;i++){
			s.cols(cnt,cnt+ssub(i).n_cols-1) = ssub(i);
			cnt+=ssub(i).n_cols;
		}

		// combine sections
		arma::Col<arma::uword> sections = arma::join_vert(
			arma::Col<arma::uword>{0}, arma::find(
			s.row(0).eval().tail_cols(s.n_cols-1)!=
			s.row(1).eval().head_cols(s.n_cols-1))+1,
			arma::Col<arma::uword>{s.n_cols});

		// get nodes
		arma::field<arma::Mat<fltp> > lines(sections.n_elem-1);
		for(arma::uword i=0;i<sections.n_elem-1;i++)
			lines(i) = arma::join_horiz(Rs.cols(s.submat(0,sections(i),0,sections(i+1)-1)),Rs.col(s(1,sections(i+1)-1)));

		// skip duplicate nodes
		for(arma::uword i=0;i<sections.n_elem-1;i++)
			lines(i).shed_cols(1+arma::find(cmn::Extra::vec_norm(arma::diff(lines(i),1,1))<tol));

		// return lines
		return lines;
	}

}}