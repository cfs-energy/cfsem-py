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
#include "gpukernels.hh"

// CUDA runtime includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
// #include <cub/cub.cuh> 

// define pi
#define CU_PI RAT_CUCONST(3.14159265358979323846)



// note that we found out the hard way that 
// armadillo is incompatible with nvcc compiler 
// at this time. Therefore, this part is written 
// more in a C style rather than of C++ style

// TODO replace sqrtf with sqrt (according to documentation 
// this is overloaded and should therefore be equally fast 

// code specific to Rat
namespace rat{namespace fmm{

	// define the float4 type
	#if defined(RAT_DOUBLE_PRECISION) && defined(RAT_CUDA_DOUBLE_PRECISION)
		typedef double4 cufltp4;
		#define MAKE_FLOAT4(x,y,z,w) make_double4(x,y,z,w);
	#else
		typedef float4 cufltp4;
		#define MAKE_FLOAT4(x,y,z,w) make_float4(x,y,z,w);
	#endif

	// addition for cufltp4
	__device__ inline cufltp4 add(const cufltp4 a, const cufltp4 b){
		return MAKE_FLOAT4(a.x + b.x, a.y + b.y, a.z + b.z, RAT_CUCONST(0.0));
	}

	// addition for double4 and cufltp4
	#ifndef RAT_CUDA_DOUBLE_PRECISION
	__device__ inline double4 add(const double4 a, const cufltp4 b){
		return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, RAT_CUCONST(0.0));
	}
	#endif

	// subtraction for cufltp4
	__device__ inline cufltp4 sub(const cufltp4 a,const cufltp4 b){
		return MAKE_FLOAT4(a.x - b.x, a.y - b.y, a.z - b.z, RAT_CUCONST(0.0));
	}

	// subtraction for cufltp4
	__device__ inline cufltp4 cross(const cufltp4 a,const cufltp4 b){
		return MAKE_FLOAT4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, RAT_CUCONST(0.0));
	}

	// subtraction for cufltp4
	__device__ inline cufltp dot(const cufltp4 a,const cufltp4 b){
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}

	// scaling for cufltp4
	__device__ inline cufltp4 scale(const cufltp4 a, const cufltp k){
		return MAKE_FLOAT4(a.x*k, a.y*k, a.z*k, RAT_CUCONST(0.0));
	}

	// scaling for cufltp4
	__device__ inline cufltp norm(const cufltp4 a){
		return sqrtf(dot(a,a));
	}

	// definition of error check function
	#define gpuErrchk(ans){gpuAssert((ans), __FILE__, __LINE__);}
	#define cublasErrchk(ans){cublasAssert((ans), __FILE__, __LINE__);}

	// simple error check function for all GPU calls
	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
		if (code != cudaSuccess){
			fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) std::exit(code);
		}
	}

	// simple error check function for all CUBLAS calls
	inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true){
		if (code != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr,"CUBLASassert: %i %s %d\n", code, file, line);
			if (abort) std::exit(code);
		}
	}

	// show info on all devices
	void GpuKernels::show_device_info(const std::set<int> &gpu_devices, const cmn::ShLogPr &lg){
		// safety return
		if(get_num_devices()==0)return;

		// output info to log
		lg->msg(2,"%sCUDA GPU Devices%s\n",KBLU,KNRM);

		// walk over enabled devices and show their info
		for(auto it=gpu_devices.begin();it!=gpu_devices.end();it++)show_device_info(*it,lg);

		// end of list
		lg->msg(-2);
	}

	// tag gpu device and cublas to reduce 
	// access time during kernel execution
	// as to avoid it from messing up the timings
	void GpuKernels::show_device_info(const int device_index, const cmn::ShLogPr &lg){
		// check if there are any devices
		if(device_index>=get_num_devices())return;

		// get GPU device properties
		cudaDeviceProp prop;
		gpuErrchk(cudaGetDeviceProperties(&prop, device_index));

        // cuda toolkit 13 deprecated these properties in cudaDeviceProp
        int clockRate;
        int memoryClockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, device_index);
        cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, device_index);

		lg->msg("%s%s%s\n",KCYN,prop.name,KNRM);
		lg->msg("Device Index: %s%i%s\n",KYEL,device_index,KNRM);
		lg->msg("Clock Rate [MHz]: %s%d%s\n",KYEL,clockRate/1000,KNRM);
		lg->msg("Memory Clock Rate [MHz]: %s%d%s\n",KYEL,memoryClockRate/1000,KNRM);
		lg->msg("Memory Bus Width [bits]: %s%d%s\n",KYEL,prop.memoryBusWidth,KNRM);
		lg->msg("Peak Memory Bandwidth [GB/s]: %s%4.2f%s\n",KYEL, 
			2.0*memoryClockRate*(prop.memoryBusWidth/8)/1.0e6,KNRM);
		lg->msg("Global Memory Size: %s%6.2f%s [GB]\n",KYEL,(float)prop.totalGlobalMem/1024/1024/1024,KNRM);
		lg->msg("Multi processor count: %s%i%s\n",KYEL,prop.multiProcessorCount,KNRM);
		lg->msg("\n");

		// char name[256];
		// size_t totalGlobalMem;
		// size_t sharedMemPerBlock;
		// int regsPerBlock;
		// int warpSize;
		// size_t memPitch;
		// int maxThreadsPerBlock;
		// int maxThreadsDim[3];
		// int maxGridSize[3];
		// int clockRate;
		// size_t totalConstMem;
		// int major;
		// int minor;
		// size_t textureAlignment;
		// int deviceOverlap;
		// int multiProcessorCount;
		// int kernelExecTimeoutEnabled;
		// int integrated;
		// int canMapHostMemory;
		// int computeMode;
		// int maxTexture1D;
		// int maxTexture2D[2];
		// int maxTexture3D[3];
		// int maxTexture1DLayered[2];
		// int maxTexture2DLayered[3];
		// size_t surfaceAlignment;
		// int concurrentKernels;
		// int ECCEnabled;
		// int pciBusID;
		// int pciDeviceID;
		// int pciDomainID;
		// int tccDriver;
		// int asyncEngineCount;
		// int unifiedAddressing;
		// int memoryClockRate;
		// int memoryBusWidth;
		// int l2CacheSize;
		// int maxThreadsPerMultiProcessor;
	}

	// get number of gpu devices
	int GpuKernels::get_num_devices(){
		int count;
		cudaError_t flag = cudaGetDeviceCount(&count);
		if(flag==cudaSuccess)return count; else return 0;
	}

	// get number of gpu devices
	void GpuKernels::set_device(const int device_index){
		gpuErrchk(cudaSetDevice(device_index));
	}

	// get number of gpu devices
	int GpuKernels::get_device(){
		int device_index;
		gpuErrchk(cudaGetDevice(&device_index));
		return device_index;
	}

	// get properties
	std::string GpuKernels::get_device_name(const int device_index){
		cudaDeviceProp prop;
		gpuErrchk(cudaGetDeviceProperties(&prop, device_index));
		return prop.name;
	}

	// // number of threads to use per block
	// const int tile_zsgemm = 16;
	// const int ntpb_zsgemm= tile_zsgemm*tile_zsgemm;

	// // Source to Target Kernel for 
	// // the calculation of magnetic field
	// __global__ void zero_stride_gemm(
	// 	int m, int n, int k,
	// 	const std::complex<cufltp> *A, int lda,
	// 	const std::complex<cufltp> *Barray[], int ldb,
	// 	std::complex<cufltp> *Carray[], int ldc,
	// 	int batchCount){

	// 	// allocate shared memory banks
	// 	__shared__ std::complex<cufltp> sh_Ma[ntpb_So2Ta];
	// 	__shared__ std::complex<cufltp> sh_Mb[ntpb_So2Ta];
		
	// 	// my index into b and c

	// 	// my (sub)column of b and c

	// 	// my row

	// 	// allocate my output

	// 	// retreive relevant pointers to A, B and C from global memory


	// 	// walk over tiles
	// 	for(int i=0;i<polesize/tilesize;i++){
	// 		// pre-load tile
	// 		// (note columnwise coalesced memory access)
	// 		// (row loads column)

	// 		// walk over column and add contribution
	// 		for(int j=0;j<tilesize;j++){

	// 		}
	// 	}

	// 	// add to global C
	// }

	// Factorial
	inline __device__ cufltp factorial(int n){
		cufltp v = RAT_CUCONST(1.0); 
		for(int i=1; i<=n; i++)v *= i; 
		return v;
	}

	// nm2fidx
	inline __device__ int nm2fidx(int n, int m){
		return n*(n+1)+m;
	}

	// nm2fidx for m>=0 only
	inline __device__ int hfnm2fidx(int n, int m){
		return n*(n+1)/2+m;
	}

	// threads settings
	const int ntpb_So2Mp = 64;

		
	// read cufltp value from global memory using only one thread of the warp
	// this should avoid reading the same value from all threads
	template<typename T>
	inline __device__ T warp_read(const T *target){
		// allocate value in each thread
		T value = 0;

		// get mask and leader
		int mask = __activemask();
		int leader = __ffs(mask) - 1; 

		// load value from global memory for first thread in warp only
		if(threadIdx.x%warpSize==leader)value = target[0];
		
		// synchronise threads in warp
		value = __shfl_sync(mask, value, leader, warpSize);

		// return synchronised value
		return value;
	}

	// aggregation function
	inline __device__ void atomic_agg(cufltp *target, cufltp value){
		// allocate accumulated values in all threads
		cufltp added_values = value;

		// get mask and leader
		int mask = __activemask();
		int leader = __ffs(mask) - 1; 

		// add values from all threads
		for(int i = 1; i<warpSize; i*=2){
			added_values += __shfl_xor_sync(mask, added_values, i);
		}

		// add accumulated values to target
		// using atomic operation on first thread
		// of warp only
		if(threadIdx.x%warpSize==leader){
			atomicAdd(target, added_values);
		}
	}

	// calculate harmonic contribution to mp
	__device__ void so2mp_core(cufltp* sharedMp, cufltp4 Ieff, cufltp Ynmre, cufltp Ynmim, int n, int m, int polesize){
		// right half m>=0 taking into account complex number
		int idx = 2*hfnm2fidx(n,m);

		// add x direction
		atomic_agg(sharedMp+idx+0*2*polesize+0, Ieff.w*Ieff.x*Ynmre); // real 
		atomic_agg(sharedMp+idx+0*2*polesize+1, -Ieff.w*Ieff.x*Ynmim); // imag

		// add y direction
		atomic_agg(sharedMp+idx+1*2*polesize+0, Ieff.w*Ieff.y*Ynmre); // real 
		atomic_agg(sharedMp+idx+1*2*polesize+1, -Ieff.w*Ieff.y*Ynmim); // imag

		// add z direction
		atomic_agg(sharedMp+idx+2*2*polesize+0, Ieff.w*Ieff.z*Ynmre); // real 
		atomic_agg(sharedMp+idx+2*2*polesize+1, -Ieff.w*Ieff.z*Ynmim); // imag
	}

	// Source to multipole kernel
	__global__ void so2mp_gpu_kernel(
		cufltp *globalMp,
		const cufltp4 *Rn, // position with respect to node
		const cufltp4 *Ieff, // element direction vector
		const long long unsigned int *first_source,
		const long long unsigned int *last_source,
		const int num_exp,
		const long long unsigned int num_nodes,
		const long long unsigned int batch_offset){

		// create localpole (dynamic allocation)
		extern __shared__ cufltp sharedMp[];
		
		// get index
		const long long unsigned int node_idx = blockIdx.x + batch_offset;

		// source indexes
		long long unsigned int sidx1 = warp_read(&first_source[node_idx]);
		long long unsigned int sidx2 = warp_read(&last_source[node_idx]);
		
		// size of multipole (only positive m>=0 half)
		unsigned int polesize = hfnm2fidx(num_exp,num_exp) + 1;
		unsigned int num_blocks = (sidx2-sidx1+1)/ntpb_So2Mp + 1;

		// synchronise
		__syncthreads();

		// set shared memory to zeros
		if(threadIdx.x==0)memset(sharedMp, 0, 2*3*polesize*sizeof(cufltp));

		// synchronise
		__syncthreads();

		// get source locations for this node
		// const long long unsigned int sidx1 = first_source[node_idx];
		// const long long unsigned int sidx2 = last_source[node_idx];
		// const int num_blocks = (sidx2-sidx1+1)/ntpb_So2Mp + 1;

		// walk over sources inside this node
		for(unsigned int i=0;i<num_blocks;i++){
			// get source index
			const long long unsigned int source_index = sidx1 + i*blockDim.x + threadIdx.x;

			// check if source index in range
			if(source_index<=sidx2){
				// get element from global memory
				cufltp4 myRn = Rn[source_index];  // relative position to node
				cufltp4 myIeff = Ieff[source_index]; // direction vector and current

				// switch relative position
				myRn = scale(myRn,-1);

				// position in spherical coordinates
				cufltp rho = sqrtf(dot(myRn,myRn));
				cufltp mu = max(-RAT_CUCONST(1.0),min(RAT_CUCONST(1.0),myRn.z/rho));
				cufltp phi = atan2f(myRn.y,myRn.x);

				// storing legendre polynomials
				cufltp Pn = RAT_CUCONST(1.0);
				cufltp fact = RAT_CUCONST(1.0);
				cufltp s = sqrtf(RAT_CUCONST(1.0) - mu*mu);

				// keep track of rho^m 
				cufltp rhom = RAT_CUCONST(1.0);

				// walk over m
				for(int m=0;m<=num_exp;m++){
					// update legendre polynomials
					// P^(m)_(n+1), P^(m)_(n), P^(m)_(n-1), respectively
					cufltp Pnm = Pn; 
					cufltp Pnm1 = Pnm; 

					// cufltp facnm = factorial(n-m);
					cufltp sqfac = sqrtf(RAT_CUCONST(1.0)/factorial(2*m));

					// calculate sin and cos of angle
					cufltp cosphim = __cosf(phi*m);
					cufltp sinphim = __sinf(phi*m);

					// calculate real and complex parts of Ynm
					cufltp Ynmre = rhom*sqfac*Pnm*cosphim;
					cufltp Ynmim = rhom*sqfac*Pnm*sinphim; // note that this is for Y_{n, -m}

					// store in multipole for diagonal
					so2mp_core(sharedMp, myIeff, Ynmre, Ynmim, m, m, polesize);

					// second
					Pnm = mu*(2*m+1)*Pnm;

					// calculate rhon
					cufltp rhon = rhom;

					// walk over n
					for(int n=m+1;n<=num_exp;n++){
						// cufltp facnm = factorial(n-m);
						sqfac = sqrtf(factorial(n-m)/factorial(n+m));

						// update rho^n
						rhon *= rho;

						// calculate real and complex parts of Ynm
						Ynmre = rhon*sqfac*Pnm*cosphim;
						Ynmim = rhon*sqfac*Pnm*sinphim; // note that this is for Y_{n, -m}

						// store in multipole
						// so2mp_core(globalMp, myIeff, Ynmre, Ynmim, rho, phi, n, m, polesize, node_idx*2*3*polesize);
						so2mp_core(sharedMp, myIeff, Ynmre, Ynmim, n, m, polesize);

						// shift Legendre polynomials eq. 2.1.54, p48
						cufltp Pnm2 = Pnm1; Pnm1 = Pnm;
						Pnm = (mu*(2*n+1)*Pnm1-(n+m)*Pnm2)/(n-m+1);
					}

					// update legendre polynomial
					Pn *= -fact*s;

					// update factor
					fact += 2;

					// update rho^m
					rhom *= rho;
				}
			}
		}

		// synchronise
		__syncthreads();

		
		// store to global memory
		// if(threadIdx.x<2*3*polesize){
		// 	globalMp[node_idx*2*3*polesize + threadIdx.x] = sharedMp[threadIdx.x];
		// }

		// store to global memory
		if(threadIdx.x==0){
			memcpy((void*)(globalMp + node_idx*2*3*polesize), (void*)sharedMp, 2*3*polesize*sizeof(cufltp));
		}
	}

	// cublas source to multipole translation kernel
	void GpuKernels::so2mp_kernel(
		std::complex<cufltp> *Mp_ptr,
		const int num_exp,
		const cufltp *Rn_ptr,
		const cufltp *Ieff_ptr,
		const long long unsigned int num_sources,
		const long long unsigned int *first_source_ptr,
		const long long unsigned int *last_source_ptr,
		const long long unsigned int num_nodes,
		const std::set<int> &gpu_devices,
		const bool is_managed){

		// polesize
		const long long unsigned int half_polesize = num_exp*(num_exp+1)/2 + num_exp + 1;

		// pointers
		cufltp *Mp_dev;
		cufltp4 *Rn_dev, *Ieff_dev;
		long long unsigned int *first_source_dev, *last_source_dev;

		// set gpu device
		gpuErrchk(cudaSetDevice(*gpu_devices.begin()));

		// create streams
		cudaStream_t stream1;
		gpuErrchk(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
		cudaStream_t stream2;
		gpuErrchk(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));


		// allocate multipoles
		long long unsigned int mp_matrix_size = 2*3*num_nodes*half_polesize;
		gpuErrchk(cudaMalloc(&Mp_dev, mp_matrix_size*sizeof(cufltp))); 
		//gpuErrchk(cudaMemset(Mp_dev, 0, mp_matrix_size*sizeof(cufltp)));

		if(is_managed){
			Rn_dev = (cufltp4*)Rn_ptr; Ieff_dev = (cufltp4*)Ieff_ptr;
		}

		// copy arrays to GPU
		else{
			gpuErrchk(cudaMalloc(&Rn_dev, num_sources*sizeof(cufltp4))); 
			gpuErrchk(cudaMemcpyAsync(Rn_dev, Rn_ptr, num_sources*sizeof(cufltp4), cudaMemcpyHostToDevice, stream1));
			
			// copy arrays to GPU
			gpuErrchk(cudaMalloc(&Ieff_dev, num_sources*sizeof(cufltp4))); 
			gpuErrchk(cudaMemcpyAsync(Ieff_dev, Ieff_ptr, num_sources*sizeof(cufltp4), cudaMemcpyHostToDevice, stream1));
		}

		// copy arrays to GPU
		gpuErrchk(cudaMalloc(&first_source_dev, num_nodes*sizeof(long long unsigned int))); 
		gpuErrchk(cudaMemcpyAsync(first_source_dev, first_source_ptr, num_nodes*sizeof(long long unsigned int), cudaMemcpyHostToDevice, stream1));
		
		// copy arrays to GPU
		gpuErrchk(cudaMalloc(&last_source_dev, num_nodes*sizeof(long long unsigned int))); 
		gpuErrchk(cudaMemcpyAsync(last_source_dev, last_source_ptr, num_nodes*sizeof(long long unsigned int), cudaMemcpyHostToDevice, stream1));

		// divide over batches
		const long long unsigned int num_nodes_batch = 1024;
		const long long unsigned int num_batches = (long long unsigned int)std::ceil(((cufltp)num_nodes)/num_nodes_batch);

		// run batches
		for(long long unsigned int i=0;i<num_batches;i++){
			// calculate offset for the batch
			const long long unsigned int batch_offset = i*num_nodes_batch;
			const int num_nodes_this_batch = int(std::min(num_nodes-batch_offset, num_nodes_batch));
			
			// call kernel
			// launch kernel for vector potential
			// int num_blocks = num_nodes;
			// std::cout<<num_blocks<<std::endl;
			so2mp_gpu_kernel<<<num_nodes_this_batch, ntpb_So2Mp, 3*2*half_polesize*sizeof(cufltp), stream1>>>(
				Mp_dev, Rn_dev, Ieff_dev, first_source_dev, last_source_dev, num_exp, num_nodes, batch_offset);	

			// synchronise
			gpuErrchk(cudaStreamSynchronize(stream1));	

			// copy back previous batch
			gpuErrchk(cudaMemcpyAsync(
				Mp_ptr + batch_offset*3*half_polesize, 
				Mp_dev + batch_offset*2*3*half_polesize, 
				num_nodes_this_batch*2*3*half_polesize*sizeof(cufltp), 
				cudaMemcpyDeviceToHost, stream2));
		}

		// synchronise
		gpuErrchk(cudaStreamSynchronize(stream2));	
		gpuErrchk(cudaGetLastError());

		// copy multipoles back
		// gpuErrchk(cudaMemcpy(Mp_ptr, Mp_dev, mp_matrix_size*sizeof(cufltp), cudaMemcpyDeviceToHost));

		// destroy
		if(!is_managed){
			gpuErrchk(cudaFree(Rn_dev));
			gpuErrchk(cudaFree(Ieff_dev));
		}
		gpuErrchk(cudaFree(Mp_dev));
		gpuErrchk(cudaFree(first_source_dev));
		gpuErrchk(cudaFree(last_source_dev));

		// clean up streams
		gpuErrchk(cudaStreamDestroy(stream1));
		gpuErrchk(cudaStreamDestroy(stream2));
		// gpuErrchk(cudaStreamSynchronize(0)); 
	}


	// cublas multipole to localpole translation kernel
	void GpuKernels::mp2lp_kernel(
		const long long unsigned int num_dim, // number of dimensions
		const int polesize, // size of the multipoles and localpoles
		std::complex<cufltp> *Lp_ptr, // matrix with target localpoles
		const long long unsigned int num_lp, // number of localpoles
		const std::complex<cufltp> *Mp_ptr, // matrix with target multipoles
		const long long unsigned int num_mp, // number of multipoles
		const std::complex<cufltp> *Mint_ptr, // matrix with transformation matrices
		const long long unsigned int num_int_type, // number of interaction types
		const long long unsigned int *type_ptr, // interaction type
		const long long unsigned int *source_node, // source node indexes
		const long long unsigned int *target_node, // target node indexes
		const long long unsigned int *mp_idx, // multipole indexes
		const long long unsigned int *lp_idx, // localpole indexes
		const long long unsigned int *first_index,
		const long long unsigned int *last_index,
		const long long unsigned int num_group,
		const std::set<int> &gpu_devices){ // requested size of the batches

		// check for nans in output
		#ifndef NDEBUG 
		// check multipoles
		for(long long unsigned int i=0;i<num_mp*polesize*num_dim;i++){
			assert(std::isnan(std::real(Mp_ptr[i]))==false);
			assert(std::isnan(std::imag(Mp_ptr[i]))==false);
		}

		// check matrices
		for(long long unsigned int i=0;i<num_int_type*polesize*polesize;i++){
			assert(std::isnan(std::real(Mint_ptr[i]))==false);
			assert(std::isnan(std::imag(Mint_ptr[i]))==false);
		}
		#endif

		// check if there is any groups
		if(num_group==0)return;

		// find batch size
		long long unsigned int max_batch_size = 0;
		for(long long unsigned int i=0;i<num_group;i++)
			max_batch_size = std::max(max_batch_size, last_index[i]-first_index[i]+1);

		// constants
		const std::complex<cufltp> alpha(RAT_CUCONST(1.0),RAT_CUCONST(0.0));
		const std::complex<cufltp> beta(RAT_CUCONST(1.0),RAT_CUCONST(0.0));

		// get number of available CPU cores
		// int num_cpus = std::thread::hardware_concurrency();
		std::list<std::future<void> > futures;

		// create a lock
		std::mutex lock;

		// make index
		std::atomic<int> idx; idx = 0;

		// walk over available cpus
		for(auto it = gpu_devices.begin();it!=gpu_devices.end();it++){
			// get device index
			const int gpu = *it;

			// start new thread and capture device index
			futures.push_back(std::async(std::launch::async,[&,gpu](){
				// set gpu device for this thread
				GpuKernels::set_device(gpu);

				// create stream
				cudaStream_t stream;
				gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

				// create cublas
				cublasHandle_t handle;
				cublasErrchk(cublasCreate(&handle));
				cublasErrchk(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
				cublasErrchk(cublasSetStream(handle,stream));

				// allocate 
				std::complex<cufltp> *Mp_dev, *Lp_dev, *Mint_dev;
				int mp_shift, lp_shift, mint_shift;

				// copy multipole matrix to GPU
				gpuErrchk(cudaMalloc(&Mp_dev, num_mp*polesize*num_dim*sizeof(std::complex<cufltp>))); 
				gpuErrchk(cudaMemcpyAsync(Mp_dev, Mp_ptr, num_mp*polesize*num_dim*sizeof(std::complex<cufltp>), cudaMemcpyHostToDevice, stream));

				// copy conversion matrices to GPU
				gpuErrchk(cudaMalloc(&Mint_dev, polesize*polesize*num_int_type*sizeof(std::complex<cufltp>))); 
				gpuErrchk(cudaMemcpyAsync(Mint_dev, Mint_ptr, polesize*polesize*num_int_type*sizeof(std::complex<cufltp>), cudaMemcpyHostToDevice, stream));

				// set leading dimensions
				mp_shift = polesize; mint_shift = polesize;

				// create target matrix on device
				gpuErrchk(cudaMalloc(&Lp_dev, num_lp*polesize*num_dim*sizeof(std::complex<cufltp>))); 
				gpuErrchk(cudaMemsetAsync(Lp_dev, 0, num_lp*polesize*num_dim*sizeof(std::complex<cufltp>), stream));
				
				// set leading dimensions
				lp_shift = polesize; 

				// host pointer arrays
				std::complex<cufltp> **A, **B, **C; // = (std::complex<cufltp>**)malloc(max_batch_size*sizeof(std::complex<cufltp>*));
				gpuErrchk(cudaMallocHost(&A, max_batch_size*sizeof(std::complex<cufltp>*)));
				gpuErrchk(cudaMallocHost(&B, max_batch_size*sizeof(std::complex<cufltp>*)));
				gpuErrchk(cudaMallocHost(&C, max_batch_size*sizeof(std::complex<cufltp>*)));

				// allocate
				std::complex<cufltp> **A_dev, **B_dev, **C_dev;
				gpuErrchk(cudaMalloc(&A_dev, max_batch_size*sizeof(std::complex<cufltp>*)));
				gpuErrchk(cudaMalloc(&B_dev, max_batch_size*sizeof(std::complex<cufltp>*)));
				gpuErrchk(cudaMalloc(&C_dev, max_batch_size*sizeof(std::complex<cufltp>*)));

				// keep firing tasks untill run out of idx
				for(;;){

					// atomic increment of index so that no two threads can do same task
					const int i = idx++; 

					// check if last index
					if(i>=(int)num_group)break;

					// get interaction list range
					const long long unsigned int fidx = first_index[i];
					const long long unsigned int lidx = last_index[i];
					const long long unsigned int num_this_batch = lidx - fidx + 1;

					// check size
					assert(lidx>=fidx);
					assert(num_this_batch<=max_batch_size);

					// setup the calculation by pointing to the correct part of each matrix
					for(long long unsigned int j=0;j<num_this_batch;j++){
						// interaction matrix
						A[j] = Mint_dev + mint_shift*polesize*type_ptr[fidx + j];
						
						// source multipole
						B[j] = Mp_dev + mp_shift*num_dim*(mp_idx[source_node[fidx + j]]-1);
						
						// target localpole
						C[j] = Lp_dev + lp_shift*num_dim*(lp_idx[target_node[fidx + j]]-1);
					}

					// synchronise - this can possibly be 
					// improved by moving before gemm call
					gpuErrchk(cudaStreamSynchronize(stream));

					// copy
					gpuErrchk(cudaMemcpyAsync(A_dev, A, num_this_batch*sizeof(std::complex<cufltp>*), cudaMemcpyHostToDevice, stream));
					gpuErrchk(cudaMemcpyAsync(B_dev, B, num_this_batch*sizeof(std::complex<cufltp>*), cudaMemcpyHostToDevice, stream));
					gpuErrchk(cudaMemcpyAsync(C_dev, C, num_this_batch*sizeof(std::complex<cufltp>*), cudaMemcpyHostToDevice, stream));

					// synchronise stream after memcopies
					gpuErrchk(cudaStreamSynchronize(stream));

					// run cuda gemm batched call
					// need to try transposed mp matrix
					#ifdef RAT_CUDA_DOUBLE_PRECISION
					 	cublasErrchk(cublasGemmBatchedEx(handle,
							CUBLAS_OP_N, CUBLAS_OP_N,
							polesize, (int)num_dim, polesize,
							(const void*)&alpha,
							(const void**)A_dev, CUDA_C_64F, mint_shift,
							(const void**)B_dev, CUDA_C_64F, mp_shift, 
							(const void*)&beta,
							(void**)C_dev, CUDA_C_64F, lp_shift,
							num_this_batch, CUBLAS_COMPUTE_64F, 
							CUBLAS_GEMM_DEFAULT_TENSOR_OP));
					#else
						cublasErrchk(cublasGemmBatchedEx(handle,
							CUBLAS_OP_N, CUBLAS_OP_N,
							polesize, (int)num_dim, polesize,
							(const void*)&alpha,
							(const void**)A_dev, CUDA_C_32F, mint_shift,
							(const void**)B_dev, CUDA_C_32F, mp_shift, 
							(const void*)&beta,
							(void**)C_dev, CUDA_C_32F, lp_shift,
							(int)num_this_batch, CUBLAS_COMPUTE_32F, // CUBLAS_COMPUTE_32F_FAST_TF32 gives nans sometimes? 
							CUBLAS_GEMM_DEFAULT_TENSOR_OP));
					#endif
				}

				// allocate 
				std::complex<cufltp> *Lp_host;

				// synchronise stream after memcopies
				gpuErrchk(cudaStreamSynchronize(stream));
				gpuErrchk(cudaGetLastError());

				// create result matrix on host
				gpuErrchk(cudaMallocHost((void**)&Lp_host, num_lp*polesize*num_dim*sizeof(std::complex<cufltp>))); 
				gpuErrchk(cudaMemcpyAsync(Lp_host, Lp_dev, num_lp*polesize*num_dim*sizeof(std::complex<cufltp>), cudaMemcpyDeviceToHost, stream));

				// synchronise stream after memcopies
				gpuErrchk(cudaStreamSynchronize(stream));

				// add values
				lock.lock();
				for(long long unsigned int j=0;j<num_lp*polesize*num_dim;j++)Lp_ptr[j] += Lp_host[j];
				lock.unlock();

				// free memory on host
				gpuErrchk(cudaFreeHost(A));
				gpuErrchk(cudaFreeHost(B));
				gpuErrchk(cudaFreeHost(C));

				// free device memory
				gpuErrchk(cudaFree(A_dev));
				gpuErrchk(cudaFree(B_dev));
				gpuErrchk(cudaFree(C_dev));

				// free multipoles and localpoles
				gpuErrchk(cudaFree(Lp_dev));
				gpuErrchk(cudaFree(Mint_dev));
				gpuErrchk(cudaFree(Mp_dev)); 

				// the host localpole
				gpuErrchk(cudaFreeHost(Lp_host));

				// free cublas
				cublasErrchk(cublasDestroy(handle));

				// clean up streams
				gpuErrchk(cudaStreamDestroy(stream));
			}));
		}

		// make sure all threads are finished
		for(auto it=futures.begin();it!=futures.end();it++)(*it).get();

		// check for nans in output
		#ifndef NDEBUG 
		for(long long unsigned int i = 0;i<num_lp*polesize*num_dim;i++){
			assert(std::isnan(std::real(Lp_ptr[i]))==false);
			assert(std::isnan(std::imag(Lp_ptr[i]))==false);
		}
		#endif


		// free memory on host
		// free(A); free(B); free(C);
	}


	// number of threads to use per block
	const int tile_So2Ta = 16; // half warp
	const int ntpb_So2Ta = tile_So2Ta*tile_So2Ta;

	// magnetic field calculation
	__device__ cufltp4 so2ta_block_H(const cufltp4 &Rs, const cufltp4 &Ieff, const cufltp4 &Rt){
		// calculate relative position
		const cufltp4 dR = sub(Rt,Rs);

		// calculate distance between source and target
		const cufltp rho = sqrtf(dR.x*dR.x + dR.y*dR.y + dR.z*dR.z);

		// third power of distance
		const cufltp rho3 = rho*rho*rho;

		// get eps3
		const cufltp eps3 = Rs.w*Rs.w*Rs.w;

		// calculate effective current
		const cufltp sphere_ratio = min(rho3/eps3,RAT_CUCONST(1.0)); 
		
		// allocate output field
		cufltp4 H{RAT_CUCONST(0.0),RAT_CUCONST(0.0),RAT_CUCONST(0.0),RAT_CUCONST(0.0)};

		// calculate magnetic field
		if(rho>RAT_CUCONST(1e-7)){
			H = scale(cross(Ieff,dR),sphere_ratio*Ieff.w/(4*CU_PI*rho3));
		}

		// return field
		return H;
	}

	// magnetic field calculation 
	// for elements with finite length
	__device__ cufltp4 so2ta_block_H_vl(const cufltp4 &Rs, const cufltp4 &Ieff, const cufltp4 &Rt){
		// left and right end of source line
		const cufltp4 R1 = sub(Rs, scale(Ieff,RAT_CUCONST(0.5)));
		const cufltp4 R2 = add(Rs, scale(Ieff,RAT_CUCONST(0.5)));

		// calculate length
		const cufltp ell = sqrtf(dot(Ieff,Ieff));

		// Distance between left end source line and target point 
		const cufltp4 dR = sub(R1, Rt);

		// minimum distance between point and source line
		const cufltp q = norm(cross(scale(dR,RAT_CUCONST(-1.0)), sub(Rt,R2)))/ell;

		// Line parameter closest to target point
		const cufltp t0 = -dot(dR, Ieff)/(ell*ell);

		// Point on line closest to target point
		const cufltp4 R01 = add(R1,scale(Ieff,t0));

		// Sign of left end of source line 
		const cufltp p1 = (1 - 2*int(t0>=0))*norm(sub(R1,R01));
		const cufltp p2 = (1 - 2*int(t0>=1))*norm(sub(R2,R01));

		// Solution of Biot-Savart law, absolute value of magnetic field
		const cufltp Ba = p2/(q*sqrtf(q*q + p2*p2)) - p1/(q*sqrtf(q*q + p1*p1));

		// Field vector, to be normalised
		const cufltp4 v = cross(Ieff,dR);

		// norm of v
		const cufltp lv = norm(v);

		// softening
		cufltp fi = RAT_CUCONST(1.0);
		if(Rs.w>RAT_CUCONST(0.0)){
			// cylinder ratio
			fi = min(q*q/(Rs.w*Rs.w),RAT_CUCONST(1.0));
		}

		// allocate output field
		cufltp4 H{RAT_CUCONST(0.0),RAT_CUCONST(0.0),RAT_CUCONST(0.0),RAT_CUCONST(0.0)};

		// check for self field
		if(q>RAT_CUCONST(1e-3)*ell)H = scale(v,-fi*Ieff.w*Ba/(lv*4*CU_PI));

		// return field
		return H;
	}


	// vector potential calculation
	__device__ cufltp4 so2ta_block_A(const cufltp4 &Rs, const cufltp4 &Ieff, const cufltp4 &Rt){
		// calculate relative position
		const cufltp4 dR = sub(Rt,Rs);

		// calculate distance between source and target
		const cufltp rho = sqrtf(dR.x*dR.x + dR.y*dR.y + dR.z*dR.z);

		// third power of distance
		const cufltp rho3 = rho*rho*rho;

		// get softening factor
		const cufltp eps3 = Rs.w*Rs.w*Rs.w;

		// calculate effective current
		const cufltp sphere_ratio = min(rho3/eps3,RAT_CUCONST(1.0)); 
		
		// allocate output field
		cufltp4 A{RAT_CUCONST(0.0),RAT_CUCONST(0.0),RAT_CUCONST(0.0),RAT_CUCONST(0.0)};

		// calculate magnetic field
		if(rho>RAT_CUCONST(1e-7)){
			const cufltp scale = RAT_CUCONST(1e-7)*Ieff.w*sphere_ratio/rho;
			A.x = Ieff.x*scale; A.y = Ieff.y*scale;	A.z = Ieff.z*scale; A.w = 0;
		}

		// return vector potential
		return A;
	}

	// vector potential calculation taking into 
	// account finite length of the elements (van Lanen)
	__device__ cufltp4 so2ta_block_A_vl(const cufltp4 &Rs, const cufltp4 &Ieff, const cufltp4 &Rt){
		// calculate Rvector
		const cufltp4 dR = sub(Rt, Rs); // xT-xS

		// calculate parameters
		const cufltp a = dot(Ieff,Ieff);
		const cufltp b = abs(RAT_CUCONST(2.0) * dot(Ieff, dR)); // abs value prevents a singularity see notes Ezra
		const cufltp c = dot(dR,dR);

		// subexpression elimination in the compiler will hopefully eliminate all cufltp calculations here
		const cufltp k1 = (b+a)/(RAT_CUCONST(2.0)*sqrtf(a))+sqrtf(a/RAT_CUCONST(4.0) + b/RAT_CUCONST(2.0) + c); 
		const cufltp k2 = (b-a)/(RAT_CUCONST(2.0)*sqrtf(a))+sqrtf(a/RAT_CUCONST(4.0) - b/RAT_CUCONST(2.0) + c);

		// calculate distance between source and target
		const cufltp rho = sqrtf(c);

		// third power of distance
		const cufltp rho3 = c*rho;

		// get softening factor
		const cufltp eps3 = Rs.w*Rs.w*Rs.w;

		// calculate effective current
		const cufltp sphere_ratio = min(rho3/eps3,RAT_CUCONST(1.0)); 

		// calculate A and return
		cufltp4 A{RAT_CUCONST(0.0),RAT_CUCONST(0.0),RAT_CUCONST(0.0),RAT_CUCONST(0.0)};

		// calculate vector potential
		//if(rho>RAT_CONST(1e-9)){
		if(k1>RAT_CUCONST(1e-7) && k2>RAT_CUCONST(1e-7) && rho>RAT_CUCONST(1e-7)){
			// calculate factor
			const cufltp scale = RAT_CUCONST(1e-7)*(logf(k1/k2)/sqrtf(a))*sphere_ratio*Ieff.w;

			// calcultae vector potential
			A.x = Ieff.x*scale; A.y = Ieff.y*scale;	A.z = Ieff.z*scale; A.w = RAT_CUCONST(0.0);
		}

		// return vector potential
		return A;
	}


	// Source to Target Kernel for 
	// the calculation of magnetic field
	__global__ void so2ta_gpu_kernel(
		cufltp4 *gl_M, const cufltp4 *gl_Rs, const cufltp4 *gl_Ieff, const cufltp4 *gl_Rt,
		const long unsigned int *gl_target_list, 
		const long unsigned int *gl_source_list_idx, const long unsigned int *gl_source_list, 
		const long unsigned int *gl_first_source, const long unsigned int *gl_last_source, 
		const long unsigned int *gl_first_target, const long unsigned int *gl_last_target,
		const bool vector_potential, const bool van_lanen){

		// allocate shared memory banks
		// for storing coordinates and output field
		__shared__ cufltp4 sh_M[ntpb_So2Ta]; // stores output field in shared memory
		__shared__ cufltp4 sh_Rt[tile_So2Ta]; // stores targets in shared memory
		__shared__ cufltp4 sh_Rs[tile_So2Ta]; // stores sources in shared memory
		__shared__ cufltp4 sh_Ieff[tile_So2Ta]; // stores source currents in shared memory
		
		// get sources
		const long unsigned int myfsidx = warp_read(&gl_source_list_idx[blockIdx.x]);
		const long unsigned int mylsidx = warp_read(&gl_source_list_idx[blockIdx.x+1])-1;
		
		// calculate number of sources
		const long unsigned int num_source_list = mylsidx - myfsidx + 1;
		
		// get target node
		const long unsigned int mytnode = warp_read(&gl_target_list[blockIdx.x]);
		
		// get targets
		const long unsigned int myft = warp_read(&gl_first_target[mytnode]);
		const long unsigned int mylt = warp_read(&gl_last_target[mytnode]);
		
		// calculate number of targets
		const unsigned int num_targets = mylt - myft + 1;
		
		// calculate number of tiles
		const unsigned int ntt = (num_targets/tile_So2Ta) + 1;

		// synchronise before using shared memory
		__syncthreads();

		// walk over target tiles
		for(unsigned int j=0;j<ntt;j++){
			// set accumulated field to zero
			sh_M[threadIdx.x] = MAKE_FLOAT4(
				RAT_CUCONST(0.0),RAT_CUCONST(0.0),
				RAT_CUCONST(0.0),RAT_CUCONST(0.0));

			// calculate inter tile index
			const unsigned int itt = threadIdx.x%tile_So2Ta;

			// calculate target index
			const unsigned int mytidx = myft + j*tile_So2Ta + itt;

			// load target indexes into shared memory
			if(threadIdx.x<tile_So2Ta && mytidx<=mylt)
				sh_Rt[threadIdx.x] = gl_Rt[mytidx];

			// walk over source nodes
			for(unsigned int i=0;i<num_source_list;i++){
				// get node
				const long unsigned int mysnode = warp_read(&gl_source_list[myfsidx + i]);

				// get sources
				const long unsigned int myfs = warp_read(&gl_first_source[mysnode]); //warp_read(&);
				const long unsigned int myls = warp_read(&gl_last_source[mysnode]); //warp_read(&);

				// calculate number of sources
				const unsigned int num_sources = myls - myfs + 1;
		
				// calculate number of tiles
				const unsigned int nst = (num_sources/tile_So2Ta) + 1;

				// walk over source tiles
				for(unsigned int k=0;k<nst;k++){
					// calculate inter tile index
					const unsigned int its = threadIdx.x/tile_So2Ta;

					// calculate my source index
					const unsigned int mysidx = myfs + k*tile_So2Ta + its;

					// copy next sources into shared memory
					if(threadIdx.x<tile_So2Ta){
						// calculate which source to load
						const unsigned int mysidxld = myfs + k*tile_So2Ta + threadIdx.x;

						// check if in range
						if(mysidxld<=myls){
							// load source coordinate
							sh_Rs[threadIdx.x] = gl_Rs[mysidxld];

							// load effective current
							sh_Ieff[threadIdx.x] = gl_Ieff[mysidxld];
						}
					}

					// synchronise before starting calculation
					__syncthreads();

					// check if target index is in range
					if(mytidx<=mylt && mysidx<=myls){
						// calculate block of source to target 
						// interactions for vector potential
						if(vector_potential){
							// vector potential taking into account the 
							// finite length of the elements
							if(van_lanen){
								sh_M[threadIdx.x] = add(sh_M[threadIdx.x],
									so2ta_block_A_vl(sh_Rs[its], sh_Ieff[its], sh_Rt[itt]));
							}

							// normal vector potential
							else{
								sh_M[threadIdx.x] = add(sh_M[threadIdx.x],
									so2ta_block_A(sh_Rs[its], sh_Ieff[its], sh_Rt[itt]));
							}
						}
						
						// calculate block of source to target 
						// interactions for magnetic field
						else{
							// special magnetic field calculation
							if(van_lanen){
								sh_M[threadIdx.x] = add(sh_M[threadIdx.x],
									so2ta_block_H_vl(sh_Rs[its], sh_Ieff[its], sh_Rt[itt]));
							}

							// normal magnetic field calculation
							else{
								sh_M[threadIdx.x] = add(sh_M[threadIdx.x],
									so2ta_block_H(sh_Rs[its], sh_Ieff[its], sh_Rt[itt]));
							}
						}
					}			

					// synchronise before changing data
					__syncthreads();	
				}
			}

			// accumulate results
			if(threadIdx.x<tile_So2Ta && mytidx<=mylt){
				// walk over tile and add to first
				for(long unsigned int i=1;i<tile_So2Ta;i++){
					sh_M[threadIdx.x] = add(sh_M[threadIdx.x],sh_M[i*tile_So2Ta + threadIdx.x]);
				}

				// set results to global memory 
				gl_M[mytidx] = sh_M[threadIdx.x];
			}

			// // accumulate results
			// for(long unsigned int i=tile_So2Ta>>1;i>0;i>>=1){
			// 	if(threadIdx.x<i*tile_So2Ta){
			// 		sh_M[threadIdx.x] = add(sh_M[threadIdx.x], sh_M[i*tile_So2Ta + threadIdx.x]);
			// 	}

			// 	// synchronise
			// 	__syncthreads();
			// }

			// // set results to global memory 
			// if(threadIdx.x<tile_So2Ta && mytidx<=mylt)
			// 	gl_M[mytidx] = sh_M[threadIdx.x];

			// synchronise
			__syncthreads();
		}
	}

	// cublas multipole to localpole translation kernel
	void GpuKernels::so2ta_kernel(
		cufltp *A_ptr, 
		cufltp *H_ptr, 
		const cufltp *Rt_ptr,
		const long unsigned int num_targets,
		const cufltp *Rs_ptr, 
		const cufltp *Ieff_ptr, 
		const long unsigned int num_sources,
		const long unsigned int* first_target_ptr,
		const long unsigned int* last_target_ptr,
		const long unsigned int num_target_nodes,
		const long unsigned int* first_source_ptr,
		const long unsigned int* last_source_ptr,
		const long unsigned int num_source_nodes,
		const long unsigned int* target_list_ptr,
		const long unsigned int* source_list_idx_ptr,
		const long unsigned int num_target_list,
		const long unsigned int* source_list_ptr,
		const long unsigned int num_source_list,
		const bool calc_A, const bool calc_H, const bool vl,
		const std::set<int> &gpu_devices, 
		const bool is_managed){

		// settings
		const long unsigned int num_blocks = 2048;
		const long unsigned int num_launches = (num_target_list-1)/num_blocks + 1;

		// get number of available CPU cores
		// int num_cpus = std::thread::hardware_concurrency();
		std::list<std::future<void> > futures;

		// make index
		std::atomic<int> idx; idx = 0;

		// mutex
		std::mutex lock;

		// walk over available cpus
		for(auto it = gpu_devices.begin();it!=gpu_devices.end();it++){
			// get device index
			const int gpu = *it;

			// start new thread and capture device index
			futures.push_back(std::async(std::launch::async,[&,gpu](){
				// set gpu device
				gpuErrchk(cudaSetDevice(gpu));

				// create stream
				cudaStream_t stream;
				gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

				// allocate pointers
				cufltp4 *Rs_dev, *Rt_dev, *Ieff_dev, *H_dev, *A_dev;
				long unsigned int *target_list_dev, *first_target_dev, *last_target_dev;
				long unsigned int *first_source_dev, *last_source_dev, *source_list_idx_dev, *source_list_dev;

				// use managed arrays
				if(is_managed){
					Rs_dev = (cufltp4*)Rs_ptr; Ieff_dev = (cufltp4*)Ieff_ptr; Rt_dev = (cufltp4*)Rt_ptr;
					A_dev = (cufltp4*)A_ptr; H_dev = (cufltp4*)H_ptr;
				}

				// copy arrays to GPU
				else{
					gpuErrchk(cudaMalloc(&Rs_dev, num_sources*sizeof(cufltp4))); 
					gpuErrchk(cudaMemcpyAsync(Rs_dev, Rs_ptr, num_sources*sizeof(cufltp4), cudaMemcpyHostToDevice,stream));
					
					gpuErrchk(cudaMalloc(&Ieff_dev, num_sources*sizeof(cufltp4))); 
					gpuErrchk(cudaMemcpyAsync(Ieff_dev, Ieff_ptr, num_sources*sizeof(cufltp4), cudaMemcpyHostToDevice,stream));
					
					gpuErrchk(cudaMalloc(&Rt_dev, num_targets*sizeof(cufltp4))); 
					gpuErrchk(cudaMemcpyAsync(Rt_dev, Rt_ptr, num_targets*sizeof(cufltp4), cudaMemcpyHostToDevice,stream));

					// allocate output vector potential and magnetic field
					if(calc_A){gpuErrchk(cudaMalloc(&A_dev, num_targets*sizeof(cufltp4))); gpuErrchk(cudaMemsetAsync(A_dev, 0, num_targets*sizeof(cufltp4),stream));}
					if(calc_H){gpuErrchk(cudaMalloc(&H_dev, num_targets*sizeof(cufltp4))); gpuErrchk(cudaMemsetAsync(H_dev, 0, num_targets*sizeof(cufltp4),stream));}

				}

				gpuErrchk(cudaMalloc(&first_target_dev, num_target_nodes*sizeof(long unsigned int))); 
				gpuErrchk(cudaMemcpyAsync(first_target_dev, first_target_ptr, num_target_nodes*sizeof(long unsigned int), cudaMemcpyHostToDevice,stream));

				gpuErrchk(cudaMalloc(&last_target_dev, num_target_nodes*sizeof(long unsigned int))); 
				gpuErrchk(cudaMemcpyAsync(last_target_dev, last_target_ptr, num_target_nodes*sizeof(long unsigned int), cudaMemcpyHostToDevice,stream));

				gpuErrchk(cudaMalloc(&first_source_dev, num_source_nodes*sizeof(long unsigned int))); 
				gpuErrchk(cudaMemcpyAsync(first_source_dev, first_source_ptr, num_source_nodes*sizeof(long unsigned int), cudaMemcpyHostToDevice,stream));

				gpuErrchk(cudaMalloc(&last_source_dev, num_source_nodes*sizeof(long unsigned int))); 
				gpuErrchk(cudaMemcpyAsync(last_source_dev, last_source_ptr, num_source_nodes*sizeof(long unsigned int), cudaMemcpyHostToDevice,stream));

				gpuErrchk(cudaMalloc(&target_list_dev, num_target_list*sizeof(long unsigned int))); 
				gpuErrchk(cudaMemcpyAsync(target_list_dev, target_list_ptr, num_target_list*sizeof(long unsigned int), cudaMemcpyHostToDevice,stream));

				gpuErrchk(cudaMalloc(&source_list_idx_dev, (num_target_list+1)*sizeof(long unsigned int))); 
				gpuErrchk(cudaMemcpyAsync(source_list_idx_dev, source_list_idx_ptr, (num_target_list+1)*sizeof(long unsigned int), cudaMemcpyHostToDevice,stream));
				
				gpuErrchk(cudaMalloc(&source_list_dev, num_source_list*sizeof(long unsigned int))); 
				gpuErrchk(cudaMemcpyAsync(source_list_dev, source_list_ptr, num_source_list*sizeof(long unsigned int), cudaMemcpyHostToDevice,stream));

				
				// create result matrix on host
				cufltp *AH_host;
				gpuErrchk(cudaMallocHost(&AH_host, num_targets*sizeof(cufltp4))); 

				// synchronise stream
				cudaStreamSynchronize(stream);

				// keep firing tasks untill run out of idx
				for(;;){
					// atomic increment of index so that no two threads can do same task
					const long unsigned int i = idx++; 

					// check if last index
					if(i>=num_launches)break;

					// number of threads
					const long unsigned int num_thread_blocks = std::min(num_blocks, num_target_list - i*num_blocks);

					// calculate offset
					const long unsigned int offset = i*num_blocks;

					// calculate vector potential
					if(calc_A){
						// launch kernel for vector potential
						so2ta_gpu_kernel<<<num_thread_blocks, ntpb_So2Ta, 0, stream>>>(
							A_dev, Rs_dev, Ieff_dev, Rt_dev, 
							target_list_dev+offset, 
							source_list_idx_dev+offset, 
							source_list_dev, 
							first_source_dev, last_source_dev, 
							first_target_dev, last_target_dev, true, vl);
					}

					// calculate magnetic field
					if(calc_H){
						// launch kernel for magnetic field
						so2ta_gpu_kernel<<<num_thread_blocks, ntpb_So2Ta, 0, stream>>>(
							H_dev, Rs_dev, Ieff_dev, Rt_dev, 
							target_list_dev+offset, 
							source_list_idx_dev+offset, 
							source_list_dev, 
							first_source_dev, last_source_dev, 
							first_target_dev, last_target_dev, false, vl);
					}

					// synchronise stream after memcopies
					gpuErrchk(cudaStreamSynchronize(stream));
					gpuErrchk(cudaGetLastError());
				}

				// get fields from GPU
				if(calc_A && !is_managed){
					// copy vector potential
					gpuErrchk(cudaMemcpyAsync(AH_host, A_dev, num_targets*sizeof(cufltp4), cudaMemcpyDeviceToHost, stream));

					// synchronise stream after memcopies
					gpuErrchk(cudaStreamSynchronize(stream));

					// add values
					lock.lock();
					for(long long unsigned int j=0;j<num_targets*4;j++)A_ptr[j] += AH_host[j];
					lock.unlock();

					// free A
					gpuErrchk(cudaFree(A_dev));
				}

				if(calc_H && !is_managed){
					// synchronise stream after memcopies
					gpuErrchk(cudaStreamSynchronize(stream));

					// copy vector potential
					gpuErrchk(cudaMemcpyAsync(AH_host, H_dev, num_targets*sizeof(cufltp4), cudaMemcpyDeviceToHost, stream));

					// synchronise stream after memcopies
					gpuErrchk(cudaStreamSynchronize(stream));

					// add values
					lock.lock();
					for(long long unsigned int j=0;j<num_targets*4;j++)H_ptr[j] += AH_host[j];
					lock.unlock();

					// free H
					gpuErrchk(cudaFree(H_dev));
				}


				// synchronise stream
				cudaStreamSynchronize(stream);

				// free host
				gpuErrchk(cudaFreeHost(AH_host));

				// free cuda
				if(!is_managed){
					gpuErrchk(cudaFree(Rs_dev));
					gpuErrchk(cudaFree(Rt_dev));
					gpuErrchk(cudaFree(Ieff_dev));
				}

				// free index matrices
				gpuErrchk(cudaFree(target_list_dev));
				gpuErrchk(cudaFree(first_target_dev));
				gpuErrchk(cudaFree(last_target_dev));
				gpuErrchk(cudaFree(first_source_dev));
				gpuErrchk(cudaFree(last_source_dev));
				gpuErrchk(cudaFree(source_list_idx_dev));
				gpuErrchk(cudaFree(source_list_dev));

				
				// clean up streams
				gpuErrchk(cudaStreamDestroy(stream));
			}));
		}

		// make sure all threads are finished
		for(auto it=futures.begin();it!=futures.end();it++)(*it).get();
	}


	// number of threads per block
	const int ntpb_direct = 256;

	// direct kernel
	// the use of shared memory was inspired by the 
	// stellar dynamics example in the CUDA toolkit
	__global__ void direct_gpu_kernel(
		cufltp4 *gl_M, const cufltp4 *gl_Rs, const cufltp4 *gl_Ieff, const cufltp4 *gl_Rt,
		const long unsigned int num_targets, const long unsigned int num_sources, 
		const long unsigned int target_offset, const bool vector_potential, const bool van_lanen){

		// create shared memory
		__shared__ cufltp4 sh_Rs[ntpb_direct];
		__shared__ cufltp4 sh_Ieff[ntpb_direct];

		// calculate index of this thread
		const long unsigned int mytarget = target_offset + 
			blockDim.x*blockIdx.x + threadIdx.x;

		// allocate stored field
		// we are always using double precision for the accumulation of field
		// as the rounding errors stack up for each source
		// cufltp4 lo_M = MAKE_FLOAT4(
		// 	RAT_CUCONST(0.0),RAT_CUCONST(0.0),
		// 	RAT_CUCONST(0.0),RAT_CUCONST(0.0));
		double4 lo_M = make_double4(
			RAT_CUCONST(0.0),RAT_CUCONST(0.0),
		 	RAT_CUCONST(0.0),RAT_CUCONST(0.0));

		// allocate target position
		cufltp4 lo_Rt;

		// find my position and current vector
		if(mytarget<num_targets)lo_Rt = gl_Rt[mytarget];

		// calculate number of tiles
		const long unsigned int num_tiles = (num_sources-1)/ntpb_direct + 1;

		// walk over tiles
		for(long unsigned int i = 0; i<num_tiles; i++){
			// calculate the index of this source
			unsigned long int mysource = i*ntpb_direct + threadIdx.x;

			// load source from global to shared memory
			if(mysource<num_sources){
				sh_Rs[threadIdx.x] = gl_Rs[mysource];
				sh_Ieff[threadIdx.x] = gl_Ieff[mysource];
			}

			// synchronise
			__syncthreads();

			// walk over loaded positions
			for(long unsigned int j=0;j<ntpb_direct;j++){
				// calculate contribution of this source
				if(i*ntpb_direct+j<num_sources && mytarget<num_targets){
					// calculate block of source to target vector potential
					if(vector_potential){
						// vector potential taking into 
						// account finite length of the elements
						if(van_lanen){
							lo_M = add(lo_M, so2ta_block_A_vl(sh_Rs[j], sh_Ieff[j], lo_Rt));
						}

						// normal vector potential
						else{
							lo_M = add(lo_M, so2ta_block_A(sh_Rs[j], sh_Ieff[j], lo_Rt));
						}
					}
					
					// calculate source to target for magnetic field
					else{
						// special magnetic field
						if(van_lanen){
							lo_M = add(lo_M, so2ta_block_H_vl(sh_Rs[j], sh_Ieff[j], lo_Rt));
						}

						// normal magnetic field
						else{
							lo_M = add(lo_M, so2ta_block_H(sh_Rs[j], sh_Ieff[j], lo_Rt));
						}
					}
				}	
			}

			// synchronise
			__syncthreads();
		}

		// store in global memory
		if(mytarget<num_targets)gl_M[mytarget] = MAKE_FLOAT4(lo_M.x,lo_M.y,lo_M.z,lo_M.w);
	}


	// cublas multipole to localpole translation kernel
	void GpuKernels::direct_kernel(
		cufltp *A_ptr, cufltp *H_ptr, const cufltp *Rt_ptr,
		const long unsigned int num_targets, const cufltp *Rs_ptr, 
		const cufltp *Ieff_ptr, const long unsigned int num_sources,
		const bool calc_A, const bool calc_H, const bool vl, 
		const std::set<int> &gpu_devices, const bool is_managed){

		// set gpu device
		gpuErrchk(cudaSetDevice(*gpu_devices.begin()));

		// create stream
		cudaStream_t stream1;
		gpuErrchk(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
		cudaStream_t stream2;
		gpuErrchk(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

		// pointers
		cufltp4 *Rs_dev, *Rt_dev, *Ieff_dev, *H_dev, *A_dev;
		
		// use pointers directly for managed memory
		if(is_managed){
			Rs_dev = (cufltp4*)Rs_ptr; Rt_dev = (cufltp4*)Rt_ptr; 
			Ieff_dev = (cufltp4*)Ieff_ptr; A_dev = (cufltp4*)A_ptr; 
			H_dev = (cufltp4*)H_ptr;
		}

		// memcopy to device
		else{
			// copy arrays to GPU
			gpuErrchk(cudaMalloc(&Rs_dev, num_sources*sizeof(cufltp4))); 
			gpuErrchk(cudaMemcpyAsync(Rs_dev, Rs_ptr, num_sources*sizeof(cufltp4), cudaMemcpyHostToDevice, stream1));
			
			gpuErrchk(cudaMalloc(&Ieff_dev, num_sources*sizeof(cufltp4))); 
			gpuErrchk(cudaMemcpyAsync(Ieff_dev, Ieff_ptr, num_sources*sizeof(cufltp4), cudaMemcpyHostToDevice, stream1));
			
			gpuErrchk(cudaMalloc(&Rt_dev, num_targets*sizeof(cufltp4))); 
			gpuErrchk(cudaMemcpyAsync(Rt_dev, Rt_ptr, num_targets*sizeof(cufltp4), cudaMemcpyHostToDevice, stream1));

			// allocate output vector potential and magnetic field
			if(calc_A)gpuErrchk(cudaMalloc(&A_dev, num_targets*sizeof(cufltp4))); 
			if(calc_H)gpuErrchk(cudaMalloc(&H_dev, num_targets*sizeof(cufltp4))); 

			// sync
			cudaStreamSynchronize(stream1);
			cudaStreamSynchronize(stream2);
		}

		// number of thread blocks
		const long unsigned int num_blocks = (num_targets-1)/ntpb_direct + 1;

		// calculate vector potential
		if(calc_A){
			// launch kernel for vector potential
			direct_gpu_kernel<<<num_blocks, ntpb_direct, 0, stream1>>>(
				(cufltp4*)A_dev, (cufltp4*)Rs_dev, (cufltp4*)Ieff_dev, 
				(cufltp4*)Rt_dev, num_targets, num_sources, 0, true, vl);
		}

		// calculate magnetic field
		if(calc_H){
			// launch kernel for vector potential
			direct_gpu_kernel<<<num_blocks, ntpb_direct, 0, stream2>>>(
				(cufltp4*)H_dev, (cufltp4*)Rs_dev, (cufltp4*)Ieff_dev, 
				(cufltp4*)Rt_dev, num_targets, num_sources, 0, false, vl);
		}

		// copy result back to CPU
		if(!is_managed){
			// copy vector potential
			if(calc_A){
				cudaStreamSynchronize(stream1);
				gpuErrchk(cudaMemcpyAsync(A_ptr, A_dev, num_targets*sizeof(cufltp4), cudaMemcpyDeviceToHost, stream1));
			}

			// copy magnetic field
			if(calc_H){
				cudaStreamSynchronize(stream2);
				gpuErrchk(cudaMemcpyAsync(H_ptr, H_dev, num_targets*sizeof(cufltp4), cudaMemcpyDeviceToHost, stream2));
			}
		}

		// sync stream
		cudaStreamSynchronize(stream1);
		cudaStreamSynchronize(stream2);
		gpuErrchk(cudaGetLastError());

		// destroy
		if(!is_managed){
			gpuErrchk(cudaFree(Rs_dev));
			gpuErrchk(cudaFree(Rt_dev));
			gpuErrchk(cudaFree(Ieff_dev));
			if(calc_A)gpuErrchk(cudaFree(A_dev));
			if(calc_H)gpuErrchk(cudaFree(H_dev));
		}
		
		// clean up streams
		gpuErrchk(cudaStreamDestroy(stream1));
		gpuErrchk(cudaStreamDestroy(stream2));
		// gpuErrchk(cudaStreamSynchronize(0)); 
	}

	// number of threads for test kernel
	const int ntpb_add = 512;

	// test gpu kernel
	__global__ void add_gpu_kernel(
		cufltp* A, cufltp *B, unsigned int nn){
		unsigned int id = blockIdx.x*ntpb_add + threadIdx.x;
		if(id<nn)A[id] += B[id];
	}

	// simple test kernel
	void GpuKernels::test_kernel(cufltp* A, cufltp *B, unsigned int nn){
		// calculate required number of threadblocks
		unsigned int num_block = nn/ntpb_add+1;

		// run kernel
		add_gpu_kernel<<<num_block, ntpb_add>>>(A,B,nn);
		
		// synchronise to sync CPU and GPU memory
		gpuErrchk(cudaDeviceSynchronize());	
	}

	// get unified memory bank
	void GpuKernels::create_cuda_managed(void** ptr, size_t size){
		gpuErrchk(cudaMallocManaged(ptr, size));
	}

	// free unified memory bank
	void GpuKernels::free_cuda_managed(void* ptr){
		gpuErrchk(cudaFree(ptr));
	}
	
	// create pinned host memory
	void GpuKernels::create_pinned(void ** ptr, size_t size){
		gpuErrchk(cudaMallocHost(ptr, size));
	}

	// free pinned host memory
	void GpuKernels::free_pinned(void* ptr){
		gpuErrchk(cudaFreeHost(ptr));
	}

	// create memory on GPU
	void GpuKernels::data2gpu(void** ptr, void* data, size_t size){
		gpuErrchk(cudaMalloc(ptr, size)); 
		gpuErrchk(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));
	}
	
	// free memory on GPU
	void GpuKernels::free_gpu(void* ptr){
		gpuErrchk(cudaFree(ptr));
	}

	// // sorting
	// void GpuKernels::sort(unsigned int *data, int num_items){
	// 	// find bits
	// 	int begin_bit = 0;
	// 	int end_bit = sizeof(unsigned int)*8;

	// 	// create device pointers
	// 	unsigned int *dev_data_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
	// 	unsigned int *dev_data_out;     // e.g., [        ...        ]

	// 	// copy data to gpu
	// 	gpuErrchk(cudaMalloc(&dev_data_in, num_items*sizeof(unsigned int))); 
	// 	gpuErrchk(cudaMemcpy(dev_data_in, data, num_items*sizeof(unsigned int), cudaMemcpyHostToDevice));
		
	// 	// allocate output to gpu
	// 	gpuErrchk(cudaMalloc(&dev_data_out, num_items*sizeof(unsigned int)));
	// 	gpuErrchk(cudaMemcpy(dev_data_out, dev_data_in, num_items*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	// 	// determine storage requirement
	// 	void *d_temp_storage = NULL;
	// 	size_t temp_storage_bytes = 0;
	// 	gpuErrchk(cub::DeviceRadixSort::SortKeys(
	// 		d_temp_storage, temp_storage_bytes, 
	// 		dev_data_in,dev_data_out,num_items, 
	// 		begin_bit, end_bit));

	// 	// Allocate temporary storage
	// 	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// 	// sort
	// 	gpuErrchk(cub::DeviceRadixSort::SortKeys(
	// 		d_temp_storage, temp_storage_bytes, 
	// 		dev_data_in,dev_data_out,num_items, 
	// 		begin_bit, end_bit));

	// 	// copy result back to host
	// 	gpuErrchk(cudaMemcpy(data, dev_data_out, num_items*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// 	// free memory
	// 	gpuErrchk(cudaFree(dev_data_in));
	// 	gpuErrchk(cudaFree(dev_data_out));
	// 	gpuErrchk(cudaFree(d_temp_storage));
	// }

}}
