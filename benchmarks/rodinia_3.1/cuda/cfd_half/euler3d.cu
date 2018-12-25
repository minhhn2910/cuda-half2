// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

//#include <cutil.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>
#include "half_operator_overload.cuh"
#include "newhalf.hpp"

 
/*
 * Options 
 * 
 */ 
#define GAMMA 1.4f
#define iterations 2000
// #ifndef block_length
// 	#define block_length 192
// #endif



#define NDIM 3
//~ #define NNB 4 //rounding error accumulated with halfprecision , use NNB = 2 for correctness checking
#define NNB 2

#define RK 3	// 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

/*
 * not options
 */

#ifdef RD_WG_SIZE_0_0
	#define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
	#define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_0 RD_WG_SIZE
#else
	#define BLOCK_SIZE_0 192
#endif

#ifdef RD_WG_SIZE_1_0
	#define BLOCK_SIZE_1 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
	#define BLOCK_SIZE_1 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_1 RD_WG_SIZE
#else
	#define BLOCK_SIZE_1 192
#endif

#ifdef RD_WG_SIZE_2_0
	#define BLOCK_SIZE_2 RD_WG_SIZE_2_0
#elif defined(RD_WG_SIZE_1)
	#define BLOCK_SIZE_2 RD_WG_SIZE_2
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_2 RD_WG_SIZE
#else
	#define BLOCK_SIZE_2 192
#endif

#ifdef RD_WG_SIZE_3_0
	#define BLOCK_SIZE_3 RD_WG_SIZE_3_0
#elif defined(RD_WG_SIZE_3)
	#define BLOCK_SIZE_3 RD_WG_SIZE_3
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_3 RD_WG_SIZE
#else
	#define BLOCK_SIZE_3 192
#endif

#ifdef RD_WG_SIZE_4_0
	#define BLOCK_SIZE_4 RD_WG_SIZE_4_0
#elif defined(RD_WG_SIZE_4)
	#define BLOCK_SIZE_4 RD_WG_SIZE_4
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE_4 RD_WG_SIZE
#else
	#define BLOCK_SIZE_4 192
#endif



// #if block_length > 128
// #warning "the kernels may fail too launch on some systems if the block length is too large"
// #endif


#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


/*
 * Generic functions
 */
template <typename T>
T* alloc(int N)
{
	T* t;
	checkCudaErrors(cudaMalloc((void**)&t, sizeof(T)*N));
	return t;
}

template <typename T>
void dealloc(T* array)
{
	checkCudaErrors(cudaFree((void*)array));
}

template <typename T>
void copy(T* dst, T* src, int N)
{
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void upload(T* dst, T* src, int N)
{
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T> //for half
void upload(T* dst, half_float::half* src, int N)
{
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void download(T* dst, T* src, int N)
{
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void download(half_float::half* dst, T* src, int N) //for half
{
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToHost));
}

void dump(half* variables, int nel, int nelr)
{
	float* h_variables = new float[nelr*NVAR];
	half_float::half* h_variables_half = new half_float::half[nelr*NVAR];
	
	download<half>(h_variables_half, variables, nelr*NVAR);
	for(int i = 0;i<nelr*NVAR;i++){
		h_variables[i] = float(h_variables_half[i]);
		
		}

	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++)
				file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
			file << std::endl;
		}
	}
	
	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
	}
	delete[] h_variables;
	delete[] h_variables_half;
}

/*
 * Element-based Cell-centered FVM solver functions
 */
__constant__ half ff_variable[NVAR];
__constant__ half3 ff_flux_contribution_momentum_x[1];
__constant__ half3 ff_flux_contribution_momentum_y[1];
__constant__ half3 ff_flux_contribution_momentum_z[1];
__constant__ half3 ff_flux_contribution_density_energy[1];

__global__ void cuda_initialize_variables(int nelr, half* variables)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	for(int j = 0; j < NVAR; j++)
		variables[i + j*nelr] = ff_variable[j];
}
void initialize_variables(int nelr, half* variables)
{
	dim3 Dg(nelr / BLOCK_SIZE_1), Db(BLOCK_SIZE_1);
	cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
	getLastCudaError("initialize_variables failed");
}

__device__ inline void compute_flux_contribution(half& density, half3& momentum, half& density_energy, half& pressure, half3& velocity, half3& fc_momentum_x, half3& fc_momentum_y, half3& fc_momentum_z, half3& fc_density_energy)
{
	fc_momentum_x.x = velocity.x*momentum.x + pressure;
	fc_momentum_x.y = velocity.x*momentum.y;
	fc_momentum_x.z = velocity.x*momentum.z;
	
	
	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y*momentum.y + pressure;
	fc_momentum_y.z = velocity.y*momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z*momentum.z + pressure;

	half de_p = density_energy+pressure;
	fc_density_energy.x = velocity.x*de_p;
	fc_density_energy.y = velocity.y*de_p;
	fc_density_energy.z = velocity.z*de_p;
}
__host__ inline void compute_flux_contribution(half_float::half& density, half3_host& momentum, half_float::half& density_energy, half_float::half& pressure, half3_host& velocity, half3_host& fc_momentum_x, half3_host& fc_momentum_y, half3_host& fc_momentum_z, half3_host& fc_density_energy)
{
	fc_momentum_x.x = velocity.x*momentum.x + pressure;
	fc_momentum_x.y = velocity.x*momentum.y;
	fc_momentum_x.z = velocity.x*momentum.z;
	
	
	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y*momentum.y + pressure;
	fc_momentum_y.z = velocity.y*momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z*momentum.z + pressure;

	half_float::half de_p = density_energy+pressure;
	fc_density_energy.x = velocity.x*de_p;
	fc_density_energy.y = velocity.y*de_p;
	fc_density_energy.z = velocity.z*de_p;
}
__device__ inline void compute_velocity(half& density, half3& momentum, half3& velocity)
{
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}
	
__device__ inline half compute_speed_sqd(half3& velocity)
{
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

__device__ inline half compute_pressure(half& density, half& density_energy, half& speed_sqd)
{
	return (__float2half(GAMMA)-__float2half(1.0f))*(density_energy - __float2half(0.5f)*density*speed_sqd);
}

__device__ inline half compute_speed_of_sound(half& density, half& pressure)
{
	return sqrtf(__float2half(GAMMA)*pressure/density);
}

__global__ void cuda_compute_step_factor(int nelr, half* variables, half* areas, half* step_factors)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	half density = variables[i + VAR_DENSITY*nelr];
	half3 momentum;
	momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	
	half density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	
	half3 velocity;       compute_velocity(density, momentum, velocity);
	half speed_sqd      = compute_speed_sqd(velocity);
	half pressure       = compute_pressure(density, density_energy, speed_sqd);
	half speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
	step_factors[i] = __float2half(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}
void compute_step_factor(int nelr, half* variables, half* areas, half* step_factors)
{
	dim3 Dg(nelr / BLOCK_SIZE_2), Db(BLOCK_SIZE_2);
	cuda_compute_step_factor<<<Dg, Db>>>(nelr, variables, areas, step_factors);		
	getLastCudaError("compute_step_factor failed");
}

/*
 *
 *
*/
__global__ void cuda_compute_flux(int nelr, int* elements_surrounding_elements, half* normals, half* variables, half* fluxes)
{
	const half smoothing_coefficient = __float2half(0.2f);
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	
	int j, nb;
	half3 normal; half normal_len;
	half factor;
	
	half density_i = variables[i + VAR_DENSITY*nelr];
	half3 momentum_i;
	momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

	half density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

	half3 velocity_i;             				compute_velocity(density_i, momentum_i, velocity_i);
	half speed_sqd_i                          = compute_speed_sqd(velocity_i);
	half speed_i                              = sqrtf(speed_sqd_i);
	half pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	half speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
	half3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
	half3 flux_contribution_i_density_energy;	
	compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);
	
	half flux_i_density = __float2half(0.0f);
	half3 flux_i_momentum;
	flux_i_momentum.x = __float2half(0.0f);
	flux_i_momentum.y = __float2half(0.0f);
	flux_i_momentum.z = __float2half(0.0f);
	half flux_i_density_energy = __float2half(0.0f);
		
	half3 velocity_nb;
	half density_nb, density_energy_nb;
	half3 momentum_nb;
	half3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
	half3 flux_contribution_nb_density_energy;	
	half speed_sqd_nb, speed_of_sound_nb, pressure_nb;
	
	
	#pragma unroll
	for(j = 0; j < NNB; j++)
	//~ for(j = 1; j < 4; j++)
	{
		nb = elements_surrounding_elements[i + j*nelr];
		normal.x = normals[i + (j + 0*NNB)*nelr];
		normal.y = normals[i + (j + 1*NNB)*nelr];
		normal.z = normals[i + (j + 2*NNB)*nelr];
		normal_len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		
		if(nb >= 0) 	// a legitimate neighbor
		{
			density_nb = variables[nb + VAR_DENSITY*nelr];
			momentum_nb.x = variables[nb + (VAR_MOMENTUM+0)*nelr];
			momentum_nb.y = variables[nb + (VAR_MOMENTUM+1)*nelr];
			momentum_nb.z = variables[nb + (VAR_MOMENTUM+2)*nelr];
			density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
			//compute_velocity(density_nb, momentum_nb, velocity_nb);
			//calling inline functions is bad for automatic parsing, use below instead
			velocity_nb.x = momentum_nb.x / density_nb;
			velocity_nb.y = momentum_nb.y / density_nb;
			velocity_nb.z = momentum_nb.z / density_nb;	
			
			speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
			pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
			speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
//			                                    compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);
//compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);
////calling inline functions is bad for automatic parsing, use below instead
//compute_flux_contribution(half2& density, half2_3& momentum, half2& density_energy, half2& pressure, half2_3& velocity, half2_3& fc_momentum_x, half2_3& fc_momentum_y, half2_3& fc_momentum_z, half2_3& fc_density_energy)
			flux_contribution_nb_momentum_x.x = velocity_nb.x*momentum_nb.x + pressure_nb;
			flux_contribution_nb_momentum_x.y = velocity_nb.x*momentum_nb.y;
			flux_contribution_nb_momentum_x.z = velocity_nb.x*momentum_nb.z;
	
	
			flux_contribution_nb_momentum_y.x = flux_contribution_nb_momentum_x.y;
			flux_contribution_nb_momentum_y.y = velocity_nb.y*momentum_nb.y + pressure_nb;
			flux_contribution_nb_momentum_y.z = velocity_nb.y*momentum_nb.z;

			flux_contribution_nb_momentum_z.x = flux_contribution_nb_momentum_x.z;
			flux_contribution_nb_momentum_z.y = flux_contribution_nb_momentum_y.z;
			flux_contribution_nb_momentum_z.z = velocity_nb.z*momentum_nb.z + pressure_nb;

			half de_p = density_energy_nb+pressure_nb;
			flux_contribution_nb_density_energy.x = velocity_nb.x*de_p;
			flux_contribution_nb_density_energy.y = velocity_nb.y*de_p;
			flux_contribution_nb_density_energy.z = velocity_nb.z*de_p;	
				
			//end compute_flux_contribution
			

			
			// artificial viscosity
			factor = -normal_len*smoothing_coefficient*float(0.5f)*(speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
			flux_i_density += factor*(density_i-density_nb);
			flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
			flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
			flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
			flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

			// accumulate cell-centered fluxes
			factor = float(0.5f)*normal.x;
			flux_i_density += factor*(momentum_nb.x+momentum_i.x);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);
			
			factor = float(0.5f)*normal.y;
			flux_i_density += factor*(momentum_nb.y+momentum_i.y);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);
			
			factor = float(0.5f)*normal.z;
			flux_i_density += factor*(momentum_nb.z+momentum_i.z);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
		}
		else if(nb == -1)	// a wing boundary
		{
			flux_i_momentum.x += normal.x*pressure_i;
			flux_i_momentum.y += normal.y*pressure_i;
			flux_i_momentum.z += normal.z*pressure_i;
		}
		else if(nb == -2) // a far field boundary
		{
			factor = float(0.5f)*normal.x;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
			flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].x+flux_contribution_i_density_energy.x);
			flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].x + flux_contribution_i_momentum_x.x);
			flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].x + flux_contribution_i_momentum_y.x);
			flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].x + flux_contribution_i_momentum_z.x);
			
			factor = float(0.5f)*normal.y;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
			flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].y+flux_contribution_i_density_energy.y);
			flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].y + flux_contribution_i_momentum_x.y);
			flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].y + flux_contribution_i_momentum_y.y);
			flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].y + flux_contribution_i_momentum_z.y);

			factor = float(0.5f)*normal.z;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
			flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].z+flux_contribution_i_density_energy.z);
			flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].z + flux_contribution_i_momentum_x.z);
			flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].z + flux_contribution_i_momentum_y.z);
			flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].z + flux_contribution_i_momentum_z.z);

		}
		
	}

	fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
	fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
	fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
	fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
	fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}
void compute_flux(int nelr, int* elements_surrounding_elements, half* normals, half* variables, half* fluxes)
{
	dim3 Dg(nelr / BLOCK_SIZE_3), Db(BLOCK_SIZE_3);
	cuda_compute_flux<<<Dg,Db>>>(nelr, elements_surrounding_elements, normals, variables, fluxes);
	getLastCudaError("compute_flux failed");
}

__global__ void cuda_time_step(int j, int nelr, half* old_variables, half* variables, half* step_factors, half* fluxes)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	half factor = step_factors[i]/float(RK+1-j);

	variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
	variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
	variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
	variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
	//~ variables[i + (VAR_MOMENTUM+0)*nelr] = fluxes[i + (VAR_MOMENTUM+0)*nelr];// old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	//~ variables[i + (VAR_MOMENTUM+1)*nelr] = fluxes[i + (VAR_MOMENTUM+1)*nelr];//old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
	//~ variables[i + (VAR_MOMENTUM+2)*nelr] = fluxes[i + (VAR_MOMENTUM+2)*nelr];//old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
}
void time_step(int j, int nelr, half* old_variables, half* variables, half* step_factors, half* fluxes)
{
	dim3 Dg(nelr / BLOCK_SIZE_4), Db(BLOCK_SIZE_4);
	cuda_time_step<<<Dg,Db>>>(j, nelr, old_variables, variables, step_factors, fluxes);
	getLastCudaError("update failed");
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
  printf("WG size of kernel:initialize = %d, WG size of kernel:compute_step_factor = %d, WG size of kernel:compute_flux = %d, WG size of kernel:time_step = %d\n", BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3, BLOCK_SIZE_4);

	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];
	
	cudaDeviceProp prop;
	int dev;
	
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDevice(&dev));
	checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
	
	printf("Name:                     %s\n", prop.name);

	// set far field conditions and load them into constant memory on the gpu
	{
		half_float::half h_ff_variable[NVAR];
		const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);
		
		h_ff_variable[VAR_DENSITY] = half_float::half(1.4);
		
		half_float::half ff_pressure = half_float::half(1.0f);
		half_float::half ff_speed_of_sound = half_float::half(sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]));
		half_float::half ff_speed = half_float::half(ff_mach)*ff_speed_of_sound;
		
		half3_host ff_velocity;
		ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
		ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
		ff_velocity.z = 0.0f;
		
		h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
		h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
		h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;
				
		h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

		half3_host h_ff_momentum;
		h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
		h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
		h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
		half3_host h_ff_flux_contribution_momentum_x;
		half3_host h_ff_flux_contribution_momentum_y;
		half3_host h_ff_flux_contribution_momentum_z;
		half3_host h_ff_flux_contribution_density_energy;
		
		compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

		// copy far field conditions to the gpu
		checkCudaErrors( cudaMemcpyToSymbol(ff_variable,          h_ff_variable,          NVAR*sizeof(half)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(half3)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(half3)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(half3)) );
		
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(half3)) );		
	}
	int nel;
	int nelr;
	
	// read in domain geometry
	half* areas;
	half* normals;
	
	
	int* elements_surrounding_elements;
	//device mem 
	
	{
		std::ifstream file(data_file_name);
	
		file >> nel;
		nelr = BLOCK_SIZE_0*((nel / BLOCK_SIZE_0 )+ std::min(1, nel % BLOCK_SIZE_0));

		float* h_areas = new float[nelr];
		int* h_elements_surrounding_elements = new int[nelr*NNB];
		float* h_normals = new float[nelr*NDIM*NNB];

				
		// read in data
		for(int i = 0; i < nel; i++)
		{
			file >> h_areas[i];
			for(int j = 0; j < NNB; j++)
			{
				file >> h_elements_surrounding_elements[i + j*nelr];
				if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
				h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
				
				for(int k = 0; k < NDIM; k++)
				{
					file >> h_normals[i + (j + k*NNB)*nelr];
					h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
				}
			}
		}
		
		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			h_areas[i] = h_areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
				for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
			}
		}
		
		half_float::half* h_areas_half = new half_float::half[nelr];
		half_float::half* h_normals_half = new half_float::half[nelr*NDIM*NNB];
		
		for(int i= 0;i<nelr; i++){
			h_areas_half[i] = half_float::half(h_areas[i]);
			}
		for(int i=0;i<nelr*NDIM*NNB; i++){
			h_normals_half[i] = half_float::half(h_normals[i]);
			
			}
		
		
		areas = alloc<half>(nelr);
		
		upload<half>(areas, h_areas_half, nelr);

		elements_surrounding_elements = alloc<int>(nelr*NNB);
		
		upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

		normals = alloc<half>(nelr*NDIM*NNB);
		upload<half>(normals, h_normals_half, nelr*NDIM*NNB);
				
		delete[] h_areas;
		delete[] h_elements_surrounding_elements;
		delete[] h_normals;
		delete[] h_areas_half;
		delete[] h_normals_half;
	}

	// Create arrays and set initial conditions
	half* variables = alloc<half>(nelr*NVAR);
	initialize_variables(nelr, variables);

	half* old_variables = alloc<half>(nelr*NVAR);   	
	half* fluxes = alloc<half>(nelr*NVAR);
	half* step_factors = alloc<half>(nelr); 

	// make sure all memory is floatly allocated before we start timing
	initialize_variables(nelr, old_variables);
	initialize_variables(nelr, fluxes);
	
	cudaMemset( (void*) step_factors, 0, sizeof(half)*nelr );
	// make sure CUDA isn't still doing something before we start timing
	cudaThreadSynchronize();

	// these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;

	StopWatchInterface *timer = 0;
	  //	unsigned int timer = 0;

	// CUT_SAFE_CALL( cutCreateTimer( &timer));
	// CUT_SAFE_CALL( cutStartTimer( timer));
	sdkCreateTimer(&timer); 
	sdkStartTimer(&timer); 
	// Begin iterations
	for(int i = 0; i < iterations; i++)
	{
		copy<half>(old_variables, variables, nelr*NVAR);
		
		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors);
		getLastCudaError("compute_step_factor failed");
	
		//~ int j =0;
		for(int j = 0; j < RK; j++)
		{
			compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
			getLastCudaError("compute_flux failed");			
			time_step(j, nelr, old_variables, variables, step_factors, fluxes);
			getLastCudaError("time_step failed");			
		}
	}

	cudaThreadSynchronize();
	//	CUT_SAFE_CALL( cutStopTimer(timer) );  
	sdkStopTimer(&timer); 

	std::cout  << (sdkGetAverageTimerValue(&timer)/1000.0)  / iterations << " seconds per iteration" << std::endl;

	std::cout << "Saving solution..." << std::endl;
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;

	
	std::cout << "Cleaning up..." << std::endl;
	dealloc<half>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<half>(normals);
	
	dealloc<half>(variables);
	dealloc<half>(old_variables);
	dealloc<half>(fluxes);
	dealloc<half>(step_factors);

	std::cout << "Done..." << std::endl;

	return 0;
}
