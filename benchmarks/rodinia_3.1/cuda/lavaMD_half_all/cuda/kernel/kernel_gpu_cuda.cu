//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200
//	plasmaKernel_gpu_2
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200

__global__ void kernel_gpu_cuda(par_str d_par_gpu,
								dim_str d_dim_gpu,
								box_str* d_box_gpu,
								FOUR_VECTOR* d_rv_gpu,
								fp* d_qv_gpu,
								FOUR_VECTOR* d_fv_gpu)
{

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	THREAD PARAMETERS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

	int bx = blockIdx.x;																// get current horizontal block index (0-n)
	int tx = threadIdx.x;															// get current horizontal thread index (0-n)
	// int ax = bx*NUMBER_THREADS+tx;
	// int wbx = bx;
	int wtx = tx;
	__half alpha_temp = __float2half(d_par_gpu.alpha);
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	DO FOR THE NUMBER OF BOXES
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

	if(bx<d_dim_gpu.number_boxes){
	// while(wbx<box_indexes_counter){

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Extract input parameters
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// parameters
		//__half a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;
		__half a2 = __hmul(__hmul(alpha_temp, alpha_temp), __float2half(2.0));
		// home box
		int first_i;
		FOUR_VECTOR* rA;
		FOUR_VECTOR* fA;
		__shared__ FOUR_VECTOR rA_shared[100];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR* rB;
		fp* qB;
		int j = 0;
		__shared__ FOUR_VECTOR rB_shared[100];
		__shared__ fp qB_shared[100];

		// common
		//~ fp r2;
		//~ fp u2;
		//~ fp vij;
		//~ fp fs;
		//~ fp fxij;
		//~ fp fyij;
		//~ fp fzij;
		
		fp r2;
		__half u2;
		__half vij;
		__half fs;
		fp fxij;
		fp fyij;
		__half fzij;
		
		THREE_VECTOR d;

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Home box
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Setup parameters
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];
		
		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Copy to shared memory
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX){
			rA_shared[wtx] = rA[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// loop over neiing boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++){

			//----------------------------------------50
			//	nei box - get pointer to the right box
			//----------------------------------------50

			if(k==0){
				pointer = bx;													// set first box to be processed to home box
			}
			else{
				pointer = d_box_gpu[bx].nei[k-1].number;							// remaining boxes are nei boxes
			}

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			// nei box - shared memory
			while(wtx<NUMBER_PAR_PER_BOX){
				rB_shared[wtx] = rB[wtx];
				qB_shared[wtx] = qB[wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation
			//----------------------------------------------------------------------------------------------------------------------------------140

			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			while(wtx<NUMBER_PAR_PER_BOX){

				// loop for the number of particles in the current nei box
				for (j=0; j<NUMBER_PAR_PER_BOX; j++){

					// r2 = rA[wtx].v + rB[j].v - DOT(rA[wtx],rB[j]); 
					// u2 = a2*r2;
					// vij= exp(-u2);
					// fs = 2.*vij;

					// d.x = rA[wtx].x  - rB[j].x;
					// fxij=fs*d.x;
					// d.y = rA[wtx].y  - rB[j].y;
					// fyij=fs*d.y;
					// d.z = rA[wtx].z  - rB[j].z;
					// fzij=fs*d.z;

					// fA[wtx].v +=  qB[j]*vij;
					// fA[wtx].x +=  qB[j]*fxij;
					// fA[wtx].y +=  qB[j]*fyij;
					// fA[wtx].z +=  qB[j]*fzij;



					r2 = (fp)rA_shared[wtx].v + (fp)rB_shared[j].v - DOT((fp)rA_shared[wtx],(fp)rB_shared[j]); 
					//~ u2 = a2*r2;
					u2 = __hmul(a2, __float2half(r2));
					
					//~ vij= exp(-u2);
					vij = hexp(__hneg(u2));
					
					//~ fs = 2*vij;
					fs = __hmul(vij,__float2half(2.0));

					d.x = (fp)rA_shared[wtx].x  - (fp)rB_shared[j].x;
					fxij=__half2float(fs)*d.x;
					d.y = (fp)rA_shared[wtx].y  - (fp)rB_shared[j].y;
					fyij=__half2float(fs)*d.y;
					d.z = (fp)rA_shared[wtx].z  - (fp)rB_shared[j].z;
					//~ fzij=fs*d.z;
					fzij= __hmul(fs, __float2half(d.z));
					fA[wtx].v += (fp)((fp)qB_shared[j]*__half2float(vij));
					fA[wtx].x += (fp)((fp)qB_shared[j]*fxij);
					fA[wtx].y += (fp)((fp)qB_shared[j]*fyij);
					fA[wtx].z += (fp)((fp)qB_shared[j]*__half2float(fzij));

				}

				// increment work thread index
				wtx = wtx + NUMBER_THREADS;

			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			__syncthreads();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation END
			//----------------------------------------------------------------------------------------------------------------------------------140

		}

		// // increment work block index
		// wbx = wbx + NUMBER_BLOCKS;

		// // synchronize - because next iteration will overwrite current shared memory
		// __syncthreads();

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

	}

}
