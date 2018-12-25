//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200
//	plasmaKernel_gpu_2
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200

__global__ void kernel_gpu_cuda(par_str d_par_gpu,
								dim_str d_dim_gpu,
								box_str* d_box_gpu,
								FOUR_VECTOR_HALF* d_rv_gpu,
								half* d_qv_gpu,
								FOUR_VECTOR_HALF* d_fv_gpu)
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
		__half a2 = __float2half(2.0*d_par_gpu.alpha*d_par_gpu.alpha);
		//__half a2 = __hmul(__hmul(alpha_temp, alpha_temp), __float2half(2.0));
		// home box
		int first_i;
		FOUR_VECTOR_HALF* rA;
		FOUR_VECTOR_HALF* fA;
		__shared__ FOUR_VECTOR_HALF rA_shared[100];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR_HALF* rB;
		half* qB;
		int j = 0;
		__shared__ FOUR_VECTOR_HALF rB_shared[100];
		__shared__ half qB_shared[100];

		// common
		//~ fp r2;
		//~ fp u2;
		//~ fp vij;
		//~ fp fs;
		//~ fp fxij;
		//~ fp fyij;
		//~ fp fzij;
		
		half r2;
		__half u2;
		__half vij;
		__half fs;
		half fxij;
		half fyij;
		__half fzij;
		
		THREE_VECTOR_HALF d;

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
		//~ k=0;{
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
			FOUR_VECTOR_HALF fA_tmp;
			
			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			//~ {
			while(wtx<NUMBER_PAR_PER_BOX){
				fA_tmp.v  = fA[wtx].v;
				fA_tmp.x  = fA[wtx].x;
				fA_tmp.y  = fA[wtx].y;
				fA_tmp.z  = fA[wtx].z;
				
				// loop for the number of particles in the current nei box
				//~ j=0;{
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



					r2 = rA_shared[wtx].v + rB_shared[j].v - DOT(rA_shared[wtx],rB_shared[j]); 
					u2 = a2*r2;
					vij= exp(-u2);
					fs = 2*vij;

					d.x = rA_shared[wtx].x  - rB_shared[j].x;
					fxij=fs*d.x;
					d.y = rA_shared[wtx].y  - rB_shared[j].y;
					fyij=fs*d.y;
					d.z = rA_shared[wtx].z  - rB_shared[j].z;
					fzij=fs*d.z;

					//~ fA[wtx].v += (qB_shared[j]*vij);
					//~ fA[wtx].x += (qB_shared[j]*fxij);
					//~ fA[wtx].y += (qB_shared[j]*fyij);
					//~ fA[wtx].z += (qB_shared[j]*fzij);

					fA_tmp.v += (qB_shared[j]*vij);
					fA_tmp.x += (qB_shared[j]*fxij);
					fA_tmp.y += (qB_shared[j]*fyij);
					fA_tmp.z += (qB_shared[j]*fzij);
					//~ fA[wtx].v += rB_shared[j].v;
					//~ fA[wtx].x += rB_shared[j].x;
					//~ fA[wtx].y += rB_shared[j].y;
					//~ fA[wtx].z += rB_shared[j].z;


				}
				fA[wtx].v = fA_tmp.v ;
				fA[wtx].x = fA_tmp.x ;
				fA[wtx].y = fA_tmp.y ;
				fA[wtx].z = fA_tmp.z ;
				

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
