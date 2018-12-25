// 14 APR 2011 Lukasz G. Szafaryn
#include <stdio.h>			// (in path known to compiler) needed by printf
#include <stdlib.h>			// (in path known to compiler) needed by malloc
#include <math.h>
#include <stdbool.h>			// (in path known to compiler) needed by true/false
#include "util/timer/timer.h"		// (in path specified here)
#include "util/num/num.h"		// (in path specified here)
#include "main.h"			// (in the current directory)
#include "kernel_gpu_cuda_wrapper.h"	// (in library path specified here)

#define EPSILON 0.001
int test_output(fp* fv_cpu, int num_space_elements, char *gold_file) {
  int i, check_v, check_x, check_y, check_z;
  FILE *gold_fp = fopen(gold_file, "r");
  float gold_v, gold_x, gold_y, gold_z;
  double max_rlt_err = 0.0;
  for (i = 0; i < 4 * num_space_elements; i+=4) {
    fscanf(gold_fp, "%f %f %f %f\n", &gold_v, &gold_x, &gold_y, &gold_z);
    if (gold_v != 0.0)
    if(fabs((gold_v - fv_cpu[i])/gold_v) > max_rlt_err)
		max_rlt_err = fabs((gold_v - fv_cpu[i])/gold_v);
    if (gold_x != 0.0)
   	if(fabs((gold_x - fv_cpu[i + 1])/gold_x) > max_rlt_err)
		max_rlt_err = fabs((gold_x - fv_cpu[i + 1])/gold_x);
    if (gold_y !=0.0)
    if(fabs((gold_y - fv_cpu[i + 2])/gold_y) > max_rlt_err)
		max_rlt_err = fabs((gold_y - fv_cpu[i + 2])/gold_y);
    if (gold_z != 0.0)
	if(fabs((gold_z - fv_cpu[i + 3])/gold_z) > max_rlt_err)
		max_rlt_err = fabs((gold_z - fv_cpu[i + 3])/gold_z);
    
 /*   check_v = ((gold_v - fv_cpu[i] >= -EPSILON) && (gold_v - fv_cpu[i] <= EPSILON));
    check_x = ((gold_x - fv_cpu[i + 1] >= -EPSILON) && (gold_x - fv_cpu[i + 1] <= EPSILON));
    check_y = ((gold_y - fv_cpu[i + 2] >= -EPSILON) && (gold_y - fv_cpu[i + 2] <= EPSILON));
    check_z = ((gold_z - fv_cpu[i + 3] >= -EPSILON) && (gold_z - fv_cpu[i + 3] <= EPSILON));
    if (!(check_v && check_x && check_y && check_z)) {
      printf("Result mismatched at element %d: %f %f %f %f vs %f %f %f %f\n",
          i, gold_v, gold_x, gold_y, gold_z,
          fv_cpu[i], fv_cpu[i + 1], fv_cpu[i + 2], fv_cpu[i + 3]);
		printf("max abs err %lf \n", max_abs_err);
		return 0;
    }
   */ 
    
  }
 
 printf("%lf,", max_rlt_err);
	
  fclose(gold_fp);
  return 1;
}

int main(int argc, char *argv [])
{
  // timer
  long long time0;
  time0 = get_time();
init_readconfig();
  // timer
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;
  long long time7;

  // counters
  int i, j, k, l, m, n;

  // system memory
  par_str par_cpu;
  dim_str dim_cpu;
  box_str* box_cpu;
  fp* rv_cpu;
  fp* qv_cpu;
  fp* fv_cpu;
  int nh;

  time1 = get_time();
  // assing default values
  dim_cpu.boxes1d_arg = 1;

  // go through arguments
  for (dim_cpu.cur_arg = 1; dim_cpu.cur_arg < argc; dim_cpu.cur_arg++) {
    // check if -boxes1d
    if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0) {
      // check if value provided
      if(argc >= dim_cpu.cur_arg + 1) {
        // check if value is a number
        if(isInteger(argv[dim_cpu.cur_arg+1]) == 1) {
          dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
          if(dim_cpu.boxes1d_arg < 0) {
            printf("ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
            return 0;
          }
          dim_cpu.cur_arg = dim_cpu.cur_arg+1;
        }
        // value is not a number
        else {
          printf("ERROR: Value to -boxes1d parameter in not a number\n");
          return 0;
        }
      }
      // value not provided
      else {
        printf("ERROR: Missing value to -boxes1d parameter\n");
        return 0;
      }
    }
    // unknown
    else {
      printf("ERROR: Unknown parameter\n");
      return 0;
    }
  }

  // Print configuration
  //printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

  time2 = get_time();
  par_cpu.alpha = 0.5;

  time3 = get_time();

  // total number of boxes
  dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

  // how many particles space has in each direction
  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
  dim_cpu.space_mem = dim_cpu.space_elem * 4 * sizeof(fp);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  // box array
  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);
  time4 = get_time();

  // allocate boxes
  box_cpu = (box_str*)malloc(dim_cpu.box_mem);

  // initialize number of home boxes
  nh = 0;

  // home boxes in z direction
  for(i=0; i<dim_cpu.boxes1d_arg; i++){
    // home boxes in y direction
    for(j=0; j<dim_cpu.boxes1d_arg; j++){
      // home boxes in x direction
      for(k=0; k<dim_cpu.boxes1d_arg; k++){

        // current home box
        box_cpu[nh].x = k;
        box_cpu[nh].y = j;
        box_cpu[nh].z = i;
        box_cpu[nh].number = nh;
        box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

        // initialize number of neighbor boxes
        box_cpu[nh].nn = 0;

        // neighbor boxes in z direction
        for (l = -1; l < 2; l++) {
          // neighbor boxes in y direction
          for (m = -1; m < 2; m++) {
            // neighbor boxes in x direction
            for (n = -1; n < 2; n++) {
              // check if (this neighbor exists) and (it is not the same as home box)
              if ((((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true &&
                    ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg &&
                     (k+n)<dim_cpu.boxes1d_arg)==true) &&
                  (l==0 && m==0 && n==0)==false) {

                // current neighbor box
                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                box_cpu[nh].nei[box_cpu[nh].nn].number = (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) +
                  (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) +
                  box_cpu[nh].nei[box_cpu[nh].nn].x;
                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                // increment neighbor box
                box_cpu[nh].nn = box_cpu[nh].nn + 1;

              }

            } // neighbor boxes in x direction
          } // neighbor boxes in y direction
        } // neighbor boxes in z direction

        // increment home box
        nh = nh + 1;

      } // home boxes in x direction
    } // home boxes in y direction
  } // home boxes in z direction
  // random generator seed set to random value - time in this case
  int seed = 7;
  srand(seed);

  // input (distances)
  rv_cpu = (fp*)malloc(dim_cpu.space_mem);
  for(i=0; i < 4 * dim_cpu.space_elem; i=i+4){
    rv_cpu[i] = (rand()%10 + 1) / 10.0;	// get a number in the range 0.1 - 1.0
    rv_cpu[i + 1] = (rand()%10 + 1) / 10.0;	// get a number in the range 0.1 - 1.0
    rv_cpu[i + 2] = (rand()%10 + 1) / 10.0;	// get a number in the range 0.1 - 1.0
    rv_cpu[i + 3] = (rand()%10 + 1) / 10.0;	// get a number in the range 0.1 - 1.0
  }

  // input (charge)
  qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    qv_cpu[i] = (rand()%10 + 1) / 10.0;	// get a number in the range 0.1 - 1.0
  }

  // output (forces)
  fv_cpu = (fp*)malloc(dim_cpu.space_mem);
  for(i=0; i < 4 * dim_cpu.space_elem; i=i+4){
    fv_cpu[i] = 0;			// set to 0, because kernels keeps adding to initial value
    fv_cpu[i + 1] = 0;			// set to 0, because kernels keeps adding to initial value
    fv_cpu[i + 2] = 0;			// set to 0, because kernels keeps adding to initial value
    fv_cpu[i + 3] = 0;			// set to 0, because kernels keeps adding to initial value
  }

  time5 = get_time();
  kernel_gpu_cuda_wrapper(par_cpu, dim_cpu, box_cpu, rv_cpu, qv_cpu, fv_cpu);
  time6 = get_time();

  //PRINT RESULT
  //for (i = 0; i < dim_cpu.space_elem; i++)
  //printf("Element %d fv: v = %f x = %f y = %f z = %f\n", i, fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
  int passed = test_output(fv_cpu, dim_cpu.space_elem, "gold_output.txt");

  free(rv_cpu);
  free(qv_cpu);
  free(fv_cpu);
  free(box_cpu);

  time7 = get_time();

  printf("Time spent in different stages of the application:\n");
  printf("%15.12f s, %15.12f % : VARIABLES\n",		(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time7-time0) * 100);
  printf("%15.12f s, %15.12f % : INPUT ARGUMENTS\n", 	(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time7-time0) * 100);
  printf("%15.12f s, %15.12f % : INPUTS\n",		(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time7-time0) * 100);
  printf("%15.12f s, %15.12f % : dim_cpu\n", 		(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time7-time0) * 100);
  printf("%15.12f s, %15.12f % : SYS MEM: ALO\n",	(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time7-time0) * 100);
  printf("%15.12f s, %15.12f % : KERNEL: COMPUTE\n",	(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time7-time0) * 100);
  printf("%15.12f s, %15.12f % : SYS MEM: FRE\n", 	(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time7-time0) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", 					(float) (time7-time0) / 1000000);
  if (passed) {
    printf("PASSED.\n");
    return 0;
  } else {
    printf("FAILED.\n");
    return 1;
  }
 
  return 1;
}
