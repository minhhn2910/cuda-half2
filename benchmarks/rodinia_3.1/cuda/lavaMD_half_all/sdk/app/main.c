// 14 APR 2011 Lukasz G. Szafaryn
#include <stdio.h>			// (in path known to compiler) needed by printf
#include <stdlib.h>			// (in path known to compiler) needed by malloc
#include <stdbool.h>			// (in path known to compiler) needed by true/false
#include "num/num.h"		// (in path specified here)
#include "main.h"			// (in the current directory)
#include "kernel_gpu_cuda_wrapper.h"	// (in library path specified here)
#include <string.h>
#include "hardware_timer.h"
//#define VERBOSE
#define EPSILON 0.001

int test_output(fp* fv_cpu, int num_space_elements, char *gold_file) {
  int i, check_v, check_x, check_y, check_z;
  FILE *gold_fp = fopen(gold_file, "r");
  float gold_v, gold_x, gold_y, gold_z;
  for (i = 0; i < 4 * num_space_elements; i+=4) {
    fscanf(gold_fp, "%f %f %f %f\n", &gold_v, &gold_x, &gold_y, &gold_z);
    check_v = ((gold_v - fv_cpu[i] >= -EPSILON) && (gold_v - fv_cpu[i] <= EPSILON));
    check_x = ((gold_x - fv_cpu[i + 1] >= -EPSILON) && (gold_x - fv_cpu[i + 1] <= EPSILON));
    check_y = ((gold_y - fv_cpu[i + 2] >= -EPSILON) && (gold_y - fv_cpu[i + 2] <= EPSILON));
    check_z = ((gold_z - fv_cpu[i + 3] >= -EPSILON) && (gold_z - fv_cpu[i + 3] <= EPSILON));
    if (!(check_v && check_x && check_y && check_z)) {
      printf("Result mismatched at element %d: %f %f %f %f vs %f %f %f %f\n",
          i, gold_v, gold_x, gold_y, gold_z,
          fv_cpu[i], fv_cpu[i + 1], fv_cpu[i + 2], fv_cpu[i + 3]);
      return 0;
    }
  }
  fclose(gold_fp);
  return 1;
}

int main(int argc, char *argv [])
{
  init_timer(timer_ctrl, timer_counter_l, timer_counter_h);
  start_timer(timer_ctrl);
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

  // assing default values
  dim_cpu.boxes1d_arg = 2;

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

#ifdef VERBOSE
  // Print configuration
  printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);
#endif

  par_cpu.alpha = 0.5;

  // total number of boxes
  dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

  // how many particles space has in each direction
  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
  dim_cpu.space_mem = dim_cpu.space_elem * 4 * sizeof(fp);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  // box array
  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

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
  for(i=0; i < 4 * dim_cpu.space_elem; i=i+1){
    rv_cpu[i] = (rand()%10 + 1) / 10.0;	// get a number in the range 0.1 - 1.0
  }

  // input (charge)
  qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    qv_cpu[i] = (rand()%10 + 1) / 10.0;	// get a number in the range 0.1 - 1.0
  }

  // output (forces)
  fv_cpu = (fp*)malloc(dim_cpu.space_mem);
  for(i=0; i < 4 * dim_cpu.space_elem; i=i+1){
    fv_cpu[i] = 0;			// set to 0, because kernels keeps adding to initial value
  }

  kernel_gpu_cuda_wrapper(par_cpu, dim_cpu, box_cpu, rv_cpu, qv_cpu, fv_cpu);

  stop_timer(timer_ctrl);
  printf("Execution time %lld us\n\r", elapsed_time());
  return 0;
}
