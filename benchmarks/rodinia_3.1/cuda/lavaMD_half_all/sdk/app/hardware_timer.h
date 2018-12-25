#ifndef HW_TIMER_H
#define HW_TIMER_H
#include <stdio.h>
#include "xtime_l.h"
#include "xil_io.h"
#include "xparameters.h"

#define timer_base 0xF8F00000
/***********************************************************
Timer Registers
************************************************************/
static volatile int *timer_counter_l=(volatile int *)(timer_base+0x200);
static volatile int *timer_counter_h=(volatile int *)(timer_base+0x204);
static volatile int *timer_ctrl=(volatile int *)(timer_base+0x208);
/***********************************************************
/***********************************************************
Function definitions
************************************************************/
void init_timer(volatile int *timer_ctrl, volatile int *timer_counter_l, volatile int *timer_counter_h){
        *timer_ctrl=0x0;
        *timer_counter_l=0x1;
        *timer_counter_h=0x0;
        DATA_SYNC;
}

void start_timer(volatile int *timer_ctrl){
        *timer_ctrl=*timer_ctrl | 0x00000001;
        DATA_SYNC;
}

void stop_timer(volatile int *timer_ctrl){
        *timer_ctrl=*timer_ctrl & 0xFFFFFFFE;
        DATA_SYNC;
}

s64 elapsed_time() {
	return (((0xFFFFFFFFFFFFFFFF & *timer_counter_h) << 32) | (0x00000000FFFFFFFF & *timer_counter_l)) / 333;
}
#endif
/*
int main()
{
    int R1, R2, R3, i;

	R1 = 1000;
	R2 = 500;
	
    print("Code Profiling\n\r");
	
    //Initialise the timer for performance monitoring
    init_timer(timer_ctrl, timer_counter_l, timer_counter_h);
    start_timer(timer_ctrl);
	
		for (i=0;i<=1000;i+=1)
		R3 = R1 + R2;

    stop_timer(timer_ctrl);
	
    printf("R3 is %u\n\r", R3);
	
    //Calculate the time for the operation
    xil_printf("Communication time %d us\n\r", (*timer_counter_l)/333);

    return 0;
}
*/
