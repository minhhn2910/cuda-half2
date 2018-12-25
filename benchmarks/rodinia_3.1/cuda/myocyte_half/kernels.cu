//=====================================================================
//	MAIN FUNCTION
//=====================================================================
#undef fp
#define fp half
#include <math.h>

//~ #define max(x,y) ( (x) < (y) ? (y) : (x) )
//~ #define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

#include <cuda_fp16.h>
#include "half_operator_overload.cuh"


__device__ void kernel_fin_2(	int timeinst,
													fp* d_initvalu,
													fp* d_finavalu,
													int offset_ecc,
													int offset_Dyad,
													int offset_SL,
													int offset_Cyt,
													fp* d_params,
													fp* d_com){

//=====================================================================
//	VARIABLES
//=====================================================================

	// input parameters
	fp BtotDyad;
	fp CaMKIItotDyad;

	// compute variables
	fp Vmyo;																			// [L]
	fp Vdyad;																			// [L]
	fp VSL;																				// [L]
	// fp kDyadSL;																			// [L/msec]
	fp kSLmyo;																			// [L/msec]
	fp k0Boff;																			// [s^-1]
	fp k0Bon;																			// [uM^-1 s^-1] kon = koff/Kd
	fp k2Boff;																			// [s^-1]
	fp k2Bon;																			// [uM^-1 s^-1]
	// fp k4Boff;																			// [s^-1]
	fp k4Bon;																			// [uM^-1 s^-1]
	fp CaMtotDyad;
	fp Bdyad;																			// [uM dyad]
	fp J_cam_dyadSL;																	// [uM/msec dyad]
	fp J_ca2cam_dyadSL;																	// [uM/msec dyad]
	fp J_ca4cam_dyadSL;																	// [uM/msec dyad]
	fp J_cam_SLmyo;																		// [umol/msec]
	fp J_ca2cam_SLmyo;																	// [umol/msec]
	fp J_ca4cam_SLmyo;																	// [umol/msec]

//=====================================================================
//	COMPUTATION
//=====================================================================

	// input parameters
	BtotDyad = d_params[1];
	CaMKIItotDyad = d_params[2];

	// ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
	d_finavalu[offset_ecc+35] = d_finavalu[offset_ecc+35] + 1e-3*d_com[0];
	d_finavalu[offset_ecc+36] = d_finavalu[offset_ecc+36] + 1e-3*d_com[1];
	d_finavalu[offset_ecc+37] = d_finavalu[offset_ecc+37] + 1e-3*d_com[2];

	// incorporate CaM diffusion between compartments
	Vmyo = __float2half(2.1454e-11);																// [L]
	Vdyad = __float2half(1.7790e-14);																// [L]
	VSL = __float2half(6.6013e-13);																// [L]
	// kDyadSL = 3.6363e-16;															// [L/msec]
	kSLmyo = __float2half(8.587e-15);																// [L/msec]
	k0Boff = __float2half(0.0014);																// [s^-1]
	k0Bon = k0Boff/0.2;																// [uM^-1 s^-1] kon = koff/Kd
	k2Boff = k0Boff/100;															// [s^-1]
	k2Bon = k0Bon;																	// [uM^-1 s^-1]
	// k4Boff = k2Boff;																// [s^-1]
	k4Bon = k0Bon;																	// [uM^-1 s^-1]
	CaMtotDyad = d_initvalu[offset_Dyad+0]
			   + d_initvalu[offset_Dyad+1]
			   + d_initvalu[offset_Dyad+2]
			   + d_initvalu[offset_Dyad+3]
			   + d_initvalu[offset_Dyad+4]
			   + d_initvalu[offset_Dyad+5]
			   + CaMKIItotDyad * (	  d_initvalu[offset_Dyad+6]
												  + d_initvalu[offset_Dyad+7]
												  + d_initvalu[offset_Dyad+8]
												  + d_initvalu[offset_Dyad+9])
			   + d_initvalu[offset_Dyad+12]
			   + d_initvalu[offset_Dyad+13]
			   + d_initvalu[offset_Dyad+14];
	Bdyad = BtotDyad - CaMtotDyad;																				// [uM dyad]
	J_cam_dyadSL = 1e-3 * (  k0Boff*d_initvalu[offset_Dyad+0] - k0Bon*Bdyad*d_initvalu[offset_SL+0]);			// [uM/msec dyad]
	J_ca2cam_dyadSL = 1e-3 * (  k2Boff*d_initvalu[offset_Dyad+1] - k2Bon*Bdyad*d_initvalu[offset_SL+1]);		// [uM/msec dyad]
	J_ca4cam_dyadSL = 1e-3 * (  k2Boff*d_initvalu[offset_Dyad+2] - k4Bon*Bdyad*d_initvalu[offset_SL+2]);		// [uM/msec dyad]

	J_cam_SLmyo = kSLmyo * (  d_initvalu[offset_SL+0] - d_initvalu[offset_Cyt+0]);								// [umol/msec]
	J_ca2cam_SLmyo = kSLmyo * (  d_initvalu[offset_SL+1] - d_initvalu[offset_Cyt+1]);							// [umol/msec]
	J_ca4cam_SLmyo = kSLmyo * (  d_initvalu[offset_SL+2] - d_initvalu[offset_Cyt+2]);							// [umol/msec]

	// ADJUST CAM Dyad
	d_finavalu[offset_Dyad+0] = d_finavalu[offset_Dyad+0] - J_cam_dyadSL;
	d_finavalu[offset_Dyad+1] = d_finavalu[offset_Dyad+1] - J_ca2cam_dyadSL;
	d_finavalu[offset_Dyad+2] = d_finavalu[offset_Dyad+2] - J_ca4cam_dyadSL;

	// ADJUST CAM Sl
	d_finavalu[offset_SL+0] = d_finavalu[offset_SL+0] + J_cam_dyadSL*Vdyad/VSL - J_cam_SLmyo/VSL;
	d_finavalu[offset_SL+1] = d_finavalu[offset_SL+1] + J_ca2cam_dyadSL*Vdyad/VSL - J_ca2cam_SLmyo/VSL;
	d_finavalu[offset_SL+2] = d_finavalu[offset_SL+2] + J_ca4cam_dyadSL*Vdyad/VSL - J_ca4cam_SLmyo/VSL;

	// ADJUST CAM Cyt
	d_finavalu[offset_Cyt+0] = d_finavalu[offset_Cyt+0] + J_cam_SLmyo/Vmyo;
	d_finavalu[offset_Cyt+1] = d_finavalu[offset_Cyt+1] + J_ca2cam_SLmyo/Vmyo;
	d_finavalu[offset_Cyt+2] = d_finavalu[offset_Cyt+2] + J_ca4cam_SLmyo/Vmyo;

}
//=====================================================================
//	MAIN FUNCTION
//=====================================================================
__device__ void kernel_ecc_2(	fp timeinst,
													fp* d_initvalu,
													fp* d_finavalu,
													int valu_offset,
													fp* d_params){

	//=====================================================================
	//	VARIABLES
	//=====================================================================

	// input parameters
	fp cycleLength;

	// variable references				// GET VARIABLES FROM MEMORY AND SAVE LOCALLY !!!!!!!!!!!!!!!!!!
	int offset_1;
	int offset_2;
	int offset_3;
	int offset_4;
	int offset_5;
	int offset_6;
	int offset_7;
	int offset_8;
	int offset_9;
	int offset_10;
	int offset_11;
	int offset_12;
	int offset_13;
	int offset_14;
	int offset_15;
	int offset_16;
	int offset_17;
	int offset_18;
	int offset_19;
	int offset_20;
	int offset_21;
	int offset_22;
	int offset_23;
	int offset_24;
	int offset_25;
	int offset_26;
	int offset_27;
	int offset_28;
	int offset_29;
	int offset_30;
	int offset_31;
	int offset_32;
	int offset_33;
	int offset_34;
	int offset_35;
	int offset_36;
	int offset_37;
	int offset_38;
	int offset_39;
	int offset_40;
	int offset_41;
	int offset_42;
	int offset_43;
	int offset_44;
	int offset_45;
	int offset_46;

	// stored input array
	fp d_initvalu_1;
	fp d_initvalu_2;
	fp d_initvalu_3;
	fp d_initvalu_4;
	fp d_initvalu_5;
	fp d_initvalu_6;
	fp d_initvalu_7;
	fp d_initvalu_8;
	fp d_initvalu_9;
	fp d_initvalu_10;
	fp d_initvalu_11;
	fp d_initvalu_12;
	fp d_initvalu_13;
	fp d_initvalu_14;
	fp d_initvalu_15;
	fp d_initvalu_16;
	fp d_initvalu_17;
	fp d_initvalu_18;
	fp d_initvalu_19;
	fp d_initvalu_20;
	fp d_initvalu_21;
	// fp d_initvalu_22;
	fp d_initvalu_23;
	fp d_initvalu_24;
	fp d_initvalu_25;
	fp d_initvalu_26;
	fp d_initvalu_27;
	fp d_initvalu_28;
	fp d_initvalu_29;
	fp d_initvalu_30;
	fp d_initvalu_31;
	fp d_initvalu_32;
	fp d_initvalu_33;
	fp d_initvalu_34;
	fp d_initvalu_35;
	fp d_initvalu_36;
	fp d_initvalu_37;
	fp d_initvalu_38;
	fp d_initvalu_39;
	fp d_initvalu_40;
	// fp d_initvalu_41;
	// fp d_initvalu_42;
	// fp d_initvalu_43;
	// fp d_initvalu_44;
	// fp d_initvalu_45;
	// fp d_initvalu_46;

	// matlab constants undefined in c
	fp pi;

	// Constants
	fp R;																			// [J/kmol*K]
	fp Frdy;																		// [C/mol]
	fp Temp;																		// [K] 310
	fp FoRT;																		//
	fp Cmem;																		// [F] membrane capacitance
	fp Qpow;

	// Cell geometry
	fp cellLength;																	// cell length [um]
	fp cellRadius;																	// cell radius [um]
	// fp junctionLength;																// junc length [um]
	// fp junctionRadius;																// junc radius [um]
	// fp distSLcyto;																	// dist. SL to cytosol [um]
	// fp distJuncSL;																	// dist. junc to SL [um]
	// fp DcaJuncSL;																	// Dca junc to SL [cm^2/sec]
	// fp DcaSLcyto;																	// Dca SL to cyto [cm^2/sec]
	// fp DnaJuncSL;																	// Dna junc to SL [cm^2/sec]
	// fp DnaSLcyto;																	// Dna SL to cyto [cm^2/sec]
	fp Vcell;																		// [L]
	fp Vmyo;
	fp Vsr;
	fp Vsl;
	fp Vjunc;
	// fp SAjunc;																		// [um^2]
	// fp SAsl;																		// [um^2]
	fp J_ca_juncsl;																	// [L/msec]
	fp J_ca_slmyo;																	// [L/msec]
	fp J_na_juncsl;																	// [L/msec]
	fp J_na_slmyo;																	// [L/msec]

	// Fractional currents in compartments
	fp Fjunc;
	fp Fsl;
	fp Fjunc_CaL;
	fp Fsl_CaL;

	// Fixed ion concentrations
	fp Cli;																			// Intracellular Cl  [mM]
	fp Clo;																			// Extracellular Cl  [mM]
	fp Ko;																			// Extracellular K   [mM]
	fp Nao;																			// Extracellular Na  [mM]
	fp Cao;																			// Extracellular Ca  [mM]
	fp Mgi;																			// Intracellular Mg  [mM]

	// Nernst Potentials
	fp ena_junc;																	// [mV]
	fp ena_sl;																		// [mV]
	fp ek;																			// [mV]
	fp eca_junc;																	// [mV]
	fp eca_sl;																		// [mV]
	fp ecl;																			// [mV]

	// Na transport parameters
	fp GNa;																			// [mS/uF]
	fp GNaB;																		// [mS/uF]
	fp IbarNaK;																		// [uA/uF]
	fp KmNaip;																		// [mM]
	fp KmKo;																		// [mM]
	// fp Q10NaK;
	// fp Q10KmNai;

	// K current parameters
	fp pNaK;
	fp GtoSlow;																		// [mS/uF]
	fp GtoFast;																		// [mS/uF]
	fp gkp;

	// Cl current parameters
	fp GClCa;																		// [mS/uF]
	fp GClB;																		// [mS/uF]
	fp KdClCa;																		// [mM]																// [mM]

	// I_Ca parameters
	fp pNa;																			// [cm/sec]
	fp pCa;																			// [cm/sec]
	fp pK;																			// [cm/sec]
	// fp KmCa;																		// [mM]
	fp Q10CaL;

	// Ca transport parameters
	fp IbarNCX;																		// [uA/uF]
	fp KmCai;																		// [mM]
	fp KmCao;																		// [mM]
	fp KmNai;																		// [mM]
	fp KmNao;																		// [mM]
	fp ksat;																			// [none]
	fp nu;																			// [none]
	fp Kdact;																		// [mM]
	fp Q10NCX;																		// [none]
	fp IbarSLCaP;																	// [uA/uF]
	fp KmPCa;																		// [mM]
	fp GCaB;																		// [uA/uF]
	fp Q10SLCaP;																	// [none]																	// [none]

	// SR flux parameters
	fp Q10SRCaP;																	// [none]
	fp Vmax_SRCaP;																	// [mM/msec] (mmol/L cytosol/msec)
	fp Kmf;																			// [mM]
	fp Kmr;																			// [mM]L cytosol
	fp hillSRCaP;																	// [mM]
	fp ks;																			// [1/ms]
	fp koCa;																		// [mM^-2 1/ms]
	fp kom;																			// [1/ms]
	fp kiCa;																		// [1/mM/ms]
	fp kim;																			// [1/ms]
	fp ec50SR;																		// [mM]

	// Buffering parameters
	fp Bmax_Naj;																	// [mM]
	fp Bmax_Nasl;																	// [mM]
	fp koff_na;																		// [1/ms]
	fp kon_na;																		// [1/mM/ms]
	fp Bmax_TnClow;																	// [mM], TnC low affinity
	fp koff_tncl;																	// [1/ms]
	fp kon_tncl;																	// [1/mM/ms]
	fp Bmax_TnChigh;																// [mM], TnC high affinity
	fp koff_tnchca;																	// [1/ms]
	fp kon_tnchca;																	// [1/mM/ms]
	fp koff_tnchmg;																	// [1/ms]
	fp kon_tnchmg;																	// [1/mM/ms]
	// fp Bmax_CaM;																	// [mM], CaM buffering
	// fp koff_cam;																	// [1/ms]
	// fp kon_cam;																		// [1/mM/ms]
	fp Bmax_myosin;																	// [mM], Myosin buffering
	fp koff_myoca;																	// [1/ms]
	fp kon_myoca;																	// [1/mM/ms]
	fp koff_myomg;																	// [1/ms]
	fp kon_myomg;																	// [1/mM/ms]
	fp Bmax_SR;																		// [mM]
	fp koff_sr;																		// [1/ms]
	fp kon_sr;																		// [1/mM/ms]
	fp Bmax_SLlowsl;																// [mM], SL buffering
	fp Bmax_SLlowj;																	// [mM]
	fp koff_sll;																	// [1/ms]
	fp kon_sll;																		// [1/mM/ms]
	fp Bmax_SLhighsl;																// [mM]
	fp Bmax_SLhighj;																// [mM]
	fp koff_slh;																	// [1/ms]
	fp kon_slh;																		// [1/mM/ms]
	fp Bmax_Csqn;																	// 140e-3*Vmyo/Vsr; [mM]
	fp koff_csqn;																	// [1/ms]
	fp kon_csqn;																	// [1/mM/ms]

	// I_Na: Fast Na Current
	fp am;
	fp bm;
	fp ah;
	fp bh;
	fp aj;
	fp bj;
	fp I_Na_junc;
	fp I_Na_sl;
	// fp I_Na;

	// I_nabk: Na Background Current
	fp I_nabk_junc;
	fp I_nabk_sl;
	// fp I_nabk;

	// I_nak: Na/K Pump Current
	fp sigma;
	fp fnak;
	fp I_nak_junc;
	fp I_nak_sl;
	fp I_nak;

	// I_kr: Rapidly Activating K Current
	fp gkr;
	fp xrss;
	fp tauxr;
	fp rkr;
	fp I_kr;

	// I_ks: Slowly Activating K Current
	fp pcaks_junc;
	fp pcaks_sl;
	fp gks_junc;
	fp gks_sl;
	fp eks;
	fp xsss;
	fp tauxs;
	fp I_ks_junc;
	fp I_ks_sl;
	fp I_ks;

	// I_kp: Plateau K current
	fp kp_kp;
	fp I_kp_junc;
	fp I_kp_sl;
	fp I_kp;

	// I_to: Transient Outward K Current (slow and fast components)
	fp xtoss;
	fp ytoss;
	fp rtoss;
	fp tauxtos;
	fp tauytos;
	fp taurtos;
	fp I_tos;

	//
	fp tauxtof;
	fp tauytof;
	fp I_tof;
	fp I_to;

	// I_ki: Time-Independent K Current
	fp aki;
	fp bki;
	fp kiss;
	fp I_ki;

	// I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
	fp I_ClCa_junc;
	fp I_ClCa_sl;
	fp I_ClCa;
	fp I_Clbk;

	// I_Ca: L-type Calcium Current
	fp dss;
	fp taud;
	fp fss;
	fp tauf;

	//
	fp ibarca_j;
	fp ibarca_sl;
	fp ibark;
	fp ibarna_j;
	fp ibarna_sl;
	fp I_Ca_junc;
	fp I_Ca_sl;
	fp I_Ca;
	fp I_CaK;
	fp I_CaNa_junc;
	fp I_CaNa_sl;
	// fp I_CaNa;
	// fp I_Catot;

	// I_ncx: Na/Ca Exchanger flux
	fp Ka_junc;
	fp Ka_sl;
	fp s1_junc;
	fp s1_sl;
	fp s2_junc;
	fp s3_junc;
	fp s2_sl;
	fp s3_sl;
	fp I_ncx_junc;
	fp I_ncx_sl;
	fp I_ncx;

	// I_pca: Sarcolemmal Ca Pump Current
	fp I_pca_junc;
	fp I_pca_sl;
	fp I_pca;

	// I_cabk: Ca Background Current
	fp I_cabk_junc;
	fp I_cabk_sl;
	fp I_cabk;

	// SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
	fp MaxSR;
	fp MinSR;
	fp kCaSR;
	fp koSRCa;
	fp kiSRCa;
	fp RI;
	fp J_SRCarel;																	// [mM/ms]
	fp J_serca;
	fp J_SRleak;																		//   [mM/ms]

	// Cytosolic Ca Buffers
	fp J_CaB_cytosol;

	// Junctional and SL Ca Buffers
	fp J_CaB_junction;
	fp J_CaB_sl;

	// SR Ca Concentrations
	fp oneovervsr;

	// Sodium Concentrations
	fp I_Na_tot_junc;																// [uA/uF]
	fp I_Na_tot_sl;																	// [uA/uF]
	fp oneovervsl;

	// Potassium Concentration
	fp I_K_tot;

	// Calcium Concentrations
	fp I_Ca_tot_junc;																// [uA/uF]
	fp I_Ca_tot_sl;																	// [uA/uF]
	// fp junc_sl;
	// fp sl_junc;
	// fp sl_myo;
	// fp myo_sl;

	//	Simulation type
	int state;																			// 0-none; 1-pace; 2-vclamp
	fp I_app;
	fp V_hold;
	fp V_test;
	fp V_clamp;
	fp R_clamp;

	//	Membrane Potential
	fp I_Na_tot;																		// [uA/uF]
	fp I_Cl_tot;																		// [uA/uF]
	fp I_Ca_tot;
	fp I_tot;

	//=====================================================================
	//	EXECUTION
	//=====================================================================

	// input parameters
	cycleLength = d_params[15];

	// variable references
	offset_1 = valu_offset;
	offset_2 = valu_offset+1;
	offset_3 = valu_offset+2;
	offset_4 = valu_offset+3;
	offset_5 = valu_offset+4;
	offset_6 = valu_offset+5;
	offset_7 = valu_offset+6;
	offset_8 = valu_offset+7;
	offset_9 = valu_offset+8;
	offset_10 = valu_offset+9;
	offset_11 = valu_offset+10;
	offset_12 = valu_offset+11;
	offset_13 = valu_offset+12;
	offset_14 = valu_offset+13;
	offset_15 = valu_offset+14;
	offset_16 = valu_offset+15;
	offset_17 = valu_offset+16;
	offset_18 = valu_offset+17;
	offset_19 = valu_offset+18;
	offset_20 = valu_offset+19;
	offset_21 = valu_offset+20;
	offset_22 = valu_offset+21;
	offset_23 = valu_offset+22;
	offset_24 = valu_offset+23;
	offset_25 = valu_offset+24;
	offset_26 = valu_offset+25;
	offset_27 = valu_offset+26;
	offset_28 = valu_offset+27;
	offset_29 = valu_offset+28;
	offset_30 = valu_offset+29;
	offset_31 = valu_offset+30;
	offset_32 = valu_offset+31;
	offset_33 = valu_offset+32;
	offset_34 = valu_offset+33;
	offset_35 = valu_offset+34;
	offset_36 = valu_offset+35;
	offset_37 = valu_offset+36;
	offset_38 = valu_offset+37;
	offset_39 = valu_offset+38;
	offset_40 = valu_offset+39;
	offset_41 = valu_offset+40;
	offset_42 = valu_offset+41;
	offset_43 = valu_offset+42;
	offset_44 = valu_offset+43;
	offset_45 = valu_offset+44;
	offset_46 = valu_offset+45;

	// stored input array
	d_initvalu_1 = d_initvalu[offset_1];
	d_initvalu_2 = d_initvalu[offset_2];
	d_initvalu_3 = d_initvalu[offset_3];
	d_initvalu_4 = d_initvalu[offset_4];
	d_initvalu_5 = d_initvalu[offset_5];
	d_initvalu_6 = d_initvalu[offset_6];
	d_initvalu_7 = d_initvalu[offset_7];
	d_initvalu_8 = d_initvalu[offset_8];
	d_initvalu_9 = d_initvalu[offset_9];
	d_initvalu_10 = d_initvalu[offset_10];
	d_initvalu_11 = d_initvalu[offset_11];
	d_initvalu_12 = d_initvalu[offset_12];
	d_initvalu_13 = d_initvalu[offset_13];
	d_initvalu_14 = d_initvalu[offset_14];
	d_initvalu_15 = d_initvalu[offset_15];
	d_initvalu_16 = d_initvalu[offset_16];
	d_initvalu_17 = d_initvalu[offset_17];
	d_initvalu_18 = d_initvalu[offset_18];
	d_initvalu_19 = d_initvalu[offset_19];
	d_initvalu_20 = d_initvalu[offset_20];
	d_initvalu_21 = d_initvalu[offset_21];
	// d_initvalu_22 = d_initvalu[offset_22];
	d_initvalu_23 = d_initvalu[offset_23];
	d_initvalu_24 = d_initvalu[offset_24];
	d_initvalu_25 = d_initvalu[offset_25];
	d_initvalu_26 = d_initvalu[offset_26];
	d_initvalu_27 = d_initvalu[offset_27];
	d_initvalu_28 = d_initvalu[offset_28];
	d_initvalu_29 = d_initvalu[offset_29];
	d_initvalu_30 = d_initvalu[offset_30];
	d_initvalu_31 = d_initvalu[offset_31];
	d_initvalu_32 = d_initvalu[offset_32];
	d_initvalu_33 = d_initvalu[offset_33];
	d_initvalu_34 = d_initvalu[offset_34];
	d_initvalu_35 = d_initvalu[offset_35];
	d_initvalu_36 = d_initvalu[offset_36];
	d_initvalu_37 = d_initvalu[offset_37];
	d_initvalu_38 = d_initvalu[offset_38];
	d_initvalu_39 = d_initvalu[offset_39];
	d_initvalu_40 = d_initvalu[offset_40];
	// d_initvalu_41 = d_initvalu[offset_41];
	// d_initvalu_42 = d_initvalu[offset_42];
	// d_initvalu_43 = d_initvalu[offset_43];
	// d_initvalu_44 = d_initvalu[offset_44];
	// d_initvalu_45 = d_initvalu[offset_45];
	// d_initvalu_46 = d_initvalu[offset_46];

	// matlab constants undefined in c
	pi = __float2half(3.1416);

	// Constants
	R = __float2half(8314);																			// [J/kmol*K]
	Frdy = __float2half(96485);																		// [C/mol]
	Temp = __float2half(310);																			// [K] 310
	FoRT = Frdy/R/Temp;																	//
	Cmem = __float2half(1.3810e-10);																	// [F] membrane capacitance
	Qpow = (Temp-310)/10;

	// Cell geometry
	cellLength = __float2half(100);																	// cell length [um]
	cellRadius = __float2half(10.25);																	// cell radius [um]
	// junctionLength = 160e-3;															// junc length [um]
	// junctionRadius = 15e-3;																// junc radius [um]
	// distSLcyto = 0.45;																	// dist. SL to cytosol [um]
	// distJuncSL = 0.5;																	// dist. junc to SL [um]
	// DcaJuncSL = 1.64e-6;																// Dca junc to SL [cm^2/sec]
	// DcaSLcyto = 1.22e-6;																// Dca SL to cyto [cm^2/sec]
	// DnaJuncSL = 1.09e-5;																// Dna junc to SL [cm^2/sec]
	// DnaSLcyto = 1.79e-5;																// Dna SL to cyto [cm^2/sec]
	//Vcell = pi*pow(cellRadius,2)*cellLength*1e-15;											// [L]
	Vcell = pi*cellRadius*cellRadius*cellLength*1e-15;
	Vmyo = 0.65*Vcell;
	Vsr = 0.035*Vcell;
	Vsl = 0.02*Vcell;
	Vjunc = 0.0539*0.01*Vcell;
	// SAjunc = 20150*pi*2*junctionLength*junctionRadius;									// [um^2]
	// SAsl = pi*2*cellRadius*cellLength;													// [um^2]
	J_ca_juncsl = __float2half(1/1.2134e12);															// [L/msec]
	J_ca_slmyo = __float2half(1/2.68510e11);															// [L/msec]
	J_na_juncsl = __float2half(1/(1.6382e12/3*100));													// [L/msec]
	J_na_slmyo = __float2half(1/(1.8308e10/3*100));													// [L/msec]

	// Fractional currents in compartments
	Fjunc = __float2half(0.11);
	Fsl = 1-Fjunc;
	Fjunc_CaL = __float2half(0.9);
	Fsl_CaL = 1-Fjunc_CaL;

	// Fixed ion concentrations
	Cli = __float2half(15);																			// Intracellular Cl  [mM]
	Clo = __float2half(150);																			// Extracellular Cl  [mM]
	Ko = __float2half(5.4);																			// Extracellular K   [mM]
	Nao = __float2half(140);																			// Extracellular Na  [mM]
	Cao = __float2half(1.8);																			// Extracellular Ca  [mM]
	Mgi = __float2half(1);																			// Intracellular Mg  [mM]

	// Nernst Potentials
	ena_junc = (1/FoRT)*log(Nao/d_initvalu_32);													// [mV]
	ena_sl = (1/FoRT)*log(Nao/d_initvalu_33);													// [mV]
	ek = (1/FoRT)*log(Ko/d_initvalu_35);														// [mV]
	eca_junc = (1/FoRT/2)*log(Cao/d_initvalu_36);												// [mV]
	eca_sl = (1/FoRT/2)*log(Cao/d_initvalu_37);													// [mV]
	ecl = (1/FoRT)*log(Cli/Clo);														// [mV]

	// Na transport parameters
	GNa =  __float2half(16.0);																		// [mS/uF]
	GNaB = __float2half(0.297e-3);																	// [mS/uF]
	IbarNaK = __float2half(1.90719);																	// [uA/uF]
	KmNaip = __float2half(11);																		// [mM]
	KmKo = __float2half(1.5);																			// [mM]
	// Q10NaK = 1.63;
	// Q10KmNai = 1.39;

	// K current parameters
	pNaK = __float2half(0.01833);
	GtoSlow = __float2half(0.06);																		// [mS/uF]
	GtoFast = __float2half(0.02);																		// [mS/uF]
	gkp = __float2half(0.001);

	// Cl current parameters
	GClCa = __float2half(0.109625);																	// [mS/uF]
	GClB = __float2half(9e-3);																		// [mS/uF]
	KdClCa = __float2half(100e-3);																	// [mM]

	// I_Ca parameters
	pNa = __float2half(1.5e-8);																		// [cm/sec]
	pCa = __float2half(5.4e-4);																		// [cm/sec]
	pK = __float2half(2.7e-7);																		// [cm/sec]
	// KmCa = 0.6e-3;																		// [mM]
	Q10CaL = __float2half(1.8);

	// Ca transport parameters
	IbarNCX = __float2half(9.0);																		// [uA/uF]
	KmCai = __float2half(3.59e-3);																	// [mM]
	KmCao = __float2half(1.3);																		// [mM]
	KmNai = __float2half(12.29);																		// [mM]
	KmNao = __float2half(87.5);																		// [mM]
	ksat = __float2half(0.27);																		// [none]
	nu = __float2half(0.35);																			// [none]
	Kdact = __float2half(0.256e-3);																	// [mM]
	Q10NCX = __float2half(1.57);																		// [none]
	IbarSLCaP = __float2half(0.0673);																	// [uA/uF]
	KmPCa = __float2half(0.5e-3);																		// [mM]
	GCaB = __float2half(2.513e-4);																	// [uA/uF]
	Q10SLCaP = __float2half(2.35);																	// [none]

	// SR flux parameters
	Q10SRCaP = __float2half(2.6);																		// [none]
	Vmax_SRCaP = __float2half(2.86e-4);																// [mM/msec] (mmol/L cytosol/msec)
	Kmf = __float2half(0.246e-3);																		// [mM]
	Kmr = __float2half(1.7);																			// [mM]L cytosol
	hillSRCaP = __float2half(1.787);																	// [mM]
	ks = __float2half(25);																			// [1/ms]
	koCa = __float2half(10);																			// [mM^-2 1/ms]
	kom = __float2half(0.06);																			// [1/ms]
	kiCa = __float2half(0.5);																			// [1/mM/ms]
	kim = __float2half(0.005);																		// [1/ms]
	ec50SR =__float2half( 0.45);																		// [mM]

	// Buffering parameters
	Bmax_Naj = __float2half(7.561);																	// [mM]
	Bmax_Nasl = __float2half(1.65);																	// [mM]
	koff_na = __float2half(1e-3);																		// [1/ms]
	kon_na = __float2half(0.1e-3);																	// [1/mM/ms]
	Bmax_TnClow = __float2half(70e-3);																// [mM], TnC low affinity
	koff_tncl = __float2half(19.6e-3);																// [1/ms]
	kon_tncl = __float2half(32.7);																	// [1/mM/ms]
	Bmax_TnChigh = __float2half(140e-3);																// [mM], TnC high affinity
	koff_tnchca = __float2half(0.032e-3);																// [1/ms]
	kon_tnchca = __float2half(2.37);																	// [1/mM/ms]
	koff_tnchmg = __float2half(3.33e-3);																// [1/ms]
	kon_tnchmg = __float2half(3e-3);																	// [1/mM/ms]
	// Bmax_CaM = 24e-3);																	// [mM], CaM buffering
	// koff_cam = 238e-3);																	// [1/ms]
	// kon_cam = 34);																		// [1/mM/ms]
	Bmax_myosin = __float2half(140e-3);																// [mM], Myosin buffering
	koff_myoca = __float2half(0.46e-3);																// [1/ms]
	kon_myoca = __float2half(13.8);																	// [1/mM/ms]
	koff_myomg = __float2half(0.057e-3);																// [1/ms]
	kon_myomg = __float2half(0.0157);																	// [1/mM/ms]
	Bmax_SR = __float2half(19*0.9e-3);																	// [mM]
	koff_sr = __float2half(60e-3);																	// [1/ms]
	kon_sr = __float2half(100);																		// [1/mM/ms]
	Bmax_SLlowsl = 37.38e-3*Vmyo/Vsl;													// [mM], SL buffering
	Bmax_SLlowj = 4.62e-3*Vmyo/Vjunc*0.1;												// [mM]
	koff_sll = __float2half(1300e-3);																	// [1/ms]
	kon_sll = __float2half(100);																		// [1/mM/ms]
	Bmax_SLhighsl = 13.35e-3*Vmyo/Vsl;													// [mM]
	Bmax_SLhighj = 1.65e-3*Vmyo/Vjunc*0.1;												// [mM]
	koff_slh = __float2half(30e-3);																	// [1/ms]
	kon_slh = __float2half(100);																		// [1/mM/ms]
	Bmax_Csqn = __float2half(2.7);																	// 140e-3*Vmyo/Vsr); [mM]
	koff_csqn = __float2half(65);																		// [1/ms]
	kon_csqn = __float2half(100);																		// [1/mM/ms]

	// I_Na: Fast Na Current
	am = 0.32*(d_initvalu_39+47.13)/(1-exp(-0.1*(d_initvalu_39+47.13)));
	bm = 0.08*exp(-d_initvalu_39/11);
	if(d_initvalu_39 >= -40.0){
		ah = __float2half(0); aj = __float2half(0);
		bh = 1/(0.13*(1+exp(-(d_initvalu_39+10.66)/11.1)));
		bj = 0.3*exp(-2.535e-7*d_initvalu_39)/(1+exp(-0.1*(d_initvalu_39+32)));
	}
	else{
		ah = 0.135*exp((80+d_initvalu_39)/-6.8);
		bh = 3.56*exp(0.079*d_initvalu_39)+3.1e5*exp(0.35*d_initvalu_39);
		aj = (-127140*exp(0.2444*d_initvalu_39)-3.474e-5*exp(-0.04391*d_initvalu_39))*(d_initvalu_39+37.78)/(1+exp(0.311*(d_initvalu_39+79.23)));
		bj = 0.1212*exp(-0.01052*d_initvalu_39)/(1+exp(-0.1378*(d_initvalu_39+40.14)));
	}
	d_finavalu[offset_1] = am*(1-d_initvalu_1)-bm*d_initvalu_1;
	d_finavalu[offset_2] = ah*(1-d_initvalu_2)-bh*d_initvalu_2;
	d_finavalu[offset_3] = aj*(1-d_initvalu_3)-bj*d_initvalu_3;
	I_Na_junc = Fjunc*GNa*pow(d_initvalu_1,3)*d_initvalu_2*d_initvalu_3*(d_initvalu_39-ena_junc);
	I_Na_sl = Fsl*GNa*pow(d_initvalu_1,3)*d_initvalu_2*d_initvalu_3*(d_initvalu_39-ena_sl);
	// I_Na = I_Na_junc+I_Na_sl;

	// I_nabk: Na Background Current
	I_nabk_junc = Fjunc*GNaB*(d_initvalu_39-ena_junc);
	I_nabk_sl = Fsl*GNaB*(d_initvalu_39-ena_sl);
	// I_nabk = I_nabk_junc+I_nabk_sl;

	// I_nak: Na/K Pump Current
	sigma = (exp(Nao/67.3)-1)/7;
	fnak = 1/(1+0.1245*exp(-0.1*d_initvalu_39*FoRT)+0.0365*sigma*exp(-d_initvalu_39*FoRT));
	I_nak_junc = Fjunc*IbarNaK*fnak*Ko /(1+pow((KmNaip/d_initvalu_32),4)) /(Ko+KmKo);
	I_nak_sl = Fsl*IbarNaK*fnak*Ko /(1+pow((KmNaip/d_initvalu_33),4)) /(Ko+KmKo);
	I_nak = I_nak_junc+I_nak_sl;

	// I_kr: Rapidly Activating K Current
	gkr = 0.03*sqrt(Ko/5.4);
	xrss = 1/(1+exp(-(d_initvalu_39+50)/7.5));
	tauxr = 1/(0.00138*(d_initvalu_39+7)/(1-exp(-0.123*(d_initvalu_39+7)))+6.1e-4*(d_initvalu_39+10)/(exp(0.145*(d_initvalu_39+10))-1));
	d_finavalu[offset_12] = (xrss-d_initvalu_12)/tauxr;
	rkr = 1/(1+exp((d_initvalu_39+33)/22.4));
	I_kr = gkr*d_initvalu_12*rkr*(d_initvalu_39-ek);

	// I_ks: Slowly Activating K Current
	pcaks_junc = -log10(d_initvalu_36)+3.0;
	pcaks_sl = -log10(d_initvalu_37)+3.0;
	gks_junc = 0.07*(0.057 +0.19/(1+ exp((-7.2+pcaks_junc)/0.6)));
	gks_sl = 0.07*(0.057 +0.19/(1+ exp((-7.2+pcaks_sl)/0.6)));
	eks = (1/FoRT)*log((Ko+pNaK*Nao)/(d_initvalu_35+pNaK*d_initvalu_34));
	xsss = 1/(1+exp(-(d_initvalu_39-1.5)/16.7));
	tauxs = 1/(7.19e-5*(d_initvalu_39+30)/(1-exp(-0.148*(d_initvalu_39+30)))+1.31e-4*(d_initvalu_39+30)/(exp(0.0687*(d_initvalu_39+30))-1));
	d_finavalu[offset_13] = (xsss-d_initvalu_13)/tauxs;
	I_ks_junc = Fjunc*gks_junc*pow(d_initvalu_12,2)*(d_initvalu_39-eks);
	I_ks_sl = Fsl*gks_sl*pow(d_initvalu_13,2)*(d_initvalu_39-eks);
	I_ks = I_ks_junc+I_ks_sl;

	// I_kp: Plateau K current
	kp_kp = 1/(1+exp(7.488-d_initvalu_39/5.98));
	I_kp_junc = Fjunc*gkp*kp_kp*(d_initvalu_39-ek);
	I_kp_sl = Fsl*gkp*kp_kp*(d_initvalu_39-ek);
	I_kp = I_kp_junc+I_kp_sl;

	// I_to: Transient Outward K Current (slow and fast components)
	xtoss = 1/(1+exp(-(d_initvalu_39+3.0)/15));
	ytoss = 1/(1+exp((d_initvalu_39+33.5)/10));
	rtoss = 1/(1+exp((d_initvalu_39+33.5)/10));
	tauxtos = 9/(1+exp((d_initvalu_39+3.0)/15))+0.5;
	tauytos = 3e3/(1+exp((d_initvalu_39+60.0)/10))+30;
	taurtos = 2800/(1+exp((d_initvalu_39+60.0)/10))+220;
	d_finavalu[offset_8] = (xtoss-d_initvalu_8)/tauxtos;
	d_finavalu[offset_9] = (ytoss-d_initvalu_9)/tauytos;
	d_finavalu[offset_40]= (rtoss-d_initvalu_40)/taurtos;
	I_tos = GtoSlow*d_initvalu_8*(d_initvalu_9+0.5*d_initvalu_40)*(d_initvalu_39-ek);									// [uA/uF]

	//
	tauxtof = 3.5*exp(-d_initvalu_39*d_initvalu_39/30/30)+1.5;
	tauytof = 20.0/(1+exp((d_initvalu_39+33.5)/10))+20.0;
	d_finavalu[offset_10] = (xtoss-d_initvalu_10)/tauxtof;
	d_finavalu[offset_11] = (ytoss-d_initvalu_11)/tauytof;
	I_tof = GtoFast*d_initvalu_10*d_initvalu_11*(d_initvalu_39-ek);
	I_to = I_tos + I_tof;

	// I_ki: Time-Independent K Current
	aki = 1.02/(1+exp(0.2385*(d_initvalu_39-ek-59.215)));
	bki =(0.49124*exp(0.08032*(d_initvalu_39+5.476-ek)) + exp(0.06175*(d_initvalu_39-ek-594.31))) /(1 + exp(-0.5143*(d_initvalu_39-ek+4.753)));
	kiss = aki/(aki+bki);
	I_ki = 0.9*sqrt(Ko/5.4)*kiss*(d_initvalu_39-ek);

	// I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
	I_ClCa_junc = Fjunc*GClCa/(1+KdClCa/d_initvalu_36)*(d_initvalu_39-ecl);
	I_ClCa_sl = Fsl*GClCa/(1+KdClCa/d_initvalu_37)*(d_initvalu_39-ecl);
	I_ClCa = I_ClCa_junc+I_ClCa_sl;
	I_Clbk = GClB*(d_initvalu_39-ecl);

	// I_Ca: L-type Calcium Current
	dss = 1/(1+exp(-(d_initvalu_39+14.5)/6.0));
	taud = dss*(1-exp(-(d_initvalu_39+14.5)/6.0))/(0.035*(d_initvalu_39+14.5));
	fss = 1/(1+exp((d_initvalu_39+35.06)/3.6))+0.6/(1+exp((50-d_initvalu_39)/20));
	tauf = 1/(0.0197*exp(-pow(0.0337*(d_initvalu_39+14.5),2))+0.02);
	d_finavalu[offset_4] = (dss-d_initvalu_4)/taud;
	d_finavalu[offset_5] = (fss-d_initvalu_5)/tauf;
	d_finavalu[offset_6] = 1.7*d_initvalu_36*(1-d_initvalu_6)-11.9e-3*d_initvalu_6;											// fCa_junc
	d_finavalu[offset_7] = 1.7*d_initvalu_37*(1-d_initvalu_7)-11.9e-3*d_initvalu_7;											// fCa_sl

	//
	ibarca_j = pCa*4*(d_initvalu_39*Frdy*FoRT) * (0.341*d_initvalu_36*exp(2*d_initvalu_39*FoRT)-0.341*Cao) /(exp(2*d_initvalu_39*FoRT)-1);
	ibarca_sl = pCa*4*(d_initvalu_39*Frdy*FoRT) * (0.341*d_initvalu_37*exp(2*d_initvalu_39*FoRT)-0.341*Cao) /(exp(2*d_initvalu_39*FoRT)-1);
	ibark = pK*(d_initvalu_39*Frdy*FoRT)*(0.75*d_initvalu_35*exp(d_initvalu_39*FoRT)-0.75*Ko) /(exp(d_initvalu_39*FoRT)-1);
	ibarna_j = pNa*(d_initvalu_39*Frdy*FoRT) *(0.75*d_initvalu_32*exp(d_initvalu_39*FoRT)-0.75*Nao)  /(exp(d_initvalu_39*FoRT)-1);
	ibarna_sl = pNa*(d_initvalu_39*Frdy*FoRT) *(0.75*d_initvalu_33*exp(d_initvalu_39*FoRT)-0.75*Nao)  /(exp(d_initvalu_39*FoRT)-1);
	I_Ca_junc = (Fjunc_CaL*ibarca_j*d_initvalu_4*d_initvalu_5*(1-d_initvalu_6)*pow(Q10CaL,Qpow))*0.45;
	I_Ca_sl = (Fsl_CaL*ibarca_sl*d_initvalu_4*d_initvalu_5*(1-d_initvalu_7)*pow(Q10CaL,Qpow))*0.45;
	I_Ca = I_Ca_junc+I_Ca_sl;
	d_finavalu[offset_43]=-I_Ca*Cmem/(Vmyo*2*Frdy)*1e3;
	I_CaK = (ibark*d_initvalu_4*d_initvalu_5*(Fjunc_CaL*(1-d_initvalu_6)+Fsl_CaL*(1-d_initvalu_7))*pow(Q10CaL,Qpow))*0.45;
	I_CaNa_junc = (Fjunc_CaL*ibarna_j*d_initvalu_4*d_initvalu_5*(1-d_initvalu_6)*pow(Q10CaL,Qpow))*0.45;
	I_CaNa_sl = (Fsl_CaL*ibarna_sl*d_initvalu_4*d_initvalu_5*(1-d_initvalu_7)*pow(Q10CaL,Qpow))*0.45;
	// I_CaNa = I_CaNa_junc+I_CaNa_sl;
	// I_Catot = I_Ca+I_CaK+I_CaNa;

	// I_ncx: Na/Ca Exchanger flux
	Ka_junc = 1/(1+pow((Kdact/d_initvalu_36),3));
	Ka_sl = 1/(1+pow((Kdact/d_initvalu_37),3));
	s1_junc = exp(nu*d_initvalu_39*FoRT)*pow(d_initvalu_32,3)*Cao;
	s1_sl = exp(nu*d_initvalu_39*FoRT)*pow(d_initvalu_33,3)*Cao;
	s2_junc = exp((nu-1)*d_initvalu_39*FoRT)*pow(Nao,3)*d_initvalu_36;
	s3_junc = (KmCai*pow(Nao,3)*(1+pow((d_initvalu_32/KmNai),3))+pow(KmNao,3)*d_initvalu_36+ pow(KmNai,3)*Cao*(1+d_initvalu_36/KmCai)+KmCao*pow(d_initvalu_32,3)+pow(d_initvalu_32,3)*Cao+pow(Nao,3)*d_initvalu_36)*(1+ksat*exp((nu-1)*d_initvalu_39*FoRT));
	s2_sl = exp((nu-1)*d_initvalu_39*FoRT)*pow(Nao,3)*d_initvalu_37;
	s3_sl = (KmCai*pow(Nao,3)*(1+pow((d_initvalu_33/KmNai),3)) + pow(KmNao,3)*d_initvalu_37+pow(KmNai,3)*Cao*(1+d_initvalu_37/KmCai)+KmCao*pow(d_initvalu_33,3)+pow(d_initvalu_33,3)*Cao+pow(Nao,3)*d_initvalu_37)*(1+ksat*exp((nu-1)*d_initvalu_39*FoRT));
	I_ncx_junc = Fjunc*IbarNCX*pow(Q10NCX,Qpow)*Ka_junc*(s1_junc-s2_junc)/s3_junc;
	I_ncx_sl = Fsl*IbarNCX*pow(Q10NCX,Qpow)*Ka_sl*(s1_sl-s2_sl)/s3_sl;
	I_ncx = I_ncx_junc+I_ncx_sl;
	d_finavalu[offset_45]=2*I_ncx*Cmem/(Vmyo*2*Frdy)*1e3;

	// I_pca: Sarcolemmal Ca Pump Current
	I_pca_junc = Fjunc*pow(Q10SLCaP,Qpow)*IbarSLCaP*pow(d_initvalu_36,__float2half(1.6))/(pow(KmPCa,__float2half(1.6))+pow(d_initvalu_36,__float2half(1.6)));
	I_pca_sl = Fsl*pow(Q10SLCaP,Qpow)*IbarSLCaP*pow(d_initvalu_37,__float2half(1.6))/(pow(KmPCa,__float2half(1.6))+pow(d_initvalu_37,__float2half(1.6)));
	I_pca = I_pca_junc+I_pca_sl;
	d_finavalu[offset_44]=-I_pca*Cmem/(Vmyo*2*Frdy)*1e3;

	// I_cabk: Ca Background Current
	I_cabk_junc = Fjunc*GCaB*(d_initvalu_39-eca_junc);
	I_cabk_sl = Fsl*GCaB*(d_initvalu_39-eca_sl);
	I_cabk = I_cabk_junc+I_cabk_sl;
	d_finavalu[offset_46]=-I_cabk*Cmem/(Vmyo*2*Frdy)*1e3;

	// SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
	MaxSR = __float2half(15);
	MinSR = __float2half(1);
	kCaSR = MaxSR - (MaxSR-MinSR)/(1+pow(ec50SR/d_initvalu_31,__float2half(2.5)));
	koSRCa = koCa/kCaSR;
	kiSRCa = kiCa*kCaSR;
	RI = 1-d_initvalu_14-d_initvalu_15-d_initvalu_16;
	d_finavalu[offset_14] = (kim*RI-kiSRCa*d_initvalu_36*d_initvalu_14)-(koSRCa*pow(d_initvalu_36,2)*d_initvalu_14-kom*d_initvalu_15);			// R
	d_finavalu[offset_15] = (koSRCa*pow(d_initvalu_36,2)*d_initvalu_14-kom*d_initvalu_15)-(kiSRCa*d_initvalu_36*d_initvalu_15-kim*d_initvalu_16);			// O
	d_finavalu[offset_16] = (kiSRCa*d_initvalu_36*d_initvalu_15-kim*d_initvalu_16)-(kom*d_initvalu_16-koSRCa*pow(d_initvalu_36,2)*RI);			// I
	J_SRCarel = ks*d_initvalu_15*(d_initvalu_31-d_initvalu_36);													// [mM/ms]
	J_serca = pow(Q10SRCaP,Qpow)*Vmax_SRCaP*(pow((d_initvalu_38/Kmf),hillSRCaP)-pow((d_initvalu_31/Kmr),hillSRCaP))
										 /(1+pow((d_initvalu_38/Kmf),hillSRCaP)+pow((d_initvalu_31/Kmr),hillSRCaP));
	J_SRleak = 5.348e-6*(d_initvalu_31-d_initvalu_36);													//   [mM/ms]

	// Sodium and Calcium Buffering
	d_finavalu[offset_17] = kon_na*d_initvalu_32*(Bmax_Naj-d_initvalu_17)-koff_na*d_initvalu_17;								// NaBj      [mM/ms]
	d_finavalu[offset_18] = kon_na*d_initvalu_33*(Bmax_Nasl-d_initvalu_18)-koff_na*d_initvalu_18;							// NaBsl     [mM/ms]

	// Cytosolic Ca Buffers
	d_finavalu[offset_19] = kon_tncl*d_initvalu_38*(Bmax_TnClow-d_initvalu_19)-koff_tncl*d_initvalu_19;						// TnCL      [mM/ms]
	d_finavalu[offset_20] = kon_tnchca*d_initvalu_38*(Bmax_TnChigh-d_initvalu_20-d_initvalu_21)-koff_tnchca*d_initvalu_20;			// TnCHc     [mM/ms]
	d_finavalu[offset_21] = kon_tnchmg*Mgi*(Bmax_TnChigh-d_initvalu_20-d_initvalu_21)-koff_tnchmg*d_initvalu_21;				// TnCHm     [mM/ms]
	d_finavalu[offset_22] = __float2half(0);																		// CaM       [mM/ms]
	d_finavalu[offset_23] = kon_myoca*d_initvalu_38*(Bmax_myosin-d_initvalu_23-d_initvalu_24)-koff_myoca*d_initvalu_23;				// Myosin_ca [mM/ms]
	d_finavalu[offset_24] = kon_myomg*Mgi*(Bmax_myosin-d_initvalu_23-d_initvalu_24)-koff_myomg*d_initvalu_24;				// Myosin_mg [mM/ms]
	d_finavalu[offset_25] = kon_sr*d_initvalu_38*(Bmax_SR-d_initvalu_25)-koff_sr*d_initvalu_25;								// SRB       [mM/ms]
	J_CaB_cytosol = d_finavalu[offset_19] + d_finavalu[offset_20] + d_finavalu[offset_21] + d_finavalu[offset_22] + d_finavalu[offset_23] + d_finavalu[offset_24] + d_finavalu[offset_25];

	// Junctional and SL Ca Buffers
	d_finavalu[offset_26] = kon_sll*d_initvalu_36*(Bmax_SLlowj-d_initvalu_26)-koff_sll*d_initvalu_26;						// SLLj      [mM/ms]
	d_finavalu[offset_27] = kon_sll*d_initvalu_37*(Bmax_SLlowsl-d_initvalu_27)-koff_sll*d_initvalu_27;						// SLLsl     [mM/ms]
	d_finavalu[offset_28] = kon_slh*d_initvalu_36*(Bmax_SLhighj-d_initvalu_28)-koff_slh*d_initvalu_28;						// SLHj      [mM/ms]
	d_finavalu[offset_29] = kon_slh*d_initvalu_37*(Bmax_SLhighsl-d_initvalu_29)-koff_slh*d_initvalu_29;						// SLHsl     [mM/ms]
	J_CaB_junction = d_finavalu[offset_26]+d_finavalu[offset_28];
	J_CaB_sl = d_finavalu[offset_27]+d_finavalu[offset_29];

	// SR Ca Concentrations
	d_finavalu[offset_30] = kon_csqn*d_initvalu_31*(Bmax_Csqn-d_initvalu_30)-koff_csqn*d_initvalu_30;						// Csqn      [mM/ms]
	oneovervsr = 1/Vsr;
	d_finavalu[offset_31] = J_serca*Vmyo*oneovervsr-(J_SRleak*Vmyo*oneovervsr+J_SRCarel)-d_finavalu[offset_30];   // Ca_sr     [mM/ms] %Ratio 3 leak current

	// Sodium Concentrations
	I_Na_tot_junc = I_Na_junc+I_nabk_junc+3*I_ncx_junc+3*I_nak_junc+I_CaNa_junc;		// [uA/uF]
	I_Na_tot_sl = I_Na_sl+I_nabk_sl+3*I_ncx_sl+3*I_nak_sl+I_CaNa_sl;					// [uA/uF]
	d_finavalu[offset_32] = -I_Na_tot_junc*Cmem/(Vjunc*Frdy)+J_na_juncsl/Vjunc*(d_initvalu_33-d_initvalu_32)-d_finavalu[offset_17];
	oneovervsl = 1/Vsl;
	d_finavalu[offset_33] = -I_Na_tot_sl*Cmem*oneovervsl/Frdy+J_na_juncsl*oneovervsl*(d_initvalu_32-d_initvalu_33)+J_na_slmyo*oneovervsl*(d_initvalu_34-d_initvalu_33)-d_finavalu[offset_18];
	d_finavalu[offset_34] = J_na_slmyo/Vmyo*(d_initvalu_33-d_initvalu_34);											// [mM/msec]

	// Potassium Concentration
	I_K_tot = I_to+I_kr+I_ks+I_ki-2*I_nak+I_CaK+I_kp;									// [uA/uF]
	d_finavalu[offset_35] = __float2half(0);															// [mM/msec]

	// Calcium Concentrations
	I_Ca_tot_junc = I_Ca_junc+I_cabk_junc+I_pca_junc-2*I_ncx_junc;						// [uA/uF]
	I_Ca_tot_sl = I_Ca_sl+I_cabk_sl+I_pca_sl-2*I_ncx_sl;								// [uA/uF]
	d_finavalu[offset_36] = -I_Ca_tot_junc*Cmem/(Vjunc*2*Frdy)+J_ca_juncsl/Vjunc*(d_initvalu_37-d_initvalu_36)
	         - J_CaB_junction+(J_SRCarel)*Vsr/Vjunc+J_SRleak*Vmyo/Vjunc;				// Ca_j
	d_finavalu[offset_37] = -I_Ca_tot_sl*Cmem/(Vsl*2*Frdy)+J_ca_juncsl/Vsl*(d_initvalu_36-d_initvalu_37)
	         + J_ca_slmyo/Vsl*(d_initvalu_38-d_initvalu_37)-J_CaB_sl;									// Ca_sl
	d_finavalu[offset_38] = -J_serca-J_CaB_cytosol +J_ca_slmyo/Vmyo*(d_initvalu_37-d_initvalu_38);
	// junc_sl=J_ca_juncsl/Vsl*(d_initvalu_36-d_initvalu_37);
	// sl_junc=J_ca_juncsl/Vjunc*(d_initvalu_37-d_initvalu_36);
	// sl_myo=J_ca_slmyo/Vsl*(d_initvalu_38-d_initvalu_37);
	// myo_sl=J_ca_slmyo/Vmyo*(d_initvalu_37-d_initvalu_38);

	// Simulation type
	state = 1;
	switch(state){
		case 0:
			I_app = __float2half(0);
			break;
		case 1:																			// pace w/ current injection at cycleLength 'cycleLength'
			if(fmod(timeinst,cycleLength) <= 5){
				I_app = __float2half(9.5);
			}
			else{
				I_app = __float2half(0.0);
			}
			break;
		case 2:
			V_hold = __float2half(-55);
			V_test = __float2half(0);
			if(timeinst>0.5 & timeinst<200.5){
				V_clamp = V_test;
			}
			else{
				V_clamp = V_hold;
			}
			R_clamp = __float2half(0.04);
			I_app = (V_clamp-d_initvalu_39)/R_clamp;
			break;
	}

	// Membrane Potential
	I_Na_tot = I_Na_tot_junc + I_Na_tot_sl;												// [uA/uF]
	I_Cl_tot = I_ClCa+I_Clbk;															// [uA/uF]
	I_Ca_tot = I_Ca_tot_junc+I_Ca_tot_sl;
	I_tot = I_Na_tot+I_Cl_tot+I_Ca_tot+I_K_tot;
	d_finavalu[offset_39] = -(I_tot-I_app);

	// Set unused output values to 0 (MATLAB does it by default)
	d_finavalu[offset_41] = __float2half(0);
	d_finavalu[offset_42] = __float2half(0);

}
//=====================================================================
//	MAIN FUNCTION
//=====================================================================
__device__ void kernel_cam_2(	fp timeinst,
													fp* d_initvalu,
													fp *d_finavalu,
													int valu_offset,
													fp* d_params,
													int params_offset,
													fp* d_com,
													int com_offset,
													fp Ca){

	//=====================================================================
	//	VARIABLES
	//=====================================================================

	// inputs
	// fp CaMtot;
	fp Btot;
	fp CaMKIItot;
	fp CaNtot;
	fp PP1tot;
	fp K;
	fp Mg;

	// variable references
	int offset_1;
	int offset_2;
	int offset_3;
	int offset_4;
	int offset_5;
	int offset_6;
	int offset_7;
	int offset_8;
	int offset_9;
	int offset_10;
	int offset_11;
	int offset_12;
	int offset_13;
	int offset_14;
	int offset_15;

	// decoding input array
	fp CaM;
	fp Ca2CaM;
	fp Ca4CaM;
	fp CaMB;
	fp Ca2CaMB;
	fp Ca4CaMB;
	fp Pb2;
	fp Pb;
	fp Pt;
	fp Pt2;
	fp Pa;
	fp Ca4CaN;
	fp CaMCa4CaN;
	fp Ca2CaMCa4CaN;
	fp Ca4CaMCa4CaN;

	// Ca/CaM parameters
	fp Kd02;																		// [uM^2]
	fp Kd24;																		// [uM^2]
	fp k20;																			// [s^-1]
	fp k02;																			// [uM^-2 s^-1]
	fp k42;																			// [s^-1]
	fp k24;																			// [uM^-2 s^-1]

	// CaM buffering (B) parameters
	fp k0Boff;																		// [s^-1]
	fp k0Bon;																		// [uM^-1 s^-1] kon = koff/Kd
	fp k2Boff;																		// [s^-1]
	fp k2Bon;																		// [uM^-1 s^-1]
	fp k4Boff;																		// [s^-1]
	fp k4Bon;																		// [uM^-1 s^-1]

	// using thermodynamic constraints
	fp k20B;																		// [s^-1] thermo constraint on loop 1
	fp k02B;																		// [uM^-2 s^-1]
	fp k42B;																		// [s^-1] thermo constraint on loop 2
	fp k24B;																		// [uM^-2 s^-1]

	// Wi Wa Wt Wp
	fp kbi;																			// [s^-1] (Ca4CaM dissocation from Wb)
	fp kib;																			// [uM^-1 s^-1]
	fp kpp1;																		// [s^-1] (PP1-dep dephosphorylation rates)
	fp Kmpp1;																		// [uM]
	fp kib2;
	fp kb2i;
	fp kb24;
	fp kb42;
	fp kta;																			// [s^-1] (Ca4CaM dissociation from Wt)
	fp kat;																			// [uM^-1 s^-1] (Ca4CaM reassociation with Wa)
	fp kt42;
	fp kt24;
	fp kat2;
	fp kt2a;

	// CaN parameters
	fp kcanCaoff;																	// [s^-1]
	fp kcanCaon;																	// [uM^-1 s^-1]
	fp kcanCaM4on;																	// [uM^-1 s^-1]
	fp kcanCaM4off;																	// [s^-1]
	fp kcanCaM2on;
	fp kcanCaM2off;
	fp kcanCaM0on;
	fp kcanCaM0off;
	fp k02can;
	fp k20can;
	fp k24can;
	fp k42can;

	// CaM Reaction fluxes
	fp rcn02;
	fp rcn24;

	// CaM buffer fluxes
	fp B;
	fp rcn02B;
	fp rcn24B;
	fp rcn0B;
	fp rcn2B;
	fp rcn4B;

	// CaN reaction fluxes
	fp Ca2CaN;
	fp rcnCa4CaN;
	fp rcn02CaN;
	fp rcn24CaN;
	fp rcn0CaN;
	fp rcn2CaN;
	fp rcn4CaN;

	// CaMKII reaction fluxes
	fp Pix;
	fp rcnCKib2;
	fp rcnCKb2b;
	fp rcnCKib;
	fp T;
	fp kbt;
	fp rcnCKbt;
	fp rcnCKtt2;
	fp rcnCKta;
	fp rcnCKt2a;
	fp rcnCKt2b2;
	fp rcnCKai;

	// CaM equations
	fp dCaM;
	fp dCa2CaM;
	fp dCa4CaM;
	fp dCaMB;
	fp dCa2CaMB;
	fp dCa4CaMB;

	// CaMKII equations
	fp dPb2;																					// Pb2
	fp dPb;																					// Pb
	fp dPt;																					// Pt
	fp dPt2;																					// Pt2
	fp dPa;																					// Pa

	// CaN equations
	fp dCa4CaN;																			// Ca4CaN
	fp dCaMCa4CaN;																	// CaMCa4CaN
	fp dCa2CaMCa4CaN;																// Ca2CaMCa4CaN
	fp dCa4CaMCa4CaN;																// Ca4CaMCa4CaN

	//=====================================================================
	//	EXECUTION
	//=====================================================================

	// inputs
	// CaMtot = d_params[params_offset];
	Btot = d_params[params_offset+1];
	CaMKIItot = d_params[params_offset+2];
	CaNtot = d_params[params_offset+3];
	PP1tot = d_params[params_offset+4];
	K = d_params[16];
	Mg = d_params[17];

	// variable references
	offset_1 = valu_offset;
	offset_2 = valu_offset+1;
	offset_3 = valu_offset+2;
	offset_4 = valu_offset+3;
	offset_5 = valu_offset+4;
	offset_6 = valu_offset+5;
	offset_7 = valu_offset+6;
	offset_8 = valu_offset+7;
	offset_9 = valu_offset+8;
	offset_10 = valu_offset+9;
	offset_11 = valu_offset+10;
	offset_12 = valu_offset+11;
	offset_13 = valu_offset+12;
	offset_14 = valu_offset+13;
	offset_15 = valu_offset+14;

	// decoding input array
	CaM				= d_initvalu[offset_1];
	Ca2CaM			= d_initvalu[offset_2];
	Ca4CaM			= d_initvalu[offset_3];
	CaMB			= d_initvalu[offset_4];
	Ca2CaMB			= d_initvalu[offset_5];
	Ca4CaMB			= d_initvalu[offset_6];
	Pb2				= d_initvalu[offset_7];
	Pb				= d_initvalu[offset_8];
	Pt				= d_initvalu[offset_9];
	Pt2				= d_initvalu[offset_10];
	Pa				= d_initvalu[offset_11];
	Ca4CaN			= d_initvalu[offset_12];
	CaMCa4CaN		= d_initvalu[offset_13];
	Ca2CaMCa4CaN	= d_initvalu[offset_14];
	Ca4CaMCa4CaN	= d_initvalu[offset_15];

	// Ca/CaM parameters
	if (Mg <= 1){
		Kd02 = 0.0025*(1+K/0.94-Mg/0.012)*(1+K/8.1+Mg/0.022);							// [uM^2]
		Kd24 = 0.128*(1+K/0.64+Mg/0.0014)*(1+K/13.0-Mg/0.153);							// [uM^2]
	}
	else{
		Kd02 = 0.0025*(1+K/0.94-1/0.012+(Mg-1)/0.060)*(1+K/8.1+1/0.022+(Mg-1)/0.068);   // [uM^2]
		Kd24 = 0.128*(1+K/0.64+1/0.0014+(Mg-1)/0.005)*(1+K/13.0-1/0.153+(Mg-1)/0.150);  // [uM^2]
	}
	k20 = __float2half(10);																			// [s^-1]
	k02 = k20/Kd02;																		// [uM^-2 s^-1]
	k42 = __float2half(500);																			// [s^-1]
	k24 = k42/Kd24;																		// [uM^-2 s^-1]

	// CaM buffering (B) parameters
	k0Boff = __float2half(0.0014);																	// [s^-1]
	k0Bon = k0Boff/0.2;																	// [uM^-1 s^-1] kon = koff/Kd
	k2Boff = k0Boff/100;																// [s^-1]
	k2Bon = k0Bon;																		// [uM^-1 s^-1]
	k4Boff = k2Boff;																	// [s^-1]
	k4Bon = k0Bon;																		// [uM^-1 s^-1]

	// using thermodynamic constraints
	k20B = k20/100;																		// [s^-1] thermo constraint on loop 1
	k02B = k02;																			// [uM^-2 s^-1]
	k42B = k42;																			// [s^-1] thermo constraint on loop 2
	k24B = k24;																			// [uM^-2 s^-1]

	// Wi Wa Wt Wp
	kbi = __float2half(2.2);																			// [s^-1] (Ca4CaM dissocation from Wb)
	kib = kbi/33.5e-3;																	// [uM^-1 s^-1]
	kpp1 = __float2half(1.72);																		// [s^-1] (PP1-dep dephosphorylation rates)
	Kmpp1 = __float2half(11.5);																		// [uM]
	kib2 = kib;
	kb2i = kib2*5;
	kb24 = k24;
	kb42 = k42*33.5e-3/5;
	kta = kbi/1000;																		// [s^-1] (Ca4CaM dissociation from Wt)
	kat = kib;																			// [uM^-1 s^-1] (Ca4CaM reassociation with Wa)
	kt42 = k42*33.5e-6/5;
	kt24 = k24;
	kat2 = kib;
	kt2a = kib*5;

	// CaN parameters
	kcanCaoff = __float2half(1);																		// [s^-1]
	kcanCaon = kcanCaoff/0.5;															// [uM^-1 s^-1]
	kcanCaM4on = __float2half(46);																	// [uM^-1 s^-1]
	kcanCaM4off = __float2half(0.0013);																// [s^-1]
	kcanCaM2on = kcanCaM4on;
	kcanCaM2off = 2508*kcanCaM4off;
	kcanCaM0on = kcanCaM4on;
	kcanCaM0off = 165*kcanCaM2off;
	k02can = k02;
	k20can = k20/165;
	k24can = k24;
	k42can = k20/2508;

	// CaM Reaction fluxes
	rcn02 = k02*pow(Ca,2)*CaM - k20*Ca2CaM;
	rcn24 = k24*pow(Ca,2)*Ca2CaM - k42*Ca4CaM;

	// CaM buffer fluxes
	B = Btot - CaMB - Ca2CaMB - Ca4CaMB;
	rcn02B = k02B*pow(Ca,2)*CaMB - k20B*Ca2CaMB;
	rcn24B = k24B*pow(Ca,2)*Ca2CaMB - k42B*Ca4CaMB;
	rcn0B = k0Bon*CaM*B - k0Boff*CaMB;
	rcn2B = k2Bon*Ca2CaM*B - k2Boff*Ca2CaMB;
	rcn4B = k4Bon*Ca4CaM*B - k4Boff*Ca4CaMB;

	// CaN reaction fluxes
	Ca2CaN = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN;
	rcnCa4CaN = kcanCaon*pow(Ca,2)*Ca2CaN - kcanCaoff*Ca4CaN;
	rcn02CaN = k02can*pow(Ca,2)*CaMCa4CaN - k20can*Ca2CaMCa4CaN;
	rcn24CaN = k24can*pow(Ca,2)*Ca2CaMCa4CaN - k42can*Ca4CaMCa4CaN;
	rcn0CaN = kcanCaM0on*CaM*Ca4CaN - kcanCaM0off*CaMCa4CaN;
	rcn2CaN = kcanCaM2on*Ca2CaM*Ca4CaN - kcanCaM2off*Ca2CaMCa4CaN;
	rcn4CaN = kcanCaM4on*Ca4CaM*Ca4CaN - kcanCaM4off*Ca4CaMCa4CaN;

	// CaMKII reaction fluxes
	Pix = 1 - Pb2 - Pb - Pt - Pt2 - Pa;
	rcnCKib2 = kib2*Ca2CaM*Pix - kb2i*Pb2;
	rcnCKb2b = kb24*pow(Ca,2)*Pb2 - kb42*Pb;
	rcnCKib = kib*Ca4CaM*Pix - kbi*Pb;
	T = Pb + Pt + Pt2 + Pa;
	kbt = 0.055*T + 0.0074*pow(T,2) + 0.015*pow(T,3);
	rcnCKbt = kbt*Pb - kpp1*PP1tot*Pt/(Kmpp1+CaMKIItot*Pt);
	rcnCKtt2 = kt42*Pt - kt24*pow(Ca,2)*Pt2;
	rcnCKta = kta*Pt - kat*Ca4CaM*Pa;
	rcnCKt2a = kt2a*Pt2 - kat2*Ca2CaM*Pa;
	rcnCKt2b2 = kpp1*PP1tot*Pt2/(Kmpp1+CaMKIItot*Pt2);
	rcnCKai = kpp1*PP1tot*Pa/(Kmpp1+CaMKIItot*Pa);

	// CaM equations
	dCaM = 1e-3*(-rcn02 - rcn0B - rcn0CaN);
	dCa2CaM = 1e-3*(rcn02 - rcn24 - rcn2B - rcn2CaN + CaMKIItot*(-rcnCKib2 + rcnCKt2a) );
	dCa4CaM = 1e-3*(rcn24 - rcn4B - rcn4CaN + CaMKIItot*(-rcnCKib+rcnCKta) );
	dCaMB = 1e-3*(rcn0B-rcn02B);
	dCa2CaMB = 1e-3*(rcn02B + rcn2B - rcn24B);
	dCa4CaMB = 1e-3*(rcn24B + rcn4B);

	// CaMKII equations
	dPb2 = 1e-3*(rcnCKib2 - rcnCKb2b + rcnCKt2b2);										// Pb2
	dPb = 1e-3*(rcnCKib + rcnCKb2b - rcnCKbt);											// Pb
	dPt = 1e-3*(rcnCKbt-rcnCKta-rcnCKtt2);												// Pt
	dPt2 = 1e-3*(rcnCKtt2-rcnCKt2a-rcnCKt2b2);											// Pt2
	dPa = 1e-3*(rcnCKta+rcnCKt2a-rcnCKai);												// Pa

	// CaN equations
	dCa4CaN = 1e-3*(rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN);							// Ca4CaN
	dCaMCa4CaN = 1e-3*(rcn0CaN - rcn02CaN);												// CaMCa4CaN
	dCa2CaMCa4CaN = 1e-3*(rcn2CaN+rcn02CaN-rcn24CaN);									// Ca2CaMCa4CaN
	dCa4CaMCa4CaN = 1e-3*(rcn4CaN+rcn24CaN);											// Ca4CaMCa4CaN

	// encode output array
	d_finavalu[offset_1] = dCaM;
	d_finavalu[offset_2] = dCa2CaM;
	d_finavalu[offset_3] = dCa4CaM;
	d_finavalu[offset_4] = dCaMB;
	d_finavalu[offset_5] = dCa2CaMB;
	d_finavalu[offset_6] = dCa4CaMB;
	d_finavalu[offset_7] = dPb2;
	d_finavalu[offset_8] = dPb;
	d_finavalu[offset_9] = dPt;
	d_finavalu[offset_10] = dPt2;
	d_finavalu[offset_11] = dPa;
	d_finavalu[offset_12] = dCa4CaN;
	d_finavalu[offset_13] = dCaMCa4CaN;
	d_finavalu[offset_14] = dCa2CaMCa4CaN;
	d_finavalu[offset_15] = dCa4CaMCa4CaN;

	// write to global variables for adjusting Ca buffering in EC coupling model
	d_finavalu[com_offset] = 1e-3*(2*CaMKIItot*(rcnCKtt2-rcnCKb2b) - 2*(rcn02+rcn24+rcn02B+rcn24B+rcnCa4CaN+rcn02CaN+rcn24CaN)); // [uM/msec]
	//d_finavalu[JCa] = 1; // [uM/msec]

}
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//		KERNEL
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

__device__ void kernel_2(	int timeinst,
											fp* initvalu,
											fp* params,
											fp* finavalu,
											fp* com){

	//======================================================================================================================================================
	// 	VARIABLES
	//======================================================================================================================================================

	// pointers
	int valu_offset_ecc;															// inivalu and finavalu offset
	int valu_offset_Dyad;														// Dyad value offset
	int valu_offset_SL;																// SL value offset
	int valu_offset_Cyt;															// Cyt value offset

	int params_offset_Dyad;													// Dyad parameters offset
	int params_offset_SL;														// SL parameters offset
	int params_offset_Cyt;														// Cyt parameters offset

	int com_offset_Dyad;															// kernel1-kernel2 Dyad communication offset
	int com_offset_SL;																// kernel1-kernel2 SL communication offset
	int com_offset_Cyt;															// kernel-kernel Cyt communication offset

	// module parameters
	fp CaDyad;																			// from ECC model, *** Converting from [mM] to [uM] ***
	fp CaSL;																				// from ECC model, *** Converting from [mM] to [uM] ***
	fp CaCyt;																			// from ECC model, *** Converting from [mM] to [uM] ***

	// counter
	int i;

	//======================================================================================================================================================
	// 	COMPUTATION
	//======================================================================================================================================================

	valu_offset_ecc = 0;
	valu_offset_Dyad = 46;
	valu_offset_SL = 61;
	valu_offset_Cyt = 76;

	params_offset_Dyad = 0;
	params_offset_SL = 5;
	params_offset_Cyt = 10;

	com_offset_Dyad = 0;
	com_offset_SL = 1;
	com_offset_Cyt = 2;

	//==================================================
	//		ECC
	//==================================================
	fp timeinst_half = __float2half(timeinst);
	// ecc function
	kernel_ecc_2(timeinst_half,
							initvalu,
							finavalu,
							valu_offset_ecc,
							params);

	//==================================================
	//		3xCAM
	//==================================================

	// specific
	CaDyad = initvalu[35]*1e3;							// from ECC model, *** Converting from [mM] to [uM] ***
	// cam function for Dyad
	kernel_cam_2(	timeinst_half,
								initvalu,
								finavalu,
								valu_offset_Dyad,
								params,
								params_offset_Dyad,
								com,
								com_offset_Dyad,
								CaDyad);

	// specific
	CaSL = initvalu[36]*1e3;								// from ECC model, *** Converting from [mM] to [uM] ***
	// cam function for Dyad
	kernel_cam_2(	timeinst_half,
								initvalu,
								finavalu,
								valu_offset_SL,
								params,
								params_offset_SL,
								com,
								com_offset_SL,
								CaSL);

	// specific
	CaCyt = initvalu[37]*1e3;							// from ECC model, *** Converting from [mM] to [uM] ***
	// cam function for Dyad
	kernel_cam_2(	timeinst_half,
								initvalu,
								finavalu,
								valu_offset_Cyt,
								params,
								params_offset_Cyt,
								com,
								com_offset_Cyt,
								CaCyt);

	//====================================================================================================
	//		SEGMENT HAPPENING 2ND IN TIME: FINAL
	//====================================================================================================

	kernel_fin_2(	timeinst,
							initvalu,
							finavalu,
							valu_offset_ecc,
							valu_offset_Dyad,
							valu_offset_SL,
							valu_offset_Cyt,
							params,
							com);

	//====================================================================================================
	//		make sure function does not return NANs and INFs
	//====================================================================================================

	for(i=0; i<EQUATIONS; i++){
		if (isnan(finavalu[i]) == 1){
			finavalu[i] = __float2half(0.0001);												// for NAN set rate of change to 0.0001
		}
		else if (isinf(finavalu[i]) == 1){
			finavalu[i] = __float2half(0.0001);												// for INF set rate of change to 0.0001
		}
	}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//		END OF KERNEL
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

}
//====================================================================================================100
//		UPDATE
//====================================================================================================100

// This file is the modified version of embedded fehlberg 7.8 solver obrained from (http://mymathlib.webtrellis.net/index.html)
// Lukasz G. Szafaryn 15 DEC 09

//====================================================================================================100
//		DESCRIPTION
//====================================================================================================100

//                                                                            //
//  Description:                                                              //
//     The Runge-Kutta-Fehlberg method is an adaptive procedure for approxi-  //
//     mating the solution of the differential equation y'(x) = f(x,y) with   //
//     initial condition y(x0) = c.  This implementation evaluates f(x,y)     //
//     thirteen times per step using embedded seventh order and eight order   //
//     Runge-Kutta estimates to estimate the not only the solution but also   //
//     the error.                                                             //
//     The next step size is then calculated using the preassigned tolerance  //
//     and error estimate.                                                    //
//     For step i+1,                                                          //
//        y[i+1] = y[i] +  h * (41/840 * k1 + 34/105 * finavalu_temp[5] + 9/35 * finavalu_temp[6]         //
//                        + 9/35 * finavalu_temp[7] + 9/280 * finavalu_temp[8] + 9/280 finavalu_temp[9] + 41/840 finavalu_temp[10] ) //
//     where                                                                  //
//     k1 = f( x[i],y[i] ),                                                   //
//     finavalu_temp[1] = f( x[i]+2h/27, y[i] + 2h*k1/27),                                  //
//     finavalu_temp[2] = f( x[i]+h/9, y[i]+h/36*( k1 + 3 finavalu_temp[1]) ),                            //
//     finavalu_temp[3] = f( x[i]+h/6, y[i]+h/24*( k1 + 3 finavalu_temp[2]) ),                            //
//     finavalu_temp[4] = f( x[i]+5h/12, y[i]+h/48*(20 k1 - 75 finavalu_temp[2] + 75 finavalu_temp[3])),                //
//     finavalu_temp[5] = f( x[i]+h/2, y[i]+h/20*( k1 + 5 finavalu_temp[3] + 4 finavalu_temp[4] ) ),                    //
//     finavalu_temp[6] = f( x[i]+5h/6, y[i]+h/108*( -25 k1 + 125 finavalu_temp[3] - 260 finavalu_temp[4] + 250 finavalu_temp[5] ) ), //
//     finavalu_temp[7] = f( x[i]+h/6, y[i]+h*( 31/300 k1 + 61/225 finavalu_temp[4] - 2/9 finavalu_temp[5]              //
//                                                            + 13/900 finavalu_temp[6]) )  //
//     finavalu_temp[8] = f( x[i]+2h/3, y[i]+h*( 2 k1 - 53/6 finavalu_temp[3] + 704/45 finavalu_temp[4] - 107/9 finavalu_temp[5]      //
//                                                      + 67/90 finavalu_temp[6] + 3 finavalu_temp[7]) ), //
//     finavalu_temp[9] = f( x[i]+h/3, y[i]+h*( -91/108 k1 + 23/108 finavalu_temp[3] - 976/135 finavalu_temp[4]        //
//                             + 311/54 finavalu_temp[5] - 19/60 finavalu_temp[6] + 17/6 finavalu_temp[7] - 1/12 finavalu_temp[8]) ), //
//     finavalu_temp[10] = f( x[i]+h, y[i]+h*( 2383/4100 k1 - 341/164 finavalu_temp[3] + 4496/1025 finavalu_temp[4]     //
//          - 301/82 finavalu_temp[5] + 2133/4100 finavalu_temp[6] + 45/82 finavalu_temp[7] + 45/164 finavalu_temp[8] + 18/41 finavalu_temp[9]) )  //
//     finavalu_temp[11] = f( x[i], y[i]+h*( 3/205 k1 - 6/41 finavalu_temp[5] - 3/205 finavalu_temp[6] - 3/41 finavalu_temp[7]        //
//                                                   + 3/41 finavalu_temp[8] + 6/41 finavalu_temp[9]) )  //
//     finavalu_temp[12] = f( x[i]+h, y[i]+h*( -1777/4100 k1 - 341/164 finavalu_temp[3] + 4496/1025 finavalu_temp[4]    //
//                      - 289/82 finavalu_temp[5] + 2193/4100 finavalu_temp[6] + 51/82 finavalu_temp[7] + 33/164 finavalu_temp[8] +   //
//                                                        12/41 finavalu_temp[9] + finavalu_temp[11]) )  //
//     x[i+1] = x[i] + h.                                                     //
//                                                                            //
//     The error is estimated to be                                           //
//        err = -41/840 * h * ( k1 + finavalu_temp[10] - finavalu_temp[11] - finavalu_temp[12])                         //
//     The step size h is then scaled by the scale factor                     //
//         scale = 0.8 * | epsilon * y[i] / [err * (xmax - x[0])] | ^ 1/7     //
//     The scale factor is further constrained 0.125 < scale < 4.0.           //
//     The new step size is h := scale * h.                                   //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//  static fp Runge_Kutta(fp (*f)(fp,fp), fp *y,          //
//                                                       fp x0, fp h) //
//                                                                            //
//  Description:                                                              //
//     This routine uses Fehlberg's embedded 7th and 8th order methods to     //
//     approximate the solution of the differential equation y'=f(x,y) with   //
//     the initial condition y = y[0] at x = x0.  The value at x + h is       //
//     returned in y[1].  The function returns err / h ( the absolute error   //
//     per step size ).                                                       //
//                                                                            //
//  Arguments:                                                                //
//     fp *f  Pointer to the function which returns the slope at (x,y) of //
//                integral curve of the differential equation y' = f(x,y)     //
//                which passes through the point (x0,y[0]).                   //
//     fp y[] On input y[0] is the initial value of y at x, on output     //
//                y[1] is the solution at x + h.                              //
//     fp x   Initial value of x.                                         //
//     fp h   Step size                                                   //
//                                                                            //
//  Return Values:                                                            //
//     This routine returns the err / h.  The solution of y(x) at x + h is    //
//     returned in y[1].                                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//		PARTICULAR SOLVER FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

__device__ void embedded_fehlberg_7_8_2(	fp h,

																			fp timeinst,
																			fp* initvalu,
																			fp* finavalu,
																			fp* parameter,

																			fp* error,
																			fp* initvalu_temp,
																			fp* finavalu_temp,
																			fp* com) {

	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	const fp c_1_11 = __float2half(41.0 / 840.0);
	const fp c6 = __float2half(34.0 / 105.0);
	const fp c_7_8= __float2half( 9.0 / 35.0);
	const fp c_9_10 = __float2half( 9.0 / 280.0);

	const fp a2 = __float2half( 2.0 / 27.0);
	const fp a3 = __float2half( 1.0 / 9.0);
	const fp a4 = __float2half( 1.0 / 6.0);
	const fp a5 = __float2half( 5.0 / 12.0);
	const fp a6 = __float2half( 1.0 / 2.0);
	const fp a7 = __float2half( 5.0 / 6.0);
	const fp a8 = __float2half( 1.0 / 6.0);
	const fp a9 = __float2half( 2.0 / 3.0);
	const fp a10 = __float2half( 1.0 / 3.0);

	const fp b31 = __float2half( 1.0 / 36.0);
	const fp b32 = __float2half( 3.0 / 36.0);
	const fp b41 = __float2half( 1.0 / 24.0);
	const fp b43 = __float2half( 3.0 / 24.0);
	const fp b51 = __float2half( 20.0 / 48.0);
	const fp b53 = __float2half( -75.0 / 48.0);
	const fp b54 = __float2half( 75.0 / 48.0);
	const fp b61 = __float2half( 1.0 / 20.0);
	const fp b64 = __float2half( 5.0 / 20.0);
	const fp b65 = __float2half( 4.0 / 20.0);
	const fp b71 = __float2half( -25.0 / 108.0);
	const fp b74 = __float2half(  125.0 / 108.0);
	const fp b75 = __float2half( -260.0 / 108.0);
	const fp b76 = __float2half(  250.0 / 108.0);
	const fp b81 = __float2half( 31.0/300.0);
	const fp b85 = __float2half( 61.0/225.0);
	const fp b86 = __float2half( -2.0/9.0);
	const fp b87 = __float2half( 13.0/900.0);
	const fp b91 = __float2half( 2.0);
	const fp b94 = __float2half( -53.0/6.0);
	const fp b95 = __float2half( 704.0 / 45.0);
	const fp b96 = __float2half( -107.0 / 9.0);
	const fp b97 = __float2half( 67.0 / 90.0);
	const fp b98 = __float2half( 3.0);
	const fp b10_1 = __float2half( -91.0 / 108.0);
	const fp b10_4 = __float2half( 23.0 / 108.0);
	const fp b10_5 = __float2half( -976.0 / 135.0);
	const fp b10_6 = __float2half( 311.0 / 54.0);
	const fp b10_7 = __float2half( -19.0 / 60.0);
	const fp b10_8 = __float2half( 17.0 / 6.0);
	const fp b10_9 = __float2half( -1.0 / 12.0);
	const fp b11_1 = __float2half( 2383.0 / 4100.0);
	const fp b11_4 = __float2half( -341.0 / 164.0);
	const fp b11_5 = __float2half( 4496.0 / 1025.0);
	const fp b11_6 = __float2half( -301.0 / 82.0);
	const fp b11_7 = __float2half( 2133.0 / 4100.0);
	const fp b11_8 = __float2half( 45.0 / 82.0);
	const fp b11_9 = __float2half( 45.0 / 164.0);
	const fp b11_10 = __float2half( 18.0 / 41.0);
	const fp b12_1 = __float2half( 3.0 / 205.0);
	const fp b12_6 = __float2half( - 6.0 / 41.0);
	const fp b12_7 = __float2half( - 3.0 / 205.0);
	const fp b12_8 = __float2half( - 3.0 / 41.0);
	const fp b12_9 = __float2half( 3.0 / 41.0);
	const fp b12_10 = __float2half( 6.0 / 41.0);
	const fp b13_1 = __float2half( -1777.0 / 4100.0);
	const fp b13_4 = __float2half( -341.0 / 164.0);
	const fp b13_5 = __float2half( 4496.0 / 1025.0);
	const fp b13_6 = __float2half( -289.0 / 82.0);
	const fp b13_7 = __float2half( 2193.0 / 4100.0);
	const fp b13_8 = __float2half( 51.0 / 82.0);
	const fp b13_9 = __float2half( 33.0 / 164.0);
	const fp b13_10 = __float2half( 12.0 / 41.0);

	const fp err_factor  = __float2half( -41.0 / 840.0);

	fp h2_7 = a2 * h;

	fp timeinst_temp;

	int i,j;

	//======================================================================================================================================================
	//		EVALUATIONS
	//======================================================================================================================================================

	for(j=0; j<13; j++){

		//===================================================================================================
		//		0
		//===================================================================================================

		if(j==0){

			timeinst_temp = timeinst;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] ;
			}

		}

		//===================================================================================================
		//		1
		//===================================================================================================

		else if(j==1){

			timeinst_temp = timeinst+h2_7;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h2_7 * (finavalu_temp[0*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		2
		//===================================================================================================

		else if(j==2){

			timeinst_temp = timeinst+a3*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b31*finavalu_temp[0*EQUATIONS+i] + b32*finavalu_temp[1*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		3
		//===================================================================================================

		else if(j==3){

			timeinst_temp = timeinst+a4*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b41*finavalu_temp[0*EQUATIONS+i] + b43*finavalu_temp[2*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		4
		//===================================================================================================

		else if(j==4){

			timeinst_temp = timeinst+a5*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b51*finavalu_temp[0*EQUATIONS+i] + b53*finavalu_temp[2*EQUATIONS+i] + b54*finavalu_temp[3*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		5
		//===================================================================================================

		else if(j==5){

			timeinst_temp = timeinst+a6*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b61*finavalu_temp[0*EQUATIONS+i] + b64*finavalu_temp[3*EQUATIONS+i] + b65*finavalu_temp[4*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		6
		//===================================================================================================

		else if(j==6){

			timeinst_temp = timeinst+a7*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b71*finavalu_temp[0*EQUATIONS+i] + b74*finavalu_temp[3*EQUATIONS+i] + b75*finavalu_temp[4*EQUATIONS+i] + b76*finavalu_temp[5*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		7
		//===================================================================================================

		else if(j==7){

			timeinst_temp = timeinst+a8*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b81*finavalu_temp[0*EQUATIONS+i] + b85*finavalu_temp[4*EQUATIONS+i] + b86*finavalu_temp[5*EQUATIONS+i] + b87*finavalu_temp[6*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		8
		//===================================================================================================

		else if(j==8){

			timeinst_temp = timeinst+a9*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b91*finavalu_temp[0*EQUATIONS+i] + b94*finavalu_temp[3*EQUATIONS+i] + b95*finavalu_temp[4*EQUATIONS+i] + b96*finavalu_temp[5*EQUATIONS+i] + b97*finavalu_temp[6*EQUATIONS+i]+ b98*finavalu_temp[7*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		9
		//===================================================================================================

		else if(j==9){

			timeinst_temp = timeinst+a10*h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b10_1*finavalu_temp[0*EQUATIONS+i] + b10_4*finavalu_temp[3*EQUATIONS+i] + b10_5*finavalu_temp[4*EQUATIONS+i] + b10_6*finavalu_temp[5*EQUATIONS+i] + b10_7*finavalu_temp[6*EQUATIONS+i] + b10_8*finavalu_temp[7*EQUATIONS+i] + b10_9*finavalu_temp[8*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		10
		//===================================================================================================

		else if(j==10){

			timeinst_temp = timeinst+h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b11_1*finavalu_temp[0*EQUATIONS+i] + b11_4*finavalu_temp[3*EQUATIONS+i] + b11_5*finavalu_temp[4*EQUATIONS+i] + b11_6*finavalu_temp[5*EQUATIONS+i] + b11_7*finavalu_temp[6*EQUATIONS+i] + b11_8*finavalu_temp[7*EQUATIONS+i] + b11_9*finavalu_temp[8*EQUATIONS+i]+ b11_10 * finavalu_temp[9*EQUATIONS+i]);
			}

		}

		//===================================================================================================
		//		11
		//===================================================================================================

		else if(j==11){

			timeinst_temp = timeinst;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b12_1*finavalu_temp[0*EQUATIONS+i] + b12_6*finavalu_temp[5*EQUATIONS+i] + b12_7*finavalu_temp[6*EQUATIONS+i] + b12_8*finavalu_temp[7*EQUATIONS+i] + b12_9*finavalu_temp[8*EQUATIONS+i] + b12_10 * finavalu_temp[9*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		12
		//===================================================================================================

		else if(j==12){

			timeinst_temp = timeinst+h;
			for(i=0; i<EQUATIONS; i++){
				initvalu_temp[i] = initvalu[i] + h * ( b13_1*finavalu_temp[0*EQUATIONS+i] + b13_4*finavalu_temp[3*EQUATIONS+i] + b13_5*finavalu_temp[4*EQUATIONS+i] + b13_6*finavalu_temp[5*EQUATIONS+i] + b13_7*finavalu_temp[6*EQUATIONS+i] + b13_8*finavalu_temp[7*EQUATIONS+i] + b13_9*finavalu_temp[8*EQUATIONS+i] + b13_10*finavalu_temp[9*EQUATIONS+i] + finavalu_temp[11*EQUATIONS+i]) ;
			}

		}

		//===================================================================================================
		//		EVALUATION
		//===================================================================================================

		kernel_2(	(int)(__half2float(timeinst_temp)),
							initvalu_temp,
							parameter,
							&finavalu_temp[j*EQUATIONS],
							com);

	}

	//======================================================================================================================================================
	//		FINAL VALUE
	//======================================================================================================================================================

	for(i=0; i<EQUATIONS; i++){
		finavalu[i]= initvalu[i] +  h * (c_1_11 * (finavalu_temp[0*EQUATIONS+i] + finavalu_temp[10*EQUATIONS+i])  + c6 * finavalu_temp[5*EQUATIONS+i] + c_7_8 * (finavalu_temp[6*EQUATIONS+i] + finavalu_temp[7*EQUATIONS+i]) + c_9_10 * (finavalu_temp[8*EQUATIONS+i] + finavalu_temp[9*EQUATIONS+i]) );
		// printf("finavalu_temp[0][%d] = %f\n", i, finavalu_temp[0][i]);
		// printf("finavalu_temp[10][%d] = %f\n", i, finavalu_temp[10][i]);
		// printf("finavalu_temp[5][%d] = %f\n", i, finavalu_temp[5][i]);
		// printf("finavalu_temp[6][%d] = %f\n", i, finavalu_temp[6][i]);
		// printf("finavalu_temp[7][%d] = %f\n", i, finavalu_temp[7][i]);
		// printf("finavalu_temp[8][%d] = %f\n", i, finavalu_temp[8][i]);
		// printf("finavalu_temp[9][%d] = %f\n", i, finavalu_temp[9][i]);
		// printf("finavalu[%d] = %f\n", i, finavalu[i]);
	}

	//======================================================================================================================================================
	//		RETURN
	//======================================================================================================================================================

	for(i=0; i<EQUATIONS; i++){
		error[i] = fabs(err_factor * (finavalu_temp[0*EQUATIONS+i] + finavalu_temp[10*EQUATIONS+i] - finavalu_temp[11*EQUATIONS+i] - finavalu_temp[12*EQUATIONS+i]));
		// printf("Error[%d] = %f\n", i, error[i]);
	}

}
//======================================================================================================================================================
//======================================================================================================================================================
//		INCLUDE
//======================================================================================================================================================
//======================================================================================================================================================


//======================================================================================================================================================
//======================================================================================================================================================
//		SOLVER FUNCTION
//======================================================================================================================================================
//======================================================================================================================================================

__global__ void solver_2(	int workload,
											int xmax,

											fp* x,
											fp* y,
											fp* params,

											fp* com,
											fp* err,
											fp* scale,
											fp* yy,
											fp* initvalu_temp,
											fp* finavalu_temp){

	//========================================================================================================================
	//	VARIABLES
	//========================================================================================================================

	// CUDA indexes
	int bx;																					// get current horizontal block index (0-n)
	int tx;																					// get current horizontal thread index (0-n)
	int tid;																					// thread identifier

	// pointers
	long y_pointer_initial;
	long y_pointer_current;
	long x_pointer_current;
	int err_pointer;
	int scale_pointer;
	int yy_pointer;
	int params_pointer;
	int initvalu_temp_pointer;
	int finavalu_temp_pointer;
	int com_pointer;

	// solver parameters
	fp err_exponent ;
	fp h_init;
	fp h;
	fp tolerance;
	int xmin;

	// temporary solver variables
	int error;
	int outside;
	fp scale_min;
	fp scale_fina;

	// counters
	int i, j, k;

	//========================================================================================================================
	//		INITIAL SETUP
	//========================================================================================================================

	// CUDA indexes
	bx = blockIdx.x;																// get current horizontal block index (0-n)
	tx = threadIdx.x;																// get current horizontal thread index (0-n)
	tid = bx*NUMBER_THREADS+tx;

	// save pointers, these pointers are one per workload, independent of time step
	err_pointer = tid*EQUATIONS;
	scale_pointer = tid*EQUATIONS;
	yy_pointer = tid*EQUATIONS;
	params_pointer = tid*PARAMETERS;
	initvalu_temp_pointer = tid*EQUATIONS;
	finavalu_temp_pointer = tid*13*EQUATIONS;
	com_pointer = tid*3;

	// solver parameters
	err_exponent = __float2half(1.0 / 7.0);
	h_init = __float2half(1);
	h = h_init;
	xmin = 0;
	tolerance = __float2half(10 / (float)(xmax - xmin));

	//========================================================================================================================
	//		RANGE AND STEP CHECKING
	//========================================================================================================================

	// Verify that the step size is positive and that the upper endpoint of integration is greater than the initial enpoint.               //
	if (xmax < xmin || h <= 0.0){
		return;
	}

	// If the upper endpoint of the independent variable agrees with the initial value of the independent variable.  Set the value of the dependent variable and return success. //
	if (xmax == xmin){
		return;
	}

	// Insure that the step size h is not larger than the length of the integration interval.                                            //
	if (h > (xmax - xmin) ) {
		h = __float2half((float)xmax - (float)xmin);
	}

	//========================================================================================================================
	//		SOLVING IF THERE ARE MANY WORKLOADS
	//========================================================================================================================

	// limit to useful threads
	if(tid<workload){

		for(k=1; k<(xmax+1); k++) {											// start after initial value

			y_pointer_initial = tid*((xmax+1)*EQUATIONS)+(k-1)*EQUATIONS;
			y_pointer_current = tid*((xmax+1)*EQUATIONS)+k*EQUATIONS;
			x_pointer_current = tid*(xmax+1)+k;

			x[x_pointer_current] = __float2half(k);															// set this to k if you want time incremente with respect to previous to be k+h, set this to k-1 if you want the increment to be h
			h = h_init;

			//==========================================================================================
			//		REINITIALIZE VARIABLES
			//==========================================================================================

			scale_fina = __float2half(1.0);

			//==========================================================================================
			//		MAKE ATTEMPTS TO MINIMIZE ERROR
			//==========================================================================================

			// make attempts to minimize error
			for (j = 0; j < ATTEMPTS; j++) {

				//============================================================
				//		REINITIALIZE VARIABLES
				//============================================================

				error = 0;
				outside = 0;
				scale_min = __float2half(MAX_SCALE_FACTOR);

				//============================================================
				//		EVALUATE ALL EQUATIONS
				//============================================================

				embedded_fehlberg_7_8_2(	h,																												// single value

																x[x_pointer_current],																				// single value
																&y[y_pointer_initial],																					// 91 array
																&y[y_pointer_current],																				// 91 array
																&params[params_pointer],																		// 18 array

																&err[err_pointer],																						// 91 array
																&initvalu_temp[initvalu_temp_pointer],													// 91 array
																&finavalu_temp[finavalu_temp_pointer],													// 13*91 array
																&com[com_pointer]);																				// 3 array

				//============================================================
				//		IF THERE WAS NO ERROR FOR ANY OF EQUATIONS, SET SCALE AND LEAVE THE LOOP
				//============================================================

				for(i=0; i<EQUATIONS; i++){
					if(err[err_pointer+i] > 0){
						error = 1;
					}
				}
				if (error != 1) {
					scale_fina = __float2half(MAX_SCALE_FACTOR);
					break;
				}

				//============================================================
				//		FIGURE OUT SCALE AS THE MINIMUM OF COMPONENT SCALES
				//============================================================

				for(i=0; i<EQUATIONS; i++){
					if(y[y_pointer_initial+i] == 0.0){
						yy[yy_pointer+i] = tolerance;
					}
					else{
						yy[yy_pointer+i] = fabs(y[y_pointer_initial+i]);
					}
					scale[scale_pointer+i] = 0.8 * pow( tolerance * yy[yy_pointer+i] / err[err_pointer+i] , err_exponent );
					if(scale[scale_pointer+i]<scale_min){
						scale_min = scale[scale_pointer+i];
					}
				}
				//~ scale_fina = min( max(scale_min, __float2half(MIN_SCALE_FACTOR)), __float2half(MAX_SCALE_FACTOR));
				fp temp_max = scale_min;
				if (scale_min < MIN_SCALE_FACTOR)
					temp_max = __float2half(MIN_SCALE_FACTOR);
				
				scale_fina = temp_max ;
				if(temp_max > MAX_SCALE_FACTOR)
					scale_fina = __float2half(MAX_SCALE_FACTOR);
				

				//============================================================
				//		IF WITHIN TOLERANCE, FINISH ATTEMPTS...
				//============================================================

				for(i=0; i<EQUATIONS; i++){
					if ( err[err_pointer+i] > ( tolerance * yy[yy_pointer+i] ) ){
						outside = 1;
					}
				}
				if (outside == 0){
					break;
				}

				//============================================================
				//		...OTHERWISE, ADJUST STEP FOR NEXT ATTEMPT
				//============================================================

				// scale next step in a default way
				h = h * scale_fina;

				// limit step to 0.9, because when it gets close to 1, it no longer makes sense, as 1 is already the next time instance (added to original algorithm)
				if (h >= 0.9) {
					h = __float2half(0.9);
				}

				// if instance+step exceeds range limit, limit to that range
				if ( x[x_pointer_current] + h > (float)xmax ){
					h = (float)xmax - x[x_pointer_current];
				}

				// if getting closer to range limit, decrease step
				else if ( x[x_pointer_current] + h + 0.5 * h > (float)xmax ){
					h = 0.5 * h;
				}

			}

			//==========================================================================================
			//		SAVE TIME INSTANCE THAT SOLVER ENDED UP USING
			//==========================================================================================

			x[x_pointer_current] = x[x_pointer_current] + h;

			//==========================================================================================
			//		IF MAXIMUM NUMBER OF ATTEMPTS REACHED AND CANNOT GIVE SOLUTION, EXIT PROGRAM WITH ERROR
			//==========================================================================================

			if ( j >= ATTEMPTS ) {
				return;
			}

		}

	}

	//========================================================================================================================
	//		FINAL RETURN
	//========================================================================================================================

	return;

//======================================================================================================================================================
//======================================================================================================================================================
//		END OF SOLVER FUNCTION
//======================================================================================================================================================
//======================================================================================================================================================

}
