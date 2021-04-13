//typedef double _Complex complex_t;

// Avoids C++ name mangling with extern "C"
/*#include <math.h>
#include <complex.h>
#include <fftw3.h>*/
#include "sub_misc_c0.h"


//#define for_python_or_c
//#define DEBUG_fft
//#define DEBUG_mp
//#define DEBUG_cost_MUSIC
//#define DEBUG_pso
//#define DEBUG_gd
//#define DEBUG_TIKR_extration_opt
//#define DEBUG_beamforming
//#define measure_time


// for C++
#include "parameter_set.h"
int get_direction_beamforming(   double **MicPos_i //[3, DEF_NUM_OF_MIC]
                                ,float **buf_i //[DEF_NUM_OF_MIC, DEF_SOR_LEN] //DEF_NUM_OF_MIC*(DEF_PROCESS_FRAME*DEF_CHUNK) or DEF_NUM_OF_MIC*DEF_SOR_LEN
                                ,double *k_TIKR //[DEF_HOPSIZE]
                                ,float bata_i
                                ,int flag_create_fftw3_plan_1d_fft_i
                                ,int flag_destroy_fftw3_plan_1d_fft_i
                                ,int flag_create_fftw3_plan_1d_ifft
                                ,int flag_destroy_fftw3_plan_1d_ifft
                                //,double flat_solution_o[DEF_NUM_OF_MIC*2] //max: DEF_NUM_OF_MIC*2, original: SorNum_guess*2
                                ,double *ang //[4]
                                ,double **p_o //[2, (DEF_PROCESS_FRAME*DEF_CHUNK)] //SorNum_guess*(DEF_PROCESS_FRAME*DEF_CHUNK)
                                );
/*void cost_MUSIC( double position[DEF_NUM_OF_MIC][3] //DEF_NUM_OF_MIC*3
                    ,double MicPos[3][DEF_NUM_OF_MIC] //3*DEF_NUM_OF_MIC
                    ,double _Complex PN[DEF_NUM_OF_MIC][DEF_NUM_OF_MIC][DEF_NUM_OF_FREQS] //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    ,double k[DEF_NFFT]
                    ,double cost[DEF_NUM_OF_MIC][DEF_NUM_OF_MIC] //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC
                    ); */ //old code
//void create_fftw3_plan_1d_fft (double _Complex *data_in ,int N ,short setting, double _Complex *data_out);
void FFT( fftw_plan fftw3_plan_1d_fft
            ,double _Complex *fftw3_data_in
            ,double _Complex *fftw3_data_out
            //,float **p //DEF_NUM_OF_MIC*(DEF_PROCESS_FRAME*DEF_CHUNK) or DEF_NUM_OF_MIC*DEF_SOR_LEN
            ,double **p
            ,double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
            );
int MUSIC_Parameter( double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                    //,int fs
                    ,double *k //DEF_NFFT/2
                    ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    //,double **eigenvalue_save //DEF_NUM_OF_FREQS*DEF_NUM_OF_MIC
                    //,int SorNum_guess
                    );
void MUSIC_Parameter_dbg( double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                    //,int fs
                    ,double *k //DEF_NFFT/2
                    ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    //,double **eigenvalue_save //DEF_NUM_OF_FREQS*DEF_NUM_OF_MIC
                    ,int *SorNum_guess
                    );
int get_direction(  double **MicPos //3*DEF_NUM_OF_MIC
					,float **buf //DEF_NUM_OF_MIC*(DEF_PROCESS_FRAME*DEF_CHUNK) or DEF_NUM_OF_MIC*DEF_SOR_LEN
                    ,int flag_create_fftw3_plan_1d_fft
                    ,int flag_destroy_fftw3_plan_1d_fft
                    ,double **solution //max: DEF_NUM_OF_MIC*2, original: SorNum_guess*2
                    //,int SorNum_guess
                    ,double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                    //,double *k //DEF_NFFT/2
                    );
double cost_MUSIC( double *position //DEF_NUM_OF_MIC*3
                    ,double **MicPos //3*DEF_NUM_OF_MIC
                    ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    ,double *k //DEF_NFFT
                    //,double cost //**cost //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC
                    ); //new code
void PSO_Localization(   double **MicPos //3*DEF_NUM_OF_MIC
                        ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                        ,double *k //DEF_NFFT
                        ,int SorNum
                        ,double **swarm //swarms*(3*dimension+2)
                        //,double swarm[DEF_PSO_SWARMS][3*pso_dimension+2]
                        );
void TIKR_extration_opt (double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                        ,double **solution //max: DEF_NUM_OF_MIC*2, original: SorNum_guess*2
                        ,int SorNum_guess
                        ,double **MicPos //3*DEF_NUM_OF_MIC
                        ,double *k_TIKR //DEF_HOPSIZE
                        ,float bata_i
                        ,int flag_create_fftw3_plan_1d_ifft
                        ,int flag_destroy_fftw3_plan_1d_ifft
                        ,double **p_o //SorNum_guess*(DEF_PROCESS_FRAME*DEF_CHUNK)
                        );
void beamforming        (double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                        ,double **solution //max: DEF_NUM_OF_MIC*2, original: SorNum_guess*2
                        ,int SorNum_guess
                        ,double **MicPos //3*DEF_NUM_OF_MIC
                        ,double *k_TIKR //DEF_HOPSIZE
                        ,float bata_i
                        ,int flag_create_fftw3_plan_1d_ifft
                        ,int flag_destroy_fftw3_plan_1d_ifft
                        ,double **p_o //SorNum_guess*(DEF_PROCESS_FRAME*DEF_CHUNK)
                        );


