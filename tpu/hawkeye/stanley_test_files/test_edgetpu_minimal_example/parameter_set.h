//typedef double _Complex complex_t;

// Avoids C++ name mangling with extern "C"
/*#include <math.h>
#include <complex.h>
#include <fftw3.h>*/


/*#define EDGE_TPU				0
#define SET_PTHREAD				1
#define TFLITE_INT8				0
#define TFLITE_UINT8			0
#define TFLITE_FP16				0
#define TFLITE_FP32FALLBACK		0
#define TFLITE_MODEL_OPT		1
#define DEF_TFLITE_NUMTHREADS 	3*/

//#define SET_DEVICE_HAWKEYE
//#define SET_DEBUG_THREAD_FUNCTION




//#include "sub_misc_c0.h"


//#define for_python_or_c
//#define DEBUG_fft
//#define DEBUG_mp
//#define DEBUG_cost_MUSIC
//#define DEBUG_pso
//#define DEBUG_gd
//#define DEBUG_TIKR_extration_opt
//#define DEBUG_beamforming
//#define measure_time




#define SET_ONE_SOURCE




// Parameter: global
#define DEF_C_SPEED 343.0
#define DEF_FS 32000.0 //48000.0
#define DEF_NUM_OF_MIC (int)8
#define DEF_MIC_BD_RADIUS 0.04

//#### signal in ####
//#define DEF_CHUNK (int)32 //1ms
//#define DEF_CHUNK (int)128 //4ms
#define DEF_CHUNK (int)1024 //32ms
#define DEF_RATE DEF_FS
#define DEF_CHANNELS (int)DEF_NUM_OF_MIC
//FORMAT = pyaudio.paInt16    

//#### array param ####
#define DEF_NWIN (int)DEF_CHUNK
#define DEF_HOPSIZE (int)(DEF_NWIN/2)
#define DEF_NFFT (int)DEF_NWIN
#define DEF_NUM_OF_FREQS (int)(DEF_NWIN/2)

//#### record param ####
#define DEF_RECEIVE_WHOLE_BUF_MULT (int)2
//#define DEF_RECORD_FRAMES (int)32
#define DEF_RECORD_FRAMES (int)DEF_CHUNK

//#define DEF_SOR_SEC 3601  //# real processing second +1
#define DEF_RECORD_SECOND_TOTAL_FRAME (int)DEF_SOR_SEC*int(DEF_RATE / DEF_CHUNK + 1)
//Process_sec = DEF_SOR_SEC - 1
//process_second_total_frame = Process_sec * int(DEF_RATE / DEF_CHUNK + 1)
//wav_out_frame = np.zeros((DEF_NUM_OF_MIC, DEF_NWIN))

//#### params , flag for multi processing ####
//#define DEF_TRUNCATE_FRAME (int)(round(1.0*DEF_FS/DEF_NWIN))  //# 1 sec frames
/*#define DEF_TRUNCATE_FRAME (int)(round(1000000.0/DEF_FS))  //# 1 sec frames
#define DEF_BLOCK_SEC (int)2
#define DEF_PROCESS_FRAME (int)(DEF_BLOCK_SEC*DEF_TRUNCATE_FRAME)  //# 4 sec
#define DEF_INPUT_TRANSITION_FRAME (int)(1*DEF_TRUNCATE_FRAME)  //# 1 sec*/
#define DEF_BLOCK_SEC (int)2
//#define DEF_INPUT_TRANSITION_FRAME (int)(round(1000000.0*DEF_RECORD_FRAMES/DEF_FS))
//#define DEF_INPUT_TRANSITION_FRAME (int)(round(1000000.0*32.0*32.0/(1.0*DEF_FS*DEF_CHUNK)))
#define DEF_INPUT_TRANSITION_FRAME (int)(round((1.0*DEF_FS)/(1.0*DEF_CHUNK)))
#define DEF_ONE_SEC (int)(DEF_INPUT_TRANSITION_FRAME*DEF_CHUNK)
#define DEF_PROCESS_FRAME (int)(DEF_BLOCK_SEC*DEF_ONE_SEC)  //# 2 sec
//#define DEF_TRUNCATE_FRAME (int)(1*DEF_INPUT_TRANSITION_FRAME)  //# 1 sec frames

//#define DEF_SOR_LEN (int)(DEF_PROCESS_FRAME*DEF_CHUNK)
//#define DEF_SOR_LEN (int)(DEF_PROCESS_FRAME)
//const int SorLen_const = DEF_PROCESS_FRAME*DEF_CHUNK; //DEF_SOR_LEN;
//#define DEF_NUM_OF_FRAME_CONST (int)(2*floor(DEF_SOR_LEN/DEF_NWIN)-1)
#define DEF_NUM_OF_FRAME_CONST (int)(2*floor(DEF_PROCESS_FRAME/DEF_NWIN)-1)
//const int DEF_NUM_OF_FRAME_CONST = 2*floor(DEF_SOR_LEN/DEF_NWIN)-1; //NumOfFrame;

//#### TIKR extraction params ####
//k_TIKR = 2 * math.pi * freqs / c
#define DEF_BATA 0.01

// Parameter: PSO_Localization
#define DEF_PSO_ITERATIONS (int)8   //# 迭代次數
#define DEF_PSO_SWARMS (int)22 //=(12-1)*(3-1)
#define DEF_PSO_DIMENSION (int)2
//boundary_max=[360,90]
//extern int pso_boundary_max[];
#define DEF_PSO_BOUNDARY_MIN (int)0
#define DEF_PSO_GRID_SECTION (int)30
#define DEF_PSO_NEIGHBOR_DIS (int)40

// for thread_detection
#define DEF_TD_PAD_TRUNCATE_LENGTH (int)126
#define DEF_TD_N_MELS (int)64
#define DEF_MAX_CLASS_LIST (int)22
/*const char *class_list[DEF_MAX_CLASS_LIST] =   { "cry"      ,"screaming"      ,"shatter" ,"bark"    ,"boiling"  ,"toilet_flush"
                                            ,"doorbell" ,"telephone_ring" ,"water"   ,"blender" ,"cupboard" 
                                            ,"cupboard" ,"cough"          ,"snoring" ,"walk"    ,"speech"   
                                            ,"alarm"    ,"cook"           ,"groan"   ,"sneeze"  ,"fall"     ,"no_evevt"};*/
