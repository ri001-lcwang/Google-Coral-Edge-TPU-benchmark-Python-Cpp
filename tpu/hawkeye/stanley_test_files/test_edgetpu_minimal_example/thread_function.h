//https://tigercosmos.xyz/post/2020/07/simple-pthread-usage/

#define measure_time
//#define debug_thread_function

//if EDGE_TPU == 1
	#include "edgetpu.h"
#include "model_utils.h"


#include <stdio.h>
//#include <cstdio>

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

//#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/optional_debug_tools.h"


#include "sub_misc_c0.h"
#include "sub_librosa_c0.h"
#include "sub_ulib_realtime_pso_c0.h"

// 2020.07.22 : (Sky)
#define	DEFAULT_DEVNAME	"MS200199500099"
#define	DEFAULT_MACADDRESS	"ac00037F12AFAF"
#define	DEFAULT_SERVERIP	"192.168.100.196"
extern char dev_sn[32];
extern char mac_address[32];
extern char server_ip[32];
//

#include "parameter_set.h"
/* Use the newer ALSA API */
#define ALSA_PCM_NEW_HW_PARAMS_API
#include <alsa/asoundlib.h>  
#include <iostream>
#include <inttypes.h>
#include <pthread.h>
#include <queue>
using namespace std; 


void detection_input(    double *inputs_i
                        ,int inputs_row_i //= 63488;
                        //,model
                        //,char class_list[100]
                        ,double sample_rate_i
                        //,int uni_sample
                        ,double **mel_pad_truncate_o
                        );
 
//#ifdef SET_PTHREAD 
  void* thread_detection(void*);
//#else
  void thread_detection(void);
//#endif
void sub_thread_detection(void);


