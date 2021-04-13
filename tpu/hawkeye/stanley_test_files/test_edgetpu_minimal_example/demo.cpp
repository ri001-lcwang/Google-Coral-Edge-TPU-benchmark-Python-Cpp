//#include <stdio.h>
//#include <cstdio>
#include "demo.h"

#define ALSA_PCM_NEW_HW_PARAMS_API
#include <alsa/asoundlib.h>

extern int EDGE_TPU		        ;
extern int SET_PTHREAD          ;		 
extern int TFLITE_INT8          ;		 
extern int TFLITE_UINT8         ;    
extern int TFLITE_FP16          ;		 
extern int TFLITE_FP32FALLBACK	;
extern int TFLITE_MODEL_OPT     ;	 
extern int DEF_TFLITE_NUMTHREADS;

extern std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;

void run_single (void) {
    SET_PTHREAD = 0;
    //printf ("\nSET_PTHREAD = %d",SET_PTHREAD);
    thread_detection();
    printf("    N");
}

void run_single_pthread (void) {
    SET_PTHREAD = 1;
    //printf ("SET_PTHREAD = %d",SET_PTHREAD);
    int ret = -1;    
    pthread_t pthread_det;
    ret = pthread_create(&pthread_det, NULL, &thread_detection, NULL);
    if (ret != 0) {
        printf ("\nCreate pthread_det error!");
        exit (1);
    }
    pthread_join(pthread_det, NULL);
    printf("    Y");
}

int main(int argc, char** argv){
    
    int i, j, k, m;
    int count_select_dev = 2;
    
    if (!edgetpu_context) {
        count_select_dev = 1;
        printf("Warning: Detect edgetpu devices is none. Run without TPU.\n");
    } else {
        printf("Note: Detect edgetpu devices is existence. Run TPU & CPU.\n");
    }
    
    printf ("\n  Benchmark result:");
    printf ("\n  Device | Opt model | Data type | Pred_threads | Accuracy(%) | Load model(s) | 1st pred(s)  | Avg 17 pred(s) | Avg 17 set_io + pred(s) | Pthread");
    
    for (i = 0; i < count_select_dev; i++) {
        if (i == count_select_dev - 1) {
            EDGE_TPU = 0;
        } else {
            EDGE_TPU = 1;
        }

        for (j = 0; j < 5; j++) {
            switch (j) { 
                case 0: {
                    TFLITE_INT8         = 1;
                    TFLITE_UINT8        = 0;
                    TFLITE_FP16	        = 0;
                    TFLITE_FP32FALLBACK = 0;
                    }
                    break; 
                case 1: {
                    TFLITE_INT8         = 0;
                    TFLITE_UINT8        = 1;
                    TFLITE_FP16	        = 0;
                    TFLITE_FP32FALLBACK = 0;
                    }
                    break; 
                case 2: { 
                    TFLITE_INT8         = 0;
                    TFLITE_UINT8        = 0;
                    TFLITE_FP16	        = 1;
                    TFLITE_FP32FALLBACK = 0;
                    }
                    break; 
                case 3: {
                    TFLITE_INT8         = 0;
                    TFLITE_UINT8        = 0;
                    TFLITE_FP16	        = 0;
                    TFLITE_FP32FALLBACK = 1;
                    }
                    break;
                default: {  //j==4
                    TFLITE_INT8         = 0;
                    TFLITE_UINT8        = 0;
                    TFLITE_FP16	        = 0;
                    TFLITE_FP32FALLBACK = 0;
                    }
                    break;
            }

            for (k = 0; k < 2; k++) {
                TFLITE_MODEL_OPT = k;                
                
                for (m = 0; m < 4; m++) {
                    DEF_TFLITE_NUMTHREADS = m + 1;
                    run_single ();
                }
                
                for (m = 0; m < 4; m++) {
                    DEF_TFLITE_NUMTHREADS = m + 1;
                    run_single_pthread ();
                }
            }

            if (EDGE_TPU == 1 && j == 1) {
                break;
            }
        }
    } 
    

    printf("\n\n\n");
    
    return 0;
}


