#include "thread_function.h"


/*char wav_header[44] = {
    	0x52, 0x49, 0x46, 0x46, 0x24, 0xE0, 0x03, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20,
    	0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x7D, 0x00, 0x00, 0x00, 0xF4, 0x01, 0x00,
    	0x04, 0x00, 0x20, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0xE0, 0x03, 0x00, 
    };*/

#include <sys/time.h>

/*const char *class_list_fname[DEF_MAX_CLASS_LIST] =
                                               { "Cry"                  ,"Screaming"      ,"Shatter"    ,"Bark"    ,"Boiling"           ,"Toilet_flush"
                                                ,"Doorbell"             ,"Telephone_ring" ,"Water_tap"  ,"Blender" ,"Cupboard" 
                                                ,"Cupboard"             ,"Cough"          ,"Snoring"    ,"Walk"    ,"Speech"   
                                                ,"Alarm"                ,"Cook"           ,"Groan"      ,"Sneeze"  ,"Fall_down"         ,"no_evevt"};*/


int EDGE_TPU		     = 1;
int SET_PTHREAD		     = 0;
int TFLITE_INT8		     = 1;
int TFLITE_UINT8         = 0;
int TFLITE_FP16		     = 0;
int TFLITE_FP32FALLBACK	 = 0;
int TFLITE_MODEL_OPT	 = 0;
int DEF_TFLITE_NUMTHREADS= 1;


std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
            edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();


//#ifdef SET_PTHREAD 
  void* thread_detection(void*) { sub_thread_detection(); pthread_exit(NULL); }
//#else
  void thread_detection(void) { sub_thread_detection(); }
//#endif
void sub_thread_detection(void)
{
    
    //struct paras *dtargs = (struct paras*) dtarguments;

    const char *class_list[DEF_MAX_CLASS_LIST] =   { "Cry"                  ,"Screaming"      ,"Shatter"    ,"Bark"    ,"Boiling"           ,"Toilet flush"
                                                ,"Doorbell"             ,"Telephone ring" ,"Water tap"  ,"Blender" ,"Cupboard / drawer" 
                                                ,"Cupboard / drawer"    ,"Cough"          ,"Snoring"    ,"Walk"    ,"Speech"   
                                                ,"Alarm"                ,"Cook"           ,"Groan"      ,"Sneeze"  ,"Fall down"         ,"no_evevt"};
    
    clock_t time0, time1, time2, time3;
    float *result = create_1d_float_array(5);

    int i, j, k, z;
    
    // detection
    // load model
    //https://coral.ai/docs/edgetpu/tflite-cpp/#run-an-inference
    #if defined(measure_time)
        time0 = clock();
    #endif 
    
    std::unique_ptr<tflite::FlatBufferModel> model;
    
    #if defined(SET_DEVICE_HAWKEYE) //Device: hawkeye
        //#if EDGE_TPU == 1        
        if (EDGE_TPU == 1) {
            //#if TFLITE_INT8 == 1
            if (TFLITE_INT8 == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    const std::string model_path = "/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_opt_int8_edgetpu.tflite";
                    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
                //#else // non opt model   
                } else {
                    const std::string model_path = "/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_int8_edgetpu.tflite";
                    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
                //#endif
                }
            //#else //TFLITE_UINT8
            } else {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    const std::string model_path = "/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_opt_uint8_edgetpu.tflite";
                    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
                //#else // non opt model   
                } else {
                    const std::string model_path = "/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_uint8_edgetpu.tflite";
                    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
                //#endif
                }
            //#endif
            }
        //#else //CPU
        } else {
            //#if TFLITE_INT8 == 1
            if (TFLITE_INT8 == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_opt_int8.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_int8.tflite");
                //#endif
                }
            //#elif TFLITE_UINT8 == 1
            } else if (TFLITE_UINT8 == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_opt_uint8.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_uint8.tflite");
                //#endif
                }
            //#elif TFLITE_FP16 == 1
            } else if (TFLITE_FP16 == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_opt_fp16.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_fp16.tflite");
                //#endif
                }
            //#elif TFLITE_FP32FALLBACK == 1
            } else if (TFLITE_FP32FALLBACK == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_opt_fp32fallback.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_fp32fallback.tflite");
                //#endif
                }
            //#else //float
            } else {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420_opt.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("/usr/bin/hawkeye/freesound_umedia_20_classes_skip_attention_model_0420.tflite");
                //#endif
                }
            //#endif
            }
        //#endif
        }
    #else //Device: RBP 
        //#if EDGE_TPU == 1        
        if (EDGE_TPU == 1) {
            //#if TFLITE_INT8 == 1
            if (TFLITE_INT8 == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    const std::string model_path = "../../../freesound_umedia_20_classes_skip_attention_model_0420_opt_int8_edgetpu.tflite";
                    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
                //#else // non opt model   
                } else {
                    const std::string model_path = "../../../freesound_umedia_20_classes_skip_attention_model_0420_int8_edgetpu.tflite";
                    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
                //#endif
                }
            //#else //TFLITE_UINT8
            } else {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    const std::string model_path = "../../../freesound_umedia_20_classes_skip_attention_model_0420_opt_uint8_edgetpu.tflite";
                    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
                //#else // non opt model   
                } else {
                    const std::string model_path = "../../../freesound_umedia_20_classes_skip_attention_model_0420_uint8_edgetpu.tflite";
                    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
                //#endif
                }
            //#endif
            }
        //#else //CPU
        } else {
            //#if TFLITE_INT8 == 1
            if (TFLITE_INT8 == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_opt_int8.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_int8.tflite");
                //#endif
                }
            //#elif TFLITE_UINT8 == 1
            } else if (TFLITE_UINT8 == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_opt_uint8.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_uint8.tflite");
                //#endif
                }
            //#elif TFLITE_FP16 == 1
            } else if (TFLITE_FP16 == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_opt_fp16.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_fp16.tflite");
                //#endif
                }
            //#elif TFLITE_FP32FALLBACK == 1
            } else if (TFLITE_FP32FALLBACK == 1) {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_opt_fp32fallback.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_fp32fallback.tflite");
                //#endif
                }
            //#else //float
            } else {
                //#if TFLITE_MODEL_OPT == 1
                if (TFLITE_MODEL_OPT == 1) {
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420_opt.tflite");
                //#else // non opt model   
                } else {                
                    model = tflite::FlatBufferModel::BuildFromFile("../../../freesound_umedia_20_classes_skip_attention_model_0420.tflite");
                //#endif
                }
            //#endif
            }
        //#endif
        }       
    #endif
    
    
    #if defined(measure_time)
        time1 = clock(); 
        result[1] = (float)(time1-time0)/(CLOCKS_PER_SEC); //Load model(s)     
        //printf("\nload model time=%f\n", (float)(time1-time0)/(CLOCKS_PER_SEC));
    #endif 
    if (!model) {
        printf("Failed to mmap model\n");
        exit(0);
    }
    
    // Build the interpreter
    //std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
    std::unique_ptr<tflite::Interpreter> interpreter;
    //#if EDGE_TPU == 1        
    if (EDGE_TPU == 1) {
        /*std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
        //std::unique_ptr<tflite::Interpreter> interpreter = BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
        interpreter = BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
        
        std::unique_ptr BuildEdgeTpuInterpreter( const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context) {
            tflite::ops::builtin::BuiltinOpResolver resolver;
            resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
            std::unique_ptr interpreter;
            if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
                std::cerr << "Failed to build interpreter." << std::endl;
            }
            // Bind given context with interpreter.
            interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
            interpreter->SetNumThreads(DEF_TFLITE_NUMTHREADS);
            if (interpreter->AllocateTensors() != kTfLiteOk) {
                std::cerr << "Failed to allocate tensors." << std::endl;
            }
            return interpreter;
        }*/
                
        //std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
        //    edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
        //edgetpu_context =
        //    edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
        interpreter = std::move(coral::BuildEdgeTpuInterpreter(*model, edgetpu_context.get()));
        interpreter->SetNumThreads(DEF_TFLITE_NUMTHREADS);
        
    //#else //CPU
    } else {
        /*tflite::ops::builtin::BuiltinOpResolver resolver;
        //std::unique_ptr<tflite::Interpreter> interpreter;
        tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
        interpreter->SetNumThreads(DEF_TFLITE_NUMTHREADS);
        interpreter->AllocateTensors();*/
        
        interpreter = std::move(coral::BuildInterpreter(*model));
        interpreter->SetNumThreads(DEF_TFLITE_NUMTHREADS);
    //#endif
    }
    
    // Resize input tensors, if desired.
  //  int input = interpreter->inputs()[0];
    //float* input_data_ptr = interpreter->typed_input_tensor<float>(input);
  //  float* input_data_ptr = interpreter->typed_tensor<float>(input);
    

    //int num_of_sor = 1; //2;
    //int detection_result;
    int input_dim0 = 126;
    int input_dim1 = 64;
    //int pred_row = 1; //4;
    const int pred_col = DEF_MAX_CLASS_LIST - 1;
    
    double **mel_pad_truncate = create_2d_double_array(DEF_TD_PAD_TRUNCATE_LENGTH ,DEF_TD_N_MELS);
    //double **buf_2d_double_thread_det = create_2d_double_array(2, DEF_PROCESS_FRAME*DEF_CHUNK);
    double *buf_1d_double_thread_det = create_1d_double_array(DEF_PROCESS_FRAME); //DEF_PROCESS_FRAME*DEF_CHUNK);
    float *output_data = create_1d_float_array(DEF_MAX_CLASS_LIST - 1); //output_size_0);
    float **output_data_ext = create_2d_float_array(3, DEF_MAX_CLASS_LIST - 1);
    int input;
    //float* input_data_ptr = interpreter->typed_input_tensor<float>(input);
    
    //#if TFLITE_INT8 == 1
        double input_data_tmp0;
        int8_t* i8_input_data_ptr;		
        int8_t* i8_output_data_ptr_0;
        int8_t* i8_output_data_ptr_1; // = interpreter->typed_tensor<float>(output_1); 
        int8_t* i8_output_data_ptr_2; // = interpreter->typed_tensor<float>(output_2); 
        int8_t* i8_output_data_ptr_3; // = interpreter->typed_tensor<float>(output_3);
    //#elif TFLITE_UINT8 == 1
        //double input_data_tmp0;
        uint8_t* ui8_input_data_ptr;		
        uint8_t* ui8_output_data_ptr_0;
        uint8_t* ui8_output_data_ptr_1; // = interpreter->typed_tensor<float>(output_1); 
        uint8_t* ui8_output_data_ptr_2; // = interpreter->typed_tensor<float>(output_2); 
        uint8_t* ui8_output_data_ptr_3; // = interpreter->typed_tensor<float>(output_3);
    //#else //float
        float* input_data_ptr;		
        float* output_data_ptr_0;
        //TfLiteIntArray* output_dims_0; // = interpreter->tensor(output_0)->dims;
        float* output_data_ptr_1; // = interpreter->typed_tensor<float>(output_1); 
        float* output_data_ptr_2; // = interpreter->typed_tensor<float>(output_2); 
        float* output_data_ptr_3; // = interpreter->typed_tensor<float>(output_3);
    //#endif
    
    
    int output_0;
    int output_1;		               
    int output_2; // = interpreter->outputs()[2];	 
    int output_3; // = interpreter->outputs()[3];
    float max_pred;
    int pred_result_index;
    

    // read wav file
    //int fd = -1;
    char *filename;
    /*char *filename = "../MS200199500008.0.1597747308.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/alarm1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/bark1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/boiling1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/cough1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/cry2.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/doorbell2.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/fall1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/groan1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/screaming1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/shatter1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/sneeze1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/snoring2.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/speech1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/telephone2.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/toilet1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/walk1.wav";
    char *filename = "../../../Data_for_demo/32k_mono_32bit/water1.wav";*/
    //int *buf_tmp = create_1d_int_array(DEF_RECORD_FRAMES);
    int *buf_tmp = create_1d_int_array(DEF_PROCESS_FRAME);
    int *buf_header = create_1d_int_array(11);
    int rc_bytes_total = 0, rc_bytes;
    short int bits_per_sample;
    FILE* file_p;
    int wavfile_data_size = DEF_PROCESS_FRAME;
    int file_class_index = 0;
    
    
    for( z = 0; z < 17; z++) { 
    //for( z = 7; z < 8; z++) {  
    //for( z = 0; z < 2; z++) {  
        // read wav file
        switch(z) { 
            case 0:
                filename = "../../../../Data_for_demo/32k_mono_32bit/alarm1.wav";
                file_class_index = 16;
                break; 
            case 1: 
                filename = "../../../../Data_for_demo/32k_mono_32bit/bark1.wav"; 
                file_class_index = 3;
                break; 
            case 2: 
                filename = "../../../../Data_for_demo/32k_mono_32bit/boiling1.wav";
                file_class_index = 4;
                break; 
            case 3:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/cough1.wav";
                file_class_index = 12;
                break;
            case 4:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/cry2.wav";
                file_class_index = 0;
                break;
            case 5: 
                filename = "../../../../Data_for_demo/32k_mono_32bit/doorbell2.wav";
                file_class_index = 6; 
                break;
            case 6:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/fall1.wav";
                file_class_index = 20;
                break;
            case 7:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/groan1.wav";
                file_class_index = 18;
                break;
            case 8:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/screaming1.wav";
                file_class_index = 1;
                break;
            case 9: 
                filename = "../../../../Data_for_demo/32k_mono_32bit/shatter1.wav"; 
                file_class_index = 2;
                break;
            case 10: 
                filename = "../../../../Data_for_demo/32k_mono_32bit/sneeze1.wav"; 
                file_class_index = 19;
                break;
            case 11:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/snoring2.wav";
                file_class_index = 13;
                break;
            case 12:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/speech1.wav";
                file_class_index = 15;
                break;
            case 13:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/telephone2.wav";
                file_class_index = 7;
                break;
            case 14:  
                filename = "../../../../Data_for_demo/32k_mono_32bit/toilet1.wav";
                file_class_index = 5;
                break;
            case 15: 
                filename = "../../../../Data_for_demo/32k_mono_32bit/walk1.wav"; 
                file_class_index = 14;
                break;
            default:  //z==16
                filename = "../../../../Data_for_demo/32k_mono_32bit/water1.wav";
                file_class_index = 8;
                break;
        }
        
        //printf("\nc z= %d", z);
        //printf("\nc filename= %s", filename);
        
        file_p = fopen(filename,"rb");
        if (file_p) {
            // read header+data
            //rc_bytes = read(fd, buf_tmp, 11 + DEF_PROCESS_FRAME);
            rc_bytes = fread(buf_header, 1, 11*sizeof(int32_t), file_p);
            /*printf("\nc header=\n");
            for( j = 0; j < 11; j++) {
                printf("%d", buf_header[j]); printf(", ");
                if (j%4==3) {
                    printf("\n");
                }
            }*/
            
            rc_bytes_total += rc_bytes;
            bits_per_sample = (buf_header[8] & 0xffff0000) >> 16;
            wavfile_data_size = buf_header[10]/4;
            //printf("\nc bits_per_sample= %d", bits_per_sample);
            //printf("\nc wavfile_data_size= %d", wavfile_data_size);
            
            /*printf("\nc raw wav data=\n");
            for( i = 0; i < DEF_BLOCK_SEC*DEF_INPUT_TRANSITION_FRAME; i++) {
                rc_bytes = fread(buf_tmp, 1, DEF_RECORD_FRAMES*sizeof(int32_t), file_p);
                rc_bytes_total += rc_bytes;
                for( j = 0; j < DEF_RECORD_FRAMES; j++) {
                    if (bits_per_sample == 32) {
                        buf_1d_double_thread_det[i*DEF_RECORD_FRAMES+j]= (1.0*(buf_tmp[j] >> 16))/pow(2.0, 15);
                    } else if (bits_per_sample == 16) {
                        buf_1d_double_thread_det[i*DEF_RECORD_FRAMES+j]= (1.0*(buf_tmp[j] & 0x0000ffff))/pow(2.0, 15);
                    } else {
                        printf("\nc not int32 or int16, do not support this format\n");
                        break;
                    }
                    printf("%d", buf_tmp[j]); printf(", ");
                    if (j%4==3) {
                        printf("\n");
                    }
                }
            }*/
            if (wavfile_data_size > DEF_PROCESS_FRAME) {
                wavfile_data_size = DEF_PROCESS_FRAME;
            }
            
            rc_bytes = fread(buf_tmp, 1, wavfile_data_size*sizeof(int32_t), file_p);
            rc_bytes_total += rc_bytes;
            //printf("\nc raw wav data=\n");
            for( i = 0; i < wavfile_data_size; i++) {
                if (bits_per_sample == 32) {
                    buf_1d_double_thread_det[i]= (1.0*(buf_tmp[i] >> 16))/pow(2.0, 15);
                } else if (bits_per_sample == 16) {
                    buf_1d_double_thread_det[i]= (1.0*(buf_tmp[i] & 0x0000ffff))/pow(2.0, 15);
                } else {
                    printf("\nc not int32 or int16, do not support this format\n");
                    break;
                }
                /*printf("%d", buf_tmp[i]); printf(", ");
                if (i%4==3) {
                    printf("\n");
                }*/
            }
            
            if (wavfile_data_size < DEF_PROCESS_FRAME) {
                for( i = wavfile_data_size; i < DEF_PROCESS_FRAME; i++) {
                    buf_1d_double_thread_det[i]= 0;
                }
            }
            //printf("\nc rc_bytes_total= %d", rc_bytes_total);
        //    printf(", rc_bytes_total/4= %d", rc_bytes_total/4);
            fclose(file_p);

            //printf(", DEF_PROCESS_FRAME= %d", DEF_PROCESS_FRAME);
            /*printf("\nc input=\n");
            for( j = 0; j < DEF_PROCESS_FRAME; j++) {
                printf("%.8f", buf_1d_double_thread_det[j]); printf(", ");
                if (j%4==3) {
                    printf("\n");
                }
            }*/
        } else {
            printf("\nc fail to open file\n");
        }
          
        #if 1  
        /*mel_pad_truncate = detection_input(  buf_1d_double_thread_det //=p_final[k, :]
                                            ,DEF_PROCESS_FRAME //DEF_PROCESS_FRAME*DEF_CHUNK //int inputs_row_i //= 63488;
                                            ,DEF_FS
                                            );*/
        
        detection_input( buf_1d_double_thread_det //=p_final[k, :]
                        ,DEF_PROCESS_FRAME //DEF_PROCESS_FRAME*DEF_CHUNK //int inputs_row_i //= 63488;
                        ,DEF_FS
                        ,&mel_pad_truncate[0]
                        );
        
    #if defined(measure_time)
        time2 = clock();
    #endif
    
        //int input = interpreter->inputs()[0];
        //float* input_data_ptr = interpreter->typed_tensor<float>(input);
        input = interpreter->inputs()[0];
      
      //#if TFLITE_INT8 == 1
      if (TFLITE_INT8 == 1) {
        i8_input_data_ptr = interpreter->typed_tensor<int8_t>(input);
      //#elif TFLITE_UINT8 == 1
      } else if (TFLITE_UINT8 == 1) {
        ui8_input_data_ptr = interpreter->typed_tensor<uint8_t>(input);
      //#else //float
      } else {
        input_data_ptr = interpreter->typed_tensor<float>(input);
      //#endif
      }
      
        //printf("\nspe c input_data_ptr= \n");
        for( i = 0; i < input_dim0; i++) {
            for( j = 0; j < input_dim1; j++) {
                // *(input_data_ptr) = (float)tensor[i][j];
              
              //#if TFLITE_INT8 == 1
              if (TFLITE_INT8 == 1) {
                //*(input_data_ptr) = (int8_t)((mel_pad_truncate[i][j] + 150.0)/150.0*255.0) - 128;
                input_data_tmp0 = (mel_pad_truncate[i][j] + 150.0)/150.0*255.0 - 128.0;
                if (input_data_tmp0 >= 0) {
                    *(i8_input_data_ptr) = floor(input_data_tmp0);
                } else {
                    *(i8_input_data_ptr) = ceil(input_data_tmp0);
                }
                //printf("%.4f", (mel_pad_truncate[i][j] + 150.0)/150.0*255.0 - 128.0); printf(", ");
                /*printf("%d", *(input_data_ptr)); printf(", ");
                if (j%4==3) {
                    printf("\n");
                }*/
                i8_input_data_ptr++;
              //#elif TFLITE_UINT8 == 1
              } else if (TFLITE_UINT8 == 1) {
                //*(input_data_ptr) = (uint8_t)((mel_pad_truncate[i][j] + 150.0)/150.0*255.0);
                input_data_tmp0 = (mel_pad_truncate[i][j] + 150.0)/150.0*255.0;
                if (input_data_tmp0 >= 0) {
                    *(ui8_input_data_ptr) = floor(input_data_tmp0);
                } else {
                    *(ui8_input_data_ptr) = ceil(input_data_tmp0);
                }
                /*printf("%u", *(input_data_ptr)); printf(", ");
                if (j%4==3) {
                    printf("\n");
                }*/
                ui8_input_data_ptr++;
              //#else //float
              } else {
                *(input_data_ptr) = (float)((mel_pad_truncate[i][j] + 150.0)/150.0);
                /*printf("%.14f", *(input_data_ptr)); printf(", ");
                if (j%4==3) {
                    printf("\n");
                }*/
                input_data_ptr++;
              //#endif
              }
                
            }
        }
        
    #if defined(measure_time)
        time0 = clock();
    #endif 
    
        interpreter->Invoke();
    
    #if defined(measure_time)
        time1 = clock(); 
        if (z == 0) {
            result[2] = (float)(time1-time0)/(CLOCKS_PER_SEC); // 1st pred(s)
        }
        
        result[3] += (float)(time1-time0)/(CLOCKS_PER_SEC); // Avg 17 pred(s)
        //printf("\nInvoke time=%f", (float)(time1-time0)/(CLOCKS_PER_SEC));
    #endif
        //    if np.max(pred) > 0.1:
        //        #        return(class_list[np.argmax(pred)]+' '+ str(pred[0,np.argmax(pred)]))
        //        return (np.argmax(pred))
        //    else:
        //        return (99)
        
        //float* output = interpreter->typed_tensor<float>(0);
        /*printf("\nResult is: \n");
        for( i = 0; i < pred_row; i++) {
            for (j = 0; j < pred_col; ++j) {
                printf("%.14f", output[i*pred_col+j]); printf(", ");
                if ((j%5==4) || ((j+1)%pred_col==0)) {
                    printf("\n");
                }
            }
        }*/
        
        output_0 = interpreter->outputs()[0];
        
      //#if TFLITE_INT8 == 1
      if (TFLITE_INT8 == 1) {
        i8_output_data_ptr_0 = interpreter->typed_tensor<int8_t>(output_0);
      //#elif TFLITE_UINT8 == 1
      } else if (TFLITE_UINT8 == 1) {
        ui8_output_data_ptr_0 = interpreter->typed_tensor<uint8_t>(output_0);
      //#else //float	
      } else {		
        output_data_ptr_0 = interpreter->typed_tensor<float>(output_0);
      //#endif
      }
        //TfLiteIntArray* output_dims_0 = interpreter->tensor(output_0)->dims;
        // assume output dims to be something like (1, 1, ... ,size)
        //auto output_size_0 = output_dims_0->data[output_dims_0->size - 1];  // 21	    
        //printf ("\nc output_data[0]= ");
        //for (i = 0; i < output_size_0; ++i) {
        for (i = 0; i < DEF_MAX_CLASS_LIST-1; ++i) {
          //#if TFLITE_INT8 == 1
          if (TFLITE_INT8 == 1) {	  
            //output_data[i] = 1.0*(*output_data_ptr_0);
            output_data[i] = 1.0*(*i8_output_data_ptr_0);
            //printf ("%d , ",*output_data_ptr_0);
            /*printf ("%f , ",output_data[i]);  
            if (i%5==4) {
                printf("\n");
            }*/
            i8_output_data_ptr_0++;
          //#elif TFLITE_UINT8 == 1
          } else if (TFLITE_UINT8 == 1) {
            output_data[i] = 1.0*(*ui8_output_data_ptr_0);  
            //printf ("%d , ",*output_data_ptr_0);
            /*printf ("%f , ",output_data[i]);
            if (i%5==4) {
                printf("\n");
            }*/
            ui8_output_data_ptr_0++;
          //#else //float	
          } else {		 
            output_data[i] = *output_data_ptr_0;   
            /*printf ("%e , ",*output_data_ptr_0);
            if (i%5==4) {
                printf("\n");
            }*/
            output_data_ptr_0++;
          //#endif
          }
            
        }
        
        output_1 = interpreter->outputs()[1];
        
      //#if TFLITE_INT8 == 1
      if (TFLITE_INT8 == 1) {
        i8_output_data_ptr_1 = interpreter->typed_tensor<int8_t>(output_1);
      //#elif TFLITE_UINT8 == 1
      } else if (TFLITE_UINT8 == 1) {
        ui8_output_data_ptr_1 = interpreter->typed_tensor<uint8_t>(output_1);
      //#else //float	
      } else {		
        output_data_ptr_1 = interpreter->typed_tensor<float>(output_1);
      //#endif
      }               
        //TfLiteIntArray* output_dims_1 = interpreter->tensor(output_1)->dims;
        //auto output_size_1 = output_dims_1->data[output_dims_1->size - 1];   // 21    
        //printf ("\nc output_data[1]= ");
        //for (i = 0; i < output_size_1; ++i) {
        for (i = 0; i < DEF_MAX_CLASS_LIST-1; ++i) {
          //#if TFLITE_INT8 == 1
          if (TFLITE_INT8 == 1) {	
            output_data_ext[0][i] = 1.0*(*i8_output_data_ptr_1);    
            //printf ("%d , ",*output_data_ptr_1);
            /*printf ("%f , ",output_data_ext[0][i]);
            if (i%5==4) {
                printf("\n");
            }*/
            i8_output_data_ptr_1++; 
          //#elif TFLITE_UINT8 == 1
          } else if (TFLITE_UINT8 == 1) {	  
            output_data_ext[0][i] = 1.0*(*ui8_output_data_ptr_1);    
            //printf ("%d , ",*output_data_ptr_1);
            /*printf ("%f , ",output_data_ext[0][i]);
            if (i%5==4) {
                printf("\n");
            }*/
            ui8_output_data_ptr_1++; 
          //#else //float	
          } else {	
            output_data_ext[0][i] = *output_data_ptr_1;	    
            /*printf ("%e , ",*output_data_ptr_1);
            if (i%5==4) {
                printf("\n");
            }*/
            output_data_ptr_1++; 
          //#endif
          }
           
        }
        
        output_2 = interpreter->outputs()[2];	
        
      //#if TFLITE_INT8 == 1
      if (TFLITE_INT8 == 1) {
        i8_output_data_ptr_2 = interpreter->typed_tensor<int8_t>(output_2);
      //#elif TFLITE_UINT8 == 1
      } else if (TFLITE_UINT8 == 1) {
        ui8_output_data_ptr_2 = interpreter->typed_tensor<uint8_t>(output_2);
      //#else //float	
      } else {		
        output_data_ptr_2 = interpreter->typed_tensor<float>(output_2);
      //#endif
      }                
        //TfLiteIntArray* output_dims_2 = interpreter->tensor(output_2)->dims;
        //auto output_size_2 = output_dims_2->data[output_dims_2->size - 1];   // 21    
        //printf ("\nc output_data[2]= ");
        //for (i = 0; i < output_size_2; ++i) {
        for (i = 0; i < DEF_MAX_CLASS_LIST-1; ++i) {
          //#if TFLITE_INT8 == 1
          if (TFLITE_INT8 == 1) {	
            output_data_ext[1][i] = 1.0*(*i8_output_data_ptr_2);    
            //printf ("%d , ",*output_data_ptr_2);
            /*printf ("%f , ",output_data_ext[1][i]);
            if (i%5==4) {
                printf("\n");
            }*/
            i8_output_data_ptr_2++;
          //#elif TFLITE_UINT8 == 1
          } else if (TFLITE_UINT8 == 1) {	  
            output_data_ext[1][i] = 1.0*(*ui8_output_data_ptr_2);    
            //printf ("%d , ",*output_data_ptr_2);
            /*printf ("%e , ",output_data_ext[1][i]);
            if (i%5==4) {
                printf("\n");
            }*/
            ui8_output_data_ptr_2++;
          //#else //float	
          } else {	
            output_data_ext[1][i] = *output_data_ptr_2;	    
            /*printf ("%e , ",*output_data_ptr_2);
            if (i%5==4) {
                printf("\n");
            }*/
            output_data_ptr_2++;
          //#endif
          }
            
        }
        
        output_3 = interpreter->outputs()[3];
        
      //#if TFLITE_INT8 == 1
      if (TFLITE_INT8 == 1) {
        i8_output_data_ptr_3 = interpreter->typed_tensor<int8_t>(output_3);
      //#elif TFLITE_UINT8 == 1
      } else if (TFLITE_UINT8 == 1) {
        ui8_output_data_ptr_3 = interpreter->typed_tensor<uint8_t>(output_3);
      //#else //float	
      } else {	
        output_data_ptr_3 = interpreter->typed_tensor<float>(output_3);
      //#endif
      }
        //TfLiteIntArray* output_dims_3 = interpreter->tensor(output_3)->dims;
        //auto output_size_3 = output_dims_3->data[output_dims_3->size - 1];   // 21    
        //printf ("\nc output_data[3]= ");
        //for (i = 0; i < output_size_3; ++i) {
        for (i = 0; i < DEF_MAX_CLASS_LIST-1; ++i) {
          //#if TFLITE_INT8 == 1
          if (TFLITE_INT8 == 1) {
            output_data_ext[2][i] = 1.0*(*i8_output_data_ptr_3);    
            //printf ("%d , ",*output_data_ptr_3);
            /*printf ("%f , ",output_data_ext[2][i]);
            if (i%5==4) {
                printf("\n");
            }*/
            i8_output_data_ptr_3++; 
          //#elif TFLITE_UINT8 == 1
          } else if (TFLITE_UINT8 == 1) {	  
            output_data_ext[2][i] = 1.0*(*ui8_output_data_ptr_3);    
            //printf ("%d , ",*output_data_ptr_3);
            /*printf ("%f , ",output_data_ext[2][i]);
            if (i%5==4) {
                printf("\n");
            }*/
            ui8_output_data_ptr_3++; 
          //#else //float	
          } else {	 
            output_data_ext[2][i] = *output_data_ptr_3;   
            /*printf ("%e , ",*output_data_ptr_3);
            if (i%5==4) {
                printf("\n");
            }*/
            output_data_ptr_3++; 
          //#endif
          }
           
        }
        
        /*pred_result_index = np_argmax_1d_float(output_data, pred_col);
        printf("\nc output_data[0] result= %d", pred_result_index); 
        for (i = 0; i < DEF_MAX_CLASS_LIST-1; ++i) {	    
            output_data[i] = output_data_ext[0][i];
        }
        pred_result_index = np_argmax_1d_float(output_data, pred_col);
        printf("\nc output_data[1] result= %d", pred_result_index); 
        for (i = 0; i < DEF_MAX_CLASS_LIST-1; ++i) {	    
            output_data[i] = output_data_ext[1][i];
        }
        pred_result_index = np_argmax_1d_float(output_data, pred_col);
        printf("\nc output_data[2] result= %d", pred_result_index);
        for (i = 0; i < DEF_MAX_CLASS_LIST-1; ++i) {	    
            output_data[i] = output_data_ext[2][i];
        } 
        pred_result_index = np_argmax_1d_float(output_data, pred_col);
        printf("\nc output_data[3] result= %d", pred_result_index); */
        
        //printf ("\nc output_data= ");
        //for (i = 0; i < output_size_0; ++i) {
        for (i = 0; i < DEF_MAX_CLASS_LIST-1; ++i) {	    
            output_data[i] = (output_data[i] + output_data_ext[0][i] + output_data_ext[1][i] + output_data_ext[2][i])/4.0;	
            /*printf ("%.8e , ",output_data[i]);
            if (i%5==4) {
                printf("\n");
            }*/
        }
        
        max_pred = np_max_1d_float(output_data, pred_col);
        pred_result_index = DEF_MAX_CLASS_LIST-1; //99;
      //#if TFLITE_INT8 == 1
      if (TFLITE_INT8 == 1) {
        if (max_pred > (0.1*255 - 128)) {
            pred_result_index = np_argmax_1d_float(output_data, pred_col);
        }
      //#elif TFLITE_UINT8 == 1
      } else if (TFLITE_UINT8 == 1) {
        if (max_pred > (0.1*255)) {
            pred_result_index = np_argmax_1d_float(output_data, pred_col);
        }
      //#else //float
      } else {		
        if (max_pred > 0.1) {
            pred_result_index = np_argmax_1d_float(output_data, pred_col);
      //#endif
      }
            //pred_result_index = np_argmax_1d_float(output_data, pred_col);
        }
        
    #if defined(measure_time)
        time3 = clock(); 
        result[4] += (float)(time3-time2)/(CLOCKS_PER_SEC); // Avg 17 set_io + pred(s)
        //printf("\nInvoke time=%f", (float)(time3-time2)/(CLOCKS_PER_SEC));
    #endif
        
        //detection_sor1_result = pred_result_index;
        /*printf("\ndetection_sor1_result= %d", pred_result_index); 
        printf("\noutput_data[pred_result_index]= %.8f", output_data[pred_result_index]); 
        printf("\nsource 1 detection result= %s\n", class_list[pred_result_index]);*/
        #endif //#if 1
        
        if (pred_result_index == file_class_index) {
            result[0] += 1; // Accuracy
        }

    }


    if (EDGE_TPU == 1) {
        printf("\n   TPU   |");
    } else {
        printf("\n   CPU   |");
    }
    
    if (TFLITE_MODEL_OPT == 1) {
        printf("     Y     |");
    } else {
        printf("     N     |");
    }
    
    if (TFLITE_INT8 == 1) {
        printf("    int8   |");
    } else if (TFLITE_UINT8 == 1) {
        printf("   uint8   |");
    } else if (TFLITE_FP16 == 1) {
        printf("     fp16  |");
    } else if (TFLITE_FP32FALLBACK == 1) {
        printf("     fp32fb|");
    } else {
        printf("     fp32  |");
    }

    printf("       %d      |", DEF_TFLITE_NUMTHREADS);
    printf("   %.2f     |", result[0]/17.0*100.0);
    printf("    %.5f    |", result[1]);
    printf("    %.5f   |", result[2]);
    printf("     %.5f    |", result[3]/17.0);
    printf("       %.5f           |", result[4]/17.0);
    

    
    free_2d_double_array(mel_pad_truncate, DEF_TD_PAD_TRUNCATE_LENGTH);
    free(buf_1d_double_thread_det);
    
    free(output_data);
    free_2d_float_array(output_data_ext, 3);
    
    free(buf_tmp);
    free(buf_header);
    //free(buf_data);
    //free(buf_data_wavfile);
    free(result);
    
    //#ifdef SET_PTHREAD
    //    pthread_exit(NULL);
    //#endif

}

void detection_input(    double *inputs_i
                        ,int inputs_row_i //= 63488;
                        //,model
                        //,char class_list[100]
                        ,double sample_rate_i
                        //,int uni_sample
                        ,double **mel_pad_truncate_o
                        )
{
    //int i, j;    

    //def detection_PC(inputs, model, class_list, sample_rate=48000, uni_sample=48000):
    //    inputs, duration = librosa.effects.trim(inputs, top_db=48)
    double top_db=48.0;
    #ifdef DEBUG_librosa
        top_db=10.5; //48.0;
    #endif
    int *duration = create_1d_int_array(2);
    
    /*#ifdef measure_time
        clock_t time0, time1; 
        time0 = clock();
    #endif*/ 

    double *inputs_trimmed = librosa_effects_trim(   inputs_i
                                                    ,inputs_row_i
                                                    ,top_db
                                                    ,0 //frame_length_i //set 0
                                                    ,0 //hop_length_i //set 0
                                                    ,&duration[0]
                                                    );

    /*#ifdef measure_time
        time1 = clock();    
        printf("\nspe c librosa_effects_trim= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif*/

    //    mel = extract(inputs, sample_rate)
    int *dim_power_to_db = create_1d_int_array(2);
    
    /*#ifdef measure_time
        //clock_t time0, time1; 
        time0 = clock();
    #endif*/
    
    double **mel = detection_extract(    inputs_trimmed
                                        ,duration[1]-duration[0] //,inputs_row_i
                                        ,sample_rate_i
                                        ,&dim_power_to_db[0]
                                        //,double **power_to_db_o //
                                        );

    /*#ifdef measure_time
        time1 = clock();    
        printf("\nspe c detection_extract= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif*/

    //    mel = pad_truncate(mel, 126)
    //[lc del] int target_length = 126;
    double pad_value = 0.0;
    
    /*#ifdef measure_time
        //clock_t time0, time1; 
        time0 = clock();
    #endif*/

    detection_pad_truncate(  mel
                            ,dim_power_to_db
                            //[lc del] ,target_length
                            ,DEF_TD_PAD_TRUNCATE_LENGTH //[lc add]
                            ,pad_value
                            ,&mel_pad_truncate_o[0]
                            );

    /*#ifdef measure_time
        time1 = clock();    
        printf("\nspe c detection_pad_truncate= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif*/

    free(duration);
    free(inputs_trimmed);
    free_2d_double_array(mel, dim_power_to_db[0]);
    
    free(dim_power_to_db);

    //return mel_pad_truncate_o;
}
