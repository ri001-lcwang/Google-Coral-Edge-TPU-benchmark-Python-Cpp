//typedef double _Complex complex_t;

// Avoids C++ name mangling with extern "C"
/*#include <math.h>
#include <complex.h>
#include <fftw3.h>*/


#include "parameter_set.h"
#include "sub_misc_c0.h"


//#define for_python_or_c
//#define DEBUG_librosa
#define measure_time


#ifdef for_python_or_c
void test_detection(     double inputs_i[] //[63488]
                        //,model
                        //,char class_list[100]
                        ,double sample_rate_i
                        //,int uni_sample
                    );
int detection(   double *inputs_i
                ,int inputs_row_i //= 63488;
                //,model
                //,char class_list[100]
                ,double sample_rate_i
                //,int uni_sample
                );
#endif
double **detection_extract(  double *inputs_i
                            ,int dim_inputs_i //63488
                            ,double sample_rate_i
                            ,int *dim_power_to_db_o
                            //,double **power_to_db_o //
                            );
void detection_pad_truncate( double **x_i
                            ,int *dim_x_i
                            ,int target_length_i
                            ,double pad_value_i
                            ,double **array_o
                            );
double *librosa_effects_trim(    double *y_i
                                ,int y_row_i
                                ,double top_db_i
                                ,int frame_length_i //set 0
                                ,int hop_length_i //set 0
                                ,int *duration_o 
                                );
int *librosa_effects__signal_to_frame_nonsilent( double *y_i
                                                ,int y_row_i
                                                ,double top_db_i
                                                ,int frame_length_i
                                                ,int hop_length_i
                                                ,int *frame_col_o
                                                //,int *non_silent_o
                                                );
double *librosa_feature_rms(     double *y_i
                                ,int y_row_i
                                ,int frame_length_i
                                ,int hop_length_i
                                ,int center_i //set 1
                                ,int *frame_col_o
                                );
int librosa_frames_to_samples(   int frames_i
                                ,int hop_length_i
                                ,int n_fft_i //set 0
                                );
void librosa_stft_dim(   int y_row_i //63488
                        ,int n_fft_i
                        ,int hop_length_i
                        ,int win_length_i //set 0
                        ,int *dim_o //{fft_window       ,y_pad           ,y_frames_row          ,y_frames_col 
                                    //,stft_matrix_row  ,stft_matrix_col ,fftw3_1d_rfft_data_in ,fftw3_1d_rfft_data_out
                                    //,hop_length}
                        );
void librosa_stft_data(  double *y_i
                        ,int y_row_i //63488
                        ,int n_fft_i
                        ,int *dim_i //{fft_window       ,y_pad           ,y_frames_row          ,y_frames_col 
                                    //,stft_matrix_row  ,stft_matrix_col ,fftw3_1d_rfft_data_in ,fftw3_1d_rfft_data_out
                                    //,hop_length}
                        //,double complex **stft_matrix_o //
                        #ifdef for_python_or_c
                            ,double complex **stft_matrix_o
                        #else //C++
                            ,double _Complex **stft_matrix_o
                        #endif
                        );
void librosa_filters_get_window(     int nx_i
                                    ,double* win_o
                                    );
void librosa_util_frame( double* x_i
                        //,int frame_length_i
                        ,int hop_length_i
                        ,int dim_frame_row_i
                        ,int dim_frame_col_i
                        ,double** frame_o
                        );
void librosa_filters_mel(    double sr_i
                            ,int n_fft_i
                            ,int n_mels_i
                            ,double fmin_i //set 0.0
                            ,double fmax_i //set 0.0
                            ,double **weights_o
                        );
double *librosa_fft_frequencies( double sr_i
                                ,int n_fft_i
                            );
double *librosa_mel_frequencies(     int n_mels_i
                                    ,double fmin_i
                                    ,double fmax_i
                                    ,int htk_i //set 0
                                );
double librosa_hz_to_mel(    double frequencies_i
                            ,int htk_i //set 0
                        );
double *librosa_mel_to_hz_1d_double( double *mels_i
                                    ,int mels_row_i
                                    ,int htk_i //set 0
                                    );
void librosa_power_to_db_np_max_1d_double(   double *s_i
                                            ,int s_row_i
                                            ,double amin_i
                                            ,double top_db_i
                                            ,double *log_spec_o
                                            ); //ref=np.max
void librosa_power_to_db_np_max_2d_double(   double **s_i
                                            ,int s_row_i
                                            ,int s_col_i
                                            ,double amin_i
                                            ,double top_db_i
                                            ,double **log_spec_o
                                            ); //ref=np.max


