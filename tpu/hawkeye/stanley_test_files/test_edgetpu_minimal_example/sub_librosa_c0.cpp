/*#include <stdio.h>
#include <stdlib.h>
#define PI acos(-1)*/
#include "sub_librosa_c0.h"

/*#pragma comment(lib , "blas.lib") 
#pragma comment(lib , "clapack.lib")

//void *malloc(size_t n) ;

#include "f2c.h"
#include "clapack.h"

#include <time.h>*/

/*
// create fftw3 1d fft plan
int fftw3_N = DEF_NWIN;
short fftw3_setting = 0;
double *fftw3_data_in = create_1d_double_array(fftw3_N);
double complex *fftw3_data_out = create_1d_double_complex_array(fftw3_N);
fftw_plan fftw3_plan_1d_fft;
switch(fftw3_setting) { 
    case 1: 
        fftw3_plan_1d_fft = fftw_plan_dft_1d(fftw3_N, fftw3_data_in, fftw3_data_out, -1, FFTW_MEASURE);
        break; 
    default:
        fftw3_plan_1d_fft = fftw_plan_dft_1d(fftw3_N, fftw3_data_in, fftw3_data_out, -1, FFTW_ESTIMATE);
}*/


#ifdef for_python_or_c
void test_detection(     double inputs_i[] //[63488]
                        //,model
                        //,char class_list[100]
                        ,double sample_rate_i
                        //,int uni_sample
                    )
{
    int inputs_row = 63488;
    #ifdef measure_time
        clock_t time0, time1; 
        time0 = clock();
    #endif
    int pred_result_index = detection(   inputs_i
                                        ,inputs_row
                                        //,model
                                        //,char class_list[100]
                                        ,sample_rate_i
                                        //,int uni_sample
                                        );

    #ifdef measure_time
        time1 = clock();    
        printf("\nspe c test_detection time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif 
    

}

int detection(   double *inputs_i
                ,int inputs_row_i //= 63488;
                //,model
                //,char class_list[100]
                ,double sample_rate_i
                //,int uni_sample
                )
{
    //def detection_PC(inputs, model, class_list, sample_rate=48000, uni_sample=48000):
    //    inputs, duration = librosa.effects.trim(inputs, top_db=48)
    double top_db=48.0;
    #ifdef DEBUG_librosa
        top_db=10.5; //48.0;
    #endif
    int *duration = create_1d_int_array(2);
    double *inputs_trimmed = librosa_effects_trim(   inputs_i
                                                    ,inputs_row_i
                                                    ,top_db
                                                    ,0 //frame_length_i //set 0
                                                    ,0 //hop_length_i //set 0
                                                    ,&duration[0]
                                                    );


    //    mel = extract(inputs, sample_rate)
    int *dim_power_to_db = create_1d_int_array(2);
    double **mel = detection_extract(    inputs_trimmed
                                        ,duration[1]-duration[0] //,inputs_row_i
                                        ,sample_rate_i
                                        ,&dim_power_to_db[0]
                                        //,double **power_to_db_o //
                                        );


    //    mel = pad_truncate(mel, 126)
    int target_length = 126;
    double pad_value = 0.0;
    double **mel_pad_truncate = detection_pad_truncate(  mel
                                                        ,dim_power_to_db
                                                        ,target_length
                                                        ,pad_value
                                                        );
    #ifdef DEBUG_librosa
        int i, j;
        //printf("\nspe c detection, librosa.effects.trim, result: inputs_trimmed=\n");
        //for( j = 0; j < duration[1] - duration[0]; j++) {
        //    printf("%.8f", inputs_trimmed[j]); printf(", ");
        //    if (j%4==3) {
        //        printf("\n");
        //    }
        //}
        //
        //printf("\nspe c detection, extract, result: dim_power_to_db=\n");
        //for( j = 0; j < 2; j++) {
        //    printf("%d", dim_power_to_db[j]); printf(", ");
        //    if (j%4==3) {
        //        printf("\n");
        //    }
        //}
        //printf("\nspe c detection, extract, result: mel=\n");
        //for( i = 0; i < dim_power_to_db[0]; i++) {
        //    for( j = 0; j < dim_power_to_db[1]; j++) {
        //        printf("%.8f", mel[i][j]); printf(", ");
        //        if (j%4==3) {
        //            printf("\n");
        //        }
        //    }
        //}
        
        printf("\nspe c detection, pad_truncate, input: mel.shape[0]= %d", dim_power_to_db[0]);
        printf("\nspe c detection, pad_truncate, input: mel.shape[1]= %d", dim_power_to_db[1]);
        printf("\nspe c detection, pad_truncate, result: mel_pad_truncate.shape[0]= %d", target_length);
        printf("\nspe c detection, pad_truncate, result: mel_pad_truncate.shape[1]= %d", dim_power_to_db[1]);
        printf("\nspe c detection, pad_truncate, result: mel_pad_truncate=\n");
        for( i = 0; i < target_length; i++) {
            for( j = 0; j < dim_power_to_db[1]; j++) {
                printf("%.8f", mel_pad_truncate[i][j]); printf(", ");
                if (j%4==3) {
                    printf("\n");
                }
            }
        }
    #endif


    //    pred = model.predict(mel.reshape((1, 126, 64, 1)))
    //    if np.max(pred) > 0.3:
    //        #        return(class_list[np.argmax(pred)]+' '+ str(pred[0,np.argmax(pred)]))
    //        return (np.argmax(pred))
    //    else:
    //        return (99)
    int pred_row = 4;
    int pred_col = 21;
    int pred_total_length = pred_row*pred_col;
    float **pred = create_2d_float_array(pred_row, pred_col);
    //pred = model.predict(mel.reshape((1, 126, 64, 1)))
    float *pred_flatten = flatten_2d_float_array(pred, pred_row, pred_col);
    float max_pred = np_max_1d_float(pred_flatten, pred_total_length);
    int pred_result_index = 99;

    if (max_pred > 0.3) {
        pred_result_index = np_argmax_1d_float(pred_flatten, pred_total_length);
    }

    free(duration);
    free(inputs_trimmed);
    free_2d_double_array(mel, dim_power_to_db[0]);
    free_2d_double_array(mel_pad_truncate, target_length);
    free(dim_power_to_db);
    free_2d_float_array(pred, pred_row);
    free(pred_flatten);

    return pred_result_index;
    
}
#endif

double **detection_extract(  double *inputs_i
                            ,int dim_inputs_i //63488
                            ,double sample_rate_i
                            ,int *dim_power_to_db_o
                            //,double **power_to_db_o //
                            )
{
    //def extract(x, sample_rate):
    //    """Transform the given signal into a logmel feature vector.11
    //    Args:
    //    x (np.ndarray): Input time-series signal.
    //    sample_rate (number): Sampling rate of signal.
    //    Returns:
    //    np.ndarray: The logmel feature vector.
    //    """
    //    # Resample to target sampling rate
    //    # x = librosa.resample(x, sample_rate, uni_sample)
    //    # Compute short-time Fourier transform
    //    D = librosa.stft(x, n_fft=1024, hop_length=512)
    int i, j, k;
    int n_fft=1024;
    int hop_length=512;
    int win_length=0;
    int *dim_stft = create_1d_int_array(9);
    librosa_stft_dim(    dim_inputs_i //63488
                        ,n_fft
                        ,hop_length
                        ,win_length //set 0
                        ,&dim_stft[0]   //{fft_window       ,y_pad           ,y_frames_row          ,y_frames_col 
                                        //,stft_matrix_row  ,stft_matrix_col ,fftw3_1d_rfft_data_in ,fftw3_1d_rfft_data_out
                                        //,hop_length}
                        );

    #ifdef for_python_or_c
        double complex **stft_matrix = create_2d_double_complex_array(dim_stft[4], dim_stft[5]);
    #else //C++
        double _Complex **stft_matrix = create_2d_double_complex_array(dim_stft[4], dim_stft[5]);
    #endif
    librosa_stft_data(   inputs_i
                        ,dim_inputs_i //63488
                        ,n_fft
                        ,dim_stft //{fft_window       ,y_pad           ,y_frames_row          ,y_frames_col 
                                  //,stft_matrix_row  ,stft_matrix_col ,fftw3_1d_rfft_data_in ,fftw3_1d_rfft_data_out
                                  //,hop_length}
                        ,&stft_matrix[0] //
                        );

    //    # Create Mel filterbank matrix
    //    mel_fb = librosa.filters.mel(sr=uni_sample,
    //                                 n_fft=1024,
    //                                 n_mels=64)
    int n_mels=64;
    double **mel_fb = create_2d_double_array(n_mels, 1 + n_fft/2);
    librosa_filters_mel(     sample_rate_i
                            ,n_fft
                            ,n_mels
                            ,0.0 //fmin_i //set 0.0
                            ,0.0 //fmax_i //set 0.0
                            ,&mel_fb[0]
                        );
    
    
    //    # Transform to Mel frequency scale
    //    S = np.dot(mel_fb, np.abs(D) ** 2).T
    double **arr_np_dot_t = create_2d_double_array(dim_stft[5] ,n_mels);
    double s=0;
    for(i = 0 ; i < n_mels ; i++) {
        for(j = 0 ; j < dim_stft[5] ; j++) {
            s = 0;
            for(k = 0 ; k < dim_stft[4] ; k++) {
                s += mel_fb[i][k]*(pow(creal(stft_matrix[k][j]), 2) + pow(cimag(stft_matrix[k][j]), 2));
            }
            arr_np_dot_t[j][i] = s;
        }
    }


    //   # Apply log nonlinearity and return as float32
    //   return librosa.power_to_db(S, ref=np.max, top_db=None)
    dim_power_to_db_o[0] = dim_stft[5];
    dim_power_to_db_o[1] = n_mels;
    double **power_to_db_o = create_and_copy_2d_double_array(dim_stft[5] ,n_mels, arr_np_dot_t);

    librosa_power_to_db_np_max_2d_double(    arr_np_dot_t
                                            ,dim_stft[5]
                                            ,n_mels
                                            ,1e-10 //amin_i
                                            ,0.0 //top_db_i
                                            ,&power_to_db_o[0]
                                            ); //ref=np.max
    
    /*#ifdef DEBUG_librosa
        printf("\nc extract, dim_stft=\n");
        for( j = 0; j < 9; j++) {
            printf("%d", dim_stft[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }
        printf("\nc extract, D=\n");
        for( i = 0; i < dim_stft[4]; i++) {
            for( j = 0; j < dim_stft[5]; j++) {
                printf("%.14f%+.14fj", creal(stft_matrix[i][j]), cimag(stft_matrix[i][j]));
                printf(", ");
                if (j%2==1) {
                    printf("\n");
                }
            }
        }
    #endif*/
    
    free_2d_double_complex_array(stft_matrix ,dim_stft[4]);
    free_2d_double_array(mel_fb ,n_mels);
    free_2d_double_array(arr_np_dot_t ,dim_stft[5]);
    free(dim_stft);
    
    return power_to_db_o;

}

void detection_pad_truncate( double **x_i
                            ,int *dim_x_i
                            ,int target_length_i
                            ,double pad_value_i
                            ,double **array_o
                            )
{
    //def pad_truncate(x, length, pad_value=0):
    //    x_len = len(x)
    //    if x_len > length:
    //        x = x[:length]
    //    elif x_len < length:
    //        padding = np.full((length - x_len,) + x.shape[1:], pad_value)
    //        x = np.concatenate((x, padding))
    //    return x    
    int i, j;
    int x_len = dim_x_i[0];
    int x_len_no_pad = x_len;
    //int x_len_total = x_len_no_pad;
    int padding = 0;

    if (x_len > target_length_i) {
        x_len_no_pad = target_length_i;
    } else if (x_len < target_length_i) { //(length - x_len,)=target_length_i-x_len, x.shape[1:]=dim_x_i[1]
        padding = target_length_i - x_len;
        //x_len_total += padding;
    }

    //x_len_total = x_len_no_pad + padding;
    
    /*double **array;
    int N = target_length_i; //x_len_total;
    int M = dim_x_i[1];

    array = (double**)malloc(N*sizeof(double *));

    if (array == NULL) {
		fprintf(stderr, "Out of memory");
		exit(0);
	}
    
    for(i = 0 ; i < N ; i++)
    {
        array[i] = (double*)malloc( M*sizeof(double) );
        if (array[i] == NULL) {
            fprintf(stderr, "Out of memory");
            exit(0);
        }
    }*/
        
    for(i = 0 ; i < x_len_no_pad ; i++) {
        for(j = 0 ; j < DEF_TD_N_MELS ; j++) {
            array_o[i][j] = x_i[i][j];
        }
    }

    if (padding > 0) {
        for(i = x_len_no_pad ; i < target_length_i ; i++) {
            for(j = 0 ; j < DEF_TD_N_MELS ; j++) {
                array_o[i][j] = pad_value_i;
            }
        }
    }
    
    /*#ifdef DEBUG_librosa
        printf("\nspe c pad_truncate, input: mel.shape[0]= %d", dim_x_i[0]);
        printf("\nspe c pad_truncate, input: mel.shape[1]= %d", dim_x_i[1]);
        printf("\nspe c pad_truncate, result: mel_pad_truncate.shape[0]= %d", target_length_i);
        printf("\nspe c pad_truncate, result: mel_pad_truncate.shape[1]= %d", dim_x_i[1]);
        printf("\nspe c pad_truncate, result: mel_pad_truncate=\n");
        for( i = 0; i < target_length_i; i++) {
            for( j = 0; j < dim_x_i[1]; j++) {
                printf("%.8f", array_o[i][j]); printf(", ");
                if (j%4==3) {
                    printf("\n");
                }
            }
        }
    #endif*/
    
    //return array;

}

double *librosa_effects_trim(    double *y_i
                                ,int y_row_i
                                ,double top_db_i
                                ,int frame_length_i //set 0
                                ,int hop_length_i //set 0
                                ,int *duration_o 
                                )
{
    //librosa/effects.py
    //def trim(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):
    //    non_silent = _signal_to_frame_nonsilent(y,
    //                                            frame_length=frame_length,
    //                                            hop_length=hop_length,
    //                                            ref=ref,
    //                                            top_db=top_db)
    //    nonzero = np.flatnonzero(non_silent)
    //    if nonzero.size > 0:
    //        # Compute the start and end positions
    //        # End position goes one frame past the last non-zero
    //        start = int(core.frames_to_samples(nonzero[0], hop_length))
    //        end = min(y.shape[-1],
    //                  int(core.frames_to_samples(nonzero[-1] + 1, hop_length)))
    //    else:
    //        # The signal only contains zeros
    //        start, end = 0, 0
    //    # Build the mono/stereo index
    //    full_index = [slice(None)] * y.ndim
    //    full_index[-1] = slice(start, end)
    //    return y[tuple(full_index)], np.asarray([start, end])
    int i;
    int frame_length = frame_length_i;
    int hop_length = hop_length_i;

    if (frame_length_i == 0) {
        frame_length = 2048;
    }

    if (hop_length_i == 0) {
        hop_length = 512;
    }

    int *frame_col = create_1d_int_array(1);
    int *non_silent = librosa_effects__signal_to_frame_nonsilent(    y_i
                                                                    ,y_row_i
                                                                    ,top_db_i
                                                                    ,frame_length
                                                                    ,hop_length
                                                                    ,&frame_col[0]
                                                                    //,int *non_silent_o
                                                                    );
        
    int nonzero_size_gt_0 = 0;
    int nonzero_min = frame_col[0];
    int nonzero_max = frame_col[0];
    int start = 0;
    int end = 0;

    for(i = 0 ; i < frame_col[0] ; i++) {
        if (non_silent[i] == 1) {
            nonzero_size_gt_0 = 1;
            nonzero_min = i;
            break;
        }
    }

    if (nonzero_min == frame_col[0] - 1) {
        nonzero_max = nonzero_min;
    } else if (nonzero_size_gt_0 == 1) {
        for(i = frame_col[0] - 1 ; i >= 0 ; i--) {
            if (non_silent[i] == 1) {
                nonzero_max = i;
                break;
            }
        }
    }

    if (nonzero_size_gt_0 == 1) {
        start = librosa_frames_to_samples(   nonzero_min
                                            ,hop_length
                                            ,0 //n_fft_i //set 0
                                         );
        
        end = librosa_frames_to_samples(     nonzero_max + 1
                                            ,hop_length
                                            ,0 //n_fft_i //set 0
                                         );
    
        if (end > y_row_i) {
            end = y_row_i;
        }
    }

    duration_o[0] = start;
    duration_o[1] = end;

    int y_trimmed_row = end - start;
    double *y_trimmed_o;    
    y_trimmed_o = (double*)malloc(y_trimmed_row*sizeof(double));    
    if (y_trimmed_o == NULL) {
		fprintf(stderr, "Out of memory");
		exit(0);
	}
    
    for(i = 0 ; i < y_trimmed_row ; i++) {
        y_trimmed_o[i] = y_i[start+i];
    }
    
    /*#ifdef DEBUG_librosa
        printf("\nspe c librosa_effects_trim, non_silent.shape= %d", frame_col[0]);
        printf("\nape c non_silent=\n");
        for( i = 0; i < frame_col[0]; i++) {
            printf("%d", non_silent[i]); printf(", ");
            if (i%4==3) {
                printf("\n");
            }
        }
        printf("\nspe c librosa_effects_trim, nonzero_size_gt_0= %d", nonzero_size_gt_0);
        printf("\nspe c librosa_effects_trim, nonzero_min= %d", nonzero_min);
        printf("\nspe c librosa_effects_trim, nonzero_max= %d", nonzero_max);
        printf("\nspe c librosa_effects_trim, start= %d", start);
        printf("\nspe c librosa_effects_trim, end= %d", end);
        printf("\nspe c librosa_effects_trim, duration_o[0]= %d", duration_o[0]);
        printf("\nspe c librosa_effects_trim, duration_o[1]= %d", duration_o[1]);
    #endif*/

    free(frame_col);
    free(non_silent);

    return y_trimmed_o;

}

int *librosa_effects__signal_to_frame_nonsilent( double *y_i
                                                ,int y_row_i
                                                ,double top_db_i
                                                ,int frame_length_i
                                                ,int hop_length_i
                                                ,int *frame_col_o
                                                //,int *non_silent_o
                                                )
{
    //librosa/effects.py
    //def _signal_to_frame_nonsilent(y, frame_length=2048, hop_length=512, top_db=60,
    //                               ref=np.max):
    //    # Convert to mono
    //    y_mono = core.to_mono(y)
    //    # Compute the MSE for the signal
    //    mse = feature.rms(y=y_mono,
    //                      frame_length=frame_length,
    //                      hop_length=hop_length)**2
    //    return (core.power_to_db(mse.squeeze(),
    //                             ref=ref,
    //                             top_db=None) > - top_db)
    int i;
    //int *frame_col_o = create_1d_int_array(1);
    double *mse = librosa_feature_rms(   y_i
                                        ,y_row_i
                                        ,frame_length_i
                                        ,hop_length_i
                                        ,1 //center_i //set 1
                                        ,&frame_col_o[0]
                                        );
    
    for(i = 0 ; i < frame_col_o[0] ; i++) {
        mse[i] = pow(mse[i], 2);
    }

    double *log_spec = create_and_copy_1d_double_array(frame_col_o[0], mse);

    librosa_power_to_db_np_max_1d_double(    mse
                                            ,frame_col_o[0]
                                            ,1e-10 //amin_i
                                            ,top_db_i
                                            ,&log_spec[0]
                                            ); //ref=np.max

    int *non_silent_o = create_1d_int_array(frame_col_o[0]);

    for(i = 0 ; i < frame_col_o[0] ; i++) {
        if (log_spec[i] > -1.0*top_db_i) {
            non_silent_o[i] = 1;
        }
    }

    free(mse);
    free(log_spec);

    return non_silent_o;

}

double *librosa_feature_rms(     double *y_i
                                ,int y_row_i
                                ,int frame_length_i
                                ,int hop_length_i
                                ,int center_i //set 1
                                ,int *frame_col_o
                                )
{
    //librosa/feature/spectral.py
    //def rms(y=None, S=None, frame_length=2048, hop_length=512,
    //        center=True, pad_mode='reflect'):
    //    if y is not None:
    //        y = to_mono(y)
    //        if center:
    //            y = np.pad(y, int(frame_length // 2), mode=pad_mode)
    //        x = util.frame(y,
    //                       frame_length=frame_length,
    //                       hop_length=hop_length)
    //        # Calculate power
    //        power = np.mean(np.abs(x)**2, axis=0, keepdims=True)
    //    elif S is not None:
    //        # Check the frame length
    //        if S.shape[0] != frame_length // 2 + 1:
    //            raise ParameterError(
    //                    'Since S.shape[0] is {}, '
    //                    'frame_length is expected to be {} or {}; '
    //                    'found {}'.format(
    //                            S.shape[0],
    //                            S.shape[0] * 2 - 2, S.shape[0] * 2 - 1,
    //                            frame_length))
    //        # power spectrogram
    //        x = np.abs(S) ** 2
    //        # Adjust the DC and sr/2 component
    //        x[0] *= 0.5
    //        if frame_length % 2 == 0:
    //            x[-1] *= 0.5
    //        # Calculate power
    //        power = 2 * np.sum(x, axis=0, keepdims=True) / frame_length**2
    //    else:
    //        raise ParameterError('Either `y` or `S` must be input.')
    //    return np.sqrt(power)
    int i, j;
    int frame_row = frame_length_i;

    if (center_i == 1) {
        frame_col_o[0] = y_row_i/hop_length_i + 1;
    } else {
        frame_col_o[0] = (y_row_i - frame_length_i)/hop_length_i + 1;
    }

    double **x = create_2d_double_array(frame_row, frame_col_o[0]);

    if (center_i == 1) {
        double *y_pad = create_1d_double_array(y_row_i + frame_length_i);
        np_pad_reflect_1d_double(y_i ,y_row_i ,frame_length_i/2 ,&y_pad[0]);
        librosa_util_frame(  y_pad
                            //,int frame_length_i
                            ,hop_length_i
                            ,frame_row
                            ,frame_col_o[0]
                            ,&x[0]
                            );
        free(y_pad);
        
    } else {
        //double *y_pad = create_and_copy_1d_double_array(y_row_i ,y_i);
        librosa_util_frame(  y_i
                            //,int frame_length_i
                            ,hop_length_i
                            ,frame_row
                            ,frame_col_o[0]
                            ,&x[0]
                            );
    
    }    

    double *power_o = create_1d_double_array(frame_col_o[0]);

    for(j = 0 ; j < frame_col_o[0] ; j++) {
        for(i = 0 ; i < frame_row ; i++) {
            //power_o[j] += x[i][j];
            power_o[j] += pow(x[i][j], 2);
        }
        
        power_o[j] = sqrt(power_o[j]/(1.0*frame_row));
    }

    //free(y_pad);
    free_2d_double_array(x, frame_row);

    return power_o;

}

int librosa_frames_to_samples(   int frames_i
                                ,int hop_length_i
                                ,int n_fft_i //set 0
                                )
{
    //librosa/core/time_frequency.py
    //def frames_to_samples(frames, hop_length=512, n_fft=None):
    //    offset = 0
    //    if n_fft is not None:
    //        offset = int(n_fft // 2)
    //    return (np.asanyarray(frames) * hop_length + offset).astype(int)    
    int offset = 0;

    if (n_fft_i != 0) {
        offset = n_fft_i/2;
    }
    
    //printf("\nspe c frames_i*hop_length_i + offset= %d", frames_i*hop_length_i + offset);
    
    return frames_i*hop_length_i + offset;

}

void librosa_stft_dim(   int y_row_i //63488
                        ,int n_fft_i
                        ,int hop_length_i
                        ,int win_length_i //set 0
                        ,int *dim_o //{fft_window       ,y_pad           ,y_frames_row          ,y_frames_col 
                                    //,stft_matrix_row  ,stft_matrix_col ,fftw3_1d_rfft_data_in ,fftw3_1d_rfft_data_out
                                    //,hop_length}
                        )
{
    //librosa/core/spectrum.py
    //@cache(level=20)
    //def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
    //         center=True, dtype=np.complex64, pad_mode='reflect'):
    //# By default, use the entire frame
    //if win_length is None:
    //    win_length = n_fft
    int win_length = win_length_i;

    if (win_length_i==0) {
        win_length = n_fft_i;
    }


    //# Set the default hop, if it's not already specified
    //if hop_length is None:
    //    hop_length = int(win_length // 4)
    int hop_length = hop_length_i;

    if (hop_length_i==0) {
        hop_length = win_length/4;
    }

    int frame_length=n_fft_i;

    dim_o[0] = win_length; //fft_window
    dim_o[1] = y_row_i+n_fft_i; //y_pad
    dim_o[2] = frame_length; //dim_y_frames_row
    //dim_o[3] = (y_row_i - frame_length)/hop_length + 1; //dim_y_frames_col
    dim_o[3] = (dim_o[1] - frame_length)/hop_length + 1; //dim_y_frames_col
    dim_o[4] = 1 + n_fft_i/2; //stft_matrix_row
    dim_o[5] = dim_o[3]; //stft_matrix_col
    dim_o[6] = n_fft_i; //fftw3_1d_rfft_data_in
    dim_o[7] = dim_o[4]; //fftw3_1d_rfft_data_out
    dim_o[8] = hop_length; //hop_length
    
}

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
                        )
{
    int i, j;
    //fft_window = get_window(window, win_length, fftbins=True)
    double *fft_window = create_1d_double_array(dim_i[0]);
    librosa_filters_get_window(       dim_i[0] //nx_i
                                    ,&fft_window[0] //win_o
                                    );


    //# Pad the window out to n_fft size
    //fft_window = util.pad_center(fft_window, n_fft)
    //# Reshape so that the window can be broadcast
    //fft_window = fft_window.reshape((-1, 1))
    //# Check audio is valid
    //util.valid_audio(y)
    //# Pad the time series so that frames are centered
    //if center:
    //    y = np.pad(y, int(n_fft // 2), mode=pad_mode)
    double *y_pad = create_1d_double_array(dim_i[1]);
    np_pad_reflect_1d_double(y_i ,y_row_i ,n_fft_i/2 ,&y_pad[0]);


    //# Window the time series.
    //y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)
    double **y_frames = create_2d_double_array(dim_i[2], dim_i[3]);
    librosa_util_frame(  y_pad
                        //,int frame_length_i
                        ,dim_i[8]
                        ,dim_i[2]
                        ,dim_i[3]
                        ,&y_frames[0]
                        );
    
    /*#ifdef DEBUG_librosa
        printf("\nspe c librosa_stft_data, dim_i=\n");
        for( j = 0; j < 9; j++) {
            printf("%d", dim_i[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }     
        
        printf("\nspe c librosa_stft_data, fft_window=\n");
        for( j = 0; j < dim_i[0]; j++) {
            printf("%.8f", fft_window[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }        
        
        printf("\nspe c librosa_stft_data, y_i=\n");
        for( j = 0; j < y_row_i; j++) {
            printf("%.8f", y_i[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }
        
        printf("\nspe c librosa_stft_data, y_pad=\n");
        for( j = 0; j < dim_i[1]; j++) {
            printf("%.8f", y_pad[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }
        printf("\nspe c librosa_stft_data, y_frames=\n");
        for( i = 0; i < dim_i[2]; i++) {
            for( j = 0; j < dim_i[3]; j++) {
                printf("%.8f", y_frames[i][j]); printf(", ");
                if (j%4==3) {
                    printf("\n");
                }
            }
        }
    #endif*/


    //# Pre-allocate the STFT matrix
    //stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
    //                       dtype=dtype,
    //                       order='F')
    //fft = get_fftlib()
    //# how many columns can we fit within MAX_MEM_BLOCK?
    //n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] *
    //                                   stft_matrix.itemsize)
    //n_columns = max(n_columns, 1)
    //for bl_s in range(0, stft_matrix.shape[1], n_columns):
    //    bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
    //    stft_matrix[:, bl_s:bl_t] = fft.rfft(fft_window *
    //                                         y_frames[:, bl_s:bl_t],
    //                                         axis=0)
    //return stft_matrix
    //int librosa_util_MAX_MEM_BLOCK = pow(2, 18);
    //int stft_matrix_itemsize = 8; //dtype=np.complex64
    //int n_columns = librosa_util_MAX_MEM_BLOCK/(dim_stft_matrix_row_o*stft_matrix_itemsize);
    //int fftw3_N = n_fft_i;
    double *fftw3_1d_rfft_data_in = create_1d_double_array(dim_i[6]);
    //double complex *fftw3_1d_rfft_data_out = create_1d_double_complex_array(dim_i[7]);
    #ifdef for_python_or_c
        double complex *fftw3_1d_rfft_data_out = create_1d_double_complex_array(dim_i[7]);
        fftw_plan fftw3_plan_1d_rfft = fftw_plan_dft_r2c_1d(dim_i[6], fftw3_1d_rfft_data_in, fftw3_1d_rfft_data_out, FFTW_ESTIMATE);
    
    #else //C++
        double _Complex *fftw3_1d_rfft_data_out = create_1d_double_complex_array(dim_i[7]);
        //fftw_complex fftw3_1d_rfft_data_out[dim_i[7]];
        //complex<double> *fftw3_1d_rfft_data_out = create_1d_double_complex_array(dim_i[7]);
        fftw_plan fftw3_plan_1d_rfft = fftw_plan_dft_r2c_1d(dim_i[6], fftw3_1d_rfft_data_in, reinterpret_cast<fftw_complex*>(fftw3_1d_rfft_data_out), FFTW_ESTIMATE);
    #endif
    for(j = 0; j < dim_i[5]; j++) {
        for(i = 0; i < dim_i[2]; i++) {
            fftw3_1d_rfft_data_in[i] = fft_window[i]*y_frames[i][j];
        }
        /*#ifdef DEBUG_librosa
            printf("\nc librosa_stft_data, fftw3_1d_rfft_data_in=\n");
            for( i = 0; i < dim_i[2]; i++) {
                printf("%.8f", fftw3_1d_rfft_data_in[i]); printf(", ");
                if (i%4==3) {
                    printf("\n");
                }
            }
        #endif*/
        fftw_execute(fftw3_plan_1d_rfft);
        /*#ifdef DEBUG_librosa
            printf("\nc librosa_stft_data, fftw3_1d_rfft_data_out=\n");
            for( i = 0; i < dim_i[2]; i++) {
                printf("%.14f%+.14fj", creal(fftw3_1d_rfft_data_out[i]), cimag(fftw3_1d_rfft_data_out[i]));
                printf(", ");
                if (i%2==1) {
                    printf("\n");
                }
            }
        #endif*/
        for(i = 0; i < dim_i[4]; i++) {
            stft_matrix_o[i][j] = fftw3_1d_rfft_data_out[i];
            //creal(stft_matrix_o[i][j]) = fftw3_1d_rfft_data_out[i][0];
            //cimag(stft_matrix_o[i][j]) = fftw3_1d_rfft_data_out[i][1];
        }
    }   
    
    free(fft_window);
    free(y_pad);
    free_2d_double_array(y_frames, dim_i[2]);
    fftw_destroy_plan(fftw3_plan_1d_rfft);
    free(fftw3_1d_rfft_data_in);
    free(fftw3_1d_rfft_data_out);
    fftw_cleanup(); //[lc add]
    
}

/*void librosa_stft(   double *y_i
                    ,int y_row_i //63488
                    ,int n_fft_i
                    ,int hop_length_i
                    ,int win_length_i //set 0
                    ,int dim_stft_matrix_row_o
                    ,int dim_stft_matrix_col_o
                    ,double complex **stft_matrix_o //
            )
{
    //librosa/core/spectrum.py
    //@cache(level=20)
    //def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
    //         center=True, dtype=np.complex64, pad_mode='reflect'):
    //# By default, use the entire frame
    //if win_length is None:
    //    win_length = n_fft
    int win_length;

    if (win_length_i==0) {
        win_length = n_fft_i;
    }


    //# Set the default hop, if it's not already specified
    //if hop_length is None:
    //    hop_length = int(win_length // 4)
    int hop_length = hop_length_i;

    if (hop_length_i==0) {
        hop_length = win_length/4;
    }


    //fft_window = get_window(window, win_length, fftbins=True)
    double *fft_window = create_1d_double_array(win_length);
    librosa_filters_get_window(       win_length //nx_i
                                    ,&fft_window[0] //win_o
                                    );


    //# Pad the window out to n_fft size
    //fft_window = util.pad_center(fft_window, n_fft)
    //# Reshape so that the window can be broadcast
    //fft_window = fft_window.reshape((-1, 1))
    //# Check audio is valid
    //util.valid_audio(y)
    //# Pad the time series so that frames are centered
    //if center:
    //    y = np.pad(y, int(n_fft // 2), mode=pad_mode)
    double *y_pad = create_1d_double_array(y_row_i+n_fft_i);
    np_pad_reflect_1d_double(y_i ,y_row_i ,n_fft_i/2 ,&y_pad[0]);


    //# Window the time series.
    //y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)
    int frame_length=n_fft_i;
    int dim_y_frame_row = frame_length;
    int dim_y_frame_col = (y_row_i - frame_length)/hop_length + 1;
    double **y_frames = create_2d_double_array(dim_y_frame_row, dim_y_frame_col);
    librosa_util_frame(  y_pad
                        //,int frame_length_i
                        ,hop_length
                        ,dim_y_frame_row
                        ,dim_y_frame_col
                        ,&y_frames[0]
                        );


    //# Pre-allocate the STFT matrix
    //stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
    //                       dtype=dtype,
    //                       order='F')
    //fft = get_fftlib()
    //# how many columns can we fit within MAX_MEM_BLOCK?
    //n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] *
    //                                   stft_matrix.itemsize)
    //n_columns = max(n_columns, 1)
    //for bl_s in range(0, stft_matrix.shape[1], n_columns):
    //    bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
    //    stft_matrix[:, bl_s:bl_t] = fft.rfft(fft_window *
    //                                         y_frames[:, bl_s:bl_t],
    //                                         axis=0)
    //return stft_matrix
    //int librosa_util_MAX_MEM_BLOCK = pow(2, 18);
    int dim_stft_matrix_row_o = 1 + n_fft_i/2;
    int dim_stft_matrix_col_o = dim_y_frame_col;
    //int stft_matrix_itemsize = 8; //dtype=np.complex64
    //int n_columns = librosa_util_MAX_MEM_BLOCK/(dim_stft_matrix_row_o*stft_matrix_itemsize);
    int fftw3_N = n_fft_i;
    double *fftw3_1d_rfft_data_in = create_1d_double_array(fftw3_N);
    double complex *fftw3_1d_rfft_data_out = create_1d_double_complex_array(dim_stft_matrix_row_o);
    fftw_plan fftw3_plan_1d_rfft = fftw_plan_dft_r2c_1d(fftw3_N, fftw3_data_in, fftw3_data_out, FFTW_ESTIMATE);
    for(j = 0; j < dim_stft_matrix_col_o; j++) {
        for(i = 0; i < dim_y_frame_row; i++) {
            fftw3_1d_rfft_data_in[i] = fft_window[i]*y_frames[i][j];
        }
        fftw_execute(fftw3_plan_1d_rfft);
        for(i = 0; i < dim_stft_matrix_row_o; i++) {
            stft_matrix[i][j] = fftw3_data_out[i];
        }
    }
    
    free(fft_window);
    free(y_pad);
    free_2d_double_array(y_frames, dim_y_frame_row);
    fftw_destroy_plan(fftw3_plan_1d_rfft);
    free(fftw3_1d_rfft_data_in);
    free(fftw3_1d_rfft_data_out);
    
}*/

void librosa_filters_get_window(     int nx_i
                                    ,double* win_o
                                    )
{
    //librosa/filters.py
    //@cache(level=10)
    //def get_window(window, Nx, fftbins=True):
    //if callable(window):
    //    return window(Nx)
    //elif (isinstance(window, (str, tuple)) or np.isscalar(window)):
    //    # TODO: if we add custom window functions in librosa, call them here
    //    return scipy.signal.get_window(window, Nx, fftbins=fftbins)
    scipy_signal_get_window_hann_double(nx_i ,&win_o[0]);


    //elif isinstance(window, (np.ndarray, list)):
    //    if len(window) == Nx:
    //        return np.asarray(window)
    //    raise ParameterError('Window size mismatch: '
    //                         '{:d} != {:d}'.format(len(window), Nx))
    //else:
    //    raise ParameterError('Invalid window specification: {}'.format(window))
    
}

void librosa_util_frame( double* x_i
                        //,int frame_length_i
                        ,int hop_length_i
                        ,int dim_frame_row_i
                        ,int dim_frame_col_i
                        ,double** frame_o
                        )
{
    //librosa/util.py
    //def frame(x, frame_length=2048, hop_length=512, axis=-1):
    //if not isinstance(x, np.ndarray):
    //    raise ParameterError('Input must be of type numpy.ndarray, '
    //                         'given type(x)={}'.format(type(x)))
    //if x.shape[axis] < frame_length:
    //    raise ParameterError('Input is too short (n={:d})'
    //                         ' for frame_length={:d}'.format(x.shape[axis], frame_length))
    //if hop_length < 1:
    //    raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))
    //n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    //strides = np.asarray(x.strides)
    //new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize
    //if axis == -1:
    //    if not x.flags['F_CONTIGUOUS']:
    //        raise ParameterError('Input array must be F-contiguous '
    //                             'for framing along axis={}'.format(axis))
    //    shape = list(x.shape)[:-1] + [frame_length, n_frames]
    //    strides = list(strides) + [hop_length * new_stride]
    //elif axis == 0:
    //    if not x.flags['C_CONTIGUOUS']:
    //        raise ParameterError('Input array must be C-contiguous '
    //                             'for framing along axis={}'.format(axis))
    //    shape = [n_frames, frame_length] + list(x.shape)[1:]
    //    strides = [hop_length * new_stride] + list(strides)
    //else:
    //    raise ParameterError('Frame axis={} must be either 0 or -1'.format(axis))
    //return as_strided(x, shape=shape, strides=strides)
    int i, j;
    
    for(j = 0; j < dim_frame_col_i; j++) {
        for(i = 0; i < dim_frame_row_i; i++) {
            frame_o[i][j] = x_i[i+j*hop_length_i];
        }
    }
    
}

void librosa_filters_mel(    double sr_i
                            ,int n_fft_i
                            ,int n_mels_i
                            ,double fmin_i //set 0.0
                            ,double fmax_i //set 0.0
                            ,double **weights_o
                        )
{
    //librosa/filters.py
    //@cache(level=10)
    //def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
    //        norm='slaney', dtype=np.float32):
    //    if fmax is None:
    //        fmax = float(sr) / 2
    int i, j;
    //double fmin=0.0;
    double fmax = fmax_i;

    if (fmax_i==0.0) {
        fmax = sr_i/2.0;
    }
    
    
    //    if norm == 1:
    //        warnings.warn('norm=1 behavior will change in librosa 0.8.0. '
    //                      "To maintain forward compatibility, use norm='slaney' instead.",
    //                      FutureWarning)
    //    elif norm == np.inf:
    //        warnings.warn('norm=np.inf behavior will change in librosa 0.8.0. '
    //                      "To maintain forward compatibility, use norm=None instead.",
    //                      FutureWarning)
    //    elif norm not in (None, 1, 'slaney', np.inf):
    //        raise ParameterError("Unsupported norm={}, must be one of: None, 1, 'slaney', np.inf".format(repr(norm)))
    //    # Initialize the weights
    //    n_mels = int(n_mels)
    //    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    //    # Center freqs of each FFT bin
    //    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    double *fftfreqs = librosa_fft_frequencies(sr_i ,n_fft_i);


    //    # 'Center freqs' of mel bands - uniformly spaced between limits
    //    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    double *mel_f = librosa_mel_frequencies( n_mels_i+2
                                            ,fmin_i
                                            ,fmax
                                            ,0
                                            );


    //    fdiff = np.diff(mel_f)
    double *fdiff = np_diff_1d_double(mel_f ,n_mels_i+2);


    //    ramps = np.subtract.outer(mel_f, fftfreqs)
    double **ramps = np_subtract_outer_1d_double_in(mel_f ,n_mels_i+2 ,fftfreqs ,1 + n_fft_i/2);


    //    for i in range(n_mels):
    //        # lower and upper slopes for all bins
    //        lower = -ramps[i] / fdiff[i]
    //        upper = ramps[i+2] / fdiff[i+1]
    //        # .. then intersect them with each other and zero
    //        weights[i] = np.maximum(0, np.minimum(lower, upper))
    //double **weights = create_2d_double_array(n_mels_i, 1 + n_fft_i/2);
    double *lower = create_1d_double_array(1 + n_fft_i/2);
    double *upper = create_1d_double_array(1 + n_fft_i/2);

    for (i=0; i<n_mels_i; i++) {
        for (j=0; j<1 + n_fft_i/2; j++) {
            lower[j] = -1.0*ramps[i][j] / fdiff[i];
            upper[j] = ramps[i+2][j] / fdiff[i+1];
            weights_o[i][j] = lower[j];

            if (lower[j] > upper[j]) {
                weights_o[i][j] = upper[j];
            }

            if (weights_o[i][j] < 0.0) {
                weights_o[i][j] = 0.0;
            }
        }
    }
    
    
    //    if norm in (1, 'slaney'):
    //        # Slaney-style mel is scaled to be approx constant energy per channel
    //        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
    //        weights *= enorm[:, np.newaxis]
    double *enorm = create_1d_double_array(n_mels_i);
    for (j=0; j<1 + n_fft_i/2; j++) {
        for (i=0; i<n_mels_i; i++) {
            enorm[i] = 2.0 / (mel_f[i+2] - mel_f[i]);
            weights_o[i][j] *= enorm[i];
        }
    }
    
    
    //    # Only check weights if f_mel[0] is positive
    //    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
    //        # This means we have an empty channel somewhere
    //        warnings.warn('Empty filters detected in mel frequency basis. '
    //                      'Some channels will produce empty responses. '
    //                      'Try increasing your sampling rate (and fmax) or '
    //                      'reducing n_mels.')
    //    return weights

    free(fftfreqs);
    free(mel_f);
    free(fdiff);
    free_2d_double_array(ramps, n_mels_i+2);
    free(lower);
    free(upper);
    free(enorm);
    
}

double *librosa_fft_frequencies( double sr_i
                                ,int n_fft_i
                            )
{
    //librosa/core/time_frequencies.py
    //def fft_frequencies(sr=22050, n_fft=2048):
    //return np.linspace(0,
    //                   float(sr) / 2,
    //                   int(1 + n_fft//2),
    //                   endpoint=True)
    return np_linspace_double(0.0 ,sr_i/2.0 ,1 + n_fft_i/2);

}

double *librosa_mel_frequencies(     int n_mels_i
                                    ,double fmin_i
                                    ,double fmax_i
                                    ,int htk_i //set 0
                                )
{
    //librosa/core/time_frequencies.py
    //def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    //    # 'Center freqs' of mel bands - uniformly spaced between limits
    //    min_mel = hz_to_mel(fmin, htk=htk)
    //    max_mel = hz_to_mel(fmax, htk=htk)
    //    mels = np.linspace(min_mel, max_mel, n_mels)
    //    return mel_to_hz(mels, htk=htk)
    double min_mel = librosa_hz_to_mel(fmin_i ,htk_i);
    double max_mel = librosa_hz_to_mel(fmax_i ,htk_i);
    double *mels = np_linspace_double(min_mel ,max_mel ,n_mels_i);
    //[lc del] return librosa_mel_to_hz_1d_double(mels ,n_mels_i ,0);
    double *mel_f = librosa_mel_to_hz_1d_double(mels ,n_mels_i ,0);
    free(mels);
    return mel_f;

}

double librosa_hz_to_mel(    double frequencies_i
                            ,int htk_i //set 0
                        )
{
    //librosa/core/time_frequencies.py
    //def hz_to_mel(frequencies, htk=False):
    //    frequencies = np.asanyarray(frequencies)
    //    if htk:
    //        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    //    # Fill in the linear part
    //    f_min = 0.0
    //    f_sp = 200.0 / 3
    //    mels = (frequencies - f_min) / f_sp
    //    # Fill in the log-scale part
    //    min_log_hz = 1000.0                         # beginning of log region (Hz)
    //    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    //    logstep = np.log(6.4) / 27.0                # step size for log region
    //    if frequencies.ndim:
    //        # If we have array data, vectorize
    //        log_t = (frequencies >= min_log_hz)
    //        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    //    elif frequencies >= min_log_hz:
    //        # If we have scalar data, heck directly
    //        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    //    return mels
    if (htk_i==1) {
        return 2595.0 * log10(1.0 + frequencies_i / 700.0);
    } else {
        double f_min = 0.0;
        double f_sp = 200.0 / 3.0;
        //double mels_o = (frequencies_i - f_min) / f_sp;
        double min_log_hz = 1000.0; //# beginning of log region (Hz)
        double min_log_mel = (min_log_hz - f_min) / f_sp; //# same (Mels)
        double logstep = log(6.4) / 27.0;

        if (frequencies_i >= min_log_hz) {
            return min_log_mel + log(frequencies_i / min_log_hz) / logstep;
        } else {
            return (frequencies_i - f_min) / f_sp;
        }
    }

}

double *librosa_mel_to_hz_1d_double( double *mels_i
                                    ,int mels_row_i
                                    ,int htk_i //set 0
                                    )
{
    //librosa/core/time_frequencies.py
    //def mel_to_hz(mels, htk=False):
    //    mels = np.asanyarray(mels)
    //    if htk:
    //        return 700.0 * (10.0**(mels / 2595.0) - 1.0)
    //    # Fill in the linear scale
    //    f_min = 0.0
    //    f_sp = 200.0 / 3
    //    freqs = f_min + f_sp * mels
    //    # And now the nonlinear scale
    //    min_log_hz = 1000.0                         # beginning of log region (Hz)
    //    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    //    logstep = np.log(6.4) / 27.0                # step size for log region
    //    if mels.ndim:
    //        # If we have vector data, vectorize
    //        log_t = (mels >= min_log_mel)
    //        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    //    elif mels >= min_log_mel:
    //        # If we have scalar data, check directly
    //        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    //    return freqs
    int i;
    double *freqs = create_1d_double_array(mels_row_i);

    if (htk_i==1) {
        for(i = 0; i < mels_row_i; i++) {        
            freqs[i] = 700.0*(pow(10.0, mels_i[i]/2595.0) - 1.0);
        }
    } else {
        double f_min = 0.0;
        double f_sp = 200.0 / 3.0;
        //freqs_o = f_min + f_sp * mels_i;
        double min_log_hz = 1000.0; //# beginning of log region (Hz)
        double min_log_mel = (min_log_hz - f_min) / f_sp; //# same (Mels)
        double logstep = log(6.4) / 27.0;

        for(i = 0; i < mels_row_i; i++) {        
            if (mels_i[i] >= min_log_mel) {
                freqs[i] = min_log_hz * exp(logstep * (mels_i[i] - min_log_mel));
            } else {
                freqs[i] = f_min + f_sp * mels_i[i];
            }
        }
    }

    return freqs;
}

void librosa_power_to_db_np_max_1d_double(   double *s_i
                                            ,int s_row_i
                                            ,double amin_i
                                            ,double top_db_i
                                            ,double *log_spec_o
                                            ) //ref=np.max
{
    //librosa/core/spectrum.py
    //@cache(level=30)
    //def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    //    S = np.asarray(S)
    //    if amin <= 0:
    //        raise ParameterError('amin must be strictly positive')
    //    if np.issubdtype(S.dtype, np.complexfloating):
    //        warnings.warn('power_to_db was called on complex input so phase '
    //                      'information will be discarded. To suppress this warning, '
    //                      'call power_to_db(np.abs(D)**2) instead.')
    //        magnitude = np.abs(S)
    //    else:
    //        magnitude = S
    //    if callable(ref):
    //        # User supplied a function to calculate reference power
    //        ref_value = ref(magnitude)
    //    else:
    //        ref_value = np.abs(ref)
    //    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    //    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    //    if top_db is not None:
    //        if top_db < 0:
    //            raise ParameterError('top_db must be non-negative')
    //        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    //    return log_spec
    int i;
    double ref_value = s_i[0];
    for (i=0; i<s_row_i; i++) {
        if (s_i[i] > ref_value) {
            ref_value = s_i[i];
        }
    }

    if (amin_i > ref_value) {
        ref_value = amin_i;
    }

    ref_value = 10.0 * log10(ref_value);

    //double *log_spec = create_and_copy_1d_double_array(s_row_i, s_i);
    for (i=0; i<s_row_i; i++) {
        if (log_spec_o[i] < amin_i) {
            log_spec_o[i] = amin_i;
        }

        log_spec_o[i] = 10.0 * log10(log_spec_o[i]) - ref_value;
    }

    if (top_db_i != 0.0) {
        double log_spec_o_max = log_spec_o[0];

        for (i=0; i<s_row_i; i++) {
            if (log_spec_o[i] > log_spec_o_max) {
                log_spec_o_max = log_spec_o[i];
            }
        }

        log_spec_o_max = log_spec_o_max - top_db_i;

        for (i=0; i<s_row_i; i++) {
            if (log_spec_o[i] < log_spec_o_max) {
                log_spec_o[i] = log_spec_o_max;
            }
        }
    }

}

void librosa_power_to_db_np_max_2d_double(   double **s_i
                                            ,int s_row_i
                                            ,int s_col_i
                                            ,double amin_i
                                            ,double top_db_i
                                            ,double **log_spec_o
                                            ) //ref=np.max
{
    //librosa/core/spectrum.py
    //@cache(level=30)
    //def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    //    S = np.asarray(S)
    //    if amin <= 0:
    //        raise ParameterError('amin must be strictly positive')
    //    if np.issubdtype(S.dtype, np.complexfloating):
    //        warnings.warn('power_to_db was called on complex input so phase '
    //                      'information will be discarded. To suppress this warning, '
    //                      'call power_to_db(np.abs(D)**2) instead.')
    //        magnitude = np.abs(S)
    //    else:
    //        magnitude = S
    //    if callable(ref):
    //        # User supplied a function to calculate reference power
    //        ref_value = ref(magnitude)
    //    else:
    //        ref_value = np.abs(ref)
    //    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    //    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    //    if top_db is not None:
    //        if top_db < 0:
    //            raise ParameterError('top_db must be non-negative')
    //        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    //    return log_spec
    int i, j;
    double ref_value = s_i[0][0];
    for (i=0; i<s_row_i; i++) {
        for (j=0; j<s_col_i; j++) {
            if (s_i[i][j] > ref_value) {
                ref_value = s_i[i][j];
            }
        }
    }

    if (amin_i > ref_value) {
        ref_value = amin_i;
    }

    ref_value = 10.0 * log10(ref_value);

    //double **log_spec = create_and_copy_2d_double_array(s_row_i, s_col_i, s_i);
    for (i=0; i<s_row_i; i++) {
        for (j=0; j<s_col_i; j++) {
            if (log_spec_o[i][j] < amin_i) {
                log_spec_o[i][j] = amin_i;
            }

            log_spec_o[i][j] = 10.0 * log10(log_spec_o[i][j]) - ref_value;

        }
    }

    if (top_db_i != 0.0) {
        double log_spec_o_max = log_spec_o[0][0];

        for (i=0; i<s_row_i; i++) {
            for (j=0; j<s_col_i; j++) {
                if (log_spec_o[i][j] > log_spec_o_max) {
                    log_spec_o_max = log_spec_o[i][j];
                }
            }
        }

        log_spec_o_max = log_spec_o_max - top_db_i;

        for (i=0; i<s_row_i; i++) {
            for (j=0; j<s_col_i; j++) {
                if (log_spec_o[i][j] < log_spec_o_max) {
                    log_spec_o[i][j] = log_spec_o_max;
                }
            }
        }
    }
}

