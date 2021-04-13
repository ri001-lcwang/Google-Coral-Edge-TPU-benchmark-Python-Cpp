/*#include <stdio.h>
#include <stdlib.h>
#define PI acos(-1)*/
#include "sub_ulib_realtime_pso_c0.h"

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


//for C++
int get_direction_beamforming(   double **MicPos_i //[3, DEF_NUM_OF_MIC]
                                ,float **buf_i //[DEF_NUM_OF_MIC, DEF_PROCESS_FRAME] //DEF_NUM_OF_MIC*(DEF_PROCESS_FRAME*DEF_CHUNK) or DEF_NUM_OF_MIC*DEF_SOR_LEN
                                ,double *k_TIKR //[DEF_HOPSIZE]
                                ,float bata_i
                                ,int flag_create_fftw3_plan_1d_fft_i
                                ,int flag_destroy_fftw3_plan_1d_fft_i
                                ,int flag_create_fftw3_plan_1d_ifft
                                ,int flag_destroy_fftw3_plan_1d_ifft
                                //,double flat_solution_o[DEF_NUM_OF_MIC*2] //max: DEF_NUM_OF_MIC*2, original: SorNum_guess*2
                                ,double *ang //[4]
                                ,double **p_o //[2, (DEF_PROCESS_FRAME*DEF_CHUNK)] //SorNum_guess*(DEF_PROCESS_FRAME*DEF_CHUNK)
                                )
{
    int i, j;
    int SorNum_guess_o = 0;
    //double **MicPos_i = deflate_2d_double_array(flat_MicPos_i, 3, DEF_NUM_OF_MIC);
    //double **buf_i = deflate_2d_double_array(flat_buf_i, DEF_NUM_OF_MIC, DEF_SOR_LEN);
    
    double **solution_o = create_2d_double_array(2, 2);
    double _Complex ***P_half_o = create_3d_double_complex_array(DEF_NUM_OF_MIC, DEF_NWIN/2, DEF_NUM_OF_FRAME_CONST);
    //double *k_o = create_1d_double_array(DEF_NFFT/2);
    
   // double **p_o = create_2d_double_array(2, DEF_PROCESS_FRAME*DEF_CHUNK);
    
    
    //printf("\nc DEF_NUM_OF_MIC*DEF_SOR_LEN= %d", DEF_NUM_OF_MIC*DEF_SOR_LEN);
    //printf("\nc DEF_NUM_OF_FRAME_CONST= %d\n", DEF_NUM_OF_FRAME_CONST);
    #ifdef measure_time
        clock_t time0, time1; 
        time0 = clock();
    #endif 
    
    //printf("\ng0");
    SorNum_guess_o = get_direction(  MicPos_i
                                    ,buf_i
                                    ,flag_create_fftw3_plan_1d_fft_i
                                    ,flag_destroy_fftw3_plan_1d_fft_i
                                    ,&solution_o[0]
                                    //,SorNum_guess_o
                                    ,&P_half_o[0]
                                    //,&k_o[0]
                                    ); 
    //printf("\ng1");
    
    /*printf("\nspe c P_half=\n");
    int k;
    for(i = 0; i < DEF_NUM_OF_MIC; i++) { 
        for( j = 0; j < DEF_NWIN/2; j++) {
            for(k = 0; k < DEF_NUM_OF_FRAME_CONST; k++) {
                printf("%f + i*%f", creal(P_half_o[i][j][k]), cimag(P_half_o[i][j][k])); printf(", ");
                if (k%2==1) {
                    printf("\n");
                }
            }
        }
    }*/
    
    beamforming        ( P_half_o 
                        ,solution_o 
                        ,SorNum_guess_o
                        ,MicPos_i 
                        ,k_TIKR
                        ,DEF_BATA
                        ,flag_create_fftw3_plan_1d_ifft
                        ,flag_destroy_fftw3_plan_1d_ifft
                        ,&p_o[0]
                        );
    //printf("\nb0");
    
    #ifdef measure_time
        time1 = clock();    
        printf("\nspe c get_direction time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif 
    /* //methon 1
    double *cost_a = flatten_2d_double_array(cost_o, DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    for (i = 0; i < DEF_NUM_OF_MIC; i++) {
        for (j = 0; j < DEF_NUM_OF_MIC; j++) {
            cost[i*DEF_NUM_OF_MIC + j]=cost_a[i*DEF_NUM_OF_MIC + j];
        }
    } */
    //methon 2
    //for (i = 0; i < DEF_NUM_OF_MIC; i++) {
   // printf("\na spe c solution angle= ");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            ang[i*2 + j]=solution_o[i][j];
           // printf("%.8lf", ang[i]);
        }
    }
    //
    ////for (i = 0; i < SorNum_guess; i++) {
    //for (i = 0; i < 2; i++) {
    //    for (j = 0; j < DEF_PROCESS_FRAME*DEF_CHUNK; j++) {
    //        flat_p_o[i*DEF_PROCESS_FRAME*DEF_CHUNK + j] = p_o[i][j];
    //    }
    //}
    //printf("\nspe c a0");
    //free_2d_double_array(MicPos_i, 3);
    //printf("\nspe c a1");
    //free_2d_double_array(buf_i, DEF_NUM_OF_MIC);
    //printf("\nspe c a2");
    free_2d_double_array(solution_o, 2); //DEF_NUM_OF_MIC);
    //printf("\nspe c a3");
    free_3d_double_complex_array(P_half_o, DEF_NUM_OF_MIC, DEF_NWIN/2);
    //printf("\nspe c a4");
    //free(k_o);
    //printf("\nspe c a5");
   // free_2d_double_array(p_o, 2);
    //printf("\nspe c a6");
    
    return SorNum_guess_o;
}


/*
void create_fftw3_plan_1d_fft (double _Complex *data_in ,int N ,short setting, double _Complex *data_out)
{
	fftw_plan fftw3_plan_1d_fft;
    
	switch(setting) { 
        case 1: 
            fftw3_plan_1d_fft = fftw_plan_dft_1d(N, data_in, data_out, -1, FFTW_MEASURE);
            break; 
        default:
            fftw3_plan_1d_fft = fftw_plan_dft_1d(N, data_in, data_out, -1, FFTW_ESTIMATE);
    }	
}*/

void FFT( fftw_plan fftw3_plan_1d_fft
            ,double _Complex *fftw3_data_in
            ,double _Complex *fftw3_data_out
            //,float **p //DEF_NUM_OF_MIC*(DEF_PROCESS_FRAME*DEF_CHUNK) or DEF_NUM_OF_MIC*DEF_SOR_LEN
            ,double **p //[DEF_NUM_OF_MIC ,DEF_PROCESS_FRAME]
            ,double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
            ) 
{
    // source code:
    //def FFT(p):
    //    # Parameter
    //    
    //    SorLen=np.size(p,1)  # 取得訊號長度
    //    # Windowing
    //    hopsize=DEF_NWIN/2
    //    NumOfFrame_const=2*math.floor(SorLen/NWIN)-1
    //    win=np.hanning(NWIN)
    //#    win=win[:-1]
    //    win=win.T
    //    # FFT
    //    NFFT=NWIN
    //    P_half=np.zeros([DEF_NUM_OF_MIC,int(NFFT/2),NumOfFrame_const],dtype=complex) # 存放頻域的音檔
    //    # ~ timea = time.time()
    //    # ~ for FrameNo in range(0,NumOfFrame_const):
    //        # ~ t_start=FrameNo*hopsize
    //        # ~ tt=np.linspace(t_start,t_start+NWIN-1,NWIN)
    //        # ~ tt = tt.astype(int)
    //        # ~ for ss in range(0,DEF_NUM_OF_MIC):  # 第幾個麥克風
    //            # ~ #source_win=p[ss,tt:tt+NWIN]*win  # window
    //            # ~ source_win=p[ss,tt]*win  # window
    //            # ~ SOURCE=np.fft.fft(source_win)     # fft
    //            # ~ SOURCE_half = SOURCE[range(int(NFFT/2))]
    //            # ~ P_half[ss,:,FrameNo]=SOURCE_half
    //    # ~ timeb = time.time()
    //    
    //    for FrameNo in range(0,NumOfFrame_const):
    //        t_start=FrameNo*hopsize
    //        tt = int(t_start)
    //        for ss in range(0,DEF_NUM_OF_MIC):  # 第幾個麥克風
    //            source_win=p[ss,tt:tt+NWIN]*win  # window
    //            SOURCE=np.fft.fft(source_win)     # fft
    //            SOURCE_half = SOURCE[0:int(NFFT/2)]
    //            P_half[ss,:,FrameNo]=SOURCE_half
    // 
    //    return P_half
    
    //int NFFT_tmp=DEF_NWIN;
    int FrameNo, t_start=0, tt=0, ss, j;
    double *win = create_1d_double_array(DEF_NWIN);
   // double _Complex ***P_half = create_3d_double_complex_array(DEF_NUM_OF_MIC, DEF_NWIN/2, DEF_NUM_OF_FRAME_CONST);
    //double *source_win = create_1d_double_array(DEF_NWIN); //replace by fftw3_data_in
    //double _Complex *SOURCE = create_1d_double_complex_array(DEF_NWIN); //replace by fftw3_data_out
    
    //win = scipy_signal_hanning_double(NWIN, 0);
    np_hanning_double(DEF_NWIN, win);
    #ifdef DEBUG_fft
        printf("\nc fft, win=");
        for( j = 0; j < DEF_NWIN; j++) {
            printf("%.8f", win[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }
    #endif
    
    for(FrameNo = 0; FrameNo < DEF_NUM_OF_FRAME_CONST; FrameNo++) {
        t_start = FrameNo*DEF_HOPSIZE;
        tt = t_start;
        /*#ifdef DEBUG_fft
            printf("\nc fft, FrameNo= %d", FrameNo);
        #endif*/
        for(ss = 0; ss < DEF_NUM_OF_MIC; ss++) {  //# 第幾個麥克風
            for( j = 0; j < DEF_NWIN; j++) {
                //source_win[j] = p[ss][tt+j]*win[j];
                fftw3_data_in[j] = p[ss][tt+j]*win[j];
            }
            fftw_execute(fftw3_plan_1d_fft);
            for( j = 0; j < DEF_NWIN/2; j++) {
                P_half[ss][j][FrameNo] = fftw3_data_out[j];
            }
            /*#ifdef DEBUG_fft
                printf("\nc fft, ss= %d", ss);
                printf("\nc fft, SOURCE= ");
                for( j = 0; j < DEF_NWIN; j++) {
                    printf("%.14f% + i*.14f", creal(fftw3_data_out[j]), cimag(fftw3_data_out[j])); printf(", ");
                    if (j%2==1) {
                        printf("\n");
                    }
                }
                printf("\nc fft, P_half= ");
                for( j = 0; j < DEF_NWIN/2; j++) {
                    printf("%.14f% + i*.14f", creal(P_half[ss][j][FrameNo]), cimag(P_half[ss][j][FrameNo])); printf(", ");
                    if (j%2==1) {
                        printf("\n");
                    }
                }
            #endif*/
        }
    }
    
    free(win); 
       
    //return P_half;
}

int MUSIC_Parameter( double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                    //,int fs
                    ,double *k //DEF_NFFT/2
                    ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    //,double **eigenvalue_save //DEF_NUM_OF_FREQS*DEF_NUM_OF_MIC
                    //,int SorNum_guess
                    ) 
{
    // source code:
    //def MUSIC_Parameter(P_half,fs):
    //    # Parameter
    //    NumOfFreqs=np.size(P_half,1)
    //    NumOfFrame_const=np.size(P_half,2)
    //    NWIN=1024
    //    NFFT=NWIN
    //    df=fs/NFFT
    //    Freqs=np.linspace(0,(NFFT/2-1)*df,int(NFFT/2))
    //    c=343
    //    w=2*math.pi*Freqs
    //    k=w/c
    //    # Transform
    //    x=np.zeros([NumOfFreqs,DEF_NUM_OF_MIC,NumOfFrame_const],dtype=complex)
    //#     for FrameNo in range(0,NumOfFrame_const):
    //#         for ff in range(0,NumOfFreqs):
    //#             x[ff,:,FrameNo]=P_half[:,ff,FrameNo]
    //    x=P_half.transpose(1,0,2)
    //        
    //    eigenvalue_save=np.zeros([NumOfFreqs,DEF_NUM_OF_MIC])      # 不同頻率的特徵值
    //
    //    # Rxx
    //    PN=np.zeros([DEF_NUM_OF_MIC,DEF_NUM_OF_MIC,NumOfFreqs],dtype=complex)
    //    eigenvector_save=np.zeros([NumOfFreqs,DEF_NUM_OF_MIC,DEF_NUM_OF_MIC],dtype=complex)
    //        for ff in range(0,NumOfFreqs):
    //        x_1=x[ff,:,:]
    //        Rxx=np.dot(x_1,x_1.conj().T)/NumOfFrame_const
    //        [eigenvalue,eigenvector]=np.linalg.eig(Rxx)
    //        sort_eigenvalue=np.argsort(abs(eigenvalue)) # 特徵值由小到大排列
    //        eigenvalue_save[ff,:]=abs(eigenvalue[sort_eigenvalue])
    //        eigenvector_save[ff,:,:]=eigenvector[:,sort_eigenvalue]  
    //        
    //
    //    ##  Check source number(use slope)  
    //    sum_eigenvalue = np.sum(eigenvalue_save, axis = 0) # Sum the eigenvalue of every frame
    //    print("eigen_value sum = ",sum_eigenvalue)
    //    for mic_count in range(DEF_NUM_OF_MIC-1,-1,-1):
    //        slope_threshold = 200
    //        if sum_eigenvalue[mic_count]-sum_eigenvalue[mic_count-1]>=slope_threshold:
    //            continue
    //        else:
    //            break
    //    SorNum_guess=DEF_NUM_OF_MIC-mic_count-1
    //#     print('The source number is ',SorNum_guess)
    //        # 取得noise subspace
    //    for ff in range(0,NumOfFreqs):
    //        sort_eigenvector=eigenvector_save[ff,:,:]
    //        Us=sort_eigenvector[:,-SorNum_guess-1:] # signal subspace
    //        PN[:,:,ff]=np.eye(DEF_NUM_OF_MIC)-np.dot(Us,Us.conj().T)
    //    return k,PN,eigenvalue_save,SorNum_guess
    
    //int NFFT_tmp=DEF_NWIN;
    int ff, mic_count, i, j, m, n;
    int SorNum_guess = 111;
    double _Complex s=0;
    double df=DEF_FS/(DEF_NFFT*1.0);
    double *Freqs = np_linspace_double(0, ((DEF_NFFT*1.0)/2-1.0)*df, DEF_NFFT/2);
    //double *k = create_1d_double_array(DEF_NFFT/2);
    for(i = 0; i < DEF_NFFT/2; i++) {
        k[i] = 2*PI*Freqs[i]/DEF_C_SPEED;
    }
    //printf("\nc mp, (DEF_NFFT/2-1)=%14f", (DEF_NFFT*1.0)/2-1.0);
    //printf("\nc mp, (DEF_NFFT/2-1)*df=%14f", ((DEF_NFFT*1.0)/2-1.0)*df);
    /*#ifdef DEBUG_mp
        printf("\nc mp, Freqs=");
        for( j = 0; j < DEF_NFFT/2; j++) {
            printf("%.14f", Freqs[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }
        printf("\nc mp, k=");
        for( j = 0; j < DEF_NFFT/2; j++) {
            printf("%.14f", k[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }
    #endif*/
    double _Complex ***x = create_3d_double_complex_array(DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC, DEF_NUM_OF_FRAME_CONST); 
    double _Complex **Rxx = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC); 
    double _Complex *eigenvalue = create_1d_double_complex_array(DEF_NUM_OF_MIC); 
    double         *eigenvalue_abs = create_1d_double_array(DEF_NUM_OF_MIC); 
    double         *tmpd0 = create_1d_double_array(DEF_NUM_OF_MIC);
    double _Complex **eigenvector = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    double _Complex ***eigenvector_save = create_3d_double_complex_array(DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    int            *sort_eigenvalue = create_1d_int_array(DEF_NUM_OF_MIC);
    double         *sum_eigenvalue = create_1d_double_array(DEF_NUM_OF_MIC);
    double _Complex **tmp0 = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
     
    double         **eigenvalue_save = create_2d_double_array(DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC); 
    
    double         slope_threshold = 20000; //400; //200; //0.5;
    
    /*printf("\nc DEF_NUM_OF_FREQS= %d", DEF_NUM_OF_FREQS);
    printf("\nc DEF_NUM_OF_FRAME_CONST= %d", DEF_NUM_OF_FRAME_CONST);
    #ifdef DEBUG_mp
        printf("\nc mp, x= \n");
    #endif*/
    for(i = 0; i < DEF_NUM_OF_FREQS; i++) {
        for(j = 0; j < DEF_NUM_OF_MIC; j++) {
            for(m = 0; m < DEF_NUM_OF_FRAME_CONST; m++) {
                x[i][j][m] = P_half[j][i][m];
                /*#ifdef DEBUG_mp
                    printf("%.14f% + i*.14f", creal(x[i][j][m]), cimag(x[i][j][m])); printf(", ");
                    if ((m+1)%2==0) {
                        printf("\n");
                    }
                #endif*/
            }
        }
    }
    
    for(ff = 0; ff < DEF_NUM_OF_FREQS; ff++) {
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                s = 0;
                for(m = 0; m < DEF_NUM_OF_FRAME_CONST; m++) {
                    s += x[ff][i][m]*conj(x[ff][j][m])/(DEF_NUM_OF_FRAME_CONST*1.0); 
                }
                Rxx[i][j] = s;
            }
        }
        
        np_linalg_eig_square_double_complex_array(DEF_NUM_OF_MIC ,Rxx ,&eigenvalue[0] ,&eigenvector[0]);
        
        /*printf("\nspe c MUSIC_Parameter, ff= %d", ff);
        printf("\nspe c MUSIC_Parameter, eigenvalue= ");
        for (int i = 0; i < DEF_NUM_OF_MIC; i++) {
            printf("%.14f+%.14fi", creal(eigenvalue[i]), cimag(eigenvalue[i])); printf(", ");
        }   
        printf("\nspe c MUSIC_Parameter, eigenvector= ");
        for (int i = 0; i < DEF_NUM_OF_MIC; i++) {
            for (int j = 0; j < DEF_NUM_OF_MIC; j++) {
                printf("%.14f+%.14fi", creal(eigenvector[i][j]), cimag(eigenvector[i][j])); printf(", ");
                if (j%2==1) {
                    printf("\n");
                }
            }
        }*/
        
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            eigenvalue_abs[i] = cabs(eigenvalue[i]); //sqrt(creal(eigenvalue[i])*creal(eigenvalue[i]) + cimag(eigenvalue[i])*cimag(eigenvalue[i])) //#+ 1j*0
        }
        
        stable_sort_1d_double(DEF_NUM_OF_MIC ,eigenvalue_abs ,&sort_eigenvalue[0]); //sort_eigenvalue=np.argsort(np.asarray(eigenvalue_abs))
        //np_sort_argsort_1d_double_complex(eigenvalue_abs ,1 ,&tmpd0[0] ,&sort_eigenvalue[0]); //# 特徵值由小到大排列
        
        for(j = 0; j < DEF_NUM_OF_MIC; j++) {
            m = sort_eigenvalue[j];
            eigenvalue_save[ff][j] = eigenvalue_abs[m]; //tmpd0[j];
        }
        
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                m = sort_eigenvalue[j];
                eigenvector_save[ff][i][j] = eigenvector[i][m];
            }
        }
        #ifdef DEBUG_mp
            /*printf("\nc mp, ff= %d", ff);
            printf("\nc mp, Rxx= ");
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                    printf("%.14f% + i*.14f", creal(Rxx[i][j]), cimag(Rxx[i][j])); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            }
            printf("\nc mp, eigenvalue= \n");
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                printf("%.14f% + i*.14f", creal(eigenvalue[i]), cimag(eigenvalue[i])); printf(", ");
                if ((i+1)%2==0) {
                    printf("\n");
                }
            }
            printf("\nc mp, eigenvector= ");
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                    printf("%.14f% + i*.14f", creal(eigenvector[i][j]), cimag(eigenvector[i][j])); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            }
            printf("\nc mp, sort_eigenvalue= ");
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                printf("%d", sort_eigenvalue[i]); printf(", ");
                if ((i+1)%4==0) {
                    printf("\n");
                }
            }
            printf("\nc mp, eigenvalue_save= ");
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                printf("%e", eigenvalue_save[ff][i]); printf(", ");
                if ((i+1)%4==0) {
                    printf("\n");
                }
            }*/
            printf("\nc mp, eigenvector_save= ");
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                    printf("%.14f% + i*.14f", creal(eigenvector_save[ff][i][j]), cimag(eigenvector_save[ff][i][j])); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            }
        #endif
    }
    
    // ##  Check source number(use slope)
    for(i = 0; i < DEF_NUM_OF_MIC; i++) {
        for(j = 0; j < DEF_NUM_OF_FREQS; j++) {
            sum_eigenvalue[i] = sum_eigenvalue[i] + eigenvalue_save[j][i];            
        }
    }
    //#ifdef DEBUG_mp
        printf("\nc mp, sum_eigenvalue= \n");
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            printf("%.14f", sum_eigenvalue[i]); printf(", ");
            if ((i+1)%4==0) {
                printf("\n");
            }
        }
    //#endif
    
    for(mic_count = DEF_NUM_OF_MIC-1; mic_count > -1; mic_count--) {       
        if (sum_eigenvalue[mic_count]-sum_eigenvalue[mic_count-1] < slope_threshold) {
            SorNum_guess = DEF_NUM_OF_MIC-mic_count-1;
            break;
        }
    }
    #ifdef DEBUG_mp
        printf("\nc mp, SorNum_guess= %d", SorNum_guess);
    #endif
    
   //# print('The source number is ',SorNum_guess)
        
    //# 取得noise subspace
    n = DEF_NUM_OF_MIC-SorNum_guess-1;
    
    for(ff = 0; ff < DEF_NUM_OF_FREQS; ff++) {
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                s = 0;
                for(m = 0; m < SorNum_guess+1; m++) {
                    s += eigenvector_save[ff][i][m+n]*conj(eigenvector_save[ff][j][m+n]);
                }
                tmp0[i][j] = s;
            }
        }
                
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                if (i==j)
                    PN[i][j][ff] = 1.0-tmp0[i][j];
                else
                    PN[i][j][ff]  = 0-tmp0[i][j];
            }
        }
    }
    
    free(Freqs);
    free_3d_double_complex_array(x, DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC);
    free_2d_double_complex_array(Rxx, DEF_NUM_OF_MIC);
    free(eigenvalue); 
    free(eigenvalue_abs); 
    free(tmpd0);
    free_2d_double_complex_array(eigenvector, DEF_NUM_OF_MIC);
    free_3d_double_complex_array(eigenvector_save, DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC);
    free(sort_eigenvalue);
    free(sum_eigenvalue);
    free_2d_double_complex_array(tmp0, DEF_NUM_OF_MIC);
    free_2d_double_array(eigenvalue_save, DEF_NUM_OF_FREQS);
    
    return SorNum_guess;
}

void MUSIC_Parameter_dbg( double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                    //,int fs
                    ,double *k //DEF_NFFT/2
                    ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    //,double **eigenvalue_save //DEF_NUM_OF_FREQS*DEF_NUM_OF_MIC
                    ,int *SorNum_guess
                    ) 
{
    // source code:
    //def MUSIC_Parameter(P_half,fs):
    //    # Parameter
    //    NumOfFreqs=np.size(P_half,1)
    //    NumOfFrame_const=np.size(P_half,2)
    //    NWIN=1024
    //    NFFT=NWIN
    //    df=fs/NFFT
    //    Freqs=np.linspace(0,(NFFT/2-1)*df,int(NFFT/2))
    //    c=343
    //    w=2*math.pi*Freqs
    //    k=w/c
    //    # Transform
    //    x=np.zeros([NumOfFreqs,DEF_NUM_OF_MIC,NumOfFrame_const],dtype=complex)
    //#     for FrameNo in range(0,NumOfFrame_const):
    //#         for ff in range(0,NumOfFreqs):
    //#             x[ff,:,FrameNo]=P_half[:,ff,FrameNo]
    //    x=P_half.transpose(1,0,2)
    //        
    //    eigenvalue_save=np.zeros([NumOfFreqs,DEF_NUM_OF_MIC])      # 不同頻率的特徵值
    //
    //    # Rxx
    //    PN=np.zeros([DEF_NUM_OF_MIC,DEF_NUM_OF_MIC,NumOfFreqs],dtype=complex)
    //    eigenvector_save=np.zeros([NumOfFreqs,DEF_NUM_OF_MIC,DEF_NUM_OF_MIC],dtype=complex)
    //        for ff in range(0,NumOfFreqs):
    //        x_1=x[ff,:,:]
    //        Rxx=np.dot(x_1,x_1.conj().T)/NumOfFrame_const
    //        [eigenvalue,eigenvector]=np.linalg.eig(Rxx)
    //        sort_eigenvalue=np.argsort(abs(eigenvalue)) # 特徵值由小到大排列
    //        eigenvalue_save[ff,:]=abs(eigenvalue[sort_eigenvalue])
    //        eigenvector_save[ff,:,:]=eigenvector[:,sort_eigenvalue]  
    //        
    //
    //    ##  Check source number(use slope)  
    //    sum_eigenvalue = np.sum(eigenvalue_save, axis = 0) # Sum the eigenvalue of every frame
    //    print("eigen_value sum = ",sum_eigenvalue)
    //    for mic_count in range(DEF_NUM_OF_MIC-1,-1,-1):
    //        slope_threshold = 200
    //        if sum_eigenvalue[mic_count]-sum_eigenvalue[mic_count-1]>=slope_threshold:
    //            continue
    //        else:
    //            break
    //    SorNum_guess=DEF_NUM_OF_MIC-mic_count-1
    //#     print('The source number is ',SorNum_guess)
    //        # 取得noise subspace
    //    for ff in range(0,NumOfFreqs):
    //        sort_eigenvector=eigenvector_save[ff,:,:]
    //        Us=sort_eigenvector[:,-SorNum_guess-1:] # signal subspace
    //        PN[:,:,ff]=np.eye(DEF_NUM_OF_MIC)-np.dot(Us,Us.conj().T)
    //    return k,PN,eigenvalue_save,SorNum_guess
    
    //int NFFT_tmp=DEF_NWIN;
    int ff, mic_count, i, j, m, n;
  //  int SorNum_guess = 111;
    double _Complex s=0;
    double df=DEF_FS/(DEF_NFFT*1.0);
    double *Freqs = np_linspace_double(0, ((DEF_NFFT*1.0)/2-1.0)*df, DEF_NFFT/2);
    //double *k = create_1d_double_array(DEF_NFFT/2);
    for(i = 0; i < DEF_NFFT/2; i++) {
        k[i] = 2*PI*Freqs[i]/DEF_C_SPEED;
    }
    //printf("\nc mp, (DEF_NFFT/2-1)=%14f", (DEF_NFFT*1.0)/2-1.0);
    //printf("\nc mp, (DEF_NFFT/2-1)*df=%14f", ((DEF_NFFT*1.0)/2-1.0)*df);
    #ifdef DEBUG_mp
        printf("\nc mp, Freqs=");
        for( j = 0; j < DEF_NFFT/2; j++) {
            printf("%.14f", Freqs[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }
        printf("\nc mp, k=");
        for( j = 0; j < DEF_NFFT/2; j++) {
            printf("%.14f", k[j]); printf(", ");
            if (j%4==3) {
                printf("\n");
            }
        }
    #endif
    double _Complex ***x = create_3d_double_complex_array(DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC, DEF_NUM_OF_FRAME_CONST); 
    double _Complex **Rxx = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC); 
    double _Complex *eigenvalue = create_1d_double_complex_array(DEF_NUM_OF_MIC); 
    double         *eigenvalue_abs = create_1d_double_array(DEF_NUM_OF_MIC); 
    double         *tmpd0 = create_1d_double_array(DEF_NUM_OF_MIC);
    double _Complex **eigenvector = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    double _Complex ***eigenvector_save = create_3d_double_complex_array(DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    int            *sort_eigenvalue = create_1d_int_array(DEF_NUM_OF_MIC);
    double         *sum_eigenvalue = create_1d_double_array(DEF_NUM_OF_MIC);
    double _Complex **tmp0 = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
     
    double         **eigenvalue_save = create_2d_double_array(DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC); 
    
    double         slope_threshold = 200; //0.5;
    
    for(i = 0; i < DEF_NUM_OF_FREQS; i++) {
        for(j = 0; j < DEF_NUM_OF_MIC; j++) {
            for(m = 0; m < DEF_NUM_OF_FRAME_CONST; m++) {
                x[i][j][m] = P_half[j][i][m];
            }
        }
    }
    
    for(ff = 0; ff < DEF_NUM_OF_FREQS; ff++) {
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                s = 0;
                for(m = 0; m < DEF_NUM_OF_FRAME_CONST; m++) {
                    s += x[ff][i][m]*conj(x[ff][j][m])/DEF_NUM_OF_FRAME_CONST; 
                }
                Rxx[i][j] = s;
            }
        }
        
        np_linalg_eig_square_double_complex_array(DEF_NUM_OF_MIC ,Rxx ,&eigenvalue[0] ,&eigenvector[0]);
        
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            eigenvalue_abs[i] = cabs(eigenvalue[i]); //sqrt(creal(eigenvalue[i])*creal(eigenvalue[i]) + cimag(eigenvalue[i])*cimag(eigenvalue[i])) //#+ 1j*0
        }
        
        stable_sort_1d_double(DEF_NUM_OF_MIC ,eigenvalue_abs ,&sort_eigenvalue[0]); //sort_eigenvalue=np.argsort(np.asarray(eigenvalue_abs))
        //np_sort_argsort_1d_double_complex(eigenvalue_abs ,1 ,&tmpd0[0] ,&sort_eigenvalue[0]); //# 特徵值由小到大排列
        
        for(j = 0; j < DEF_NUM_OF_MIC; j++) {
            m = sort_eigenvalue[j];
            eigenvalue_save[ff][j] = eigenvalue_abs[m]; //tmpd0[j];
        }
        
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                m = sort_eigenvalue[j];
                eigenvector_save[ff][i][j] = eigenvector[i][m];
            }
        }
    }
    
    // ##  Check source number(use slope)
    for(i = 0; i < DEF_NUM_OF_MIC; i++) {
        for(j = 0; j < DEF_NUM_OF_FREQS; j++) {
            sum_eigenvalue[i] = sum_eigenvalue[i] + eigenvalue_save[j][i];
        }
    }
    
    for(mic_count = DEF_NUM_OF_MIC-1; mic_count > -1; mic_count--) {       
        if (sum_eigenvalue[mic_count]-sum_eigenvalue[mic_count-1] < slope_threshold) {
            SorNum_guess[0] = DEF_NUM_OF_MIC-mic_count-1;
            break;
        }
    }
    #ifdef DEBUG_mp
        printf("\nc mp, SorNum_guess= %d", SorNum_guess[0]);
    #endif
    
   //# print('The source number is ',SorNum_guess)
        
    //# 取得noise subspace
    n = DEF_NUM_OF_MIC-SorNum_guess[0]-1;
    
    for(ff = 0; ff < DEF_NUM_OF_FREQS; ff++) {
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                s = 0;
                for(m = 0; m < SorNum_guess[0]+1; m++) {
                    s += eigenvector_save[ff][i][m+n]*conj(eigenvector_save[ff][j][m+n]);
                }
                tmp0[i][j] = s;
            }
        }
                
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(j = 0; j < DEF_NUM_OF_MIC; j++) {
                if (i==j)
                    PN[i][j][ff] = 1.0-tmp0[i][j];
                else
                    PN[i][j][ff]  = 0-tmp0[i][j];
            }
        }
    }
    
    free(Freqs);
    free_3d_double_complex_array(x, DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC);
    free_2d_double_complex_array(Rxx, DEF_NUM_OF_MIC);
    free(eigenvalue); 
    free(eigenvalue_abs); 
    free(tmpd0);
    free_2d_double_complex_array(eigenvector, DEF_NUM_OF_MIC);
    free_3d_double_complex_array(eigenvector_save, DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC);
    free(sort_eigenvalue);
    free(sum_eigenvalue);
    free_2d_double_complex_array(tmp0, DEF_NUM_OF_MIC);
    free_2d_double_array(eigenvalue_save, DEF_NUM_OF_FREQS);
    
    //return SorNum_guess;
}

int get_direction(   double **MicPos //3*DEF_NUM_OF_MIC
                    ,float **buf //[DEF_NUM_OF_MIC, DEF_PROCESS_FRAME] or DEF_NUM_OF_MIC*DEF_SOR_LEN
                    ,int flag_create_fftw3_plan_1d_fft
                    ,int flag_destroy_fftw3_plan_1d_fft
                    ,double **solution // 2*2 //max: DEF_NUM_OF_MIC*2, original: SorNum_guess*2
                    //,int SorNum_guess
                    ,double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*NumOfFrame_const
                    //,double *k //DEF_NFFT/2
                    ) 
{
    // source code:
    //def get_direction(buf):
    //    
    //    p = np.zeros([DEF_NUM_OF_MIC,process_frame*DEF_CHUNK], dtype='float32')  # 放置讀入的時域音檔
    //    # Normalize(current int16 for range -32738~32768)
    //#     for Mic in range(0,DEF_NUM_OF_MIC):
    //#         p[Mic,:] = buf[Mic,:]/math.pow(2,15)   
    //    p[:,:] = buf[:,:]/math.pow(2,15) 
    //    ## FFT
    //    aaa = time.time()
    //    
    //    P_half = FFT(p)
    //    bbb = time.time()
    //    print("fft time = ",bbb - aaa)
    //    
    //    # Get MUSIC Parameter
    //    k, PN, eigenvalue_save, SorNum_guess = MUSIC_Parameter(P_half, fs)
    // 
    //     
    //    if SorNum_guess == 0:           # 判定為沒有聲源
    //        solution = 0;
    //    else:                         # 有聲源
    //        # PSO Localization
    //        
    //        swarm = PSO_Localization(MicPos,PN,k,SorNum_guess)
    //        
    //#        print (time.clock()-t0)
    //        # Find source position
    //        solution = np.zeros([SorNum_guess,2])   # 每個聲源的水平角與仰角
    //        sorted_swarm = swarm[np.lexsort(-swarm.T)]  # 按照fitness值大小排序(由大到小)
    //        solution[0,:] = sorted_swarm[0,:2]
    //        swarm_point = 1
    //        solution_point = 1
    //        for SorFind in range(1,len(swarm)):
    //            if solution_point >= SorNum_guess:
    //                break
    //            for check_solution in range(0,len(solution)):  # 檢查每個已選出的解
    //                repeat_flag = 0  
    //                if np.linalg.norm(sorted_swarm[SorFind,:2]-solution[check_solution,:])<=40: # 距離不夠遠
    //                    repeat_flag=1 
    //                    swarm_point+=1
    //                    break
    //            if repeat_flag==0:
    //                solution[solution_point,:]=sorted_swarm[SorFind,:2]
    //                solution_point+=1
    //                swarm_point+=1              
    //    return solution,SorNum_guess,P_half,k    # 返回水平角與仰角
    
    int i, j, n, SorFind, check_solution;
    int m = 3*DEF_PSO_DIMENSION+2;
    int repeat_flag = 0, solution_point = 1;
    //int swarm_point = 1;
    int SorNum_guess;
    double tmp0 = 0, tmp1 = 0;
    double **p = create_2d_double_array(DEF_NUM_OF_MIC, DEF_PROCESS_FRAME); //DEF_SOR_LEN);
    double _Complex ***PN = create_3d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC, DEF_NUM_OF_FREQS);
    double **swarm = create_2d_double_array(DEF_PSO_SWARMS, 3*DEF_PSO_DIMENSION+2);
    //double **eigenvalue_save = create_2d_double_array(DEF_NUM_OF_FREQS, DEF_NUM_OF_MIC);
    double **sorted_swarm = create_2d_double_array(DEF_PSO_SWARMS, 3*DEF_PSO_DIMENSION+2);
    double **swarm_t_tmp = create_2d_double_array(3*DEF_PSO_DIMENSION+2, DEF_PSO_SWARMS);
    int *swarm_sort_tmp = create_1d_int_array(DEF_PSO_SWARMS);
    double *k = create_1d_double_array((int)DEF_NFFT/2);
    
    // fftw3 plan
    int fftw3_N = DEF_NWIN;
//    short fftw3_setting = 0;
    //fftw_complex *fftw3_data_in;
    //fftw3_data_in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * fftw3_N); // = create_1d_double_complex_array(fftw3_N);
    //fftw_complex *fftw3_data_out;
    //fftw3_data_out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * fftw3_N); // = create_1d_double_complex_array(fftw3_N);
    double _Complex *fftw3_data_in = create_1d_double_complex_array(fftw3_N);
    double _Complex *fftw3_data_out = create_1d_double_complex_array(fftw3_N);
    
    #ifdef measure_time
        clock_t time0, time1;
        time0 = clock();
    #endif
    fftw_plan fftw3_plan_1d_fft = fftw_plan_dft_1d(  fftw3_N
                                                    ,reinterpret_cast<fftw_complex*>(fftw3_data_in)
                                                    ,reinterpret_cast<fftw_complex*>(fftw3_data_out)
                                                    ,-1
                                                    ,FFTW_ESTIMATE);
    #ifdef measure_time
        time1 = clock();
        printf("\nspe c create_fftw3_plan_1d_fft time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif
    
    /*if (flag_create_fftw3_plan_1d_fft == 1) {
	//	time0 = clock();    
        //create_fftw3_plan_1d_fft (fftw3_data_in ,fftw3_N ,fftw3_setting, fftw3_data_out);
        switch(fftw3_setting) { 
            case 1: 
                fftw3_plan_1d_fft = fftw_plan_dft_1d(fftw3_N, fftw3_data_in, fftw3_data_out, -1, FFTW_MEASURE);
                break; 
            default:
                fftw3_plan_1d_fft = fftw_plan_dft_1d(fftw3_N, fftw3_data_in, fftw3_data_out, -1, FFTW_ESTIMATE);
        }
    //    time1 = clock();
    //    printf("spe c create_fftw3_plan_1d_fft time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
	}*/
    
    // code start:
    //#ifdef DEBUG_gd
    //    printf("\nc gd, p= ");
    //#endif
    
    
    for(i = 0; i < DEF_NUM_OF_MIC; i++) {
        //for(j = 0; j < DEF_SOR_LEN; j++) {
        for(j = 0; j < DEF_PROCESS_FRAME; j++) {
            //p[i][j] = (double)(buf[i][j]/pow(2.0,15));
            //p[i][j] = (double)(50.0*buf[i][j]/pow(2.0,31));
            p[i][j] = (double)buf[i][j];
            //#ifdef DEBUG_gd
                //printf("%e", p[i][j]); printf(", ");
                //if ((j+1)%4==0) {
                //    printf("\n");
                //}
            //#endif
        }
    }
    
    #ifdef measure_time
        time0 = clock();
    #endif  
    
    FFT( fftw3_plan_1d_fft
        ,&fftw3_data_in[0]
        ,fftw3_data_out
        ,p
        ,&P_half[0]
        );
    #ifdef measure_time
        time1 = clock();
        printf("\nspe c FFT time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif
    
    //# Get MUSIC Parameter
    #ifdef measure_time
        time0 = clock();
    #endif  
    
    SorNum_guess = MUSIC_Parameter( P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                    //,fs
                    ,&k[0]
                    ,&PN[0] //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    //,&eigenvalue_save[0] //DEF_NUM_OF_FREQS*DEF_NUM_OF_MIC
                    //,SorNum_guess
                    );
    #ifdef measure_time
        time1 = clock();
        printf("\nspe c MUSIC_Parameter time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif
    //# 判定為沒有聲源
    //for(i = 0; i < DEF_NUM_OF_MIC; i++) {
    //    for(j = 0; j < 2; j++) {
    //        solution[i][j] = 0;
    //    }
    //} 

    if (SorNum_guess != 0) { //# 有聲源
        #ifdef measure_time
            time0 = clock();
        #endif
        PSO_Localization(MicPos //3*DEF_NUM_OF_MIC
                        ,PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                        ,k //DEF_NFFT/2
                        ,SorNum_guess
                        ,&swarm[0] //swarms*(3*dimension+2)
                        //,double swarm[DEF_PSO_SWARMS][3*DEF_PSO_DIMENSION+2]
                        );
        #ifdef measure_time
            time1 = clock();
            printf("\nspe c PSO_Localization time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
        #endif
        //#print (time.clock()-t0)
        //# Find source position
        //solution = np.zeros([SorNum_guess,2])   # 每個聲源的水平角與仰角
        
        //sorted_swarm = swarm[np.lexsort(-swarm.T)]  # 按照fitness值大小排序(由大到小)        
        for(i = 0; i < m; i++) { //-swarm.T
            for(j = 0; j < DEF_PSO_SWARMS; j++) {
                swarm_t_tmp[i][j] = -1.0*swarm[j][i];
            }
        }

        np_lexsort_2d_double_v1(m ,DEF_PSO_SWARMS ,swarm_t_tmp ,&swarm_sort_tmp[0]);
        /*#ifdef DEBUG_gd
            printf("\nc gd, swarm= ");
            for (i = 0; i < DEF_PSO_SWARMS; i++) {
                for (j = 0; j < m; j++) {
                    printf("%e", swarm[i][j]); printf(", ");
                    if ((j+1)%4==0) {
                        printf("\n");
                    }
                }
                printf("\n");
            }
            
            printf("\nc gd, -swarm.T= ");
            for (i = 0; i < m; i++) {
                for (j = 0; j < DEF_PSO_SWARMS; j++) {
                    printf("%e", swarm_t_tmp[i][j]); printf(", ");
                    if ((j+1)%4==0) {
                        printf("\n");
                    }
                }
                printf("\n");
            }
            
            printf("\nc gd, lexsort(-swarm.T)= ");
            for (i = 0; i < DEF_PSO_SWARMS; i++) {
                printf("%d", swarm_sort_tmp[i]); printf(", ");
                if ((i+1)%4==0) {
                    printf("\n");
                }
            }
            printf("\nc gd, sorted_swarm= ");
            for (i = 0; i < DEF_PSO_SWARMS; i++) {
                for (j = 0; j < m; j++) {
                    printf("%lf", sorted_swarm[i][j]); printf(", ");
                    if ((j+1)%4==0) {
                        printf("\n");
                    }
                }
                printf("\n");
            }
        #endif*/

        for(i = 0; i < DEF_PSO_SWARMS; i++) {
            n = swarm_sort_tmp[i];
            for(j = 0; j < m; j++) {
                //n = swarm_sort_tmp[j];
                //sorted_swarm[i][j] = swarm[i][n];
                sorted_swarm[i][j] = swarm[n][j];
            }
        }
        #ifdef DEBUG_gd
            printf("\nc gd, sorted_swarm= ");
            for (i = 0; i < DEF_PSO_SWARMS; i++) {
                for (j = 0; j < m; j++) {
                    printf("%lf", sorted_swarm[i][j]); printf(", ");
                    if ((j+1)%4==0) {
                        printf("\n");
                    }
                }
                printf("\n");
            }
        #endif         

        //solution[0,:] = sorted_swarm[0,:2]
        for(j = 0; j < 2; j++) {
            solution[0][j] = sorted_swarm[0][j];
        }        

        //swarm_point = 1;
        //solution_point = 1;
        
        //for SorFind in range(1,len(swarm)):
        //    if solution_point >= SorNum_guess:
        //        break
        //    for check_solution in range(0,len(solution)):  # 檢查每個已選出的解
        //        repeat_flag = 0  
        //        if np.linalg.norm(sorted_swarm[SorFind,:2]-solution[check_solution,:])<=40: # 距離不夠遠
        //            repeat_flag=1 
        //            swarm_point+=1
        //            break
        //    if repeat_flag==0:
        //        solution[solution_point,:]=sorted_swarm[SorFind,:2]
        //        solution_point+=1
        //        swarm_point+=1
        for(SorFind = 1; SorFind < DEF_PSO_SWARMS; SorFind++) {
            //if (solution_point >= SorNum_guess) {
            if (solution_point >= 2) { 
                break;
            }
            
            //for(check_solution = 0; check_solution < SorNum_guess; check_solution++) { // # 檢查每個已選出的解
            for(check_solution = 0; check_solution < 2; check_solution++) { // # 檢查每個已選出的解
                repeat_flag = 0;                
                tmp1 = 0;
                for(j = 0; j < 2; j++) {
                    tmp0 = sorted_swarm[SorFind][j] - solution[check_solution][j];
                    tmp1 += pow(tmp0, 2);
                }
                tmp0 = sqrt(tmp1);
                #ifdef DEBUG_gd
                    printf("\nc gd, SorFind= %d", SorFind);
                    printf("\nc gd, check_solution= %d", check_solution);
                    printf("\nc gd, np.linalg.norm= %lf", tmp0);
                #endif
                if (tmp0 <= 40.0) { // # 距離不夠遠
                    repeat_flag=1; 
                    //swarm_point+=1;                    
                    #ifdef DEBUG_gd
                        printf("\nc gd, break np.linalg.norm");
                    #endif
                    break;
                }
            }            
            
            if (repeat_flag==0) { 
                for(j = 0; j < 2; j++) {
                    solution[solution_point][j] = sorted_swarm[SorFind][j];
                } 
                
                solution_point+=1;
                //swarm_point+=1;                   
                #ifdef DEBUG_gd
                    printf("\nc gd, if repeat_flag==0");
                #endif
            } 
            #ifdef DEBUG_gd
                printf("\nc gd, solution_point= %d", solution_point);
                printf("\nc gd, solution= ");
                for (i = 0; i < DEF_NUM_OF_MIC; i++) {
                    for (j = 0; j < 2; j++) {
                        printf("%e", solution[i][j]); printf(", ");
                        if ((j+1)%4==0) {
                            printf("\n");
                        }
                    }
                    printf("\n");
                }
            #endif               
        }

    }

    if (flag_destroy_fftw3_plan_1d_fft == 1) {
        #ifdef measure_time
            //clock_t time0, time1;
            time0 = clock(); 
        #endif   
        fftw_destroy_plan(fftw3_plan_1d_fft);
        free(fftw3_data_in);
        free(fftw3_data_out);
        fftw_cleanup();
        #ifdef measure_time
            time1 = clock();
            printf("\nspe c destroy_fftw3_plan_1d_fft time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
        #endif
	}
    
    free_2d_double_array(p, DEF_NUM_OF_MIC);
    free_3d_double_complex_array(PN, DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    free_2d_double_array(swarm, DEF_PSO_SWARMS);
    //free_2d_double_array(eigenvalue_save, DEF_NUM_OF_FREQS);
    free_2d_double_array(sorted_swarm, DEF_PSO_SWARMS);
    free_2d_double_array(swarm_t_tmp, m);
    free(swarm_sort_tmp);
    free(k);
    
    return SorNum_guess;
}

double cost_MUSIC( double *position //1*3
                    ,double **MicPos //3*DEF_NUM_OF_MIC
                    ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    ,double *k //DEF_NFFT/2
                    //,double cost
                    ) 
{
    int i, j, m;
    double cost = 0;
    double **manifold_expand_tmp = create_2d_double_array(60-10, DEF_NUM_OF_MIC);
    double **manifold_expand = create_2d_double_array(DEF_NUM_OF_MIC, 60-10);
    double _Complex **w_expand = create_2d_double_complex_array(DEF_NUM_OF_MIC, 60-10);
    double _Complex ***temp1 = create_3d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC, 60-10);
    double _Complex **temp2 = create_2d_double_complex_array(DEF_NUM_OF_MIC, 60-10);
    double s=0;
    /*double _Complex *cost_tmp;   
    cost_tmp = malloc((60-10)*sizeof(double _Complex));    
    if (cost_tmp == NULL) {
		fprintf(stderr, "Out of memory");
		exit(0);
	}*/
    double _Complex *cost_tmp = create_1d_double_complex_array(60-10);
    
    //manifold_expand = (np.tile(np.dot(position, MicPos), (len(k[10:60]), 1))).T
   // printf("\n np.dot(position, MicPos)= ");
    #ifdef DEBUG_cost_MUSIC
        printf("\n manifold_expand= ");
    #endif
    for(j = 0; j < DEF_NUM_OF_MIC; j++) {
        s = 0;
        for(m = 0; m < 3; m++) {
            //s += position[0][m]*MicPos[m][j];
            s += position[m]*MicPos[m][j];
        }
        manifold_expand_tmp[0][j] = s;
       // printf("%.14lf", manifold_expand_tmp[0][j]); printf(", ");
    }
    for(i = 1; i < 60-10; i++) {
        for(j = 0; j < DEF_NUM_OF_MIC; j++) {
            manifold_expand_tmp[i][j] = manifold_expand_tmp[0][j];
        }
    }
    for(i = 0; i < DEF_NUM_OF_MIC; i++) {
        for(j = 0; j < 60-10; j++) {
            manifold_expand[i][j] = manifold_expand_tmp[j][i];
            #ifdef DEBUG_cost_MUSIC
                printf("%.14lf", manifold_expand[i][j]); printf(", ");
                if (j%2==1) {
                    printf("\n");
                }
            #endif
        }
    }  
    
    //w_expand = np.exp(1j * k[10:60] * manifold_expand).reshape(1, DEF_NUM_OF_MIC, len(k[10:60]))
    //w_expand_temp = w_expand.transpose(1, 0, 2)
    #ifdef DEBUG_cost_MUSIC
        printf("\n w_expand_temp= ");
    #endif
    for(i = 0; i < DEF_NUM_OF_MIC; i++) {
        for(j = 0; j < 60-10; j++) {
            //w_expand[i][j] = cexp(1j * k[j+10] * manifold_expand[i][j]);
            w_expand[i][j] = cexp(k[j+10] * manifold_expand[i][j] * I);
            #ifdef DEBUG_cost_MUSIC
                printf("%.14f% + i*.14f", creal(w_expand[i][j]), cimag(w_expand[i][j])); printf(", ");
                if (j%2==1) {
                    printf("\n");
                }
            #endif
        }
    } 
    
    //temp1 = np.tile(w_expand_temp, (DEF_NUM_OF_MIC, 1))
    #ifdef DEBUG_cost_MUSIC
        printf("\n temp1= ");
    #endif
    for(j = 0; j < DEF_NUM_OF_MIC; j++) {
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            for(m = 0; m < 60-10; m++) {
                temp1[i][j][m] = w_expand[i][m];
                #ifdef DEBUG_cost_MUSIC
                    printf("%.14f% + i*.14f", creal(temp1[i][j][m]), cimag(temp1[i][j][m])); printf(", ");
                    if (m%2==1) {
                        printf("\n");
                    }
                #endif
            }
        }
    } 
    
    //temp2 = np.sum(temp1.conj() * PN[:, :, 10:60], axis=0)
    #ifdef DEBUG_cost_MUSIC
        printf("\n temp2= ");
    #endif
    /*for(i = 0; i < DEF_NUM_OF_MIC; i++) {
        for(j = 0; j < DEF_NUM_OF_MIC; j++) {
            for(m = 0; m < 60-10; m++) {
                temp1[i][j][m] = conj(temp1[i][j][m])*PN[i][j][m+10];
            }
        }
    }
    for(j = 0; j < DEF_NUM_OF_MIC; j++) {
        for(m = 0; m < 60-10; m++) {
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                temp2[j][m] += temp1[i][j][m];
            }
            #ifdef DEBUG_cost_MUSIC
                printf("%1.10f%+1.10fj", creal(temp2[j][m]), cimag(temp2[j][m])); printf(", ");
                if (m%2==1) {
                    printf("\n");
                }
            #endif
        }
    }*/
    for(j = 0; j < DEF_NUM_OF_MIC; j++) {
        for(m = 0; m < 60-10; m++) {
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                temp1[i][j][m] = conj(temp1[i][j][m])*PN[i][j][m+10];
            }
            for(i = 0; i < DEF_NUM_OF_MIC; i++) {
                temp2[j][m] += temp1[i][j][m];
            }
            #ifdef DEBUG_cost_MUSIC
                printf("%1.10f%+1.10fj", creal(temp2[j][m]), cimag(temp2[j][m])); printf(", ");
                if (m%2==1) {
                    printf("\n");
                }
            #endif
        }
    }
    
    //cost = np.sum(temp2 * w_expand.reshape(DEF_NUM_OF_MIC, len(k[10:60])), axis=0)
    #ifdef DEBUG_cost_MUSIC
        printf("\n cost 0= ");
    #endif
    /*for(j = 0; j < 60-10; j++) {
        cost_tmp[j] += temp2[0][j]*w_expand[0][j];
    }
    for(i = 1; i < DEF_NUM_OF_MIC; i++) {
        for(j = 0; j < 60-10; j++) {
            cost_tmp[j] += temp2[i][j]*w_expand[i][j];
            #ifdef DEBUG_cost_MUSIC
                printf("%.14f% + i*.14f", creal(cost_tmp[j]), cimag(cost_tmp[j])); printf(", ");
                if (j%2==1) {
                    printf("\n");
                }
            #endif
        }
    }*/
    for(j = 0; j < 60-10; j++) {
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            temp2[i][j] = temp2[i][j]*w_expand[i][j];
        }
        for(i = 0; i < DEF_NUM_OF_MIC; i++) {
            cost_tmp[j] += temp2[i][j];
        }
        #ifdef DEBUG_cost_MUSIC
            printf("%.10f%+.10fj", creal(cost_tmp[j]), cimag(cost_tmp[j])); printf(", ");
            if (j%2==1) {
                printf("\n");
            }
        #endif
    }
    
    //cost = np.sum(abs(1 / cost))
    //cost = 0;
    for(i = 0; i < 60-10; i++) {
        cost += cabs(1.0 / cost_tmp[i]);
        #ifdef DEBUG_cost_MUSIC
            printf("\n cost 1= %.14f", cost);
        #endif
    }
    
    free_2d_double_array(manifold_expand_tmp, 60-10);
    free_2d_double_array(manifold_expand, DEF_NUM_OF_MIC);
    free_2d_double_complex_array(w_expand, DEF_NUM_OF_MIC); 
    free_3d_double_complex_array(temp1, DEF_NUM_OF_MIC, DEF_NUM_OF_MIC); 
    free_2d_double_complex_array(temp2, DEF_NUM_OF_MIC); 
    free(cost_tmp); 
    
   // printf("b cost= %f\n", *cost);    
    return cost;
}

/*
void sub_cost_MUSIC( double position[DEF_NUM_OF_MIC*3] //DEF_NUM_OF_MIC*3
                    ,double MicPos[3*DEF_NUM_OF_MIC] //3*DEF_NUM_OF_MIC
                    ,double _Complex PN[DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS] //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    ,double k[DEF_NFFT]
                    ,double cost[DEF_NUM_OF_MIC*DEF_NUM_OF_MIC] //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC
                    ) */
/*void cost_MUSIC( double **position //DEF_NUM_OF_MIC*3
                    ,double **MicPos //3*DEF_NUM_OF_MIC
                    ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                    ,double *k
                    ,double **cost //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC
                    ) 
{
    int ff, i, j, m;
    //double **cost = create_2d_double_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    double _Complex **w = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    double _Complex s=0;
    double _Complex **tmp0 = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    //complex_t **tmp1 = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_NUM_OF_MIC);
    
    for (ff = 10; ff < 60; ff++)
    {
        //w=np.exp(1j*k[ff]*np.dot(position,MicPos))
        for (i = 0; i < DEF_NUM_OF_MIC; i++)
        {
            for (j = 0; j < DEF_NUM_OF_MIC; j++)
            {
                s = 0;
                for (m = 0; m < 3; m++)
                {
                    s += position[i][m]*MicPos[m][j];
                    //s += position[i*3 + m]*MicPos[m*DEF_NUM_OF_MIC+j];
                }
                w[i][j] = cexp(1j*k[ff]*s);
            }
        }
        
        //cost+=abs(1/np.dot(np.dot(w.conj(),PN[:,:,ff]),w.T))
        for (i = 0; i < DEF_NUM_OF_MIC; i++)
        {
            for (j = 0; j < DEF_NUM_OF_MIC; j++)
            {
                s = 0;
                for (m = 0; m < DEF_NUM_OF_MIC; m++)
                {
                    s += conj(w[i][m])*PN[m][j][ff];
                    //s += conj(w[i][m])*PN[m*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS + j*DEF_NUM_OF_FREQS + ff];
                }
                tmp0[i][j] = s;
            }
        }
        for (i = 0; i < DEF_NUM_OF_MIC; i++)
        {
            for (j = 0; j < DEF_NUM_OF_MIC; j++)
            {
                s = 0;
                for (m = 0; m < DEF_NUM_OF_MIC; m++)
                {
                    s += tmp0[i][m]*w[j][m];
                }
                s = 1/s;
                cost[i][j] += sqrt(pow(creal(s), 2) + pow(cimag(s), 2));
                //cost[i*DEF_NUM_OF_MIC + j] += sqrt(pow(creal(s), 2) + pow(cimag(s), 2));
            }
        }
    }
    
    free_2d_double_complex_array(w, DEF_NUM_OF_MIC);
    free_2d_double_complex_array(tmp0, DEF_NUM_OF_MIC);      
    //return cost;
} */

void PSO_Localization(   double **MicPos //3*DEF_NUM_OF_MIC
                        ,double _Complex ***PN //DEF_NUM_OF_MIC*DEF_NUM_OF_MIC*DEF_NUM_OF_FREQS
                        ,double *k //DEF_NFFT/2
                        ,int SorNum
                        ,double **swarm //swarms*(3*dimension+2)
                        //,double swarm[DEF_PSO_SWARMS][3*DEF_PSO_DIMENSION+2]
                        )
{
    //boundary_max=[360,90]
    int boundary_max[] = { 360, 90};
    
    int theta_grid, phi_grid, i_1, i_2, i, times, SorNo, dim, j, m=0;
    int P_cluster=0, index=0, magnitude=0, grid_time=0;
    float inertia=1;
    float correction_factor_body=1;
    float correction_factor_group=1;
    double cost=0;

    double **distance = create_2d_double_array(DEF_PSO_SWARMS, DEF_PSO_SWARMS);
    int *vote_time = create_1d_int_array(DEF_PSO_SWARMS);
    int *neighbor_gbest = create_1d_int_array(DEF_PSO_SWARMS);
    //double **swarm = create_2d_double_array(DEF_PSO_SWARMS, 3*DEF_PSO_DIMENSION+2);
    double *angle = create_1d_double_array(DEF_PSO_DIMENSION);
    double *position = create_1d_double_array(3);
   // printf("\n c SorNum=%d", SorNum);
    double tmp_d0=0;
    
    // [lc]
    //for theta_grid in range(grid_section,boundary_max[0],grid_section):
    //    for phi_grid in range(30,60,30):
    //        swarm[grid_time,0]=theta_grid
    //        swarm[grid_time,1]=phi_grid
    //        grid_time+=1
   // printf("\n spe c swarm=\n");
    for (theta_grid = DEF_PSO_GRID_SECTION; theta_grid < boundary_max[0]; theta_grid+= DEF_PSO_GRID_SECTION) {
        for (phi_grid = 30; phi_grid < 60; phi_grid+= 30) {
            i=grid_time;
            swarm[i][0] = theta_grid;
            swarm[i][1] = phi_grid;
            grid_time+=1;
            //printf("%f", swarm[i][0]);
            //printf(", ");
            //printf("%f", swarm[i][1]);
            //printf(", ");
        }
    }
    
    // [lc]
    //for i_1 in range(0,swarms):
    //    for i_2 in range(0,swarms):
    //        distance[i_1,i_2]=np.linalg.norm([swarm[i_1,0]-swarm[i_2,0],swarm[i_1,1]-swarm[i_2,1]])
    //        distance[i_2,i_1]=distance[i_1,i_2] 
    //neighbor_gbest=np.zeros([swarms,1])   # 儲存每個粒子的 local best 的 index
    //vote_time=np.zeros([swarms,1])        # 累計每個粒子被當成local best的次數
   // printf("\n spe c distance=\n"); 
    for (i_1 = 0; i_1 < DEF_PSO_SWARMS; i_1++) {
        for (i_2 = 0; i_2 < DEF_PSO_SWARMS; i_2++) {
            distance[i_1][i_2] = sqrt(pow(swarm[i_1][0]-swarm[i_2][0], 2)+pow(swarm[i_1][1]-swarm[i_2][1], 2));
            distance[i_2][i_1] = distance[i_1][i_2];
            //printf("%e", distance[i_1][i_2]);
            //printf(", ");
        }
    }    
        
    // [lc]
    //# Main
    //for times in range(0,iterations):
    //    # 迭代相關參數修正
    //    P_cluster=0                       # 聚集在可能聲源位置附近的粒子數量
    //    for SorNo in range(0,SorNum):
    //        # 取得聚集最多粒子的聚集數量與其index
    //        index = np.argmax(vote_time,axis=0)
    //        magnitude = vote_time[index,0]
    //        P_cluster+=magnitude             # 累計次數 
    //        vote_time[index,0]=0          # 清除紀錄
    //    
    //    vote_time=np.zeros([swarms,1])        # 重新累計每個粒子被當成local best的次數
    //
    //    # 權重參數修正
    //    inertia*=swarms/(swarms+P_cluster)
    //    correction_factor_body*=swarms/(swarms+P_cluster)
    //    correction_factor_group*=(swarms+ 0.5*P_cluster)/swarms
    //    
    //
    //    for i in range(0,swarms):
    //        for dim in range(0,dimension):
    //            swarm[i,dim]+=swarm[i,dimension+dim]/1.2
    //            # 檢查邊界
    //            if swarm[i,dim]>=boundary_max[dim] or swarm[i,dim]<=boundary_min:
    //                swarm[i,dim]=boundary_max[dim]*np.random.random()
    //        # 計算Fitness
    //        angle=swarm[i,:dimension]
    //        position=np.array([math.cos(angle[1]/(180/math.pi))*math.cos(angle[0]/(180/math.pi)),math.cos(angle[1]/(180/math.pi))*math.sin(angle[0]/(180/math.pi)),math.sin(angle[1]/(180/math.pi))])
    //        cost = cost_MUSIC(position,MicPos,PN,k)     
    //        swarm[i,3*dimension+1]=cost
    //        # 修正Personal best
    //        if cost>swarm[i,3*dimension]:
    //            swarm[i,2*dimension:3*dimension]=swarm[i,0:dimension]
    //            swarm[i,3*dimension]=cost
    //        # 修正Local best
    //        neighbor_gbest[i,0]=i      # 先把自己當作自己的local best      
    //        for i_2 in range(0,swarms):
    //            distance[i,i_2]=np.linalg.norm([swarm[i,0]-swarm[i_2,0],swarm[i,1]-swarm[i_2,1]]) # 修正距離
    //            if distance[i,i_2]<neighbor_dis:
    //                if swarm[i_2,3*dimension+1]>= swarm[int(neighbor_gbest[i,0]),3*dimension+1]:
    //                    neighbor_gbest[i,0]=i_2     # 更新該粒子的local best
    //        vote_time[int(neighbor_gbest[i,0]),0]+=1     # 累計該粒子被當成local best 的次數
    //                
    //    # 計算下一刻速度
    //    for i in range(0,swarms):
    //        for dim in range(0,dimension):
    //            swarm[i,dimension+dim]=np.random.random()*inertia*swarm[i,dimension+dim]+np.random.random()*(swarm[i,2*dimension+dim]-swarm[i,dim])+correction_factor_group*np.random.random()*(swarm[int(neighbor_gbest[i,0]),2*dimension+dim]-swarm[i,dim])
  #if defined DEBUG_pso || defined DEBUG_cost_MUSIC || defined DEBUG_gd
    for (times = 0; times < 2; times++) {
  #else
    for (times = 0; times < DEF_PSO_ITERATIONS; times++) {
  #endif
        #ifdef DEBUG_pso
            printf("\n spe c times= %d", times);
        #endif
        P_cluster = 0;
        for (SorNo = 0; SorNo < SorNum; SorNo++) {
            index = np_argmax_1d_int(vote_time, DEF_PSO_SWARMS);
            magnitude = vote_time[index];
            P_cluster+=magnitude; //# 累計次數 
            vote_time[index]=0; //# 清除紀錄
            #ifdef DEBUG_pso
                printf("\n spe c SorNo, index, magnitude, P_cluster= %d", SorNo); printf(", %d", index); printf(", %d", magnitude); printf(", %d", P_cluster);
            #endif
        }
        
        fill_1d_int_array(vote_time, DEF_PSO_SWARMS, 0); //# 重新累計每個粒子被當成local best的次數
    
        //# 權重參數修正
        inertia*=DEF_PSO_SWARMS/(DEF_PSO_SWARMS+ 1.0*P_cluster);
        correction_factor_body*=DEF_PSO_SWARMS/(DEF_PSO_SWARMS+ 1.0*P_cluster);
        correction_factor_group*=(DEF_PSO_SWARMS+ 0.5*P_cluster)/DEF_PSO_SWARMS;
        #ifdef DEBUG_pso
            printf("\n spe c inertia, correction_factor_body, correction_factor_group= %lf", inertia); printf(", %lf", correction_factor_body); printf(", %lf", correction_factor_group);
        #endif
      #ifdef DEBUG_cost_MUSIC
        for (i = 0; i < 5; i++) {
      #else
        for (i = 0; i < DEF_PSO_SWARMS; i++) {
      #endif        
            #ifdef DEBUG_pso
                printf("\n spe c i= %d", i);
            #endif
            for (dim = 0; dim < DEF_PSO_DIMENSION; dim++) {
                #ifdef DEBUG_pso
                    printf("\n spe c dim= %d", dim);
                #endif
                swarm[i][dim]+=swarm[i][DEF_PSO_DIMENSION+dim]/1.2;
                #ifdef DEBUG_pso
                    printf("\n spe c swarm[i][dim]= %lf", swarm[i][dim]);
                #endif
                //# 檢查邊界
                if (swarm[i][dim]>=boundary_max[dim] || swarm[i][dim]<=DEF_PSO_BOUNDARY_MIN) {
                    #if defined DEBUG_pso || defined DEBUG_cost_MUSIC || defined DEBUG_gd
                        tmp_d0 = 0.5; //random()/(RAND_MAX + 1.0);
                    #else
                        tmp_d0 = random()/(RAND_MAX + 1.0);
                    #endif                    
                    swarm[i][dim]=boundary_max[dim]*tmp_d0; //random();
                    #ifdef DEBUG_pso
                        printf("\n spe c swarm[i][dim] if= %lf", swarm[i][dim]);
                    #endif
                }
            }
            //# 計算Fitness
            for (j = 0; j < DEF_PSO_DIMENSION; j++) {
                angle[j]=swarm[i][j];
                #ifdef DEBUG_pso
                    printf("\n spe c angle[j]= %lf", angle[j]);
                #endif
            }
                
            position[0]=cos(angle[1]/(180/PI))*cos(angle[0]/(180/PI));
            position[1]=cos(angle[1]/(180/PI))*sin(angle[0]/(180/PI));
            position[2]=sin(angle[1]/(180/PI));            
            cost = cost_MUSIC(position, MicPos, PN, k); //, cost);     
            swarm[i][3*DEF_PSO_DIMENSION+1]=cost;
            #if defined DEBUG_pso || defined DEBUG_cost_MUSIC
                printf("\n spe c position, cost, swarm[i][3*dimension+1]= %lf", position[0]); printf(", %lf", position[1]); printf(", %lf", position[2]); 
                printf(", %lf", cost); printf(", %lf", swarm[i][3*DEF_PSO_DIMENSION+1]);
            #endif
            
            //# 修正Personal best
            if (cost>swarm[i][3*DEF_PSO_DIMENSION]) {
                for (j = 0; j < DEF_PSO_DIMENSION; j++) {
                    swarm[i][j+2*DEF_PSO_DIMENSION]=swarm[i][j];
                    #ifdef DEBUG_pso
                        printf("\n spe c swarm[i][j+2*DEF_PSO_DIMENSION]= %lf", swarm[i][j+2*DEF_PSO_DIMENSION]);
                    #endif
                }
                swarm[i][3*DEF_PSO_DIMENSION]=cost;
                #ifdef DEBUG_pso
                    printf("\n spe c swarm[i][3*DEF_PSO_DIMENSION]= %lf", swarm[i][3*DEF_PSO_DIMENSION]);
                #endif
            }
            //# 修正Local best
            neighbor_gbest[i]=i;      //# 先把自己當作自己的local best
            #ifdef DEBUG_pso
                printf("\n spe c neighbor_gbest[i]= %d", neighbor_gbest[i]);
            #endif
            for (i_2 = 0; i_2 < DEF_PSO_SWARMS; i_2++) {
                #ifdef DEBUG_pso
                    printf("\n spe c i_2= %d", i_2);
                #endif
                distance[i][i_2]=np_linalg_norm_double(swarm[i][0]-swarm[i_2][0], swarm[i][1]-swarm[i_2][1]); //np.linalg.norm([swarm[i,0]-swarm[i_2,0],swarm[i,1]-swarm[i_2,1]]) # 修正距離
                #ifdef DEBUG_pso
                    printf("\n spe c distance[i][i_2]= %lf", distance[i][i_2]);
                #endif
                if (distance[i][i_2]<DEF_PSO_NEIGHBOR_DIS) {
                    m = neighbor_gbest[i];
                    if (swarm[i_2][3*DEF_PSO_DIMENSION+1]>= swarm[m][3*DEF_PSO_DIMENSION+1]) { //swarm[int(neighbor_gbest[i][0]),3*dimension+1])
                        neighbor_gbest[i]=i_2;     //# 更新該粒子的local best
                        #ifdef DEBUG_pso
                            printf("\n spe c neighbor_gbest[i] if= %d", neighbor_gbest[i]);
                        #endif
                    }
                }
            }
            m = neighbor_gbest[i];
            vote_time[m]+=1;     //# 累計該粒子被當成local best 的次數
            #ifdef DEBUG_pso
                printf("\n spe c vote_time[m]= %d", vote_time[m]);
            #endif
        }
        //# 計算下一刻速度
        for (i = 0; i < DEF_PSO_SWARMS; i++) {
            for (dim = 0; dim < DEF_PSO_DIMENSION; dim++) {
                m = neighbor_gbest[i];
                #if defined DEBUG_pso || defined DEBUG_cost_MUSIC || defined DEBUG_gd
                    tmp_d0 = 0.5; //random()/(RAND_MAX + 1.0);
                #else
                    tmp_d0 = random()/(RAND_MAX + 1.0);
                #endif
                swarm[i][DEF_PSO_DIMENSION+dim]= tmp_d0*inertia*swarm[i][DEF_PSO_DIMENSION+dim]
                                            +tmp_d0*(swarm[i][2*DEF_PSO_DIMENSION+dim]-swarm[i][dim])
                                            +correction_factor_group*tmp_d0*(swarm[m][2*DEF_PSO_DIMENSION+dim]-swarm[i][dim]);
                #ifdef DEBUG_pso
                    printf("\n spe c swarm[i][DEF_PSO_DIMENSION+dim] f= %.8lf", swarm[i][DEF_PSO_DIMENSION+dim]);
                #endif
            }
        }
    }
    
    #ifdef DEBUG_pso
        printf("\n spe c swarm return= \n");
        for (i = 0; i < DEF_PSO_SWARMS; i++) {
            printf("\n");
            for (j = 0; j < 3*DEF_PSO_DIMENSION+2; j++) {
                printf("%.8lf", swarm[i][j]); printf(", ");
                if ((j+1)%(3*DEF_PSO_DIMENSION-1)==0) {
                    printf("\n");
                }
            }
        }
    #endif
    
    free_2d_double_array(distance, DEF_PSO_SWARMS);
    free(vote_time);
    free(neighbor_gbest);
    free(angle);
    free(position);
           
    //return swarm;
}

void TIKR_extration_opt (double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                        ,double **solution //2*2 //max: DEF_NUM_OF_MIC*2, original: SorNum_guess*2
                        ,int SorNum_guess
                        ,double **MicPos //3*DEF_NUM_OF_MIC
                        ,double *k_TIKR //DEF_HOPSIZE
                        ,float bata_i
                        ,int flag_create_fftw3_plan_1d_ifft
                        ,int flag_destroy_fftw3_plan_1d_ifft
                        ,double **p_o //2*(DEF_PROCESS_FRAME*DEF_CHUNK) //SorNum_guess*(DEF_PROCESS_FRAME*DEF_CHUNK)
                        )
{
    //source code:
    //def TIKR_extration_opt (P_half, solution, SorNum_guess, MicPos, k_TIKR, bata):
    //     
    //    NumOfFrame=np.size(P_half,2)
    //#    print('solution_IN_EXTRACT=',solution)
    //    p=np.zeros([SorNum_guess,process_frame*DEF_CHUNK],float)
    //    
    //    TIKR_spectrum = np.zeros([freqs.size,SorNum_guess,NumOfFrame],dtype = 'complex')
    //    
    //    #print("freqs.size=",freqs.size)
    //    sig_in_f_half = P_half
    //    for sor_num in range(0,SorNum_guess):
    //        kappa = np.array([math.sin(math.radians(solution[1,sor_num]))*math.cos(math.radians(solution[0,sor_num])), 
    //                          math.sin(math.radians(solution[1,sor_num]))*math.sin(math.radians(solution[0,sor_num])), 
    //                          math.cos(math.radians(solution[1,sor_num]))]).reshape(1,3)
    //        k_temp = 1j*np.array(np.dot(kappa,MicPos))
    //        k_temp2 = k_temp.T*k_TIKR
    //        A = np.exp(k_temp2)
    //        A_plam = A.conj()
    //        #TIKR
    //        for frame_no in range(0,NumOfFrame):
    //            TIKR_spectrum[:,sor_num,frame_no] = (np.sum(((np.tile(1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)),(DEF_NUM_OF_MIC,1)))*A_plam*sig_in_f_half[:,:,frame_no]),axis=0)).reshape(freqs.size)
    //    #thresholding
    //#     time_a = time.time()
    //    max_value = np.max(np.abs(TIKR_spectrum), axis=1)
    //    for sor_num in range(0, SorNum_guess):
    //        compare_table = (np.abs(TIKR_spectrum[:, sor_num, :]) < max_value)
    //        compare_table = np.where(compare_table == True, 0.2,1)
    //        TIKR_spectrum[:, sor_num, :] = TIKR_spectrum[:, sor_num, :]*compare_table
    //    
    //#     time_b = time.time()
    //    
    //#    print("extraction_time=",time_b-time_a)
    //
    //    
    //    # ifft to time domain
    //    for frame_no in range(0,NumOfFrame):
    //        t_start = int(frame_no*hopsize)
    //        for sor_num in range(0,SorNum_guess):
    //            amp_TIKR_spectrum = TIKR_spectrum[:,sor_num,frame_no]
    //            p_total = np.fft.irfft(np.concatenate((amp_TIKR_spectrum,np.zeros([1, ]), np.flip((amp_TIKR_spectrum[1:]).conj(),0)),axis = 0),NWIN)
    //            p[sor_num,t_start:t_start+NWIN] = p[sor_num,t_start:t_start+NWIN] + p_total
    //    
    //    return p
    
    int i, j, m;
    int sor_num, frame_no, t_start = 0;
    
    int SorNum_guess_tmp = 2;
    if (SorNum_guess > 2)
        SorNum_guess_tmp = 2;
    else
        SorNum_guess_tmp = SorNum_guess;
        
    //double s=0;
    double _Complex ***TIKR_spectrum = create_3d_double_complex_array(DEF_HOPSIZE, 2, DEF_NUM_OF_FRAME_CONST); //SorNum_guess, DEF_NUM_OF_FRAME_CONST);
    double ***TIKR_spectrum_abs = create_3d_double_array(DEF_HOPSIZE, 2, DEF_NUM_OF_FRAME_CONST); //SorNum_guess, DEF_NUM_OF_FRAME_CONST);
    double **max_value = create_2d_double_array(DEF_HOPSIZE, DEF_NUM_OF_FRAME_CONST);
    double _Complex ***sig_in_f_half = create_and_copy_3d_double_complex_array(DEF_NUM_OF_MIC, (int)(DEF_NWIN/2), DEF_NUM_OF_FRAME_CONST, P_half);
    double  *kappa = create_1d_double_array(3);
    double  *k_temp = create_1d_double_array(DEF_NUM_OF_MIC);
    double **k_temp2 = create_2d_double_array(DEF_NUM_OF_MIC, DEF_HOPSIZE);
    double _Complex **A = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_HOPSIZE);
    double _Complex **A_plam = create_2d_double_complex_array(DEF_NUM_OF_MIC, DEF_HOPSIZE);
    double  **compare_table = create_2d_double_array_inivalue(DEF_HOPSIZE, DEF_NUM_OF_FRAME_CONST, 1.0);
    //double _Complex *arr_ifft0 = create_1d_double_complex_array(DEF_NWIN);
    
    // fftw3 plan
    int fftw3_N = DEF_NWIN;
//    short fftw3_setting = 0;
    //fftw_complex *fftw3_data_in;
    //fftw3_data_in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * fftw3_N); // = create_1d_double_complex_array(fftw3_N);
    //fftw_complex *fftw3_data_out;
    //fftw3_data_out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * fftw3_N); // = create_1d_double_complex_array(fftw3_N);
    double _Complex *fftw3_data_in = create_1d_double_complex_array(fftw3_N);
    double _Complex *fftw3_data_out = create_1d_double_complex_array(fftw3_N);
    
    #ifdef measure_time
        clock_t time0, time1;
        time0 = clock();
    #endif
    //fftw_plan fftw3_plan_1d_ifft = fftw_plan_dft_1d(fftw3_N, fftw3_data_out, fftw3_data_in, 1, FFTW_ESTIMATE);
    fftw_plan fftw3_plan_1d_ifft = fftw_plan_dft_1d( fftw3_N
                                                    ,reinterpret_cast<fftw_complex*>(fftw3_data_in)
                                                    ,reinterpret_cast<fftw_complex*>(fftw3_data_out)
                                                    ,1
                                                    ,FFTW_ESTIMATE);
    #ifdef measure_time
        time1 = clock();
        printf("\nspe c create_fftw3_plan_1d_ifft time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
    #endif
    
    /*#ifdef DEBUG_TIKR_extration_opt
        printf("\n spe c TIKR_extration_opt, solution= \n");
        for (i = 0; i < 2; i++) {
            printf("\n");
            for (m = 0; m < SorNum_guess; m++) {
                printf("%.8lf", solution[i][m]); printf(", ");
                if ((m+1)%4==0) {
                    printf("\n");
                }
            }
        }printf("\n spe c TIKR_extration_opt, MicPos= \n");
        for (i = 0; i < 3; i++) {
            printf("\n");
            for (m = 0; m < DEF_NUM_OF_MIC; m++) {
                printf("%.8lf", MicPos[i][m]); printf(", ");
                if ((m+1)%4==0) {
                    printf("\n");
                }
            }
        }
    #endif*/
    
    //for(sor_num = 0 ; sor_num < SorNum_guess ; sor_num++) {    
    for(sor_num = 0 ; sor_num < SorNum_guess_tmp ; sor_num++) {
        kappa[0] = sin(deg2rad_double(solution[1][sor_num]))*cos(deg2rad_double(solution[0][sor_num])); 
        kappa[1] = sin(deg2rad_double(solution[1][sor_num]))*sin(deg2rad_double(solution[0][sor_num])); 
        kappa[2] = cos(deg2rad_double(solution[1][sor_num]));
        
        //k_temp = np.array(np.dot(kappa,MicPos))
        for(j = 0 ; j < DEF_NUM_OF_MIC ; j++) {
            //s = 0;
            k_temp[j] = 0;
            for(m = 0 ; m < 3 ; m++) {
                //s += kappa[k]*MicPos[k][j];
                k_temp[j] += kappa[m]*MicPos[m][j];
            }
            //k_temp[j] = s;
        }
        
        //k_temp2 = (k_temp).T*k_TIKR
        //A = np.exp(1j*k_temp2)
        //A_plam = A.conj()
        for(i = 0 ; i < DEF_NUM_OF_MIC ; i++) {
            for(j = 0 ; j < DEF_HOPSIZE ; j++) {
                k_temp2[i][j] = k_temp[i]*k_TIKR[j]; //k_temp2 = (k_temp).T*k_TIKR
                //A[i][j] = cexp(1j*k_temp2[i][j]); //A = np.exp(1j*k_temp2)
                A[i][j] = cexp(k_temp2[i][j]*I); //A = np.exp(1j*k_temp2)
                A_plam[i][j] = conj(A[i][j]); //A_plam = A.conj()
                A[i][j] = pow(creal(A_plam[i][j]), 2) + pow(cimag(A_plam[i][j]), 2) + 0*j; //A*A_plam
                //cimag(A[i][j]) = 0;
            }
        }    
        /*#ifdef DEBUG_TIKR_extration_opt
            printf("\n spe c TIKR_extration_opt, sor_num= %d", sor_num);
            printf("\n spe c TIKR_extration_opt, kappa= ");
            for (i = 0; i < 3; i++) {
                printf("\n");
                printf("%.8lf", kappa[i]); printf(", ");
            }
            printf("\n spe c TIKR_extration_opt, k_temp=np.dot(kappa,MicPos)= ");
            for (i = 0; i < DEF_NUM_OF_MIC; i++) {
                printf("\n");
                printf("%.8lf", k_temp[i]); printf(", ");
            }
            printf("\n spe c TIKR_extration_opt, k_temp2=(k_temp).T*k_TIKR= ");
            for (i = 0; i < DEF_NUM_OF_MIC; i++) {
                printf("\n");
                for (j = 0; j < DEF_HOPSIZE; j++) {
                    printf("%.8lf", k_temp2[i][j]); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            }
            printf("\n spe c TIKR_extration_opt, A=np.exp(1j*k_temp2)= ");
            for (i = 0; i < DEF_NUM_OF_MIC; i++) {
                printf("\n");
                for (j = 0; j < DEF_HOPSIZE; j++) {
                    printf("%.14f% + i*.14f", creal(cexp(1j*k_temp2[i][j])), cimag(cexp(1j*k_temp2[i][j]))); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            }
            printf("\n spe c TIKR_extration_opt, A*A_plam= ");
            for (i = 0; i < DEF_NUM_OF_MIC; i++) {
                printf("\n");
                for (j = 0; j < DEF_HOPSIZE; j++) {
                    printf("%.14f% + i*.14f", creal(A[i][j]), cimag(A[i][j])); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            }
        #endif*/
        
        //for frame_no in range(0,NumOfFrame):
        //    TIKR_spectrum[:,sor_num,frame_no] = (np.sum(((np.tile(1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)),(DEF_NUM_OF_MIC,1)))*A_plam*sig_in_f_half[:,:,frame_no]),axis=0)).reshape(freqs.size)
        // step 1. calc (np.tile(1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)),(DEF_NUM_OF_MIC,1)))*A_plam
        // step 1.0 calc A[0][j] = np.sum(A*A_plam,axis=0)
        for(j = 0 ; j < DEF_HOPSIZE ; j++) {            
            for(i = 1 ; i < DEF_NUM_OF_MIC ; i++) {
                A[0][j] += A[i][j];
            }
        }
        /*#ifdef DEBUG_TIKR_extration_opt
            printf("\n spe c TIKR_extration_opt, sor_num= %d", sor_num);
            printf("\n spe c TIKR_extration_opt, np.sum(A*A_plam,axis=0)= ");
            printf("\n");
            for (j = 0; j < DEF_HOPSIZE; j++) {
                printf("%.14f% + i*.14f", creal(A[0][j]), cimag(A[0][j])); printf(", ");
                if ((j+1)%2==0) {
                    printf("\n");
                }
            }
        #endif*/
        
        // step 1.1 calc A[0][j] = 1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)) 
        for(j = 0 ; j < DEF_HOPSIZE ; j++) { 
            A[0][j] = 1/(A[0][j] + pow(bata_i, 2));
        }
        /*#ifdef DEBUG_TIKR_extration_opt
            printf("\n spe c TIKR_extration_opt, sor_num= %d", sor_num);
            printf("\n spe c TIKR_extration_opt, 1/(np.sum(A*A_plam,axis=0)+np.power(bata,2))= ");
            printf("\n");
            for (j = 0; j < DEF_HOPSIZE; j++) {
                printf("%.14f% + i*.14f", creal(A[0][j]), cimag(A[0][j])); printf(", ");
                if ((j+1)%2==0) {
                    printf("\n");
                }
            }
        #endif*/
            
        // step 1.2 calc A = (np.tile(1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)),(DEF_NUM_OF_MIC,1)))*A_plam 
        for(j = 0 ; j < DEF_HOPSIZE ; j++) { 
            /*for(i = 1 ; i < DEF_NUM_OF_MIC ; i++) {
                A[i][j] = A[0][j];
            }*/
            for(i = 1 ; i < DEF_NUM_OF_MIC ; i++) {
                A[i][j] = A[0][j]*A_plam[i][j];
            }
            A[0][j] *= A_plam[0][j];
        }
        /*#ifdef DEBUG_TIKR_extration_opt
            printf("\n spe c TIKR_extration_opt, sor_num= %d", sor_num);
            printf("\n spe c TIKR_extration_opt, (np.tile(1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)),(DEF_NUM_OF_MIC,1)))*A_plam= \n");
            for(i = 0 ; i < DEF_NUM_OF_MIC ; i++) {
                for (j = 0; j < DEF_HOPSIZE; j++) {
                    printf("%.14f% + i*.14f", creal(A[i][j]), cimag(A[i][j])); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            }
        #endif*/
            
        // step 1.3 calc A = (np.tile(1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)),(DEF_NUM_OF_MIC,1)))*A_plam 
        /*for(j = 0 ; j < DEF_HOPSIZE ; j++) { 
            for(i = 0 ; i < DEF_NUM_OF_MIC ; i++) {
                A[i][j] *= A_plam[i][j];
            }
        }*/
            
        for(frame_no = 0 ; frame_no < DEF_NUM_OF_FRAME_CONST ; frame_no++) {    
            // step 2.1 calc A_plam = ((np.tile(1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)),(DEF_NUM_OF_MIC,1)))*A_plam*sig_in_f_half[:,:,frame_no])               
            for(j = 0 ; j < DEF_HOPSIZE ; j++) {
                for(i = 0 ; i < DEF_NUM_OF_MIC ; i++) {
                    A_plam[i][j] = A[i][j]*sig_in_f_half[i][j][frame_no];
                }
            }
                
            // step 2.2 calc np.sum(((np.tile(1/(np.sum(A*A_plam,axis=0)+np.power(bata,2)),(DEF_NUM_OF_MIC,1)))*A_plam*sig_in_f_half),axis=0)               
            for(j = 0 ; j < DEF_HOPSIZE ; j++) {
                TIKR_spectrum[j][sor_num][frame_no] = A_plam[0][j];
                for(i = 1 ; i < DEF_NUM_OF_MIC ; i++) {
                    TIKR_spectrum[j][sor_num][frame_no] += A_plam[i][j];
                }
            }
        } 
    }    
    /*#ifdef DEBUG_TIKR_extration_opt
        printf("\n spe c TIKR_extration_opt, TIKR_spectrum= \n");
        for (i = 0; i < DEF_HOPSIZE; i++) {
            printf("\n");
            //for (j = 0; j < SorNum_guess; j++) {
            for (j = 0; j < SorNum_guess_tmp; j++) {
                printf("\n");
                for (m = 0; m < DEF_NUM_OF_FRAME_CONST; m++) {
                    printf("%.14f% + i*.14f", creal(TIKR_spectrum[i][j][m]), cimag(TIKR_spectrum[i][j][m])); printf(", ");
                    if ((m+1)%2==0) {
                        printf("\n");
                    }
                }
            }
        }
    #endif*/
    
    //#thresholding
    //    max_value = np.max(np.abs(TIKR_spectrum), axis=1)
    for(i = 0 ; i < DEF_HOPSIZE ; i++) {
        for(m = 0 ; m < DEF_NUM_OF_FRAME_CONST ; m++) {
            //for(j = 0 ; j < SorNum_guess ; j++) {
            for(j = 0 ; j < SorNum_guess_tmp ; j++) {
                TIKR_spectrum_abs[i][j][m] = cabs(TIKR_spectrum[i][j][m]);
            }
            
            max_value[i][m] = TIKR_spectrum_abs[i][0][m];
            
            if (SorNum_guess > 1) {
                //for(j = 1 ; j < SorNum_guess ; j++) {
                for(j = 1 ; j < SorNum_guess_tmp ; j++) {
                    if (TIKR_spectrum_abs[i][j][m] > max_value[i][m]) {
                        max_value[i][m] = TIKR_spectrum_abs[i][j][m];
                    }
                }
            }
        }
    }    
    /*#ifdef DEBUG_TIKR_extration_opt
        printf("\n spe c TIKR_extration_opt, max_value= \n");
        for (i = 0; i < DEF_HOPSIZE; i++) {
            printf("\n");
            for (m = 0; m < DEF_NUM_OF_FRAME_CONST; m++) {
                printf("%.8lf", max_value[i][m]); printf(", ");
                if ((m+1)%4==0) {
                    printf("\n");
                }
            }
        }
    #endif*/
    
    //for(sor_num = 0 ; sor_num < SorNum_guess ; sor_num++) {    
    for(sor_num = 0 ; sor_num < SorNum_guess_tmp ; sor_num++) {
        //compare_table = (np.abs(TIKR_spectrum[:, sor_num, :]) < max_value)
        //compare_table = np.where(compare_table == True, 0.2,1)
        //TIKR_spectrum[:, sor_num, :] = TIKR_spectrum[:, sor_num, :]*compare_table
        for(i = 0 ; i < DEF_HOPSIZE ; i++) {
            for(m = 0 ; m < DEF_NUM_OF_FRAME_CONST ; m++) {
                if (TIKR_spectrum_abs[i][sor_num][m] < max_value[i][m]) {
                    compare_table[i][m] = 0.2;
                }
                
                TIKR_spectrum[i][sor_num][m] = TIKR_spectrum[i][sor_num][m]*compare_table[i][m];
            }
        } 
    }    
    /*#ifdef DEBUG_TIKR_extration_opt
        printf("\n spe c TIKR_extration_opt, TIKR_spectrum= \n");
        for (i = 0; i < DEF_HOPSIZE; i++) {
            printf("\n");
            //for (j = 0; j < SorNum_guess; j++) {
            for (j = 0; j < SorNum_guess_tmp; j++) {
                printf("\n");
                for (m = 0; m < DEF_NUM_OF_FRAME_CONST; m++) {
                    printf("%.14f% + i*.14f", creal(TIKR_spectrum[i][j][m]), cimag(TIKR_spectrum[i][j][m])); printf(", ");
                    if ((m+1)%2==0) {
                        printf("\n");
                    }
                }
            }
        }
    #endif*/
    
    // ifft to time domain
    for(frame_no = 0 ; frame_no < DEF_NUM_OF_FRAME_CONST ; frame_no++) {
        t_start = frame_no*DEF_HOPSIZE;
        /*#ifdef DEBUG_TIKR_extration_opt
            printf("\n spe c TIKR_extration_opt, frame_no= %d", frame_no);
        #endif*/
            
        //for(sor_num = 0 ; sor_num < SorNum_guess ; sor_num++) {
        for(sor_num = 0 ; sor_num < SorNum_guess_tmp ; sor_num++) {
            //amp_TIKR_spectrum = TIKR_spectrum[:,sor_num,frame_no]
            //p_total = np.fft.irfft(np.concatenate((amp_TIKR_spectrum,np.zeros([1, ]), np.flip((amp_TIKR_spectrum[1:]).conj(),0)),axis = 0),NWIN)            
            //step 1. calc fftw3_data_in = np.concatenate((amp_TIKR_spectrum,np.zeros([1, ]), np.flip((amp_TIKR_spectrum[1:]).conj(),0)),axis = 0)   
            /*#ifdef DEBUG_TIKR_extration_opt
                printf("\n spe c TIKR_extration_opt, sor_num= %d", sor_num);
            #endif*/
            for(i = 0 ; i < DEF_HOPSIZE ; i++) {
                fftw3_data_in[i] = TIKR_spectrum[i][sor_num][frame_no];
            }
            
            fftw3_data_in[DEF_HOPSIZE] = 0;
            
            for(i = 1 ; i < DEF_HOPSIZE ; i++) {
                fftw3_data_in[DEF_HOPSIZE+i] = conj(TIKR_spectrum[DEF_HOPSIZE-i][sor_num][frame_no]);
            }    
            /*#ifdef DEBUG_TIKR_extration_opt
                printf("\n spe c TIKR_extration_opt, fftw3_data_in= \n");
                for (i = 0; i < DEF_NWIN; i++) {
                    printf("%.14f% + i*.14f", creal(fftw3_data_in[i]), cimag(fftw3_data_in[i])); printf(", ");
                    if ((i+1)%2==0) {
                        printf("\n");
                    }
                }
            #endif*/
            
            //step 2. calc p_total = np.fft.irfft(arr_ifft0,NWIN)
            fftw_execute(fftw3_plan_1d_ifft);    
            /*#ifdef DEBUG_TIKR_extration_opt
                printf("\n spe c TIKR_extration_opt, fftw3_data_out= \n");
                for (i = 0; i < DEF_NWIN; i++) {
                    printf("%.14f% + i*.14f", creal(fftw3_data_out[i]), cimag(fftw3_data_out[i])); printf(", ");
                    if ((i+1)%2==0) {
                        printf("\n");
                    }
                }
            #endif*/
            
            //p[sor_num,t_start:t_start+NWIN] = p[sor_num,t_start:t_start+NWIN] + p_total
            for(j = t_start ; j < t_start+DEF_NWIN ; j++) {
                p_o[sor_num][j] += creal(fftw3_data_out[j-t_start])/(1.0*fftw3_N);
            }
            #ifdef DEBUG_TIKR_extration_opt
                /*printf("\n spe c TIKR_extration_opt, p_o= \n");
                for (j = 0; j < DEF_PROCESS_FRAME*DEF_CHUNK; j++) {
                    printf("%.14f", p_o[sor_num][j]); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }*/
                if ((frame_no==0) || (frame_no==(int)(DEF_NUM_OF_FRAME_CONST/2)) || (frame_no==(int)(DEF_NUM_OF_FRAME_CONST-1))) {
                    printf("\n spe c TIKR_extration_opt, frame_no= %d", frame_no);
                    printf("\n spe c TIKR_extration_opt, sor_num= %d", sor_num);
                    printf("\n spe c TIKR_extration_opt, p_o= \n");
                    for (j = 0; j < DEF_PROCESS_FRAME*DEF_CHUNK; j++) {
                        printf("%.14f", p_o[sor_num][j]); printf(", ");
                        if ((j+1)%2==0) {
                            printf("\n");
                        }
                    }
                }
            #endif
        }
    }
    #ifdef DEBUG_TIKR_extration_opt
        printf("\n spe c TIKR_extration_opt, DEF_NUM_OF_FRAME_CONST*DEF_HOPSIZE= %d\n", DEF_NUM_OF_FRAME_CONST*DEF_HOPSIZE);
        printf("\n spe c TIKR_extration_opt, SorNum_guess*(DEF_PROCESS_FRAME*DEF_CHUNK)= %d\n", SorNum_guess*(DEF_PROCESS_FRAME*DEF_CHUNK));
    #endif

    if (flag_destroy_fftw3_plan_1d_ifft == 1) {
        #ifdef measure_time
            //clock_t time0, time1;
            time0 = clock(); 
        #endif   
        fftw_destroy_plan(fftw3_plan_1d_ifft);
        free(fftw3_data_in);
        free(fftw3_data_out);
        fftw_cleanup();
        #ifdef measure_time
            time1 = clock();
            printf("\nspe c destroy_fftw3_plan_1d_ifft time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
        #endif
	}
    
    
    free_3d_double_complex_array(TIKR_spectrum, DEF_HOPSIZE, 2); //SorNum_guess);
    free_3d_double_array(TIKR_spectrum_abs, DEF_HOPSIZE, 2); //SorNum_guess);
    free_2d_double_array(max_value, DEF_HOPSIZE); //[lc add]
    free_3d_double_complex_array(sig_in_f_half, DEF_NUM_OF_MIC, DEF_NWIN/2);
    free(kappa);
    free(k_temp);
    free_2d_double_array(k_temp2, DEF_NUM_OF_MIC);
    free_2d_double_complex_array(A, DEF_NUM_OF_MIC);
    free_2d_double_complex_array(A_plam, DEF_NUM_OF_MIC);
    free_2d_double_array(compare_table, DEF_HOPSIZE);
           
    //return p_o;
}

void beamforming        (double _Complex ***P_half //DEF_NUM_OF_MIC*(DEF_NWIN/2)*DEF_NUM_OF_FRAME_CONST
                        ,double **solution //2*2 //max: DEF_NUM_OF_MIC*2, original: SorNum_guess*2
                        ,int SorNum_guess
                        ,double **MicPos //3*DEF_NUM_OF_MIC
                        ,double *k_TIKR //DEF_HOPSIZE
                        ,float bata_i
                        ,int flag_create_fftw3_plan_1d_ifft
                        ,int flag_destroy_fftw3_plan_1d_ifft
                        ,double **p_o //2*(DEF_PROCESS_FRAME*DEF_CHUNK) //SorNum_guess*(DEF_PROCESS_FRAME*DEF_CHUNK)
                        )
{
    //source code:
    //def beamforming(solution, SorNum_guess, P_half, k):
    //    Run_iters=0     # 目前進行第幾次定位
    //    count = 0
    //    if SorNum_guess==0:   # 判斷為沒有聲源
    //        p=0
    //#         print('There is no sound source!!')
    //    elif SorNum_guess==1:
    //#         print('The horizontal angle is',solution[0,0])
    //#         print('The vertical angle is',solution[0,1])
    //        solution=solution.T
    //        #time_a = time.time()
    //        #p=TIKR_extration_wi_mmse_opt(P_half, solution, SorNum_guess, MicPos, k_TIKR, bata)
    //        p=TIKR_extration_opt(P_half, solution, SorNum_guess, MicPos, k_TIKR, bata)
    //        #time_b = time.time()
    //        #print("extraction_time=",time_b-time_a)
    //        
    //        #write file
    //        p_final = np.zeros((1,p.size))
    //        for i in range(0,SorNum_guess):
    //            #p[i,:] = mmse_de_noise(p[i,:])
    //            p[i,:] = p[i,:]/(2*max(abs(p[i,:])))
    //            
    //            #librosa.output.write_wav('source_'+str(i)+'_signal_extraction_out.wav',(p[i,:]*32767).astype(np.int16),RATE)
    //            #print("block.value==============",block.value)
    //            #scipy.io.wavfile.write('block'+str(block.value)+'source_'+str(i)+'_sig_ext_out.wav',RATE,(p[i,:]*32767).astype(np.int16))
    //        #print("write extraction end")
    //            
    //    elif SorNum_guess==2:
    //#         print('Souce1')
    //#         print('The horizontal angle is',solution[0,0])
    //#         print('The verticle angle is',solution[0,1])
    //#         print('Souce2')
    //#         print('The horizontal angle is',solution[1,0])
    //#         print('The verticle angle is',solution[1,1])
    //        solution=solution.T
    //        #time_a = time.time()
    //        p=TIKR_extration_opt(P_half, solution, SorNum_guess, MicPos, k_TIKR, bata)
    //        #p=TIKR_extration_wi_mmse_opt(P_half, solution, SorNum_guess, MicPos, k_TIKR, bata)
    //        #time_b = time.time()
    //        #print("extraction_time=",time_b-time_a)
    //        #write file
    //#         aa = time.time()
    //        p_final = np.zeros((2,int(p.size/2)))
    //        for i in range(0,SorNum_guess):
    //            p[i,:] = p[i,:]/(2*max(abs(p[i,:])))
    //
    //            #p_final[i,:] = mmse_de_noise(p[i,:])
    //#         bb = time.time()
    //#        print("de-noise time = ",bb-aa)
    //            #librosa.output.write_wav('source_'+str(i)+'_signal_extraction_out.wav',p[i,:],RATE)
    //#            scipy.io.wavfile.write('block'+str(block.value)+'source_'+str(i)+'_sig_ext_out.wav',RATE,(p[i,:]*32767).astype(np.int16))
    //            #scipy.io.wavfile.write('block'+str(block.value)+'source_'+str(i)+'_sig_ext_out.wav',RATE,(p[i,:]*32767).astype(np.int16))
    //#         print("write extraction end")
    //    else:
    //        p=0
    //#         print('Too much source!!')
    //    return p
    
    int i, j;
    //int Run_iters=0;     //# 目前進行第幾次定位
    //int count = 0;    
    double max_value = 0;
    double **solution_trans = create_2d_double_array(2, 2); //SorNum_guess);
    
    if ( (SorNum_guess==1) || (SorNum_guess==2) ) {
//#         print('Souce1')
//#         print('The horizontal angle is',solution[0,0])
//#         print('The verticle angle is',solution[0,1])
//#         print('Souce2')
//#         print('The horizontal angle is',solution[1,0])
//#         print('The verticle angle is',solution[1,1])        
        for(i = 0 ; i < 2 ; i++) { //solution=solution.T
            for(j = 0 ; j < SorNum_guess ; j++) {
                solution_trans[i][j] = solution[j][i];
            }
        }
        
        TIKR_extration_opt( P_half, solution_trans, SorNum_guess, MicPos, k_TIKR, DEF_BATA,
                            flag_create_fftw3_plan_1d_ifft, flag_destroy_fftw3_plan_1d_ifft,
                            &p_o[0]);   
        /*#ifdef DEBUG_beamforming
            printf("\n spe c beamforming, p_o= \n");
            for (i = 0; i < SorNum_guess; i++) {
                for (j = 0; j < DEF_PROCESS_FRAME*DEF_CHUNK; j++) {
                    printf("%.14f", p_o[i][j]); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            }
        #endif*/
                            
        //p_final = np.zeros((2,int(p.size/2)))
        for(i = 0 ; i < SorNum_guess ; i++) {
            // calc max(abs(p[i,:]))
            max_value = fabs(p_o[i][0]);
            
            //for(j = 1 ; j < DEF_PROCESS_FRAME*DEF_CHUNK ; j++) {
            for(j = 1 ; j < DEF_PROCESS_FRAME ; j++) {
                if (fabs(p_o[i][j]) > max_value) {
                    max_value = fabs(p_o[i][j]);
                }
            }   
            #ifdef DEBUG_beamforming
                printf("\n spe c beamforming, i= %d", i);
                printf("\n spe c beamforming, max_value= %.14f", max_value);
                /*printf("\n spe c beamforming, TIKR_extration_opt p_o= \n");
                for (j = 0; j < DEF_PROCESS_FRAME*DEF_CHUNK; j++) {
                    printf("%.14f", p_o[i][j]); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }*/
            #endif
            
            max_value *= 2.0;
            
            //#p[i,:] = mmse_de_noise(p[i,:])
            //p[i,:] = p[i,:]/(2*max(abs(p[i,:])))
            //for(j = 0 ; j < DEF_PROCESS_FRAME*DEF_CHUNK ; j++) {
            for(j = 0 ; j < DEF_PROCESS_FRAME ; j++) {
                p_o[i][j] /= max_value;
            }    
            /*#ifdef DEBUG_beamforming
                //printf("\n spe c beamforming, i= %d\n", i);
                printf("\n spe c beamforming, p[i,:] = p[i,:]/(2*max(abs(p[i,:])))= \n");
                for (j = 0; j < DEF_PROCESS_FRAME*DEF_CHUNK; j++) {
                    printf("%.14f", p_o[i][j]); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
                }
            #endif*/
        }
    }

    free_2d_double_array(solution_trans, 2);
    
}

