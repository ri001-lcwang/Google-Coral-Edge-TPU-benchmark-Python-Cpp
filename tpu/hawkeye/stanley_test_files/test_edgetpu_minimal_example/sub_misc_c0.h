//typedef double _Complex complex_t;

// Avoids C++ name mangling with extern "C"
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#include <stdio.h>
#include <stdlib.h>
#define PI acos(-1)
//#include "sub_ulib_realtime_pso_c0.h"

//#pragma comment(lib , "blas.lib") 
//#pragma comment(lib , "clapack.lib")

//void *malloc(size_t n) ;

//#include "f2c.h"
//#include "clapack.h"

#include <time.h>

//sudo apt-get install liblapacke-dev
//#include <lapacke.h>
#include "lapacke.h"



//#define for_python_or_c
//#define DEBUG_np_linalg_eig_double_complex
//#define DEBUG_np_lexsort_2d_double

// Parameter: 
/*#define SMALL 1.0E-30
#define NEARZERO 1.0E-10

#define max(a,b) ((a>b)?a:b)*/


#ifdef for_python_or_c
    //void flat_np_lexsort_2d_double( int row_i ,int col_i ,double p[row_i*col_i] ,int lexsort_o[col_i]);
    void flat_np_lexsort_2d_double_v1( int row_i ,int col_i ,double p[] ,int lexsort_o[]);
    void flat_np_lexsort_2d_double_v2( int row_i ,int col_i ,double p[] ,int lexsort_o[]);
    /*void flat_np_linalg_eig_double_complex(  int row_in
                                            ,int max_iteration_in //最大迭代次數，讓資料更準確
                                            ,double complex flat_A_in[] // 2d
                                            ,double complex eigenvalue_o[]
                                            ,double complex flat_eigenvector_o[] // 2d
                                            );*/
    void test_np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double complex flat_A_in[]
                                                    ,double complex eigenvalue_o[]
                                                    ,double complex flat_eigenvector_o[]);
    void flat_np_linalg_eig_square_double_complex_array( int row_in
                                                        ,double complex *flat_A_in // 2d
                                                        ,double complex *eigenvalue_o
                                                        ,double complex *flat_eigenvector_o // 2d
                                                        );


    double deg2rad_double(double degrees);
    int *create_1d_int_array(int N);
    float *create_1d_float_array(int N);
    double *create_1d_double_array(int N);
    double complex *create_1d_double_complex_array(int N);
    int **create_2d_int_array(int N, int M);
    float **create_2d_float_array(int N, int M);
    double **create_2d_double_array(int N, int M);
    double **create_2d_double_array_inivalue(int N, int M, double inivalue);
    double complex **create_2d_double_complex_array(int N, int M);
    double complex **create_2d_double_complex_identity_array(int N);
    double ***create_3d_double_array(int N, int M, int L);
    double complex ***create_3d_double_complex_array(int N, int M, int L);
    void free_2d_int_array(int** p, int N);
    void free_2d_float_array(float** p, int N);
    void free_2d_double_array(double** p, int N);
    void free_2d_double_complex_array(double complex** p, int N);
    void free_3d_double_array(double*** p, int N, int M);
    void free_3d_double_complex_array(double complex*** p, int N, int M);
    float *flatten_2d_float_array(float** p, int N, int M);
    double *flatten_2d_double_array(double** p, int N, int M);
    double complex *flatten_2d_double_complex_array(double complex** p, int N, int M);
    double *flatten_3d_double_array(double*** p, int N, int M, int L);
    double complex *flatten_3d_double_complex_array(double complex*** p, int N, int M, int L);
    float **deflate_2d_float_array(float* p, int N, int M);
    double **deflate_2d_double_array(double* p, int N, int M);
    double complex **deflate_2d_double_complex_array(double complex* p, int N, int M);
    double ***deflate_3d_double_array(double* p, int N, int M, int L);
    double complex ***deflate_3d_double_complex_array(double complex* p, int N, int M, int L);
    void fill_1d_int_array(int* p, int dim, int value);
    double *create_and_copy_1d_double_array(int a_col_in ,double* a_in);
    void copy_1d_double_complex_array(int a_col_in ,double complex* a_in, double complex* b_o);
    void copy_2d_double_complex_array(int a_row_in ,int a_col_in ,double complex** a_in ,double complex** b_o);
    double **create_and_copy_2d_double_array(int N ,int M ,double** a_in);
    double complex ***create_and_copy_3d_double_complex_array(int N ,int M ,int L ,double complex*** a_in);
    //double int_to_double(int input);
    void np_dot_2d_double(int a_row_in ,int a_col_in ,int b_col_in ,double** a_in, double** b_in, double** c_o);
    void np_dot_2d_double_complex(int a_row_in ,int a_col_in ,int b_col_in ,double complex** a_in, double complex** b_in, double complex** c_o);
    int np_argmax_1d_int(int *a, int n);
    int np_argmax_1d_float(float *a, int n);
    float np_max_1d_float(float *a, int n);
    double np_linalg_norm_double(double a, double b);
    //float *scipy_signal_hanning_float(int N ,short itype);
    //double *scipy_signal_hanning_double(int N ,short itype);
    void np_hanning_double(int M_i, double* np_hanning_o);
    void scipy_signal_get_window_hann_double(int nx_i ,double* win_o);
    void np_pad_reflect_1d_double(double* a_i ,int a_row_i ,int pad_width_i ,double* b_o);
    double *np_linspace_double(double start ,double stop ,int num);
    double *np_diff_1d_double(double *a_i ,int a_row_i);
    double **np_subtract_outer_1d_double_in(double *a_i ,int a_row_i ,double *b_i ,int b_row_i);
    double complex Norm2_1d_double_complex(int row_in ,double complex *a);
    /*void qr_decomposition_square_double_complex( int row_in
                                                ,double complex **A_in
                                                ,double complex **Q_o
                                                ,double complex **R_o);
    void eigenvector_from_eigenvalue_square_double_complex(  int row_in
                                                            ,double complex **A_in
                                                            ,double complex *eigenvalue_in
                                                            ,double complex **eigenvector_o);
    void np_linalg_eig_double_complex(   int row_in
                                        ,int max_iteration_in //最大迭代次數，讓資料更準確
                                        ,double complex **A_in
                                        ,double complex *eigenvalue_o
                                        ,double complex **eigenvector_o);*/

    double norm_double(double A_in); // =norm() in C++
    double norm_double_complex(double complex A_in); // =norm() in C++
    void matSca_2d_double_complex_array(int a_row_in ,int a_col_in ,double complex scalar_in ,double complex** A_in ,double complex** C_o); // Scalar multiple of matrix
    void matLin_2d_double_complex_array( int a_row_in ,int a_col_in ,double complex a_scalar_in ,double complex** A_in 
                                        ,double complex b_scalar_in ,double complex** B_in ,double complex** C_o); // Linear combination of matrices
    double matNorm_1d_double_complex_array( int a_col_in ,double complex* A_in); // Complex vector norm
    double matNorm_2d_double_array( int a_row_in ,int a_col_in ,double** A_in); // matrix norm
    double matNorm_2d_double_complex_array( int a_row_in ,int a_col_in ,double complex** A_in); // Complex matrix norm
    double subNorm_square_double_complex_array( int a_row_in ,double complex** A_in); // Below leading diagonal of square matrix
    /*double complex shift_square_double_complex_array(int row_in ,double complex** A_in);
    void Hessenberg_square_double_complex_array(int row_in ,double complex** A_in ,double complex** P_o ,double complex** H_o);
    void QRFactoriseGivens_square_double_complex_array(int row_in ,double complex** A_in ,double complex** Q_o ,double complex** R_o);
    void QRHessenberg_square_double_complex_array(int row_in ,double complex** A_in ,double complex** P_o ,double complex** T_o);
    void eigenvectorUpper_square_double_complex_array(int row_in ,double complex** T_in ,double complex** E_o);*/
    void np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double complex **A_in
                                                    ,double complex *eigenvalue_o
                                                    ,double complex **eigenvector_o);
    void stable_sort_1d_double(int col_i ,double* p_i ,int* stable_sort_o);
    void np_argsort_1d_double(int size ,double* p_in ,int* argsort_o);
    void np_argsort_1d_double_complex(int size ,double complex* p_in ,int* argsort_o);
    void np_lexsort_2d_double_v1(int row ,int col ,double** p_in ,int* lexsort_o);
    void np_lexsort_2d_double_v2(int row ,int col ,double** p_in ,int* lexsort_o);
#else //C++
    //extern "C" {
    //extern void zgeev(char jobvl, char jobvr, int n, double complex *a, int lda, double complex *w, double complex *vl, int ldvl, double complex *vr, int ldvr, int *info);
    //}
    /* ZGEEV prototype */
    /*extern void zgeev(char * jobvl, char * jobvr, int * n,
		   double _Complex * A, int * lda,
		   double _Complex * w, double _Complex * vl, int * ldvl,
		   double _Complex * vr, int * ldvr,
		   double _Complex * work, int * lwork,
		   double * rwork, int * info);*/
    void flat_np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double _Complex *flat_A_in
                                                    ,double _Complex *eigenvalue_o
                                                    ,double _Complex *flat_eigenvector_o);
    double deg2rad_double(double degrees);
    int *create_1d_int_array(int N);
    float *create_1d_float_array(int N);
    double *create_1d_double_array(int N);
    double _Complex *create_1d_double_complex_array(int N);
    char *create_1d_char_array(int N);
    short **create_2d_short_array(int N, int M);
    int **create_2d_int_array(int N, int M);
    float **create_2d_float_array(int N, int M);
    double **create_2d_double_array(int N, int M);
    double **create_2d_double_array_inivalue(int N, int M, double inivalue);
    double _Complex **create_2d_double_complex_array(int N, int M);
    double _Complex **create_2d_double_complex_identity_array(int N);
    double ***create_3d_double_array(int N, int M, int L);
    double _Complex ***create_3d_double_complex_array(int N, int M, int L);
    void free_2d_int_array(int** p, int N);
    void free_2d_float_array(float** p, int N);
    void free_2d_double_array(double** p, int N);
    void free_2d_double_complex_array(double _Complex** p, int N);
    void free_3d_double_array(double*** p, int N, int M);
    void free_3d_double_complex_array(double _Complex*** p, int N, int M);
    float *flatten_2d_float_array(float** p, int N, int M);
    double *flatten_2d_double_array(double** p, int N, int M);
    double _Complex *flatten_2d_double_complex_array(double _Complex** p, int N, int M);
    double *flatten_3d_double_array(double*** p, int N, int M, int L);
    double _Complex *flatten_3d_double_complex_array(double _Complex*** p, int N, int M, int L);
    float **deflate_2d_float_array(float* p, int N, int M);
    double **deflate_2d_double_array(double* p, int N, int M);
    double _Complex **deflate_2d_double_complex_array(double _Complex* p, int N, int M);
    double ***deflate_3d_double_array(double* p, int N, int M, int L);
    double _Complex ***deflate_3d_double_complex_array(double _Complex* p, int N, int M, int L);
    void fill_1d_int_array(int* p, int dim, int value);
    double *create_and_copy_1d_double_array(int a_col_in ,double* a_in);
    void copy_1d_double_complex_array(int a_col_in ,double _Complex* a_in, double _Complex* b_o);
    void copy_2d_double_complex_array(int a_row_in ,int a_col_in ,double _Complex** a_in ,double _Complex** b_o);
    double **create_and_copy_2d_double_array(int N ,int M ,double** a_in);
    double _Complex ***create_and_copy_3d_double_complex_array(int N ,int M ,int L ,double _Complex*** a_in);
    //double int_to_double(int input);
    void np_dot_2d_double(int a_row_in ,int a_col_in ,int b_col_in ,double** a_in, double** b_in, double** c_o);
    void np_dot_2d_double_complex(int a_row_in ,int a_col_in ,int b_col_in ,double _Complex** a_in, double _Complex** b_in, double _Complex** c_o);
    int np_argmax_1d_int(int *a, int n);
    int np_argmax_1d_float(float *a, int n);
    float np_max_1d_float(float *a, int n);
    double np_linalg_norm_double(double a, double b);
    //float *scipy_signal_hanning_float(int N ,short itype);
    //double *scipy_signal_hanning_double(int N ,short itype);
    void np_hanning_double(int M_i, double* np_hanning_o);
    void scipy_signal_get_window_hann_double(int nx_i ,double* win_o);
    void np_pad_reflect_1d_double(double* a_i ,int a_row_i ,int pad_width_i ,double* b_o);
    double *np_linspace_double(double start ,double stop ,int num);
    double *np_diff_1d_double(double *a_i ,int a_row_i);
    double **np_subtract_outer_1d_double_in(double *a_i ,int a_row_i ,double *b_i ,int b_row_i);
    double _Complex Norm2_1d_double_complex(int row_in ,double _Complex *a);
    /*void qr_decomposition_square_double_complex( int row_in
                                                ,double _Complex **A_in
                                                ,double _Complex **Q_o
                                                ,double _Complex **R_o);
    void eigenvector_from_eigenvalue_square_double_complex(  int row_in
                                                            ,double _Complex **A_in
                                                            ,double _Complex *eigenvalue_in
                                                            ,double _Complex **eigenvector_o);
    void np_linalg_eig_double_complex(   int row_in
                                        ,int max_iteration_in //最大迭代次數，讓資料更準確
                                        ,double _Complex **A_in
                                        ,double _Complex *eigenvalue_o
                                        ,double _Complex **eigenvector_o);*/

    double norm_double(double A_in); // =norm() in C++
    double norm_double_complex(double _Complex A_in); // =norm() in C++
    void matSca_2d_double_complex_array(int a_row_in ,int a_col_in ,double _Complex scalar_in ,double _Complex** A_in ,double _Complex** C_o); // Scalar multiple of matrix
    void matLin_2d_double_complex_array( int a_row_in ,int a_col_in ,double _Complex a_scalar_in ,double _Complex** A_in 
                                        ,double _Complex b_scalar_in ,double _Complex** B_in ,double _Complex** C_o); // Linear combination of matrices
    double matNorm_1d_double_complex_array( int a_col_in ,double _Complex* A_in); // Complex vector norm
    double matNorm_2d_double_array( int a_row_in ,int a_col_in ,double** A_in); // matrix norm
    double matNorm_2d_double_complex_array( int a_row_in ,int a_col_in ,double _Complex** A_in); // Complex matrix norm
    double subNorm_square_double_complex_array( int a_row_in ,double _Complex** A_in); // Below leading diagonal of square matrix
    /*double _Complex shift_square_double_complex_array(int row_in ,double _Complex** A_in);
    void Hessenberg_square_double_complex_array(int row_in ,double _Complex** A_in ,double _Complex** P_o ,double _Complex** H_o);
    void QRFactoriseGivens_square_double_complex_array(int row_in ,double _Complex** A_in ,double _Complex** Q_o ,double _Complex** R_o);
    void QRHessenberg_square_double_complex_array(int row_in ,double _Complex** A_in ,double _Complex** P_o ,double _Complex** T_o);
    void eigenvectorUpper_square_double_complex_array(int row_in ,double _Complex** T_in ,double _Complex** E_o);*/
    void np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double _Complex **A_in
                                                    ,double _Complex *eigenvalue_o
                                                    ,double _Complex **eigenvector_o);
    void stable_sort_1d_double(int col_i ,double* p_i ,int* stable_sort_o);
    void np_argsort_1d_double(int size ,double* p_in ,int* argsort_o);
    void np_argsort_1d_double_complex(int size ,double _Complex* p_in ,int* argsort_o);
    void np_lexsort_2d_double_v1(int row ,int col ,double** p_in ,int* lexsort_o);
    void np_lexsort_2d_double_v2(int row ,int col ,double** p_in ,int* lexsort_o);
#endif


//#include <string.h>
//int get_memory_usage_kb(long* vmrss_kb, long* vmsize_kb);
