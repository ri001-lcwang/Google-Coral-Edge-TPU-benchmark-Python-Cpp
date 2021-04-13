/*#include <stdio.h>
#include <stdlib.h>
#define PI acos(-1)
#include "sub_ulib_realtime_pso_c0.h"

#pragma comment(lib , "blas.lib") 
#pragma comment(lib , "clapack.lib")

//void *malloc(size_t n) ;

//#include "f2c.h"
//#include "clapack.h"

#include <time.h>*/

#include "sub_misc_c0.h"


#ifdef for_python_or_c
    //void flat_np_lexsort_2d_double_v1( int row_i ,int col_i ,double p[row_i*col_i] ,int lexsort_o[col_i])
    void flat_np_lexsort_2d_double_v1( int row_i ,int col_i ,double p[] ,int lexsort_o[])
    {
        //int i;
        double **p_in;
        p_in = deflate_2d_double_array(p, row_i, col_i);
        //int *lexsort = create_1d_int_array(col_i);
    
       // #ifdef DEBUG_np_lexsort_2d_double
            clock_t time0, time1;
            time0 = clock();
       // #endif    
        np_lexsort_2d_double_v1(row_i ,col_i ,p_in ,&lexsort_o[0]); // ,&lexsort[0]);
       // #ifdef DEBUG_np_lexsort_2d_double
            time1 = clock();
            printf("\n spe c v1 np_lexsort time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
       // #endif

        #ifdef DEBUG_np_lexsort_2d_double
            printf("c lexsort_o= ");
            for (int i = 0; i < col_i; i++) {
                //lexsort_o[i] = lexsort[i];
                printf("%d", lexsort_o[i]); printf(", ");
            }

            //printf("\nsub c count=%u\n", lexsort_o);
            //printf("\nsub c count=%u\n", *(&lexsort_o));
            printf("\nsub c count=%u\n", *(lexsort_o));
            printf("\nsub c count=%f\n", *(p));
            //printf("\nsub c count=%u\n", *double(p[0]));
            //printf("\nsub c count=%u\n", *double(p[0][0]));
            //printf("\nsub c count=%u\n", *(&lexsort_o + 1) - lexsort_o);
            printf("\nsub c sizeof (p_in)=%u", sizeof (p_in));
            printf("\nsub c sizeof (p_in)=%u", sizeof (*p_in));
            printf("\nsub c sizeof (p_in[0])=%u", sizeof (&p_in));
        #endif

        free_2d_double_array(p_in, row_i);
        //free(lexsort);

    }

    //void flat_np_lexsort_2d_double_v2( int row_i ,int col_i ,double p[row_i*col_i] ,int lexsort_o[col_i])
    void flat_np_lexsort_2d_double_v2( int row_i ,int col_i ,double p[] ,int lexsort_o[])
    {
        //int i;
        double **p_in;
        p_in = deflate_2d_double_array(p, row_i, col_i);
        //int *lexsort = create_1d_int_array(col_i);
    
       // #ifdef DEBUG_np_lexsort_2d_double
            clock_t time0, time1;
            time0 = clock();
       // #endif    
        np_lexsort_2d_double_v2(row_i ,col_i ,p_in ,&lexsort_o[0]); // ,&lexsort[0]);
       // #ifdef DEBUG_np_lexsort_2d_double
            time1 = clock();
            printf("\n spe c v2 np_lexsort time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
       // #endif

        #ifdef DEBUG_np_lexsort_2d_double
            printf("c lexsort_o= ");
            for (int i = 0; i < col_i; i++) {
                //lexsort_o[i] = lexsort[i];
                printf("%d", lexsort_o[i]); printf(", ");
            }

            //printf("\nsub c count=%u\n", lexsort_o);
            //printf("\nsub c count=%u\n", *(&lexsort_o));
            printf("\nsub c count=%u\n", *(lexsort_o));
            printf("\nsub c count=%f\n", *(p));
            //printf("\nsub c count=%u\n", *double(p[0]));
            //printf("\nsub c count=%u\n", *double(p[0][0]));
            //printf("\nsub c count=%u\n", *(&lexsort_o + 1) - lexsort_o);
            printf("\nsub c sizeof (p_in)=%u", sizeof (p_in));
            printf("\nsub c sizeof (p_in)=%u", sizeof (*p_in));
            printf("\nsub c sizeof (p_in[0])=%u", sizeof (&p_in));
        #endif

        free_2d_double_array(p_in, row_i);
        //free(lexsort);

    }

    /*void flat_np_linalg_eig_double_complex(  int row_in
                                            ,int max_iteration_in //最大迭代次數，讓資料更準確
                                            ,double complex flat_A_in[] // 2d
                                            ,double complex eigenvalue_o[]
                                            ,double complex flat_eigenvector_o[] // 2d
                                            )
    {
        //int i;
        double complex **A_in = deflate_2d_double_complex_array(flat_A_in, row_in, row_in);
        double complex **eigenvector_o = create_2d_double_complex_array(row_in, row_in);
    
       // #ifdef DEBUG_np_linalg_eig_double_complex
            clock_t time0, time1;
            time0 = clock();
       // #endif    
        np_linalg_eig_double_complex(   row_in
                                        ,max_iteration_in //最大迭代次數，讓資料更準確
                                        ,A_in
                                        ,&eigenvalue_o[0]
                                        ,&eigenvector_o[0]);
       // #ifdef DEBUG_np_linalg_eig_double_complex
            time1 = clock();
            printf("\n spe c np_linalg_eig time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
       // #endif

        #ifdef DEBUG_np_linalg_eig_double_complex
            //printf("\nflat_A_in[0]*flat_A_in[1]= %.14f%+.14fj", creal(flat_A_in[0]*flat_A_in[1]), cimag(flat_A_in[0]*flat_A_in[1]));
            //printf("\nflat_A_in[0]/flat_A_in[1]= %.14f%+.14fj", creal(flat_A_in[0]/flat_A_in[1]), cimag(flat_A_in[0]/flat_A_in[1]));
            printf("\nflat_A_in[0]^2= %.14f%+.14fj", creal(cpow(flat_A_in[0],2)), cimag(cpow(flat_A_in[0],2)));
            printf("\nc square array in= ");
            for (int i = 0; i < row_in; i++) {
                for (int j = 0; j < row_in; j++) {
                    printf("%.14f%+.14fj", creal(A_in[i][j]), cimag(A_in[i][j])); printf(", ");
                    if (j%row_in==row_in-1) {
                        printf("\n");
                    }
                }
            }
            printf("\nc eigenvalue_o= ");
            for (int i = 0; i < row_in; i++) {
                printf("%.14f%+.14fj", creal(eigenvalue_o[i]), cimag(eigenvalue_o[i])); printf(", ");
            }

            printf("\nc eigenvector_o= ");
        #endif

        for (int i = 0; i < row_in; i++) {
            for (int j = 0; j < row_in; j++) {
                flat_eigenvector_o[i*row_in+j] = eigenvector_o[i][j];
                #ifdef DEBUG_np_linalg_eig_double_complex
                    printf("%.14f%+.14fj", creal(eigenvector_o[i][j]), cimag(eigenvector_o[i][j])); printf(", ");
                    if (j%row_in==row_in-1) {
                        printf("\n");
                    }
                #endif
            }
        }

        free_2d_double_complex_array(A_in, row_in);
        free_2d_double_complex_array(eigenvector_o, row_in);

    }*/

    void test_np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double complex flat_A_in[]
                                                    ,double complex eigenvalue_o[]
                                                    ,double complex flat_eigenvector_o[])
    {
        //int i;
        double complex **A_in = deflate_2d_double_complex_array(flat_A_in, row_in, row_in);
        double complex **eigenvector_o = create_2d_double_complex_array(row_in, row_in);
    
       // #ifdef DEBUG_np_linalg_eig_double_complex
            //clock_t time0, time1;
            //time0 = clock();
       // #endif    
        np_linalg_eig_square_double_complex_array(   row_in
                                                    ,A_in //,A_in
                                                    ,&eigenvalue_o[0]
                                                    ,&eigenvector_o[0] //,&eigenvector_o[0]
                                                    );
       // #ifdef DEBUG_np_linalg_eig_double_complex
            //time1 = clock();
            //printf("\n spe c np_linalg_eig time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
       // #endif

        #ifdef DEBUG_np_linalg_eig_double_complex
            //printf("\nflat_A_in[0]*flat_A_in[1]= %.14f%+.14fj", creal(flat_A_in[0]*flat_A_in[1]), cimag(flat_A_in[0]*flat_A_in[1]));
            //printf("\nflat_A_in[0]/flat_A_in[1]= %.14f%+.14fj", creal(flat_A_in[0]/flat_A_in[1]), cimag(flat_A_in[0]/flat_A_in[1]));
            //printf("\nflat_A_in[0]^2= %.14f%+.14fj", creal(cpow(flat_A_in[0],2)), cimag(cpow(flat_A_in[0],2)));
           /* printf("\nc square array in= \n");
            for (int i = 0; i < row_in; i++) {
                for (int j = 0; j < row_in; j++) {
                    printf("%.14f%+.14fj", creal(A_in[i][j]), cimag(A_in[i][j])); printf(", ");
                    if (j%row_in==row_in-1) {
                        printf("\n");
                    }
                }
            }*/
            printf("\nc eigenvalue_o= ");
            for (int i = 0; i < row_in; i++) {
                printf("%.14f%+.14fj", creal(eigenvalue_o[i]), cimag(eigenvalue_o[i])); printf(", ");
            }   
        #endif

        for (int i = 0; i < row_in; i++) {
            for (int j = 0; j < row_in; j++) {
                flat_eigenvector_o[i*row_in+j] = eigenvector_o[i][j];
                #ifdef DEBUG_np_linalg_eig_double_complex
                    printf("%.14f%+.14fj", creal(eigenvector_o[i][j]), cimag(eigenvector_o[i][j])); printf(", ");
                    if (j%row_in==row_in-1) {
                        printf("\n");
                    }
                #endif
            }
        }

        free(flat_A_in);
        free(flat_eigenvector_o);

    }

    void flat_np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double complex *flat_A_in
                                                    ,double complex *eigenvalue_o
                                                    ,double complex *flat_eigenvector_o)
    {   
        //https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_zgeev_row.c.htm
        //https://www.ibm.com/support/knowledgecenter/SSFHY8_6.2/reference/am5gr_hsgeevx.html
        //http://www.netlib.org/clapack/old/complex16/zgeev.c
        //http://www.icl.utk.edu/~mgates3/docs/lapack.html


        double complex *P_o = create_1d_double_complex_array(row_in*row_in);
        //double complex **T_o = create_2d_double_complex_array(row_in ,row_in);
        //double complex **E_o = create_2d_double_complex_array(row_in ,row_in);

        int n = row_in, lda = row_in, ldvl = row_in, ldvr = row_in, info=1;
        //MKL_Complex16 w[N], vl[LDVL*N], vr[LDVR*N];
        //MKL_Complex16 a[LDA*N] = {
        //   {-3.84,  2.25}, {-8.94, -4.75}, { 8.95, -6.53}, {-9.87,  4.82},
        //   {-0.66,  0.83}, {-4.40, -3.82}, {-3.50, -4.26}, {-3.15,  7.36},
        //   {-3.99, -4.73}, {-5.88, -6.60}, {-3.36, -0.40}, {-0.75,  5.23},
        //   { 7.74,  4.18}, { 3.66, -7.53}, { 2.58,  3.60}, { 4.59,  5.41}
        //};

        #ifdef DEBUG_np_linalg_eig_double_complex
            clock_t time0, time1;
            time0 = clock();
        #endif
    /*    double complex *a = create_1d_double_complex_array(row_in*row_in);
        a[0]=  750543.69428599+0j; a[1]=  1879355.49996126+0j; a[2]=  3008167.30563653+0j; a[3]=  4136979.11131179+0j; a[4]=  5265790.91698706+0j; a[5]=  6394602.72266233+0j; a[6]=  7523414.52833759+0j; a[7]=  8652226.33401285+0j;
    a[8]= 1879355.49996126+0j; a[9]=  5265767.46286309+0j; a[10]=  8652179.42576492+0j; a[11]= 12038591.38866674+0j; a[12]= 15425003.35156858+0j; a[13]= 18811415.31447038+0j; a[14]= 22197827.27737221+0j; a[15]= 25584239.24027405+0j;
    a[16]= 3008167.30563653+0j; a[17]=  8652179.42576492+0j; a[18]= 14296191.5458933 +0j; a[19]= 19940203.66602169+0j; a[20]= 25584215.78615009+0j; a[21]= 31228227.90627848+0j; a[22]= 36872240.02640685+0j; a[23]= 42516252.14653525+0j;
    a[24]= 4136979.11131179+0j; a[25]= 12038591.38866674+0j; a[26]= 19940203.66602169+0j; a[27]= 27841815.94337663+0j; a[28]= 35743428.22073162+0j; a[29]= 43645040.49808656+0j; a[30]= 51546652.77544151+0j; a[31]= 59448265.05279647+0j;
    a[32]= 5265790.91698706+0j; a[33]= 15425003.35156858+0j; a[34]= 25584215.78615009+0j; a[35]= 35743428.22073162+0j; a[36]= 45902640.65531312+0j; a[37]= 56061853.08989465+0j; a[38]= 66221065.52447614+0j; a[39]= 76380277.95905769+0j;
    a[40]= 6394602.72266233+0j; a[41]= 18811415.31447038+0j; a[42]= 31228227.90627848+0j; a[43]= 43645040.49808656+0j; a[44]= 56061853.08989465+0j; a[45]= 68478665.6817027 +0j; a[46]= 80895478.27351078+0j; a[47]= 93312290.86531891+0j;
    a[48]= 7523414.52833759+0j; a[49]= 22197827.27737221+0j; a[50]= 36872240.02640685+0j; a[51]= 51546652.77544151+0j; a[52]=        66221065.5+0j; a[53]=        80895478.3+0j; a[54]=        95569891.0+0j; a[55]=         110244304+0j;
    a[56]= 8652226.33401285+0j; a[57]= 25584239.24027405+0j; a[58]= 42516252.14653525+0j; a[59]= 59448265.05279647+0j; a[60]=        76380278.0+0j; a[61]=        93312290.9+0j; a[62]=         110244304+0j; a[63]=         127176317+0j;    
        */
        //info = LAPACKE_zgeev( LAPACK_ROW_MAJOR, 'V', 'V', n, a, lda, w, vl, ldvl, vr, ldvr );
        info = LAPACKE_zgeev( LAPACK_ROW_MAJOR, 'N', 'V', n, flat_A_in, lda, eigenvalue_o, P_o, ldvl, flat_eigenvector_o, ldvr );
        //free(a);
        #ifdef DEBUG_np_linalg_eig_double_complex
            time1 = clock();
            printf("\n spe c np_linalg_eig time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
        #endif


        //QRHessenberg_square_double_complex_array(row_in ,A_in ,&P_o[0] ,&T_o[0]);
        //eigenvectorUpper_square_double_complex_array(row_in ,T_o ,&E_o[0]); //,&eigenvector_o[0]); //,&E_o[0]);
        //np_dot_2d_double_complex(row_in ,row_in ,row_in ,P_o ,E_o ,&eigenvector_o[0]); //matMul( P, E );

        #ifdef DEBUG_np_linalg_eig_double_complex
            if (info==0)
                printf("\nsub c calc eigen success");
            else
                printf("\nsub c calc eigen fail");

            int i;        
            printf("\nsub c eigenvalue_o= ");

            for (i = 0; i < row_in; i++) {
                //eigenvalue_o[i] = T_o[i][i];
                printf("\n");
                printf("%.14f%+.14fj", creal(eigenvalue_o[i]), cimag(eigenvalue_o[i])); printf(", ");
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
            }

            printf("\nsub c eigenvector_o= ");
            for (i = 0; i < row_in*row_in; i++) {
                //for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f%+.14fj", creal(flat_eigenvector_o[i]), cimag(flat_eigenvector_o[i])); printf(", ");
                //} 
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
                // Check matrix norm of   A v - lambda v
                //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
            }
        #endif

       // free_2d_double_complex_array(P_o, row_in);
       // free_2d_double_complex_array(T_o, row_in);
        //free_2d_double_complex_array(E_o, row_in);
        free(P_o);
    }

    double deg2rad_double(double degrees) {
        return degrees*PI/180.0;
    };

    int *create_1d_int_array(int N) /* Allocate the array */
    {
        int i;
        int *array;    
        array = malloc(N*sizeof(int));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }

    float *create_1d_float_array(int N) /* Allocate the array */
    {
        int i;
        float *array;    
        array = malloc(N*sizeof(float));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }

    double *create_1d_double_array(int N) /* Allocate the array */
    {
        int i;
        double *array;    
        array = malloc(N*sizeof(double));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }

    double complex *create_1d_double_complex_array(int N) /* Allocate the array */
    {
        int i;
        double complex *array;    
        array = malloc(N*sizeof(double complex));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }
    
    short **create_2d_short_array(int N, int M)
    {
        // Check if allocation succeeded. (check for NULL pointer) 
        int i, j;
        short **array;    
        array = (short **)malloc(N*sizeof(short *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}
        
        for(i = 0 ; i < N ; i++)
        {
            array[i] = (short *)malloc( M*sizeof(short) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }
        
        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    int **create_2d_int_array(int N, int M) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        int **array;    
        array = malloc(N*sizeof(int *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(int) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    float **create_2d_float_array(int N, int M) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        float **array;    
        array = malloc(N*sizeof(float *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(float) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    double **create_2d_double_array(int N, int M) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        double **array;    
        array = malloc(N*sizeof(double *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    double **create_2d_double_array_inivalue(int N, int M, double inivalue) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        double **array;    
        array = malloc(N*sizeof(double *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = inivalue; //j;
        return array;
    }

    double complex **create_2d_double_complex_array(int N, int M) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        double complex **array;    
        array = malloc(N*sizeof(double complex *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double complex) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    double complex **create_2d_double_complex_identity_array(int N) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        double complex **array;    
        array = malloc(N*sizeof(double complex *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( N*sizeof(double complex) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++) {
            for(j = 0 ; j < N ; j++) {
                if (i == j)
                    array[i][j] = 1;
                else
                    array[i][j] = 0;
            }
        }
        return array;
    }

    double ***create_3d_double_array(int N, int M, int L) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j, k;
        double ***array;    
        array = malloc(N*sizeof(double **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double*) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = malloc( L*sizeof(double) );
                if (array[i][j] == NULL) {
    			fprintf(stderr, "Out of memory");
    			exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = 0;
        return array;
    }

    double complex ***create_3d_double_complex_array(int N, int M, int L) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j, k;
        double complex ***array;    
        array = malloc(N*sizeof(double complex **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double complex*) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = malloc( L*sizeof(double complex) );
    			if (array[i][j] == NULL) {
    			fprintf(stderr, "Out of memory");
    			exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = 0;
        return array;
    }

    void free_2d_int_array(int** p, int N) {
        int i;
        for(i = 0 ; i < N ; i++)
            free(p[i]);
        free(p);
    }

    void free_2d_float_array(float** p, int N) {
        int i;
        for(i = 0 ; i < N ; i++)
            free(p[i]);
        free(p);
    }

    void free_2d_double_array(double** p, int N) {
        int i;
        for(i = 0 ; i < N ; i++)
            free(p[i]);
        free(p);
    }

    void free_2d_double_complex_array(double complex** p, int N) {
        int i;
        for(i = 0 ; i < N ; i++)
            free(p[i]);
        free(p);
    }

    void free_3d_double_array(double*** p, int N, int M) {
        int i, j;
    	for (i = 0; i < N; i++) 
    	{
    		for (j = 0; j < M; j++)
    			free(p[i][j]);
    		free(p[i]);
    	}
    	free(p);
    }

    void free_3d_double_complex_array(double complex*** p, int N, int M) {
        int i, j;
    	for (i = 0; i < N; i++) 
    	{
    		for (j = 0; j < M; j++) {
    			free(p[i][j]);
            }
    		free(p[i]);
    	}
    	free(p);
    }

    float *flatten_2d_float_array(float** p, int N, int M)
    {
        int i, j;
        float *array;   
        array = malloc(N*M*sizeof(float));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i*M + j] = p[i][j];
        return array;
    }

    double *flatten_2d_double_array(double** p, int N, int M)
    {
        int i, j;
        double *array;   
        array = malloc(N*M*sizeof(double));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i*M + j] = p[i][j];
        return array;
    }

    double complex *flatten_2d_double_complex_array(double complex** p, int N, int M)
    {
        int i, j;
        double complex *array;   
        array = malloc(N*M*sizeof(double complex));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i*M + j] = p[i][j];
        return array;
    }

    double *flatten_3d_double_array(double*** p, int N, int M, int L)
    {
        int i, j, k;
        double *array;   
        array = malloc(N*M*L*sizeof(double));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i*M*L + j*L + k] = p[i][j][k];
        return array;
    }

    double complex *flatten_3d_double_complex_array(double complex*** p, int N, int M, int L)
    {
        int i, j, k;
        double complex *array;    
        array = malloc(N*M*L*sizeof(double complex));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i*M*L + j*L + k] = p[i][j][k];
        return array;
    }

    float **deflate_2d_float_array(float* p, int N, int M)
    {
        int i, j;
        float **array;     
        array = malloc(N*sizeof(float *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(float) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = p[i*M + j];
        return array;
    }

    double **deflate_2d_double_array(double* p, int N, int M)
    {
        int i, j;
        double **array;     
        array = malloc(N*sizeof(double *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = p[i*M + j];
        return array;
    }

    double complex **deflate_2d_double_complex_array(double complex* p, int N, int M)
    {
        int i, j;
        double complex **array;     
        array = malloc(N*sizeof(double complex *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double complex) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = p[i*M + j];
        return array;
    }

    double ***deflate_3d_double_array(double* p, int N, int M, int L)
    {
        int i, j, k;
        double ***array;     
        array = malloc(N*sizeof(double **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double*) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = malloc( L*sizeof(double) );
    				if (array[i][j] == NULL) {
    				fprintf(stderr, "Out of memory");
    				exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = p[i*M*L + j*L + k];
        return array;
    }

    double complex ***deflate_3d_double_complex_array(double complex* p, int N, int M, int L)
    {
        int i, j, k;
        double complex ***array;     
        array = malloc(N*sizeof(double complex **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double complex *) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = malloc( L*sizeof(double complex) );
    				if (array[i][j] == NULL) {
    				fprintf(stderr, "Out of memory");
    				exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = p[i*M*L + j*L + k];
        return array;
    }

    void fill_1d_int_array(int* p, int dim, int value) {
        int i;
        for(i = 0 ; i < dim ; i++)
            p[i] = value;
    }

    double *create_and_copy_1d_double_array(int a_col_in ,double* a_in) // Allocate the array
    {
        int i;
        double *array;    
        array = malloc(a_col_in*sizeof(double));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < a_col_in ; i++)
            array[i] = a_in[i];
        return array;
    }

    void copy_1d_double_complex_array(int a_col_in ,double complex* a_in, double complex* b_o)
    {
        int i;
        //int N = sizeof (a) / sizeof (a[0]);

        for(i = 0 ; i < a_col_in ; i++)
            b_o[i] = a_in[i];
    }

    void copy_2d_double_complex_array(int a_row_in ,int a_col_in ,double complex** a_in ,double complex** b_o)
    {
        int i, j;
        //int N = sizeof (a) / sizeof (a[0]);
    	//int M = sizeof (a) / sizeof (a[1]);

        for(i = 0 ; i < a_row_in ; i++)
            for(j = 0 ; j < a_col_in ; j++)
                b_o[i][j] = a_in[i][j];
    }

    double **create_and_copy_2d_double_array(int N ,int M ,double** a_in)
    {
        int i, j;
        //int N = sizeof (a) / sizeof (a[0]);
    	//int M = sizeof (a) / sizeof (a[1]);        
        double **array;    
        array = malloc(N*sizeof(double *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = a_in[i][j];

        return array;
    }

    double complex ***create_and_copy_3d_double_complex_array(int N ,int M ,int L ,double complex*** a_in)
    {
        int i, j, k;
        //int N = sizeof (a) / sizeof (a[0]);
    	//int M = sizeof (a) / sizeof (a[1]);        
        double complex ***array;    
        array = malloc(N*sizeof(double complex **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double complex*) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = malloc( L*sizeof(double complex) );
    			if (array[i][j] == NULL) {
    			fprintf(stderr, "Out of memory");
    			exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = a_in[i][j][k];

        return array;
    }

    /*double int_to_double(int input)
    {
        double output;
        char temp[20];

        sprintf("%d", input);

        temp[19] = 0x00;

        sscanf("%lf", &output);

        return output;
    }*/

    void np_dot_2d_double(int a_row_in ,int a_col_in ,int b_col_in ,double** a_in, double** b_in, double** c_o)
    {
        // usage:
        // c[N][L] = dot product (a[N][M], b[M][L])
        int i, j, k;
        //int N = sizeof (a) / sizeof (a[0]);
        //int M = sizeof (a) / sizeof (a[1]);
        //int L = sizeof (b) / sizeof (b[1]);
        double s=0;
        for(i = 0 ; i < a_row_in ; i++) {
            for(j = 0 ; j < b_col_in ; j++) {
                s = 0;
                for(k = 0 ; k < a_col_in ; k++) {
                    s += a_in[i][k]*b_in[k][j];
                }
                c_o[i][j] = s;
            }
        }
    }

    void np_dot_2d_double_complex(int a_row_in ,int a_col_in ,int b_col_in ,double complex** a_in, double complex** b_in, double complex** c_o)
    {
        // usage:
        // c[N][L] = dot product (a[N][M], b[M][L])
        int i, j, k;
        //int N = sizeof (a) / sizeof (a[0]);
        //int M = sizeof (a) / sizeof (a[1]);
        //int L = sizeof (b) / sizeof (b[1]);
        double complex s=0;
        for(i = 0 ; i < a_row_in ; i++) {
            for(j = 0 ; j < b_col_in ; j++) {
                s = 0;
                for(k = 0 ; k < a_col_in ; k++) {
                    s += a_in[i][k]*b_in[k][j];
                }
                c_o[i][j] = s;
            }
        }
    }

    int np_argmax_1d_int(int *a, int n)
    {
        if(n <= 0) return -1;
        int i, max_i = 0;
        int max = a[0];
        for(i = 1; i < n; ++i){
            if(a[i] > max){
                max = a[i];
                max_i = i;
            }
        }
        return max_i;
    }

    int np_argmax_1d_float(float *a, int n)
    {
        if(n <= 0) return -1;
        int i, max_i = 0;
        float max = a[0];
        for(i = 1; i < n; ++i){
            if(a[i] > max){
                max = a[i];
                max_i = i;
            }
        }
        return max_i;
    }

    float np_max_1d_float(float *a, int n)
    {
        if(n <= 0) return -1;
        int i;
        float max = a[0];
        for(i = 1; i < n; ++i){
            if(a[i] > max){
                max = a[i];
            }
        }
        return max;
    }

    double np_linalg_norm_double(double a, double b)
    {
        return sqrt(pow(a, 2) + pow(b, 2));
    }

    /*float *scipy_signal_hanning_float(int N ,short itype) 
    {
        // usage:
            //itype = 1 --> periodic
            //itype = 0 --> symmetric= numpy.hanning
            //default itype=0 (symmetric)

        int half, i, idx, n;
        float *w = create_1d_float_array(N);

        //w = (float*) calloc(N, sizeof(float));
        //memset(w, 0, N*sizeof(float));

        if(itype==1)    //periodic function
            n = N-1;
        else
            n = N;

        if(n%2==0)
        {
            half = n/2;
            for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                w[i] = 0.5 * (1 - cos(2*PI*(i+1) / (n+1)));

            idx = half-1;
            for(i=half; i<n; i++) {
                w[i] = w[idx];
                idx--;
            }
        }
        else
        {
            half = (n+1)/2;
            for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                w[i] = 0.5 * (1 - cos(2*PI*(i+1) / (n+1)));

            idx = half-2;
            for(i=half; i<n; i++) {
                w[i] = w[idx];
                idx--;
            }
        }

        if(itype==1)    //periodic function
        {
            for(i=N-1; i>=1; i--)
                w[i] = w[i-1];
            w[0] = 0.0;
        }
        return(w);
    }

    double *scipy_signal_hanning_double(int N ,short itype) 
    {
        // usage:
            //itype = 1 --> periodic
            //itype = 0 --> symmetric= numpy.hanning
            //default itype=0 (symmetric)

        int half, i, idx, n;
        double *w = create_1d_double_array(N);

        //w = (float*) calloc(N, sizeof(float));
        //memset(w, 0, N*sizeof(float));

        if(itype==1)    //periodic function
            n = N-1;
        else
            n = N;

        if(n%2==0)
        {
            half = n/2;
            for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                w[i] = 0.5 * (1 - cos(2*PI*(i+1) / (n+1)));

            idx = half-1;
            for(i=half; i<n; i++) {
                w[i] = w[idx];
                idx--;
            }
        }
        else
        {
            half = (n+1)/2;
            for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                w[i] = 0.5 * (1 - cos(2*PI*(i+1) / (n+1)));

            idx = half-2;
            for(i=half; i<n; i++) {
                w[i] = w[idx];
                idx--;
            }
        }

        if(itype==1)    //periodic function
        {
            for(i=N-1; i>=1; i--)
                w[i] = w[i-1];
            w[0] = 0.0;
        }
        return(w);
    }*/

    void np_hanning_double(int M_i, double* np_hanning_o) 
    {
        int i;
        for(i = 0; i < M_i; i++) {
            np_hanning_o[i] = 0.5*(1 - cos(2*PI*i/(M_i - 1)));
        }
    }

    void scipy_signal_get_window_hann_double(int nx_i ,double* win_o) 
    {
        int i;
        for(i = 0; i < nx_i; i++) {
            win_o[i] = 0.5*(1 - cos(2*PI*(1.0*i)/(1.0*nx_i)));
        }
    }

    void np_pad_reflect_1d_double(double* a_i ,int a_row_i ,int pad_width_i ,double* b_o) 
    {
        int i;
        int j = pad_width_i+a_row_i; //2*pad_width_i; 
        int k = a_row_i - 2;
        for(i = 0; i < pad_width_i; i++) {        
            //b_o[i] = a_i[j-i];               
            b_o[i] = a_i[pad_width_i-i];        
            //b_o[pad_width_i+a_row_i+i] = a_i[k-i];        
            b_o[j+i] = a_i[k-i];
        }

        for(i = 0; i < a_row_i; i++) {        
            b_o[pad_width_i+i] = a_i[i];
        }
    }

    double *np_linspace_double(double start ,double stop ,int num) 
    {
        double *line = create_1d_double_array(num);
        double delta = (stop-start)/(num*1.0-1.0);
        for (int i=0; i<num; i++) {
                line[i] = start + (1.0*i*delta);
        }
        return line;
    }

    double *np_diff_1d_double(double *a_i ,int a_row_i) 
    {
        int b_row = a_row_i - 1;
        double *b = create_1d_double_array(b_row);
        for (int i=0; i<b_row; i++) {
                b[i] = a_i[i+1] - a_i[i];
        }
        return b;
    }

    double **np_subtract_outer_1d_double_in(double *a_i ,int a_row_i ,double *b_i ,int b_row_i) 
    {
        double **c = create_2d_double_array(a_row_i, b_row_i);
        for (int i=0; i<a_row_i; i++) {
            for (int j=0; j<b_row_i; j++) {
                c[i][j] = a_i[i] - b_i[j];
            }
        }
        return c;
    }

    /*
    double complex Norm2_1d_double_complex(int row_in ,double complex *a)
    {
    	int i;
        //int row_num = sizeof (a) / sizeof (a[0]);
        double complex norm2_ans = 0;

    	for (i = 0; i < row_in; i++)
    		norm2_ans+=a[i]*a[i];
    	//norm2_ans = sqrt(norm2_ans);
        norm2_ans = sqrt(pow(creal(norm2_ans), 2) + pow(cimag(norm2_ans), 2));
    	return norm2_ans;
    }

    void qr_decomposition_square_double_complex( int row_in
                                                ,double complex **A_in
                                                ,double complex **Q_o
                                                ,double complex **R_o)
    {
    	int i, j, k, i1, i2;
        //int row_num = sizeof (A) / sizeof (A[0]);
        double complex *col_A = create_1d_double_complex_array(row_in);
        double complex *col_Q = create_1d_double_complex_array(row_in);
        double complex temp   = 0;

    	//施密特正交化
    	for (j = 0; j < row_in; j++)
    	{
    		for (i = 0; i < row_in; i++) { //把A的第j列存入col_A中		
    			col_A[i] = A_in[i][j];
    			col_Q[i] = A_in[i][j];
    		}
    		for (k = 0; k < j; k++) { //計算第j列以前		
    			R_o[k][j] = 0;
    			for (i1 = 0; i1 < row_in; i1++) { //R=Q'A(Q'即Q的轉置) 即Q的第k列和A的第j列做內積
    				R_o[k][j] += col_A[i1]*Q_o[i1][k];//Q的第k列
    			}
    			for (i2 = 0; i2 < row_in; i2++) {
    				col_Q[i2] -= R_o[k][j]*Q_o[i2][k];
    			}
    		}
    
    		temp = Norm2_1d_double_complex(row_in, col_Q);
    		R_o[j][j] = temp;
    		for (i = 0; i < row_in; i++) {
    			//單位化Q
    			Q_o[i][j] = col_Q[i]/temp;
    		}
    	}

    	free(col_A);
    	free(col_Q);
    }

    void eigenvector_from_eigenvalue_square_double_complex(  int row_in
                                                            ,double complex **A_in
                                                            ,double complex *eigenvalue_in
                                                            ,double complex **eigenvector_o)
    {
    	int count, i, j, i1, j1, i2, j2;
        //int row_num = sizeof (A) / sizeof (A[0]);
        double complex eValue=0, coe=0, sum1=0, sum2=0;
    	double complex **temp = create_2d_double_complex_array(row_in, row_in);
    
        //CopyMatrix(A, &temp);
    	for (count = 0; count < row_in; count++) {
    		eValue = eigenvalue_in[count]; //當前的特徵值
    		copy_2d_double_complex_array(A_in, &temp[0]); //這個每次都要重新複製，因為後面會破壞原矩陣(剛開始沒注意到這個找bug找了好久。。)		
            for (i = 0; i < row_in; i++) {
    			temp[i][i] -= eValue;
    		}        
    		//將temp化為階梯型矩陣(歸一性)對角線值為一
    		for (i = 0; i < row_in - 1; i++) {
    			coe = temp[i][i];            
    			for (j = i; j < row_in; j++) {
    				temp[i][j] /= coe; //讓對角線值為一
    			}            
    			for (i1 = i + 1; i1 < row_in; i1++) {
    				coe = temp[i1][i];				
                    for (j1 = i; j1 < row_in; j1++) {
    					temp[i1][j1] -= coe*temp[i][j1];
    				}
    			}
    		}
    		//讓最後一行為1
    		sum1 = eigenvector_o[row_in-1][count] = 1;
    		for (i2 = row_in - 2; i2 >= 0; i2--) {
    			sum2 = 0;
    			for (j2 = i2 + 1; j2 < row_in; j2++) {
    				sum2 += temp[i2][j2]*eigenvector_o[j2][count];
    			}
    			sum2 = -sum2/temp[i2][i2];
    			sum1 += sum2*sum2;
    			eigenvector_o[i2][count] = sum2;
    		}
    		//sum1 = sqrt(sum1); //當前列的模
    		for (i = 0; i < row_in; i++) {
    			//單位化
    			eigenvector_o[i][count] /= sum1;
    		}
    	}
    
        free_2d_double_complex_array(temp, row_in);
    }

    void np_linalg_eig_double_complex(   int row_in
                                        ,int max_iteration_in // > 0, 最大迭代次數，讓資料更準確
                                        ,double complex **A_in
                                        ,double complex *eigenvalue_o
                                        ,double complex **eigenvector_o)
    {
        //https://www.itread01.com/content/1549432108.html
        int i;
        //int row_num = sizeof (A) / sizeof (A[0]);
        double complex **temp = create_2d_double_complex_array(row_in, row_in);
        double complex **temp_q = create_2d_double_complex_array(row_in, row_in);
        double complex **temp_r = create_2d_double_complex_array(row_in, row_in);

        copy_2d_double_complex_array(A_in, &temp[0]);
        //使用QR分解求矩陣特徵值
    	for (i = 0; i < max_iteration_in; ++i) {
            qr_decomposition_square_double_complex(row_in ,temp, &temp_q[0], &temp_r[0]);
            np_dot_2d_double_complex(temp_r, temp_q, &temp[0]);
        }
        for (i = 0; i < row_in; ++i) {
            eigenvalue_o[i] = temp[i][i];
        }

    	eigenvector_from_eigenvalue_square_double_complex( row_in ,A_in ,eigenvalue_o ,&eigenvector_o[0]);

        free_2d_double_complex_array(temp, row_in);
        free_2d_double_complex_array(temp_q, row_in);
        free_2d_double_complex_array(temp_r, row_in);
    }*/

    double norm_double(double A_in) // =norm() in C++
    {
        return pow(A_in, 2);
    }

    double norm_double_complex(double complex A_in) // =norm() in C++
    {
        return pow(creal(A_in), 2) + pow(cimag(A_in), 2);
    }

    void matSca_2d_double_complex_array(int a_row_in ,int a_col_in ,double complex scalar_in ,double complex** A_in ,double complex** C_o) // Scalar multiple of matrix
    {
        //int m = A.size(),   n = A[0].size();
        int i, j;
        //matrix C = A;
        for (i = 0; i < a_row_in; i++ )
        {
            for (j = 0; j < a_col_in; j++ )
                C_o[i][j] = scalar_in*A_in[i][j]; //C[i][j] *= c;
        }
        //return C_o;
    }

    void matLin_2d_double_complex_array( int a_row_in ,int a_col_in ,double complex a_scalar_in ,double complex** A_in 
                                        ,double complex b_scalar_in ,double complex** B_in ,double complex** C_o)  // Linear combination of matrices
    {
        //int m = A.size(),   n = A[0].size();   assert( B.size() == m && B[0].size() == n );
        int i, j;
        matSca_2d_double_complex_array(a_row_in ,a_col_in ,a_scalar_in ,A_in ,&C_o[0]); //matSca( a, A );  

        for (i = 0; i < a_row_in; i++ )
        {
            for (j = 0; j < a_col_in; j++ )
                C_o[i][j] += b_scalar_in*B_in[i][j];
        }
        //return C_o;
    }

    double matNorm_1d_double_complex_array( int a_col_in ,double complex* A_in) // Complex vector norm
    {
        //int m = A.size();
        int i;
        double result = 0.0;
        for (i = 0; i < a_col_in; i++ )
        {
            result += norm_double_complex(A_in[i]); //norm( A[i][j] );
        }
        return sqrt( result );
    }

    double matNorm_2d_double_array( int a_row_in ,int a_col_in ,double** A_in) // matrix norm
    {
        //int m = A.size(),   n = A[0].size();
        int i, j;
        double result = 0.0, norm = 0;
        for (i = 0; i < a_row_in; i++ )
        {
            for (j = 0; j < a_col_in; j++ ) {
                norm = norm_double(A_in[i][j]);
                result += norm;
                //printf("\nc matNorm, result= %.8lf", result);
            }
        }
        return sqrt( result );
    }

    double matNorm_2d_double_complex_array( int a_row_in ,int a_col_in ,double complex** A_in) // Complex matrix norm
    {
        //int m = A.size(),   n = A[0].size();
        int i, j;
        double result = 0.0, norm = 0;
        for (i = 0; i < a_row_in; i++ )
        {
            for (j = 0; j < a_col_in; j++ ) {
                norm = norm_double_complex(A_in[i][j]);
                result += norm; //norm( A[i][j] );
                //printf("\nc matNorm, result= %.8lf", result);
            }
        }
        return sqrt( result );
    }

    double subNorm_square_double_complex_array( int a_row_in ,double complex** A_in) // Below leading diagonal of square matrix
    {
        //int n = T.size();   assert( T[0].size() == n );
        int i, j;
        double result = 0.0;
        for (i = 1; i < a_row_in; i++ )
        {
            for (j = 0; j < i; j++ ) result += norm_double_complex(A_in[i][j]); //norm( T[i][j] );
        }
        return sqrt( result );
    }

    /*double complex shift_square_double_complex_array(int row_in ,double complex** A_in) // Wilkinson shift in QR algorithm
    {
        //int N = A.size();
        int i = row_in - 1;
        double e = 0, f = 0;
        double complex a = 0, b = 0, c = 0, d = 0, delta = 0, s1 = 0, s2 = 0, s = 0;
    //  while ( i > 0 && abs( A[i][i-1] ) < NEARZERO ) i--;     // Deflation (not sure about this)

        s = 0.0;
        if ( i > 0 )
        {
           a = A_in[i-1][i-1];
           b = A_in[i-1][i];
           c = A_in[i][i-1];
           d = A_in[i][i];        // Bottom-right elements
           delta = csqrt( ( a + d ) * ( a + d ) - 4.0 * ( a * d - b * c ) ); 
           s1 = 0.5 * ( a + d + delta );
           s2 = 0.5 * ( a + d - delta );
           e = norm_double_complex(s1 - d);
           f = norm_double_complex(s2 - d);
           s = ( e < f ? s1 : s2 ); //s = ( norm( s1 - d ) < norm( s2 - d ) ? s1 : s2 );
        }
        return s;
    }

    void Hessenberg_square_double_complex_array( int row_in 
                                                ,double complex** A_in 
                                                ,double complex** P_o 
                                                ,double complex** H_o)
    //http://www.cplusplus.com/forum/beginner/220486/2/
    // Reduce A to Hessenberg form A = P H P-1 where P is unitary, H is Hessenberg
    //                             i.e. P-1 A P = H
    // A Hessenberg matrix is upper triangular plus single non-zero diagonal below main diagonal
    {
        //int N = A.size();
        int i, j, k;
        double xlength = 0, axk = 0, ulength = 0;
        double complex rho = 1, xk = 0;
        double complex *U = create_1d_double_complex_array(row_in);
        double complex **P_tmp = create_2d_double_complex_identity_array(row_in); //P = identity( N );
        double complex **PK = create_2d_double_complex_identity_array(row_in);
        double complex **H_tmp; //H = A;    
        H_tmp = malloc(row_in*sizeof(double complex *));    
        if (H_tmp == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}    
        for(i = 0 ; i < row_in ; i++)
        {
            H_tmp[i] = malloc( row_in*sizeof(double complex) );
            if (H_tmp[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }
        for(i = 0 ; i < row_in ; i++) //H = A;
            for(j = 0 ; j < row_in ; j++)
                H_tmp[i][j] = A_in[i][j];

        for (k = 0; k < row_in - 2; k++ )             // k is the working column
        {
            // X vector, based on the elements from k+1 down in the kth column
            for (i = k + 1; i < row_in; i++ ) xlength += norm_double_complex(H_tmp[i][k]); //norm( H[i][k] );
            xlength = sqrt(xlength); 

            // U vector ( normalise X - rho.|x|.e_k )
            //vec U( N, 0.0 );
            xk = H_tmp[k+1][k];
            axk = cabs(xk);
            if ( axk > NEARZERO ) rho = -xk / axk;
            U[k+1] = xk - rho * xlength;
            ulength = norm_double_complex(U[k+1]); //norm( U[k+1] );
            for (i = k + 2; i < row_in; i++ )
            {
               U[i] = H_tmp[i][k];
               ulength += norm_double_complex(U[i]); //norm( U[i] );
            }
            ulength = max( sqrt( ulength ), SMALL );
            for ( i = k + 1; i < row_in; i++ ) U[i] /= ulength;   
            // Householder matrix: P = I - 2 U U*T
            //matrix PK = identity( N );
            for(i = 0 ; i < row_in ; i++) {
                for(j = 0 ; j < row_in ; j++) {
                    if (i == j)
                        PK[i][j] = 1;
                    else
                        PK[i][j] = 0;
                }
            }

            for (i = k + 1; i < row_in; i++ )
            {
               for (j = k + 1; j < row_in; j++ ) PK[i][j] -= 2.0 * U[i] * conj( U[j] );
            }    
            // Transform as PK*T H PK.   Note: PK is unitary, so PK*T = P
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,H_tmp ,PK ,&P_o[0]); //H = matMul( PK, matMul( H, PK ) );
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,PK ,P_o ,&H_o[0]); //H = matMul( PK, matMul( H, PK ) );
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,P_tmp ,PK ,&P_o[0]); //P = matMul( P, PK );
            for(i = 0 ; i < row_in ; i++)
                for(j = 0 ; j < row_in ; j++)
                    H_tmp[i][j] = H_o[i][j];

            for(i = 0 ; i < row_in ; i++)
                for(j = 0 ; j < row_in ; j++)
                    P_tmp[i][j] = P_o[i][j];
        }
        //#ifdef DEBUG_np_linalg_eig_double_complex
        //    printf("\nc Hessenberg, P_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f%+.14fj", creal(P_o[i][j]), cimag(P_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //        // Check matrix norm of   A v - lambda v
        //        //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
        //    }
        //    
        //    printf("\nc Hessenberg, H_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f%+.14fj", creal(H_o[i][j]), cimag(H_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //        // Check matrix norm of   A v - lambda v
        //        //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
        //    }
        //#endif    

        free(U);
        free_2d_double_complex_array(P_tmp, row_in);
        free_2d_double_complex_array(PK, row_in);
        free_2d_double_complex_array(H_tmp, row_in);
    }

    void QRFactoriseGivens_square_double_complex_array(  int row_in 
                                                        ,double complex** A_in 
                                                        ,double complex** Q_o 
                                                        ,double complex** R_o)
    {
        // Factorises a Hessenberg matrix A as QR, where Q is unitary and R is upper triangular
        // Uses N-1 Givens rotations
        //int N = A.size(); 
        int i, j, k, m;
        double length = 1;
        double complex c = 0, s = 0, c_tmp = 0, s_tmp = 0, cstar = 0, sstar = 0;
        double complex **RR = create_2d_double_complex_array(row_in ,row_in);
        double complex **QQ = create_2d_double_complex_identity_array(row_in);   
        for(i = 0 ; i < row_in ; i++) //R = A;
            for(j = 0 ; j < row_in ; j++)
                R_o[i][j] = A_in[i][j];  

        for(i = 0 ; i < row_in ; i++) { //Q = identity( N );
            for(j = 0 ; j < row_in ; j++) {
                if (i == j)
                    Q_o[i][j] = 1;
                else
                    Q_o[i][j] = 0;
            }
        }

        for (i = 1; i < row_in; i++ )       // i is the row number
        {
            j = i - 1;                   // aiming to zero the element one place below the diagonal
            if (cabs( R_o[i][j] ) < SMALL ) continue;  
            // Form the Givens matrix        
            c =        R_o[j][j]  ;           
            s = -conj( R_o[i][j] );                       
            c_tmp = norm_double_complex(c);                      
            s_tmp = norm_double_complex(s);                      
            length = sqrt( c_tmp + s_tmp); //sqrt( norm( c ) + norm( s ) );    
            c /= length;               
            s /= length;               
            cstar = conj( c );         //  G*T = ( c* -s )     G = (  c  s  )     <--- j
            sstar = conj( s );         //        ( s*  c )         ( -s* c* )     <--- i

            for(k = 0 ; k < row_in ; k++) //matrix RR = R;
                for(m = 0 ; m < row_in ; m++)
                    RR[k][m] = R_o[k][m];

            for(k = 0 ; k < row_in ; k++) //matrix QQ = Q;
                for(m = 0 ; m < row_in ; m++)
                    QQ[k][m] = Q_o[k][m];

            for (m = 0; m < row_in; m++ ) 
            {
                R_o[j][m] = cstar * RR[j][m] - s     * RR[i][m];
                R_o[i][m] = sstar * RR[j][m] + c     * RR[i][m];    // Should force R[i][j] = 0.0
                Q_o[m][j] = c     * QQ[m][j] - sstar * QQ[m][i];
                Q_o[m][i] = s     * QQ[m][j] + cstar * QQ[m][i];
            }
        }
        //#ifdef DEBUG_np_linalg_eig_double_complex
        //    printf("\nc QRFactoriseGivens, A_in= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f%+.14fj", creal(A_in[i][j]), cimag(A_in[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //    
        //    printf("\nc QRFactoriseGivens, Q_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f%+.14fj", creal(Q_o[i][j]), cimag(Q_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //    
        //    printf("\nc QRFactoriseGivens, R_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f%+.14fj", creal(R_o[i][j]), cimag(R_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //#endif    

        free_2d_double_complex_array(RR, row_in);
        free_2d_double_complex_array(QQ, row_in);
    }

    void QRHessenberg_square_double_complex_array(   int row_in 
                                                    ,double complex** A_in 
                                                    ,double complex** P_o 
                                                    ,double complex** T_o)
    // Apply the QR algorithm to the matrix A. 
    //
    // Multi-stage:
    //    - transform to a Hessenberg matrix
    //    - apply QR factorisation based on Givens rotations
    //    - uses (single) Wilkinson shift - double-shift version in development
    //
    // Should give a Shur decomposition A = P T P-1 where P is unitary, T is upper triangular
    //                             i.e. P-1 A P = T
    // Eigenvalues of A should be the diagonal elements of T
    // If A is hermitian T would be diagonal and the eigenvectors would be the columns of P
    {
        //const int ITERMAX = 10000;
        //const double TOLERANCE = 1.0e-10;
        int ITERMAX = 10000;
        double TOLERANCE = 1.0e-10;
    
        //int N = A.size();
        int i, j;
        int iter = 1;
        double residual = 1.0, residual_tmp = 0;
        double complex mu = 1;
        double complex **Q = create_2d_double_complex_array(row_in ,row_in);
        double complex **R = create_2d_double_complex_array(row_in ,row_in);
        double complex **Told = create_2d_double_complex_array(row_in ,row_in);
        double complex **array_identity = create_2d_double_complex_identity_array(row_in);
        double complex **P_tmp = create_2d_double_complex_array(row_in ,row_in);
        double complex **T_tmp = create_2d_double_complex_array(row_in ,row_in);
    
        //matrix Q( N, vec( N ) ), R( N, vec( N ) ), Told( N, vec( N ) );
        //matrix I = identity( N );
    
        // Stage 1: transform to Hessenberg matrix ( T = Hessenberg matrix, P = unitary transformation )
        Hessenberg_square_double_complex_array(row_in ,A_in ,&P_tmp[0] ,&T_o[0]); //Hessenberg( A, P, T );
        //#ifdef DEBUG_np_linalg_eig_double_complex
        //    printf("\nc QRHessenberg, P_tmp= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) {
        //            printf("%.14f%+.14fj", creal(P_tmp[i][j]), cimag(P_tmp[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //    
        //    printf("\nc QRHessenberg, T_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) {
        //            printf("%.14f%+.14fj", creal(T_o[i][j]), cimag(T_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //#endif 
    
        // Stage 2: apply QR factorisation (using Givens rotations)
        while( residual > TOLERANCE && iter < ITERMAX )
        {
            for(i = 0 ; i < row_in ; i++) //Told = T;
                for(j = 0 ; j < row_in ; j++)
                    Told[i][j] = T_o[i][j];
    
            // Spectral shift
            mu = shift_square_double_complex_array(row_in ,T_o); //cmplx mu = shift( T );
            if (cabs( mu ) < NEARZERO ) mu = 1.0;   // prevent unitary matrices causing a problem
            matLin_2d_double_complex_array(row_in ,row_in ,1.0 ,T_o ,-mu ,array_identity ,&T_tmp[0]); //T = matLin( 1.0, T, -mu, I );
    
            // Basic QR algorithm by Givens rotation
            QRFactoriseGivens_square_double_complex_array(row_in ,T_tmp ,&Q[0] ,&R[0]); //QRFactoriseGivens( T, Q, R );
            //#ifdef DEBUG_np_linalg_eig_double_complex
            //    printf("\nc QRHessenberg, mu= %.14f%+.14fj", creal(mu), cimag(mu)); printf("\n");
            //    printf("\nc QRHessenberg, cabs( mu )= %.8lf", cabs( mu )); printf("\n");
            //    printf("\nc QRHessenberg, matLin, T_tmp= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f%+.14fj", creal(T_tmp[i][j]), cimag(T_tmp[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    
            //    printf("\nc QRHessenberg, QRFactoriseGivens, Q= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f%+.14fj", creal(Q[i][j]), cimag(Q[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    
            //    printf("\nc QRHessenberg, QRFactoriseGivens, R= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f%+.14fj", creal(R[i][j]), cimag(R[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //#endif
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,R ,Q ,&T_tmp[0]); //T = matMul( R, Q );
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,P_tmp ,Q ,&P_o[0]); //P = matMul( P, Q );
    
            // Reverse shift
            matLin_2d_double_complex_array(row_in ,row_in ,1.0 ,T_tmp ,mu ,array_identity ,&T_o[0]); //T = matLin( 1.0, T, mu, I );
            //#ifdef DEBUG_np_linalg_eig_double_complex
            //    printf("\nc QRHessenberg, matMul( R, Q ), T_tmp= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f%+.14fj", creal(T_tmp[i][j]), cimag(T_tmp[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    
            //    printf("\nc QRHessenberg, matMul( P, Q ), P_o= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f%+.14fj", creal(P_o[i][j]), cimag(P_o[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    
            //    printf("\nc QRHessenberg, matLin, T_o= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f%+.14fj", creal(T_o[i][j]), cimag(T_o[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //#endif
    
            // Calculate residuals
            //residual = matNorm( matLin( 1.0, T, -1.0, Told ) ); // change on iteration
            matLin_2d_double_complex_array(row_in ,row_in ,1.0 ,T_o ,-1.0 ,Told ,&T_tmp[0]);
            residual = matNorm_2d_double_complex_array(row_in ,row_in ,T_tmp);
            residual_tmp = subNorm_square_double_complex_array(row_in ,T_o);
            //#ifdef DEBUG_np_linalg_eig_double_complex
            //    printf("\nc QRHessenberg, Calculate residuals, matLin, T_tmp= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f%+.14fj", creal(T_tmp[i][j]), cimag(T_tmp[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    printf("\nc QRHessenberg, Calculate residuals, residual matNorm= %.8lf", residual); printf("\n");
            //    printf("\nc QRHessenberg, residual_tmp= %.8lf", residual_tmp); printf("\n");
            //#endif
            residual += residual_tmp; //residual += subNorm( T ); // below-diagonal elements
    //      cout << "\nIteration: " << iter << "   Residual: " << residual << endl;
            iter++; 
            //#ifdef DEBUG_np_linalg_eig_double_complex
            //    printf("\nc QRHessenberg, Calculate residuals, residual= %.8lf", residual); printf("\n");
            //#endif

        }
        //cout << "\nQR iterations: " << iter << "   Residual: " << residual << endl;
        //if ( residual > TOLERANCE ) cout << "***** WARNING ***** QR algorithm not converged\n";
        if ( residual > TOLERANCE ) printf("\n***** WARNING ***** QR algorithm not converged");
        printf("\nc QRHessenberg, iter= %d", iter); printf("\n");
        printf("\nc QRHessenberg, residual= %.14f", residual); printf("\n");
        printf("\nc QRHessenberg, residual_tmp= %.14f", residual_tmp); printf("\n");

        #ifdef DEBUG_np_linalg_eig_double_complex
            printf("\nc QRHessenberg, P_o= \n");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f%+.14fj", creal(P_o[i][j]), cimag(P_o[i][j])); printf(", ");
                } 
                if (i%(row_in-1)==0) {
                    printf("\n");
                }
            }

            printf("\nc QRHessenberg, T_o= \n");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f%+.14fj", creal(T_o[i][j]), cimag(T_o[i][j])); printf(", ");
                } 
                if (i%(row_in-1)==0) {
                    printf("\n");
                }
            }
        #endif

        free_2d_double_complex_array(Q, row_in);
        free_2d_double_complex_array(R, row_in);
        free_2d_double_complex_array(Told, row_in);
        free_2d_double_complex_array(array_identity, row_in);
        free_2d_double_complex_array(P_tmp, row_in);
        free_2d_double_complex_array(T_tmp, row_in);
    }

    void eigenvectorUpper_square_double_complex_array(int row_in ,double complex** T_in ,double complex** E_o)
    // Find the eigenvectors of upper-triangular matrix T; returns them as column vectors of matrix E
    // The eigenvalues are necessarily the diagonal elements of T
    // NOTE: if there are repeated eigenvalues, then THERE MAY NOT BE N EIGENVECTORS
    {
        //bool fullset = true;
        //int N = T.size();
        //E = matrix( N, vec( N, 0.0 ) );               // Columns of E will hold the eigenvectors    
        int i, j, k, L, ok = 1;
        double length = 1;
        double complex lambda = 0;
        double complex *V = create_1d_double_complex_array(row_in);
        double complex **TT = create_2d_double_complex_array(row_in ,row_in);

        for(i = 0 ; i < row_in ; i++) //matrix TT = T;
            for(j = 0 ; j < row_in ; j++)
                TT[i][j] = T_in[i][j];

        for (L = row_in - 1; L >= 0; L-- )            // find Lth eigenvector, working from the bottom
        {
            ok = 1; //bool ok = true;
            lambda = T_in[L][L]; //cmplx lambda = T[L][L];
            for (k = 0; k < row_in; k++ ) {
                V[k] = 0; //vec V( N, 0.0 );
                TT[k][k] = T_in[k][k] - lambda; // TT = T - lambda I
            } // Solve TT.V = 0
            V[L] = 1.0;                                // free choice of this component
            for (i = L - 1; i >= 0; i-- )         // back-substitute for other components
            {
                V[i] = 0.0;
                for (j = i + 1; j <= L; j++ ) V[i] -= TT[i][j] * V[j];
                if ( cabs( TT[i][i] ) < NEARZERO )       // problem with repeated eigenvalues
                {
                    if ( cabs( V[i] ) > NEARZERO ) ok = 0; //false;     // incomplete set; use the lower-L one only
                    V[i] = 0.0;
                }
                else
                {
                    V[i] = V[i] / TT[i][i];
                }
            }    
            if ( ok==1 )
            {
                // Normalise
                length = matNorm_1d_double_complex_array(row_in ,V); //double length = vecNorm( V );    
                for (i = 0; i <= L; i++ ) E_o[i][L] = V[i] / length;
            }
            else
            {
                //fullset = false;
                for (i = 0; i <= L; i++ ) E_o[i][L] = 0.0;
            }
        }   
        //if ( !fullset )
        //{
        //    cout << "\n***** WARNING ***** Can't find N independent eigenvectors\n";
        //    cout << "   Some will be set to zero\n";
        //}
        if (ok==0) printf("\n***** WARNING ***** Can't find N independent eigenvectors, some will be set to zero"); 

        #ifdef DEBUG_np_linalg_eig_double_complex
            printf("\nc eigenvectorUpper, E_o= \n");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f%+.14fj", creal(E_o[i][j]), cimag(E_o[i][j])); printf(", ");
                } 
                if (i%(row_in-1)==0) {
                    printf("\n");
                }
                // Check matrix norm of   A v - lambda v
                //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
            }
        #endif  

        free(V);
        free_2d_double_complex_array(TT, row_in);
        //return fullset;
    }

    void np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double complex **A_in
                                                    ,double complex *eigenvalue_o
                                                    ,double complex **eigenvector_o)
    {   
        int i, j;
        double complex **P_o = create_2d_double_complex_array(row_in ,row_in);
        double complex **T_o = create_2d_double_complex_array(row_in ,row_in);
        double complex **E_o = create_2d_double_complex_array(row_in ,row_in);

        QRHessenberg_square_double_complex_array(row_in ,A_in ,&P_o[0] ,&T_o[0]);
        eigenvectorUpper_square_double_complex_array(row_in ,T_o ,&E_o[0]); //,&eigenvector_o[0]); //,&E_o[0]);
        np_dot_2d_double_complex(row_in ,row_in ,row_in ,P_o ,E_o ,&eigenvector_o[0]); //matMul( P, E );

        #ifdef DEBUG_np_linalg_eig_double_complex
            printf("\nsub c eigenvalue_o= ");
        #endif
        for (i = 0; i < row_in; i++) {
            eigenvalue_o[i] = T_o[i][i];
            #ifdef DEBUG_np_linalg_eig_double_complex
                printf("\n");
                printf("%.14f%+.14fj", creal(eigenvalue_o[i]), cimag(eigenvalue_o[i])); printf(", ");
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
            #endif
        }

        #ifdef DEBUG_np_linalg_eig_double_complex
            printf("\nsub c eigenvector_o E_o= ");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f%+.14fj", creal(E_o[i][j]), cimag(E_o[i][j])); printf(", ");
                } 
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
                // Check matrix norm of   A v - lambda v
                //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
            }

            printf("\nsub c eigenvector_o= ");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f%+.14fj", creal(eigenvector_o[i][j]), cimag(eigenvector_o[i][j])); printf(", ");
                } 
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
                // Check matrix norm of   A v - lambda v
                //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
            }
        #endif

        free_2d_double_complex_array(P_o, row_in);
        free_2d_double_complex_array(T_o, row_in);
        //free_2d_double_complex_array(E_o, row_in);
    }*/

    void np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double complex **A_in
                                                    ,double complex *eigenvalue_o
                                                    ,double complex **eigenvector_o)
    {
        //int i;
       // double complex **A_in = deflate_2d_double_complex_array(flat_A_in, row_in, row_in);
       // double complex **eigenvector_o = create_2d_double_complex_array(row_in, row_in);
        double complex *flat_A_in = flatten_2d_double_complex_array(A_in, row_in, row_in);
        double complex *flat_eigenvector_o = create_1d_double_complex_array(row_in*row_in);
    
        /*printf("\nc np A_in= ");
        for (int i = 0; i < row_in; i++) {
            for (int j = 0; j < row_in; j++) {
                printf("%.14f%+.14fj", creal(A_in[i][j]), cimag(A_in[i][j])); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
            }
        }
        printf("\nc np flat_A_in= ");
        for (int i = 0; i < row_in*row_in; i++) {
            printf("%.14f%+.14fj", creal(flat_A_in[i]), cimag(flat_A_in[i])); printf(", ");
            if ((i+1)%2==0) {
                printf("\n");
            }
        }*/
       // #ifdef DEBUG_np_linalg_eig_double_complex
            //clock_t time0, time1;
            //time0 = clock();
       // #endif    
        flat_np_linalg_eig_square_double_complex_array(   row_in
                                                    ,flat_A_in //,A_in
                                                    ,&eigenvalue_o[0]
                                                    ,&flat_eigenvector_o[0] //,&eigenvector_o[0]
                                                    );
       // #ifdef DEBUG_np_linalg_eig_double_complex
            //time1 = clock();
            //printf("\n spe c np_linalg_eig time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
       // #endif

        #ifdef DEBUG_np_linalg_eig_double_complex
            //printf("\nflat_A_in[0]*flat_A_in[1]= %.14f%+.14fj", creal(flat_A_in[0]*flat_A_in[1]), cimag(flat_A_in[0]*flat_A_in[1]));
            //printf("\nflat_A_in[0]/flat_A_in[1]= %.14f%+.14fj", creal(flat_A_in[0]/flat_A_in[1]), cimag(flat_A_in[0]/flat_A_in[1]));
            //printf("\nflat_A_in[0]^2= %.14f%+.14fj", creal(cpow(flat_A_in[0],2)), cimag(cpow(flat_A_in[0],2)));
           /* printf("\nc square array in= \n");
            for (int i = 0; i < row_in; i++) {
                for (int j = 0; j < row_in; j++) {
                    printf("%.14f%+.14fj", creal(A_in[i][j]), cimag(A_in[i][j])); printf(", ");
                    if (j%row_in==row_in-1) {
                        printf("\n");
                    }
                }
            }*/
            printf("\nc eigenvalue_o= ");
            for (int i = 0; i < row_in; i++) {
                printf("%.14f%+.14fj", creal(eigenvalue_o[i]), cimag(eigenvalue_o[i])); printf(", ");
            }   
        #endif

        for (int i = 0; i < row_in; i++) {
            for (int j = 0; j < row_in; j++) {
                eigenvector_o[i][j] = flat_eigenvector_o[i*row_in+j];
                #ifdef DEBUG_np_linalg_eig_double_complex
                    printf("%.14f%+.14fj", creal(eigenvector_o[i][j]), cimag(eigenvector_o[i][j])); printf(", ");
                    if (j%row_in==row_in-1) {
                        printf("\n");
                    }
                #endif
            }
        }

        free(flat_A_in);
        free(flat_eigenvector_o);

    }

    void stable_sort_1d_double(int col_i ,double* p_i ,int* stable_sort_o)
    {
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, j;
        int min = 0, m = 0;
        double compare = 0, key = 0;
        double *arr_sort = create_and_copy_1d_double_array(col_i ,p_i);

        for (i = 0; i < col_i; i++)    
    		stable_sort_o[i] = i;     

        //https://www.geeksforgeeks.org/stable-selection-sort/
        for (i = 0; i < col_i - 1; i++) {  
            // Find minimum element from arr[i] to arr[n - 1]. 
            min = i;
            //#ifdef DEBUG_np_lexsort_2d_double
            //    printf("\nsub c step 2.1 i= %d", i); 
            //#endif
            for (j = i + 1; j < col_i; j++) {
                // if (flag == 1) {
                    compare = arr_sort[min] - arr_sort[j];
                // }
                // else {
                //     compare = arr_sort[j] - arr_sort[min];
                // }

                if (compare > 0)
                    min = j;
            }

            // Move minimum element at current i. 
            key = arr_sort[min];
            m = stable_sort_o[min];
            while (min > i)  
            { 
                arr_sort[min] = arr_sort[min - 1]; 
                stable_sort_o[min] = stable_sort_o[min - 1];
                min--; 
            } 
            arr_sort[i] = key;
            stable_sort_o[i] = m; 
        }

        free(arr_sort);

    }

    //void np_argsort_1d_double_complex(int size ,double complex* p_in ,int flag ,double complex* sort_o ,int* argsort_o)
    void np_argsort_1d_double_complex(int size ,double complex* p_in ,int* argsort_o)
    {
        // #ifdef DEBUG_np_lexsort_2d_double
            clock_t time0, time1;
            time0 = clock();
        // #endif
    
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, N=2, M=size;
        //int size = sizeof (p) / sizeof (p[0]);

        //copy_1d_double_complex_array( p, sort_o);
        double **array;    
        array = malloc(N*sizeof(double complex *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = malloc( M*sizeof(double complex) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < M ; i++) {
            array[0][i] = cimag(p_in[i]);
            array[1][i] = creal(p_in[i]);
        }

    	np_lexsort_2d_double_v2(N ,M ,array ,&argsort_o[0]);

        //for (i = 0; i < size - 1; i++) {
    	//	k = i;
    	//	for (j = i + 1; j < size; j++) {
    	//		if (flag == 1) {
    	//			if (sort_o[k] > sort_o[j]) {
    	//				k = j;
    	//			}
    	//		}
    	//		else {
    	//			if (sort_o[k] < sort_o[j]) {
    	//				k = j;
    	//			}
    	//		}
    	//	}
    	//	if (k != i) {			
    	//		temp = sort_o[i];
    	//		sort_o[i] = sort_o[k];
    	//		sort_o[k] = temp;			
    	//		index = argsort_o[i];
    	//		argsort_o[i] = argsort_o[k];
    	//		argsort_o[k] = index;
    	//	}
    	//}
        #ifdef DEBUG_np_lexsort_2d_double
            printf("\nsub c argsort_o= ");
            for (i = 0; i < M; i++) {
                printf("%d", argsort_o[i]); printf(", ");
            }
        #endif

        //long vmrss, vmsize;
        //get_memory_usage_kb(&vmrss, &vmsize);
        //printf("\nspe c v1 Current memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n", vmrss, vmsize);

        free_2d_double_array(array, N);
    
        // #ifdef DEBUG_np_lexsort_2d_double
            time1 = clock();
            printf("\n spe c v1 np_argsort time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
        // #endif
    
    }

    /*
    //void np_lexsort_2d_double(double** p ,int flag ,int* lexsort_o)
    void np_lexsort_2d_double(double** p ,int* lexsort_o)
    {
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, j, k, m=0, max_index_eq=0, min=0, has_eq_element=0, cnt_index_eq=1;
        int row = sizeof (p) / sizeof (p[0]);
        int col = sizeof (p) / sizeof (p[1]);
        double key=0, compare=0;
        double *pri_sort_key = create_1d_double_array(col);
        double *index_eq = create_1d_double_array(col);

        for(i = 0 ; i < col ; i++) // primary sort key: copy last row of p
            pri_sort_key[i] = p[row-1][i];

        for (i = 0; i < col; i++) 
    		lexsort_o[i] = i;

        //https://www.geeksforgeeks.org/stable-selection-sort/
        //stableSelectionSort(int a[], int n) 
        // Iterate through array elements 
        for (i = 0; i < col - 1; i++) {  
            // Find minimum element from arr[i] to arr[n - 1]. 
            min = i; 
            for (j = i + 1; j < col; j++) {
                //if (a[min] > a[j]) 
                //    min = j;
               // if (flag == 1) {
                    compare = pri_sort_key[min] - pri_sort_key[j];
               // }
               // else {
               //     compare = pri_sort_key[j] - pri_sort_key[min];
               // }

                if (compare > 0)
                    min = j;

                // equal element indicator   
                if (pri_sort_key[i] == pri_sort_key[j]) {
                    //if (index_eq[i] == 0) {
                    //    max_index_eq += 1;
                    //    index_eq[i] = max_index_eq;
                    //}
                    //
                    //if (index_eq[j] == 0)
                    //    index_eq[j] = index_eq[i];
                    has_eq_element = 1;
                }
            }
    
            // Move minimum element at current i. 
            key = pri_sort_key[min];
            m = lexsort_o[min];
            while (min > i)  
            { 
                pri_sort_key[min] = pri_sort_key[min - 1]; 
                lexsort_o[min] = lexsort_o[min - 1];
                min--; 
            } 
            pri_sort_key[i] = key;
            lexsort_o[i] = m; 
        }

        cnt_row = row - 1;

        while ((cnt_row > 0) && (has_eq_element == 1)) {
            // mark equal element for cnt_row
            if (cnt_row == row - 1)
                //col_tmp = col - 1;
            else {
                //m = arr_eq_par[cnt_row+1][col/2+1]; // (cnt_row+1)'s cnt_eq_set
                //col_tmp = arr_eq_par[cnt_row+1][2*m+1]; // (cnt_row+1)'s cnt_eq_set's eq_set inc
                arr_eq_par[cnt_row][col/2]; // reset cnt_row's max_index_eq
                arr_eq_par[cnt_row][col/2+1]; // reset cnt_row's cnt_eq_set
            }

            for (i = eq_start_pt; i < eq_start_pt + eq_inc; i++) {
                j = i+1;
                if (arr_sort[i] == arr_sort[j]) {
                    if (arr_index_eq[i] == 0) {
                        m = arr_eq_par[cnt_row][col/2]; // max_index_eq
                        arr_eq_par[cnt_row][col/2] ++; // max_index_eq ++
                        arr_index_eq[i] = 1;
                        arr_eq_par[cnt_row][2*m] = i; // save (max_index_eq-1)'s eq_set start pt
                        arr_eq_par[cnt_row][2*m+1] ++; // (max_index_eq-1)'s eq_set inc ++
                    }

                    if (arr_index_eq[j] == 0) {
                        arr_index_eq[j] = 1;
                        arr_eq_par[cnt_row][2*m+1] ++; // max_index_eq's eq_set inc ++
                    }
                }
            }

            // sort multiple set of equal elements in the last row by multiple columns in multiple rows
            for (k = 0; k < max_index_eq; k++) { // multiple set of equal elements
                record_start = 1;
                cnt_tmp = 0;
                row_tmp = row - 1;
                has_eq_element = 0;
                stop_row_compare = 0;
                for (i = 0; i < col; i++) {
                    if (index_eq[i] == k+1) {
                        if (record_start == 1) {
                            start_index_eq_key = i; // record start position of column in the last row
                            record_start = 0;
                        }
                        //eq_key[cnt_tmp] = pri_sort_key[i];
                        index_eq_key[cnt_tmp] = lexsort_o[i]; // copy index of this set of equal elements
                        cnt_tmp ++; // count the amount of this set of equal elements
                    }
                }

                while (stop_row_compare == 0) {
                    for (i = 0; i < cnt_tmp; i++) {
                        row_tmp --;
                        m = index_eq_key[i];
                        eq_key[i] = p[row_tmp][m]; // copy this set of equal elements for sorting
                    }

                    // stable sort
                    for (i = 0; i < cnt_tmp; i++) {
                        min = i; 
                        for (j = i + 1; j < col; j++) {
                            compare = eq_key[min] - eq_key[j];

                            if (compare > 0)
                                min = j;  

                            if (eq_key[i] == eq_key[j])
                                has_eq_element = 1;

                        }

                        // Move minimum element at current i. 
                        key = eq_key[min];
                        m = index_eq_key[min];
                        while (min > i)  
                        { 
                            eq_key[min] = eq_key[min - 1]; 
                            index_eq_key[min] = index_eq_key[min - 1];
                            min--; 
                        } 
                        eq_key[i] = key;
                        index_eq_key[i] = m; 
                    }

                    if ((has_eq_element == 0) || (row_tmp == 0))
                        stop_row_compare = 1;
                }            

                // copy sorting result of this set of equal elements to output
                for (i = 0; i < cnt_tmp; i++) {
                    m = i + start_index_eq_key;
                    lexsort_o[m] = index_eq_key[i];
                }
            }
        }
    
    } */

    //void np_lexsort_2d_double_v1(double** p ,int flag ,int* lexsort_o)
    void np_lexsort_2d_double_v1(int row ,int col ,double** p_in ,int* lexsort_o)
    {
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, j;
        int min = 0, m = 0;
        int finish = 0, has_eq_element = 0, exec_step_3 = 0;
        int cnt_eq_set_num = 0, eq_start_pt = 0, eq_inc = 0;
        //int row = sizeof (p_in[0][0]) / sizeof (p_in);
        //int col = sizeof (p_in) + 1;
        int cnt_row = row - 1;
        int half_col = col/2; // == max eq set
        int col_even = 2*half_col;
        int loop_count = col;
        double compare = 0, key = 0;
        double *arr_sort = create_1d_double_array(col);
        int *arr_index_sort = create_1d_int_array(col);
        int *arr_index_eq = create_1d_int_array(col);
        int **arr_eq_par = create_2d_int_array(row, col_even + 2);

        for (i = 0; i < col; i++) 
    		//lexsort_o[i] = i;   
    		arr_index_sort[i] = i;     


        while (finish == 0) {
            // step 1. cp cnt_row to memory, including index and elements
            if (cnt_row == row - 1) {           
                for(i = 0 ; i < col ; i++) // cp p_in[row-1][0 ~ col-1] to arr_sort[0 ~ col-1]
                    arr_sort[i] = p_in[row-1][i];
            } else {
                cnt_eq_set_num = arr_eq_par[cnt_row+1][col_even+1]; // (cnt_row+1)'s cnt_eq_set
                eq_start_pt = arr_eq_par[cnt_row+1][2*cnt_eq_set_num]; // (cnt_row+1)'s start pt
                eq_inc = arr_eq_par[cnt_row+1][2*cnt_eq_set_num+1]; // (cnt_row+1)'s inc            

                for (i = 0; i < eq_inc; i++) { // cp lexsort_o[eq_start_pt ~ (eq_start_pt + eq_inc - 1)] to arr_index_sort[0 ~ eq_inc - 1]
                    arr_index_sort[i] = lexsort_o[eq_start_pt+i];
                }

                for (i = 0; i < eq_inc; i++) { // cp p_in[cnt_row][j] to arr_sort[i]
                    j = arr_index_sort[i];
                    arr_sort[i] = p_in[cnt_row][j];
                }
            }
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 1 cnt_row= %d", cnt_row);
                printf(", cnt_eq_set_num= %d", cnt_eq_set_num);
                printf(", eq_start_pt= %d", eq_start_pt);
                printf(", eq_inc= %d", eq_inc);
                printf("\nsub c step 1 arr_sort= ");
                for (i = 0; i < col; i++) {
                    printf("%f", arr_sort[i]); printf(", ");
                }
                printf("\nsub c step 1 arr_index_sort= ");
                for (i = 0; i < col; i++) {
                    printf("%d", arr_index_sort[i]); printf(", ");
                }
            #endif

            //step 2. sort
            //step 2.1 stable sort: https://www.geeksforgeeks.org/stable-selection-sort/
            // Iterate through array elements
            if (cnt_row < row - 1)
                loop_count = eq_inc;

            if (arr_sort[loop_count - 1] > 1)
                key = arr_sort[loop_count - 1] - 1;
            else            
                key = arr_sort[loop_count - 1] + 1;

            for (i = 0; i < loop_count - 1; i++) {  
                // Find minimum element from arr[i] to arr[n - 1]. 
                min = i;
                //#ifdef DEBUG_np_lexsort_2d_double
                //    printf("\nsub c step 2.1 i= %d", i); 
                //#endif
                for (j = i + 1; j < loop_count; j++) {
                   // if (flag == 1) {
                        compare = arr_sort[min] - arr_sort[j];
                   // }
                   // else {
                   //     compare = arr_sort[j] - arr_sort[min];
                   // }

                    if (compare > 0)
                        min = j;

                    // equal element indicator   
                    //if (arr_sort[i] == arr_sort[j]) {  
                    if (i == 0) {
                        if ((arr_sort[i] == arr_sort[j]) || (key == arr_sort[j])) {
                            //if (index_eq[i] == 0) {
                            //    max_index_eq += 1;
                            //    index_eq[i] = max_index_eq;
                            //}
                            //
                            //if (index_eq[j] == 0)
                            //    index_eq[j] = index_eq[i];
                            has_eq_element = 1;
                        }

                        key = arr_sort[j];
                    }
                    //#ifdef DEBUG_np_lexsort_2d_double
                    //    printf("\nsub c step 2.1 j= %d", j);
                    //    printf(", arr_sort[min]= %f", arr_sort[min]);
                    //    printf(", arr_sort[i]= %f", arr_sort[i]);
                    //    printf(", arr_sort[j]= %f", arr_sort[j]);
                    //#endif
                }

                // Move minimum element at current i. 
                key = arr_sort[min];
                m = arr_index_sort[min];
                while (min > i)  
                { 
                    arr_sort[min] = arr_sort[min - 1]; 
                    arr_index_sort[min] = arr_index_sort[min - 1];
                    min--; 
                } 
                arr_sort[i] = key;
                arr_index_sort[i] = m; 
            }
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 2.1 loop_count= %d", loop_count);
                printf("\nsub c step 2.1 has_eq_element= %d", has_eq_element);
                printf("\nsub c step 2.1 arr_sort= ");
                for (i = 0; i < loop_count; i++) {
                    printf("%f", arr_sort[i]); printf(", ");
                }
                printf("\nsub c step 2.1 arr_index_sort= ");
                for (i = 0; i < loop_count; i++) {
                    printf("%d", arr_index_sort[i]); printf(", ");
                }
            #endif

            //step 2.2 cp sort result to lexsort_o
            /*if (cnt_row == row - 1) {          
                for(i = 0 ; i < col ; i++) // cp arr_index_sort to lexsort_o 
                    lexsort_o[i] = arr_index_sort[i];
            } else {
                for (i = 0; i < eq_inc; i++)  // cp arr_index_sort to lexsort_o
                    lexsort_o[eq_start_pt+i] = arr_index_sort[i];
            }*/
            for (i = 0; i < loop_count; i++)  // cp arr_index_sort to lexsort_o
                lexsort_o[eq_start_pt+i] = arr_index_sort[i];

            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 2.2 lexsort_o= ");
                for (i = 0; i < col; i++) {
                    printf("%d", lexsort_o[i]); printf(", ");
                }
            #endif

            //step 2.3 condition: finish, continueously from bottom to top or begin from top to bottom
            if ((has_eq_element == 1) && (cnt_row > 0)) {
                exec_step_3 = 1; // go to step 3
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 2.3 (has_eq_element == 1) && (cnt_row > 0)");
                #endif
            } else {
                //search (max_eq_set - cnt_eq_set > 0) from cnt_row+1 to row-1 if (cnt_row < row - 1):
                if (cnt_row == row -1) {
                    finish = 1;
                    exec_step_3 = 0;
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 finish = 1; exec_step_3 = 0;");
                    #endif
                } else {
                    // search (max_eq_set - (cnt_eq_set + 1) > 0) 
                    // from cnt_row+1(for cnt_row == 0) or cnt_row(for cnt_row > 0) to row-1: // top down search  
                    //if (cnt_row == 0) search_st_row = cnt_row+1
                    //else search_st_row = cnt_row
                    //for (i==search_st_row; i < row; i++)
                    for (i = cnt_row+1; i < row; i++) {
                        if (arr_eq_par[i][col_even] - (arr_eq_par[i][col_even+1] + 1) > 0) {
                            cnt_row = i - 1;
                            //for (j==cnt_row; j >= 0; j--) // clear max_eq_set and cnt_eq_set from cnt_row to 0
                            //    arr_eq_par[j][col/2] = 0
                            //    arr_eq_par[j][col/2+1] = 0
                            arr_eq_par[i][col_even+1] ++;
                            finish = 0;
                            exec_step_3 = 0; // go to step 1
                            #ifdef DEBUG_np_lexsort_2d_double
                                printf("\nsub c step 2.3 i= %d", i);
                                printf("\nsub c step 2.3 max_index_eq= %d", arr_eq_par[i][col_even]);
                                printf("\nsub c step 2.3 cnt_eq_set= %d", arr_eq_par[i][col_even+1]);
                            #endif
                            break;
                        } else if (i == row - 1) { // note: arr_eq_par[i][col/2] - (arr_eq_par[i][col/2+1] + 1) == 0
                            finish = 1;
                            exec_step_3 = 0; // skip step 3
                            break;
                        }
                    }
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 cnt_row= %d", cnt_row);
                        printf("\nsub c step 2.3 finish= %d", finish);
                        printf("\nsub c step 2.3 exec_step_3= %d", exec_step_3);
                    #endif
                }
            }

            has_eq_element = 0;

            //step 3. mark eq and save associated parameters        
            if (exec_step_3 == 1) {
                //step 3.1 mark equal elements
                //& step 3.2 save cnt_row's start pt and inc for each eq set and max_eq_set (arr_eq_par[cnt_row][0 ~ 2*max_eq_set - 1, col/2])
                //search eq elements range
                if (cnt_row == row - 1) 
                    loop_count = col - 1;
                else
                    loop_count = eq_inc;

                if (cnt_row < row - 1) {
                    //m = arr_eq_par[cnt_row+1][col/2+1]; // (cnt_row+1)'s cnt_eq_set
                    //col_tmp = arr_eq_par[cnt_row+1][2*m+1]; // (cnt_row+1)'s cnt_eq_set's eq_set inc
                    arr_eq_par[cnt_row][col_even] = 0; // reset cnt_row's max_index_eq
                    arr_eq_par[cnt_row][col_even+1] = 0; // reset cnt_row's cnt_eq_set                

                    for (i = 0; i < loop_count; i++) {
                        m = 2*i;
                        arr_eq_par[cnt_row][m+1] = 0; // reset cnt_row's cnt_eq_set's eq_set inc
                        arr_index_eq[i] = 0; // reset arr_index_eq
                    }

                    loop_count --;
                }
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 3 search eq elements range= %d", loop_count+1);
                #endif

                //m = 0;  
                //for (i = 0; i < eq_inc; i++) {  
                for (i = 0; i < loop_count; i++) {
                    j = i+1;
                    if (arr_sort[i] == arr_sort[j]) {
                        if (arr_index_eq[i] == 0) { // find an eq_set start pt
                            m = 2*arr_eq_par[cnt_row][col_even]; // 2*max_index_eq
                            arr_eq_par[cnt_row][col_even] ++; // max_index_eq ++
                            arr_index_eq[i] = 1; // mark this eq element
                            //arr_eq_par[cnt_row][m] = i; // save (max_index_eq-1)'s eq_set start pt
                            arr_eq_par[cnt_row][m] = eq_start_pt + i; // save (max_index_eq-1)'s eq_set start pt
                            arr_eq_par[cnt_row][m+1] ++; // (max_index_eq-1)'s eq_set inc ++
                            //printf("\nm= %d", m);
                        }

                        if (arr_index_eq[j] == 0) {
                            arr_index_eq[j] = 1;
                            arr_eq_par[cnt_row][m+1] ++; // max_index_eq's eq_set inc ++
                        }
                    }
                }
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[cnt_row][col_even]);
                    printf("\nsub c step 3 cnt_eq_set= %d", arr_eq_par[cnt_row][col_even+1]);
                    printf("\nsub c step 3 eq_set start pt= ");
                    for (i = 0; i < half_col; i++) {
                        printf("%d", arr_eq_par[cnt_row][2*i]); printf(", ");
                    }
                    printf("\nsub c step 3 eq_set inc= ");
                    for (i = 0; i < half_col; i++) {
                        printf("%d", arr_eq_par[cnt_row][2*i+1]); printf(", ");
                    }
                #endif            

                //step 3.3 cnt_row --
                cnt_row --;

                //step 3.4 go to step 1.            
            }          
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c end all step, arr_eq_par= ");
                for(i = row -1 ; i >= 0 ; i--) {
                    printf("\n");
                    for(j = 0 ; j <= col_even+1 ; j++) {
                        printf("%d", arr_eq_par[i][j]); printf(", ");
                    }
                }
            #endif     
        }
        #ifdef DEBUG_np_lexsort_2d_double
            printf("\nsub c lexsort_o= ");
            for (i = 0; i < col; i++) {
                printf("%d", lexsort_o[i]); printf(", ");
            }
        #endif

        //long vmrss, vmsize;
        //get_memory_usage_kb(&vmrss, &vmsize);
        //printf("\nspe c v1 Current memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n", vmrss, vmsize);

        free(arr_sort);
        free(arr_index_sort);
        free(arr_index_eq);
        free_2d_int_array(arr_eq_par, row);    


    }

    //void np_lexsort_2d_double_v2(double** p ,int flag ,int* lexsort_o)
    void np_lexsort_2d_double_v2(int row ,int col ,double** p_in ,int* lexsort_o)
    {
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, j;
        int min = 0, m = 0;
        int finish = 0, has_eq_element = 0, exec_step_3 = 0;
        int cnt_eq_set_num = 0, eq_start_pt = 0, eq_inc = 0;
        //int row = sizeof (p_in[0][0]) / sizeof (p_in);
        //int col = sizeof (p_in) + 1;
        //int cnt_row = row - 1;
        int half_col = col/2; // == max eq set
        int col_even = 2*half_col;
        int loop_count = col;
        double compare = 0, key = 0;
        double *arr_sort = create_1d_double_array(col);
        int *arr_index_sort = create_1d_int_array(col);
        int *arr_index_eq = create_1d_int_array(col);

        int max_row_size = 1, cnt_row_eq_par = 0, arr_eq_par_col_size = col_even + 2;
        //int **arr_eq_par = create_2d_int_array(row, col_even + 2);
        //int **arr_eq_par = create_2d_int_array(max_row_size, col_even + 2);
        int *arr_eq_par = create_1d_int_array(arr_eq_par_col_size);

        for (i = 0; i < col; i++) 
    		//lexsort_o[i] = i;   
    		arr_index_sort[i] = i;     


        while (finish == 0) {
            // step 1. cp cnt_row to memory, including index and elements
            //if (cnt_row == row - 1) {
            if (cnt_row_eq_par == 0) {           
                for(i = 0 ; i < col ; i++) // cp p_in[row-1][0 ~ col-1] to arr_sort[0 ~ col-1]
                    arr_sort[i] = p_in[row-1][i];
            } else {
                //cnt_eq_set_num = arr_eq_par[cnt_row+1][col_even+1]; // (cnt_row+1)'s cnt_eq_set
                //eq_start_pt = arr_eq_par[cnt_row+1][2*cnt_eq_set_num]; // (cnt_row+1)'s start pt
                //eq_inc = arr_eq_par[cnt_row+1][2*cnt_eq_set_num+1]; // (cnt_row+1)'s inc
              //  cnt_row = row - 1 - cnt_row_eq_par; 
              //  cnt_eq_set_num = arr_eq_par[cnt_row_eq_par-1][col_even+1]; // (cnt_row+1)'s cnt_eq_set
              //  eq_start_pt = arr_eq_par[cnt_row_eq_par-1][2*cnt_eq_set_num]; // (cnt_row+1)'s start pt
              //  eq_inc = arr_eq_par[cnt_row_eq_par-1][2*cnt_eq_set_num+1]; // (cnt_row+1)'s inc
                m = (cnt_row_eq_par - 1)*arr_eq_par_col_size; 
                cnt_eq_set_num = arr_eq_par[m+col_even+1]; // (cnt_row+1)'s cnt_eq_set
                eq_start_pt = arr_eq_par[m+2*cnt_eq_set_num]; // (cnt_row+1)'s start pt
                eq_inc = arr_eq_par[m+2*cnt_eq_set_num+1]; // (cnt_row+1)'s inc
                for (i = 0; i < eq_inc; i++) { // cp lexsort_o[eq_start_pt ~ (eq_start_pt + eq_inc - 1)] to arr_index_sort[0 ~ eq_inc - 1]
                    arr_index_sort[i] = lexsort_o[eq_start_pt+i];
                }

                for (i = 0; i < eq_inc; i++) { // cp p_in[cnt_row][j] to arr_sort[i]
                    j = arr_index_sort[i];
                    arr_sort[i] = p_in[row-1-cnt_row_eq_par][j];
                }
            }
            #ifdef DEBUG_np_lexsort_2d_double
                //printf("\nsub c step 1 cnt_row= %d", cnt_row);
                printf("\nsub c step 1 cnt_row_eq_par= %d", cnt_row_eq_par);
                printf(", cnt_eq_set_num= %d", cnt_eq_set_num);
                printf(", eq_start_pt= %d", eq_start_pt);
                printf(", eq_inc= %d", eq_inc);
                printf("\nsub c step 1 arr_sort= ");
                for (i = 0; i < col; i++) {
                    printf("%f", arr_sort[i]); printf(", ");
                }
                printf("\nsub c step 1 arr_index_sort= ");
                for (i = 0; i < col; i++) {
                    printf("%d", arr_index_sort[i]); printf(", ");
                }
            #endif

            //step 2. sort
            //step 2.1 stable sort: https://www.geeksforgeeks.org/stable-selection-sort/
            // Iterate through array elements
            //if (cnt_row < row - 1)
            if (cnt_row_eq_par > 0)
                loop_count = eq_inc;

            if (arr_sort[loop_count - 1] > 1)
                key = arr_sort[loop_count - 1] - 1;
            else            
                key = arr_sort[loop_count - 1] + 1;

            for (i = 0; i < loop_count - 1; i++) {  
                // Find minimum element from arr[i] to arr[n - 1]. 
                min = i;
                //#ifdef DEBUG_np_lexsort_2d_double
                //    printf("\nsub c step 2.1 i= %d", i); 
                //#endif
                for (j = i + 1; j < loop_count; j++) {
                   // if (flag == 1) {
                        compare = arr_sort[min] - arr_sort[j];
                   // }
                   // else {
                   //     compare = arr_sort[j] - arr_sort[min];
                   // }

                    if (compare > 0)
                        min = j;

                    // equal element indicator   
                    //if (arr_sort[i] == arr_sort[j]) {  
                    if (i == 0) {
                        if ((arr_sort[i] == arr_sort[j]) || (key == arr_sort[j])) {
                            //if (index_eq[i] == 0) {
                            //    max_index_eq += 1;
                            //    index_eq[i] = max_index_eq;
                            //}
                            //
                            //if (index_eq[j] == 0)
                            //    index_eq[j] = index_eq[i];
                            has_eq_element = 1;
                        }

                        key = arr_sort[j];
                    }
                    //#ifdef DEBUG_np_lexsort_2d_double
                    //    printf("\nsub c step 2.1 j= %d", j);
                    //    printf(", arr_sort[min]= %f", arr_sort[min]);
                    //    printf(", arr_sort[i]= %f", arr_sort[i]);
                    //    printf(", arr_sort[j]= %f", arr_sort[j]);
                    //#endif
                }

                // Move minimum element at current i. 
                key = arr_sort[min];
                m = arr_index_sort[min];
                while (min > i)  
                { 
                    arr_sort[min] = arr_sort[min - 1]; 
                    arr_index_sort[min] = arr_index_sort[min - 1];
                    min--; 
                } 
                arr_sort[i] = key;
                arr_index_sort[i] = m; 
            }
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 2.1 loop_count= %d", loop_count);
                printf("\nsub c step 2.1 has_eq_element= %d", has_eq_element);
                printf("\nsub c step 2.1 arr_sort= ");
                for (i = 0; i < loop_count; i++) {
                    printf("%f", arr_sort[i]); printf(", ");
                }
                printf("\nsub c step 2.1 arr_index_sort= ");
                for (i = 0; i < loop_count; i++) {
                    printf("%d", arr_index_sort[i]); printf(", ");
                }
            #endif

            //step 2.2 cp sort result to lexsort_o
            /*if (cnt_row == row - 1) {          
                for(i = 0 ; i < col ; i++) // cp arr_index_sort to lexsort_o 
                    lexsort_o[i] = arr_index_sort[i];
            } else {
                for (i = 0; i < eq_inc; i++)  // cp arr_index_sort to lexsort_o
                    lexsort_o[eq_start_pt+i] = arr_index_sort[i];
            }*/
            for (i = 0; i < loop_count; i++)  // cp arr_index_sort to lexsort_o
                lexsort_o[eq_start_pt+i] = arr_index_sort[i];

            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 2.2 lexsort_o= ");
                for (i = 0; i < col; i++) {
                    printf("%d", lexsort_o[i]); printf(", ");
                }
            #endif

            //step 2.3 condition: finish, continueously from bottom to top or begin from top to bottom
            //if ((has_eq_element == 1) && (cnt_row > 0)) {
            if ((has_eq_element == 1) && (cnt_row_eq_par < row - 1)) {
                exec_step_3 = 1; // go to step 3
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 2.3 (has_eq_element == 1) && (cnt_row > 0)");
                #endif
            } else {
                //search (max_eq_set - cnt_eq_set > 0) from cnt_row+1 to row-1 if (cnt_row < row - 1):
                //if (cnt_row == row -1) {
                if (cnt_row_eq_par == 0) {
                    finish = 1;
                    exec_step_3 = 0;
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 finish = 1; exec_step_3 = 0;");
                    #endif
                } else {
                    // search (max_eq_set - (cnt_eq_set + 1) > 0) 
                    // from cnt_row+1(for cnt_row == 0) or cnt_row(for cnt_row > 0) to row-1: // top down search  
                    //if (cnt_row == 0) search_st_row = cnt_row+1
                    //else search_st_row = cnt_row
                    //for (i==search_st_row; i < row; i++)
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 cnt_row_eq_par= %d", cnt_row_eq_par);
                      //  printf("\nsub c step 2.3 max_index_eq= %d", arr_eq_par[cnt_row_eq_par-1][col_even]);
                      //  printf("\nsub c step 2.3 cnt_eq_set= %d", arr_eq_par[cnt_row_eq_par-1][col_even+1]);
                        min = (cnt_row_eq_par - 1)*arr_eq_par_col_size;
                        printf("\nsub c step 2.3 max_index_eq= %d", arr_eq_par[min+col_even]);
                        printf("\nsub c step 2.3 cnt_eq_set= %d", arr_eq_par[min+col_even+1]);
                    #endif
                  //  for (i = cnt_row+1; i < row; i++) {
                  //      if (arr_eq_par[i][col_even] - (arr_eq_par[i][col_even+1] + 1) > 0) {
                  //          cnt_row = i - 1;
                    for (i = cnt_row_eq_par-1; i >= 0; i--) {
                      //  if (arr_eq_par[i][col_even] - (arr_eq_par[i][col_even+1] + 1) > 0) {
                        m = i*arr_eq_par_col_size;
                        if (arr_eq_par[m+col_even] - (arr_eq_par[m+col_even+1] + 1) > 0) {
                            cnt_row_eq_par = i + 1;
                            //for (j==cnt_row; j >= 0; j--) // clear max_eq_set and cnt_eq_set from cnt_row to 0
                            //    arr_eq_par[j][col/2] = 0
                            //    arr_eq_par[j][col/2+1] = 0
                          //  arr_eq_par[i][col_even+1] ++;
                            arr_eq_par[m+col_even+1] ++;
                            finish = 0;
                            exec_step_3 = 0; // go to step 1
                            #ifdef DEBUG_np_lexsort_2d_double
                                printf("\nsub c step 2.3 i= %d", i);
                                printf("\nsub c step 2.3 max_index_eq= %d", arr_eq_par[m+col_even]);
                                printf("\nsub c step 2.3 cnt_eq_set= %d", arr_eq_par[m+col_even+1]);
                            #endif
                            break;
                        //} else if (i == row - 1) { // note: arr_eq_par[i][col/2] - (arr_eq_par[i][col/2+1] + 1) == 0
                        } else if (i == 0) { // note: arr_eq_par[i][col/2] - (arr_eq_par[i][col/2+1] + 1) == 0
                            finish = 1;
                            exec_step_3 = 0; // skip step 3
                            break;
                        }
                    }
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 cnt_row_eq_par= %d", cnt_row_eq_par);
                        printf("\nsub c step 2.3 finish= %d", finish);
                        printf("\nsub c step 2.3 exec_step_3= %d", exec_step_3);
                    #endif
                }
            }

            has_eq_element = 0;

            //step 3. mark eq and save associated parameters        
            if (exec_step_3 == 1) {
                //step 3.1 mark equal elements
                //& step 3.2 save cnt_row's start pt and inc for each eq set and max_eq_set (arr_eq_par[cnt_row][0 ~ 2*max_eq_set - 1, col/2])
                //search eq elements range
                //if (cnt_row == row - 1)
                if (cnt_row_eq_par == 0) 
                    loop_count = col - 1;
                else
                    loop_count = eq_inc;

                //cnt_row_eq_par = row - 1 - cnt_row;

                if (cnt_row_eq_par == max_row_size) {
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 3 cnt_row_eq_par= %d", cnt_row_eq_par);
                        printf("\nsub c step 3 max_row_size= %d", max_row_size);
                    #endif
                    max_row_size ++;

                    // Reallocate rows
                    /*for (i = 0; i < max_row_size; i++) {
                        arr_eq_par[i] = (int*)realloc(arr_eq_par[i], (col_even + 2)*sizeof(int));
                        if (arr_eq_par[i] == NULL) {
                            fprintf(stderr, "Out of memory");
                            exit(0);
                        }
                    }*/
                    //arr_eq_par = realloc(arr_eq_par, max_row_size*(col_even + 2)*sizeof(int*));
                    //arr_eq_par = realloc(arr_eq_par, max_row_size*sizeof(int*));
                    arr_eq_par = realloc(arr_eq_par, max_row_size*arr_eq_par_col_size*sizeof(int));
                    if (arr_eq_par == NULL) {
                        fprintf(stderr, "Out of memory");
                        exit(0);
                    }
                    for(i = cnt_row_eq_par*arr_eq_par_col_size ; i < max_row_size*arr_eq_par_col_size ; i++)
                        arr_eq_par[i] = 0;        
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 3 Reallocate rows= ");
                        for(i = 0 ; i < max_row_size*arr_eq_par_col_size ; i++) {
                            printf("%d", arr_eq_par[i]); printf(", ");
                        }
                    #endif
                    /*arr_eq_par = realloc(arr_eq_par, max_row_size*sizeof(int *));    
                    if (arr_eq_par == NULL) {
                        fprintf(stderr, "realloc, Out of memory");
                        exit(0);
                    }*/
                    // Reallocate rows
                    /*for (i = cnt_row_eq_par; i < max_row_size; i++) {
                        //arr_eq_par[i] = (int*)realloc(arr_eq_par[i], (col_even + 2)*sizeof(int));
                        arr_eq_par[i] = realloc(arr_eq_par[i], (col_even + 2)*sizeof(int));
                        if (arr_eq_par[i] == NULL) {
                            fprintf(stderr, "\nOut of memory");
                            exit(0);
                        }
                    }*/
                  //  printf("\nsub c step 3 realloc");
                    /*
                    for(i = cnt_row_eq_par ; i < max_row_size ; i++)
                        for(j = 0 ; j < col_even + 2 ; j++)
                            arr_eq_par[i][j] = 0;

                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 3 Reallocate rows= ");
                        for(i = 0 ; i < max_row_size ; i++) {
                            for(j = 0 ; j < col_even + 2 ; j++) {
                                printf("%d", arr_eq_par[i][j]); printf(", ");
                            }
                        }
                    #endif*/
                }

                m = cnt_row_eq_par*arr_eq_par_col_size;
                //if (cnt_row < row - 1) {
                if (cnt_row_eq_par > 0) {
                    //arr_eq_par[cnt_row][col_even] = 0; // reset cnt_row's max_index_eq
                    //arr_eq_par[cnt_row][col_even+1] = 0; // reset cnt_row's cnt_eq_set
                  //  arr_eq_par[cnt_row_eq_par][col_even] = 0; // reset cnt_row's max_index_eq
                  //  arr_eq_par[cnt_row_eq_par][col_even+1] = 0; // reset cnt_row's cnt_eq_set
                    arr_eq_par[m+col_even] = 0; // reset cnt_row's max_index_eq
                    arr_eq_par[m+col_even+1] = 0; // reset cnt_row's cnt_eq_set                

                    for (i = 0; i < loop_count; i++) {
                      //  m = 2*i;
                        //arr_eq_par[cnt_row][m+1] = 0; // reset cnt_row's cnt_eq_set's eq_set inc
                      //  arr_eq_par[cnt_row_eq_par][m+1] = 0; // reset cnt_row's cnt_eq_set's eq_set inc
                        arr_eq_par[m+2*i+1] = 0; // reset cnt_row's cnt_eq_set's eq_set inc
                        arr_index_eq[i] = 0; // reset arr_index_eq
                    }

                    loop_count --;
                }
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 3 search eq elements range= %d", loop_count+1);
                #endif

                //m = 0;  
                //for (i = 0; i < eq_inc; i++) {  
                for (i = 0; i < loop_count; i++) {
                    j = i+1;
                    if (arr_sort[i] == arr_sort[j]) {
                        if (arr_index_eq[i] == 0) { // find an eq_set start pt
                            //m = 2*arr_eq_par[cnt_row][col_even]; // 2*max_index_eq
                            //arr_eq_par[cnt_row][col_even] ++; // max_index_eq ++
                            //arr_index_eq[i] = 1; // mark this eq element
                            ////arr_eq_par[cnt_row][m] = i; // save (max_index_eq-1)'s eq_set start pt
                            //arr_eq_par[cnt_row][m] = eq_start_pt + i; // save (max_index_eq-1)'s eq_set start pt
                            //arr_eq_par[cnt_row][m+1] ++; // (max_index_eq-1)'s eq_set inc ++
                          //  m = 2*arr_eq_par[cnt_row_eq_par][col_even]; // 2*max_index_eq
                          //  arr_eq_par[cnt_row_eq_par][col_even] ++; // max_index_eq ++
                            min = 2*arr_eq_par[m+col_even]; // 2*max_index_eq
                            //printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[m+col_even]);
                            arr_eq_par[m+col_even] ++; // max_index_eq ++
                            //printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[m+col_even]);
                            arr_index_eq[i] = 1; // mark this eq element
                            //arr_eq_par[cnt_row][m] = i; // save (max_index_eq-1)'s eq_set start pt
                          //  arr_eq_par[cnt_row_eq_par][m] = eq_start_pt + i; // save (max_index_eq-1)'s eq_set start pt
                          //  arr_eq_par[cnt_row_eq_par][m+1] ++; // (max_index_eq-1)'s eq_set inc ++
                            arr_eq_par[m+min] = eq_start_pt + i; // save (max_index_eq-1)'s eq_set start pt
                            arr_eq_par[m+min+1] ++; // (max_index_eq-1)'s eq_set inc ++
                            //printf("\nm= %d", m);
                        }

                        if (arr_index_eq[j] == 0) {
                            arr_index_eq[j] = 1;
                            //arr_eq_par[cnt_row][m+1] ++; // max_index_eq's eq_set inc ++
                          //  arr_eq_par[cnt_row_eq_par][m+1] ++; // max_index_eq's eq_set inc ++
                            arr_eq_par[m+min+1] ++; // max_index_eq's eq_set inc ++
                        }
                    }
                }
                #ifdef DEBUG_np_lexsort_2d_double
                  //  printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[cnt_row_eq_par][col_even]);
                  //  printf("\nsub c step 3 cnt_eq_set= %d", arr_eq_par[cnt_row_eq_par][col_even+1]);
                    printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[m+col_even]);
                    printf("\nsub c step 3 cnt_eq_set= %d", arr_eq_par[m+col_even+1]);
                    printf("\nsub c step 3 eq_set start pt= ");
                    for (i = 0; i < half_col; i++) {
                      //  printf("%d", arr_eq_par[cnt_row_eq_par][2*i]); printf(", ");
                        printf("%d", arr_eq_par[m+2*i]); printf(", ");
                    }
                    printf("\nsub c step 3 eq_set inc= ");
                    for (i = 0; i < half_col; i++) {
                      //  printf("%d", arr_eq_par[cnt_row_eq_par][2*i+1]); printf(", ");
                        printf("%d", arr_eq_par[m+2*i+1]); printf(", ");
                    }
                #endif            

                //step 3.3 cnt_row --
                //cnt_row --;
                cnt_row_eq_par ++;

                //step 3.4 go to step 1.            
            }         
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c end all step, arr_eq_par= ");
                for(i = 0 ; i < max_row_size ; i++) {
                    printf("\n");
                    for(j = 0 ; j < arr_eq_par_col_size ; j++) {
                        printf("%d", arr_eq_par[i*arr_eq_par_col_size+j]); printf(", ");
                    }
                }
            #endif     
        }
        #ifdef DEBUG_np_lexsort_2d_double
            printf("\nsub c lexsort_o= ");
            for (i = 0; i < col; i++) {
                printf("%d", lexsort_o[i]); printf(", ");
            }
        #endif

        //long vmrss, vmsize;
        //get_memory_usage_kb(&vmrss, &vmsize);
        //printf("\nspe c v2 Current memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n", vmrss, vmsize);

        free(arr_sort);
        free(arr_index_sort);
        free(arr_index_eq);
        //free_2d_int_array(arr_eq_par, row);
      //  free_2d_int_array(arr_eq_par, cnt_row_eq_par);
        free(arr_eq_par);        

    }
#else //C++
    void flat_np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double _Complex *flat_A_in
                                                    ,double _Complex *eigenvalue_o
                                                    ,double _Complex *flat_eigenvector_o)
    {   
        //https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_zgeev_row.c.htm
        //https://www.ibm.com/support/knowledgecenter/SSFHY8_6.2/reference/am5gr_hsgeevx.html
        //http://www.netlib.org/clapack/old/complex16/zgeev.c
        //http://www.icl.utk.edu/~mgates3/docs/lapack.html


        double _Complex *P_o = create_1d_double_complex_array(row_in*row_in);
        //double _Complex **T_o = create_2d_double_complex_array(row_in ,row_in);
        //double _Complex **E_o = create_2d_double_complex_array(row_in ,row_in);

        int n = row_in, lda = row_in, ldvl = row_in, ldvr = row_in, info=1;
        //MKL_Complex16 w[N], vl[LDVL*N], vr[LDVR*N];
        //MKL_Complex16 a[LDA*N] = {
        //   {-3.84,  2.25}, {-8.94, -4.75}, { 8.95, -6.53}, {-9.87,  4.82},
        //   {-0.66,  0.83}, {-4.40, -3.82}, {-3.50, -4.26}, {-3.15,  7.36},
        //   {-3.99, -4.73}, {-5.88, -6.60}, {-3.36, -0.40}, {-0.75,  5.23},
        //   { 7.74,  4.18}, { 3.66, -7.53}, { 2.58,  3.60}, { 4.59,  5.41}
        //};

        #ifdef DEBUG_np_linalg_eig_double_complex
            clock_t time0, time1;
            time0 = clock();
        #endif
    /*    double complex *a = create_1d_double_complex_array(row_in*row_in);
        a[0]=  750543.69428599+0j; a[1]=  1879355.49996126+0j; a[2]=  3008167.30563653+0j; a[3]=  4136979.11131179+0j; a[4]=  5265790.91698706+0j; a[5]=  6394602.72266233+0j; a[6]=  7523414.52833759+0j; a[7]=  8652226.33401285+0j;
    a[8]= 1879355.49996126+0j; a[9]=  5265767.46286309+0j; a[10]=  8652179.42576492+0j; a[11]= 12038591.38866674+0j; a[12]= 15425003.35156858+0j; a[13]= 18811415.31447038+0j; a[14]= 22197827.27737221+0j; a[15]= 25584239.24027405+0j;
    a[16]= 3008167.30563653+0j; a[17]=  8652179.42576492+0j; a[18]= 14296191.5458933 +0j; a[19]= 19940203.66602169+0j; a[20]= 25584215.78615009+0j; a[21]= 31228227.90627848+0j; a[22]= 36872240.02640685+0j; a[23]= 42516252.14653525+0j;
    a[24]= 4136979.11131179+0j; a[25]= 12038591.38866674+0j; a[26]= 19940203.66602169+0j; a[27]= 27841815.94337663+0j; a[28]= 35743428.22073162+0j; a[29]= 43645040.49808656+0j; a[30]= 51546652.77544151+0j; a[31]= 59448265.05279647+0j;
    a[32]= 5265790.91698706+0j; a[33]= 15425003.35156858+0j; a[34]= 25584215.78615009+0j; a[35]= 35743428.22073162+0j; a[36]= 45902640.65531312+0j; a[37]= 56061853.08989465+0j; a[38]= 66221065.52447614+0j; a[39]= 76380277.95905769+0j;
    a[40]= 6394602.72266233+0j; a[41]= 18811415.31447038+0j; a[42]= 31228227.90627848+0j; a[43]= 43645040.49808656+0j; a[44]= 56061853.08989465+0j; a[45]= 68478665.6817027 +0j; a[46]= 80895478.27351078+0j; a[47]= 93312290.86531891+0j;
    a[48]= 7523414.52833759+0j; a[49]= 22197827.27737221+0j; a[50]= 36872240.02640685+0j; a[51]= 51546652.77544151+0j; a[52]=        66221065.5+0j; a[53]=        80895478.3+0j; a[54]=        95569891.0+0j; a[55]=         110244304+0j;
    a[56]= 8652226.33401285+0j; a[57]= 25584239.24027405+0j; a[58]= 42516252.14653525+0j; a[59]= 59448265.05279647+0j; a[60]=        76380278.0+0j; a[61]=        93312290.9+0j; a[62]=         110244304+0j; a[63]=         127176317+0j;    
        */
        //info = LAPACKE_zgeev( LAPACK_ROW_MAJOR, 'V', 'V', n, a, lda, w, vl, ldvl, vr, ldvr );
        info = LAPACKE_zgeev( LAPACK_ROW_MAJOR, 'N', 'V', n, flat_A_in, lda, eigenvalue_o, P_o, ldvl, flat_eigenvector_o, ldvr );
        //https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/zgeev_ex.c.htm
        //http://th.if.uj.edu.pl/~adamr/zadania/C/2019/lapack/lapack-example.c
       /* char jobvl = 'N', jobvr = 'V';
        int ldwork = 2*n;
        zgeev(&jobvl, &jobvr, &n, flat_A_in, &lda, eigenvalue_o, P_o, &ldvl, flat_eigenvector_o, &ldvr,
        P_o, &ldwork, (double *)P_o, &info);*/
        //free(a);
        #ifdef DEBUG_np_linalg_eig_double_complex
            time1 = clock();
            printf("\n spe c np_linalg_eig time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
        #endif


        //QRHessenberg_square_double_complex_array(row_in ,A_in ,&P_o[0] ,&T_o[0]);
        //eigenvectorUpper_square_double_complex_array(row_in ,T_o ,&E_o[0]); //,&eigenvector_o[0]); //,&E_o[0]);
        //np_dot_2d_double_complex(row_in ,row_in ,row_in ,P_o ,E_o ,&eigenvector_o[0]); //matMul( P, E );

        #ifdef DEBUG_np_linalg_eig_double_complex
            if (info==0)
                printf("\nsub c calc eigen success");
            else
                printf("\nsub c calc eigen fail");

            int i;        
            printf("\nsub c eigenvalue_o= ");

            for (i = 0; i < row_in; i++) {
                //eigenvalue_o[i] = T_o[i][i];
                printf("\n");
                printf("%.14f% + i*.14f", creal(eigenvalue_o[i]), cimag(eigenvalue_o[i])); printf(", ");
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
            }

            printf("\nsub c eigenvector_o= ");
            for (i = 0; i < row_in*row_in; i++) {
                //for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f% + i*.14f", creal(flat_eigenvector_o[i]), cimag(flat_eigenvector_o[i])); printf(", ");
                //} 
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
                // Check matrix norm of   A v - lambda v
                //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
            }
        #endif

       // free_2d_double_complex_array(P_o, row_in);
       // free_2d_double_complex_array(T_o, row_in);
        //free_2d_double_complex_array(E_o, row_in);
        free(P_o);
    }
    
    double deg2rad_double(double degrees) {
        return degrees*PI/180.0;
    };

    int *create_1d_int_array(int N) /* Allocate the array */
    {
        int i;
        int *array;    
        array = (int*)malloc(N*sizeof(int));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }

    float *create_1d_float_array(int N) /* Allocate the array */
    {
        int i;
        float *array;    
        array = (float*)malloc(N*sizeof(float));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }

    double *create_1d_double_array(int N) /* Allocate the array */
    {
        int i;
        double *array;    
        array = (double*)malloc(N*sizeof(double));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }

    double _Complex *create_1d_double_complex_array(int N) /* Allocate the array */
    {
        int i;
        double _Complex *array;    
        array = (double _Complex*)malloc(N*sizeof(double _Complex));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }

    char *create_1d_char_array(int N) /* Allocate the array */
    {
        int i;
        char *array;    
        array = (char*)malloc(N*sizeof(char));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            array[i] = 0; //j;
        return array;
    }

    int **create_2d_int_array(int N, int M) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        int **array;    
        array = (int**)malloc(N*sizeof(int *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (int*)malloc( M*sizeof(int) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    float **create_2d_float_array(int N, int M) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        float **array;    
        array = (float**)malloc(N*sizeof(float *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (float*)malloc( M*sizeof(float) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    double **create_2d_double_array(int N, int M) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        double **array;    
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
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    double **create_2d_double_array_inivalue(int N, int M, double inivalue) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        double **array;    
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
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = inivalue; //j;
        return array;
    }

    double _Complex **create_2d_double_complex_array(int N, int M) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        double _Complex **array;    
        array = (double _Complex**)malloc(N*sizeof(double _Complex *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double _Complex*)malloc( M*sizeof(double _Complex) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = 0; //j;
        return array;
    }

    double _Complex **create_2d_double_complex_identity_array(int N) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j;
        double _Complex **array;    
        array = (double _Complex**)malloc(N*sizeof(double _Complex *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double _Complex*)malloc( N*sizeof(double _Complex) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++) {
            for(j = 0 ; j < N ; j++) {
                if (i == j)
                    array[i][j] = 1;
                else
                    array[i][j] = 0;
            }
        }
        return array;
    }

    double ***create_3d_double_array(int N, int M, int L) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j, k;
        double ***array;    
        array = (double***)malloc(N*sizeof(double **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double**)malloc( M*sizeof(double*) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = (double*)malloc( L*sizeof(double) );
                if (array[i][j] == NULL) {
    			fprintf(stderr, "Out of memory");
    			exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = 0;
        return array;
    }

    double _Complex ***create_3d_double_complex_array(int N, int M, int L) /* Allocate the array */
    {
        /* Check if allocation succeeded. (check for NULL pointer) */
        int i, j, k;
        double _Complex ***array;    
        array = (double _Complex***)malloc(N*sizeof(double _Complex **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double _Complex**)malloc( M*sizeof(double _Complex*) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = (double _Complex*)malloc( L*sizeof(double _Complex) );
    			if (array[i][j] == NULL) {
    			fprintf(stderr, "Out of memory");
    			exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = 0;
        return array;
    }

    void free_2d_int_array(int** p, int N) {
        int i;
        for(i = 0 ; i < N ; i++)
            free(p[i]);
        free(p);
    }

    void free_2d_float_array(float** p, int N) {
        int i;
        for(i = 0 ; i < N ; i++)
            free(p[i]);
        free(p);
    }

    void free_2d_double_array(double** p, int N) {
        int i;
        for(i = 0 ; i < N ; i++)
            free(p[i]);
        free(p);
    }

    void free_2d_double_complex_array(double _Complex** p, int N) {
        int i;
        for(i = 0 ; i < N ; i++)
            free(p[i]);
        free(p);
    }

    void free_3d_double_array(double*** p, int N, int M) {
        int i, j;
    	for (i = 0; i < N; i++) 
    	{
    		for (j = 0; j < M; j++)
    			free(p[i][j]);
    		free(p[i]);
    	}
    	free(p);
    }

    void free_3d_double_complex_array(double _Complex*** p, int N, int M) {
        int i, j;
    	for (i = 0; i < N; i++) 
    	{
    		for (j = 0; j < M; j++) {
    			free(p[i][j]);
            }
    		free(p[i]);
    	}
    	free(p);
    }

    float *flatten_2d_float_array(float** p, int N, int M)
    {
        int i, j;
        float *array;   
        array = (float*)malloc(N*M*sizeof(float));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i*M + j] = p[i][j];
        return array;
    }

    double *flatten_2d_double_array(double** p, int N, int M)
    {
        int i, j;
        double *array;   
        array = (double*)malloc(N*M*sizeof(double));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i*M + j] = p[i][j];
        return array;
    }

    double _Complex *flatten_2d_double_complex_array(double _Complex** p, int N, int M)
    {
        int i, j;
        double _Complex *array;   
        array = (double _Complex*)malloc(N*M*sizeof(double _Complex));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i*M + j] = p[i][j];
        return array;
    }

    double *flatten_3d_double_array(double*** p, int N, int M, int L)
    {
        int i, j, k;
        double *array;   
        array = (double*)malloc(N*M*L*sizeof(double));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i*M*L + j*L + k] = p[i][j][k];
        return array;
    }

    double _Complex *flatten_3d_double_complex_array(double _Complex*** p, int N, int M, int L)
    {
        int i, j, k;
        double _Complex *array;    
        array = (double _Complex*)malloc(N*M*L*sizeof(double _Complex));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i*M*L + j*L + k] = p[i][j][k];
        return array;
    }

    float **deflate_2d_float_array(float* p, int N, int M)
    {
        int i, j;
        float **array;     
        array = (float**)malloc(N*sizeof(float *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (float*)malloc( M*sizeof(float) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = p[i*M + j];
        return array;
    }

    double **deflate_2d_double_array(double* p, int N, int M)
    {
        int i, j;
        double **array;     
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
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = p[i*M + j];
        return array;
    }

    double _Complex **deflate_2d_double_complex_array(double _Complex* p, int N, int M)
    {
        int i, j;
        double _Complex **array;     
        array = (double _Complex**)malloc(N*sizeof(double _Complex *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double _Complex*)malloc( M*sizeof(double _Complex) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = p[i*M + j];
        return array;
    }

    double ***deflate_3d_double_array(double* p, int N, int M, int L)
    {
        int i, j, k;
        double ***array;     
        array = (double***)malloc(N*sizeof(double **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double**)malloc( M*sizeof(double*) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = (double*)malloc( L*sizeof(double) );
    				if (array[i][j] == NULL) {
    				fprintf(stderr, "Out of memory");
    				exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = p[i*M*L + j*L + k];
        return array;
    }

    double _Complex ***deflate_3d_double_complex_array(double _Complex* p, int N, int M, int L)
    {
        int i, j, k;
        double _Complex ***array;     
        array = (double _Complex***)malloc(N*sizeof(double _Complex **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double _Complex**)malloc( M*sizeof(double _Complex *) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = (double _Complex*)malloc( L*sizeof(double _Complex) );
    				if (array[i][j] == NULL) {
    				fprintf(stderr, "Out of memory");
    				exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = p[i*M*L + j*L + k];
        return array;
    }

    void fill_1d_int_array(int* p, int dim, int value) {
        int i;
        for(i = 0 ; i < dim ; i++)
            p[i] = value;
    }

    double *create_and_copy_1d_double_array(int a_col_in ,double* a_in) // Allocate the array
    {
        int i;
        double *array;    
        array = (double*)malloc(a_col_in*sizeof(double));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < a_col_in ; i++)
            array[i] = a_in[i];
        return array;
    }

    void copy_1d_double_complex_array(int a_col_in ,double _Complex* a_in, double _Complex* b_o)
    {
        int i;
        //int N = sizeof (a) / sizeof (a[0]);

        for(i = 0 ; i < a_col_in ; i++)
            b_o[i] = a_in[i];
    }

    void copy_2d_double_complex_array(int a_row_in ,int a_col_in ,double _Complex** a_in ,double _Complex** b_o)
    {
        int i, j;
        //int N = sizeof (a) / sizeof (a[0]);
    	//int M = sizeof (a) / sizeof (a[1]);

        for(i = 0 ; i < a_row_in ; i++)
            for(j = 0 ; j < a_col_in ; j++)
                b_o[i][j] = a_in[i][j];
    }

    double **create_and_copy_2d_double_array(int N ,int M ,double** a_in)
    {
        int i, j;
        //int N = sizeof (a) / sizeof (a[0]);
    	//int M = sizeof (a) / sizeof (a[1]);        
        double **array;    
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
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                array[i][j] = a_in[i][j];

        return array;
    }

    double _Complex ***create_and_copy_3d_double_complex_array(int N ,int M ,int L ,double _Complex*** a_in)
    {
        int i, j, k;
        //int N = sizeof (a) / sizeof (a[0]);
    	//int M = sizeof (a) / sizeof (a[1]);        
        double _Complex ***array;    
        array = (double _Complex***)malloc(N*sizeof(double _Complex **));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double _Complex**)malloc( M*sizeof(double _Complex*) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }

            for (j = 0; j < M; j++)
    		{
    			array[i][j] = (double _Complex*)malloc( L*sizeof(double _Complex) );
    			if (array[i][j] == NULL) {
    			fprintf(stderr, "Out of memory");
    			exit(0);
    			}
    		}
        }

        for(i = 0 ; i < N ; i++)
            for(j = 0 ; j < M ; j++)
                for(k = 0 ; k < L ; k++)
                    array[i][j][k] = a_in[i][j][k];

        return array;
    }

    /*double int_to_double(int input)
    {
        double output;
        char temp[20];

        sprintf("%d", input);

        temp[19] = 0x00;

        sscanf("%lf", &output);

        return output;
    }*/

    void np_dot_2d_double(int a_row_in ,int a_col_in ,int b_col_in ,double** a_in, double** b_in, double** c_o)
    {
        // usage:
        // c[N][L] = dot product (a[N][M], b[M][L])
        int i, j, k;
        //int N = sizeof (a) / sizeof (a[0]);
        //int M = sizeof (a) / sizeof (a[1]);
        //int L = sizeof (b) / sizeof (b[1]);
        double s=0;
        for(i = 0 ; i < a_row_in ; i++) {
            for(j = 0 ; j < b_col_in ; j++) {
                s = 0;
                for(k = 0 ; k < a_col_in ; k++) {
                    s += a_in[i][k]*b_in[k][j];
                }
                c_o[i][j] = s;
            }
        }
    }

    void np_dot_2d_double_complex(int a_row_in ,int a_col_in ,int b_col_in ,double _Complex** a_in, double _Complex** b_in, double _Complex** c_o)
    {
        // usage:
        // c[N][L] = dot product (a[N][M], b[M][L])
        int i, j, k;
        //int N = sizeof (a) / sizeof (a[0]);
        //int M = sizeof (a) / sizeof (a[1]);
        //int L = sizeof (b) / sizeof (b[1]);
        double _Complex s=0;
        for(i = 0 ; i < a_row_in ; i++) {
            for(j = 0 ; j < b_col_in ; j++) {
                s = 0;
                for(k = 0 ; k < a_col_in ; k++) {
                    s += a_in[i][k]*b_in[k][j];
                }
                c_o[i][j] = s;
            }
        }
    }

    int np_argmax_1d_int(int *a, int n)
    {
        if(n <= 0) return -1;
        int i, max_i = 0;
        int max = a[0];
        for(i = 1; i < n; ++i){
            if(a[i] > max){
                max = a[i];
                max_i = i;
            }
        }
        return max_i;
    }

    int np_argmax_1d_float(float *a, int n)
    {
        if(n <= 0) return -1;
        int i, max_i = 0;
        float max = a[0];
        for(i = 1; i < n; ++i){
            if(a[i] > max){
                max = a[i];
                max_i = i;
            }
        }
        return max_i;
    }

    float np_max_1d_float(float *a, int n)
    {
        if(n <= 0) return -1;
        int i;
        float max = a[0];
        for(i = 1; i < n; ++i){
            if(a[i] > max){
                max = a[i];
            }
        }
        return max;
    }

    double np_linalg_norm_double(double a, double b)
    {
        return sqrt(pow(a, 2) + pow(b, 2));
    }

    /*float *scipy_signal_hanning_float(int N ,short itype) 
    {
        // usage:
            //itype = 1 --> periodic
            //itype = 0 --> symmetric= numpy.hanning
            //default itype=0 (symmetric)

        int half, i, idx, n;
        float *w = create_1d_float_array(N);

        //w = (float*) calloc(N, sizeof(float));
        //memset(w, 0, N*sizeof(float));

        if(itype==1)    //periodic function
            n = N-1;
        else
            n = N;

        if(n%2==0)
        {
            half = n/2;
            for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                w[i] = 0.5 * (1 - cos(2*PI*(i+1) / (n+1)));

            idx = half-1;
            for(i=half; i<n; i++) {
                w[i] = w[idx];
                idx--;
            }
        }
        else
        {
            half = (n+1)/2;
            for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                w[i] = 0.5 * (1 - cos(2*PI*(i+1) / (n+1)));

            idx = half-2;
            for(i=half; i<n; i++) {
                w[i] = w[idx];
                idx--;
            }
        }

        if(itype==1)    //periodic function
        {
            for(i=N-1; i>=1; i--)
                w[i] = w[i-1];
            w[0] = 0.0;
        }
        return(w);
    }

    double *scipy_signal_hanning_double(int N ,short itype) 
    {
        // usage:
            //itype = 1 --> periodic
            //itype = 0 --> symmetric= numpy.hanning
            //default itype=0 (symmetric)

        int half, i, idx, n;
        double *w = create_1d_double_array(N);

        //w = (float*) calloc(N, sizeof(float));
        //memset(w, 0, N*sizeof(float));

        if(itype==1)    //periodic function
            n = N-1;
        else
            n = N;

        if(n%2==0)
        {
            half = n/2;
            for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                w[i] = 0.5 * (1 - cos(2*PI*(i+1) / (n+1)));

            idx = half-1;
            for(i=half; i<n; i++) {
                w[i] = w[idx];
                idx--;
            }
        }
        else
        {
            half = (n+1)/2;
            for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                w[i] = 0.5 * (1 - cos(2*PI*(i+1) / (n+1)));

            idx = half-2;
            for(i=half; i<n; i++) {
                w[i] = w[idx];
                idx--;
            }
        }

        if(itype==1)    //periodic function
        {
            for(i=N-1; i>=1; i--)
                w[i] = w[i-1];
            w[0] = 0.0;
        }
        return(w);
    }*/

    void np_hanning_double(int M_i, double* np_hanning_o) 
    {
        int i;
        for(i = 0; i < M_i; i++) {
            np_hanning_o[i] = 0.5*(1 - cos(2*PI*i/(M_i - 1)));
        }
    }

    void scipy_signal_get_window_hann_double(int nx_i ,double* win_o) 
    {
        int i;
        for(i = 0; i < nx_i; i++) {
            win_o[i] = 0.5*(1 - cos(2*PI*(1.0*i)/(1.0*nx_i)));
        }
    }

    void np_pad_reflect_1d_double(double* a_i ,int a_row_i ,int pad_width_i ,double* b_o) 
    {
        int i;
        int j = pad_width_i+a_row_i; //2*pad_width_i; 
        int k = a_row_i - 2;
        for(i = 0; i < pad_width_i; i++) {        
            //b_o[i] = a_i[j-i];               
            b_o[i] = a_i[pad_width_i-i];        
            //b_o[pad_width_i+a_row_i+i] = a_i[k-i];        
            b_o[j+i] = a_i[k-i];
        }

        for(i = 0; i < a_row_i; i++) {        
            b_o[pad_width_i+i] = a_i[i];
        }
    }

    double *np_linspace_double(double start ,double stop ,int num) 
    {
        double *line = create_1d_double_array(num);
        double delta = (stop-start)/(num*1.0-1.0);
        for (int i=0; i<num; i++) {
                line[i] = start + (1.0*i*delta);
        }
        return line;
    }

    double *np_diff_1d_double(double *a_i ,int a_row_i) 
    {
        int b_row = a_row_i - 1;
        double *b = create_1d_double_array(b_row);
        for (int i=0; i<b_row; i++) {
                b[i] = a_i[i+1] - a_i[i];
        }
        return b;
    }

    double **np_subtract_outer_1d_double_in(double *a_i ,int a_row_i ,double *b_i ,int b_row_i) 
    {
        double **c = create_2d_double_array(a_row_i, b_row_i);
        for (int i=0; i<a_row_i; i++) {
            for (int j=0; j<b_row_i; j++) {
                c[i][j] = a_i[i] - b_i[j];
            }
        }
        return c;
    }

    /*
    double _Complex Norm2_1d_double_complex(int row_in ,double _Complex *a)
    {
    	int i;
        //int row_num = sizeof (a) / sizeof (a[0]);
        double _Complex norm2_ans = 0;

    	for (i = 0; i < row_in; i++)
    		norm2_ans+=a[i]*a[i];
    	//norm2_ans = sqrt(norm2_ans);
        norm2_ans = sqrt(pow(creal(norm2_ans), 2) + pow(cimag(norm2_ans), 2));
    	return norm2_ans;
    }

    void qr_decomposition_square_double_complex( int row_in
                                                ,double _Complex **A_in
                                                ,double _Complex **Q_o
                                                ,double _Complex **R_o)
    {
    	int i, j, k, i1, i2;
        //int row_num = sizeof (A) / sizeof (A[0]);
        double _Complex *col_A = create_1d_double_complex_array(row_in);
        double _Complex *col_Q = create_1d_double_complex_array(row_in);
        double _Complex temp   = 0;

    	//施密特正交化
    	for (j = 0; j < row_in; j++)
    	{
    		for (i = 0; i < row_in; i++) { //把A的第j列存入col_A中		
    			col_A[i] = A_in[i][j];
    			col_Q[i] = A_in[i][j];
    		}
    		for (k = 0; k < j; k++) { //計算第j列以前		
    			R_o[k][j] = 0;
    			for (i1 = 0; i1 < row_in; i1++) { //R=Q'A(Q'即Q的轉置) 即Q的第k列和A的第j列做內積
    				R_o[k][j] += col_A[i1]*Q_o[i1][k];//Q的第k列
    			}
    			for (i2 = 0; i2 < row_in; i2++) {
    				col_Q[i2] -= R_o[k][j]*Q_o[i2][k];
    			}
    		}
    
    		temp = Norm2_1d_double_complex(row_in, col_Q);
    		R_o[j][j] = temp;
    		for (i = 0; i < row_in; i++) {
    			//單位化Q
    			Q_o[i][j] = col_Q[i]/temp;
    		}
    	}

    	free(col_A);
    	free(col_Q);
    }

    void eigenvector_from_eigenvalue_square_double_complex(  int row_in
                                                            ,double _Complex **A_in
                                                            ,double _Complex *eigenvalue_in
                                                            ,double _Complex **eigenvector_o)
    {
    	int count, i, j, i1, j1, i2, j2;
        //int row_num = sizeof (A) / sizeof (A[0]);
        double _Complex eValue=0, coe=0, sum1=0, sum2=0;
    	double _Complex **temp = create_2d_double_complex_array(row_in, row_in);
    
        //CopyMatrix(A, &temp);
    	for (count = 0; count < row_in; count++) {
    		eValue = eigenvalue_in[count]; //當前的特徵值
    		copy_2d_double_complex_array(A_in, &temp[0]); //這個每次都要重新複製，因為後面會破壞原矩陣(剛開始沒注意到這個找bug找了好久。。)		
            for (i = 0; i < row_in; i++) {
    			temp[i][i] -= eValue;
    		}        
    		//將temp化為階梯型矩陣(歸一性)對角線值為一
    		for (i = 0; i < row_in - 1; i++) {
    			coe = temp[i][i];            
    			for (j = i; j < row_in; j++) {
    				temp[i][j] /= coe; //讓對角線值為一
    			}            
    			for (i1 = i + 1; i1 < row_in; i1++) {
    				coe = temp[i1][i];				
                    for (j1 = i; j1 < row_in; j1++) {
    					temp[i1][j1] -= coe*temp[i][j1];
    				}
    			}
    		}
    		//讓最後一行為1
    		sum1 = eigenvector_o[row_in-1][count] = 1;
    		for (i2 = row_in - 2; i2 >= 0; i2--) {
    			sum2 = 0;
    			for (j2 = i2 + 1; j2 < row_in; j2++) {
    				sum2 += temp[i2][j2]*eigenvector_o[j2][count];
    			}
    			sum2 = -sum2/temp[i2][i2];
    			sum1 += sum2*sum2;
    			eigenvector_o[i2][count] = sum2;
    		}
    		//sum1 = sqrt(sum1); //當前列的模
    		for (i = 0; i < row_in; i++) {
    			//單位化
    			eigenvector_o[i][count] /= sum1;
    		}
    	}
    
        free_2d_double_complex_array(temp, row_in);
    }

    void np_linalg_eig_double_complex(   int row_in
                                        ,int max_iteration_in // > 0, 最大迭代次數，讓資料更準確
                                        ,double _Complex **A_in
                                        ,double _Complex *eigenvalue_o
                                        ,double _Complex **eigenvector_o)
    {
        //https://www.itread01.com/content/1549432108.html
        int i;
        //int row_num = sizeof (A) / sizeof (A[0]);
        double _Complex **temp = create_2d_double_complex_array(row_in, row_in);
        double _Complex **temp_q = create_2d_double_complex_array(row_in, row_in);
        double _Complex **temp_r = create_2d_double_complex_array(row_in, row_in);

        copy_2d_double_complex_array(A_in, &temp[0]);
        //使用QR分解求矩陣特徵值
    	for (i = 0; i < max_iteration_in; ++i) {
            qr_decomposition_square_double_complex(row_in ,temp, &temp_q[0], &temp_r[0]);
            np_dot_2d_double_complex(temp_r, temp_q, &temp[0]);
        }
        for (i = 0; i < row_in; ++i) {
            eigenvalue_o[i] = temp[i][i];
        }

    	eigenvector_from_eigenvalue_square_double_complex( row_in ,A_in ,eigenvalue_o ,&eigenvector_o[0]);

        free_2d_double_complex_array(temp, row_in);
        free_2d_double_complex_array(temp_q, row_in);
        free_2d_double_complex_array(temp_r, row_in);
    }*/

    double norm_double(double A_in) // =norm() in C++
    {
        return pow(A_in, 2);
    }

    double norm_double_complex(double _Complex A_in) // =norm() in C++
    {
        return pow(creal(A_in), 2) + pow(cimag(A_in), 2);
    }

    void matSca_2d_double_complex_array(int a_row_in ,int a_col_in ,double _Complex scalar_in ,double _Complex** A_in ,double _Complex** C_o) // Scalar multiple of matrix
    {
        //int m = A.size(),   n = A[0].size();
        int i, j;
        //matrix C = A;
        for (i = 0; i < a_row_in; i++ )
        {
            for (j = 0; j < a_col_in; j++ )
                C_o[i][j] = scalar_in*A_in[i][j]; //C[i][j] *= c;
        }
        //return C_o;
    }

    void matLin_2d_double_complex_array( int a_row_in ,int a_col_in ,double _Complex a_scalar_in ,double _Complex** A_in 
                                        ,double _Complex b_scalar_in ,double _Complex** B_in ,double _Complex** C_o)  // Linear combination of matrices
    {
        //int m = A.size(),   n = A[0].size();   assert( B.size() == m && B[0].size() == n );
        int i, j;
        matSca_2d_double_complex_array(a_row_in ,a_col_in ,a_scalar_in ,A_in ,&C_o[0]); //matSca( a, A );  

        for (i = 0; i < a_row_in; i++ )
        {
            for (j = 0; j < a_col_in; j++ )
                C_o[i][j] += b_scalar_in*B_in[i][j];
        }
        //return C_o;
    }

    double matNorm_1d_double_complex_array( int a_col_in ,double _Complex* A_in) // Complex vector norm
    {
        //int m = A.size();
        int i;
        double result = 0.0;
        for (i = 0; i < a_col_in; i++ )
        {
            result += norm_double_complex(A_in[i]); //norm( A[i][j] );
        }
        return sqrt( result );
    }

    double matNorm_2d_double_array( int a_row_in ,int a_col_in ,double** A_in) // matrix norm
    {
        //int m = A.size(),   n = A[0].size();
        int i, j;
        double result = 0.0, norm = 0;
        for (i = 0; i < a_row_in; i++ )
        {
            for (j = 0; j < a_col_in; j++ ) {
                norm = norm_double(A_in[i][j]);
                result += norm;
                //printf("\nc matNorm, result= %.8lf", result);
            }
        }
        return sqrt( result );
    }

    double matNorm_2d_double_complex_array( int a_row_in ,int a_col_in ,double _Complex** A_in) // Complex matrix norm
    {
        //int m = A.size(),   n = A[0].size();
        int i, j;
        double result = 0.0, norm = 0;
        for (i = 0; i < a_row_in; i++ )
        {
            for (j = 0; j < a_col_in; j++ ) {
                norm = norm_double_complex(A_in[i][j]);
                result += norm; //norm( A[i][j] );
                //printf("\nc matNorm, result= %.8lf", result);
            }
        }
        return sqrt( result );
    }

    double subNorm_square_double_complex_array( int a_row_in ,double _Complex** A_in) // Below leading diagonal of square matrix
    {
        //int n = T.size();   assert( T[0].size() == n );
        int i, j;
        double result = 0.0;
        for (i = 1; i < a_row_in; i++ )
        {
            for (j = 0; j < i; j++ ) result += norm_double_complex(A_in[i][j]); //norm( T[i][j] );
        }
        return sqrt( result );
    }

    /*double _Complex shift_square_double_complex_array(int row_in ,double _Complex** A_in) // Wilkinson shift in QR algorithm
    {
        //int N = A.size();
        int i = row_in - 1;
        double e = 0, f = 0;
        double _Complex a = 0, b = 0, c = 0, d = 0, delta = 0, s1 = 0, s2 = 0, s = 0;
    //  while ( i > 0 && abs( A[i][i-1] ) < NEARZERO ) i--;     // Deflation (not sure about this)

        s = 0.0;
        if ( i > 0 )
        {
           a = A_in[i-1][i-1];
           b = A_in[i-1][i];
           c = A_in[i][i-1];
           d = A_in[i][i];        // Bottom-right elements
           delta = csqrt( ( a + d ) * ( a + d ) - 4.0 * ( a * d - b * c ) ); 
           s1 = 0.5 * ( a + d + delta );
           s2 = 0.5 * ( a + d - delta );
           e = norm_double_complex(s1 - d);
           f = norm_double_complex(s2 - d);
           s = ( e < f ? s1 : s2 ); //s = ( norm( s1 - d ) < norm( s2 - d ) ? s1 : s2 );
        }
        return s;
    }

    void Hessenberg_square_double_complex_array( int row_in 
                                                ,double _Complex** A_in 
                                                ,double _Complex** P_o 
                                                ,double _Complex** H_o)
    //http://www.cplusplus.com/forum/beginner/220486/2/
    // Reduce A to Hessenberg form A = P H P-1 where P is unitary, H is Hessenberg
    //                             i.e. P-1 A P = H
    // A Hessenberg matrix is upper triangular plus single non-zero diagonal below main diagonal
    {
        //int N = A.size();
        int i, j, k;
        double xlength = 0, axk = 0, ulength = 0;
        double _Complex rho = 1, xk = 0;
        double _Complex *U = create_1d_double_complex_array(row_in);
        double _Complex **P_tmp = create_2d_double_complex_identity_array(row_in); //P = identity( N );
        double _Complex **PK = create_2d_double_complex_identity_array(row_in);
        double _Complex **H_tmp; //H = A;    
        H_tmp = malloc(row_in*sizeof(double _Complex *));    
        if (H_tmp == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}    
        for(i = 0 ; i < row_in ; i++)
        {
            H_tmp[i] = malloc( row_in*sizeof(double _Complex) );
            if (H_tmp[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }
        for(i = 0 ; i < row_in ; i++) //H = A;
            for(j = 0 ; j < row_in ; j++)
                H_tmp[i][j] = A_in[i][j];

        for (k = 0; k < row_in - 2; k++ )             // k is the working column
        {
            // X vector, based on the elements from k+1 down in the kth column
            for (i = k + 1; i < row_in; i++ ) xlength += norm_double_complex(H_tmp[i][k]); //norm( H[i][k] );
            xlength = sqrt(xlength); 

            // U vector ( normalise X - rho.|x|.e_k )
            //vec U( N, 0.0 );
            xk = H_tmp[k+1][k];
            axk = cabs(xk);
            if ( axk > NEARZERO ) rho = -xk / axk;
            U[k+1] = xk - rho * xlength;
            ulength = norm_double_complex(U[k+1]); //norm( U[k+1] );
            for (i = k + 2; i < row_in; i++ )
            {
               U[i] = H_tmp[i][k];
               ulength += norm_double_complex(U[i]); //norm( U[i] );
            }
            ulength = max( sqrt( ulength ), SMALL );
            for ( i = k + 1; i < row_in; i++ ) U[i] /= ulength;   
            // Householder matrix: P = I - 2 U U*T
            //matrix PK = identity( N );
            for(i = 0 ; i < row_in ; i++) {
                for(j = 0 ; j < row_in ; j++) {
                    if (i == j)
                        PK[i][j] = 1;
                    else
                        PK[i][j] = 0;
                }
            }

            for (i = k + 1; i < row_in; i++ )
            {
               for (j = k + 1; j < row_in; j++ ) PK[i][j] -= 2.0 * U[i] * conj( U[j] );
            }    
            // Transform as PK*T H PK.   Note: PK is unitary, so PK*T = P
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,H_tmp ,PK ,&P_o[0]); //H = matMul( PK, matMul( H, PK ) );
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,PK ,P_o ,&H_o[0]); //H = matMul( PK, matMul( H, PK ) );
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,P_tmp ,PK ,&P_o[0]); //P = matMul( P, PK );
            for(i = 0 ; i < row_in ; i++)
                for(j = 0 ; j < row_in ; j++)
                    H_tmp[i][j] = H_o[i][j];

            for(i = 0 ; i < row_in ; i++)
                for(j = 0 ; j < row_in ; j++)
                    P_tmp[i][j] = P_o[i][j];
        }
        //#ifdef DEBUG_np_linalg_eig_double_complex
        //    printf("\nc Hessenberg, P_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f% + i*.14f", creal(P_o[i][j]), cimag(P_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //        // Check matrix norm of   A v - lambda v
        //        //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
        //    }
        //    
        //    printf("\nc Hessenberg, H_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f% + i*.14f", creal(H_o[i][j]), cimag(H_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //        // Check matrix norm of   A v - lambda v
        //        //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
        //    }
        //#endif    

        free(U);
        free_2d_double_complex_array(P_tmp, row_in);
        free_2d_double_complex_array(PK, row_in);
        free_2d_double_complex_array(H_tmp, row_in);
    }

    void QRFactoriseGivens_square_double_complex_array(  int row_in 
                                                        ,double _Complex** A_in 
                                                        ,double _Complex** Q_o 
                                                        ,double _Complex** R_o)
    {
        // Factorises a Hessenberg matrix A as QR, where Q is unitary and R is upper triangular
        // Uses N-1 Givens rotations
        //int N = A.size(); 
        int i, j, k, m;
        double length = 1;
        double _Complex c = 0, s = 0, c_tmp = 0, s_tmp = 0, cstar = 0, sstar = 0;
        double _Complex **RR = create_2d_double_complex_array(row_in ,row_in);
        double _Complex **QQ = create_2d_double_complex_identity_array(row_in);   
        for(i = 0 ; i < row_in ; i++) //R = A;
            for(j = 0 ; j < row_in ; j++)
                R_o[i][j] = A_in[i][j];  

        for(i = 0 ; i < row_in ; i++) { //Q = identity( N );
            for(j = 0 ; j < row_in ; j++) {
                if (i == j)
                    Q_o[i][j] = 1;
                else
                    Q_o[i][j] = 0;
            }
        }

        for (i = 1; i < row_in; i++ )       // i is the row number
        {
            j = i - 1;                   // aiming to zero the element one place below the diagonal
            if (cabs( R_o[i][j] ) < SMALL ) continue;  
            // Form the Givens matrix        
            c =        R_o[j][j]  ;           
            s = -conj( R_o[i][j] );                       
            c_tmp = norm_double_complex(c);                      
            s_tmp = norm_double_complex(s);                      
            length = sqrt( c_tmp + s_tmp); //sqrt( norm( c ) + norm( s ) );    
            c /= length;               
            s /= length;               
            cstar = conj( c );         //  G*T = ( c* -s )     G = (  c  s  )     <--- j
            sstar = conj( s );         //        ( s*  c )         ( -s* c* )     <--- i

            for(k = 0 ; k < row_in ; k++) //matrix RR = R;
                for(m = 0 ; m < row_in ; m++)
                    RR[k][m] = R_o[k][m];

            for(k = 0 ; k < row_in ; k++) //matrix QQ = Q;
                for(m = 0 ; m < row_in ; m++)
                    QQ[k][m] = Q_o[k][m];

            for (m = 0; m < row_in; m++ ) 
            {
                R_o[j][m] = cstar * RR[j][m] - s     * RR[i][m];
                R_o[i][m] = sstar * RR[j][m] + c     * RR[i][m];    // Should force R[i][j] = 0.0
                Q_o[m][j] = c     * QQ[m][j] - sstar * QQ[m][i];
                Q_o[m][i] = s     * QQ[m][j] + cstar * QQ[m][i];
            }
        }
        //#ifdef DEBUG_np_linalg_eig_double_complex
        //    printf("\nc QRFactoriseGivens, A_in= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f% + i*.14f", creal(A_in[i][j]), cimag(A_in[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //    
        //    printf("\nc QRFactoriseGivens, Q_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f% + i*.14f", creal(Q_o[i][j]), cimag(Q_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //    
        //    printf("\nc QRFactoriseGivens, R_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
        //            printf("%.14f% + i*.14f", creal(R_o[i][j]), cimag(R_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //#endif    

        free_2d_double_complex_array(RR, row_in);
        free_2d_double_complex_array(QQ, row_in);
    }

    void QRHessenberg_square_double_complex_array(   int row_in 
                                                    ,double _Complex** A_in 
                                                    ,double _Complex** P_o 
                                                    ,double _Complex** T_o)
    // Apply the QR algorithm to the matrix A. 
    //
    // Multi-stage:
    //    - transform to a Hessenberg matrix
    //    - apply QR factorisation based on Givens rotations
    //    - uses (single) Wilkinson shift - double-shift version in development
    //
    // Should give a Shur decomposition A = P T P-1 where P is unitary, T is upper triangular
    //                             i.e. P-1 A P = T
    // Eigenvalues of A should be the diagonal elements of T
    // If A is hermitian T would be diagonal and the eigenvectors would be the columns of P
    {
        //const int ITERMAX = 10000;
        //const double TOLERANCE = 1.0e-10;
        int ITERMAX = 10000;
        double TOLERANCE = 1.0e-10;
    
        //int N = A.size();
        int i, j;
        int iter = 1;
        double residual = 1.0, residual_tmp = 0;
        double _Complex mu = 1;
        double _Complex **Q = create_2d_double_complex_array(row_in ,row_in);
        double _Complex **R = create_2d_double_complex_array(row_in ,row_in);
        double _Complex **Told = create_2d_double_complex_array(row_in ,row_in);
        double _Complex **array_identity = create_2d_double_complex_identity_array(row_in);
        double _Complex **P_tmp = create_2d_double_complex_array(row_in ,row_in);
        double _Complex **T_tmp = create_2d_double_complex_array(row_in ,row_in);
    
        //matrix Q( N, vec( N ) ), R( N, vec( N ) ), Told( N, vec( N ) );
        //matrix I = identity( N );
    
        // Stage 1: transform to Hessenberg matrix ( T = Hessenberg matrix, P = unitary transformation )
        Hessenberg_square_double_complex_array(row_in ,A_in ,&P_tmp[0] ,&T_o[0]); //Hessenberg( A, P, T );
        //#ifdef DEBUG_np_linalg_eig_double_complex
        //    printf("\nc QRHessenberg, P_tmp= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) {
        //            printf("%.14f% + i*.14f", creal(P_tmp[i][j]), cimag(P_tmp[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //    
        //    printf("\nc QRHessenberg, T_o= \n");
        //    for (i = 0; i < row_in; i++) {
        //        for (j = 0; j < row_in; j++ ) {
        //            printf("%.14f% + i*.14f", creal(T_o[i][j]), cimag(T_o[i][j])); printf(", ");
        //        } 
        //        if (i%(row_in-1)==0) {
        //            printf("\n");
        //        }
        //    }
        //#endif 
    
        // Stage 2: apply QR factorisation (using Givens rotations)
        while( residual > TOLERANCE && iter < ITERMAX )
        {
            for(i = 0 ; i < row_in ; i++) //Told = T;
                for(j = 0 ; j < row_in ; j++)
                    Told[i][j] = T_o[i][j];
    
            // Spectral shift
            mu = shift_square_double_complex_array(row_in ,T_o); //cmplx mu = shift( T );
            if (cabs( mu ) < NEARZERO ) mu = 1.0;   // prevent unitary matrices causing a problem
            matLin_2d_double_complex_array(row_in ,row_in ,1.0 ,T_o ,-mu ,array_identity ,&T_tmp[0]); //T = matLin( 1.0, T, -mu, I );
    
            // Basic QR algorithm by Givens rotation
            QRFactoriseGivens_square_double_complex_array(row_in ,T_tmp ,&Q[0] ,&R[0]); //QRFactoriseGivens( T, Q, R );
            //#ifdef DEBUG_np_linalg_eig_double_complex
            //    printf("\nc QRHessenberg, mu= %.14f% + i*.14f", creal(mu), cimag(mu)); printf("\n");
            //    printf("\nc QRHessenberg, cabs( mu )= %.8lf", cabs( mu )); printf("\n");
            //    printf("\nc QRHessenberg, matLin, T_tmp= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f% + i*.14f", creal(T_tmp[i][j]), cimag(T_tmp[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    
            //    printf("\nc QRHessenberg, QRFactoriseGivens, Q= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f% + i*.14f", creal(Q[i][j]), cimag(Q[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    
            //    printf("\nc QRHessenberg, QRFactoriseGivens, R= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f% + i*.14f", creal(R[i][j]), cimag(R[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //#endif
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,R ,Q ,&T_tmp[0]); //T = matMul( R, Q );
            np_dot_2d_double_complex(row_in ,row_in ,row_in ,P_tmp ,Q ,&P_o[0]); //P = matMul( P, Q );
    
            // Reverse shift
            matLin_2d_double_complex_array(row_in ,row_in ,1.0 ,T_tmp ,mu ,array_identity ,&T_o[0]); //T = matLin( 1.0, T, mu, I );
            //#ifdef DEBUG_np_linalg_eig_double_complex
            //    printf("\nc QRHessenberg, matMul( R, Q ), T_tmp= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f% + i*.14f", creal(T_tmp[i][j]), cimag(T_tmp[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    
            //    printf("\nc QRHessenberg, matMul( P, Q ), P_o= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f% + i*.14f", creal(P_o[i][j]), cimag(P_o[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    
            //    printf("\nc QRHessenberg, matLin, T_o= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f% + i*.14f", creal(T_o[i][j]), cimag(T_o[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //#endif
    
            // Calculate residuals
            //residual = matNorm( matLin( 1.0, T, -1.0, Told ) ); // change on iteration
            matLin_2d_double_complex_array(row_in ,row_in ,1.0 ,T_o ,-1.0 ,Told ,&T_tmp[0]);
            residual = matNorm_2d_double_complex_array(row_in ,row_in ,T_tmp);
            residual_tmp = subNorm_square_double_complex_array(row_in ,T_o);
            //#ifdef DEBUG_np_linalg_eig_double_complex
            //    printf("\nc QRHessenberg, Calculate residuals, matLin, T_tmp= \n");
            //    for (i = 0; i < row_in; i++) {
            //        for (j = 0; j < row_in; j++ ) {
            //            printf("%.14f% + i*.14f", creal(T_tmp[i][j]), cimag(T_tmp[i][j])); printf(", ");
            //        } 
            //        if (i%(row_in-1)==0) {
            //            printf("\n");
            //        }
            //    }
            //    printf("\nc QRHessenberg, Calculate residuals, residual matNorm= %.8lf", residual); printf("\n");
            //    printf("\nc QRHessenberg, residual_tmp= %.8lf", residual_tmp); printf("\n");
            //#endif
            residual += residual_tmp; //residual += subNorm( T ); // below-diagonal elements
    //      cout << "\nIteration: " << iter << "   Residual: " << residual << endl;
            iter++; 
            //#ifdef DEBUG_np_linalg_eig_double_complex
            //    printf("\nc QRHessenberg, Calculate residuals, residual= %.8lf", residual); printf("\n");
            //#endif

        }
        //cout << "\nQR iterations: " << iter << "   Residual: " << residual << endl;
        //if ( residual > TOLERANCE ) cout << "***** WARNING ***** QR algorithm not converged\n";
        if ( residual > TOLERANCE ) printf("\n***** WARNING ***** QR algorithm not converged");
        printf("\nc QRHessenberg, iter= %d", iter); printf("\n");
        printf("\nc QRHessenberg, residual= %.14f", residual); printf("\n");
        printf("\nc QRHessenberg, residual_tmp= %.14f", residual_tmp); printf("\n");

        #ifdef DEBUG_np_linalg_eig_double_complex
            printf("\nc QRHessenberg, P_o= \n");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f% + i*.14f", creal(P_o[i][j]), cimag(P_o[i][j])); printf(", ");
                } 
                if (i%(row_in-1)==0) {
                    printf("\n");
                }
            }

            printf("\nc QRHessenberg, T_o= \n");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f% + i*.14f", creal(T_o[i][j]), cimag(T_o[i][j])); printf(", ");
                } 
                if (i%(row_in-1)==0) {
                    printf("\n");
                }
            }
        #endif

        free_2d_double_complex_array(Q, row_in);
        free_2d_double_complex_array(R, row_in);
        free_2d_double_complex_array(Told, row_in);
        free_2d_double_complex_array(array_identity, row_in);
        free_2d_double_complex_array(P_tmp, row_in);
        free_2d_double_complex_array(T_tmp, row_in);
    }

    void eigenvectorUpper_square_double_complex_array(int row_in ,double _Complex** T_in ,double _Complex** E_o)
    // Find the eigenvectors of upper-triangular matrix T; returns them as column vectors of matrix E
    // The eigenvalues are necessarily the diagonal elements of T
    // NOTE: if there are repeated eigenvalues, then THERE MAY NOT BE N EIGENVECTORS
    {
        //bool fullset = true;
        //int N = T.size();
        //E = matrix( N, vec( N, 0.0 ) );               // Columns of E will hold the eigenvectors    
        int i, j, k, L, ok = 1;
        double length = 1;
        double _Complex lambda = 0;
        double _Complex *V = create_1d_double_complex_array(row_in);
        double _Complex **TT = create_2d_double_complex_array(row_in ,row_in);

        for(i = 0 ; i < row_in ; i++) //matrix TT = T;
            for(j = 0 ; j < row_in ; j++)
                TT[i][j] = T_in[i][j];

        for (L = row_in - 1; L >= 0; L-- )            // find Lth eigenvector, working from the bottom
        {
            ok = 1; //bool ok = true;
            lambda = T_in[L][L]; //cmplx lambda = T[L][L];
            for (k = 0; k < row_in; k++ ) {
                V[k] = 0; //vec V( N, 0.0 );
                TT[k][k] = T_in[k][k] - lambda; // TT = T - lambda I
            } // Solve TT.V = 0
            V[L] = 1.0;                                // free choice of this component
            for (i = L - 1; i >= 0; i-- )         // back-substitute for other components
            {
                V[i] = 0.0;
                for (j = i + 1; j <= L; j++ ) V[i] -= TT[i][j] * V[j];
                if ( cabs( TT[i][i] ) < NEARZERO )       // problem with repeated eigenvalues
                {
                    if ( cabs( V[i] ) > NEARZERO ) ok = 0; //false;     // incomplete set; use the lower-L one only
                    V[i] = 0.0;
                }
                else
                {
                    V[i] = V[i] / TT[i][i];
                }
            }    
            if ( ok==1 )
            {
                // Normalise
                length = matNorm_1d_double_complex_array(row_in ,V); //double length = vecNorm( V );    
                for (i = 0; i <= L; i++ ) E_o[i][L] = V[i] / length;
            }
            else
            {
                //fullset = false;
                for (i = 0; i <= L; i++ ) E_o[i][L] = 0.0;
            }
        }   
        //if ( !fullset )
        //{
        //    cout << "\n***** WARNING ***** Can't find N independent eigenvectors\n";
        //    cout << "   Some will be set to zero\n";
        //}
        if (ok==0) printf("\n***** WARNING ***** Can't find N independent eigenvectors, some will be set to zero"); 

        #ifdef DEBUG_np_linalg_eig_double_complex
            printf("\nc eigenvectorUpper, E_o= \n");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f% + i*.14f", creal(E_o[i][j]), cimag(E_o[i][j])); printf(", ");
                } 
                if (i%(row_in-1)==0) {
                    printf("\n");
                }
                // Check matrix norm of   A v - lambda v
                //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
            }
        #endif  

        free(V);
        free_2d_double_complex_array(TT, row_in);
        //return fullset;
    }

    void np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double _Complex **A_in
                                                    ,double _Complex *eigenvalue_o
                                                    ,double _Complex **eigenvector_o)
    {   
        int i, j;
        double _Complex **P_o = create_2d_double_complex_array(row_in ,row_in);
        double _Complex **T_o = create_2d_double_complex_array(row_in ,row_in);
        double _Complex **E_o = create_2d_double_complex_array(row_in ,row_in);

        QRHessenberg_square_double_complex_array(row_in ,A_in ,&P_o[0] ,&T_o[0]);
        eigenvectorUpper_square_double_complex_array(row_in ,T_o ,&E_o[0]); //,&eigenvector_o[0]); //,&E_o[0]);
        np_dot_2d_double_complex(row_in ,row_in ,row_in ,P_o ,E_o ,&eigenvector_o[0]); //matMul( P, E );

        #ifdef DEBUG_np_linalg_eig_double_complex
            printf("\nsub c eigenvalue_o= ");
        #endif
        for (i = 0; i < row_in; i++) {
            eigenvalue_o[i] = T_o[i][i];
            #ifdef DEBUG_np_linalg_eig_double_complex
                printf("\n");
                printf("%.14f% + i*.14f", creal(eigenvalue_o[i]), cimag(eigenvalue_o[i])); printf(", ");
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
            #endif
        }

        #ifdef DEBUG_np_linalg_eig_double_complex
            printf("\nsub c eigenvector_o E_o= ");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f% + i*.14f", creal(E_o[i][j]), cimag(E_o[i][j])); printf(", ");
                } 
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
                // Check matrix norm of   A v - lambda v
                //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
            }

            printf("\nsub c eigenvector_o= ");
            for (i = 0; i < row_in; i++) {
                for (j = 0; j < row_in; j++ ) { //V[j] = E[j][i];
                    printf("%.14f% + i*.14f", creal(eigenvector_o[i][j]), cimag(eigenvector_o[i][j])); printf(", ");
                } 
                if ((i+1)%row_in==0) {
                    printf("\n");
                }
                // Check matrix norm of   A v - lambda v
                //cout << "Check error: " << vecNorm( vecLin( 1.0, matVec( A, V ), -lambda, V ) ) << endl;
            }
        #endif

        free_2d_double_complex_array(P_o, row_in);
        free_2d_double_complex_array(T_o, row_in);
        //free_2d_double_complex_array(E_o, row_in);
    }*/

    void np_linalg_eig_square_double_complex_array(  int row_in
                                                    ,double _Complex **A_in
                                                    ,double _Complex *eigenvalue_o
                                                    ,double _Complex **eigenvector_o)
    {
        //int i;
       // double _Complex **A_in = deflate_2d_double_complex_array(flat_A_in, row_in, row_in);
       // double _Complex **eigenvector_o = create_2d_double_complex_array(row_in, row_in);
        double _Complex *flat_A_in = flatten_2d_double_complex_array(A_in, row_in, row_in);
        double _Complex *flat_eigenvector_o = create_1d_double_complex_array(row_in*row_in);
    
        /*printf("\nc np A_in= ");
        for (int i = 0; i < row_in; i++) {
            for (int j = 0; j < row_in; j++) {
                printf("%.14f% + i*.14f", creal(A_in[i][j]), cimag(A_in[i][j])); printf(", ");
                    if ((j+1)%2==0) {
                        printf("\n");
                    }
            }
        }
        printf("\nc np flat_A_in= ");
        for (int i = 0; i < row_in*row_in; i++) {
            printf("%.14f% + i*.14f", creal(flat_A_in[i]), cimag(flat_A_in[i])); printf(", ");
            if ((i+1)%2==0) {
                printf("\n");
            }
        }*/
       // #ifdef DEBUG_np_linalg_eig_double_complex
            //clock_t time0, time1;
            //time0 = clock();
       // #endif    
        flat_np_linalg_eig_square_double_complex_array(   row_in
                                                    ,flat_A_in //,A_in
                                                    ,&eigenvalue_o[0]
                                                    ,&flat_eigenvector_o[0] //,&eigenvector_o[0]
                                                    );
       // #ifdef DEBUG_np_linalg_eig_double_complex
            //time1 = clock();
            //printf("\n spe c np_linalg_eig time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
       // #endif

        #ifdef DEBUG_np_linalg_eig_double_complex
            //printf("\nflat_A_in[0]*flat_A_in[1]= %.14f% + i*.14f", creal(flat_A_in[0]*flat_A_in[1]), cimag(flat_A_in[0]*flat_A_in[1]));
            //printf("\nflat_A_in[0]/flat_A_in[1]= %.14f% + i*.14f", creal(flat_A_in[0]/flat_A_in[1]), cimag(flat_A_in[0]/flat_A_in[1]));
            //printf("\nflat_A_in[0]^2= %.14f% + i*.14f", creal(cpow(flat_A_in[0],2)), cimag(cpow(flat_A_in[0],2)));
           /* printf("\nc square array in= \n");
            for (int i = 0; i < row_in; i++) {
                for (int j = 0; j < row_in; j++) {
                    printf("%.14f% + i*.14f", creal(A_in[i][j]), cimag(A_in[i][j])); printf(", ");
                    if (j%row_in==row_in-1) {
                        printf("\n");
                    }
                }
            }*/
            printf("\nc eigenvalue_o= ");
            for (int i = 0; i < row_in; i++) {
                printf("%.14f% + i*.14f", creal(eigenvalue_o[i]), cimag(eigenvalue_o[i])); printf(", ");
            }   
        #endif

        for (int i = 0; i < row_in; i++) {
            for (int j = 0; j < row_in; j++) {
                eigenvector_o[i][j] = flat_eigenvector_o[i*row_in+j];
                #ifdef DEBUG_np_linalg_eig_double_complex
                    printf("%.14f% + i*.14f", creal(eigenvector_o[i][j]), cimag(eigenvector_o[i][j])); printf(", ");
                    if (j%row_in==row_in-1) {
                        printf("\n");
                    }
                #endif
            }
        }

        free(flat_A_in);
        free(flat_eigenvector_o);

    }

    void stable_sort_1d_double(int col_i ,double* p_i ,int* stable_sort_o)
    {
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, j;
        int min = 0, m = 0;
        double compare = 0, key = 0;
        double *arr_sort = create_and_copy_1d_double_array(col_i ,p_i);

        for (i = 0; i < col_i; i++)    
    		stable_sort_o[i] = i;     

        //https://www.geeksforgeeks.org/stable-selection-sort/
        for (i = 0; i < col_i - 1; i++) {  
            // Find minimum element from arr[i] to arr[n - 1]. 
            min = i;
            //#ifdef DEBUG_np_lexsort_2d_double
            //    printf("\nsub c step 2.1 i= %d", i); 
            //#endif
            for (j = i + 1; j < col_i; j++) {
                // if (flag == 1) {
                    compare = arr_sort[min] - arr_sort[j];
                // }
                // else {
                //     compare = arr_sort[j] - arr_sort[min];
                // }

                if (compare > 0)
                    min = j;
            }

            // Move minimum element at current i. 
            key = arr_sort[min];
            m = stable_sort_o[min];
            while (min > i)  
            { 
                arr_sort[min] = arr_sort[min - 1]; 
                stable_sort_o[min] = stable_sort_o[min - 1];
                min--; 
            } 
            arr_sort[i] = key;
            stable_sort_o[i] = m; 
        }

        free(arr_sort);

    }

    //void np_argsort_1d_double_complex(int size ,double _Complex* p_in ,int flag ,double _Complex* sort_o ,int* argsort_o)
    void np_argsort_1d_double_complex(int size ,double _Complex* p_in ,int* argsort_o)
    {
        // #ifdef DEBUG_np_lexsort_2d_double
            clock_t time0, time1;
            time0 = clock();
        // #endif
    
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, N=2, M=size;
        //int size = sizeof (p) / sizeof (p[0]);

        //copy_1d_double_complex_array( p, sort_o);
        double **array;    
        array = (double**)malloc(N*sizeof(double _Complex *));    
        if (array == NULL) {
    		fprintf(stderr, "Out of memory");
    		exit(0);
    	}

        for(i = 0 ; i < N ; i++)
        {
            array[i] = (double*)malloc( M*sizeof(double _Complex) );
            if (array[i] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }

        for(i = 0 ; i < M ; i++) {
            array[0][i] = cimag(p_in[i]);
            array[1][i] = creal(p_in[i]);
        }

    	np_lexsort_2d_double_v2(N ,M ,array ,&argsort_o[0]);

        //for (i = 0; i < size - 1; i++) {
    	//	k = i;
    	//	for (j = i + 1; j < size; j++) {
    	//		if (flag == 1) {
    	//			if (sort_o[k] > sort_o[j]) {
    	//				k = j;
    	//			}
    	//		}
    	//		else {
    	//			if (sort_o[k] < sort_o[j]) {
    	//				k = j;
    	//			}
    	//		}
    	//	}
    	//	if (k != i) {			
    	//		temp = sort_o[i];
    	//		sort_o[i] = sort_o[k];
    	//		sort_o[k] = temp;			
    	//		index = argsort_o[i];
    	//		argsort_o[i] = argsort_o[k];
    	//		argsort_o[k] = index;
    	//	}
    	//}
        #ifdef DEBUG_np_lexsort_2d_double
            printf("\nsub c argsort_o= ");
            for (i = 0; i < M; i++) {
                printf("%d", argsort_o[i]); printf(", ");
            }
        #endif

        //long vmrss, vmsize;
        //get_memory_usage_kb(&vmrss, &vmsize);
        //printf("\nspe c v1 Current memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n", vmrss, vmsize);

        free_2d_double_array(array, N);
    
        // #ifdef DEBUG_np_lexsort_2d_double
            time1 = clock();
            printf("\n spe c v1 np_argsort time= %f\n", (double)(time1-time0)/(CLOCKS_PER_SEC));
        // #endif
    
    }

    /*
    //void np_lexsort_2d_double(double** p ,int flag ,int* lexsort_o)
    void np_lexsort_2d_double(double** p ,int* lexsort_o)
    {
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, j, k, m=0, max_index_eq=0, min=0, has_eq_element=0, cnt_index_eq=1;
        int row = sizeof (p) / sizeof (p[0]);
        int col = sizeof (p) / sizeof (p[1]);
        double key=0, compare=0;
        double *pri_sort_key = create_1d_double_array(col);
        double *index_eq = create_1d_double_array(col);

        for(i = 0 ; i < col ; i++) // primary sort key: copy last row of p
            pri_sort_key[i] = p[row-1][i];

        for (i = 0; i < col; i++) 
    		lexsort_o[i] = i;

        //https://www.geeksforgeeks.org/stable-selection-sort/
        //stableSelectionSort(int a[], int n) 
        // Iterate through array elements 
        for (i = 0; i < col - 1; i++) {  
            // Find minimum element from arr[i] to arr[n - 1]. 
            min = i; 
            for (j = i + 1; j < col; j++) {
                //if (a[min] > a[j]) 
                //    min = j;
               // if (flag == 1) {
                    compare = pri_sort_key[min] - pri_sort_key[j];
               // }
               // else {
               //     compare = pri_sort_key[j] - pri_sort_key[min];
               // }

                if (compare > 0)
                    min = j;

                // equal element indicator   
                if (pri_sort_key[i] == pri_sort_key[j]) {
                    //if (index_eq[i] == 0) {
                    //    max_index_eq += 1;
                    //    index_eq[i] = max_index_eq;
                    //}
                    //
                    //if (index_eq[j] == 0)
                    //    index_eq[j] = index_eq[i];
                    has_eq_element = 1;
                }
            }
    
            // Move minimum element at current i. 
            key = pri_sort_key[min];
            m = lexsort_o[min];
            while (min > i)  
            { 
                pri_sort_key[min] = pri_sort_key[min - 1]; 
                lexsort_o[min] = lexsort_o[min - 1];
                min--; 
            } 
            pri_sort_key[i] = key;
            lexsort_o[i] = m; 
        }

        cnt_row = row - 1;

        while ((cnt_row > 0) && (has_eq_element == 1)) {
            // mark equal element for cnt_row
            if (cnt_row == row - 1)
                //col_tmp = col - 1;
            else {
                //m = arr_eq_par[cnt_row+1][col/2+1]; // (cnt_row+1)'s cnt_eq_set
                //col_tmp = arr_eq_par[cnt_row+1][2*m+1]; // (cnt_row+1)'s cnt_eq_set's eq_set inc
                arr_eq_par[cnt_row][col/2]; // reset cnt_row's max_index_eq
                arr_eq_par[cnt_row][col/2+1]; // reset cnt_row's cnt_eq_set
            }

            for (i = eq_start_pt; i < eq_start_pt + eq_inc; i++) {
                j = i+1;
                if (arr_sort[i] == arr_sort[j]) {
                    if (arr_index_eq[i] == 0) {
                        m = arr_eq_par[cnt_row][col/2]; // max_index_eq
                        arr_eq_par[cnt_row][col/2] ++; // max_index_eq ++
                        arr_index_eq[i] = 1;
                        arr_eq_par[cnt_row][2*m] = i; // save (max_index_eq-1)'s eq_set start pt
                        arr_eq_par[cnt_row][2*m+1] ++; // (max_index_eq-1)'s eq_set inc ++
                    }

                    if (arr_index_eq[j] == 0) {
                        arr_index_eq[j] = 1;
                        arr_eq_par[cnt_row][2*m+1] ++; // max_index_eq's eq_set inc ++
                    }
                }
            }

            // sort multiple set of equal elements in the last row by multiple columns in multiple rows
            for (k = 0; k < max_index_eq; k++) { // multiple set of equal elements
                record_start = 1;
                cnt_tmp = 0;
                row_tmp = row - 1;
                has_eq_element = 0;
                stop_row_compare = 0;
                for (i = 0; i < col; i++) {
                    if (index_eq[i] == k+1) {
                        if (record_start == 1) {
                            start_index_eq_key = i; // record start position of column in the last row
                            record_start = 0;
                        }
                        //eq_key[cnt_tmp] = pri_sort_key[i];
                        index_eq_key[cnt_tmp] = lexsort_o[i]; // copy index of this set of equal elements
                        cnt_tmp ++; // count the amount of this set of equal elements
                    }
                }

                while (stop_row_compare == 0) {
                    for (i = 0; i < cnt_tmp; i++) {
                        row_tmp --;
                        m = index_eq_key[i];
                        eq_key[i] = p[row_tmp][m]; // copy this set of equal elements for sorting
                    }

                    // stable sort
                    for (i = 0; i < cnt_tmp; i++) {
                        min = i; 
                        for (j = i + 1; j < col; j++) {
                            compare = eq_key[min] - eq_key[j];

                            if (compare > 0)
                                min = j;  

                            if (eq_key[i] == eq_key[j])
                                has_eq_element = 1;

                        }

                        // Move minimum element at current i. 
                        key = eq_key[min];
                        m = index_eq_key[min];
                        while (min > i)  
                        { 
                            eq_key[min] = eq_key[min - 1]; 
                            index_eq_key[min] = index_eq_key[min - 1];
                            min--; 
                        } 
                        eq_key[i] = key;
                        index_eq_key[i] = m; 
                    }

                    if ((has_eq_element == 0) || (row_tmp == 0))
                        stop_row_compare = 1;
                }            

                // copy sorting result of this set of equal elements to output
                for (i = 0; i < cnt_tmp; i++) {
                    m = i + start_index_eq_key;
                    lexsort_o[m] = index_eq_key[i];
                }
            }
        }
    
    } */

    //void np_lexsort_2d_double_v1(double** p ,int flag ,int* lexsort_o)
    void np_lexsort_2d_double_v1(int row ,int col ,double** p_in ,int* lexsort_o)
    {
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, j;
        int min = 0, m = 0;
        int finish = 0, has_eq_element = 0, exec_step_3 = 0;
        int cnt_eq_set_num = 0, eq_start_pt = 0, eq_inc = 0;
        //int row = sizeof (p_in[0][0]) / sizeof (p_in);
        //int col = sizeof (p_in) + 1;
        int cnt_row = row - 1;
        int half_col = col/2; // == max eq set
        int col_even = 2*half_col;
        int loop_count = col;
        double compare = 0, key = 0;
        double *arr_sort = create_1d_double_array(col);
        int *arr_index_sort = create_1d_int_array(col);
        int *arr_index_eq = create_1d_int_array(col);
        int **arr_eq_par = create_2d_int_array(row, col_even + 2);

        for (i = 0; i < col; i++) 
    		//lexsort_o[i] = i;   
    		arr_index_sort[i] = i;     


        while (finish == 0) {
            // step 1. cp cnt_row to memory, including index and elements
            if (cnt_row == row - 1) {           
                for(i = 0 ; i < col ; i++) // cp p_in[row-1][0 ~ col-1] to arr_sort[0 ~ col-1]
                    arr_sort[i] = p_in[row-1][i];
            } else {
                cnt_eq_set_num = arr_eq_par[cnt_row+1][col_even+1]; // (cnt_row+1)'s cnt_eq_set
                eq_start_pt = arr_eq_par[cnt_row+1][2*cnt_eq_set_num]; // (cnt_row+1)'s start pt
                eq_inc = arr_eq_par[cnt_row+1][2*cnt_eq_set_num+1]; // (cnt_row+1)'s inc            

                for (i = 0; i < eq_inc; i++) { // cp lexsort_o[eq_start_pt ~ (eq_start_pt + eq_inc - 1)] to arr_index_sort[0 ~ eq_inc - 1]
                    arr_index_sort[i] = lexsort_o[eq_start_pt+i];
                }

                for (i = 0; i < eq_inc; i++) { // cp p_in[cnt_row][j] to arr_sort[i]
                    j = arr_index_sort[i];
                    arr_sort[i] = p_in[cnt_row][j];
                }
            }
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 1 cnt_row= %d", cnt_row);
                printf(", cnt_eq_set_num= %d", cnt_eq_set_num);
                printf(", eq_start_pt= %d", eq_start_pt);
                printf(", eq_inc= %d", eq_inc);
                printf("\nsub c step 1 arr_sort= ");
                for (i = 0; i < col; i++) {
                    printf("%f", arr_sort[i]); printf(", ");
                }
                printf("\nsub c step 1 arr_index_sort= ");
                for (i = 0; i < col; i++) {
                    printf("%d", arr_index_sort[i]); printf(", ");
                }
            #endif

            //step 2. sort
            //step 2.1 stable sort: https://www.geeksforgeeks.org/stable-selection-sort/
            // Iterate through array elements
            if (cnt_row < row - 1)
                loop_count = eq_inc;

            if (arr_sort[loop_count - 1] > 1)
                key = arr_sort[loop_count - 1] - 1;
            else            
                key = arr_sort[loop_count - 1] + 1;

            for (i = 0; i < loop_count - 1; i++) {  
                // Find minimum element from arr[i] to arr[n - 1]. 
                min = i;
                //#ifdef DEBUG_np_lexsort_2d_double
                //    printf("\nsub c step 2.1 i= %d", i); 
                //#endif
                for (j = i + 1; j < loop_count; j++) {
                   // if (flag == 1) {
                        compare = arr_sort[min] - arr_sort[j];
                   // }
                   // else {
                   //     compare = arr_sort[j] - arr_sort[min];
                   // }

                    if (compare > 0)
                        min = j;

                    // equal element indicator   
                    //if (arr_sort[i] == arr_sort[j]) {  
                    if (i == 0) {
                        if ((arr_sort[i] == arr_sort[j]) || (key == arr_sort[j])) {
                            //if (index_eq[i] == 0) {
                            //    max_index_eq += 1;
                            //    index_eq[i] = max_index_eq;
                            //}
                            //
                            //if (index_eq[j] == 0)
                            //    index_eq[j] = index_eq[i];
                            has_eq_element = 1;
                        }

                        key = arr_sort[j];
                    }
                    //#ifdef DEBUG_np_lexsort_2d_double
                    //    printf("\nsub c step 2.1 j= %d", j);
                    //    printf(", arr_sort[min]= %f", arr_sort[min]);
                    //    printf(", arr_sort[i]= %f", arr_sort[i]);
                    //    printf(", arr_sort[j]= %f", arr_sort[j]);
                    //#endif
                }

                // Move minimum element at current i. 
                key = arr_sort[min];
                m = arr_index_sort[min];
                while (min > i)  
                { 
                    arr_sort[min] = arr_sort[min - 1]; 
                    arr_index_sort[min] = arr_index_sort[min - 1];
                    min--; 
                } 
                arr_sort[i] = key;
                arr_index_sort[i] = m; 
            }
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 2.1 loop_count= %d", loop_count);
                printf("\nsub c step 2.1 has_eq_element= %d", has_eq_element);
                printf("\nsub c step 2.1 arr_sort= ");
                for (i = 0; i < loop_count; i++) {
                    printf("%f", arr_sort[i]); printf(", ");
                }
                printf("\nsub c step 2.1 arr_index_sort= ");
                for (i = 0; i < loop_count; i++) {
                    printf("%d", arr_index_sort[i]); printf(", ");
                }
            #endif

            //step 2.2 cp sort result to lexsort_o
            /*if (cnt_row == row - 1) {          
                for(i = 0 ; i < col ; i++) // cp arr_index_sort to lexsort_o 
                    lexsort_o[i] = arr_index_sort[i];
            } else {
                for (i = 0; i < eq_inc; i++)  // cp arr_index_sort to lexsort_o
                    lexsort_o[eq_start_pt+i] = arr_index_sort[i];
            }*/
            for (i = 0; i < loop_count; i++)  // cp arr_index_sort to lexsort_o
                lexsort_o[eq_start_pt+i] = arr_index_sort[i];

            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 2.2 lexsort_o= ");
                for (i = 0; i < col; i++) {
                    printf("%d", lexsort_o[i]); printf(", ");
                }
            #endif

            //step 2.3 condition: finish, continueously from bottom to top or begin from top to bottom
            if ((has_eq_element == 1) && (cnt_row > 0)) {
                exec_step_3 = 1; // go to step 3
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 2.3 (has_eq_element == 1) && (cnt_row > 0)");
                #endif
            } else {
                //search (max_eq_set - cnt_eq_set > 0) from cnt_row+1 to row-1 if (cnt_row < row - 1):
                if (cnt_row == row -1) {
                    finish = 1;
                    exec_step_3 = 0;
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 finish = 1; exec_step_3 = 0;");
                    #endif
                } else {
                    // search (max_eq_set - (cnt_eq_set + 1) > 0) 
                    // from cnt_row+1(for cnt_row == 0) or cnt_row(for cnt_row > 0) to row-1: // top down search  
                    //if (cnt_row == 0) search_st_row = cnt_row+1
                    //else search_st_row = cnt_row
                    //for (i==search_st_row; i < row; i++)
                    for (i = cnt_row+1; i < row; i++) {
                        if (arr_eq_par[i][col_even] - (arr_eq_par[i][col_even+1] + 1) > 0) {
                            cnt_row = i - 1;
                            //for (j==cnt_row; j >= 0; j--) // clear max_eq_set and cnt_eq_set from cnt_row to 0
                            //    arr_eq_par[j][col/2] = 0
                            //    arr_eq_par[j][col/2+1] = 0
                            arr_eq_par[i][col_even+1] ++;
                            finish = 0;
                            exec_step_3 = 0; // go to step 1
                            #ifdef DEBUG_np_lexsort_2d_double
                                printf("\nsub c step 2.3 i= %d", i);
                                printf("\nsub c step 2.3 max_index_eq= %d", arr_eq_par[i][col_even]);
                                printf("\nsub c step 2.3 cnt_eq_set= %d", arr_eq_par[i][col_even+1]);
                            #endif
                            break;
                        } else if (i == row - 1) { // note: arr_eq_par[i][col/2] - (arr_eq_par[i][col/2+1] + 1) == 0
                            finish = 1;
                            exec_step_3 = 0; // skip step 3
                            break;
                        }
                    }
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 cnt_row= %d", cnt_row);
                        printf("\nsub c step 2.3 finish= %d", finish);
                        printf("\nsub c step 2.3 exec_step_3= %d", exec_step_3);
                    #endif
                }
            }

            has_eq_element = 0;

            //step 3. mark eq and save associated parameters        
            if (exec_step_3 == 1) {
                //step 3.1 mark equal elements
                //& step 3.2 save cnt_row's start pt and inc for each eq set and max_eq_set (arr_eq_par[cnt_row][0 ~ 2*max_eq_set - 1, col/2])
                //search eq elements range
                if (cnt_row == row - 1) 
                    loop_count = col - 1;
                else
                    loop_count = eq_inc;

                if (cnt_row < row - 1) {
                    //m = arr_eq_par[cnt_row+1][col/2+1]; // (cnt_row+1)'s cnt_eq_set
                    //col_tmp = arr_eq_par[cnt_row+1][2*m+1]; // (cnt_row+1)'s cnt_eq_set's eq_set inc
                    arr_eq_par[cnt_row][col_even] = 0; // reset cnt_row's max_index_eq
                    arr_eq_par[cnt_row][col_even+1] = 0; // reset cnt_row's cnt_eq_set                

                    for (i = 0; i < loop_count; i++) {
                        m = 2*i;
                        arr_eq_par[cnt_row][m+1] = 0; // reset cnt_row's cnt_eq_set's eq_set inc
                        arr_index_eq[i] = 0; // reset arr_index_eq
                    }

                    loop_count --;
                }
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 3 search eq elements range= %d", loop_count+1);
                #endif

                //m = 0;  
                //for (i = 0; i < eq_inc; i++) {  
                for (i = 0; i < loop_count; i++) {
                    j = i+1;
                    if (arr_sort[i] == arr_sort[j]) {
                        if (arr_index_eq[i] == 0) { // find an eq_set start pt
                            m = 2*arr_eq_par[cnt_row][col_even]; // 2*max_index_eq
                            arr_eq_par[cnt_row][col_even] ++; // max_index_eq ++
                            arr_index_eq[i] = 1; // mark this eq element
                            //arr_eq_par[cnt_row][m] = i; // save (max_index_eq-1)'s eq_set start pt
                            arr_eq_par[cnt_row][m] = eq_start_pt + i; // save (max_index_eq-1)'s eq_set start pt
                            arr_eq_par[cnt_row][m+1] ++; // (max_index_eq-1)'s eq_set inc ++
                            //printf("\nm= %d", m);
                        }

                        if (arr_index_eq[j] == 0) {
                            arr_index_eq[j] = 1;
                            arr_eq_par[cnt_row][m+1] ++; // max_index_eq's eq_set inc ++
                        }
                    }
                }
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[cnt_row][col_even]);
                    printf("\nsub c step 3 cnt_eq_set= %d", arr_eq_par[cnt_row][col_even+1]);
                    printf("\nsub c step 3 eq_set start pt= ");
                    for (i = 0; i < half_col; i++) {
                        printf("%d", arr_eq_par[cnt_row][2*i]); printf(", ");
                    }
                    printf("\nsub c step 3 eq_set inc= ");
                    for (i = 0; i < half_col; i++) {
                        printf("%d", arr_eq_par[cnt_row][2*i+1]); printf(", ");
                    }
                #endif            

                //step 3.3 cnt_row --
                cnt_row --;

                //step 3.4 go to step 1.            
            }          
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c end all step, arr_eq_par= ");
                for(i = row -1 ; i >= 0 ; i--) {
                    printf("\n");
                    for(j = 0 ; j <= col_even+1 ; j++) {
                        printf("%d", arr_eq_par[i][j]); printf(", ");
                    }
                }
            #endif     
        }
        #ifdef DEBUG_np_lexsort_2d_double
            printf("\nsub c lexsort_o= ");
            for (i = 0; i < col; i++) {
                printf("%d", lexsort_o[i]); printf(", ");
            }
        #endif

        //long vmrss, vmsize;
        //get_memory_usage_kb(&vmrss, &vmsize);
        //printf("\nspe c v1 Current memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n", vmrss, vmsize);

        free(arr_sort);
        free(arr_index_sort);
        free(arr_index_eq);
        free_2d_int_array(arr_eq_par, row);    


    }

    //void np_lexsort_2d_double_v2(double** p ,int flag ,int* lexsort_o)
    void np_lexsort_2d_double_v2(int row ,int col ,double** p_in ,int* lexsort_o)
    {
    	// usage:
        // flag=0: min first, ascending
        // flag=1: max first, descending
        int i, j;
        int min = 0, m = 0;
        int finish = 0, has_eq_element = 0, exec_step_3 = 0;
        int cnt_eq_set_num = 0, eq_start_pt = 0, eq_inc = 0;
        //int row = sizeof (p_in[0][0]) / sizeof (p_in);
        //int col = sizeof (p_in) + 1;
        //int cnt_row = row - 1;
        int half_col = col/2; // == max eq set
        int col_even = 2*half_col;
        int loop_count = col;
        double compare = 0, key = 0;
        double *arr_sort = create_1d_double_array(col);
        int *arr_index_sort = create_1d_int_array(col);
        int *arr_index_eq = create_1d_int_array(col);

        int max_row_size = 1, cnt_row_eq_par = 0, arr_eq_par_col_size = col_even + 2;
        //int **arr_eq_par = create_2d_int_array(row, col_even + 2);
        //int **arr_eq_par = create_2d_int_array(max_row_size, col_even + 2);
        int *arr_eq_par = create_1d_int_array(arr_eq_par_col_size);

        for (i = 0; i < col; i++) 
    		//lexsort_o[i] = i;   
    		arr_index_sort[i] = i;     


        while (finish == 0) {
            // step 1. cp cnt_row to memory, including index and elements
            //if (cnt_row == row - 1) {
            if (cnt_row_eq_par == 0) {           
                for(i = 0 ; i < col ; i++) // cp p_in[row-1][0 ~ col-1] to arr_sort[0 ~ col-1]
                    arr_sort[i] = p_in[row-1][i];
            } else {
                //cnt_eq_set_num = arr_eq_par[cnt_row+1][col_even+1]; // (cnt_row+1)'s cnt_eq_set
                //eq_start_pt = arr_eq_par[cnt_row+1][2*cnt_eq_set_num]; // (cnt_row+1)'s start pt
                //eq_inc = arr_eq_par[cnt_row+1][2*cnt_eq_set_num+1]; // (cnt_row+1)'s inc
              //  cnt_row = row - 1 - cnt_row_eq_par; 
              //  cnt_eq_set_num = arr_eq_par[cnt_row_eq_par-1][col_even+1]; // (cnt_row+1)'s cnt_eq_set
              //  eq_start_pt = arr_eq_par[cnt_row_eq_par-1][2*cnt_eq_set_num]; // (cnt_row+1)'s start pt
              //  eq_inc = arr_eq_par[cnt_row_eq_par-1][2*cnt_eq_set_num+1]; // (cnt_row+1)'s inc
                m = (cnt_row_eq_par - 1)*arr_eq_par_col_size; 
                cnt_eq_set_num = arr_eq_par[m+col_even+1]; // (cnt_row+1)'s cnt_eq_set
                eq_start_pt = arr_eq_par[m+2*cnt_eq_set_num]; // (cnt_row+1)'s start pt
                eq_inc = arr_eq_par[m+2*cnt_eq_set_num+1]; // (cnt_row+1)'s inc
                for (i = 0; i < eq_inc; i++) { // cp lexsort_o[eq_start_pt ~ (eq_start_pt + eq_inc - 1)] to arr_index_sort[0 ~ eq_inc - 1]
                    arr_index_sort[i] = lexsort_o[eq_start_pt+i];
                }

                for (i = 0; i < eq_inc; i++) { // cp p_in[cnt_row][j] to arr_sort[i]
                    j = arr_index_sort[i];
                    arr_sort[i] = p_in[row-1-cnt_row_eq_par][j];
                }
            }
            #ifdef DEBUG_np_lexsort_2d_double
                //printf("\nsub c step 1 cnt_row= %d", cnt_row);
                printf("\nsub c step 1 cnt_row_eq_par= %d", cnt_row_eq_par);
                printf(", cnt_eq_set_num= %d", cnt_eq_set_num);
                printf(", eq_start_pt= %d", eq_start_pt);
                printf(", eq_inc= %d", eq_inc);
                printf("\nsub c step 1 arr_sort= ");
                for (i = 0; i < col; i++) {
                    printf("%f", arr_sort[i]); printf(", ");
                }
                printf("\nsub c step 1 arr_index_sort= ");
                for (i = 0; i < col; i++) {
                    printf("%d", arr_index_sort[i]); printf(", ");
                }
            #endif

            //step 2. sort
            //step 2.1 stable sort: https://www.geeksforgeeks.org/stable-selection-sort/
            // Iterate through array elements
            //if (cnt_row < row - 1)
            if (cnt_row_eq_par > 0)
                loop_count = eq_inc;

            if (arr_sort[loop_count - 1] > 1)
                key = arr_sort[loop_count - 1] - 1;
            else            
                key = arr_sort[loop_count - 1] + 1;

            for (i = 0; i < loop_count - 1; i++) {  
                // Find minimum element from arr[i] to arr[n - 1]. 
                min = i;
                //#ifdef DEBUG_np_lexsort_2d_double
                //    printf("\nsub c step 2.1 i= %d", i); 
                //#endif
                for (j = i + 1; j < loop_count; j++) {
                   // if (flag == 1) {
                        compare = arr_sort[min] - arr_sort[j];
                   // }
                   // else {
                   //     compare = arr_sort[j] - arr_sort[min];
                   // }

                    if (compare > 0)
                        min = j;

                    // equal element indicator   
                    //if (arr_sort[i] == arr_sort[j]) {  
                    if (i == 0) {
                        if ((arr_sort[i] == arr_sort[j]) || (key == arr_sort[j])) {
                            //if (index_eq[i] == 0) {
                            //    max_index_eq += 1;
                            //    index_eq[i] = max_index_eq;
                            //}
                            //
                            //if (index_eq[j] == 0)
                            //    index_eq[j] = index_eq[i];
                            has_eq_element = 1;
                        }

                        key = arr_sort[j];
                    }
                    //#ifdef DEBUG_np_lexsort_2d_double
                    //    printf("\nsub c step 2.1 j= %d", j);
                    //    printf(", arr_sort[min]= %f", arr_sort[min]);
                    //    printf(", arr_sort[i]= %f", arr_sort[i]);
                    //    printf(", arr_sort[j]= %f", arr_sort[j]);
                    //#endif
                }

                // Move minimum element at current i. 
                key = arr_sort[min];
                m = arr_index_sort[min];
                while (min > i)  
                { 
                    arr_sort[min] = arr_sort[min - 1]; 
                    arr_index_sort[min] = arr_index_sort[min - 1];
                    min--; 
                } 
                arr_sort[i] = key;
                arr_index_sort[i] = m; 
            }
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 2.1 loop_count= %d", loop_count);
                printf("\nsub c step 2.1 has_eq_element= %d", has_eq_element);
                printf("\nsub c step 2.1 arr_sort= ");
                for (i = 0; i < loop_count; i++) {
                    printf("%f", arr_sort[i]); printf(", ");
                }
                printf("\nsub c step 2.1 arr_index_sort= ");
                for (i = 0; i < loop_count; i++) {
                    printf("%d", arr_index_sort[i]); printf(", ");
                }
            #endif

            //step 2.2 cp sort result to lexsort_o
            /*if (cnt_row == row - 1) {          
                for(i = 0 ; i < col ; i++) // cp arr_index_sort to lexsort_o 
                    lexsort_o[i] = arr_index_sort[i];
            } else {
                for (i = 0; i < eq_inc; i++)  // cp arr_index_sort to lexsort_o
                    lexsort_o[eq_start_pt+i] = arr_index_sort[i];
            }*/
            for (i = 0; i < loop_count; i++)  // cp arr_index_sort to lexsort_o
                lexsort_o[eq_start_pt+i] = arr_index_sort[i];

            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c step 2.2 lexsort_o= ");
                for (i = 0; i < col; i++) {
                    printf("%d", lexsort_o[i]); printf(", ");
                }
            #endif

            //step 2.3 condition: finish, continueously from bottom to top or begin from top to bottom
            //if ((has_eq_element == 1) && (cnt_row > 0)) {
            if ((has_eq_element == 1) && (cnt_row_eq_par < row - 1)) {
                exec_step_3 = 1; // go to step 3
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 2.3 (has_eq_element == 1) && (cnt_row > 0)");
                #endif
            } else {
                //search (max_eq_set - cnt_eq_set > 0) from cnt_row+1 to row-1 if (cnt_row < row - 1):
                //if (cnt_row == row -1) {
                if (cnt_row_eq_par == 0) {
                    finish = 1;
                    exec_step_3 = 0;
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 finish = 1; exec_step_3 = 0;");
                    #endif
                } else {
                    // search (max_eq_set - (cnt_eq_set + 1) > 0) 
                    // from cnt_row+1(for cnt_row == 0) or cnt_row(for cnt_row > 0) to row-1: // top down search  
                    //if (cnt_row == 0) search_st_row = cnt_row+1
                    //else search_st_row = cnt_row
                    //for (i==search_st_row; i < row; i++)
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 cnt_row_eq_par= %d", cnt_row_eq_par);
                      //  printf("\nsub c step 2.3 max_index_eq= %d", arr_eq_par[cnt_row_eq_par-1][col_even]);
                      //  printf("\nsub c step 2.3 cnt_eq_set= %d", arr_eq_par[cnt_row_eq_par-1][col_even+1]);
                        min = (cnt_row_eq_par - 1)*arr_eq_par_col_size;
                        printf("\nsub c step 2.3 max_index_eq= %d", arr_eq_par[min+col_even]);
                        printf("\nsub c step 2.3 cnt_eq_set= %d", arr_eq_par[min+col_even+1]);
                    #endif
                  //  for (i = cnt_row+1; i < row; i++) {
                  //      if (arr_eq_par[i][col_even] - (arr_eq_par[i][col_even+1] + 1) > 0) {
                  //          cnt_row = i - 1;
                    for (i = cnt_row_eq_par-1; i >= 0; i--) {
                      //  if (arr_eq_par[i][col_even] - (arr_eq_par[i][col_even+1] + 1) > 0) {
                        m = i*arr_eq_par_col_size;
                        if (arr_eq_par[m+col_even] - (arr_eq_par[m+col_even+1] + 1) > 0) {
                            cnt_row_eq_par = i + 1;
                            //for (j==cnt_row; j >= 0; j--) // clear max_eq_set and cnt_eq_set from cnt_row to 0
                            //    arr_eq_par[j][col/2] = 0
                            //    arr_eq_par[j][col/2+1] = 0
                          //  arr_eq_par[i][col_even+1] ++;
                            arr_eq_par[m+col_even+1] ++;
                            finish = 0;
                            exec_step_3 = 0; // go to step 1
                            #ifdef DEBUG_np_lexsort_2d_double
                                printf("\nsub c step 2.3 i= %d", i);
                                printf("\nsub c step 2.3 max_index_eq= %d", arr_eq_par[m+col_even]);
                                printf("\nsub c step 2.3 cnt_eq_set= %d", arr_eq_par[m+col_even+1]);
                            #endif
                            break;
                        //} else if (i == row - 1) { // note: arr_eq_par[i][col/2] - (arr_eq_par[i][col/2+1] + 1) == 0
                        } else if (i == 0) { // note: arr_eq_par[i][col/2] - (arr_eq_par[i][col/2+1] + 1) == 0
                            finish = 1;
                            exec_step_3 = 0; // skip step 3
                            break;
                        }
                    }
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 2.3 cnt_row_eq_par= %d", cnt_row_eq_par);
                        printf("\nsub c step 2.3 finish= %d", finish);
                        printf("\nsub c step 2.3 exec_step_3= %d", exec_step_3);
                    #endif
                }
            }

            has_eq_element = 0;

            //step 3. mark eq and save associated parameters        
            if (exec_step_3 == 1) {
                //step 3.1 mark equal elements
                //& step 3.2 save cnt_row's start pt and inc for each eq set and max_eq_set (arr_eq_par[cnt_row][0 ~ 2*max_eq_set - 1, col/2])
                //search eq elements range
                //if (cnt_row == row - 1)
                if (cnt_row_eq_par == 0) 
                    loop_count = col - 1;
                else
                    loop_count = eq_inc;

                //cnt_row_eq_par = row - 1 - cnt_row;

                if (cnt_row_eq_par == max_row_size) {
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 3 cnt_row_eq_par= %d", cnt_row_eq_par);
                        printf("\nsub c step 3 max_row_size= %d", max_row_size);
                    #endif
                    max_row_size ++;

                    // Reallocate rows
                    /*for (i = 0; i < max_row_size; i++) {
                        arr_eq_par[i] = (int*)realloc(arr_eq_par[i], (col_even + 2)*sizeof(int));
                        if (arr_eq_par[i] == NULL) {
                            fprintf(stderr, "Out of memory");
                            exit(0);
                        }
                    }*/
                    //arr_eq_par = realloc(arr_eq_par, max_row_size*(col_even + 2)*sizeof(int*));
                    //arr_eq_par = realloc(arr_eq_par, max_row_size*sizeof(int*));
                    arr_eq_par = (int*)realloc(arr_eq_par, max_row_size*arr_eq_par_col_size*sizeof(int));
                    if (arr_eq_par == NULL) {
                        fprintf(stderr, "Out of memory");
                        exit(0);
                    }
                    for(i = cnt_row_eq_par*arr_eq_par_col_size ; i < max_row_size*arr_eq_par_col_size ; i++)
                        arr_eq_par[i] = 0;        
                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 3 Reallocate rows= ");
                        for(i = 0 ; i < max_row_size*arr_eq_par_col_size ; i++) {
                            printf("%d", arr_eq_par[i]); printf(", ");
                        }
                    #endif
                    /*arr_eq_par = realloc(arr_eq_par, max_row_size*sizeof(int *));    
                    if (arr_eq_par == NULL) {
                        fprintf(stderr, "realloc, Out of memory");
                        exit(0);
                    }*/
                    // Reallocate rows
                    /*for (i = cnt_row_eq_par; i < max_row_size; i++) {
                        //arr_eq_par[i] = (int*)realloc(arr_eq_par[i], (col_even + 2)*sizeof(int));
                        arr_eq_par[i] = realloc(arr_eq_par[i], (col_even + 2)*sizeof(int));
                        if (arr_eq_par[i] == NULL) {
                            fprintf(stderr, "\nOut of memory");
                            exit(0);
                        }
                    }*/
                  //  printf("\nsub c step 3 realloc");
                    /*
                    for(i = cnt_row_eq_par ; i < max_row_size ; i++)
                        for(j = 0 ; j < col_even + 2 ; j++)
                            arr_eq_par[i][j] = 0;

                    #ifdef DEBUG_np_lexsort_2d_double
                        printf("\nsub c step 3 Reallocate rows= ");
                        for(i = 0 ; i < max_row_size ; i++) {
                            for(j = 0 ; j < col_even + 2 ; j++) {
                                printf("%d", arr_eq_par[i][j]); printf(", ");
                            }
                        }
                    #endif*/
                }

                m = cnt_row_eq_par*arr_eq_par_col_size;
                //if (cnt_row < row - 1) {
                if (cnt_row_eq_par > 0) {
                    //arr_eq_par[cnt_row][col_even] = 0; // reset cnt_row's max_index_eq
                    //arr_eq_par[cnt_row][col_even+1] = 0; // reset cnt_row's cnt_eq_set
                  //  arr_eq_par[cnt_row_eq_par][col_even] = 0; // reset cnt_row's max_index_eq
                  //  arr_eq_par[cnt_row_eq_par][col_even+1] = 0; // reset cnt_row's cnt_eq_set
                    arr_eq_par[m+col_even] = 0; // reset cnt_row's max_index_eq
                    arr_eq_par[m+col_even+1] = 0; // reset cnt_row's cnt_eq_set                

                    for (i = 0; i < loop_count; i++) {
                      //  m = 2*i;
                        //arr_eq_par[cnt_row][m+1] = 0; // reset cnt_row's cnt_eq_set's eq_set inc
                      //  arr_eq_par[cnt_row_eq_par][m+1] = 0; // reset cnt_row's cnt_eq_set's eq_set inc
                        arr_eq_par[m+2*i+1] = 0; // reset cnt_row's cnt_eq_set's eq_set inc
                        arr_index_eq[i] = 0; // reset arr_index_eq
                    }

                    loop_count --;
                }
                #ifdef DEBUG_np_lexsort_2d_double
                    printf("\nsub c step 3 search eq elements range= %d", loop_count+1);
                #endif

                //m = 0;  
                //for (i = 0; i < eq_inc; i++) {  
                for (i = 0; i < loop_count; i++) {
                    j = i+1;
                    if (arr_sort[i] == arr_sort[j]) {
                        if (arr_index_eq[i] == 0) { // find an eq_set start pt
                            //m = 2*arr_eq_par[cnt_row][col_even]; // 2*max_index_eq
                            //arr_eq_par[cnt_row][col_even] ++; // max_index_eq ++
                            //arr_index_eq[i] = 1; // mark this eq element
                            ////arr_eq_par[cnt_row][m] = i; // save (max_index_eq-1)'s eq_set start pt
                            //arr_eq_par[cnt_row][m] = eq_start_pt + i; // save (max_index_eq-1)'s eq_set start pt
                            //arr_eq_par[cnt_row][m+1] ++; // (max_index_eq-1)'s eq_set inc ++
                          //  m = 2*arr_eq_par[cnt_row_eq_par][col_even]; // 2*max_index_eq
                          //  arr_eq_par[cnt_row_eq_par][col_even] ++; // max_index_eq ++
                            min = 2*arr_eq_par[m+col_even]; // 2*max_index_eq
                            //printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[m+col_even]);
                            arr_eq_par[m+col_even] ++; // max_index_eq ++
                            //printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[m+col_even]);
                            arr_index_eq[i] = 1; // mark this eq element
                            //arr_eq_par[cnt_row][m] = i; // save (max_index_eq-1)'s eq_set start pt
                          //  arr_eq_par[cnt_row_eq_par][m] = eq_start_pt + i; // save (max_index_eq-1)'s eq_set start pt
                          //  arr_eq_par[cnt_row_eq_par][m+1] ++; // (max_index_eq-1)'s eq_set inc ++
                            arr_eq_par[m+min] = eq_start_pt + i; // save (max_index_eq-1)'s eq_set start pt
                            arr_eq_par[m+min+1] ++; // (max_index_eq-1)'s eq_set inc ++
                            //printf("\nm= %d", m);
                        }

                        if (arr_index_eq[j] == 0) {
                            arr_index_eq[j] = 1;
                            //arr_eq_par[cnt_row][m+1] ++; // max_index_eq's eq_set inc ++
                          //  arr_eq_par[cnt_row_eq_par][m+1] ++; // max_index_eq's eq_set inc ++
                            arr_eq_par[m+min+1] ++; // max_index_eq's eq_set inc ++
                        }
                    }
                }
                #ifdef DEBUG_np_lexsort_2d_double
                  //  printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[cnt_row_eq_par][col_even]);
                  //  printf("\nsub c step 3 cnt_eq_set= %d", arr_eq_par[cnt_row_eq_par][col_even+1]);
                    printf("\nsub c step 3 max_index_eq= %d", arr_eq_par[m+col_even]);
                    printf("\nsub c step 3 cnt_eq_set= %d", arr_eq_par[m+col_even+1]);
                    printf("\nsub c step 3 eq_set start pt= ");
                    for (i = 0; i < half_col; i++) {
                      //  printf("%d", arr_eq_par[cnt_row_eq_par][2*i]); printf(", ");
                        printf("%d", arr_eq_par[m+2*i]); printf(", ");
                    }
                    printf("\nsub c step 3 eq_set inc= ");
                    for (i = 0; i < half_col; i++) {
                      //  printf("%d", arr_eq_par[cnt_row_eq_par][2*i+1]); printf(", ");
                        printf("%d", arr_eq_par[m+2*i+1]); printf(", ");
                    }
                #endif            

                //step 3.3 cnt_row --
                //cnt_row --;
                cnt_row_eq_par ++;

                //step 3.4 go to step 1.            
            }         
            #ifdef DEBUG_np_lexsort_2d_double
                printf("\nsub c end all step, arr_eq_par= ");
                for(i = 0 ; i < max_row_size ; i++) {
                    printf("\n");
                    for(j = 0 ; j < arr_eq_par_col_size ; j++) {
                        printf("%d", arr_eq_par[i*arr_eq_par_col_size+j]); printf(", ");
                    }
                }
            #endif     
        }
        #ifdef DEBUG_np_lexsort_2d_double
            printf("\nsub c lexsort_o= ");
            for (i = 0; i < col; i++) {
                printf("%d", lexsort_o[i]); printf(", ");
            }
        #endif

        //long vmrss, vmsize;
        //get_memory_usage_kb(&vmrss, &vmsize);
        //printf("\nspe c v2 Current memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n", vmrss, vmsize);

        free(arr_sort);
        free(arr_index_sort);
        free(arr_index_eq);
        //free_2d_int_array(arr_eq_par, row);
      //  free_2d_int_array(arr_eq_par, cnt_row_eq_par);
        free(arr_eq_par);        

    }
#endif


// https://github.com/lemire/CMemoryUsage
/*
// https://hpcf.umbc.edu/general-productivity/checking-memory-usage/
// Look for lines in the procfile contents like: 
// VmRSS:         5560 kB
// VmSize:         5560 kB
// Grab the number between the whitespace and the "kB"
// If 1 is returned in the end, there was a serious problem 
// (we could not find one of the memory usages)
#include "memory.h"
int get_memory_usage_kb(long* vmrss_kb, long* vmsize_kb)
{
    // Get the the current process' status file from the proc filesystem
    FILE* procfile = fopen("/proc/self/status", "r");

    long to_read = 8192;
    char buffer[to_read];
    int read = fread(buffer, sizeof(char), to_read, procfile);
    fclose(procfile);

    short found_vmrss = 0;
    short found_vmsize = 0;
    char* search_result;

    // Look through proc status contents line by line 
    char delims[] = "\n";
    char* line = strtok(buffer, delims);

    while (line != NULL && (found_vmrss == 0 || found_vmsize == 0) )
    {
        search_result = strstr(line, "VmRSS:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmrss_kb);
            found_vmrss = 1;
        }

        search_result = strstr(line, "VmSize:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmsize_kb);
            found_vmsize = 1;
        }

        line = strtok(NULL, delims);
    }

    return (found_vmrss == 1 && found_vmsize == 1) ? 0 : 1;
} */
