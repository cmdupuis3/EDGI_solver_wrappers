
#include "edgi_openblas.hpp"

template<typename MTYPE, svd_mat_t SVD_TYPE>
edgi_openblas_t<MTYPE, SVD_TYPE>::edgi_openblas_t(int nx_in, int ny_in, MTYPE* mat_in, int Nthreads_in){
    this->Nthreads = Nthreads_in;
    this->nx = nx_in;
    this->ny = ny_in;
    this->mat = mat_in;

    this->results.U = nullptr;
    this->results.S = nullptr;
    this->results.VT = nullptr;
    this->results.V = nullptr;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
edgi_openblas_t<MTYPE, SVD_TYPE>::~edgi_openblas_t(){
    //delete[] this->results.S;
    //delete[] this->results.U;
    //delete[] this->results.VT;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::initialize(){

    //omp_set_num_threads(this->Nthreads);
    openblas_set_num_threads(this->Nthreads);

    time(&(this->start_time));

    return;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::finalize(){

    time(&(this->end_time));
    cout << this->Nthreads << " OpenMP threads -> " << difftime(this->end_time, this->start_time) << " seconds" << endl;

    return;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::svd(){
    cout << "This type is not supported for *svd routines." << endl;
    return;
}
template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::sdd(){
    cout << "This type is not supported for *sdd routines." << endl;
    return;
}
template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::evd(){
    cout << "This type is not supported for *evd routines." << endl;
    return;
}
template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::evr(int Neigs){
    cout << "This type is not supported for *evr routines." << endl;
    return;
}

template<>
void edgi_openblas_t<float, SVD_MAT_GENERAL>::svd(){

    this->initialize();

    int min_xy = (this->nx < this->ny) ? this->nx : this->ny;

    this->results.U = new float[this->nx*this->nx];
    this->results.S = new float[min_xy];
    this->results.VT = new float[this->ny*this->ny];

    char jobu = 'A';
    char jobvt = 'A';
    float* superb;

    this->status = LAPACKE_sgesvd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobu,               // jobu
        jobvt,              // jobvt
        this->nx,           // M
        this->ny,           // N
        this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        this->results.U,    // U
        this->nx,           // LDU
        this->results.VT,   // VT
        this->ny,           // LDVT
        superb);            // superb

    this->finalize();

    return;
}

/*
template<>
void edgi_openblas_t< complex<float>, SVD_MAT_GENERAL >::svd(){

    this->initialize();

    int min_xy = (this->nx < this->ny) ? this->nx : this->ny;

    this->results.U = new complex<float>[this->nx*this->nx];
    this->results.S = new complex<float>[min_xy];
    this->results.VT = new complex<float>[this->ny*this->ny];

    char jobu = 'A';
    char jobvt = 'A';
    float* superb;

    this->status = LAPACKE_cgesvd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobu,               // jobu
        jobvt,              // jobvt
        this->nx,           // M
        this->ny,           // N
        (lapack_complex_float*)this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        (lapack_complex_float*)this->results.U,    // U
        this->nx,           // LDU
        (lapack_complex_float*)this->results.VT,   // VT
        this->ny,           // LDVT
        superb);            // superb

    this->finalize();

    return;
}
*/

template<>
void edgi_openblas_t<double, SVD_MAT_GENERAL>::svd(){

    int min_xy = (this->nx < this->ny) ? this->nx : this->ny;

    this->results.U = new double[this->nx*this->nx];
    this->results.S = new double[min_xy];
    this->results.VT = new double[this->ny*this->ny];

    char jobu = 'A';
    char jobvt = 'A';
    double* superb;

    this->status = LAPACKE_dgesvd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobu,               // jobu
        jobvt,              // jobvt
        this->nx,           // M
        this->ny,           // N
        this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        this->results.U,    // U
        this->nx,           // LDU
        this->results.VT,   // VT
        this->ny,           // LDVT
        superb);            // superb

    this->finalize();

    return;
}

/*
template<>
void edgi_openblas_t< complex<double>, SVD_MAT_GENERAL >::svd(){

    this->initialize();

    int min_xy = (this->nx < this->ny) ? this->nx : this->ny;

    this->results.U = new complex<double>[this->nx*this->nx];
    this->results.S = new complex<double>[min_xy];
    this->results.VT = new complex<double>[this->ny*this->ny];

    char jobu = 'A';
    char jobvt = 'A';
    double* superb;

    this->status = LAPACKE_cgesvd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobu,               // jobu
        jobvt,              // jobvt
        this->nx,           // M
        this->ny,           // N
        (lapack_complex_double*)this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        (lapack_complex_double*)this->results.U,    // U
        this->nx,           // LDU
        (lapack_complex_double*)this->results.VT,   // VT
        this->ny,           // LDVT
        superb);            // superb

    this->finalize();

    return;
}
*/

template<>
void edgi_openblas_t<float, SVD_MAT_SYMMETRIC>::svd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new float[this->nx*this->nx];
    this->results.S = new float[nx];
    this->results.VT = new float[this->ny*this->ny];

    char jobu = 'A';
    char jobvt = 'A';
    float* superb;

    this->status = LAPACKE_sgesvd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobu,               // jobu
        jobvt,              // jobvt
        this->nx,           // M
        this->ny,           // N
        this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        this->results.U,    // U
        this->nx,           // LDU
        this->results.VT,   // VT
        this->ny,           // LDVT
        superb);            // superb

    this->finalize();

    return;
}

template<>
void edgi_openblas_t<double, SVD_MAT_SYMMETRIC>::svd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new double[this->nx*this->nx];
    this->results.S = new double[nx];
    this->results.VT = new double[this->ny*this->ny];

    char jobu = 'A';
    char jobvt = 'A';
    double* superb;

    this->status = LAPACKE_dgesvd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobu,               // jobu
        jobvt,              // jobvt
        this->nx,           // M
        this->ny,           // N
        this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        this->results.U,    // U
        this->nx,           // LDU
        this->results.VT,   // VT
        this->ny,           // LDVT
        superb);            // superb

    this->finalize();

    return;
}

template<>
void edgi_openblas_t< complex<float>, SVD_MAT_HERMITIAN >::svd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new complex<float>[this->nx*this->nx];
    this->results.S = new float[nx];
    this->results.VT = new complex<float>[this->ny*this->ny];

    char jobu = 'A';
    char jobvt = 'A';
    float* superb;

    this->status = LAPACKE_cgesvd(
        LAPACK_ROW_MAJOR,                           // matrix_layout
        jobu,                                       // jobu
        jobvt,                                      // jobvt
        this->nx,                                   // M
        this->ny,                                   // N
        (lapack_complex_float*)this->mat,           // A
        this->nx,                                   // LDA
        this->results.S,                            // S
        (lapack_complex_float*)this->results.U,     // U
        this->nx,                                   // LDU
        (lapack_complex_float*)this->results.VT,    // VT
        this->ny,                                   // LDVT
        superb);                                    // superb

    this->finalize();

    return;
}

template<>
void edgi_openblas_t< complex<double>, SVD_MAT_HERMITIAN >::svd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new complex<double>[this->nx*this->nx];
    this->results.S = new double[nx];
    this->results.VT = new complex<double>[this->ny*this->ny];

    char jobu = 'A';
    char jobvt = 'A';
    double* superb;

    this->status = LAPACKE_zgesvd(
        LAPACK_ROW_MAJOR,                           // matrix_layout
        jobu,                                       // jobu
        jobvt,                                      // jobvt
        this->nx,                                   // M
        this->ny,                                   // N
        (lapack_complex_double*)this->mat,          // A
        this->nx,                                   // LDA
        this->results.S,                            // S
        (lapack_complex_double*)this->results.U,    // U
        this->nx,                                   // LDU
        (lapack_complex_double*)this->results.VT,   // VT
        this->ny,                                   // LDVT
        superb);                                    // superb

    this->finalize();

    return;
}

template<>
void edgi_openblas_t<float, SVD_MAT_GENERAL>::sdd(){

    this->initialize();

    int min_xy = (this->nx < this->ny) ? this->nx : this->ny;

    this->results.U = new float[this->nx*this->nx];
    this->results.S = new float[min_xy];
    this->results.VT = new float[this->ny*this->ny];

    char jobz = 'A';

    this->status = LAPACKE_sgesdd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        this->nx,           // M
        this->ny,           // N
        this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        this->results.U,    // U
        this->nx,           // LDU
        this->results.VT,   // VT
        this->ny);          // LDVT

    this->finalize();

    return;
}

/*
template<>
void edgi_openblas_t< complex<float>, SVD_MAT_GENERAL >::sdd(){

    this->initialize();

    int min_xy = (this->nx < this->ny) ? this->nx : this->ny;

    this->results.U = new complex<float>[this->nx*this->nx];
    this->results.S = new float[min_xy];
    this->results.VT = new complex<float>[this->ny*this->ny];

    PLASMA_Alloc_Workspace_cgesdd(this->nx, this->ny, desc);
    PLASMA_cgesdd(PlasmaVec,                                // jobu
                  PlasmaVec,                                // jobvt
                  this->nx,                                 // M
                  this->ny,                                 // N
                  (PLASMA_Complex32_t*)this->mat,           // A
                  this->nx,                                 // LDA
                  this->results.S,//,                        // S
                  *(this->desc),                                    // descT
                  (PLASMA_Complex32_t*)this->results.U,   // U
                  this->nx,                                 // LDU
                  (PLASMA_Complex32_t*)this->results.VT,  // VT
                  this->ny);                                // LDVT

    this->finalize();

    return;
}
*/

template<>
void edgi_openblas_t<double, SVD_MAT_GENERAL>::sdd(){

    this->initialize();

    int min_xy = (this->nx < this->ny) ? this->nx : this->ny;

    this->results.U = new double[this->nx*this->nx];
    this->results.S = new double[min_xy];
    this->results.VT = new double[this->ny*this->ny];

    char jobz = 'A';

    this->status = LAPACKE_dgesdd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        this->nx,           // M
        this->ny,           // N
        this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        this->results.U,    // U
        this->nx,           // LDU
        this->results.VT,   // VT
        this->ny);          // LDVT

    this->finalize();

    return;
}

/*
template<>
void edgi_openblas_t< complex<double>, SVD_MAT_GENERAL >::sdd(){

    this->initialize();

    int min_xy = (this->nx < this->ny) ? this->nx : this->ny;

    this->results.U = new complex<double>[this->nx*this->nx];
    this->results.S = new double[min_xy];
    this->results.VT = new complex<double>[this->ny*this->ny];

    PLASMA_Alloc_Workspace_zgesdd(this->nx, this->ny, desc);
    PLASMA_zgesdd(PlasmaVec,                                // jobu
                  PlasmaVec,                                // jobvt
                  this->nx,                                 // M
                  this->ny,                                 // N
                  (PLASMA_Complex64_t*)this->mat,           // A
                  this->nx,                                 // LDA
                  this->results.S,//,                        // S
                  *(this->desc),                                    // descT
                  (PLASMA_Complex64_t*)this->results.U,   // U
                  this->nx,                                 // LDU
                  (PLASMA_Complex64_t*)this->results.VT,  // VT
                  this->ny);                                // LDVT

    this->finalize();

    return;
}
*/

template<>
void edgi_openblas_t<float, SVD_MAT_SYMMETRIC>::sdd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new float[this->nx*this->nx];
    this->results.S = new float[nx];
    this->results.VT = new float[this->ny*this->ny];

    char jobz = 'A';

    this->status = LAPACKE_sgesdd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        this->nx,           // M
        this->ny,           // N
        this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        this->results.U,    // U
        this->nx,           // LDU
        this->results.VT,   // VT
        this->ny);          // LDVT

    this->finalize();

    return;
}

template<>
void edgi_openblas_t<double, SVD_MAT_SYMMETRIC>::sdd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new double[this->nx*this->nx];
    this->results.S = new double[nx];
    this->results.VT = new double[this->ny*this->ny];

    char jobz = 'A';

    this->status = LAPACKE_dgesdd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        this->nx,           // M
        this->ny,           // N
        this->mat,          // A
        this->nx,           // LDA
        this->results.S,    // S
        this->results.U,    // U
        this->nx,           // LDU
        this->results.VT,   // VT
        this->ny);          // LDVT

    this->finalize();

    return;
}

template<>
void edgi_openblas_t< complex<float>, SVD_MAT_HERMITIAN >::sdd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new complex<float>[this->nx*this->nx];
    this->results.S = new float[nx];
    this->results.VT = new complex<float>[this->ny*this->ny];

    char jobz = 'A';

    this->status = LAPACKE_cgesdd(
        LAPACK_ROW_MAJOR,                           // matrix_layout
        jobz,                                       // jobz
        this->nx,                                   // M
        this->ny,                                   // N
        (lapack_complex_float*)this->mat,           // A
        this->nx,                                   // LDA
        this->results.S,                            // S
        (lapack_complex_float*)this->results.U,     // U
        this->nx,                                   // LDU
        (lapack_complex_float*)this->results.VT,    // VT
        this->ny);                                  // LDVT

    this->finalize();

    return;
}

template<>
void edgi_openblas_t< complex<double>, SVD_MAT_HERMITIAN >::sdd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new complex<double>[this->nx*this->nx];
    this->results.S = new double[nx];
    this->results.VT = new complex<double>[this->ny*this->ny];

    char jobz = 'A';

    this->status = LAPACKE_zgesdd(
        LAPACK_ROW_MAJOR,                           // matrix_layout
        jobz,                                       // jobz
        this->nx,                                   // M
        this->ny,                                   // N
        (lapack_complex_double*)this->mat,          // A
        this->nx,                                   // LDA
        this->results.S,                            // S
        (lapack_complex_double*)this->results.U,    // U
        this->nx,                                   // LDU
        (lapack_complex_double*)this->results.VT,   // VT
        this->ny);                                  // LDVT

    this->finalize();

    return;
}


template<>
void edgi_openblas_t<float, SVD_MAT_SYMMETRIC>::evd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new float[this->nx*this->nx];
    this->results.S = new float[nx];

    char jobz = 'V';
    char uplo = 'U';

    /** Solver will save results in-situ, so copy data to results matrix first */
    for(int i = 0; i < this->nx*this->nx; i++){
        this->results.U[i] = this->mat[i];
    }

    this->status = LAPACKE_ssyevd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        uplo,               // uplo
        this->nx,           // N
        this->results.U,    // A
        this->nx,           // LDA
        this->results.S);   // W

    this->finalize();

    return;
}

template<>
void edgi_openblas_t<double, SVD_MAT_SYMMETRIC>::evd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new double[this->nx*this->nx];
    this->results.S = new double[nx];

    char jobz = 'V';
    char uplo = 'U';

    /** Solver will save results in-situ, so copy data to results matrix first */
    for(int i = 0; i < this->nx*this->nx; i++){
        this->results.U[i] = this->mat[i];
    }

    this->status = LAPACKE_dsyevd(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        uplo,               // uplo
        this->nx,           // N
        this->results.U,    // A
        this->nx,           // LDA
        this->results.S);   // W

    this->finalize();

    return;
}

template<>
void edgi_openblas_t< complex<float>, SVD_MAT_HERMITIAN >::evd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new complex<float>[this->nx*this->nx];
    this->results.S = new float[nx];

    char jobz = 'V';
    char uplo = 'U';

    /** Solver will save results in-situ, so copy data to results matrix first */
    for(int i = 0; i < this->nx*this->nx; i++){
        this->results.U[i] = this->mat[i];
    }

    this->status = LAPACKE_cheevd(
        LAPACK_ROW_MAJOR,                           // matrix_layout
        jobz,                                       // jobz
        uplo,                                       // uplo
        this->nx,                                   // N
        (lapack_complex_float*) this->results.U,    // A
        this->nx,                                   // LDA
        this->results.S);                           // W

    this->finalize();

    return;
}

template<>
void edgi_openblas_t< complex<double>, SVD_MAT_HERMITIAN >::evd(){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new complex<double>[this->nx*this->nx];
    this->results.S = new double[nx];

    char jobz = 'V';
    char uplo = 'U';

    /** Solver will save results in-situ, so copy data to results matrix first */
    for(int i = 0; i < this->nx*this->nx; i++){
        this->results.U[i] = this->mat[i];
    }

    this->status = LAPACKE_zheevd(
        LAPACK_ROW_MAJOR,                           // matrix_layout
        jobz,                                       // jobz
        uplo,                                       // uplo
        this->nx,                                   // N
        (lapack_complex_double*) this->results.U,   // A
        this->nx,                                   // LDA
        this->results.S);                           // W

    this->finalize();

    return;
}

template<>
void edgi_openblas_t<float, SVD_MAT_SYMMETRIC>::evr(int Neigs){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new float[this->nx*this->nx];
    this->results.S = new float[nx];

    char jobz = 'V';
    char range = 'A';
    char uplo = 'U';
    int* nComputedEigs;
    int* isuppz;

    this->status = LAPACKE_ssyevr(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        range,              // range
        uplo,               // uplo
        this->nx,           // N
        this->mat,          // A
        this->nx,           // LDA
        0, 0, 0, 0,         // vl, vu, il, iu
        0,                  // abstol
        nComputedEigs,      // m
        this->results.S,    // W
        this->results.U,    // z
        this->nx,           // ldz
        isuppz);            // isuppz

    this->finalize();

    return;
}

template<>
void edgi_openblas_t<double, (svd_mat_t)SVD_MAT_SYMMETRIC>::evr(int Neigs){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new double[this->nx*this->nx];
    this->results.S = new double[nx];

    char jobz = 'V';
    char range = 'A';
    char uplo = 'U';
    int* nComputedEigs;
    int* isuppz;

    this->status = LAPACKE_dsyevr(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        range,              // range
        uplo,               // uplo
        this->nx,           // N
        this->mat,          // A
        this->nx,           // LDA
        0, 0, 0, 0,         // vl, vu, il, iu
        0,                  // abstol
        nComputedEigs,      // m
        this->results.S,    // W
        this->results.U,    // z
        this->nx,           // ldz
        isuppz);            // isuppz

    this->finalize();

    return;
}

template<>
void edgi_openblas_t< complex<float>, SVD_MAT_HERMITIAN >::evr(int Neigs){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new complex<float>[this->nx*this->nx];
    this->results.S = new float[nx];

    char jobz = 'V';
    char range = 'A';
    char uplo = 'U';
    int* nComputedEigs;
    int* isuppz;

    this->status = LAPACKE_cheevr(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        range,              // range
        uplo,               // uplo
        this->nx,           // N
        (lapack_complex_float*)this->mat, // A
        this->nx,           // LDA
        0, 0, 0, 0,         // vl, vu, il, iu
        0,                  // abstol
        nComputedEigs,      // m
        this->results.S,    // W
        (lapack_complex_float*)this->results.U, // z
        this->nx,           // ldz
        isuppz);            // isuppz

    this->finalize();

    return;
}

template<>
void edgi_openblas_t< complex<double>, SVD_MAT_HERMITIAN >::evr(int Neigs){

    assert(this->nx == this->ny);

    this->initialize();

    this->results.U = new complex<double>[this->nx*this->nx];
    this->results.S = new double[nx];

    char jobz = 'V';
    char range = 'A';
    char uplo = 'U';
    int* nComputedEigs;
    int* isuppz;

    this->status = LAPACKE_zheevr(
        LAPACK_ROW_MAJOR,   // matrix_layout
        jobz,               // jobz
        range,              // range
        uplo,               // uplo
        this->nx,           // N
        (lapack_complex_double*)this->mat, // A
        this->nx,           // LDA
        0, 0, 0, 0,         // vl, vu, il, iu
        0,                  // abstol
        nComputedEigs,      // m
        this->results.S,    // W
        (lapack_complex_double*)this->results.U, // z
        this->nx,           // ldz
        isuppz);            // isuppz

    this->finalize();

    return;
}


template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::svd_test(){

    if(this->mat){
        // error: mat must be a nullptr to call test routines
    }
    this->mat = new MTYPE[this->nx*this->ny];
    this->gen_rand();
    this->svd();

    return;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::sdd_test(){

    if(this->mat){
        // error: mat must be a nullptr to call test routines
    }
    this->mat = new MTYPE[this->nx*this->ny];
    this->gen_rand();
    this->sdd();

    return;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::evd_test(){

    if(this->mat){
        // error: mat must be a nullptr to call test routines
    }
    this->mat = new MTYPE[this->nx*this->ny];
    this->gen_rand();
    this->evd();

    return;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::evr_test(int Neigs){

    if(this->mat){
        // error: mat must be a nullptr to call test routines
    }
    this->mat = new MTYPE[this->nx*this->ny];
    this->gen_rand();
    this->evr(Neigs);

    return;
}



template<typename MTYPE, svd_mat_t SVD_TYPE>
void edgi_openblas_t<MTYPE, SVD_TYPE>::gen_rand(){
    // error
    return;
}

template<>
void edgi_openblas_t<float, SVD_MAT_GENERAL>::gen_rand(){
    srand(time(NULL));
    int i = 0;
    #pragma omp for
    for(i = 0; i < this->nx*this->ny; i++){
        this->mat[i] = (rand() % 1000) / 100;
    }
    return;
}
template<>
void edgi_openblas_t< complex<float>, SVD_MAT_GENERAL >::gen_rand(){
    srand(time(NULL));
    int i = 0;
    #pragma omp for
    for(i = 0; i < this->nx*this->ny; i++){
        this->mat[i] = complex<float>((rand() % 1000) / 100, (rand() % 1000) / 100);
    }
    return;
}
template<>
void edgi_openblas_t<double, SVD_MAT_GENERAL>::gen_rand(){
    srand(time(NULL));
    int i = 0;
    #pragma omp for
    for(i = 0; i < this->nx*this->ny; i++){
        this->mat[i] = (rand() % 1000) / 100;
    }
    return;
}
template<>
void edgi_openblas_t< complex<double>, SVD_MAT_GENERAL >::gen_rand(){
    srand(time(NULL));
    int i = 0;
    #pragma omp for
    for(i = 0; i < this->nx*this->ny; i++){
        this->mat[i] = complex<double>((rand() % 1000) / 100, (rand() % 1000) / 100);
    }
    return;
}

template<>
void edgi_openblas_t<float, SVD_MAT_SYMMETRIC>::gen_rand(){
    srand(time(NULL));
    int i = 0, j = 0;
    #pragma omp for
    for(i = 0; i < this->nx; i++){
        for(j = i; j < this->ny; j++){
            this->mat[i*this->ny + j] = (rand() % 1000) / 100;
            this->mat[j*this->nx + i] = mat[i*this->ny + j];
        }
    }
    return;
}
template<>
void edgi_openblas_t< complex<float>, SVD_MAT_SYMMETRIC >::gen_rand(){
    srand(time(NULL));
    int i = 0, j = 0;
    #pragma omp for
    for(i = 0; i < this->nx; i++){
        for(j = i; j < this->ny; j++){
            this->mat[i*this->ny + j] = complex<float>((rand() % 1000) / 100, (rand() % 1000) / 100);
            this->mat[j*this->nx + i] = mat[i*this->ny + j];
        }
    }
    return;

}
template<>
void edgi_openblas_t<double, SVD_MAT_SYMMETRIC>::gen_rand(){
    srand(time(NULL));
    int i = 0, j = 0;
    #pragma omp for
    for(i = 0; i < this->nx; i++){
        for(j = i; j < this->ny; j++){
            this->mat[i*this->ny + j] = (rand() % 1000) / 100;
            this->mat[j*this->nx + i] = mat[i*this->ny + j];
        }
    }
    return;
}
template<>
void edgi_openblas_t< complex<double>, SVD_MAT_SYMMETRIC >::gen_rand(){
    srand(time(NULL));
    int i = 0, j = 0;
    #pragma omp for
    for(i = 0; i < this->nx; i++){
        for(j = i; j < this->ny; j++){
            this->mat[i*this->ny + j] = complex<double>((rand() % 1000) / 100, (rand() % 1000) / 100);
            this->mat[j*this->nx + i] = mat[i*this->ny + j];
        }
    }
    return;
}


template<>
void edgi_openblas_t< complex<float>, SVD_MAT_HERMITIAN >::gen_rand(){
    srand(time(NULL));
    int i = 0, j = 0;
    #pragma omp for
    for(i = 0; i < this->nx; i++){
        for(j = i; j < this->ny; j++){
            this->mat[i*this->ny + j] = complex<float>((rand() % 1000) / 100, (rand() % 1000) / 100);
            this->mat[j*this->nx + i] = conj(mat[i*this->ny + j]);
        }
    }
    return;
}
template<>
void edgi_openblas_t< complex<double>, SVD_MAT_HERMITIAN >::gen_rand(){
    srand(time(NULL));
    int i = 0, j = 0;
    #pragma omp for
    for(i = 0; i < this->nx; i++){
        for(j = i; j < this->ny; j++){
            this->mat[i*this->ny + j] = complex<double>((rand() % 1000) / 100, (rand() % 1000) / 100);
            this->mat[j*this->nx + i] = conj(mat[i*this->ny + j]);
        }
    }
    return;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
MTYPE* edgi_openblas_t<MTYPE, SVD_TYPE>::get_U(){
    if(!this->results.U){
        cout << "U does not currently exist." << endl;
        // fatal
    }

    return this->results.U;
}
template<typename MTYPE, svd_mat_t SVD_TYPE>
auto edgi_openblas_t<MTYPE, SVD_TYPE>::get_S(){
    if(!this->results.S){
        cout << "S does not currently exist." << endl;
        // fatal
    }

    return this->results.S;
}
template<typename MTYPE, svd_mat_t SVD_TYPE>
MTYPE* edgi_openblas_t<MTYPE, SVD_TYPE>::get_VT(){
    if(!this->results.VT){
        cout << "VT does not currently exist." << endl;
        // fatal
    }

    return this->results.VT;
}

/** Note: This routine transposes VT in-place; in other words, VT will be V after calling this! */
template<typename MTYPE, svd_mat_t SVD_TYPE>
MTYPE* edgi_openblas_t<MTYPE, SVD_TYPE>::get_V(){
    // fatal
    return nullptr;
}
template<>
float* edgi_openblas_t<float, SVD_MAT_GENERAL>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_somatcopy(
                CblasRowMajor,      // order
                CblasTrans,         // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
float* edgi_openblas_t<float, SVD_MAT_SYMMETRIC>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_somatcopy(
                CblasRowMajor,      // order
                CblasTrans,         // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
float* edgi_openblas_t<float, SVD_MAT_HERMITIAN>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_somatcopy(
                CblasRowMajor,      // order
                CblasConjTrans,     // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
double* edgi_openblas_t<double, SVD_MAT_GENERAL>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_domatcopy(
                CblasRowMajor,      // order
                CblasTrans,         // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
double* edgi_openblas_t<double, SVD_MAT_SYMMETRIC>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_domatcopy(
                CblasRowMajor,      // order
                CblasTrans,         // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
double* edgi_openblas_t<double, SVD_MAT_HERMITIAN>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_domatcopy(
                CblasRowMajor,      // order
                CblasConjTrans,     // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
complex<float>* edgi_openblas_t<complex<float>, SVD_MAT_GENERAL>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_comatcopy(
                CblasRowMajor,      // order
                CblasConjTrans,     // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
complex<float>* edgi_openblas_t<complex<float>, SVD_MAT_SYMMETRIC>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_comatcopy(
                CblasRowMajor,      // order
                CblasConjTrans,     // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
complex<float>* edgi_openblas_t<complex<float>, SVD_MAT_HERMITIAN>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_comatcopy(
                CblasRowMajor,      // order
                CblasConjTrans,     // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
complex<double>* edgi_openblas_t<complex<double>, SVD_MAT_GENERAL>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_domatcopy(
                CblasRowMajor,      // order
                CblasConjTrans,     // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
complex<double>* edgi_openblas_t<complex<double>, SVD_MAT_SYMMETRIC>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_domatcopy(
                CblasRowMajor,      // order
                CblasConjTrans,     // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}
template<>
complex<double>* edgi_openblas_t<complex<double>, SVD_MAT_HERMITIAN>::get_V(){
    if(!this->results.V){
        if(this->results.VT){

            this->initialize();

            cblas_domatcopy(
                CblasRowMajor,      // order
                CblasConjTrans,     // trans
                this->ny,           // rows
                this->nx,           // cols
                1,                  // alpha
                this->results.VT,   // A
                this->nx,           // lda
                this->results.V,    // B
                this->ny);          // ldb

            this->finalize();
        }
    }

    return this->results.V;
}



template<typename MTYPE, svd_mat_t SVD_TYPE>
int edgi_openblas_t<MTYPE, SVD_TYPE>::get_U_dimension(){
    return this->nx;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
int edgi_openblas_t<MTYPE, SVD_TYPE>::get_S_dimension(){
    return (this->nx < this->ny) ? this->nx : this->ny;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
int edgi_openblas_t<MTYPE, SVD_TYPE>::get_V_dimension(){
    return this->ny;
}

template<typename MTYPE, svd_mat_t SVD_TYPE>
MTYPE edgi_openblas_t<MTYPE, SVD_TYPE>::get_U_max(){
    MTYPE max = -1;
    for(int i = 0; i < this->nx * this->nx; i++){
        if(this->results.U[i] > max) max = this->results.U[i];
    }
    return max;
}
template<typename MTYPE, svd_mat_t SVD_TYPE>
MTYPE edgi_openblas_t<MTYPE, SVD_TYPE>::get_V_max(){
    MTYPE max = -1;
    if(this->results.VT){
        for(int i = 0; i < this->ny * this->ny; i++){
            if(this->results.VT[i] > max) max = this->results.VT[i];
        }
    }else{
        for(int i = 0; i < this->ny * this->ny; i++){
            if(this->results.V[i] > max) max = this->results.V[i];
        }
    }
    return max;
}





