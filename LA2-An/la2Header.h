
#ifndef LA1_H_
#define LA1_H_

/*
    Computes dot product of matrix (of n by n) and vector (size n)
    Here the matrix is stored in a 1D array
    @param: mat = double*, vec = double*, n = int
    @output: result = double*
*/
double* mvp1(double* mat, double* vec, int n);

/*
    Computes dot product of matrix (of rowAmount by n) and vector (size n)
    Here the matrix is stored in a 1D array
    @param: mat = double*, vec = double*, n = int, rowAmount = int
    @output: result = double*
*/
double* subMatrixVectorProduct(double* mat, double* vec, int n, int rowAmount);

/*
    Computes dot product of matrix (of n by n) and vector (size n)
    Here the matrix is stored in a 2D array
    @param: mat = double**, vec = double*, n = int
    @output: result = double*
*/
double* mvp2(double** mat, double* vec, int n);

/*
    Frees space used by matrix, vector, and dot product result
    Here the matrix is stored in a 2D array
    @param: mat = double**, vec = double*, result = double*, n = int
    @output: void
*/
void freeMat2(double** mat, double* vec, double* result, int n);

/*
    Populate given matrix (of size n by n) with 2's along diagonal and 1's on the off diagonal
    Here the matrix is stored in a 2D array
    @param: mat = double**, n = int
    @output: void
*/
void assignMat2(double** mat, int n);

/*
    Populates given vector (of size n) with 1's
    Here the vec is stored in a 1D array
    @param: vec = double*, n = int
    @output: void
*/
void assignVec(double* vec, int n);

/*
    Prints the matrix, vector, and dot product in corresponding order
    Here the matrix is stored in a 2D array
    @param: mat = double**, vec = double*, result = double*, n = int
    @output: void
*/
void printMatVec2(double **mat, double *vec, double *result, int n);

/*
    Allocates space for vector of size n
    @param: n = int
    @output: vector = double*
*/
double* allocVec(int n);

/*
    Allocates space for matrix of size n by n
    Here the matrix is stored in a 2D array
    @param: n = int
    @output: matrix = double**
*/
double** allocMat2(int n);

/*
    Frees space used by matrix, vector, and dot product result
    Here the matrix is stored in a 1D array
    @param: mat = double*, vec = double*, result = double*, n = int
    @output: void
*/
void freeMat1(double* mat, double* vec, double* result, int n);

/*
    Allocates space for matrix of size n by n
    Here the matrix is stored in a 1D array
    @param: n = int
    @output: matrix = double*
*/
double* allocMat1(int n);

/*
    Prints the matrix, vector, and dot product in corresponding order
    Here the matrix is stored in a 1D array
    @param: mat = double*, vec = double*, result = double*, n = int
    @output: void
*/
void printMatVec1(double *mat, double *vec, double *result, int n);

/*
    Populate given matrix (of size n by n) with 2's along diagonal and 1's on the off diagonal
    Here the matrix is stored in a 1D array
    @param: mat = double**, n = int
    @output: void
*/
void assignMat1(double* mat, int n);

#endif

