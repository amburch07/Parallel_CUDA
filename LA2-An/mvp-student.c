#include <stdio.h>
#include <stdlib.h>
// When compiling, using -lm tag at the end
#include <math.h>
#include "mpi.h"
#include "la2Header.h"

/*
Function prototypes for the matrix vector product.
You MUST implement both of these functions
in addition to the others listed in the PDF
*/

// Matrix-Vector product operation where matrix = double*
double* mvp1(double* mat, double* vec, int n){
    double *result = (double*) malloc(n * sizeof(double));
    double sum = 0.0;
    // Keep track of what "row" we're on for mat
    int totalNumOfElements = n * n;

    // Manually updating vecIndex - for vec and result
    int vecIndex = 0; 
    for(int i = 0; i < totalNumOfElements; i+=n){
        for(int j = i; j < (i + n); j++){
            sum += mat[j] * vec[vecIndex];
        }
        result[vecIndex] = sum;
        sum = 0.0;
        vecIndex = vecIndex + 1;
    }

    return result;
}

// Used for performing matrix vector product for each sub matrix and vector
double* subMatrixVectorProduct(double* mat, double* vec, int n, int rowAmount){
    double *result = (double*) malloc(rowAmount * sizeof(double));
    // Keep track of what "row" we're on for mat
    int totalNumOfElements = n * rowAmount;

    // Manually updating vecIndex - for vec and result
    int vecIndex = 0; 
    for(int i = 0; i < totalNumOfElements; i+=n){
        double sum = 0.0;
        for(int j = i; j < (i + n); j++){
            sum += mat[j] * vec[vecIndex];
        }
        result[vecIndex] = sum;
        vecIndex = vecIndex + 1;
    }

    return result;
}


// Matrix-Vector product operation where matrix = double**
double* mvp2(double** mat, double* vec, int n) {
    double* result = (double*) malloc(n * sizeof(double));
   
    double sum = 0.0;

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            sum += (mat[i][j] * vec[j]);
        }
        result[i] = sum;
        sum = 0.0;
    }

    return result;
}

// BELOW ARE double** mat (Stores as 2D) methods - REMEMBER IT HAS TO WORK FOR ANY n

// Where we are freeing the allocated memory for the matrix
void freeMat2(double** mat, double* vec, double* result, int n) {
    // Free each of the rows
    for(int i = 0; i < n; i++){
        free(mat[i]);
    }
    free(mat);
    free(vec);
    free(result);
}

// Where we are assigning the matrix values themselves, the 2D array - Assuming square matrix
void assignMat2(double** mat, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++){
            // Populating the matrix to use 2’s and 1’s for the off diagonals when i==j
            if(i == j && i != (n-1) && j != (n-1)){
                mat[i][j] = 2.0;
                mat[i][j + 1] = 1.0;
                mat[i + 1][j] = 1.0;
            }
            // There's some issue where it won't populate the very last cell so doing it manually
            if(i == (n-1) && j == (n-1)){
                mat[i][j] = 2.0;
            }
        }
    }
}

// Where we are assigning the vector values themselves, the column vector of 1's - REUSE for both mat type inputs
void assignVec(double* vec, int n) {
    for(int i = 0; i < n; i++){
        vec[i] = 1;
    }
}

// Where we are printing the result from the matrix and vector multiplication 
void printMatVec2(double **mat, double *vec, double *result, int n) {
    // Looping through rows
    for(int i = 0; i < n; i++){
        // Printing out each element in row of mat - Looping through cols
        for(int j = 0; j < n; j++){
            printf("%f\t", mat[i][j]);
        }
        // Printing element in row of vec
        printf("%f\t", vec[i]);
        // Try to print equal sign in the "middle" of rows depending on dimensions
        if(i == (round(n/2.0)-1.0) && i != 0 && n > 2){
            printf("=\t");
        }
        else if (i == (round(n/2.0)-1.0) && n <= 2){
            printf("=\t");
        }
        else{
            printf("\t");
        }
        // Printing element in row of result
        printf("%f", result[i]);
        printf("\n");
    }
}
 
// Where we allocate memory for the vector - REUSE for both mat type inputs
double* allocVec(int n) {
    double *vec = (double*) malloc(n * sizeof(double));
    return vec;
}

// Where we allocate memory for the matrix (Stores as 2D)
double** allocMat2(int n) {
    double **mat = (double **) malloc(n * sizeof(double *));
    for(int i = 0; i < n; i++){
        mat[i] = (double *) malloc(n * sizeof(double));
    }
    return mat;
}

// BELOW ARE double* mat (Stores as 1D) methods - REMEMBER IT HAS TO WORK FOR ANY n

void freeMat1(double* mat, double* vec, double* result, int n) {
    // All 1 - D
    free(mat);
    free(vec);
    free(result);
}

// Where we allocate memory for the matrix (Stores as 1D)
double* allocMat1(int n) {
    double *mat = (double*) malloc(pow(n, 2) * sizeof(double *));
    return mat;
}

// Where we are printing the result from the matrix and vector multiplication - here both are 1-D
void printMatVec1(double *mat, double *vec, double *result, int n) {
    // To help us keep track of when to print out new row
    int totalNumOfElements = n * n;
    int vecIndex = 0;

    // Looping through all elements in mat and vec
    for(int i = 0; i < totalNumOfElements; i+=n){
        
        // Following same logic in mvp1 function
        for(int j = i; j < (i + n); j++){
            printf("%f\t", mat[j]);
        }
        
        // Printing element in row of vec
        printf("%f\t", vec[vecIndex]);

        // Try to print equal sign in the "middle" of rows depending on dimensions
        if(i == (round(totalNumOfElements/2.0)-1.0) && i != 0 && n > 2){
            printf("=\t");
        }
        else if (i == (round(totalNumOfElements/2.0)-1.0) && n <= 2){
            printf("=\t");
        }
        else{
            printf("\t");
        }
        // Printing element in row of result
        printf("%f", result[vecIndex]);
        printf("\n");

        // Going onto next row
        vecIndex = vecIndex + 1;
    }
}

// Where we are assigning values in the "matrix" but here it is 1-D though - Assuming square matrix
void assignMat1(double* mat, int n) {
    if(n == 1){
        mat[0] = 2.0;
    }
    if(n == 2){
        mat[0] = 2.0;
        mat[1] = 1.0;
        mat[2] = 1.0;
        mat[3] = 2.0;
    }
    if(n > 2){
        int totalNumOfElements = n * n;

        // Set the first element to 2 and second element to 1
        mat[0] = 2.0;
        mat[1] = 1.0;

        // To help keep track of when to add 2 and 1 - add them in a pattern
        int mod = 1;
        // Start with second "row" of matrix
        for(int i = n; i < totalNumOfElements; i+=n){
            // Following same logic in mvp1 function
            for(int j = i; j < (i + n); j++){
                if(j%n == mod){
                    mat[j] = 2.0;
                    mat[j - 1] = 1.0;
                    // Avoiding index out of bounds error
                    if((j+1)< totalNumOfElements){
                        mat[j + 1] = 1.0;
                    }
                }
            }
            // Going to next "row", increment mod
            mod = mod + 1;
        }
    }
}

//Main function - Where we parallelize matrix vector multiplication
int main(int argc, char **argv){
    // Get dimension of square matrix and vector from argv (In terminal:$ /_.x _pass # here_)
    if(argc == 1){
        printf("Please input an int for dimensions as parameter.\n");
    }

    // Need to convert argv[1] to an int since argv takes in chars
    int n = atoi(argv[1]);

    // Checking parameter
    if(n <= 0){
        printf("Please enter a positive integer greater than 0 as parameter.\n");
    }
    
    // Set up the vector and matrix
    double *vec2 = allocVec(n);
    double **mat2 = allocMat2(n);
    assignVec(vec2, n);
    assignMat2(mat2, n);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Seeing if amount of rank is not equally divisible by dimension - class example
    if(n % numRanks != 0){
        if(rank == 0){
            printf("Need another N.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Splitting up the matrix into seperate "sub matrices" for each rank
    int rowAmountPerRank = n / numRanks;
    double* subMatrix = (double*) malloc(n*rowAmountPerRank*sizeof(double));
    double* result = (double*) malloc(n * sizeof(double));
    double* matrixAs1DArr;

    if (rank == 0) {
        matrixAs1DArr = (double*) malloc(n*n*sizeof(double));
        // Convert 2d matrix to 1d matrx
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrixAs1DArr[i * n + j] = mat2[i][j];
            }
        }
        double endTimeForConvertingMat = MPI_Wtime();
        // printf("Rank %d Converting time %f\n", rank, endTimeForConvertingMat - startTime);
    } 

    // Start time here
    double startTime = MPI_Wtime();
    MPI_Scatter(matrixAs1DArr, rowAmountPerRank * n, MPI_DOUBLE, subMatrix, rowAmountPerRank * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double startTimeForCalculatingSubResult = MPI_Wtime();
    double* subResult = subMatrixVectorProduct(subMatrix, vec2, n, rowAmountPerRank);
    double endTimeForCalculatingSubResult = MPI_Wtime();
    printf("Rank %d Sub result calculation time %f\n", rank, endTimeForCalculatingSubResult - startTimeForCalculatingSubResult);

    MPI_Gather(subResult, rowAmountPerRank, MPI_DOUBLE, result, rowAmountPerRank, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double endFullTime = MPI_Wtime();

    if (rank == 0) {
        printf("Full Time: %f\n", endFullTime - startTime);
    }

    MPI_Finalize();

    // Free memory
    // freeMat2(mat2, vec2, result, n);
    // free(subResult);
    // free(subMatrix);
    // free(matrixAs1DArr);

    return 0;
}
