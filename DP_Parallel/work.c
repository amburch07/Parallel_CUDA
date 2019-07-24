#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

static int m,n;

//headers
double* assignVec(double* vec, char **argv);
double* assignMat(double* mat, char **argv);
void printMatVec(double* mat, double* vec, double* result );

//if you want to input the full matrix use pbs_work,otherwise use pbs_workMat
//for both: first user inputs are the matrix dimensions
int main(int argc, char **argv){
        m = atoi(argv[1]);
        n = atoi(argv[2]);

	int rank, numRanks;
        MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double startfull = MPI_Wtime();

	int numele = m / numRanks; //int divison, may have leftovers
        int mystart = rank * numele;
        int myend =  mystart + numele;
        if(rank==numRanks-1) { //deal with leftovers
                myend = m;
                numele = myend - mystart;
        }

	double startalloc=MPI_Wtime(); //time alloc+init
	double *vec = malloc(m * sizeof(double));
        double *mat = malloc(m * n*sizeof(double));
        double *result = (double*)malloc(numele*sizeof(double));

	double *resultAll;
	if (rank==0) {
		resultAll = (double*)malloc(m*sizeof(double));
	}
	if (argc <= 3) { //did user give mat/vec or just dims?
                if (rank==0) printf("No matrix/vector entered, will use default assignment (m*n)\n");
                for (int i=0; i<(m*n);i++) {
                        mat[i] = 1.0;
                }
		for (int j=0; j<n;j++) {
			vec[j] = 2.0;
		}
        }
	else {
                vec = assignVec(vec, argv);
                mat = assignMat(mat, argv);
        }
	double endalloc = MPI_Wtime();

	double startcalc = MPI_Wtime();
	int count =0;;
        for(int i=mystart; i<myend; i++) {
                for(int j=0;j<n;j++) {
			result[count] += mat[n*i+j]*vec[j];
                }
		count++;
        };
	double endcalc = MPI_Wtime();

	double startgather = MPI_Wtime();
	MPI_Gather(result, numele, MPI_DOUBLE, resultAll, numele, MPI_DOUBLE,0,MPI_COMM_WORLD);
	double endgather = MPI_Wtime();

	double endfull = MPI_Wtime();

	if (rank==0) {
		printf("Number of Ranks: %d\n",numRanks);
		printf("Full Time: %f\n",endfull-startfull);
		printf("Alloc+Init Time: %f\n", endalloc-startalloc);
		printf("Calc Time: %f\n", endcalc-startcalc);
		printf("Gather Time: %f\n\n", endgather-startgather);
		if (m+n < 20){ printMatVec(mat,vec,resultAll); }
		free(resultAll);
	}
	free(vec);
        free(mat);
        free(result);

	MPI_Finalize();
	return 0;
}



//helper method to print out dot product
void printMatVec(double* mat, double* vec, double* result ){
        //printmat
        for (int i=0; i<m;i++) {
                for (int j=0; j<n;j++) {
                        printf("%f ",mat[i+j]);
                }
                printf("\n");
        }
        printf("    *\n");

        //print vec
        for (int i=0; i<(n);i++) {
                printf("%f\n",vec[i]);
        }
        printf("    =\n");

        //print result
        for (int i=0; i<m;i++) {
                printf("%f\n",result[i]);
        }
}

//if user gives matrix and vector then read in
double* assignMat(double* mat, char **argv){
        int counter = 3;
        for(int i = 0; i < (m*n); i++){
                mat[i] = atoi(argv[counter]);
                counter++;
        };
        return mat;
}
double* assignVec(double* vec, char **argv){
        int counter = (m*n)+3;
        for(int i = 0; i < n; i++){ vec[i] = atoi(argv[counter]); };
        return vec;
}

