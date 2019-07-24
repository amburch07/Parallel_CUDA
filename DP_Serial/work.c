#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int m, n;

//headers
double* assignVec(double* vec, char **argv);
double* assignMat(double* mat, char **argv);
void printMatVec(double* mat, double* vec, double* result);

//if you want to input the full matrix use pbs_work,otherwise use pbs_workMat
//for both: first user inputs are the matrix dimensions
int main(int argc, char **argv){
        m = atoi(argv[1]);
        n = atoi(argv[2]);

    	clock_t startfull = clock();
	clock_t startalloc = clock(); //time alloc+init

	double *vec = malloc(m * sizeof(double));
	double *mat = malloc(m * n*sizeof(double));
        double *result = (double*)malloc(m*sizeof(double));
	if (argc <= 3) { //did user give mat/vec or just dims?
                printf("No matrix/vector entered, will use default assignment (m*n)\n");
                for (int i=0; i<(m*n); i++){ mat[i] = 1.0; }
		for (int j=0; j<n; j++){ vec[j] = 2.0; }
        }
	else {
                vec = assignVec(vec, argv);
                mat = assignMat(mat, argv);
        }
	clock_t endalloc = clock();

	clock_t startcalc = clock();
	int count =0;
        for(int i=0; i<m; i++) {
                for(int j=0; j<n; j++) { result[count] += mat[n*i+j]*vec[j]; }
		count++;
        };
	clock_t endcalc = clock();
	clock_t endfull = clock();

	printf("Full Time: %f\n",((double) (endfull - startfull))/CLOCKS_PER_SEC);
	printf("Alloc+Init Time: %f\n", ((double) (endalloc - startalloc))/CLOCKS_PER_SEC);
	printf("Calc Time: %f\n\n", ((double) (endcalc - startcalc))/CLOCKS_PER_SEC);
	if (m+n < 20){ printMatVec(mat,vec,result);}

	free(vec); free(mat); free(result);
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


