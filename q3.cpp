/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <stdio.h> //this library is needed for printf function
#include <stdlib.h> //this library is needed for rand() function
#include <windows.h> //this library is needed for pause() function
#include <time.h> //this library is needed for clock() function
#include <math.h> //this library is needed for abs()
#include <pmmintrin.h>
#include <process.h>
//#include <chrono>
#include <iostream>
#include <immintrin.h>

void initialize();
void initialize_again();
void slow_routine(float alpha, float beta);//you will optimize this routine
void q3_vectorized(float alpha, float beta);
unsigned short int Compare(float alpha, float beta);
unsigned short int equal(float const a, float const b) ;

#define N 8192 //input size
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N];

#define TIMES_TO_RUN 1 //how many times the function will run
#define EPSILON 0.0001

int main() {

float alpha=0.23f, beta=0.45f;

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now

	initialize();

	start_1 = clock(); //start the timer 

	for (int i = 0; i < TIMES_TO_RUN; i++)//this loop is needed to get an accurate ex.time value
 		slow_routine(alpha,beta);
		

	end_1 = clock(); //end the timer 

	printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time

	if (Compare(alpha,beta) == 0)
		printf("\nCorrect Result\n");
	else 
		printf("\nINcorrect Result\n");

	system("pause"); //this command does not let the output window to close

	return 0; //normally, by returning zero, we mean that the program ended successfully. 
}


void initialize(){

unsigned int    i,j;

//initialization
for (i=0;i<N;i++)
for (j=0;j<N;j++){
A[i][j]= 1.1f;

}

for (i=0;i<N;i++){
z[i]=(i%9)*0.8f;
x[i]=0.1f;
u1[i]=(i%9)*0.2f;
u2[i]=(i%9)*0.3f;
v1[i]=(i%9)*0.4f;
v2[i]=(i%9)*0.5f;
w[i]=0.0f;
y[i]=(i%9)*0.7f;
}

}

void initialize_again(){

unsigned int    i,j;

//initialization
for (i=0;i<N;i++)
for (j=0;j<N;j++){
A[i][j]= 1.1f;

}

for (i=0;i<N;i++){
z[i]=(i%9)*0.8f;
x[i]=0.1f;
test[i]=0.0f;
u1[i]=(i%9)*0.2f;
u2[i]=(i%9)*0.3f;
v1[i]=(i%9)*0.4f;
v2[i]=(i%9)*0.5f;
y[i]=(i%9)*0.7f;
}

}

//you will optimize this routine
void slow_routine(float alpha, float beta){

unsigned int i,j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];


  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < N; i++) 
    x[i] = x[i] + z[i];


  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];


}

void q3_vectorized(float alpha, float beta) {
	unsigned int i, j;

	__m128 alphaVec, betaVec, num1, num2, num3, num4, num5, num6;
	__m256 num7, num8, num9;

	//Vectorized constants for use in loops
	alphaVec = _mm_set_ps1(alpha);
	betaVec = _mm_set_ps1(beta);

	//A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
	for (i = 0; i < (N / 4) * 4; i += 4) {
		num2 = _mm_loadu_ps(&u1[i]);
		num3 = _mm_loadu_ps(&u2[i]);
		for (j = 0; j < (N / 4) * 4; j+= 4) {
			num1 = _mm_loadu_ps(&A[i][j]);
			num4 = _mm_loadu_ps(&v1[j]);
			num5 = _mm_loadu_ps(&v2[j]);
			num1 = _mm_add_ps(num1, num2); //A[][] + u1[]
			num1 = _mm_fmadd_ps(num1, num4, num3); //((A[][] + u1[]) * v1[]) + u2[]
			num1 = _mm_mul_ps(num1, num5); // (above) * v2
		}
	}
	
	for (i = (N / 4) * 4; i < N; i++) {
		for (j = (N / 4) * 4; j < N; j++) {
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}



	//x[i] = x[i] + beta * A[j][i] * y[j];
	for (i = 0; i < (N / 4) * 4; i += 4) {
		num1 = _mm_loadu_ps(&x[i]);
		for (j = 0; j < (N / 4) * 4; j+= 4) {
			num2 = _mm_loadu_ps(&y[j]);
			num3 = _mm_loadu_ps(&A[j][i]);
			num1 = _mm_add_ps(num1, betaVec); //x[] + beta
			num2 = _mm_mul_ps(num2, num3); //A[][] * y[]
			num1 = _mm_mul_ps(num1, num2); //x[] * (A[][] * y[])
			_mm_storeu_ps(&x[i], num1); //MULTIPLICATION COULD CAUSE ERRORS?????????
		}
	}

	for (i = (N / 4) * 4; i < N; i++) {
		for (j = (N / 4) * 4; j < N; j++) {
			x[i] = x[i] + beta * A[j][i] * y[j];
		}
	}



	//x[i] = x[i] + z[i]
	for (i = 0; i < (N / 4) * 4; i += 4) { //As number may not be divisiable by 4, we can only vectorize certain amount
		num1 = _mm_loadu_ps(&x[i]); //packs our 4 values of x[]
		num2 = _mm_loadu_ps(&z[i]); //Same as above for z[]
		num3 = _mm_add_ps(num1, num2); //Adds all 4 at once
		_mm_storeu_ps(&x[i], num3); //stores results back to x 
	}

	//Non vectorized code to allow excess to run
	for (i = (N / 4) * 4; i < N; i++) {
		x[i] = x[i] + z[i];
	}


	//w[i] = w[i] +  alpha * A[i][j] * x[j];
	for (i = 0; i < N; i++) {
		num1 = _mm_loadu_ps(&w[i]);
		for (j = 0; j < N; j++) {
			num2 = _mm_loadu_ps(&A[i][j]);
			num3 = _mm_loadu_ps(&x[j]);
			num1 = _mm_add_ps(num1, alphaVec);
			num2 = _mm_mul_ps(num2, num3);

		}
	}
}


unsigned short int Compare(float alpha, float beta) {

unsigned int i,j;

initialize_again();


  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];


  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];


  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
     test[i]= test[i] + alpha * A[i][j] * x[j];
     } }



    for (j = 0; j < N; j++){
	if (equal(w[j],test[j]) == 1){
	  printf("\n %f %f",test[j], w[j]);
		return -1;
		}
		}

	return 0;
}




unsigned short int equal(float const a, float const b) {
	
	if (fabs(a-b)/fabs(a) < EPSILON)
		return 0; //success
	else
		return 1;
}



