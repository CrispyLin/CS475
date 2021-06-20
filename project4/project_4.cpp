//Author: Xinwei Lin
//Student ID: 933332253
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <xmmintrin.h>

#define SSE_WIDTH		4

//global variables
const int NUM_TRIALS = 10;

//function prototypes
void SimdMul(float *, float *, float *, int);
float SimdMulSum(float*, float *, int);
void generate_rand_nums(float*);
void initialize_array(float*);
float my_own(float*, float*);

int main( int argc, char *argv[ ] )
{
	srand (time(NULL));
	int peek_performance_my_own = 9999;
	int peek_performance_SSE = 9999;
	float sum_my_own = 0.0;
	float sum_SSE = 0.0;
	float * array_A = new float [ARRAY_SIZE];
	float * array_B = new float [ARRAY_SIZE];
	//float * array_C = new float [ARRAY_SIZE];
	generate_rand_nums(array_A);
	generate_rand_nums(array_B);

	for(int i=0; i<NUM_TRIALS;i++){
		// my own calculation
		double time0 = omp_get_wtime();
		sum_my_own = my_own(array_A, array_B);
		double time1 = omp_get_wtime();

		// calculate performance, unit is second
		int performance = (int) (ARRAY_SIZE * 3) / ( time1 - time0 ) / 1000000.;
		// if this time the proformance is smaller, which means running fast/better than before
		// replace peek performance
		if (peek_performance_my_own > performance)
		{	
			peek_performance_my_own = performance;
			//std::cout<<"Peek_performance_my_own: "<< peek_performance_my_own<<std::endl;
		}


		// below is SIMD SSE calculation
		time0 = omp_get_wtime();
		sum_SSE = SimdMulSum(array_A, array_B, ARRAY_SIZE);
		time1 = omp_get_wtime();
		performance = (int) (ARRAY_SIZE * 3) / ( time1 - time0 ) / 1000000.;
		if (peek_performance_SSE > performance)
		{
			peek_performance_SSE = performance;
			//std::cout<<"Peek_performance_SSE: "<< peek_performance_SSE<<std::endl;
		}
	}
	// calculate speed up
	// S = Psse/Pnon-sse = Tnon-sse/Tsse
	// here I use S = Psse/Pnon-sse
	float speed_up = (float) peek_performance_SSE / peek_performance_my_own;
	printf("Array size: %d  NumOfTrials: %d  Speed Up: %f  peek performance of non SSE: %d  peek performance of SSE: %d\n", ARRAY_SIZE, NUM_TRIALS, speed_up, peek_performance_my_own, peek_performance_SSE);
}


float my_own(float* array_A, float* array_B)
{
	// do multiplication
	float sum = 0.0;
	for(int i=0; i<ARRAY_SIZE;i++)
	{
		sum += (float) array_A[i] * array_B[i];
	}

	return sum;
}


void initialize_array(float * a)
{
	for(int i=0;i<ARRAY_SIZE;i++){
		a[i] = 0.0;
	}
	return;
}


void generate_rand_nums(float* a){
	for (int i=0; i<ARRAY_SIZE; i++){
		a[i] = (float) (rand() % 10 + 1.0); // generate num between 1-10
	}
	return; 
}


void SimdMul( float *a, float *b,   float *c,   int len )
{
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;
	register float *pc = c;
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		_mm_storeu_ps( pc,  _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
		pc += SSE_WIDTH;
	}

	for( int i = limit; i < len; i++ )
	{
		c[i] = a[i] * b[i];
	}
}


float SimdMulSum( float *a, float *b, int len )
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	__m128 ss = _mm_loadu_ps( &sum[0] );
	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps( &sum[0], ss );

	for( int i = limit; i < len; i++ )
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}