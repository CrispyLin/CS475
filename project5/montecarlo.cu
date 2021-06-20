#include "helper_cuda.h"
#include "helper_functions.h"
#include <fstream>
using namespace std;

// defining variables here
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK   128
#endif


#ifndef NUMTRIALS
#define NUMTRIALS   2048
#endif
// 4 096 4K
// 8 192 8K
// 16 384  16k
// 32 768  32k
// 65 536  64k
// 131 072   128k
// 262 144   256k
// 524 288   512k
// 1 048 576  1M


// these two #defines are just to label things
// other than that, they do nothing:
#define IN
#define OUT


// global variables
// ranges for the random numbers:
const float GMIN =	20.0;	// ground distance in meters
const float GMAX =	30.0;	// ground distance in meters
const float HMIN =	10.0;	// cliff height in meters
const float HMAX =	20.0;	// cliff height in meters
const float DMIN  =	10.0;	// distance to castle in meters
const float DMAX  =	20.0;	// distance to castle in meters
const float VMIN  =	10.0;	// intial cnnonball velocity in meters / sec
const float VMAX  =	30.0;	// intial cnnonball velocity in meters / sec
const float THMIN = 	30.0;	// cannonball launch angle in degrees
const float THMAX =	70.0;	// cannonball launch angle in degrees

const float GRAVITY =	-9.8;	// acceleraion due to gravity in meters / sec^2
const float TOL = 5.0;		// tolerance in cannonball hitting the castle in meters
ofstream fp;

// function prototypes:
void CudaCheckError( );
//inline float Radians( float d );
float		Ranf( float, float );
int		Ranf( int, int );
void		TimeOfDaySeed( );
__device__ float Radians( float d );
__global__ void MonteCarlo( float *dvs, float *dths, float *dgs, float *dhs, float *dds, int *dhits );



int main( int argc, char* argv[ ] )
{
    TimeOfDaySeed( );
    fp.open("montecarlo.csv", ofstream::app);
    printf("ThreadsPerBlock: %d  NUMTRIALS: %d  ", THREADS_PER_BLOCK, NUMTRIALS);
    int dev = findCudaDevice(argc, (const char **)argv);

    // better to define these here so that the rand() calls don't get into the thread timing:
    float *hvs   = new float [NUMTRIALS];
    float *hths  = new float [NUMTRIALS];
    float *hgs   = new float [NUMTRIALS];
    float *hhs   = new float [NUMTRIALS];
    float *hds   = new float [NUMTRIALS];
    int   *hhits = new int   [NUMTRIALS];

    // fill the random-value arrays:
    for( int n = 0; n < NUMTRIALS; n++ )
    {
        hvs[n]  = Ranf(  VMIN,  VMAX );
        hths[n] = Ranf( THMIN, THMAX );
        hgs[n]  = Ranf(  GMIN,  GMAX );
        hhs[n]  = Ranf(  HMIN,  HMAX );
        hds[n]  = Ranf(  DMIN,  DMAX );
        hhits[n] = 0;
    }


    // allocate device memory:
    float *dvs, *dths, *dgs, *dhs, *dds;
    int   *dhits;

    cudaMalloc( &dvs,   NUMTRIALS*sizeof(float) );
    cudaMalloc( &dths,  NUMTRIALS*sizeof(float) );
    cudaMalloc( &dgs,   NUMTRIALS*sizeof(float) );
    cudaMalloc( &dhs,   NUMTRIALS*sizeof(float) );
    cudaMalloc( &dds,   NUMTRIALS*sizeof(float) );
    cudaMalloc( &dhits, NUMTRIALS*sizeof(int) );
    CudaCheckError( );

    // copy host memory to the device:
    cudaMemcpy( dvs,  hvs,  NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dths, hths, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dgs,  hgs,  NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dhs,  hhs,  NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dds,  hds,  NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
    CudaCheckError( );

    // setup the execution parameters:
    dim3 grid( NUMTRIALS/THREADS_PER_BLOCK, 1, 1 );
    dim3 threads( THREADS_PER_BLOCK, 1, 1 );

    // allocate cuda events that we'll use for timing:
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    CudaCheckError( );

    // let the gpu go quiet:
    cudaDeviceSynchronize( );

    // record the start event:
    cudaEventRecord( start, NULL );
    CudaCheckError( );

    // execute the kernel:
    MonteCarlo<<< grid, threads >>>( IN dvs, IN dths, IN dgs, IN dhs, IN dds, OUT dhits );

    // record the stop event:
    cudaEventRecord( stop, NULL );
    CudaCheckError( );

    // wait for the stop event to complete:
    cudaDeviceSynchronize( );
    cudaEventSynchronize( stop );
    CudaCheckError( );

    float msecTotal = 0.0f;
    cudaEventElapsedTime( &msecTotal, start, stop );
    CudaCheckError( );

    // compute and print the performance
    float TrialsPerSecond = (float) (NUMTRIALS / (float)(msecTotal/1000)); //unit is trials/second
    float performance = (float) (TrialsPerSecond / (1000000.)); //unit is MegaTrials/second
    printf("Performance: %6.3f  ", performance);
    fp << performance << ",";
    fp.close();
    
    // copy result from the device to the host:
    cudaMemcpy( hhits, dhits, NUMTRIALS*sizeof(int), cudaMemcpyDeviceToHost );
    CudaCheckError( );

    // add up the hhits[ ] array: :
    int Total_hits = 0;
    for (int i =0; i<NUMTRIALS; i++)
    {
        Total_hits += hhits[i];
    }

    // compute and print the probability:
    float Hit_probability = (float)Total_hits/ (float)NUMTRIALS * 100.0;
	printf("Probability: %%%6.3f\n", Hit_probability); //print newline character here
    

    // clean up host memory:
    delete [ ] hvs;
    delete [ ] hths;
    delete [ ] hgs;
    delete [ ] hhs;
    delete [ ] hds;
    delete [ ] hhits;

    // clean up device memory:
    cudaFree( dvs );
    cudaFree( dths );
    cudaFree( dgs );
    cudaFree( dhs );
    cudaFree( dds );
    cudaFree( dhits );
    CudaCheckError( );

	return 0;
}


void CudaCheckError( )
{
    cudaError_t e = cudaGetLastError( );
    if( e != cudaSuccess )
    {
            fprintf( stderr, "CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e) );
    }
}


// degrees-to-radians -- callable from the device:
__device__ float Radians( float d )
{
        return (M_PI/180.f) * d;
}


// the kernel:
__global__ void MonteCarlo( float *dvs, float *dths, float *dgs, float *dhs, float *dds, int *dhits )
{
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // randomize everything:
    float v   = dvs[gid];
    float thr = Radians( dths[gid] );
    float vx  = v * cos(thr);
    float vy  = v * sin(thr);
    float  g  =  dgs[gid];
    float  h  =  dhs[gid];
    float  d  =  dds[gid];

    int numHits = 0;

    // see if the ball doesn't even reach the cliff:
    float t = -vy / ( 0.5*GRAVITY );
    float x = vx * t;
    if( x > g )
    {
        t = (float) (g/vx);
        float y = (float) vy*t + 0.5 * GRAVITY * t * t;
        if (y > h)
        {
            float a = 0.5*GRAVITY;
            float b = vy;
            float c = -h;
            float disc = b*b - 4.f*a*c;
            if(disc >= 0.)
            {
                disc = sqrtf( disc );
                float t1 = (-b + disc ) / ( 2.f*a );	// time to intersect high ground
                float t2 = (-b - disc ) / ( 2.f*a );	// time to intersect high ground

                // only care about the second intersection
                float tmax = t1;
                if( t2 > t1 )
                    tmax = t2;
                float upperDist = vx * tmax  -  g;
                if (fabs( upperDist - d ) <= TOL)
                {
                    numHits = 1;
                }
            }
        }       
    }
    dhits[gid] = numHits;
}


float Ranf( float low, float high )
{
        float r = (float) rand();               // 0 - RAND_MAX
        float t = r  /  (float) RAND_MAX;       // 0. - 1.

        return   low  +  t * ( high - low );
}

int Ranf( int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = ceil( (float)ihigh );

        return (int) Ranf(low,high);
}


// call this if you want to force your program to use
// a different random number sequence every time you run it:
void TimeOfDaySeed( )
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time( &timer );
	double seconds = difftime( timer, mktime(&y2k) );
	unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
	srand( seed );
}