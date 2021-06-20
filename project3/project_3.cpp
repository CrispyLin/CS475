#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>



//global variables
const float GRAIN_GROWS_PER_MONTH = 15;
const float ONE_DEER_EATS_PER_MONTH =   1;


const float AVG_PRECIP_PER_MONTH =  7.0;	// average
const float AMP_PRECIP_PER_MONTH =  6.0;	// plus or minus
const float RANDOM_PRECIP = 2.0;	// plus or minus noise

const float AVG_TEMP =	60.0;	// average
const float AMP_TEMP =	20.0;	// plus or minus
const float RANDOM_TEMP = 10.0;	// plus or minus noise

const float MIDTEMP =   40.0;
const float MIDPRECIP = 10.0;

const float Artificial_Rainfall = 5.0;

int	NowYear;		// 2021 - 2026
int	NowMonth;		// 0 - 11
unsigned int seed = 0;

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	NowNumDeer;		// number of deer in the current population
float NowArtificialRainfall;

omp_lock_t	Lock;
int		NumInThreadTeam;
int		NumAtBarrier;
int		NumGone;

// function prototypes
float Ranf( unsigned int *,  float, float);
int Ranf( unsigned int *, int, int);
float SQR(float);
void Deer();
void Grain();
void Watcher();
void ArtificialRainfall();
void InitBarrier( int );
void WaitBarrier( );


int main( int argc, char *argv[ ] )
{
    #ifndef _OPENMP
	fprintf( stderr, "No OpenMP support!\n" );
	return 1;
    #endif

    srand (time(NULL));
    // starting date and time:
    NowMonth =    0;
    NowYear  = 2021;

    // starting state (feel free to change this if you want):
    NowNumDeer = 2;
    NowHeight =  2.;
    NowArtificialRainfall = rand() % 11 + 5;

    // starting NowPrecip and NowTemp
    float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

    float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
    if( NowPrecip < 0. )
        NowPrecip = 0.;


    InitBarrier( NUMT );
    omp_set_num_threads( NUMT );	// same as # of sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Deer( );
        }

        #pragma omp section
        {
            Grain( );
        }

        #pragma omp section
        {
            Watcher( );
        }
        
        #pragma omp section
        {
            ArtificialRainfall( );	// your own
        }
    }       // implied barrier -- all functions must return in order
	// to allow any of them to get past here
}


void Deer()
{
    while( NowYear<2027 )
    {
        // calculate next num of deer
        int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)( NowHeight );
        if( nextNumDeer < carryingCapacity )
                nextNumDeer++;
        else{
            if( nextNumDeer > carryingCapacity )
                    nextNumDeer--;
        }
        if( nextNumDeer < 0 )
            nextNumDeer = 0;

        WaitBarrier();

        // assign next nextNumDeer to NowNumDeer
        NowNumDeer = nextNumDeer;
        WaitBarrier();

        //do nothing
        WaitBarrier();
    }
}


void Grain()
{
    while( NowYear<2027 )
    {
        //calculate next height
        float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
        float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );

        float nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        if( nextHeight < 0. ) 
            nextHeight = 0.;

        WaitBarrier();

        //assign nextHeight to NowHeight
        NowHeight = nextHeight;
        
        WaitBarrier();

        //do nothing
        WaitBarrier();
    }
}


void Watcher()
{
    while( NowYear<2027 )
    {
        //do nothing
        WaitBarrier();

        //do nothing
        WaitBarrier();

        // write out Now state data
        // print out values for temperature, precipitation, number of deer, 
        // height of the grain, and your own-choice quantity as a function of month number.
        printf("%d/%d Temp: %f Precip: %f NumOfDeer: %d HeightOfGrain: %f AR: %f\n", NowMonth, NowYear, (5./9.)*(NowTemp-32), NowPrecip*2.45, NowNumDeer, NowHeight*2.45, NowArtificialRainfall*2.45);

        // increment Time
        NowMonth ++;
        if(NowMonth >= 12){
            NowYear ++;
            NowMonth = 0;
        }

        // calculate new environemnt parameters which are NowPrecip and NowTemp for next loop
        float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

        float temp = AVG_TEMP - AMP_TEMP * cos( ang );
        NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
        NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
        if( NowPrecip < 0. )
            NowPrecip = 0.;

        NowPrecip += NowArtificialRainfall;
        WaitBarrier();
    }
}


void ArtificialRainfall()
{
    while( NowYear<2027 )
    {
        //calculate next Artificial Rainfall
        float nextArtificialRainfall = rand() % 11 + 5;  

        WaitBarrier();

        //assign nextAR to NowAR
        NowArtificialRainfall = nextArtificialRainfall;
        
        WaitBarrier();

        //do something
        WaitBarrier();
    }
}


float Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


int Ranf( unsigned int *seedp, int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
}


float SQR( float x )
{
        return x*x;
}


// specify how many threads will be in the barrier:
//	(also init's the Lock)
void InitBarrier( int n )
{
        NumInThreadTeam = n;
        NumAtBarrier = 0;
	omp_init_lock( &Lock );
}


// have the calling thread wait here until all the other threads catch up:
void WaitBarrier( )
{
        omp_set_lock( &Lock );
        {
                NumAtBarrier++;
                if( NumAtBarrier == NumInThreadTeam )
                {
                        NumGone = 0;
                        NumAtBarrier = 0;
                        // let all other threads get back to what they were doing
			// before this one unlocks, knowing that they might immediately
			// call WaitBarrier( ) again:
                        while( NumGone != NumInThreadTeam-1 );
                        omp_unset_lock( &Lock );
                        return;
                }
        }
        omp_unset_lock( &Lock );

        while( NumAtBarrier != 0 );	// this waits for the nth thread to arrive

        #pragma omp atomic
        NumGone++;			// this flags how many threads have returned
}
