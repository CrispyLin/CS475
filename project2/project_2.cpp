#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define XMIN     -1.
#define XMAX      1.
#define YMIN     -1.
#define YMAX      1.

#define N	0.7

float Height( int, int );	// function prototype

int main( int argc, char *argv[ ] )
{
    #ifndef _OPENMP
	fprintf( stderr, "No OpenMP support!\n" );
	return 1;
    #endif

    omp_set_num_threads( NUMT );
    double time0 = omp_get_wtime( );
	// the area of a single full-sized tile:
	// (not all tiles are full-sized, though)

	float fullTileArea = (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) )  *
				( ( YMAX - YMIN )/(float)(NUMNODES-1) )  );

    float halfTileArea = (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) )  *
				( ( YMAX - YMIN )/(float)(NUMNODES-1) )  ) / 2;

    float quarterTileArea = (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) )  *
				( ( YMAX - YMIN )/(float)(NUMNODES-1) )  ) / 4;

	// sum up the weighted heights into the variable "volume"
	// using an OpenMP for loop and a reduction:
    float volume = 0.0;
	#pragma omp parallel for collapse(2), default(none), shared(fullTileArea, halfTileArea, quarterTileArea), reduction(+:volume)
    for( int iv = 0; iv < NUMNODES; iv++ )
    {
        for( int iu = 0; iu < NUMNODES; iu++ )
        {
            float z = Height( iu, iv );
            float area = 0.0;
            // determine if this node is corner node
            if((iv == NUMNODES-1 && iu == NUMNODES-1) || (iv == 0 && iu == 0) || (iv == 0 && iu == NUMNODES-1) || (iv == NUMNODES-1 && iu == 0))
                area = quarterTileArea;
            else if(iv == NUMNODES-1 || iv == 0 || iu ==0 || iu == NUMNODES-1)
                area = halfTileArea;
            else
                area = fullTileArea;
            float v = (float)area * z;
            volume += v;
        }
    }
    volume *= 2;
    double time1 = omp_get_wtime( );
	double megaHeightPerSecond = (double) (NUMNODES * NUMNODES) / ( time1 - time0 ) / 1000000.;
    fprintf(stderr, "%2d threads : %8d NUMNODES ; volume = %f ; megaHeight Computed Per Second = %6.2lf\n",
		NUMT, NUMNODES, volume, megaHeightPerSecond);
}

float Height( int iu, int iv )	// iu,iv = 0 .. NUMNODES-1
{
	float x = -1.  +  2.*(float)iu /(float)(NUMNODES-1);	// -1. to +1.
	float y = -1.  +  2.*(float)iv /(float)(NUMNODES-1);	// -1. to +1.

	float xn = pow( fabs(x), (double)N );
	float yn = pow( fabs(y), (double)N );
	float r = 1. - xn - yn;
	if( r <= 0. )
	        return 0.;
	float height = pow( r, 1./(float)N );
	return height;
}
