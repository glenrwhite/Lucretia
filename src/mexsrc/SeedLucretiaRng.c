/* SeedLucretiaRng.c
 * MATLAB-callable seed for the per-thread xoshiro256** RNG used by the
 * OpenMP-parallelised tracking kernels.
 *
 * Usage:
 *
 *     SeedLucretiaRng( seed )         % seed is a non-negative integer
 *     v = SeedLucretiaRng('version')  % return the version string
 *
 * Notes:
 *
 * - This affects ONLY the C-side per-thread RNG that the synchrotron-
 *   radiation sampling (SRSpectrumHB / SRSpectrumAW / poidev) consumes
 *   inside parallel tracking loops.  MATLAB's own rand / randn streams
 *   are not touched.  When Lucretia is built with the 'mlrand' option
 *   the MATLAB streams are used instead and this function has no
 *   tracking effect.
 *
 * - Call this BEFORE entering a parallel tracking call so that every
 *   worker thread sees the new master seed on its next RNG call.  Each
 *   thread re-seeds its private xoshiro state on first use after a
 *   master-seed change.
 *
 * AUTH: 27-Apr-2026 */

#include <mex.h>
#include "matrix.h"
#include "LucretiaGlobalAccess.h"
#include <string.h>

char SeedLucretiaRngVersion[] =
    "SeedLucretiaRng Matlab version = 27-Apr-2026" ;

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  double s ;

  if ( nrhs != 1 )
    mexErrMsgIdAndTxt( "Lucretia:SeedLucretiaRng:nargin",
        "Usage: SeedLucretiaRng(seed) or SeedLucretiaRng('version')" ) ;

  if ( mxIsChar( prhs[0] ) )
  {
    char buf[16] ;
    if ( mxGetString( prhs[0], buf, sizeof(buf) ) == 0
         && strcmp( buf, "version" ) == 0 )
    {
      plhs[0] = mxCreateString( SeedLucretiaRngVersion ) ;
      return ;
    }
    mexErrMsgIdAndTxt( "Lucretia:SeedLucretiaRng:badArg",
        "String argument must be 'version'" ) ;
  }

  if ( !mxIsNumeric( prhs[0] ) || mxGetNumberOfElements( prhs[0] ) != 1 )
    mexErrMsgIdAndTxt( "Lucretia:SeedLucretiaRng:badArg",
        "Seed must be a numeric scalar" ) ;

  s = mxGetScalar( prhs[0] ) ;
  if ( s < 0 || !mxIsFinite(s) )
    mexErrMsgIdAndTxt( "Lucretia:SeedLucretiaRng:badArg",
        "Seed must be a non-negative finite scalar" ) ;

  LucretiaSeedThreadRng( (unsigned long long) s ) ;
}
