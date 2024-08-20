#include "macros.h"
#include "fldind.h"
#define FIELD_TYPE Field_Model

module eom

  use constants
  use fftw3
  implicit none

  integer, parameter :: nLat  = 2048
  integer, parameter :: nyq   = nLat/2+1
  integer, parameter :: kspec = nLat/2
  integer, parameter :: nFld  = 1
  integer, parameter :: nVar  = 2*nFld*nLat+1

  real(dl), parameter :: lambda  = 1.5
  real(dl), parameter :: nu      = 2.e-3
  real(dl), parameter :: m2eff   = 4._dl*nu*(-1._dl+lambda**2._dl)

  real(dl), parameter :: lenLat  = 100._dl/(2.*nu)**0.5
  real(dl), parameter :: dx      = lenLat/dble(nLat)
  real(dl), parameter :: dk      = twopi/lenLat
  real(dl), parameter :: alph    = 8._dl

  real(dl), dimension(1:nVar), target :: yvec
  type(transformPair1D) :: tPair

contains

  subroutine derivs(yc,yp)
    real(dl), dimension(:), intent(in)  :: yc
    real(dl), dimension(:), intent(out) :: yp

    yp(TIND) = 1._dl

    yp(DFLD) = -4._dl*nu* ( sin(yc(FLD)) + 0.5_dl*lambda**2._dl * sin(2._dl*yc(FLD)) )

    tPair%realSpace(:) = yc(FLD)
    call laplacian_1d_wtype(tPair,dk)
    yp(DFLD) = yp(DFLD) + tPair%realSpace(:)

    yp(FLD)  = yc(DFLD)
  end subroutine derivs

end module eom
