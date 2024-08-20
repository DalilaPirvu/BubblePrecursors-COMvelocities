module constants
  integer, parameter :: dl = kind(1.d0)
  real(dl), parameter :: twopi =  6.2831853071795864769252867665590

  !> Initialise fluctuations in linear perturbation theory approximation
  !> type : (1) sets vacuum
  !>        (2) thermal+vacuum
  !>        (3) only thermal
  integer, parameter :: type = 3
  integer, parameter :: seedfac = 9!12

  integer, parameter :: nLat     = 1024
  integer, parameter :: nTimeMax = nLat
  integer, parameter :: lSim = 0, nSim = 1

  real(dl), parameter :: phi0   = twopi / 4.5_dl
  real(dl), parameter :: temp   = 0.1
  real(dl), parameter :: lambda = 1.5

  real(dl), parameter :: nu     = 2.e-3
  real(dl), parameter :: lenLat = 50._dl/(2.*nu)**0.5
  real(dl), parameter :: m2eff  = 4._dl*nu*(-1._dl+lambda**2._dl)

  real(dl), parameter :: dx   = lenLat/dble(nLat)
  real(dl), parameter :: dk   = twopi/lenLat
  real(dl), parameter :: alph = 16._dl

  real(dl), parameter :: fldinit = twopi / 2._dl
  real(dl), parameter :: mominit = 0._dl

  integer, parameter :: nyq   = nLat/2+1
  integer, parameter :: kspec = nLat/2
  integer, parameter :: nFld  = 1
  integer, parameter :: nVar  = 2*nFld*nLat+1

  ! Normalise assuming the Box-Mueller transform gives a complex random deviate with unit variance
  real(dl), parameter :: norm = 1._dl / phi0 / sqrt(2._dl * lenLat)  

!!!!!!!!!!!!!!!!!!!!!!
! define box average <cos(phi)> = c = -0.7 as the moment the decay took place
! from that moment, we evolve another nLat/2 timesteps so that tv fills whole volume
! interrupt evolution if no decay occured up to nTimeMax
!!!!!!!!!!!!!!!!!!!!!!

end module constants
