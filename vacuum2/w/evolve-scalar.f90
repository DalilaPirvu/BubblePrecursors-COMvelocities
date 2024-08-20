#include "macros.h"
#define SMOOTH 1

program Gross_Pitaevskii_1d
  use, intrinsic :: iso_c_binding
  use gaussianRandomField
  use integrator
  use constants
  use eom
  implicit none

  integer :: sim
  real(dl), pointer :: time
  real(dl), dimension(:,:), pointer :: fld
  type(transformPair1D) :: tPairgsq

  fld(1:nLat,1:2) => yvec(1:nVar-1) ! store the field in yvec
  time => yvec(nVar) ! last position stores the time?
  call initialize_rand(93286123,seedfac)
  call setup(nVar)

  ! change this script to vary over specific parameters, as needed
  do sim = 0, nSim-1
      call initialise_fields(fld, m2eff, temp, type, nyq, norm, fldinit, mominit)
      if (sim >= lSim) then
 !         if (ANY(simList==sim)) then
          call time_evolve(sim, temp, alph)
          print*, "Simulation ", sim, " out of ", nSim , " done!"
 !         endif
      endif
  end do
  print*, "All Done!"

contains

  subroutine initialise_fields(fld, m2, temp, type, kc, norm, fldinit, mominit)
    real(dl), dimension(:,:), intent(inout) :: fld
    real(dl), intent(in) :: m2, temp, norm, fldinit, mominit
    integer,  intent(in) :: type, kc

    fld(:,1)   = fldinit
    fld(:,2)   = mominit
    yvec(nVar) = 0 ! Add a tcur pointer here

    call initialize_linear_fluctuations(fld, m2, temp, type, kc, norm)
  end subroutine initialise_fields

  subroutine time_evolve(sim, temp, alp) 
    real(dl) :: temp
    real(dl) :: dt
    real(dl) :: dtout
    real(dl) :: sum_cos_fld
    real(dl) :: alp
    integer :: i, j, k, m
    integer :: sim
    logical :: bool

    dt = dx / alp
!    if (dt > dx) print*, "Warning, violating Courant condition"
    dtout = dt * alp

    call output_fields(fld, dtout, sim, temp)

    i = 1
    k = 1
    bool = .True.

    do while ( i <= nLat/6. )
       do j = 1, int(alp)
          call gl10(yvec, dt)
       end do
       call output_fields(fld, dtout, sim, temp)

       if ( bool ) then
          sum_cos_fld = 0._dl
          do m = 1, nLat
             sum_cos_fld = sum_cos_fld + cos(fld(m,1))
          end do
          sum_cos_fld = sum_cos_fld/nLat

          if ( sum_cos_fld > -0.7_dl ) then
             bool = .False.
          end if

          k = k + 1
          if ( k == nTimeMax) then
             exit
          end if
       else

          i = i + 1
       end if
    end do
  end subroutine time_evolve

  subroutine initialize_linear_fluctuations(fld, m2, temp, type, kc, norm)
    real(dl), dimension(:,:), intent(inout) :: fld
    real(dl), intent(in) :: m2, temp, norm
    integer, intent(in) :: type, kc

    real(dl), dimension(1:size(fld(:,1))) :: df
    real(dl), dimension(1:size(fld(:,1))/2+1) :: spec, w2eff
    integer :: i, j, nn

    nn = size(spec)
    spec(:) = 0._dl
    do i = 1, nn
       w2eff(i) = m2 + (dk*(i-1))**2._dl
    enddo

    select case (type)
       case (1)  ! Vacuum fluctuations
          spec(2:kc) = norm / w2eff(2:kc)**0.25
       case (2)  ! Thermal + Vacuum
          spec(2:kc) = norm / w2eff(2:kc)**0.25 * sqrt(2._dl/(exp(w2eff(2:kc)**0.5/temp)-1._dl)+1._dl)
       case (3)  ! Only Thermal
          spec(2:kc) = norm / w2eff(2:kc)**0.25 * sqrt(2._dl/(exp(w2eff(2:kc)**0.5/temp)-1._dl))
    end select

    df(:) = 0._dl
    call generate_1dGRF(df, spec(1:nn))  ! check if this is correct
    fld(:,1) = fld(:,1) + df(:)

    do j = 2, nn
        spec(j) = spec(j)*(w2eff(j)**0.5)
    enddo

    df(:) = 0._dl
    call generate_1dGRF(df, spec(1:nn))
    fld(:,2) = fld(:,2) + df(:)
  end subroutine initialize_linear_fluctuations

  subroutine setup(nVar)
    integer, intent(in) :: nVar
    call init_integrator(nVar)
    call initialize_transform_1d(tPair,nLat)
    call initialize_transform_1d(tPairgsq,nLat)
  end subroutine setup

  character(len=20) function str(k)
    integer, intent(in) :: k
    write (str, *) k
    str = adjustl(str)
  end function str

  character(len=20) function real_str(k)
    real(dl), intent(in) :: k
    write (real_str, '(f12.4)') k
    real_str = adjustl(real_str)
  end function real_str

  subroutine output_fields(fld, dtout, sim, temp)
    real(dl), dimension(1:nLat, 1:2) :: fld
    real(dl), dimension(1:nLat) :: grad
!    real(dl), dimension(1:nLat) :: kinetic
!    real(dl), dimension(1:nLat) :: gradient
!    real(dl), dimension(1:nLat) :: potential
    real(dl) :: dtout, temp
    integer  :: m, sim
    logical  :: o
    integer, parameter :: oFile = 98

    tPairgsq%realSpace(:) = fld(:,1)
    call grad_1d_wtype(tPairgsq, dk)
    grad(:) = tPairgsq%realSpace(:)

!    kinetic(:)   = 0.5_dl*(fld(:,2)**2._dl)
!    gradient(:)  = 0.5_dl*(grad(:)**2._dl)
!    potential(:) = 4._dl*nu * (-cos(fld(:,1)) + 0.5_dl*lambda**2._dl * sin(fld(:,1))**2._dl)

    inquire(file='/gpfs/dpirvu/vacuum_potential/x'//trim(str(nLat))//'_phi0'//trim(real_str(phi0))//'_lambda'//trim(real_str(lambda))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_fields.dat', opened=o)

    if (.not.o) then
       open(unit=oFile,file='/gpfs/dpirvu/vacuum_potential/x'//trim(str(nLat))//'_phi0'//trim(real_str(phi0))//'_lambda'//trim(real_str(lambda))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_fields.dat')
       write(oFile,*) "# Lattice Parameters dx = ", dx, "nLat = ", nLat, " lenLat = ", lenLat, "dk = ", dk
       write(oFile,*) "# Time Stepping parameters dt_out = ", dtout, " m2eff = ", m2eff, " phi0 = ", phi0, " temperature = ", temp
    endif

    do m = 1, nLat
       write(oFile,*) fld(m,:), grad(m)
    end do

  end subroutine output_fields

end program Gross_Pitaevskii_1d
