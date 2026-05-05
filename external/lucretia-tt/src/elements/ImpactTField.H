#pragma once

#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <string>
#include <vector>


namespace lucretiatt {
namespace elements {

/** Element backed by an ImpactT-format `rfdata*` field-map file.
 *
 *  File layout (two Fourier blocks):
 *    NF1, zmin1, zmax1, zlen1, a0, (a1,b1), (a2,b2), ...   (E_z block)
 *    NF2, zmin2, zmax2, zlen2, a0, (a1,b1), (a2,b2), ...   (B_z block)
 *  with NF = 1 + 2 * (number of Fourier modes). NF=1 means only the DC
 *  (a0) term is present; ImpactT uses NF=1 with a0=0 to denote "no field
 *  in this block". Both blocks are read; if a block has no real data,
 *  its scale_E or scale_B will typically be 0 in the lattice line and
 *  the kernel returns 0 for that field.
 *
 *  On-axis evaluation:
 *    F(z) = a0 + sum_{n=1}^{Nmodes} [ a_n cos(2 pi n z / zlen)
 *                                  +  b_n sin(2 pi n z / zlen) ]
 *  where z is measured relative to the field-map's intrinsic origin
 *  (zmin offset is folded in by the reader so we sample F directly at
 *  z_local in [zmin, zmax]; we use z_for_eval = z_local - zmin centred
 *  on the period).
 *
 *  Time + Cartesian projection (axisymmetric paraxial):
 *    Ez = scale_E * F_E(z) * cos(omega t + phase)
 *    Bz = scale_B * F_B(z) * cos(omega t + phase)
 *    dEz/dz = scale_E * F_E'(z) * cos(omega t + phase)
 *    dBz/dz = scale_B * F_B'(z) * cos(omega t + phase)
 *    dEz/dt = -omega * scale_E * F_E(z) * sin(omega t + phase)
 *    dBz/dt = -omega * scale_B * F_B(z) * sin(omega t + phase)
 *
 *    E_r = -(r/2) dEz/dz       Ex = -(x/2) dEz/dz   + (y/2) dBz/dt
 *    E_phi = -(r/2) dBz/dt     Ey = -(y/2) dEz/dz   - (x/2) dBz/dt
 *    B_r = -(r/2) dBz/dz       Bx = -(x/2) dBz/dz   - (y/(2c^2)) dEz/dt
 *    B_phi = (r/(2c^2)) dEz/dt By = -(y/2) dBz/dz   + (x/(2c^2)) dEz/dt
 *
 *  For DC (omega = 0), the cos becomes cos(phase) (constant) and the
 *  sin term -> 0, so dEz/dt = dBz/dt = 0 and the cross terms vanish.
 *
 *  Coordinate convention: m_z is the lab-frame z position to which the
 *  field map's intrinsic z = 0 corresponds (i.e. the field's "centre"
 *  is at lab z = m_z). The field is evaluated for z_local = z_lab - m_z
 *  in [zmin, zmax]. */
struct ImpactTField
    : public Named
    , public Position3D
    , public TimeWindow
{
    ImpactTField () = default;

    ImpactTField (
        std::string  name,
        amrex::Real  z_lab_edge,    // lab z corresponding to field z = 0
        amrex::Real  blength,        // m, "real used" field range = [z_lab_edge, z_lab_edge+blength]
        const std::string& path,
        amrex::Real  scale_E,        // V/m
        amrex::Real  scale_B,        // T
        amrex::Real  f_RF,           // Hz (0 for DC)
        amrex::Real  phase_rad);     // radians

    /** Evaluate the on-axis E_z or B_z Fourier sum at field-local z.
     *
     *  IMPACT-T (SolRF.f90 getfldtfour_SolRF) convention:
     *      F(z) = a0/2 + sum_{n=1}^{Nmodes}
     *               [ a_n cos(2 pi n zz / zlen) + b_n sin(2 pi n zz / zlen) ]
     *      zz   = z_local - zstart - zlen/2
     *  where z_local = lab_z - m_z (offset from element zedge), zstart is
     *  the file's first stored z (= "distance before zedge", typically
     *  negative), and zlen is the file's stored Fourier period.
     *
     *  Note the a0/2 factor (standard Fourier convention) and the
     *  argument centering on the period midpoint (zstart + zlen/2),
     *  not on zstart. For the LCLS-style 1.5-cell gun with
     *  zstart = -zlen/2 (symmetric file range about the cathode),
     *  zz = z_local, so the cathode (z_local = 0) sits on the
     *  argument-zero point where cos(arg)=1 for every mode and the
     *  Fourier sum gives the FULL peak field at the cathode. (Earlier
     *  buggy version used a0 + sum a_n cos(2 pi n (z-zmin)/zlen): that
     *  put the cathode at the SIGN-ALTERNATING sum which collapses to
     *  ~0 by destructive interference.)
     *
     *  Returns the sum value F(zz) and writes its z-derivative dF/dz to
     *  *dF_dz_out. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real fourier_eval (
        const amrex::Real* coeffs, int nf,
        amrex::Real zlen, amrex::Real zz,
        amrex::Real* dF_dz_out)
    {
        amrex::Real F = amrex::Real(0.5) * coeffs[0];
        amrex::Real dF = amrex::Real(0.0);
        if (nf > 1) {
            constexpr amrex::Real two_pi = amrex::Real(6.28318530717958647692);
            const amrex::Real wbase = two_pi / zlen;
            const int n_modes = (nf - 1) / 2;
            for (int n = 1; n <= n_modes; ++n) {
                const amrex::Real arg = wbase * amrex::Real(n) * zz;
                const amrex::Real ca  = std::cos(arg);
                const amrex::Real sa  = std::sin(arg);
                const amrex::Real an  = coeffs[2*n - 1];
                const amrex::Real bn  = coeffs[2*n];
                F  += an * ca + bn * sa;
                dF += -an * (wbase * amrex::Real(n)) * sa
                    +  bn * (wbase * amrex::Real(n)) * ca;
            }
        }
        if (dF_dz_out) { *dF_dz_out = dF; }
        return F;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = 0.0; By = 0.0; Bz = 0.0;

        constexpr amrex::Real c        = amrex::Real(299792458.0);
        constexpr amrex::Real inv_2c2  = amrex::Real(1.0) / (amrex::Real(2.0) * c * c);

        const amrex::Real z_local = z - m_z;

        // Time factor (constant for both E and B blocks)
        const amrex::Real cos_t = std::cos(m_omega * t + m_phase);
        const amrex::Real sin_t = std::sin(m_omega * t + m_phase);

        amrex::Real Ez_axis = 0.0, dEz_dz = 0.0, dEz_dt = 0.0;
        amrex::Real Bz_axis = 0.0, dBz_dz = 0.0, dBz_dt = 0.0;

        // ImpactT type-105 convention (per the manual): "the real used
        // field range in z is [zedge, zedge+Blength]". So gating is on
        // the LATTICE-line length (m_blength), not the file's intrinsic
        // [zmin, zmax]. The file z range is generally wider than the
        // physical element (the negative-z half is Fourier-padding for
        // periodicity); only the [0, Blength] slice of the file is used.
        // file_z = z_local = lab_z - m_z, with no zmin offset.
        if (z_local < amrex::Real(0.0) || z_local > m_blength) {
            return;   // outside the element's lab-z extent: zero field
        }
        if (m_E_nf > 0) {
            amrex::Real dF;
            const amrex::Real zz = z_local - m_E_zmin - amrex::Real(0.5) * m_E_zlen;
            amrex::Real F = fourier_eval(m_E_coeffs.data(), m_E_nf,
                                         m_E_zlen, zz, &dF);
            Ez_axis = m_scale_E * F  * cos_t;
            dEz_dz  = m_scale_E * dF * cos_t;
            dEz_dt  = -m_omega  * m_scale_E * F * sin_t;
        }
        if (m_B_nf > 0) {
            amrex::Real dF;
            const amrex::Real zz = z_local - m_B_zmin - amrex::Real(0.5) * m_B_zlen;
            amrex::Real F = fourier_eval(m_B_coeffs.data(), m_B_nf,
                                         m_B_zlen, zz, &dF);
            Bz_axis = m_scale_B * F  * cos_t;
            dBz_dz  = m_scale_B * dF * cos_t;
            dBz_dt  = -m_omega  * m_scale_B * F * sin_t;
        }

        Ez = Ez_axis;
        Bz = Bz_axis;

        Ex = amrex::Real(-0.5) * x * dEz_dz + amrex::Real( 0.5) * y * dBz_dt;
        Ey = amrex::Real(-0.5) * y * dEz_dz - amrex::Real( 0.5) * x * dBz_dt;
        Bx = amrex::Real(-0.5) * x * dBz_dz - inv_2c2 * y * dEz_dt;
        By = amrex::Real(-0.5) * y * dBz_dz + inv_2c2 * x * dEz_dt;
    }

    // Block 1: E_z Fourier
    int                       m_E_nf    = 0;
    amrex::Real               m_E_zmin  = 0.0;
    amrex::Real               m_E_zmax  = 0.0;
    amrex::Real               m_E_zlen  = 0.0;
    std::vector<amrex::Real>  m_E_coeffs;

    // Block 2: B_z Fourier
    int                       m_B_nf    = 0;
    amrex::Real               m_B_zmin  = 0.0;
    amrex::Real               m_B_zmax  = 0.0;
    amrex::Real               m_B_zlen  = 0.0;
    std::vector<amrex::Real>  m_B_coeffs;

    // Lattice-line scaling, timing, and gated z extent.
    // m_blength is the ImpactT type-105 V2 ("Blength") parameter: the
    // "real used" field range in lab z is [m_z, m_z + m_blength]. The
    // file's intrinsic zmin/zmax often extends BEYOND this -- e.g. for
    // the LCLS solenoid rfdata102, file z ~ [-0.49, +0.49] but Blength
    // = 0.49308, so only the upper half of the file represents the
    // physical solenoid (the negative-z half is Fourier padding for
    // periodicity).
    amrex::Real m_blength = 0.0;     // m
    amrex::Real m_scale_E = 0.0;     // V/m
    amrex::Real m_scale_B = 0.0;     // T
    amrex::Real m_omega   = 0.0;     // rad/s (= 2 pi f_RF)
    amrex::Real m_phase   = 0.0;     // rad
};

} // namespace elements
} // namespace lucretiatt
