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

    /** Evaluate F, dF/dz, d^2F/dz^2, d^3F/dz^3 at field-local zz.
     *  Used by the higher-order paraxial expansion that ImpactT applies
     *  in getfldtfour_SolRF lines 868-877:
     *    f1  = -(d^2F + F*(omega/c)^2) / 4   (correction to E_z)
     *    f1p = -(d^3F + dF*(omega/c)^2) / 4  (correction to E_x, E_y)
     *  Then E_z gets +f1*r^2, E_x/E_y get an additional -x*(f1p*r^2/4),
     *  and B_x/B_y get an additional contribution from f1*r^2/4.
     *
     *  These corrections are 2nd-order in r and matter most when the
     *  field has large second derivative (transition between cells in
     *  a multi-cell gun, where Ez changes rapidly with z). For LCLS
     *  partcl-seed they account for the residual chirp-slope mismatch
     *  vs ImpactT in the gun's second half (z=0.06-0.10m). */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void fourier_eval_higher (
        const amrex::Real* coeffs, int nf,
        amrex::Real zlen, amrex::Real zz,
        amrex::Real& F, amrex::Real& dF, amrex::Real& ddF, amrex::Real& dddF)
    {
        F    = amrex::Real(0.5) * coeffs[0];
        dF   = amrex::Real(0.0);
        ddF  = amrex::Real(0.0);
        dddF = amrex::Real(0.0);
        if (nf > 1) {
            constexpr amrex::Real two_pi = amrex::Real(6.28318530717958647692);
            const amrex::Real wbase = two_pi / zlen;
            const int n_modes = (nf - 1) / 2;
            for (int n = 1; n <= n_modes; ++n) {
                const amrex::Real k   = wbase * amrex::Real(n);
                const amrex::Real arg = k * zz;
                const amrex::Real ca  = std::cos(arg);
                const amrex::Real sa  = std::sin(arg);
                const amrex::Real an  = coeffs[2*n - 1];
                const amrex::Real bn  = coeffs[2*n];
                F    += an * ca + bn * sa;
                dF   += k    * (-an * sa + bn * ca);
                ddF  += k*k  * (-an * ca - bn * sa);
                dddF += k*k*k* ( an * sa - bn * ca);
            }
        }
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
        // Higher-order paraxial f1, f1p corrections (ImpactT
        // getfldtfour_SolRF lines 868-877). These add 2nd-order-in-r
        // corrections that account for off-axis curvature of the
        // standing-wave field. f1 corrects E_z by +f1*r^2; f1p
        // corrects E_x/E_y by -x*f1p*r^2/4; both also contribute to
        // B_x/B_y via the time-varying part. xlrep = omega/c.
        amrex::Real f1     = amrex::Real(0.0);
        amrex::Real f1p    = amrex::Real(0.0);
        amrex::Real ez1    = amrex::Real(0.0);  // = scale_E * F
        amrex::Real ezp1   = amrex::Real(0.0);  // = scale_E * dF/dz
        if (m_E_nf > 0) {
            amrex::Real F, dF, ddF, dddF;
            const amrex::Real zz = z_local - m_E_zmin - amrex::Real(0.5) * m_E_zlen;
            fourier_eval_higher(m_E_coeffs.data(), m_E_nf, m_E_zlen, zz,
                                F, dF, ddF, dddF);
            ez1   = m_scale_E * F;
            ezp1  = m_scale_E * dF;
            const amrex::Real ezpp1 = m_scale_E * ddF;
            const amrex::Real ezppp = m_scale_E * dddF;
            const amrex::Real xlrep = m_omega / c;
            const amrex::Real xlrep2 = xlrep * xlrep;
            f1  = -(ezpp1 + ez1  * xlrep2) * amrex::Real(0.25);
            f1p = -(ezppp + ezp1 * xlrep2) * amrex::Real(0.25);
            Ez_axis = ez1  * cos_t;
            dEz_dz  = ezp1 * cos_t;
            dEz_dt  = -m_omega * ez1 * sin_t;
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

        // Higher-order corrections (in 1/m^2 effective at given r^2)
        const amrex::Real r2     = x*x + y*y;
        const amrex::Real f1_r2  = f1  * r2;
        const amrex::Real f1p_r2 = f1p * r2;

        Ez = (ez1 + f1_r2) * cos_t;
        Bz = Bz_axis;

        // Ex, Ey: paraxial transverse with f1p correction (ImpactT line 873)
        Ex = -x * (ezp1 * amrex::Real(0.5) + f1p_r2 * amrex::Real(0.25)) * cos_t
           + amrex::Real(0.5) * y * dBz_dt;
        Ey = -y * (ezp1 * amrex::Real(0.5) + f1p_r2 * amrex::Real(0.25)) * cos_t
           - amrex::Real(0.5) * x * dBz_dt;

        // Bx, By: dEz/dt-driven contribution with f1 correction (ImpactT line 876)
        // tmpbx = pos(2)*xlrep/c * (ez1/2 + f1*r^2/4) * sin
        const amrex::Real xlrep_over_c = m_omega / (c * c);
        Bx = -amrex::Real(0.5) * x * dBz_dz
           + y * xlrep_over_c * (ez1 * amrex::Real(0.5) + f1_r2 * amrex::Real(0.25)) * sin_t;
        By = -amrex::Real(0.5) * y * dBz_dz
           - x * xlrep_over_c * (ez1 * amrex::Real(0.5) + f1_r2 * amrex::Real(0.25)) * sin_t;
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
