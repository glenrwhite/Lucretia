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

/** Axisymmetric solenoid loaded from a 2D B_z(r, z) HDF5 map.
 *
 *  HDF5 layout (lucretia-tt schema, matches MATLAB's writeFieldMap2D
 *  helper which writes a [Nr, Nz] array):
 *    /Bz             - (Nr, Nz) double, HDF5 C order
 *                      (on disk r slowest, z fastest)
 *    /grid_spacing   - (2,) double = [dr, dz]   (m)
 *    /grid_origin    - (2,) double = [r0, z0]   (m;
 *                      r0 typically 0, z0 relative to element start m_z)
 *
 *  At construction, dB_z/dz is precomputed on the (Nz, Nr) grid via
 *  centered differences (forward/backward at endpoints) so gather time
 *  is just a bilinear interpolation in (r, z).
 *
 *  gather_E_B uses paraxial expansion:
 *     B_z(x,y,z) = B_z(r, z_local)
 *     B_r        = -(r/2) * (dB_z/dz)(r, z_local)
 *  In Cartesian:  B_x = (x/r) * B_r,  B_y = (y/r) * B_r
 *  The (1/r) and r factors cancel so the kernel is well-defined on axis. */
struct Solenoid3D
    : public Named
    , public Position3D
    , public TimeWindow
{
    Solenoid3D () = default;

    Solenoid3D (
        std::string  name,
        amrex::Real  z0_element,
        amrex::Real  length,
        const std::string& path);

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real /*t*/,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = 0.0; By = 0.0; Bz = 0.0;

        const amrex::Real z_local = z - m_z;
        if (z_local < amrex::Real(0.0) || z_local > m_length) { return; }

        const amrex::Real r = std::sqrt(x*x + y*y);

        const amrex::Real fr = (r - m_r0) / m_dr;
        const amrex::Real fz = (z_local - m_z0) / m_dz;
        const int ir = int(std::floor(fr));
        const int iz = int(std::floor(fz));
        if (ir < 0 || ir >= m_Nr - 1 ||
            iz < 0 || iz >= m_Nz - 1) {
            return;
        }
        const amrex::Real wr = fr - amrex::Real(ir);
        const amrex::Real wz = fz - amrex::Real(iz);

        const amrex::Real w00 = (1-wr)*(1-wz);
        const amrex::Real w10 = (  wr)*(1-wz);
        const amrex::Real w01 = (1-wr)*(  wz);
        const amrex::Real w11 = (  wr)*(  wz);

        auto sample = [&] (const amrex::Real* data) -> amrex::Real {
            // (Nr, Nz) layout: data[ir, iz] -> data[ir * Nz + iz]
            const int s00 = ir   * m_Nz + iz;
            const int s01 = s00  + 1;          // +1 in iz
            const int s10 = s00  + m_Nz;       // +1 in ir
            const int s11 = s10  + 1;
            return w00*data[s00] + w10*data[s10] + w01*data[s01] + w11*data[s11];
        };

        const amrex::Real Bz_v = sample(m_Bz.data());
        const amrex::Real dBz  = sample(m_dBz_dz.data());
        const amrex::Real Br_over_r = amrex::Real(-0.5) * dBz;

        Bx = x * Br_over_r;
        By = y * Br_over_r;
        Bz = Bz_v;
    }

    int         m_Nr = 0, m_Nz = 0;
    amrex::Real m_dr = 0.0, m_dz = 0.0;
    amrex::Real m_r0 = 0.0, m_z0 = 0.0;

    std::vector<amrex::Real> m_Bz;        // (Nz, Nr) row-major
    std::vector<amrex::Real> m_dBz_dz;    // same shape, precomputed
};

} // namespace elements
} // namespace lucretiatt
