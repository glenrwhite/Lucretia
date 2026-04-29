#pragma once

#include "mixin/Named.H"
#include "mixin/Position3D.H"
#include "mixin/TimeWindow.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <memory>
#include <string>
#include <vector>


namespace lucretiatt {
namespace elements {

/** 3D Cartesian field map loaded from a custom-schema HDF5 file.
 *
 *  HDF5 layout (lucretia-tt schema, matches MATLAB h5write of a
 *  [Nx, Ny, Nz] array written via the writeFieldMap3D helper):
 *    /Bx, /By, /Bz   - (Nx, Ny, Nz) double, HDF5 C order
 *                      (i.e. on disk, x slowest, z fastest); optional
 *    /Ex, /Ey, /Ez   - same shape as B; optional
 *    /grid_spacing   - (3,) double = [dx, dy, dz]   (m)
 *    /grid_origin    - (3,) double = [x0, y0, z0]   (m;
 *                      x0,y0 are absolute lab coords, z0 is relative
 *                      to the element start m_z)
 *
 *  At least one of {Bx,By,Bz} or {Ex,Ey,Ez} must be present. Missing
 *  components default to zero.
 *
 *  Off-grid queries return zero (no extrapolation). Spatial gating is
 *  also applied to [m_z, m_z + m_length] before the grid lookup.
 *
 *  Optional RF modulation: if m_rf_modulated is set, all loaded
 *  components are multiplied by cos(omega * t + phase) at gather time.
 *  Use this for an RF cavity provided as a static field map. */
struct FieldMap3D
    : public Named
    , public Position3D
    , public TimeWindow
{
    FieldMap3D () = default;

    FieldMap3D (
        std::string  name,
        amrex::Real  z0_element,
        amrex::Real  length,
        const std::string& path,
        bool         rf_modulated = false,
        amrex::Real  f_RF = 0.0,
        amrex::Real  phase = 0.0);

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void gather_E_B (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t,
                     amrex::Real& Ex, amrex::Real& Ey, amrex::Real& Ez,
                     amrex::Real& Bx, amrex::Real& By, amrex::Real& Bz) const
    {
        Ex = 0.0; Ey = 0.0; Ez = 0.0;
        Bx = 0.0; By = 0.0; Bz = 0.0;

        const amrex::Real z_local = z - m_z;
        if (z_local < amrex::Real(0.0) || z_local > m_length) { return; }

        // Convert to grid-local coords.
        const amrex::Real fx = (x - m_x0) / m_dx;
        const amrex::Real fy = (y - m_y0) / m_dy;
        const amrex::Real fz = (z_local - m_z0) / m_dz;

        const int ix = int(std::floor(fx));
        const int iy = int(std::floor(fy));
        const int iz = int(std::floor(fz));
        if (ix < 0 || ix >= m_Nx - 1 ||
            iy < 0 || iy >= m_Ny - 1 ||
            iz < 0 || iz >= m_Nz - 1) {
            return;
        }
        const amrex::Real wx = fx - amrex::Real(ix);
        const amrex::Real wy = fy - amrex::Real(iy);
        const amrex::Real wz = fz - amrex::Real(iz);

        // Trilinear weights for the 8 corners.
        const amrex::Real w000 = (1-wx)*(1-wy)*(1-wz);
        const amrex::Real w100 = (  wx)*(1-wy)*(1-wz);
        const amrex::Real w010 = (1-wx)*(  wy)*(1-wz);
        const amrex::Real w110 = (  wx)*(  wy)*(1-wz);
        const amrex::Real w001 = (1-wx)*(1-wy)*(  wz);
        const amrex::Real w101 = (  wx)*(1-wy)*(  wz);
        const amrex::Real w011 = (1-wx)*(  wy)*(  wz);
        const amrex::Real w111 = (  wx)*(  wy)*(  wz);

        auto sample = [&] (const amrex::Real* data) -> amrex::Real {
            // HDF5 C-order layout (x slowest, z fastest):
            //   data[ix, iy, iz] -> data[(ix*Ny + iy)*Nz + iz]
            const int s000 = (ix   * m_Ny + iy  ) * m_Nz + iz;
            const int s001 = s000 + 1;             // +1 in iz
            const int s010 = s000 + m_Nz;          // +1 in iy
            const int s011 = s010 + 1;
            const int s100 = s000 + m_Ny * m_Nz;   // +1 in ix
            const int s101 = s100 + 1;
            const int s110 = s100 + m_Nz;
            const int s111 = s110 + 1;
            return w000*data[s000] + w100*data[s100]
                 + w010*data[s010] + w110*data[s110]
                 + w001*data[s001] + w101*data[s101]
                 + w011*data[s011] + w111*data[s111];
        };

        if (m_has_E) {
            Ex = sample(m_Ex.data());
            Ey = sample(m_Ey.data());
            Ez = sample(m_Ez.data());
        }
        if (m_has_B) {
            Bx = sample(m_Bx.data());
            By = sample(m_By.data());
            Bz = sample(m_Bz.data());
        }

        if (m_rf_modulated) {
            constexpr amrex::Real two_pi = amrex::Real(6.28318530717958647692);
            const amrex::Real ct = std::cos(two_pi * m_f_RF * t + m_phase);
            Ex *= ct; Ey *= ct; Ez *= ct;
            Bx *= ct; By *= ct; Bz *= ct;
        }
    }

    // Grid metadata
    int         m_Nx = 0, m_Ny = 0, m_Nz = 0;
    amrex::Real m_dx = 0.0, m_dy = 0.0, m_dz = 0.0;
    amrex::Real m_x0 = 0.0, m_y0 = 0.0, m_z0 = 0.0;  // grid origin (relative to m_z)

    // Field data (CPU memory; GPU port deferred to Phase 7).
    std::vector<amrex::Real> m_Ex, m_Ey, m_Ez;
    std::vector<amrex::Real> m_Bx, m_By, m_Bz;
    bool        m_has_E = false;
    bool        m_has_B = false;

    // Optional RF modulation
    bool        m_rf_modulated = false;
    amrex::Real m_f_RF         = 0.0;
    amrex::Real m_phase        = 0.0;
};

} // namespace elements
} // namespace lucretiatt
