#include "Solenoid3D.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <hdf5.h>

#include <stdexcept>


namespace lucretiatt {
namespace elements {

namespace {

bool read_1d (hid_t file, const char* name, std::vector<double>& out)
{
    if (H5Lexists(file, name, H5P_DEFAULT) <= 0) { return false; }
    hid_t dset  = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2] = {0, 0};
    int ndim = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    hsize_t total = 1;
    for (int i = 0; i < ndim; ++i) { total *= dims[i]; }
    out.resize(total);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    return true;
}

bool read_2d (hid_t file, const char* name,
              std::vector<double>& out, int dims_out[2])
{
    if (H5Lexists(file, name, H5P_DEFAULT) <= 0) { return false; }
    hid_t dset  = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    int ndim = H5Sget_simple_extent_ndims(space);
    if (ndim != 2) {
        H5Sclose(space); H5Dclose(dset);
        throw std::runtime_error(
            std::string("Solenoid3D: dataset '") + name +
            "' has " + std::to_string(ndim) + " dims, expected 2");
    }
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    dims_out[0] = int(dims[0]);
    dims_out[1] = int(dims[1]);
    out.resize(dims[0] * dims[1]);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    return true;
}

template <class T>
std::vector<T> to_real (const std::vector<double>& src)
{
    std::vector<T> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) { out[i] = T(src[i]); }
    return out;
}

} // anonymous


Solenoid3D::Solenoid3D (
    std::string  name,
    amrex::Real  z0_element,
    amrex::Real  length,
    const std::string& path)
    : Named(std::move(name))
    , Position3D(0.0, 0.0, z0_element, length)
    , TimeWindow()
{
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error(
            "Solenoid3D: cannot open HDF5 file '" + path + "'");
    }

    std::vector<double> spacing, origin;
    if (!read_1d(file, "/grid_spacing", spacing) || spacing.size() != 2) {
        H5Fclose(file);
        throw std::runtime_error("Solenoid3D: /grid_spacing must be size-2 in '" + path + "'");
    }
    if (!read_1d(file, "/grid_origin", origin) || origin.size() != 2) {
        H5Fclose(file);
        throw std::runtime_error("Solenoid3D: /grid_origin must be size-2 in '" + path + "'");
    }
    m_dr = amrex::Real(spacing[0]);
    m_dz = amrex::Real(spacing[1]);
    m_r0 = amrex::Real(origin[0]);
    m_z0 = amrex::Real(origin[1]);

    int dims[2] = {0, 0};
    std::vector<double> tmp;
    if (!read_2d(file, "/Bz", tmp, dims)) {
        H5Fclose(file);
        throw std::runtime_error("Solenoid3D: /Bz dataset missing in '" + path + "'");
    }
    H5Fclose(file);

    // HDF5 dim 0 = r (slowest), dim 1 = z (fastest). Matches MATLAB's
    // writeFieldMap2D output for a [Nr, Nz] array.
    m_Nr = dims[0];
    m_Nz = dims[1];
    m_Bz = to_real<amrex::Real>(tmp);

    // Pre-compute dB_z/dz on the grid via centered differences
    // (forward/backward at the longitudinal endpoints).
    m_dBz_dz.assign(m_Bz.size(), 0.0);
    for (int ir = 0; ir < m_Nr; ++ir) {
        for (int iz = 0; iz < m_Nz; ++iz) {
            const int idx = ir * m_Nz + iz;
            amrex::Real dB;
            if (iz == 0) {
                dB = (m_Bz[ir*m_Nz + (iz+1)] - m_Bz[idx]) / m_dz;
            } else if (iz == m_Nz - 1) {
                dB = (m_Bz[idx] - m_Bz[ir*m_Nz + (iz-1)]) / m_dz;
            } else {
                dB = (m_Bz[ir*m_Nz + (iz+1)] - m_Bz[ir*m_Nz + (iz-1)])
                     / (amrex::Real(2.0) * m_dz);
            }
            m_dBz_dz[idx] = dB;
        }
    }

    amrex::Print() << "[Solenoid3D:" << this->name() << "] loaded "
                   << path << "  shape (Nr,Nz)=(" << m_Nr << "," << m_Nz << ")"
                   << "  (dr,dz)=(" << m_dr << "," << m_dz << ") m"
                   << "  (r0,z0)=(" << m_r0 << "," << m_z0 << ") m\n";
}

} // namespace elements
} // namespace lucretiatt
