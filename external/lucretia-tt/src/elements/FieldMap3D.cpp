#include "FieldMap3D.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <hdf5.h>

#include <stdexcept>


namespace lucretiatt {
namespace elements {

namespace {

/** Read a 1D dataset of fixed-size doubles into a std::vector. Returns
 *  true if the dataset exists and was read; false if absent. */
bool read_dataset_1d (hid_t file, const char* name, std::vector<double>& out)
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

/** Read a 3D dataset shaped (Nz, Ny, Nx) (C order) into a vector of doubles.
 *  Returns dims[3] = {Nz, Ny, Nx}. */
bool read_dataset_3d (hid_t file, const char* name,
                      std::vector<double>& out, int dims_out[3])
{
    if (H5Lexists(file, name, H5P_DEFAULT) <= 0) { return false; }
    hid_t dset  = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    int ndim = H5Sget_simple_extent_ndims(space);
    if (ndim != 3) {
        H5Sclose(space); H5Dclose(dset);
        throw std::runtime_error(
            std::string("FieldMap3D: dataset '") + name +
            "' has " + std::to_string(ndim) + " dims, expected 3");
    }
    hsize_t dims[3] = {0, 0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    dims_out[0] = int(dims[0]);
    dims_out[1] = int(dims[1]);
    dims_out[2] = int(dims[2]);
    out.resize(dims[0] * dims[1] * dims[2]);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    return true;
}

/** Convert a vector<double> in-place to vector<amrex::Real> (no-op if double). */
template <class T>
std::vector<T> to_real (const std::vector<double>& src)
{
    std::vector<T> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) { out[i] = T(src[i]); }
    return out;
}

} // anonymous


FieldMap3D::FieldMap3D (
    std::string name,
    amrex::Real z0_element,
    amrex::Real length,
    const std::string& path,
    bool        rf_modulated,
    amrex::Real f_RF,
    amrex::Real phase)
    : Named(std::move(name))
    , Position3D(0.0, 0.0, z0_element, length)
    , TimeWindow()
    , m_rf_modulated(rf_modulated)
    , m_f_RF(f_RF)
    , m_phase(phase)
{
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error(
            "FieldMap3D: cannot open HDF5 file '" + path + "'");
    }

    // Grid metadata
    std::vector<double> spacing, origin;
    if (!read_dataset_1d(file, "/grid_spacing", spacing) || spacing.size() != 3) {
        H5Fclose(file);
        throw std::runtime_error("FieldMap3D: /grid_spacing must be size-3 in '" + path + "'");
    }
    if (!read_dataset_1d(file, "/grid_origin", origin) || origin.size() != 3) {
        H5Fclose(file);
        throw std::runtime_error("FieldMap3D: /grid_origin must be size-3 in '" + path + "'");
    }
    m_dx = amrex::Real(spacing[0]);
    m_dy = amrex::Real(spacing[1]);
    m_dz = amrex::Real(spacing[2]);
    m_x0 = amrex::Real(origin[0]);
    m_y0 = amrex::Real(origin[1]);
    m_z0 = amrex::Real(origin[2]);

    // Field components
    int dims_E[3] = {0, 0, 0};
    int dims_B[3] = {0, 0, 0};
    std::vector<double> tmp;

    // HDF5 dim 0 = x (slowest in C order), dim 2 = z (fastest).
    // Matches what MATLAB's h5write produces for a [Nx, Ny, Nz] array.
    auto set_dims_or_check = [&] (const int got[3], int& Nx, int& Ny, int& Nz) {
        if (Nx == 0 && Ny == 0 && Nz == 0) {
            Nx = got[0]; Ny = got[1]; Nz = got[2];
        } else if (got[0] != Nx || got[1] != Ny || got[2] != Nz) {
            H5Fclose(file);
            throw std::runtime_error("FieldMap3D: component shape mismatch in '" + path + "'");
        }
    };

    int Nx = 0, Ny = 0, Nz = 0;

    if (read_dataset_3d(file, "/Ex", tmp, dims_E)) {
        set_dims_or_check(dims_E, Nx, Ny, Nz);
        m_Ex = to_real<amrex::Real>(tmp);
        m_has_E = true;
    }
    if (read_dataset_3d(file, "/Ey", tmp, dims_E)) {
        set_dims_or_check(dims_E, Nx, Ny, Nz);
        m_Ey = to_real<amrex::Real>(tmp);
        m_has_E = true;
    }
    if (read_dataset_3d(file, "/Ez", tmp, dims_E)) {
        set_dims_or_check(dims_E, Nx, Ny, Nz);
        m_Ez = to_real<amrex::Real>(tmp);
        m_has_E = true;
    }
    if (read_dataset_3d(file, "/Bx", tmp, dims_B)) {
        set_dims_or_check(dims_B, Nx, Ny, Nz);
        m_Bx = to_real<amrex::Real>(tmp);
        m_has_B = true;
    }
    if (read_dataset_3d(file, "/By", tmp, dims_B)) {
        set_dims_or_check(dims_B, Nx, Ny, Nz);
        m_By = to_real<amrex::Real>(tmp);
        m_has_B = true;
    }
    if (read_dataset_3d(file, "/Bz", tmp, dims_B)) {
        set_dims_or_check(dims_B, Nx, Ny, Nz);
        m_Bz = to_real<amrex::Real>(tmp);
        m_has_B = true;
    }

    H5Fclose(file);

    if (!m_has_E && !m_has_B) {
        throw std::runtime_error(
            "FieldMap3D: no E or B components found in '" + path + "'");
    }

    m_Nx = Nx; m_Ny = Ny; m_Nz = Nz;

    // Fill missing components with zero so the kernel can sample them
    // unconditionally.
    const size_t total = size_t(m_Nx) * m_Ny * m_Nz;
    if (m_has_E) {
        if (m_Ex.empty()) { m_Ex.assign(total, 0.0); }
        if (m_Ey.empty()) { m_Ey.assign(total, 0.0); }
        if (m_Ez.empty()) { m_Ez.assign(total, 0.0); }
    }
    if (m_has_B) {
        if (m_Bx.empty()) { m_Bx.assign(total, 0.0); }
        if (m_By.empty()) { m_By.assign(total, 0.0); }
        if (m_Bz.empty()) { m_Bz.assign(total, 0.0); }
    }

    amrex::Print() << "[FieldMap3D:" << this->name() << "] loaded "
                   << path << "  shape=(" << m_Nx << "," << m_Ny << "," << m_Nz << ")"
                   << "  d=(" << m_dx << "," << m_dy << "," << m_dz << ") m"
                   << "  origin=(" << m_x0 << "," << m_y0 << "," << m_z0 << ") m"
                   << "  E=" << (m_has_E?"yes":"no")
                   << "  B=" << (m_has_B?"yes":"no")
                   << "  rf=" << (m_rf_modulated?"yes":"no") << "\n";
}

} // namespace elements
} // namespace lucretiatt
