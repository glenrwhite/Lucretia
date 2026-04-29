#include "ImpactTField.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <cmath>
#include <fstream>
#include <stdexcept>


namespace lucretiatt {
namespace elements {

namespace {

/** Read the next double from a stream, skipping any whitespace.
 *  Throws on EOF / parse error. */
double read_one (std::ifstream& f, const std::string& path)
{
    double v;
    if (!(f >> v)) {
        throw std::runtime_error(
            "ImpactTField: unexpected EOF / parse error in '" + path + "'");
    }
    return v;
}

/** Read one ImpactT rfdata Fourier block: NF, zmin, zmax, zlen, then NF coefficients. */
void read_block (std::ifstream& f, const std::string& path,
                 int& nf, amrex::Real& zmin, amrex::Real& zmax,
                 amrex::Real& zlen, std::vector<amrex::Real>& coeffs)
{
    nf   = int(std::round(read_one(f, path)));
    zmin = amrex::Real(read_one(f, path));
    zmax = amrex::Real(read_one(f, path));
    zlen = amrex::Real(read_one(f, path));
    coeffs.resize(nf);
    for (int i = 0; i < nf; ++i) {
        coeffs[i] = amrex::Real(read_one(f, path));
    }
}

} // anonymous


ImpactTField::ImpactTField (
    std::string  name,
    amrex::Real  z_lab_edge,
    const std::string& path,
    amrex::Real  scale_E,
    amrex::Real  scale_B,
    amrex::Real  f_RF,
    amrex::Real  phase_rad)
    : Named(std::move(name))
    , Position3D(0.0, 0.0, z_lab_edge, amrex::Real(0.0))   // length set below
    , TimeWindow()
    , m_scale_E(scale_E)
    , m_scale_B(scale_B)
    , m_omega(amrex::Real(2.0 * 3.14159265358979323846) * f_RF)
    , m_phase(phase_rad)
{
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("ImpactTField: cannot open '" + path + "'");
    }

    read_block(f, path, m_E_nf, m_E_zmin, m_E_zmax, m_E_zlen, m_E_coeffs);
    read_block(f, path, m_B_nf, m_B_zmin, m_B_zmax, m_B_zlen, m_B_coeffs);

    // Position3D::m_length encompasses both blocks for diagnostic purposes;
    // the actual gating in gather_E_B is per-block on its own [zmin, zmax].
    const amrex::Real z_lo = std::min(m_E_zmin, m_B_zmin);
    const amrex::Real z_hi = std::max(m_E_zmax, m_B_zmax);
    m_length = z_hi - z_lo;

    amrex::Print() << "[ImpactTField:" << this->name() << "] loaded " << path
                   << "  E: NF=" << m_E_nf
                       << " z=[" << m_E_zmin << "," << m_E_zmax << "] m"
                   << "  B: NF=" << m_B_nf
                       << " z=[" << m_B_zmin << "," << m_B_zmax << "] m"
                   << "  scale_E=" << m_scale_E << " V/m"
                   << "  scale_B=" << m_scale_B << " T"
                   << "  f_RF="   << (m_omega / (2.0 * 3.14159265358979323846))
                   << " Hz"
                   << "  phase=" << m_phase << " rad"
                   << "\n";
}

} // namespace elements
} // namespace lucretiatt
