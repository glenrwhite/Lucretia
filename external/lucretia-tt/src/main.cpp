#include "elements/All.H"
#include "io/BeamIO.H"
#include "particles/TimeBunch.H"
#include "spacecharge/SpaceCharge.H"
#include "spacecharge/SpaceChargeSlice.H"
#include "tracking/TrackingLoop.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#if defined(AMREX_USE_MPI)
#   include <mpi.h>
#endif

#include <hdf5.h>

#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>


namespace {

constexpr const char* kVersion = "0.1.0";

void print_banner ()
{
    amrex::Print() << "================================================\n"
                   << "  lucretia-tt " << kVersion << " (Phase 1)\n"
                   << "  Time-based 3D PIC tracker for Lucretia\n"
                   << "================================================\n";
}


void make_geometry (amrex::Geometry& geom,
                    amrex::BoxArray& ba,
                    amrex::DistributionMapping& dm)
{
    amrex::ParmParse pp("geom");

    amrex::Vector<amrex::Real> lo, hi;
    amrex::Vector<int>         ncell;
    if (!pp.queryarr("lo",    lo))    { lo    = {-0.05, -0.05,  0.0}; }
    if (!pp.queryarr("hi",    hi))    { hi    = { 0.05,  0.05,  1.0}; }
    if (!pp.queryarr("ncell", ncell)) { ncell = {32, 32, 32}; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        lo.size() == 3 && hi.size() == 3 && ncell.size() == 3,
        "geom.lo / geom.hi / geom.ncell must each have 3 entries");

    amrex::IntVect iv_lo(0, 0, 0);
    amrex::IntVect iv_hi(ncell[0]-1, ncell[1]-1, ncell[2]-1);
    amrex::Box     domain(iv_lo, iv_hi);
    amrex::RealBox rb({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]});
    amrex::Array<int, AMREX_SPACEDIM> is_per{0, 0, 0};
    geom.define(domain, rb, amrex::CoordSys::cartesian, is_per);
    ba = amrex::BoxArray(domain);
    ba.maxSize(16);
    dm = amrex::DistributionMapping(ba);
}


/** Build the lattice from ParmParse `lattice.elements` + per-element keys.
 *  When `lattice.elements` is unset, returns a default smoke lattice
 *  (Drift3D + BeamMonitor every 10 steps). */
lucretiatt::elements::Lattice build_lattice ()
{
    using namespace lucretiatt::elements;
    Lattice lattice;

    amrex::ParmParse pp_lat("lattice");
    std::vector<std::string> names;
    pp_lat.queryarr("elements", names);
    if (names.empty()) {
        lattice.emplace_back(Drift3D("DRIFT_01", 0.0, 0.0, 0.0, 1.0));
        lattice.emplace_back(BeamMonitor("MON_01", 10));
        return lattice;
    }

    for (auto const& name : names) {
        amrex::ParmParse pp_el(("lattice." + name).c_str());
        std::string type;
        pp_el.get("type", type);

        if (type == "drift3d") {
            amrex::Real length = 1.0;
            pp_el.query("length", length);
            lattice.emplace_back(Drift3D(name, 0.0, 0.0, 0.0, length));
        }
        else if (type == "beam_monitor") {
            int dump_every = 0;
            pp_el.get("dump_every", dump_every);
            lattice.emplace_back(BeamMonitor(name, dump_every));
        }
        else if (type == "uniform_b") {
            amrex::Real Bx = 0.0, By = 0.0, Bz = 0.0;
            pp_el.query("Bx", Bx);
            pp_el.query("By", By);
            pp_el.query("Bz", Bz);
            lattice.emplace_back(UniformB(name, Bx, By, Bz));
        }
        else if (type == "sw_cavity") {
            amrex::Real z          = 0.0;
            amrex::Real length     = 0.0;
            int         n_cells    = 1;
            amrex::Real f_RF       = 2.856e9;
            amrex::Real peak_field = 0.0;
            amrex::Real phase      = 0.0;
            pp_el.query("z",          z);
            pp_el.get  ("length",     length);
            pp_el.query("n_cells",    n_cells);
            pp_el.query("f_RF",       f_RF);
            pp_el.get  ("peak_field", peak_field);
            pp_el.query("phase",      phase);
            lattice.emplace_back(SWCavity(
                name, z, length, n_cells, f_RF, peak_field, phase));
        }
        else if (type == "quad") {
            amrex::Real z        = 0.0;
            amrex::Real length   = 0.0;
            amrex::Real gradient = 0.0;
            pp_el.query("z",        z);
            pp_el.get  ("length",   length);
            pp_el.get  ("gradient", gradient);
            lattice.emplace_back(QuadAnalytic(name, z, length, gradient));
        }
        else if (type == "tw_cavity") {
            amrex::Real z          = 0.0;
            amrex::Real length     = 0.0;
            amrex::Real f_RF       = 2.856e9;
            amrex::Real peak_field = 0.0;
            amrex::Real phase      = 0.0;
            amrex::Real v_phase    = 299792458.0;
            pp_el.query("z",          z);
            pp_el.get  ("length",     length);
            pp_el.query("f_RF",       f_RF);
            pp_el.get  ("peak_field", peak_field);
            pp_el.query("phase",      phase);
            pp_el.query("v_phase",    v_phase);
            lattice.emplace_back(TWCavity(
                name, z, length, f_RF, peak_field, phase, v_phase));
        }
        else if (type == "solenoid_analytic") {
            amrex::Real z       = 0.0;
            amrex::Real length  = 0.0;
            amrex::Real peak_Bz = 0.0;
            amrex::Real sigma   = 0.0;
            amrex::Real z_peak  = 0.0;
            pp_el.query("z",        z);
            pp_el.get  ("length",   length);
            pp_el.get  ("peak_Bz",  peak_Bz);
            pp_el.get  ("sigma",    sigma);
            pp_el.query("z_peak",   z_peak);
            lattice.emplace_back(SolenoidAnalytic(
                name, z, length, peak_Bz, sigma, z_peak));
        }
        else if (type == "solenoid_3d") {
            amrex::Real z      = 0.0;
            amrex::Real length = 0.0;
            std::string path;
            pp_el.query("z",      z);
            pp_el.get  ("length", length);
            pp_el.get  ("path",   path);
            lattice.emplace_back(Solenoid3D(name, z, length, path));
        }
        else if (type == "field_map_3d") {
            amrex::Real z          = 0.0;
            amrex::Real length     = 0.0;
            std::string path;
            int         rf         = 0;
            amrex::Real f_RF       = 0.0;
            amrex::Real phase      = 0.0;
            pp_el.query("z",          z);
            pp_el.get  ("length",     length);
            pp_el.get  ("path",       path);
            pp_el.query("rf_modulated", rf);
            pp_el.query("f_RF",       f_RF);
            pp_el.query("phase",      phase);
            lattice.emplace_back(FieldMap3D(
                name, z, length, path, rf != 0, f_RF, phase));
        }
        else if (type == "impact_t_field") {
            amrex::Real z_lab_edge = 0.0;
            amrex::Real length     = 0.0;
            std::string path;
            amrex::Real scale_E = 0.0;
            amrex::Real scale_B = 0.0;
            amrex::Real f_RF    = 0.0;
            amrex::Real phase   = 0.0;
            pp_el.query("z",       z_lab_edge);
            pp_el.get  ("length",  length);          // ImpactT type-105 Blength
            pp_el.get  ("path",    path);
            pp_el.query("scale_E", scale_E);
            pp_el.query("scale_B", scale_B);
            pp_el.query("f_RF",    f_RF);
            pp_el.query("phase",   phase);
            lattice.emplace_back(ImpactTField(
                name, z_lab_edge, length, path, scale_E, scale_B, f_RF, phase));
        }
        else if (type == "cathode_source") {
            amrex::Real z_cathode = 0.0;
            int         image_charge = 1;
            int         n_total = 0;
            amrex::Real total_charge = 0.0;
            std::string pulse_shape_str = "gaussian";
            amrex::Real pulse_duration = 0.0;
            amrex::Real pulse_t0 = 0.0;
            std::string xy_profile_str = "gaussian";
            amrex::Real spot_size = 0.0;
            amrex::Real mte = 0.0;
            // SuperGaussian extras (ignored unless shape == super_gaussian)
            amrex::Real pulse_alpha         = 1.0;
            amrex::Real pulse_slope         = 0.0;
            amrex::Real transverse_alpha    = 1.0;
            amrex::Real transverse_truncate = 0.0;
            // ImpactT-style emission options
            int         longitudinal_thermal = 0;
            amrex::Real z_init_spread        = 0.0;
            pp_el.query("z_cathode",            z_cathode);
            pp_el.query("image_charge",         image_charge);
            pp_el.query("n_macroparticles_total", n_total);
            pp_el.query("total_charge",         total_charge);
            pp_el.query("pulse_shape",          pulse_shape_str);
            pp_el.query("pulse_duration",       pulse_duration);
            pp_el.query("pulse_t0",             pulse_t0);
            pp_el.query("transverse_profile",   xy_profile_str);
            pp_el.query("spot_size",            spot_size);
            pp_el.query("mte",                  mte);
            pp_el.query("pulse_alpha",          pulse_alpha);
            pp_el.query("pulse_slope",          pulse_slope);
            pp_el.query("transverse_alpha",     transverse_alpha);
            pp_el.query("transverse_truncate",  transverse_truncate);
            pp_el.query("longitudinal_thermal", longitudinal_thermal);
            pp_el.query("z_init_spread",        z_init_spread);

            CathodeSource::PulseShape pulse_shape;
            if      (pulse_shape_str == "flat_top")       pulse_shape = CathodeSource::PulseShape::FlatTop;
            else if (pulse_shape_str == "super_gaussian") pulse_shape = CathodeSource::PulseShape::SuperGaussian;
            else                                          pulse_shape = CathodeSource::PulseShape::Gaussian;

            CathodeSource::TransverseProfile xy_profile;
            if      (xy_profile_str == "uniform_disk")    xy_profile = CathodeSource::TransverseProfile::UniformDisk;
            else if (xy_profile_str == "super_gaussian")  xy_profile = CathodeSource::TransverseProfile::SuperGaussian;
            else                                          xy_profile = CathodeSource::TransverseProfile::Gaussian;

            lattice.emplace_back(CathodeSource(
                name, z_cathode, image_charge != 0,
                n_total, total_charge,
                pulse_shape, pulse_duration, pulse_t0,
                xy_profile, spot_size, mte,
                pulse_alpha, pulse_slope,
                transverse_alpha, transverse_truncate,
                longitudinal_thermal != 0, z_init_spread));
        }
        else if (type == "collimator") {
            amrex::Real z_start = 0.0;
            amrex::Real z_end   = 0.0;
            amrex::Real xmin    = -1e30, xmax = 1e30;
            amrex::Real ymin    = -1e30, ymax = 1e30;
            int         kill_bk = 0;
            pp_el.query("z_start",       z_start);
            pp_el.query("z_end",         z_end);
            pp_el.query("xmin",          xmin);
            pp_el.query("xmax",          xmax);
            pp_el.query("ymin",          ymin);
            pp_el.query("ymax",          ymax);
            pp_el.query("kill_backward", kill_bk);
            lattice.emplace_back(Collimator(name, z_start, z_end,
                                            xmin, xmax, ymin, ymax,
                                            kill_bk != 0));
        }
        else if (type == "wakefield") {
            amrex::Real z_start  = 0.0;
            amrex::Real z_end    = 0.0;
            amrex::Real iris_a   = 0.0;
            amrex::Real gap_g    = 0.0;
            amrex::Real period_L = 0.0;
            int         long_on  = 1;
            int         trans_on = 1;
            int         n_slices = 200;
            pp_el.query("z_start",      z_start);
            pp_el.query("z_end",        z_end);
            pp_el.query("iris_a",       iris_a);
            pp_el.query("gap_g",        gap_g);
            pp_el.query("period_L",     period_L);
            pp_el.query("longitudinal", long_on);
            pp_el.query("transverse",   trans_on);
            pp_el.query("n_slices",     n_slices);
            lattice.emplace_back(WakeField(
                name, z_start, z_end, iris_a, gap_g, period_L,
                long_on != 0, trans_on != 0, n_slices));
        }
        else {
            amrex::Abort("Unknown element type: '" + type
                         + "' for element '" + name + "'");
        }
    }
    return lattice;
}


void seed_from_h5 (lucretiatt::particles::TimeBunch& bunch,
                   const std::string& path);

void seed_from_partcl_data (lucretiatt::particles::TimeBunch& bunch,
                            const std::string& path,
                            amrex::Real        q_total,
                            amrex::Real        t_init);

void seed_beam (lucretiatt::particles::TimeBunch& bunch)
{
    amrex::ParmParse pp_beam("beam");

    // ImpactT-style ASCII partcl.data seed: bypasses CathodeSource
    // entirely, loading per-particle (x, y, z, p_dimensionless) at a
    // single t_birth = t_init. Used for cross-code calibration where
    // we need lucretia-tt to start from EXACTLY the same particles as
    // ImpactT (eliminating CathodeSource's emission model from the
    // source of disagreement).
    std::string partcl_file;
    if (pp_beam.query("partcl_file", partcl_file) && !partcl_file.empty()) {
        amrex::Real q_total = 0.0;
        amrex::Real t_init  = 0.0;
        pp_beam.get  ("partcl_q_total", q_total);   // total |Q| (C, positive)
        pp_beam.query("t_init",         t_init);    // ImpactT Tini (s)
        seed_from_partcl_data(bunch, partcl_file, q_total, t_init);
        return;
    }

    std::string seed_file;
    if (pp_beam.query("seed_file", seed_file) && !seed_file.empty()) {
        seed_from_h5(bunch, seed_file);
        return;
    }

    int n_particles = 16;
    pp_beam.query("n_particles", n_particles);

    amrex::Real x = 0.0, y = 0.0, z = 0.0;
    amrex::Real ux = 0.0, uy = 0.0, uz = 0.0;
    amrex::Real weight = 1.0;
    pp_beam.query("x",  x);
    pp_beam.query("y",  y);
    pp_beam.query("z",  z);
    pp_beam.query("ux", ux);
    pp_beam.query("uy", uy);
    pp_beam.query("uz", uz);
    pp_beam.query("weight", weight);

    bunch.AddParticles(n_particles, x, y, z, ux, uy, uz, weight);
}


/** Read a 1D dataset of doubles into a std::vector. Returns true if
 *  read; false if absent. */
bool h5_read_1d (hid_t file, const char* name, std::vector<double>& out)
{
    if (H5Lexists(file, name, H5P_DEFAULT) <= 0) { return false; }
    hid_t dset  = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hsize_t n = 0;
    int ndim = H5Sget_simple_extent_ndims(space);
    if (ndim == 1) {
        H5Sget_simple_extent_dims(space, &n, nullptr);
    } else {
        // accept (n,1) or (1,n)
        hsize_t dims[2] = {0, 0};
        H5Sget_simple_extent_dims(space, dims, nullptr);
        n = dims[0] * dims[1];
    }
    out.resize(n);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    return true;
}


/** Read /x /y /z /ux /uy /uz /w from an HDF5 file and seed the bunch. */
void seed_from_h5 (lucretiatt::particles::TimeBunch& bunch,
                   const std::string& path)
{
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error(
            "seed_from_h5: cannot open '" + path + "'");
    }

    std::vector<double> x, y, z, ux, uy, uz, w;
    const bool ok =
        h5_read_1d(file, "/x",  x)  &&
        h5_read_1d(file, "/y",  y)  &&
        h5_read_1d(file, "/z",  z)  &&
        h5_read_1d(file, "/ux", ux) &&
        h5_read_1d(file, "/uy", uy) &&
        h5_read_1d(file, "/uz", uz);
    h5_read_1d(file, "/w", w);   // optional; default 1.0
    H5Fclose(file);

    if (!ok) {
        throw std::runtime_error(
            "seed_from_h5: '" + path + "' missing one of /x /y /z /ux /uy /uz");
    }
    const size_t n = x.size();
    if (y.size() != n || z.size() != n ||
        ux.size() != n || uy.size() != n || uz.size() != n) {
        throw std::runtime_error("seed_from_h5: position/velocity arrays size mismatch");
    }

    // Cast doubles -> ParticleReal (no-op when AMReX uses double).
    std::vector<amrex::ParticleReal>
        xs(n), ys(n), zs(n), uxs(n), uys(n), uzs(n);
    for (size_t i = 0; i < n; ++i) {
        xs[i]  = amrex::ParticleReal(x[i]);
        ys[i]  = amrex::ParticleReal(y[i]);
        zs[i]  = amrex::ParticleReal(z[i]);
        uxs[i] = amrex::ParticleReal(ux[i]);
        uys[i] = amrex::ParticleReal(uy[i]);
        uzs[i] = amrex::ParticleReal(uz[i]);
    }

    if (w.empty()) {
        // Single weight broadcast
        bunch.AddParticlesFromArrays(int(n), xs.data(), ys.data(), zs.data(),
                                     uxs.data(), uys.data(), uzs.data(),
                                     amrex::ParticleReal(1.0), amrex::ParticleReal(0.0));
    } else if (w.size() == 1) {
        bunch.AddParticlesFromArrays(int(n), xs.data(), ys.data(), zs.data(),
                                     uxs.data(), uys.data(), uzs.data(),
                                     amrex::ParticleReal(w[0]), amrex::ParticleReal(0.0));
    } else if (w.size() == n) {
        // Per-particle weight: emit in groups by unique weight (or particle by
        // particle if weights vary). Phase 4A test only needs uniform weight.
        const double w0 = w[0];
        bool uniform = true;
        for (size_t i = 1; i < n; ++i) {
            if (w[i] != w0) { uniform = false; break; }
        }
        if (uniform) {
            bunch.AddParticlesFromArrays(int(n), xs.data(), ys.data(), zs.data(),
                                         uxs.data(), uys.data(), uzs.data(),
                                         amrex::ParticleReal(w0), amrex::ParticleReal(0.0));
        } else {
            for (size_t i = 0; i < n; ++i) {
                bunch.AddParticlesFromArrays(1,
                    &xs[i], &ys[i], &zs[i], &uxs[i], &uys[i], &uzs[i],
                    amrex::ParticleReal(w[i]), amrex::ParticleReal(0.0));
            }
        }
    } else {
        throw std::runtime_error("seed_from_h5: /w size must be 0, 1, or N");
    }

    amrex::Print() << "[seed_from_h5] loaded " << n << " particles from "
                   << path << "\n";
}


/** Read an ImpactT-style ASCII partcl.data file:
 *
 *      <N>
 *      x1 px1 y1 py1 z1 pz1
 *      x2 px2 y2 py2 z2 pz2
 *      ...
 *
 *  Lengths are in metres; momenta are dimensionless gamma*beta_axis
 *  (i.e. p_axis / (m_e * c)). Convert to lucretia-tt's WarpX-style
 *  proper velocity u = gamma*v_axis [m/s] via u = p_dimensionless * c.
 *
 *  All particles get the same t_birth = t_init (typically the ImpactT
 *  Tini header). They are loaded "alive" — no CathodeSource emission
 *  window logic. Tracking should be started at t = t_init for the
 *  emission timing to match ImpactT.
 *
 *  weight = q_total / (N * |e|) so that the macro charge q*w yields
 *  the correct total bunch charge (q is set to electron_charge in
 *  TimeBunch::AddParticlesFromArrays). Sign convention: q_total is
 *  the magnitude of the bunch charge in coulombs (positive). */
void seed_from_partcl_data (lucretiatt::particles::TimeBunch& bunch,
                            const std::string& path,
                            amrex::Real        q_total,
                            amrex::Real        t_init)
{
    using amrex::ParticleReal;
    using amrex::Real;

    // All ranks open + parse the file: AddParticlesFromArrays deposits
    // only on the IO rank but its trailing Redistribute() is collective,
    // so an early-return on non-IO ranks would deadlock the collective.
    // The file is small (~3 MB for 50k particles) and the parse is
    // single-process anyway, so the duplicate read cost is negligible
    // compared to the rest of the run.
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error(
            "seed_from_partcl_data: cannot open '" + path + "'");
    }

    // Header line: total particle count.
    int n = 0;
    {
        std::string line;
        if (!std::getline(f, line)) {
            throw std::runtime_error(
                "seed_from_partcl_data: '" + path + "' is empty");
        }
        std::istringstream iss(line);
        if (!(iss >> n) || n <= 0) {
            throw std::runtime_error(
                "seed_from_partcl_data: bad header in '" + path + "'");
        }
    }

    std::vector<ParticleReal> xs(n), ys(n), zs(n);
    std::vector<ParticleReal> uxs(n), uys(n), uzs(n);
    std::vector<ParticleReal> t_births(n, ParticleReal(t_init));

    // partcl.data dimensionless momenta are p_axis/(m_e c). lucretia-tt
    // uses u_axis = gamma * v_axis [m/s], which equals p_dimensionless * c.
    constexpr Real c = Real(2.99792458e8);

    for (int i = 0; i < n; ++i) {
        std::string line;
        while (std::getline(f, line)) {
            // Skip blank lines (some writers emit a trailing newline)
            bool blank = true;
            for (char ch : line) {
                if (!std::isspace(static_cast<unsigned char>(ch))) { blank = false; break; }
            }
            if (!blank) { break; }
        }
        std::istringstream iss(line);
        double x_m, px_dim, y_m, py_dim, z_m, pz_dim;
        if (!(iss >> x_m >> px_dim >> y_m >> py_dim >> z_m >> pz_dim)) {
            throw std::runtime_error(
                "seed_from_partcl_data: short read at particle "
                + std::to_string(i + 1) + " of " + std::to_string(n)
                + " in '" + path + "'");
        }
        xs[i]  = ParticleReal(x_m);
        ys[i]  = ParticleReal(y_m);
        zs[i]  = ParticleReal(z_m);
        uxs[i] = ParticleReal(px_dim * c);
        uys[i] = ParticleReal(py_dim * c);
        uzs[i] = ParticleReal(pz_dim * c);
    }

    constexpr Real e_charge_mag = Real(1.602176634e-19);
    const Real weight = q_total / (Real(n) * e_charge_mag);

    bunch.AddParticlesFromArrays(
        n,
        xs.data(),  ys.data(),  zs.data(),
        uxs.data(), uys.data(), uzs.data(),
        ParticleReal(weight),
        t_births.data());

    // Quick provenance summary so the run log makes the seed source
    // unambiguous (these runs are normally cross-code calibration
    // experiments and the seed identity matters).
    Real mean_x = 0, mean_y = 0, mean_z = 0;
    Real mean_uz = 0;
    for (int i = 0; i < n; ++i) {
        mean_x  += xs[i];   mean_y  += ys[i];   mean_z  += zs[i];
        mean_uz += uzs[i];
    }
    mean_x /= n; mean_y /= n; mean_z /= n; mean_uz /= n;
    amrex::Print() << "[seed_from_partcl_data] loaded " << n
                   << " particles from " << path << "\n"
                   << "    Q_total = " << q_total << " C"
                   << "  weight = " << weight
                   << "  t_init = " << t_init << " s\n"
                   << "    <x> = " << mean_x  << " m"
                   << "  <y> = "   << mean_y  << " m"
                   << "  <z> = "   << mean_z  << " m"
                   << "  <uz> = "  << mean_uz << " m/s\n";
}


/** Run BeamMonitor::should_dump checks across the lattice and invoke
 *  the writer for each that fires this step. */
void maybe_dump (
    lucretiatt::io::OpenPMDWriter*           writer,
    const lucretiatt::elements::Lattice&     lattice,
    lucretiatt::particles::TimeBunch&        bunch,
    int                                      step,
    amrex::Real                              t)
{
    if (!writer) { return; }
    for (auto const& el : lattice) {
        std::visit([&] (auto const& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, lucretiatt::elements::BeamMonitor>) {
                if (e.should_dump(step)) {
                    writer->write(bunch, step, t);
                }
            }
        }, el);
    }
}

} // namespace


int main (int argc, char* argv[])
{
#if defined(AMREX_USE_MPI)
    AMREX_ALWAYS_ASSERT(MPI_SUCCESS == MPI_Init(&argc, &argv));
#endif

    amrex::Initialize(argc, argv);
    {
        BL_PROFILE_VAR("main()", pmain);
        print_banner();

        // ---- Geometry ----
        amrex::Geometry            geom;
        amrex::BoxArray            ba;
        amrex::DistributionMapping dm;
        make_geometry(geom, ba, dm);

        // ---- Beam ----
        lucretiatt::particles::TimeBunch bunch(geom, dm, ba);
        seed_beam(bunch);
        bunch.PrintSummary("seeded");

        // ---- Lattice ----
        auto lattice = build_lattice();
        amrex::Print() << "Lattice (" << lattice.size() << " elements):\n";
        int el_idx = 0;
        for (auto const& el : lattice) {
            std::visit([&] (auto const& e) {
                amrex::Print() << "  [" << el_idx << "] " << e.name() << "\n";
            }, el);
            ++el_idx;
        }

        // ---- Output writer ----
        std::unique_ptr<lucretiatt::io::OpenPMDWriter> writer;
        const bool any_dump = std::any_of(lattice.begin(), lattice.end(),
            [] (auto const& el) {
                return std::visit([] (auto const& e) {
                    using T = std::decay_t<decltype(e)>;
                    if constexpr (std::is_same_v<T, lucretiatt::elements::BeamMonitor>) {
                        return e.m_dump_every > 0;
                    } else { return false; }
                }, el);
            });
        std::string out_dir = "diags";
        amrex::ParmParse().query("out_dir", out_dir);
        if (any_dump) {
            amrex::UtilCreateDirectoryDestructive(out_dir, /*clean*/ false);
            const std::string path_pattern = out_dir + "/openpmd_%07T.h5";
            writer = std::make_unique<lucretiatt::io::OpenPMDWriter>(path_pattern);
            amrex::Print() << "openPMD output: " << path_pattern << "\n";
        }

        // ---- Space charge (optional) ----
        std::unique_ptr<lucretiatt::spacecharge::SpaceCharge>      sc;
        std::unique_ptr<lucretiatt::spacecharge::SpaceChargeSlice> slice_sc;
        {
            int sc_enabled        = 0;
            int sc_comoving       = 0;
            int sc_adaptive       = 0;
            amrex::Real sc_pad_factor = 5.0;
            amrex::Real sc_min_pad_xy = 1.0e-3;
            amrex::Real sc_min_pad_z  = 1.0e-3;
            int sc_image_enabled  = 0;
            amrex::Real sc_image_z_cath = 0.0;
            amrex::Real sc_image_cutoff = 0.05;
            int slice_sc_enabled  = 0;
            int slice_n           = 256;
            amrex::Real slice_radius_factor = 2.0;
            amrex::Real slice_gamma_off     = std::numeric_limits<amrex::Real>::infinity();
            amrex::ParmParse pp_sc("space_charge");
            pp_sc.query("enabled",  sc_enabled);
            pp_sc.query("comoving", sc_comoving);
            pp_sc.query("adaptive", sc_adaptive);
            int sc_exact_range = 0;
            pp_sc.query("exact_range", sc_exact_range);
            pp_sc.query("pad_factor", sc_pad_factor);
            pp_sc.query("min_pad_xy", sc_min_pad_xy);
            pp_sc.query("min_pad_z",  sc_min_pad_z);
            amrex::Real sc_resize_hyst = 2.0;
            pp_sc.query("resize_hyst", sc_resize_hyst);
            pp_sc.query("image_plane_enabled", sc_image_enabled);
            pp_sc.query("image_plane_z",       sc_image_z_cath);
            pp_sc.query("image_cutoff",        sc_image_cutoff);
            pp_sc.query("slice_enabled",       slice_sc_enabled);
            pp_sc.query("slice_n",             slice_n);
            pp_sc.query("slice_radius_factor", slice_radius_factor);
            pp_sc.query("slice_gamma_off",     slice_gamma_off);
            int slice_profile = 0;
            pp_sc.query("slice_profile",       slice_profile);
            int sc_rho_smooth_passes = 0;
            pp_sc.query("rho_smooth_passes",   sc_rho_smooth_passes);
            int sc_hybrid_z = 0;
            pp_sc.query("hybrid_z_adaptive",   sc_hybrid_z);
            int sc_shape_order = 1;
            pp_sc.query("shape_order",         sc_shape_order);
            if (sc_enabled != 0) {
                sc = std::make_unique<lucretiatt::spacecharge::SpaceCharge>(
                    geom, ba, dm);
                sc->set_comoving(sc_comoving != 0);
                if (sc_adaptive != 0) {
                    sc->set_adaptive(true, sc_pad_factor, sc_min_pad_xy, sc_min_pad_z);
                }
                sc->set_resize_hyst(sc_resize_hyst);
                if (sc_hybrid_z != 0) {
                    sc->set_hybrid_z_adaptive(true, sc_pad_factor, sc_min_pad_z);
                }
                if (sc_exact_range != 0) {
                    sc->set_exact_range(true);
                }
                if (sc_image_enabled != 0) {
                    sc->set_image_plane(sc_image_z_cath, sc_image_cutoff);
                }
                if (sc_rho_smooth_passes > 0) {
                    sc->set_rho_smooth_passes(sc_rho_smooth_passes);
                }
                if (sc_shape_order != 1) {
                    sc->set_shape_order(sc_shape_order);
                }
                amrex::Print() << "Space charge: enabled"
                               << (sc_exact_range ? "  (ImpactT-exact-range mesh)"
                                                  : (sc_hybrid_z   ? "  (hybrid: static-xy + adaptive-z)"
                                                  : (sc_adaptive   ? "  (adaptive padded mesh)"
                                                  : (sc_comoving   ? "  (co-moving mesh)"
                                                                   : "  (static mesh)"))))
                               << (sc_image_enabled ? "  (cathode image charges)" : "");
                if (sc_rho_smooth_passes > 0) {
                    amrex::Print() << "  (rho-smooth " << sc_rho_smooth_passes << " pass)";
                }
                if (sc_shape_order != 1) {
                    amrex::Print() << "  (shape order=" << sc_shape_order
                                   << " " << (sc_shape_order == 2 ? "TSC" : "?") << ")";
                }
                amrex::Print() << "\n";
            }
            if (slice_sc_enabled != 0) {
                slice_sc = std::make_unique<lucretiatt::spacecharge::SpaceChargeSlice>(
                    slice_n, slice_radius_factor);
                slice_sc->set_gamma_off(slice_gamma_off);
                slice_sc->set_profile(slice_profile);
                amrex::Print() << "1D longitudinal slice SC: enabled  (n_slices="
                               << slice_n
                               << ",  profile=" << (slice_profile == 1 ? "Gaussian-disk" : "uniform-disk")
                               << ",  bunch_radius_factor=" << slice_radius_factor
                               << " * sigma_xy"
                               << (std::isfinite(slice_gamma_off)
                                   ? std::string(",  off above gamma=") +
                                     std::to_string(slice_gamma_off)
                                   : std::string(""))
                               << ")\n";
            }
        }

        // ---- Time-stepping (with optional change-of-dt schedule) ----
        // tracking.dt           : initial dt (s)
        // tracking.n_steps      : total number of steps to take
        // tracking.dt_change_t  : if set, switch dt at this simulation time
        // tracking.dt_after     : dt to use after dt_change_t (default = dt)
        // tracking.t_start      : initial t (default 0). Useful when matching
        //                         a code that uses a non-zero initial time
        //                         (e.g. ImpactT's Tini header).
        // tracking.behind_cathode_z, tracking.behind_cathode_betazini :
        //                         when both set (and betazini > 0), enables
        //                         ImpactT-style universal-betazini drift
        //                         for particles with z < behind_cathode_z.
        amrex::Real dt           = 1.0e-12;
        int         n_steps      = 100;
        amrex::Real dt_change_t  = std::numeric_limits<amrex::Real>::infinity();
        amrex::Real dt_change_z  = std::numeric_limits<amrex::Real>::infinity();
        amrex::Real dt_after     = -1.0;
        amrex::Real t_start      = 0.0;
        amrex::Real behind_cathode_z        = std::numeric_limits<amrex::Real>::lowest();
        amrex::Real behind_cathode_betazini = 0.0;
        int         use_centroid_phase      = 0;
        amrex::Real centroid_t_offset       = 0.0;
        int         use_midstep_field       = 0;
        int         use_particle_phase      = 0;
        int         use_dkd_integrator      = 0;
        int         disable_self_force      = 0;
        int         self_force_direct       = 0;
        int         sc_use_b_field          = 0;
        std::string trace_file;
        int         trace_every             = 1;
        {
            amrex::ParmParse pp_track("tracking");
            pp_track.query("dt", dt);
            pp_track.query("n_steps", n_steps);
            pp_track.query("dt_change_t", dt_change_t);
            pp_track.query("dt_change_z", dt_change_z);
            pp_track.query("dt_after",    dt_after);
            pp_track.query("t_start",     t_start);
            pp_track.query("behind_cathode_z",        behind_cathode_z);
            pp_track.query("behind_cathode_betazini", behind_cathode_betazini);
            pp_track.query("use_centroid_phase",      use_centroid_phase);
            pp_track.query("centroid_t_offset",       centroid_t_offset);
            pp_track.query("use_midstep_field",       use_midstep_field);
            pp_track.query("use_particle_phase",      use_particle_phase);
            pp_track.query("use_dkd_integrator",      use_dkd_integrator);
            pp_track.query("disable_self_force",      disable_self_force);
            pp_track.query("self_force_direct",       self_force_direct);
            pp_track.query("sc_use_b_field",          sc_use_b_field);
            pp_track.query("trace_file",              trace_file);
            pp_track.query("trace_every",             trace_every);
        }
        if (dt_after <= 0.0) { dt_after = dt; }

        lucretiatt::tracking::TrackingLoop tracker;
        if (behind_cathode_betazini > 0.0
            && behind_cathode_z > std::numeric_limits<amrex::Real>::lowest()) {
            tracker.set_behind_cathode_drift(behind_cathode_z, behind_cathode_betazini);
            amrex::Print() << "[tracking] behind-cathode drift enabled: z<"
                           << behind_cathode_z << " m, betazini="
                           << behind_cathode_betazini
                           << " (v=" << (behind_cathode_betazini * 299792458.0)
                           << " m/s)\n";
            // Also auto-enable the SC z-filter so behind-cathode drifting
            // particles don't extend the SC mesh below the cathode plane
            // (matches ImpactT's flagpos=1 mode during emission).
            if (sc) {
                sc->set_z_filter_min(true, behind_cathode_z);
                amrex::Print() << "[tracking] SC z-filter enabled at z="
                               << behind_cathode_z << " m (skip particles below cathode)\n";
            }
        }
        if (use_centroid_phase != 0) {
            tracker.set_centroid_field_phase(true, centroid_t_offset);
            amrex::Print() << "[tracking] centroid-z field phase enabled "
                           << "(t_eff = z_centroid/c + offset; offset = "
                           << centroid_t_offset << " s)\n";
        }
        if (use_midstep_field != 0) {
            tracker.set_midstep_field(true);
            amrex::Print() << "[tracking] midstep-position field gather enabled\n";
        }
        if (use_particle_phase != 0) {
            tracker.set_particle_phase(true);
            amrex::Print() << "[tracking] per-particle phase enabled "
                           << "(t_eff = z_particle/c per particle)\n";
        }
        if (use_dkd_integrator != 0) {
            tracker.set_dkd_integrator(true);
            amrex::Print() << "[tracking] ImpactT-style drift-kick-drift "
                           << "integrator + first-order emission enabled\n";
        }
        if (disable_self_force != 0) {
            tracker.set_self_force_disabled(true);
            amrex::Print() << "[tracking] SC self-force LUT subtraction DISABLED "
                           << "(diagnostic — kSelfForceFactor effectively 0)\n";
        }
        if (sc_use_b_field != 0) {
            tracker.set_sc_use_b_field(true);
            amrex::Print() << "[tracking] SC: explicit B field via Boris pusher "
                           << "(matches ImpactT; replaces 1/gamma^2 boost shortcut)\n";
        }
        if (self_force_direct != 0) {
            tracker.set_self_force_direct(true);
            amrex::Print() << "[tracking] SC self-force computed DIRECTLY each step "
                           << "(no LUT; ~7x cost; eliminates LUT-rebuild noise)\n";
        }
        amrex::Real t = t_start;
        bool        switched = false;

        // Per-step trace file for cross-code calibration. Writes one
        // CSV row per (trace_every) step containing bunch-mean diagnostics.
        std::ofstream trace_out;
        if (!trace_file.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            trace_out.open(trace_file);
            if (!trace_out) {
                amrex::Print() << "[trace] WARNING: cannot open " << trace_file << "\n";
            } else {
                trace_out << "step,t,n_alive,z_mean,sigma_z,gamma_mean,sigma_gamma,"
                             "x_mean,sigma_x,vz_mean,t_centroid_eff\n";
                amrex::Print() << "[trace] writing per-step bunch stats to "
                               << trace_file << " (every " << trace_every
                               << " step)\n";
            }
        }
        // Lambda: compute and write trace row for current step.
        constexpr amrex::Real kTraceC  = 299792458.0;
        constexpr amrex::Real kTraceMe = 9.1093837015e-31;
        auto write_trace_row = [&] (int step, amrex::Real cur_t) {
            if (!trace_out.is_open()) return;
            using namespace lucretiatt::particles;
            using PIter = amrex::ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
            constexpr int lev = 0;
            amrex::Real sum_x = 0, sum_x2 = 0, sum_z = 0, sum_z2 = 0;
            amrex::Real sum_g = 0, sum_g2 = 0, sum_vz = 0;
            long n_alive = 0;
            constexpr amrex::Real inv_c2 = amrex::Real(1.0) / (kTraceC * kTraceC);
            for (PIter pti(bunch, lev); pti.isValid(); ++pti) {
                auto& soa = pti.GetStructOfArrays();
                const int np = pti.numParticles();
                const auto& xs  = soa.GetRealData(RealSoA::x);
                const auto& zs  = soa.GetRealData(RealSoA::z);
                const auto& uxs = soa.GetRealData(RealSoA::px);
                const auto& uys = soa.GetRealData(RealSoA::py);
                const auto& uzs = soa.GetRealData(RealSoA::pz);
                const auto& alives = soa.GetIntData(IntSoA::alive);
                for (int i = 0; i < np; ++i) {
                    if (alives[i] == 0) { continue; }
                    const amrex::Real ux = uxs[i], uy = uys[i], uz = uzs[i];
                    const amrex::Real g = std::sqrt(amrex::Real(1.0)
                        + (ux*ux + uy*uy + uz*uz) * inv_c2);
                    sum_x  += xs[i];   sum_x2 += xs[i]*xs[i];
                    sum_z  += zs[i];   sum_z2 += zs[i]*zs[i];
                    sum_g  += g;       sum_g2 += g*g;
                    sum_vz += uz / g;
                    ++n_alive;
                }
            }
            amrex::ParallelDescriptor::ReduceLongSum(n_alive);
            amrex::ParallelDescriptor::ReduceRealSum(sum_x);
            amrex::ParallelDescriptor::ReduceRealSum(sum_x2);
            amrex::ParallelDescriptor::ReduceRealSum(sum_z);
            amrex::ParallelDescriptor::ReduceRealSum(sum_z2);
            amrex::ParallelDescriptor::ReduceRealSum(sum_g);
            amrex::ParallelDescriptor::ReduceRealSum(sum_g2);
            amrex::ParallelDescriptor::ReduceRealSum(sum_vz);
            if (n_alive == 0) return;
            const amrex::Real inv_N = amrex::Real(1.0) / amrex::Real(n_alive);
            const amrex::Real x_mean = sum_x * inv_N;
            const amrex::Real z_mean = sum_z * inv_N;
            const amrex::Real g_mean = sum_g * inv_N;
            const amrex::Real vz_mean = sum_vz * inv_N;
            const amrex::Real var_x = std::max(sum_x2*inv_N - x_mean*x_mean, amrex::Real(0.0));
            const amrex::Real var_z = std::max(sum_z2*inv_N - z_mean*z_mean, amrex::Real(0.0));
            const amrex::Real var_g = std::max(sum_g2*inv_N - g_mean*g_mean, amrex::Real(0.0));
            const amrex::Real sx = std::sqrt(var_x);
            const amrex::Real sz = std::sqrt(var_z);
            const amrex::Real sg = std::sqrt(var_g);
            // Effective field-gather time used by centroid-phase mode.
            const amrex::Real t_centroid_eff = (use_centroid_phase != 0)
                ? (z_mean / kTraceC + centroid_t_offset)
                : cur_t;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                trace_out << step << ',' << cur_t << ',' << n_alive << ','
                          << z_mean << ',' << sz << ','
                          << g_mean << ',' << sg << ','
                          << x_mean << ',' << sx << ','
                          << vz_mean << ',' << t_centroid_eff << '\n';
                trace_out.flush();
            }
        };

        // Step 0 = initial state (pre-push).
        maybe_dump(writer.get(), lattice, bunch, 0, t);
        write_trace_row(0, t);

        // Progress reports every 10% of steps (or every step if n_steps < 10).
        // Includes wall time and ETA so long Injector / production runs are
        // monitorable without enabling per-step BeamMonitor dumps.
        const int    prog_every    = std::max(1, n_steps / 10);
        const double t_wall_start  = amrex::second();
        amrex::Print() << "[tracking] starting " << n_steps
                       << " steps; reporting every " << prog_every
                       << " step(s)\n";

        for (int s = 1; s <= n_steps; ++s) {
            // Switch dt either at a wall-clock time (dt_change_t) OR when the
            // bunch z-centroid passes a lab z position (dt_change_z). The
            // z-trigger mirrors ImpactT's `change_timestep_1` element, which
            // fires when the bunch reaches a specified z (not at a fixed t).
            // Either trigger latches `switched` permanently.
            if (!switched && dt_change_z < std::numeric_limits<amrex::Real>::infinity()) {
                // Compute bunch-mean z over alive particles (single-rank
                // sweep; cheap because we only do it until the latch fires).
                using namespace amrex;
                using namespace lucretiatt::particles;
                using PIter = ParIterSoA<RealSoA::nattribs, IntSoA::nattribs>;
                Real sum_z = 0.0;  long n_alive = 0;
                for (PIter pti(bunch, 0); pti.isValid(); ++pti) {
                    auto& soa = pti.GetStructOfArrays();
                    const int np = pti.numParticles();
                    const auto& zs     = soa.GetRealData(RealSoA::z);
                    const auto& alives = soa.GetIntData(IntSoA::alive);
                    for (int i = 0; i < np; ++i) {
                        if (alives[i] == 0) continue;
                        sum_z += zs[i];
                        ++n_alive;
                    }
                }
                ParallelDescriptor::ReduceRealSum(sum_z);
                ParallelDescriptor::ReduceLongSum(n_alive);
                if (n_alive > 0 && (sum_z / Real(n_alive)) >= dt_change_z) {
                    amrex::Print() << "[tracking] switched dt: "
                                   << dt << " -> " << dt_after
                                   << " s at z_centroid = "
                                   << (sum_z / Real(n_alive)) << " m"
                                   << " (target z = " << dt_change_z << " m, t = "
                                   << t << " s, step " << s << ")\n";
                    switched = true;
                }
            }
            if (!switched && t >= dt_change_t) {
                amrex::Print() << "[tracking] switched dt: "
                               << dt << " -> " << dt_after
                               << " s at t = " << t << " s (step " << s << ")\n";
                switched = true;
            }
            const amrex::Real cur_dt = switched ? dt_after : dt;
            tracker.step(bunch, lattice, t, cur_dt, sc.get(), slice_sc.get());
            t += cur_dt;
            maybe_dump(writer.get(), lattice, bunch, s, t);
            if (s % trace_every == 0) {
                write_trace_row(s, t);
            }

            if (s % prog_every == 0 || s == n_steps) {
                const long n_now      = bunch.TotalNumberOfParticles(true, false);
                const int  pct        = int(std::lround(100.0 * double(s) / double(n_steps)));
                const double t_wall   = amrex::second() - t_wall_start;
                const double t_per    = t_wall / double(s);
                const double eta_s    = t_per * double(n_steps - s);
                amrex::Print() << "[tracking] " << pct << "%  step " << s
                               << "/" << n_steps
                               << "  t = " << t << " s"
                               << "  N = " << n_now
                               << "  wall = " << t_wall << " s"
                               << "  ETA = " << eta_s << " s\n";
            }
        }

        amrex::Print() << "Done after " << n_steps
                       << " steps; t_final = " << t << " s\n";

        BL_PROFILE_VAR_STOP(pmain);
    }
    amrex::Finalize();

#if defined(AMREX_USE_MPI)
    AMREX_ALWAYS_ASSERT(MPI_SUCCESS == MPI_Finalize());
#endif

    return 0;
}
