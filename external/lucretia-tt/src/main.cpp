#include "elements/All.H"
#include "io/BeamIO.H"
#include "particles/TimeBunch.H"
#include "spacecharge/SpaceCharge.H"
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
#include <memory>
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
            std::string path;
            amrex::Real scale_E = 0.0;
            amrex::Real scale_B = 0.0;
            amrex::Real f_RF    = 0.0;
            amrex::Real phase   = 0.0;
            pp_el.query("z",       z_lab_edge);
            pp_el.get  ("path",    path);
            pp_el.query("scale_E", scale_E);
            pp_el.query("scale_B", scale_B);
            pp_el.query("f_RF",    f_RF);
            pp_el.query("phase",   phase);
            lattice.emplace_back(ImpactTField(
                name, z_lab_edge, path, scale_E, scale_B, f_RF, phase));
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
            pp_el.query("z_cathode",      z_cathode);
            pp_el.query("image_charge",   image_charge);
            pp_el.query("n_macroparticles_total", n_total);
            pp_el.query("total_charge",   total_charge);
            pp_el.query("pulse_shape",    pulse_shape_str);
            pp_el.query("pulse_duration", pulse_duration);
            pp_el.query("pulse_t0",       pulse_t0);
            pp_el.query("transverse_profile", xy_profile_str);
            pp_el.query("spot_size",      spot_size);
            pp_el.query("mte",            mte);

            CathodeSource::PulseShape pulse_shape =
                (pulse_shape_str == "flat_top")
                    ? CathodeSource::PulseShape::FlatTop
                    : CathodeSource::PulseShape::Gaussian;
            CathodeSource::TransverseProfile xy_profile =
                (xy_profile_str == "uniform_disk")
                    ? CathodeSource::TransverseProfile::UniformDisk
                    : CathodeSource::TransverseProfile::Gaussian;

            lattice.emplace_back(CathodeSource(
                name, z_cathode, image_charge != 0,
                n_total, total_charge,
                pulse_shape, pulse_duration, pulse_t0,
                xy_profile, spot_size, mte));
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

void seed_beam (lucretiatt::particles::TimeBunch& bunch)
{
    amrex::ParmParse pp_beam("beam");

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
        std::unique_ptr<lucretiatt::spacecharge::SpaceCharge> sc;
        {
            int sc_enabled = 0;
            amrex::ParmParse("space_charge").query("enabled", sc_enabled);
            if (sc_enabled != 0) {
                sc = std::make_unique<lucretiatt::spacecharge::SpaceCharge>(
                    geom, ba, dm);
                amrex::Print() << "Space charge: enabled\n";
            }
        }

        // ---- Time-stepping ----
        amrex::Real dt      = 1.0e-12;
        int         n_steps = 100;
        {
            amrex::ParmParse pp_track("tracking");
            pp_track.query("dt", dt);
            pp_track.query("n_steps", n_steps);
        }

        lucretiatt::tracking::TrackingLoop tracker;
        amrex::Real t = 0.0;

        // Step 0 = initial state (pre-push).
        maybe_dump(writer.get(), lattice, bunch, 0, t);

        for (int s = 1; s <= n_steps; ++s) {
            tracker.step(bunch, lattice, t, dt, sc.get());
            t += dt;
            maybe_dump(writer.get(), lattice, bunch, s, t);
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
