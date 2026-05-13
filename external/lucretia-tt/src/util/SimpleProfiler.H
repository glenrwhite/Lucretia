#pragma once

// SimpleProfiler: scope-based wall-clock accumulator for ad-hoc
// runtime hotspot profiling. Not thread-safe; use only from the
// orchestrating thread (e.g., main + the per-step kick driver).
//
// Usage:
//     {
//         lucretiatt::util::ScopeTimer t("phase_name");
//         // ... work ...
//     }   // Timer adds elapsed time to "phase_name" total on destruction.
//
// Print all accumulated phase totals at the end:
//     lucretiatt::util::SimpleProfiler::instance().print();
//
// Designed for one-off profiling runs; remove the scope timers when
// no longer needed (or wrap them in a compile-time switch).

#include <AMReX_Print.H>

#include <chrono>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace lucretiatt {
namespace util {

class SimpleProfiler {
public:
    static SimpleProfiler& instance ()
    {
        static SimpleProfiler s_instance;
        return s_instance;
    }

    void add (const std::string& name, double seconds)
    {
        m_totals[name] += seconds;
        m_counts[name] += 1;
    }

    void print () const
    {
        // Build the report into a single string then emit via amrex::Print()
        // so it's interleaved correctly with AMReX's other output (and
        // routed through the same stdout that the AMReX profiler uses).
        double grand = 0.0;
        for (auto const& kv : m_totals) { grand += kv.second; }
        std::ostringstream ss;
        ss << "\n[SimpleProfiler] phase wall-time summary:\n";
        ss << std::left << std::setw(28) << "phase"
           << std::right << std::setw(12) << "total[s]"
           << std::setw(10) << "count"
           << std::setw(14) << "per-call[ms]"
           << std::setw(10) << "% total" << "\n";
        ss << std::string(74, '-') << "\n";
        std::vector<std::pair<std::string, double>> sorted(
            m_totals.begin(), m_totals.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](auto const& a, auto const& b) { return a.second > b.second; });
        for (auto const& kv : sorted) {
            const long c = m_counts.count(kv.first) ? m_counts.at(kv.first) : 0;
            const double per_call_ms = (c > 0) ? (kv.second / double(c) * 1000.0) : 0.0;
            const double pct = (grand > 0.0) ? (kv.second / grand * 100.0) : 0.0;
            ss << std::left << std::setw(28) << kv.first
               << std::right << std::fixed << std::setprecision(3) << std::setw(12) << kv.second
               << std::setw(10) << c
               << std::setw(14) << per_call_ms
               << std::setw(9) << std::setprecision(1) << pct << "%" << "\n";
        }
        ss << std::string(74, '-') << "\n";
        ss << std::left << std::setw(28) << "TOTAL"
           << std::right << std::setw(12) << std::setprecision(3) << grand << "\n";
        amrex::Print() << ss.str();
    }

private:
    SimpleProfiler () = default;
    std::map<std::string, double> m_totals;
    std::map<std::string, long>   m_counts;
};

class ScopeTimer {
public:
    explicit ScopeTimer (const std::string& name)
        : m_name(name)
        , m_start(std::chrono::steady_clock::now()) {}
    ~ScopeTimer ()
    {
        const auto end = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(end - m_start).count();
        SimpleProfiler::instance().add(m_name, secs);
    }
private:
    std::string m_name;
    std::chrono::steady_clock::time_point m_start;
};

} // namespace util
} // namespace lucretiatt
