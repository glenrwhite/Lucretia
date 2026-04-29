#pragma once

#include <string>
#include <utility>


namespace lucretiatt {
namespace elements {

/** Mixin: gives an element a human-readable name (for logs / diagnostics). */
struct Named
{
    Named () = default;
    explicit Named (std::string n) : m_name(std::move(n)) {}

    [[nodiscard]] const std::string& name () const noexcept { return m_name; }

    std::string m_name;
};

} // namespace elements
} // namespace lucretiatt
