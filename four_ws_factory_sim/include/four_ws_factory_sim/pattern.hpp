#pragma once
#include <array>
#include <cmath>
#include <cstdint>

namespace factory_sim {
// R2 low-discrepancy sampling over the solid angle. Each head has an independent
// phase. The sequence continues across frames instead of repeating a raster.
inline std::array<double,3> pattern_direction(std::uint64_t index, unsigned head) {
  const double n = static_cast<double>(index) + 1.;
  const auto fract = [](double a) { return a - std::floor(a); };
  const double azimuth = 6.283185307179586 * fract(.7548776662466927*n + .137*head);
  constexpr double sin_min = -.12186934340514748;  // sin(-7 degrees)
  constexpr double sin_max = .788010753606722;     // sin(52 degrees)
  const double z = sin_min + (sin_max-sin_min)*fract(.5698402909980532*n + .193*head);
  const double xy = std::sqrt(1.-z*z);
  return {xy*std::cos(azimuth),xy*std::sin(azimuth),z};
}
}
