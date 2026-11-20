#pragma once
#include <algorithm>
#include <cmath>

namespace factory_sim {
constexpr double pi = 3.14159265358979323846;
constexpr double half_pi = pi / 2.0;

struct WheelCommand {
  double angle;
  double surface_speed;
};

// A steerable wheel may roll in either direction. Near the +/-90 degree stops
// a small directional error costs less than turning the wheel through 180 degrees.
inline WheelCommand wheel_command(double vx, double vy, double current_angle) {
  const double speed = std::hypot(vx, vy);
  if (speed < 0.001) return {current_angle, 0.0};
  const double heading = std::atan2(vy, vx);
  double best_cost = 1e9;
  WheelCommand best{current_angle, 0.0};
  for (double offset : {-pi, 0.0, pi}) {
    const double candidate = std::clamp(heading + offset, -half_pi, half_pi);
    const double error = std::abs(std::sin(heading - candidate));
    const double cost = error + 0.023 * std::abs(current_angle - candidate);
    if (cost < best_cost) {
      best_cost = cost;
      best = {candidate, vx * std::cos(candidate) + vy * std::sin(candidate)};
    }
  }
  return best;
}
}  // namespace factory_sim
