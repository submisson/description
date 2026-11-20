#include <cassert>
#include <cmath>
#include "four_ws_factory_sim/steering.hpp"
#include "four_ws_factory_sim/pattern.hpp"

int main() {
  using namespace factory_sim;
  for (double a = -pi; a <= pi; a += .04) {
    auto command = wheel_command(std::cos(a), std::sin(a), 0.0);
    assert(std::abs(command.angle) <= half_pi);
    assert(std::hypot(std::cos(a) - command.surface_speed * std::cos(command.angle),
                      std::sin(a) - command.surface_speed * std::sin(command.angle)) < .075);
  }
  auto crab = wheel_command(0., -.1, -half_pi);
  assert(std::abs(crab.angle + half_pi) < 1e-6 && crab.surface_speed > .09);
  auto opposite = wheel_command(-.002, -.1, -half_pi);
  assert(std::abs(opposite.angle + half_pi) < .001);
  auto reverse = wheel_command(0., .1, -half_pi);
  assert(std::abs(reverse.angle + half_pi) < .001 && reverse.surface_speed < -.09);
  assert(wheel_command(0., 0., .63).angle == .63);
  for (unsigned head=0;head<5;++head) {
    std::array<unsigned,4> counts{};
    for (unsigned n=0;n<20000;++n) {
      auto v=pattern_direction(n,head);
      assert(std::abs(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]-1.)<1e-12);
      assert(v[2]>=std::sin(-7.*pi/180) && v[2]<=std::sin(52.*pi/180));
      ++counts[2*(v[0]>0)+(v[1]>0)];
    }
    for (auto count:counts) assert(count>4950 && count<5050);
    assert(pattern_direction(0,head)!=pattern_direction(20000,head));
  }
}
