#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/UpdateInfo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include "four_ws_factory_sim/steering.hpp"

namespace factory_sim {
using Joint = gazebo::physics::JointPtr;
using Twist = geometry_msgs::msg::Twist;
using JointState = sensor_msgs::msg::JointState;

class FactoryVehicleControl : public gazebo::ModelPlugin {
 public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = std::move(model);
    node_ = gazebo_ros::Node::Get(sdf);
    base_ = model_->GetLink("base_link");
    if (!base_) throw std::runtime_error("Missing base_link");
    clear_reach_ = sdf->Get<double>("clear_reach");
    const std::array<std::string, 4> ids{"FL", "FR", "RL", "RR"};
    for (size_t i = 0; i < ids.size(); ++i) {
      steer_[i] = bind_joint("steer_" + ids[i] + "_joint");
      wheels_[i] = bind_joint("wheel_" + ids[i] + "_joint");
    }
    inner_ = bind_joint("reach_inner_joint");
    outer_ = bind_joint("reach_outer_joint");
    shift_ = bind_joint("side_shift_joint");
    lift_ = bind_joint("fork_lift_joint");
    state_pub_ = node_->create_publisher<JointState>("/joint_states", 10);
    odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);
    velocity_sub_ = node_->create_subscription<Twist>("/cmd_vel", 1, [this](Twist::SharedPtr msg) {
      if (!(std::isfinite(msg->linear.x) && std::isfinite(msg->linear.y) &&
            std::isfinite(msg->angular.z))) return;
      std::lock_guard<std::mutex> lock(mutex_);
      command_ = *msg;
      fresh_ = true;
    });
    tool_sub_ = node_->create_subscription<JointState>("/fork/targets", 1,
        [this](JointState::SharedPtr msg) {
          if (msg->name.size() != msg->position.size()) return;
          std::lock_guard<std::mutex> lock(mutex_);
          Targets requested = targets_;
          for (size_t i = 0; i < msg->name.size(); ++i) {
            auto value = msg->position[i];
            if (!std::isfinite(value)) return;
            if (msg->name[i] == "reach") requested.reach = value;
            else if (msg->name[i] == "side_shift_joint") requested.shift = value;
            else if (msg->name[i] == "fork_lift_joint") requested.lift = value;
            else return;
          }
          targets_ = requested;
        });
    update_ = gazebo::event::Events::ConnectWorldUpdateBegin(
        std::bind(&FactoryVehicleControl::step, this, std::placeholders::_1));
  }

  void Reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    command_ = Twist();
    targets_ = Targets();
    fresh_ = false;
    last_command_ = -1.0;
    last_pub_ = -1.0;
  }

 private:
  struct Targets { double reach = 0.0, shift = 0.0, lift = 0.0; };

  Joint bind_joint(const std::string &name) {
    auto joint = model_->GetJoint(name);
    if (!joint) throw std::runtime_error("Missing Gazebo joint " + name);
    joint->SetParam("fmax", 0, joint->GetEffortLimit(0));
    joints_.push_back(joint);
    return joint;
  }

  static void servo(const Joint &joint, double target) {
    const double goal = std::clamp(target, joint->LowerLimit(0), joint->UpperLimit(0));
    const double vmax = joint->GetVelocityLimit(0);
    joint->SetParam("vel", 0, std::clamp(7.0 * (goal - joint->Position(0)), -vmax, vmax));
  }

  void drive(const Twist &twist) {
    // Twist origin is the rear axle (base_link). Include omega cross r for each wheel.
    constexpr std::array<double, 4> xs{1.2165,1.2165,0.,0.};
    constexpr std::array<double, 4> ys{.343148044,-.342851956,.342851956,-.343148044};
    std::array<WheelCommand, 4> commands;
    double max_speed = .8, max_error = 0.;
    for (size_t i = 0; i < 4; ++i) {
      const double vx = twist.linear.x - twist.angular.z * ys[i];
      const double vy = twist.linear.y + twist.angular.z * xs[i];
      const double current = steer_[i]->Position(0);
      commands[i] = wheel_command(vx, vy, current);
      max_speed = std::max(max_speed, std::abs(commands[i].surface_speed));
      max_error = std::max(max_error, std::abs(commands[i].angle - current));
      servo(steer_[i], commands[i].angle);
    }
    const double scale = max_error < .16 ? .8 / max_speed : 0.;
    for (size_t i = 0; i < 4; ++i)
      wheels_[i]->SetParam("vel", 0, commands[i].surface_speed * scale / .175);
  }

  void step(const gazebo::common::UpdateInfo &info) {
    const double now = info.simTime.Double();
    Twist velocity;
    Targets t;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (fresh_) { last_command_ = now; fresh_ = false; }
      if (last_command_ >= 0. && now >= last_command_ && now - last_command_ < .50)
        velocity = command_;
      t = targets_;
    }
    drive(velocity);
    double reach = inner_->Position(0) + outer_->Position(0);
    // Below the deck, hold both stages out until the toes have risen clear.
    // A simultaneous extend/lower request therefore extends first.
    if (reach < clear_reach_ - .006) t.lift = std::max(t.lift, -.005);
    if (lift_->Position(0) < -.012) t.reach = std::max(t.reach, clear_reach_);
    servo(inner_, t.reach / 2.);
    servo(outer_, t.reach / 2.);
    servo(shift_, t.shift);
    servo(lift_, t.lift);
    if (now < last_pub_ || now - last_pub_ >= .025) {
      last_pub_ = now;
      publish(info);
    }
  }

  void publish(const gazebo::common::UpdateInfo &info) {
    JointState js;
    js.header.stamp.sec = info.simTime.sec;
    js.header.stamp.nanosec = info.simTime.nsec;
    for (const auto &j : joints_) {
      js.name.push_back(j->GetName());
      js.position.push_back(j->Position(0));
      js.velocity.push_back(j->GetVelocity(0));
    }
    state_pub_->publish(js);
    const auto pose = base_->WorldPose();
    const auto linear = pose.Rot().RotateVectorReverse(base_->WorldLinearVel());
    const auto angular = pose.Rot().RotateVectorReverse(base_->WorldAngularVel());
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = js.header.stamp;
    odom.header.frame_id = "odom";
    odom.child_frame_id = "base_link";
    auto &p = odom.pose.pose;
    p.position.x=pose.Pos().X(); p.position.y=pose.Pos().Y(); p.position.z=pose.Pos().Z();
    p.orientation.x=pose.Rot().X(); p.orientation.y=pose.Rot().Y();
    p.orientation.z=pose.Rot().Z(); p.orientation.w=pose.Rot().W();
    auto &v=odom.twist.twist;
    v.linear.x=linear.X(); v.linear.y=linear.Y(); v.linear.z=linear.Z();
    v.angular.x=angular.X(); v.angular.y=angular.Y(); v.angular.z=angular.Z();
    odom_pub_->publish(odom);
    geometry_msgs::msg::TransformStamped tf;
    tf.header = odom.header;
    tf.child_frame_id = odom.child_frame_id;
    tf.transform.translation.x=p.position.x; tf.transform.translation.y=p.position.y;
    tf.transform.translation.z=p.position.z; tf.transform.rotation=p.orientation;
    broadcaster_->sendTransform(tf);
  }

  gazebo::physics::ModelPtr model_;
  gazebo::physics::LinkPtr base_;
  gazebo_ros::Node::SharedPtr node_;
  std::array<Joint,4> steer_, wheels_;
  Joint inner_, outer_, shift_, lift_;
  std::vector<Joint> joints_;
  std::mutex mutex_;
  Twist command_;
  Targets targets_;
  bool fresh_ = false;
  double clear_reach_ = 0., last_command_ = -1., last_pub_ = -1.;
  rclcpp::Subscription<Twist>::SharedPtr velocity_sub_;
  rclcpp::Subscription<JointState>::SharedPtr tool_sub_;
  rclcpp::Publisher<JointState>::SharedPtr state_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
  gazebo::event::ConnectionPtr update_;  // Disconnect the update callback first on destruction.
};
GZ_REGISTER_MODEL_PLUGIN(FactoryVehicleControl)
}  // namespace factory_sim
