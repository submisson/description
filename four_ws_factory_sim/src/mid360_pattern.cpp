#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <boost/pointer_cast.hpp>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/UpdateInfo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/physics/RayShape.hh>
#include <gazebo_ros/node.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include "four_ws_factory_sim/pattern.hpp"

namespace factory_sim {
using Cloud = sensor_msgs::msg::PointCloud2;

class Mid360Pattern : public gazebo::ModelPlugin {
 public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override {
    node_=gazebo_ros::Node::Get(sdf);
    points_=sdf->Get<unsigned int>("points_per_frame");
    if (points_==0 || points_>100000) throw std::runtime_error("Invalid sample count");
    ray_=boost::dynamic_pointer_cast<gazebo::physics::RayShape>(
        model->GetWorld()->Physics()->CreateShape("ray", gazebo::physics::CollisionPtr()));
    if (!ray_) throw std::runtime_error("The active physics engine has no ray query");
    for (size_t i=0; i<heads_.size(); ++i) {
      auto &h=heads_[i];
      const std::string id="l"+std::to_string(i+1);
      h.frame="mid360_"+id;
      h.link=model->GetLink(h.frame);
      if (!h.link) throw std::runtime_error("Missing lidar link "+h.frame);
      h.pub=node_->create_publisher<Cloud>("/sensing/lidar/"+id+"/pointcloud_before_sync",
                                           rclcpp::SensorDataQoS());
      h.hits.reserve(points_);
    }
    connection_=gazebo::event::Events::ConnectWorldUpdateBegin(
        std::bind(&Mid360Pattern::step,this,std::placeholders::_1));
  }

  void Reset() override {
    epoch_=-1;
    serial_=0;
    next_=0;
    rng_.seed(1947);
    noise_.reset();
    for (auto &h:heads_) h.hits.clear();
  }

 private:
  struct Hit { float x,y,z,intensity; std::uint32_t offset; };
  static_assert(sizeof(Hit)==20,"Unexpected PointCloud2 record layout");
  struct Head {
    std::string frame;
    gazebo::physics::LinkPtr link;
    rclcpp::Publisher<Cloud>::SharedPtr pub;
    std::vector<Hit> hits;
  };
  static constexpr std::int64_t period=100000000;  // 10 Hz

  void emit(Head &h) {
    Cloud msg;
    msg.header.frame_id=h.frame;
    msg.header.stamp.sec=static_cast<std::int32_t>(epoch_/1000000000);
    msg.header.stamp.nanosec=static_cast<std::uint32_t>(epoch_%1000000000);
    msg.height=1;
    msg.width=static_cast<std::uint32_t>(h.hits.size());
    const std::uint16_t one=1;
    msg.is_bigendian=*reinterpret_cast<const unsigned char*>(&one)==0;
    msg.is_dense=true;
    msg.point_step=sizeof(Hit);
    msg.row_step=msg.width*msg.point_step;
    const std::array<std::string,5> names{"x","y","z","intensity","offset_time"};
    for (unsigned i=0;i<names.size();++i) {
      sensor_msgs::msg::PointField field;
      field.name=names[i];
      field.offset=4*i;
      field.count=1;
      field.datatype=i==4 ? field.UINT32 : field.FLOAT32;
      msg.fields.push_back(field);
    }
    msg.data.resize(msg.row_step);
    if (!h.hits.empty()) std::memcpy(msg.data.data(),h.hits.data(),msg.row_step);
    h.pub->publish(msg);
    h.hits.clear();
  }

  void scan(Head &head, unsigned id, const ignition::math::Pose3d &pose,
            std::uint32_t offset) {
    const auto u=pattern_direction(serial_,id);
    const ignition::math::Vector3d dir(u[0],u[1],u[2]);
    const auto world=pose.Rot().RotateVector(dir);
    ray_->SetPoints(pose.Pos()+world*.10,pose.Pos()+world*40.);
    double distance=0.;
    std::string entity;
    ray_->GetIntersection(distance,entity);
    if (entity.empty() || !std::isfinite(distance) || distance<0. || distance>=39.9) return;
    const double r=.10+distance+noise_(rng_);
    if (r<.10 || r>40.) return;
    head.hits.push_back({static_cast<float>(u[0]*r),static_cast<float>(u[1]*r),
                         static_cast<float>(u[2]*r),0.f,offset});
  }

  void step(const gazebo::common::UpdateInfo &info) {
    const std::int64_t now=static_cast<std::int64_t>(info.simTime.sec)*1000000000+info.simTime.nsec;
    if (epoch_<0 || now<epoch_ || now-epoch_>2*period) {
      Reset();
      epoch_=now;
    }
    std::array<ignition::math::Pose3d,5> poses;
    std::array<bool,5> active;
    for (size_t i=0;i<heads_.size();++i) {
      poses[i]=heads_[i].link->WorldPose();
      active[i]=heads_[i].pub->get_subscription_count()>0;
    }
    // A common frame clock synchronizes the heads. The moving pose is sampled at
    // each physics step. A late subscriber receives a complete frame on the next period.
    while (true) {
      if (next_==points_) {
        if (now<epoch_+period) return;
        for (size_t i=0;i<heads_.size();++i) {
          if (active[i]) emit(heads_[i]); else heads_[i].hits.clear();
        }
        epoch_+=period;
        next_=0;
      }
      const auto offset=static_cast<std::uint32_t>(period*next_/points_);
      if (epoch_+offset>now) return;
      for (size_t i=0;i<heads_.size();++i)
        if (active[i]) scan(heads_[i],static_cast<unsigned>(i),poses[i],offset);
      ++next_;
      ++serial_;
    }
  }

  gazebo_ros::Node::SharedPtr node_;
  gazebo::physics::RayShapePtr ray_;
  std::array<Head,5> heads_;
  std::mt19937 rng_{1947};
  std::normal_distribution<double> noise_{0.,.003};
  std::uint64_t serial_=0;
  unsigned next_=0,points_=0;
  std::int64_t epoch_=-1;
  gazebo::event::ConnectionPtr connection_;
};
GZ_REGISTER_MODEL_PLUGIN(Mid360Pattern)
}
