#!/usr/bin/env python3
"""Synchronize five fixed MID-360/CARLA PointCloud2 inputs, transform, crop and merge.

Requires an existing robot_state_publisher. This is not a Livox hardware driver,
scan-pattern simulator, or motion-distortion corrector.
"""
from collections import deque
from pathlib import Path
import math
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener
from pointcloud_utils import (decode_xyzi, rotation_from_quaternion, transform_xyzi,
                              crop_xyzi, encode_xyzi, take_synchronized)


def stamp_ns(message):
    return message.header.stamp.sec*1_000_000_000 + message.header.stamp.nanosec


class Mid360CloudMerger(Node):
    def __init__(self):
        super().__init__('mid360_cloud_merger')
        defaults = {
            'input_topics': [f'/sensing/lidar/l{i}/pointcloud_before_sync' for i in range(1, 6)],
            'output_topic': '/sensing/lidar/concatenated/pointcloud',
            'output_frame': 'base_link',
            'sync_tolerance_sec': 0.05,
            'queue_size': 10,
            'crop_box_filter_self_param_path': '',
            'crop_box_filter_mirror_param_path': '',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        value = lambda name: self.get_parameter(name).value
        topics = value('input_topics')
        if len(topics) != 5 or len(set(topics)) != 5:
            raise ValueError('Exactly five distinct MID-360 input topics are required')
        if value('output_frame') != 'base_link':
            raise ValueError('The supplied crop boxes require output_frame=base_link')
        self.frame = value('output_frame')
        tolerance = float(value('sync_tolerance_sec'))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('sync_tolerance_sec must be finite and nonnegative')
        self.tolerance_ns = int(tolerance*1_000_000_000)
        queue_size = int(value('queue_size'))
        if queue_size < 1:
            raise ValueError('queue_size must be positive')
        self.buffers = [deque(maxlen=queue_size) for _ in topics]
        self.latest_stamp = [None] * len(topics)
        self.boxes = []
        for name in ('crop_box_filter_self_param_path', 'crop_box_filter_mirror_param_path'):
            path = value(name)
            if not path:
                continue
            with Path(path).open(encoding='utf-8') as stream:
                box = yaml.safe_load(stream)['/**']['ros__parameters']
            for axis in ('x', 'y', 'z'):
                lo, hi = float(box['min_'+axis]), float(box['max_'+axis])
                if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                    raise ValueError(f'Invalid crop box in {path}')
            self.boxes.append(box)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(PointCloud2, value('output_topic'), qos_profile_sensor_data)
        self.subscriptions_ = [self.create_subscription(
            PointCloud2, topic, lambda msg, index=i: self.on_cloud(index, msg), qos_profile_sensor_data)
            for i, topic in enumerate(topics)]
        self.get_logger().info('Waiting for all five MID-360 streams and fixed sensor TFs')

    def on_cloud(self, index, message):
        timestamp = stamp_ns(message)
        previous = self.latest_stamp[index]
        if previous is not None and timestamp < previous:
            # A CARLA /clock reset needs a clean epoch across all inputs.
            for buffer in self.buffers:
                buffer.clear()
            self.latest_stamp = [None]*len(self.buffers)
        elif previous == timestamp:
            return
        self.latest_stamp[index] = timestamp
        self.buffers[index].append(message)
        while True:
            group = take_synchronized(self.buffers, stamp_ns, self.tolerance_ns)
            if group is None:
                break
            self.publish_group(group)

    def publish_group(self, messages):
        arrays = []
        try:
            for message in messages:
                if not message.header.frame_id:
                    raise ValueError('Point cloud frame_id is empty')
                points = decode_xyzi(message)
                if message.header.frame_id != self.frame:
                    # Every sensor is fixed relative to base_link. Latest TF avoids
                    # future extrapolation at startup; there is no motion compensation.
                    transform = self.tf_buffer.lookup_transform(self.frame, message.header.frame_id, Time()).transform
                    q, t = transform.rotation, transform.translation
                    points = transform_xyzi(points, rotation_from_quaternion(q.x,q.y,q.z,q.w), [t.x,t.y,t.z])
                arrays.append(crop_xyzi(points, self.boxes))
        except (TransformException, ValueError, KeyError, TypeError) as error:
            self.get_logger().warning(f'Dropped synchronized group: {error}', throttle_duration_sec=5.0)
            return
        points = np.concatenate(arrays, axis=0)
        output = PointCloud2()
        output.header.frame_id = self.frame
        output.header.stamp = max(messages, key=stamp_ns).header.stamp
        output.height, output.width = 1, len(points)
        output.fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
                         for i, name in enumerate(('x', 'y', 'z', 'intensity'))]
        output.is_bigendian = False
        output.point_step, output.row_step = 16, len(points)*16
        output.data = encode_xyzi(points)
        output.is_dense = True
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Mid360CloudMerger()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
