from pathlib import Path
from tempfile import TemporaryDirectory

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from four_ws_factory_sim.vehicle import descriptions, PROFILES


def start(context):
    name='four_ws_factory_sim'
    paths={k:Path(get_package_share_directory(k)) for k in
           (name,'four_ws_vehicle_description','four_ws_sensor_kit_description','four_ws_sensor_kit_launch')}
    share=paths[name]
    urdf,sdf=descriptions(paths,LaunchConfiguration('lidar_profile').perform(context),
                         LaunchConfiguration('lidar_backend').perform(context))
    temp=TemporaryDirectory(prefix='four_ws_factory_')
    vehicle=Path(temp.name)/'vehicle.sdf'
    vehicle.write_text(sdf,encoding='utf-8')
    gazebo=Path(get_package_share_directory('gazebo_ros'))/'launch'/'gazebo.launch.py'
    common={'use_sim_time':True}

    def cleanup(_context):
        temp.cleanup()
        return []

    def sim_node(executable,**kwargs):
        return Node(package=name,executable=executable,output='screen',parameters=[common],**kwargs)

    return [
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI',''),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gazebo)),
            launch_arguments={'world':str(share/'worlds'/'factory.world'),
                              'gui':LaunchConfiguration('gui'),'verbose':'true','pause':'false'}.items()),
        Node(package='robot_state_publisher',executable='robot_state_publisher',
             parameters=[dict(common,robot_description=ParameterValue(urdf,value_type=str))],output='screen'),
        Node(package='gazebo_ros',executable='spawn_entity.py',output='screen',
             arguments=['-entity','four_ws','-file',str(vehicle),'-z','.004','-timeout','150']),
        Node(package='four_ws_sensor_kit_launch',executable='merge_mid360_clouds.py',output='screen',
             parameters=[str(paths['four_ws_sensor_kit_launch']/'config'/'mid360_cloud_merger.param.yaml'),
                         dict(common,crop_box_filter_self_param_path=str(
                             paths['four_ws_sensor_kit_launch']/'config'/'crop_box_filter_self.param.yaml'))]),
        sim_node('pallet_detector'),
        sim_node('pick_pallet',condition=IfCondition(LaunchConfiguration('demo'))),
        sim_node('check_pick',condition=IfCondition(LaunchConfiguration('verify'))),
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gui',default_value='true'),
        DeclareLaunchArgument('demo',default_value='false'),
        DeclareLaunchArgument('verify',default_value='false'),
        DeclareLaunchArgument('lidar_profile',default_value='standard',choices=list(PROFILES)),
        DeclareLaunchArgument('lidar_backend',default_value='ray',choices=['ray','gpu_ray','pattern']),
        OpaqueFunction(function=start),
    ])
