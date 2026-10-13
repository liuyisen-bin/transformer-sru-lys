"""Launch only the model-24999 sensor adapter; no policy or motor commands."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory('transformer_sru_lidar_adapter'))
                 / 'config' / 'livox_24999.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config),
        Node(package='transformer_sru_lidar_adapter', executable='lidar_adapter_node',
             name='transformer_sru_lidar_adapter', output='screen',
             parameters=[LaunchConfiguration('config')]),
    ])
