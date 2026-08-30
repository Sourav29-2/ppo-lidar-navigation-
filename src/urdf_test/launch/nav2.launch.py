import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetRemap,Node

def generate_launch_description():
    package_name = 'urdf_test'
    
    # Path to your freshly saved map file
    map_file_path = os.path.join(get_package_share_directory(package_name), 'maps', 'my_obstacles_map.yaml')

    # Force simulation time alignment
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_yaml_file = LaunchConfiguration('map', default=map_file_path)

    # Path to your custom Nav2 parameters file
    nav2_params_path = os.path.join(get_package_share_directory(package_name), 'config', 'nav2_params.yaml')

    # Include the official Nav2 bringup pipeline wrapped in a GroupAction for topic remapping
    nav2_bringup = GroupAction(
        actions=[
            SetRemap(src='/cmd_vel', dst='/cmd_vel_nav_nav2'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'bringup_launch.py')
                ]),
                launch_arguments={
                    'map': map_yaml_file,
                    'use_sim_time': 'true',
                    'autostart': 'true',
                    'params_file': nav2_params_path
                }.items()
            )
        ]
    )
    # Add the Twist Mux Node directly to handle topic priorities
    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'topics.navigation.topic': 'cmd_vel_nav',
            'topics.navigation.timeout': 0.5,
            'topics.navigation.priority': 10,
            'topics.teleop.topic': 'cmd_vel_teleop',
            'topics.teleop.timeout': 0.5,
            'topics.teleop.priority': 100 # Higher priority overrides autonomous goals instantly!
        }],
        remappings=[('cmd_vel_out', 'cmd_vel')]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock'),
        DeclareLaunchArgument('map', default_value=map_file_path, description='Full path to map yaml file to load'),
        nav2_bringup,
        twist_mux
    ])