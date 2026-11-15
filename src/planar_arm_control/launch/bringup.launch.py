"""
bringup.launch.py  —  controller + GUI.

Controller parameters can be overridden from the command line, e.g.
    ros2 launch planar_arm_control bringup.launch.py publish_rate_hz:=100.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("publish_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("max_joint_velocity", default_value="1.0"),
        DeclareLaunchArgument("max_joint_acceleration", default_value="2.0"),
        DeclareLaunchArgument("min_duration", default_value="0.5"),
        DeclareLaunchArgument("backend", default_value="sim"),
        DeclareLaunchArgument("approach_offset", default_value="0.5"),
    ]

    def p(name, type_):
        return ParameterValue(LaunchConfiguration(name), value_type=type_)

    controller = Node(
        package="planar_arm_control",
        executable="controller_node",
        name="controller_node",
        output="screen",
        parameters=[{
            "publish_rate_hz": p("publish_rate_hz", float),
            "max_joint_velocity": p("max_joint_velocity", float),
            "max_joint_acceleration": p("max_joint_acceleration", float),
            "min_duration": p("min_duration", float),
            "backend": p("backend", str),
        }],
    )

    pick_place = Node(
        package="planar_arm_control",
        executable="pick_place_node",
        name="pick_place_node",
        output="screen",
        parameters=[{"approach_offset": p("approach_offset", float)}],
    )

    gui = Node(
        package="planar_arm_control",
        executable="gui_node",
        name="gui_node",
        output="screen",
    )

    return LaunchDescription(args + [controller, pick_place, gui])
