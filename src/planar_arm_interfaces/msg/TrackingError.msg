# Per-joint trajectory tracking, published by the controller at the control rate.
# In "position" mode the error is identically zero (ideal servo); in "pid"
# mode it shows how well the PID tracks the planned reference.

std_msgs/Header header
float64[] reference     # q_ref [rad]
float64[] measured      # q [rad]
float64[] error         # q_ref - q [rad]
float64[] effort        # applied joint torque [N·m] (empty in position mode)
