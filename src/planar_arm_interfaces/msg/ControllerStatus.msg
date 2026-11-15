# Controller state, published at a low rate (~10 Hz) for GUIs and sequencers.

std_msgs/Header header
string state            # "IDLE" or "MOVING"
string control_mode     # active control mode, e.g. "position"
float64 goal_x          # Cartesian goal of the last accepted command
float64 goal_y
bool projected          # last goal was projected onto the workspace boundary
float64 progress        # 0..1 along the active trajectory (1 when idle)
string message          # last planner message
