# Model 24999 LiDAR slice: elevation-major, then azimuth, then XYZ.
# header.stamp is the original input cloud stamp; frame_id is virtual sensor yaw.
std_msgs/Header header
# /secondary_imu is headerless: this is ROS reception time, NOT measurement time.
builtin_interfaces/Time imu_received_stamp
# Signed (IMU reception - input cloud stamp), seconds. Not hardware synchronization.
float32 imu_cloud_offset_s
# False permits numerical preview only; motor consumers must require true.
bool extrinsics_verified
uint16 rows
uint16 columns
float32 normalization_distance_m
uint32 input_points
uint32 rejected_points
uint32 occupied_cells
# Separate real measurements from points inferred under a flat-floor assumption.
uint32 observed_cells
uint32 floor_filled_cells
float64 floor_height_m
uint32 floor_fit_inlier_points
float64 floor_fit_rms_m
# The exact normalized pose used by the adapter; still receipt-time matched.
float64[4] torso_quaternion_wxyz
# Fixed 16 x 41 x 3, XYZ / 5 m. Empty cells are exactly zero.
float32[1968] xyz
