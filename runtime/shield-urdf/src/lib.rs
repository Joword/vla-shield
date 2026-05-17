//! URDF parsing, forward kinematics, and Cartesian forbidden zones for VLA-Shield.

pub mod error;
pub mod forbidden_zone;
pub mod forward_kinematics;
pub mod urdf_loader;

pub use error::UrdfError;
pub use forbidden_zone::{AxisAlignedBox, point_in_aabb};
pub use forward_kinematics::{
    UrdfKinematicChain, SINGULARITY_MANIPULABILITY_THRESHOLD,
};
pub use urdf_loader::{JointSpec, UrdfRobot};
