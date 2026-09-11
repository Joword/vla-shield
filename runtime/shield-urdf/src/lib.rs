//! URDF parsing, forward kinematics, and Cartesian forbidden zones for VLA-Shield.

pub mod error;
pub mod forbidden_zone;
pub mod forward_kinematics;
pub mod geom;
pub mod urdf_loader;

pub use error::UrdfError;
pub use forbidden_zone::{point_in_aabb, AxisAlignedBox};
pub use forward_kinematics::{UrdfKinematicChain, SINGULARITY_MANIPULABILITY_THRESHOLD};
pub use geom::{merge_geoms, parse_link_aabbs, synthesize_link_aabbs};
pub use urdf_loader::{JointSpec, UrdfRobot};
