use pyo3::prelude::*;

pub mod blink;
pub use blink::blink;

pub mod constants;

pub mod disc_eclipse;
pub use disc_eclipse::disc_eclipse;

pub mod errors;

pub mod face;
pub use face::*;

pub mod fblink;
pub use fblink::*;

pub mod ingress_egress;
pub use ingress_egress::ingress_egress;

pub mod jacobi;
pub use jacobi::*;

pub mod lobes;
pub use lobes::lobe1;
pub use lobes::lobe2;

pub mod phases;
pub use phases::*;

pub mod planck;
pub use planck::*;

pub mod point;
pub use point::Point;

pub mod pot_min;
pub use pot_min::*;

pub mod potential;
pub use potential::*;

pub mod rcirc;
pub use rcirc::*;

pub mod ref_sphere;
pub use ref_sphere::*;

pub mod roche_context;
pub use roche_context::RocheContext;

pub mod roche_shadow;
pub use roche_shadow::*;

pub mod set_earth;
pub use set_earth::*;

pub mod solve_triads;
pub use solve_triads::findi;

pub mod sphere_eclipse;
pub use sphere_eclipse::*;

pub mod star_eclipse;
pub use star_eclipse::star_eclipse;

pub mod stream_physics;
pub use stream_physics::*;

pub mod vec3;
pub use vec3::Vec3;

pub mod vel_transform;
pub use vel_transform::vel_transform;

pub mod vstream_physics;
pub use vstream_physics::*;

pub mod x_lagrange;
pub use x_lagrange::*;

pub mod zeta_rlobe_eggleton;
pub use zeta_rlobe_eggleton::*;


#[pyclass(from_py_object, eq, eq_int)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Star {
    Primary = 1,
    Secondary = 2,
}

pub type Etype = Vec<(f64, f64)>;

// Python module
#[pymodule]
mod roche {

    #[pymodule_export]
    use crate::point::Point;

    #[pymodule_export]
    use crate::Star;

    #[pymodule_export]
    use crate::vec3::Vec3;
    
    #[pymodule_export]
    use crate::face::face;
    #[pymodule_export]
    use crate::fblink::fblink;
    
    #[pymodule_export]
    use crate::ingress_egress::ingress_egress_wrapper;
    
    #[pymodule_export]
    use crate::jacobi::jacobi;
    
    #[pymodule_export]
    use crate::lobes::{
        lobe1_py,
        lobe2_py,
        vlobe1_py,
        vlobe2_py
    };
    
    #[pymodule_export]
    use crate::phases::{
        wdradius,
        wdphases,
        bsphases
    };
    
    #[pymodule_export]
    use crate::planck::{
        planck,
        dplanck,
        dlpdlt
    };
    
    #[pymodule_export]
    use crate::potential::{
        rpot,
        rpot1,
        rpot2,
        drpot,
        drpot1,
        drpot2,
        rpot_val,
        rpot_val_grad
    };
    
    #[pymodule_export]
    use crate::rcirc::rcirc;
    
    #[pymodule_export]
    use crate::ref_sphere::ref_sphere;
    
    #[pymodule_export]
    use crate::roche_shadow::roche_shadow_py;
    
    #[pymodule_export]
    use crate::set_earth::{
        set_earth_iangle,
        set_earth
    };
    
    #[pymodule_export]
    use crate::sphere_eclipse::{
        sphere_eclipse_wrapper,
        sphere_eclipse_vector_wrapper
    };
    
    #[pymodule_export]
    use crate::solve_triads::{
        findi,
        findq,
        findphi
    };
    
    #[pymodule_export]
    use crate::stream_physics::{
        stradv_py,
        rocacc,
        strinit,
        stream_py,
        streamr_py,
        strmnx_wrapper,
        brightspot_position,
        bspot
    };
    
    #[pymodule_export]
    use crate::vstream_physics::vstream_reg_py;
    
    #[pymodule_export]
    use crate::x_lagrange::{
        x_l1,
        x_l1_1,
        x_l1_2,
        x_l2,
        x_l3
    };
    
    #[pymodule_export]
    use crate::zeta_rlobe_eggleton::{
        zeta_rlobe_eggleton,
        dzetadq_rlobe_eggleton
    };
}
