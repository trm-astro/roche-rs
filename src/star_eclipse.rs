use crate::errors::RocheError;
use crate::{Etype, Star, Vec3};
use crate::{ingress_egress, sphere_eclipse};

///
/// Covenience routine which wraps up the code to compute a star eclipse allowing for roche-dirstortion or not.
///
/// Arguments:
///
/// * `q`: mass ratio
/// * `r`: radius of star
/// * `spin`: spin/orbital frequency factor
/// * `ffac`: roche filling factor
/// * `iangle`: inclination angle
/// * `posn`: position of point
/// * `delta`: accuracy in phase
/// * `roche`: account for roche distortion or not
/// * `eclipses`: set of ingress/egress pairs
///
pub fn star_eclipse(
    q: f64,
    spin: f64,
    r: f64,
    ffac: f64,
    iangle: f64,
    posn: &Vec3,
    delta: f64,
    roche: bool,
    star: Star,
    eclipses: &mut Etype,
) -> Result<(), RocheError> {
    let ri = iangle.to_radians();
    let (sini, cosi) = ri.sin_cos();
    let cofm = match star {
        Star::Primary => Vec3::cofm1(),
        Star::Secondary => Vec3::cofm2(),
    };
    let mut lam1: f64 = 0.0;
    let mut lam2: f64 = 0.0;
    let mut ingress: f64 = 0.0;
    let mut egress: f64 = 0.0;
    // let mut eclipses = Etype::new();
    if (roche
        && ingress_egress(
            q,
            star,
            spin,
            ffac,
            iangle,
            delta,
            posn,
            &mut ingress,
            &mut egress,
        )?)
        || (!roche
            && sphere_eclipse(
                cosi,
                sini,
                posn,
                &cofm,
                r,
                &mut ingress,
                &mut egress,
                &mut lam1,
                &mut lam2,
            ))
    {
        eclipses.push((ingress, egress));
    }
    Ok(())
}
