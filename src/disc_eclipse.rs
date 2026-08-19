use crate::errors::RocheError;
use crate::{Etype, Vec3};
use std::f64::consts::TAU;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
// This enumerates the 5 possible outcomes of the LOSC intersection with a circle.
pub enum Circle {
    // Line of sight cone starts at or above the circle of interest
    Above,
    // Line of sight circle is everywhere inside circle of interest
    Inside,
    // Line of sight circle is everywhere outside circle of interest
    Outside,
    // Line of sight circle is separated from the circle of interest
    Separate,
    // Line of sight circle cone intersects the circle of interest
    Crossing,
}

///
/// disc_eclipse works out phase ranges during which a cylindrically symmetric, flared disc
/// running between a pair of radii eclipses a given point.
///
/// Arguments:
///
/// * `iangle`:  the orbital inclination, degrees. 90 = edge on.
/// * `r`:       the position vector of the point in question (units of binary separation)
/// * `rdisc1`: inner disc  radius, units of separation
/// * `rdisc2`: outer disc radius, units of separation
/// * `beta`: exponent of flaring, so that the height scales as r**beta. beta should be >= 1
/// * `height disc`: height at unit radius in disc (even if it does not exist)
///
/// Returns:
///
/// * a vector of ingress and egress phase pairs during which the point in question is eclipsed.
///   The ingress phase will always be between 0 and 1 while the egress phase will be larger than this, but
///   by no more than 1 cycle. If the vector is null, no eclipse takes place.
///
pub fn disc_eclipse(
    iangle: f64,
    rdisc1: f64,
    rdisc2: f64,
    beta: f64,
    height: f64,
    r: &Vec3,
) -> Result<Etype, RocheError> {
    if beta < 1.0 {
        return Err(RocheError::ParameterError(
            "beta must be >= 1.0".to_string(),
        ));
    }

    // Compute and store cosine and sine of inclination if need be.
    // let mut iangle_old: f64 = -1.0e30;
    let sini: f64;
    let cosi: f64;
    (sini, cosi) = iangle.to_radians().sin_cos();
    // }

    let mut temp = Etype::new();

    // Compute height of disc at outer boundary
    let h_out: f64 = height * rdisc2.powf(beta);

    // Deal with points too high ever to be eclipsed whatever the inclination
    if r.z >= h_out {
        return Ok(temp);
    }

    // Special case of exactly edge-on, only curved outer edge matters.
    if cosi == 0.0 {
        if r.z.abs() < h_out {
            let rxy: f64 = (r.x * r.x + r.y * r.y).sqrt();
            if rxy <= rdisc2 {
                temp.push((0.0, 1.0));
            } else {
                let subtend: f64 = (rdisc2 / rxy).asin() / TAU;
                let pcen: f64 = r.y.atan2(-r.x) / TAU;
                let mut ingress: f64 = pcen - subtend;
                ingress -= ingress.floor();
                let egress: f64 = ingress + 2.0 * subtend;
                temp.push((ingress, egress));
            }
        }
        return Ok(temp);
    }

    // Work out distance from axis
    let rxy: f64 = (r.x * r.x + r.y * r.y).sqrt();

    if rdisc1 < rxy && rxy < rdisc2 && r.z.abs() < height * rxy.powf(beta) {
        // Point is inside disc and so is eclipsed
        temp.push((0.0, 1.1));
        return Ok(temp);
    }

    let tani: f64 = sini / cosi;
    let mut result: Circle;

    let mut phase: f64 = 0.0;
    let mut ingress: f64;
    let mut egress: f64;

    if rxy < rdisc2 && r.z >= height * rdisc1.max(rxy).powf(beta) {
        // Point is in approximately conical region above the disc. Just need to check whether
        // it is not occulted by the edge of the disc
        result = circle_eclipse(rxy, r.z, h_out, rdisc2, tani, &mut phase);

        if result == Circle::Outside {
            // point will be occulted by the disc edge at all phases
            temp.push((0.0, 1.1));
        } else if result == Circle::Crossing {
            // point partially occulted by disc edge; work out phases
            let phi0: f64 = r.y.atan2(-r.x) / TAU;
            ingress = phi0 + phase;
            ingress -= ingress.floor();
            egress = ingress + 1.0 - 2.0 * phase;
            temp.push((ingress, egress));
        }
        return Ok(temp);
    }

    // Compute the radius of circle formed by LOSC in the plane of
    // the lower outer rim of the disc
    let rcone_lo: f64 = 0.0_f64.max(tani * (-h_out - r.z));

    // Circle encloses rim, so no intersection
    if rcone_lo >= rxy + rdisc2 {
        return Ok(temp);
    }

    // Compute the radius of circle formed by LOSC in the plane of
    // the upper outer rim of the disc
    let rcone_hi: f64 = tani * (h_out - r.z);

    // Circle disjoint from rim, so no intersection
    if rxy >= rcone_hi + rdisc2 {
        return Ok(temp);
    }

    // For the moment we pretend that the disc has no hole at its centre, so
    // that we are simply interested in the phases over which eclipse occurs.
    // At this point we are guaranteed that this will happen. All events are
    // symmetrically located around a phase defined by x and y only which will
    // be calculated at the end. We therefore just find the half range which
    // is called 'eclipse_phase' below.

    let eclipse_phase: f64;
    if rxy + rcone_lo <= rdisc2 {
        // Cone swept out by line of sight always inside lower face so total eclipse
        eclipse_phase = 0.5;
    } else if rxy <= rdisc2 {
        // Points that project close to the z axis which are only
        // partially obscured by the disc hovering above them.
        // this means they must be below -HOUT
        eclipse_phase = cut_phase(rxy, rcone_lo, rdisc2);
    } else {
        // Points further from the z axis than the outer rim of the disc that
        // will be eclipsed.
        if rcone_hi * rcone_hi + rdisc2 * rdisc2 >= rxy * rxy
            && rcone_lo * rcone_lo + rdisc2 * rdisc2 <= rxy * rxy
        {
            // In this case it is the curved outer disc rim that sets the limit
            eclipse_phase = (rdisc2 / rxy).asin() / TAU;
        } else if rcone_hi * rcone_hi + rdisc2 * rdisc2 < rxy * rxy {
            // In this case it is upper outer rim that sets the limit
            eclipse_phase = cut_phase(rxy, rcone_hi, rdisc2);
        } else {
            // In this case it is lower outer rim that sets the limit
            eclipse_phase = cut_phase(rxy, rcone_lo, rdisc2);
        }
    }

    // At this point we have covered all cases for the eclipse, whilst ignoring the
    // possibility of seeing the point through the hole in the middle of the disc.
    // Now let's calculate the 'appear_phase' if any.

    // First compute height of disc at inner boundary
    let h_in: f64 = height * rdisc1.powf(beta);

    let mut appear_phase: f64 = -1.0;

    if r.z < -h_out {
        // In this case the LOSC has to run through 4 circles which are the upper and
        // lower outer and inner rims.

        // First, the lower outer rim
        result = circle_eclipse(rxy, r.z, -h_out, rdisc2, tani, &mut phase);
        if result == Circle::Inside {
            appear_phase = 0.5;
        } else if result == Circle::Crossing {
            appear_phase = phase;
        }

        // Second, the lower inner rim
        if appear_phase > 0.0 {
            result = circle_eclipse(rxy, r.z, -h_in, rdisc1, tani, &mut phase);
            if result == Circle::Crossing {
                appear_phase = appear_phase.min(phase);
            } else if result != Circle::Inside {
                appear_phase = -1.0;
            }
        }

        // Third, the upper inner rim
        if appear_phase > 0.0 {
            result = circle_eclipse(rxy, r.z, h_in, rdisc1, tani, &mut phase);
            if result == Circle::Crossing {
                appear_phase = appear_phase.min(phase);
            } else if result != Circle::Inside {
                appear_phase = -1.0;
            }
        }

        // Fourth, the upper outer rim
        if appear_phase > 0.0 {
            result = circle_eclipse(rxy, r.z, h_out, rdisc2, tani, &mut phase);
            if result == Circle::Crossing {
                appear_phase = appear_phase.min(phase);
            } else if result != Circle::Inside {
                appear_phase = -1.0;
            }
        }
    } else if rxy < rdisc1 {
        if r.z < -h_in {
            // Points hovering around underside of disc. Have to consider just three circles

            // First, the lower inner rim
            result = circle_eclipse(rxy, r.z, -h_in, rdisc1, tani, &mut phase);
            if result == Circle::Inside {
                appear_phase = 0.5;
            } else if result == Circle::Crossing {
                appear_phase = phase;
            }

            // Second, the upper inner rim
            if appear_phase > 0.0 {
                result = circle_eclipse(rxy, r.z, h_in, rdisc1, tani, &mut phase);
                if result == Circle::Crossing {
                    appear_phase = appear_phase.min(phase);
                } else if result != Circle::Inside {
                    appear_phase = -1.0;
                }
            }

            // Third, the upper outer rim
            if appear_phase > 0.0 {
                result = circle_eclipse(rxy, r.z, h_out, rdisc2, tani, &mut phase);
                if result == Circle::Crossing {
                    appear_phase = appear_phase.min(phase);
                } else if result != Circle::Inside {
                    appear_phase = -1.0;
                }
            }
        } else if r.z < h_in {
            // Points inside hole in middle of disc. Have to consider just two circles

            // First, the upper inner rim
            result = circle_eclipse(rxy, r.z, h_in, rdisc1, tani, &mut phase);
            if result == Circle::Inside {
                appear_phase = 0.5;
            } else if result == Circle::Crossing {
                appear_phase = phase;
            }

            // Second, the upper outer rim
            if appear_phase > 0.0 {
                result = circle_eclipse(rxy, r.z, h_out, rdisc2, tani, &mut phase);
                if result == Circle::Crossing {
                    appear_phase = appear_phase.min(phase);
                } else if result != Circle::Inside {
                    appear_phase = -1.0;
                }
            }
        }
    }

    // Here is the central phase
    let phi0: f64 = r.y.atan2(-r.x) / TAU;

    if appear_phase <= 0.0 {
        ingress = phi0 - eclipse_phase;
        ingress -= ingress.floor();
        egress = ingress + 2.0 * eclipse_phase;
        temp.push((ingress, egress));
    } else if appear_phase < eclipse_phase {
        ingress = phi0 - eclipse_phase;
        ingress -= ingress.floor();
        egress = ingress + (eclipse_phase - appear_phase);
        temp.push((ingress, egress));
        ingress = phi0 + appear_phase;
        ingress -= ingress.floor();
        egress = ingress + (eclipse_phase - appear_phase);
        temp.push((ingress, egress));
    }

    Ok(temp)
}

pub fn circle_eclipse(
    rxy: f64,
    z: f64,
    zcirc: f64,
    radius: f64,
    tani: f64,
    phase: &mut f64,
) -> Circle {
    // point above circle
    if z >= zcirc {
        return Circle::Above;
    }
    let rcone: f64 = tani * (zcirc - z);

    // line-of-sight always outside the circle
    if rcone >= rxy + radius {
        return Circle::Outside;
    }

    // line-of-sight circle separate from the circle
    if rxy >= rcone + radius {
        return Circle::Separate;
    }

    // line-of-sight always outside the circle
    if rxy + rcone <= radius {
        return Circle::Inside;
    }

    // crossing case
    *phase = cut_phase(rxy, rcone, radius);

    Circle::Crossing
}

pub fn cut_phase(rxy: f64, rcone: f64, radius: f64) -> f64 {
    // Temporary checks
    if rxy + rcone <= radius {
        panic!("rxy + rcone <= radius");
    }
    if rxy >= radius + rcone {
        panic!("rxy >= radius + rcone");
    }
    if rcone >= radius + rxy {
        panic!("rcone >= radius + rxy");
    }

    ((rxy * rxy + rcone * rcone - radius * radius) / (2.0 * rcone * rxy)).acos() / TAU
}

#[cfg(test)]
mod tests {
    use super::*;

    // Eclipse windows must be centred on the phase at which the disc lies
    // between the point and the observer: phi0 = atan2(y, -x) in cycles.
    // These pin a regression where phi0 was accidentally computed in
    // radians, scattering the windows across arbitrary phases.

    fn centre_of(iangle: f64, r: &Vec3) -> f64 {
        let eclipses = disc_eclipse(iangle, 0.05, 0.35, 1.5, 0.02, r).unwrap();
        assert_eq!(eclipses.len(), 1, "expected a single eclipse window");
        let (ingress, egress) = eclipses[0];
        assert!(ingress < egress);
        (ingress + egress) / 2.0 % 1.0
    }

    #[test]
    fn window_centred_on_half_for_point_beyond_disc_on_x_axis() {
        let centre = centre_of(82.0, &Vec3::new(1.0, 0.0, -0.15));
        assert!((centre - 0.5).abs() < 1e-6, "centre = {centre}");
    }

    #[test]
    fn window_centred_on_quarter_for_point_beyond_disc_on_y_axis() {
        let centre = centre_of(82.0, &Vec3::new(0.0, 1.0, -0.15));
        assert!((centre - 0.25).abs() < 1e-6, "centre = {centre}");
    }

    #[test]
    fn no_eclipse_for_point_above_disc() {
        let eclipses =
            disc_eclipse(82.0, 0.05, 0.35, 1.5, 0.02, &Vec3::new(1.0, 0.0, 0.05)).unwrap();
        assert!(eclipses.is_empty());
    }

    #[test]
    fn beta_of_exactly_one_is_allowed() {
        assert!(disc_eclipse(82.0, 0.05, 0.35, 1.0, 0.02, &Vec3::new(1.0, 0.0, -0.15)).is_ok());
    }
}
