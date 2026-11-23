use crate::errors::RocheError;
use crate::{Star, Vec3, brightspot_position, ingress_egress};
use crate::{fblink, set_earth_iangle, x_l1};
use pyo3::prelude::*;
use std::f64::consts::{FRAC_PI_2, TAU};


///
/// Computes scaled radius of white dwarf, r1 = R1/a given
/// the mass ratio, orbital inclination and phase width of the
/// white dwarf's egress (or ingress).
/// 
/// Arguments:
/// 
/// * `q`: mass ratio M2/M1
/// * `iangle`: orbital inclination (degrees)
/// * `dpwd`: phase-width of WD ingress/egress feature
/// * `ntheta`: number of points on limb of white dwarf when using wdphases during this routine.
/// * `dr`: tolerance on scaled radius
/// * `rmax`: maximum scaled radius to assume
/// 
/// Returns:
/// 
/// * scaled radius of WD
/// 
#[pyfunction]
#[pyo3(signature = (q, iangle, dpwd, ntheta=100, dr=1.0e-5, rmax=0.1))]
pub fn wdradius(q: f64, iangle: f64, dpwd: f64, ntheta: i32, dr: f64, rmax: f64) -> Result<f64, RocheError> {
    let mut r1: f64;
    let mut r1_lo: f64 = 0.0;
    let mut r1_hi: f64 = rmax;
    let mut phi3: f64;
    let mut phi4: f64;
    let mut dp: f64;
    while r1_hi - r1_lo > dr {
        r1 = (r1_lo + r1_hi) / 2.0;
        (phi3, phi4) = wdphases(q, iangle, r1, -1.0, ntheta)?;
        dp = phi4 - phi3;
        if dp > dpwd {r1_hi = r1} else {r1_lo = r1}
    }
    Ok((r1_lo + r1_hi) / 2.0)
}


///
/// phi3, phi4 = wdphases(q, iangle, r1, r2=-1, ntheta=200)
///
/// Returns the third and fourth contact phases of the white dwarf.
///
/// q      -- mass ratio = M2/M1
/// iangle -- orbital inclination, degrees
/// r1     -- scaled white dwarf radius = R1/a
/// r2     -- scaled secondary radius, < 0 for Roche lobe filling
/// ntheta -- number of angles to compute at the limb of white dwarf.
///           (used over quadrants)
///
/// The routine searches points equally-spaced at quadrants of the limb
/// of the white dwarf to determine the contact phases. It will fail if
/// there is no eclipse at all by raising a RocheError. For partial eclipses
/// there will be a valid 'fourth' contact (marking the end of eclipse still)
/// but the third contact will be set = -1.
///
#[pyfunction]
#[pyo3(signature = (q, iangle, r1, r2=-1.0, ntheta=200))]
pub fn wdphases(
    q: f64,
    iangle: f64,
    r1: f64,
    r2: f64,
    ntheta: i32,
) -> Result<(f64, f64), RocheError> {
    let ffac: f64 = if r2 <= 0.0 {
        1.0
    } else {
        r2 / (1.0 - x_l1(q)?)
    };
    // fourth contact
    let mut phi4lo: f64 = 0.0;
    let mut phi4hi: f64 = 0.25;
    let mut phi4: f64 = 0.0;

    if !eclipsed_4(q, iangle, phi4lo, r1, ffac, ntheta)? {
        let message = format!("no eclipse at all for q={}, i={}, r1={}", q, iangle, r1);
        return Err(RocheError::WdphasesError(message));
    }

    while phi4hi - phi4lo > r1 / (ntheta as f64) / 10.0 {
        phi4 = (phi4lo + phi4hi) / 2.0;
        if eclipsed_4(q, iangle, phi4, r1, ffac, ntheta)? {
            phi4lo = phi4;
        } else {
            phi4hi = phi4;
        }
    }

    // third contact
    let mut phi3lo: f64 = 0.0;
    let mut phi3hi: f64 = 0.25;
    let mut phi3: f64 = 0.0;

    if uneclipsed_3(q, iangle, phi3lo, r1, ffac, ntheta)? {
        return Ok((-1.0, phi4));
    }

    while phi3hi - phi3lo > r1 / (ntheta as f64) / 10.0 {
        phi3 = (phi3lo + phi3hi) / 2.0;
        if uneclipsed_3(q, iangle, phi3, r1, ffac, ntheta)? {
            phi3hi = phi3;
        } else {
            phi3lo = phi3;
        }
    }

    Ok((phi3, phi4))
}

///
/// Returns x, y vectors which define the projected limb of white dwarf
///when viewed at orbital inclination = iangle and orbital phase = phase
///
fn xyv(iangle: f64, r1: f64, phase: f64) -> (Vec3, Vec3) {
    let (sinp, cosp) = (TAU * phase).sin_cos();
    let x = Vec3::new(-r1 * sinp, r1 * cosp, 0.0);
    let iangle_radians = iangle.to_radians();
    let (sini, cosi) = iangle_radians.sin_cos();
    let y = Vec3::new(-r1 * cosi * cosp, -r1 * cosi * sinp, r1 * sini);
    (x, y)
}

///
/// Says whether any of the upper-left quadrant of the WD is uneclipsed at
/// phase = phase 'any' means all of the ntheta points computed uniformly
/// around the quadrant. This can be used to define the 3rd contact
///
fn uneclipsed_3(
    q: f64,
    iangle: f64,
    phase: f64,
    r1: f64,
    ffac: f64,
    ntheta: i32,
) -> Result<bool, RocheError> {
    let (x, y) = xyv(iangle, r1, phase);
    let mut theta: f64;
    let mut v: Vec3;
    let mut sint: f64;
    let mut cost: f64;
    for i in 0..ntheta {
        theta = FRAC_PI_2 * (i as f64 / (ntheta as f64 - 1.0));
        (sint, cost) = theta.sin_cos();
        v = -x * cost + y * sint;
        if !fblink(
            q,
            Star::Secondary,
            1.0,
            ffac,
            1.0e-5,
            &set_earth_iangle(iangle, phase),
            &v,
        )? {
            return Ok(true);
        }
    }

    Ok(false)
}

///
/// Says whether any of lower-right quadrant of the WD is eclipsed at
/// phase = phase 'Any' means any of ntheta points computed uniformly around the quadrant.
/// This can be used to define the 4th contact
///
fn eclipsed_4(
    q: f64,
    iangle: f64,
    phase: f64,
    r1: f64,
    ffac: f64,
    ntheta: i32,
) -> Result<bool, RocheError> {
    let (x, y) = xyv(iangle, r1, phase);
    let mut theta: f64;
    let mut v: Vec3;
    let mut sint: f64;
    let mut cost: f64;
    for i in 0..ntheta {
        theta = FRAC_PI_2 * (i as f64 / (ntheta as f64 - 1.0));
        (sint, cost) = theta.sin_cos();
        v = x * cost - y * sint;
        if fblink(
            q,
            Star::Secondary,
            1.0,
            ffac,
            1.0e-5,
            &set_earth_iangle(iangle, phase),
            &v,
        )? {
            return Ok(true);
        }
    }

    Ok(false)
}

#[pyfunction]
// #[pyo3(signature = (q, iangle, rbs))]
pub fn bsphases(q: f64, iangle: f64, rbs: f64) -> Result<(f64, f64), RocheError> {
    let r = brightspot_position(q, rbs, 1.0e-7, 1.0e-2)?;
    let mut ingress: f64 = 0.0;
    let mut egress: f64 = 0.0;
    let eclipse = ingress_egress(
        q,
        Star::Secondary,
        1.0,
        1.0,
        iangle,
        1.0e-7,
        &r,
        &mut ingress,
        &mut egress,
    )?;
    if !eclipse {
        return Err(RocheError::WdphasesError(
            "point is not eclipsed".to_string(),
        ));
    }
    Ok((ingress - 1.0, egress - 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wdradius_test() -> Result<(), RocheError> {
        assert_eq!(wdradius(0.2, 87.0, 0.008, 100, 1.0e-5, 0.1)?, 0.026266479492187505);
        Ok(())
    }

    #[test]
    fn wdphases_test() -> Result<(), RocheError> {
        // Values from trm.roche.wdphases
        assert_eq!(
            wdphases(0.2, 90.0, 0.015, 0.20, 200)?,
            (0.027637481689453125, 0.032169342041015625)
        );
        assert_eq!(
            wdphases(0.2, 85.0, 0.015, 0.20, 200)?,
            (0.023677825927734375, 0.029010772705078125)
        );
        assert_eq!(
            wdphases(0.2, 80.0, 0.015, 0.20, 200)?,
            (-1.0, 0.015842437744140625)
        );
        assert!(wdphases(0.2, 60.0, 0.015, 0.20, 200).is_err());
        Ok(())
    }

    #[test]
    fn bsphases_test() -> Result<(), RocheError> {
        // Values from trm.roche.bsphases
        let (phi_in, phi_eg) = bsphases(0.2, 90.0, 0.2)?;
        assert!((phi_in - -0.016291638617189408).abs() < 1.0e-7);
        assert!((phi_eg - 0.07258765600962591).abs() < 1.0e-7);

        let (phi_in, phi_eg) = bsphases(0.2, 85.0, 0.2)?;
        assert!((phi_in - -0.013903522665196566).abs() < 1.0e-7);
        assert!((phi_eg - 0.07018328081120417).abs() < 1.0e-7);

        assert!(bsphases(0.2, 60.0, 0.2).is_err());
        Ok(())
    }
}
