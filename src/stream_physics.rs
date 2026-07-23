use crate::errors::RocheError;
use crate::x_l1;
use crate::{Vec3, vel_transform};
use bulirsch::{self, Integrator};
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};

///
/// strinit sets a particle just inside the L1 point with the
/// correct velocity as given in Lubow and Shu.
///
/// Arguments:
///
/// * `q`: mass ratio = M2/M1
///
/// Returns:
///
/// * start position
/// * start velocity
///
#[pyfunction]
pub fn strinit(q: f64) -> Result<(Vec3, Vec3), RocheError> {
    const SMALL: f64 = 1.0e-5;
    let rl1: f64 = x_l1(q)?;
    let mu: f64 = q / (1.0 + q);
    let a: f64 = (1.0 - mu) / rl1.powi(3) + mu / (1.0 - rl1).powi(3);
    let lambda1: f64 = (((a - 2.0) + (a * (9.0 * a - 8.0)).sqrt()) / 2.0).sqrt();
    let m1: f64 = (lambda1 * lambda1 - 2.0 * a - 1.0) / 2.0 / lambda1;

    let r: Vec3 = Vec3::new(rl1 - SMALL, -m1 * SMALL, 0.0);
    let v: Vec3 = Vec3::new(-lambda1 * SMALL, -lambda1 * m1 * SMALL, 0.0);

    Ok((r, v))
}

///
/// stream works by integrating the equations of motion for the Roche
/// potential using Burlisch-Stoer integration. Every time the distance
/// from the last point exceeds step, it interpolates and stores a new
/// point. This allows one not to spend loads of points on regions where
/// nothing is happening.
///
/// Arguments:
///
/// * `q`:    mass ratio = M2/M1. Stream flows from star 2 to 1.
/// * `step`: step between points (units of separation).
/// * `n_points`:    number of points to compute.
///
/// Returns:
///
/// * `x`:    array of x values returned.
/// * `y`:    array of y values returned.
///
pub fn stream(q: f64, step: f64, n_points: usize) -> Result<(Vec<f64>, Vec<f64>), RocheError> {
    if n_points < 2 {
        return Err(RocheError::ParameterError(
            "Need at least 2 points in the stream.".to_string(),
        ));
    }

    if step <= 0.0 || step > 1.0 {
        return Err(RocheError::ParameterError(
            "Step size must be between 0.0 and 1.0".to_string(),
        ));
    }

    if q <= 0.0 {
        return Err(RocheError::ParameterError("q = {} <= 0".to_string()));
    }

    let mut x_arr: Vec<f64> = vec![];
    let mut y_arr: Vec<f64> = vec![];

    // Initialise stream
    let rl1: f64 = x_l1(q)?;
    let (mut r, mut v) = strinit(q)?;

    // Store L1 as first point
    x_arr.push(rl1);
    y_arr.push(0.0);

    let mut lp: usize = 0;

    // Store interpolation between L1 and initial point if
    // step has been set small enough

    let mut dist: f64 = (r.x - rl1).hypot(r.y);

    let frac: f64;

    if dist > step {
        frac = step / dist;
        x_arr.push(rl1 + (r.x - rl1) * frac);
        y_arr.push(r.y * frac);
        lp += 1;
    }

    // set up Bulirsch-Stoer integrator
    let system = OrbitalSystem { q };
    let mut integrator = Integrator::default()
        .with_abs_tol(1.0e-8)
        .with_rel_tol(1.0e-8)
        .into_adaptive();
    // Initialise arrays
    let mut y = ndarray::array![r.x, r.y, r.z, v.x, v.y, v.z];
    let mut y_next = ndarray::Array::zeros(y.raw_dim());

    let mut delta_t: f64 = 1.0e-3;
    let smax: f64 = (1.0e-3_f64).min(step / 2.0);

    let mut vel: f64;
    while lp < n_points - 1 {
        integrator
            .step(&system, delta_t, y.view(), y_next.view_mut())
            .unwrap();
        y.assign(&y_next);

        r.set(y[0], y[1], y[2]);
        v.set(y[3], y[4], y[5]);
        dist = (r.x - x_arr[lp]).hypot(r.y - y_arr[lp]);
        if dist > step {
            let frac: f64 = step / dist;
            x_arr.push(x_arr[lp] + (r.x - x_arr[lp]) * frac);
            y_arr.push(y_arr[lp] + (r.y - y_arr[lp]) * frac);
            lp += 1;
        }
        vel = v.x.hypot(v.y);
        delta_t = (smax / vel).min(delta_t);
    }

    Ok((x_arr, y_arr))
}

///
/// stream works by integrating the equations of motion for the Roche
/// potential using Burlisch-Stoer integration. Every time the distance
/// from the last point exceeds step, it interpolates and stores a new
/// point. This allows one not to spend loads of points on regions where
/// nothing is happening.
///
/// Arguments:
///
/// * `q`:    mass ratio = M2/M1. Stream flows from star 2 to 1.
/// * `step`: step between points (units of separation).
/// * `n_points`:    number of points to compute.
///
/// Returns:
///
/// * `x`:    array of x values returned.
/// * `y`:    array of y values returned.
///
#[pyfunction]
#[pyo3(name = "stream", signature = (q, step, n_points=200))]
pub fn stream_py(py: Python, q: f64, step: f64, n_points: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (x_arr, y_arr) = stream(q, step, n_points)?;
    Ok((x_arr.into_pyarray(py).unbind(), y_arr.into_pyarray(py).unbind()))
}

///
/// strmnx finds the next point at which stream is closest or furthest
/// from primary.
///
/// Arguments:
///
/// * `q`: mass ratio = M2/M1
/// * `r`: initial and final position
/// * `v`: initial and final velocity
/// * `acc`: accuracy in time to locate minimum/maximum.
///
///
pub fn strmnx(q: f64, r: &mut Vec3, v: &mut Vec3, acc: f64) -> Result<(), RocheError> {
    let mut dir: f64;
    let mut lo: f64;
    let mut hi: f64;
    let mut ro: Vec3 = *r;
    let mut vo: Vec3 = *v;

    let mut delta_t: f64 = 1.0e-2;

    // Store initial direction
    dir = r.dot(v);
    let dir1: f64 = dir;

    // set up Bulirsch-Stoer integrator
    let system = OrbitalSystem { q };
    let mut integrator = Integrator::default()
        .with_abs_tol(1.0e-8)
        .with_rel_tol(1.0e-8)
        .into_adaptive();
    // Initialise arrays
    let mut y = ndarray::array![r.x, r.y, r.z, v.x, v.y, v.z];
    let mut y_next = ndarray::Array::zeros(y.raw_dim());
    let mut yo = y.clone();

    while (dir > 0.0 && dir1 > 0.0) || (dir < 0.0 && dir1 < 0.0) {
        ro = *r;
        vo = *v;
        yo = y.clone();
        integrator
            .step(&system, delta_t, y.view(), y_next.view_mut())
            .unwrap();
        y.assign(&y_next);
        r.set(y[0], y[1], y[2]);
        v.set(y[3], y[4], y[5]);
        dir = r.dot(v);
    }

    //   Now refine by reinitialising and binary chopping until
    //   close enough to requested radius.

    lo = 0.0;
    hi = delta_t;
    while (hi - lo).abs() > acc {
        delta_t = (lo + hi) / 2.0;
        y = yo.clone();
        *r = ro;
        *v = vo;
        integrator
            .step(&system, delta_t, y.view(), y_next.view_mut())
            .unwrap();
        y.assign(&y_next);

        r.set(y[0], y[1], y[2]);
        v.set(y[3], y[4], y[5]);
        dir = r.dot(v);
        if (dir > 0.0 && dir1 < 0.0) || (dir < 0.0 && dir1 > 0.0) {
            hi = delta_t;
        } else {
            lo = delta_t;
        }
    }

    Ok(())
}

// wrapper for python library, avoiding mutable references

///
/// Calculates position & velocity of n-th turning point of stream.
/// x,y,vx1,vy1,vx2,vy2 = strmnx(q, n=1, acc=1.e-7), q = M2/M1.
/// Two sets of velocities are reported, the first for the pure stream,
/// the second for the disk at that point.
///
/// Arguments:
///
/// * `q`: mass ratio = M2/M1
/// * `n`: turning point number
/// * `acc`: accuracy in time to locate minimum/maximum.
///
/// Returns:
/// (x, y, vx1, vy1, vx2, vy2)
///
#[pyfunction]
#[pyo3(name = "strmnx")]
#[pyo3(signature = (q, n=1, acc=1.0e-7))]
pub fn strmnx_wrapper(
    q: f64,
    n: usize,
    acc: f64,
) -> Result<(f64, f64, f64, f64, f64, f64), RocheError> {
    let (mut r, mut v) = strinit(q)?;
    for _ in 0..n {
        strmnx(q, &mut r, &mut v, acc)?
    }
    let (tvx1, tvy1) = vel_transform(q, 1, r.x, r.y, v.x, v.y)?;
    let (tvx2, tvy2) = vel_transform(q, 2, r.x, r.y, v.x, v.y)?;
    Ok((r.x, r.y, tvx1, tvy1, tvx2, tvy2))
}

///
/// streamr works by integrating the equations of motion for the Roche
/// potential using Burlisch-Stoer integration. It stops when the stream
/// reaches a target radius or a minimum radius, whichever is the larger.
///
/// Arguments:
///
/// * `q`: mass ratio = M2/M1. Stream flows from star 2 to 1.
/// * `rad`: Radius to aim for. If this is less than the minimum, the stream will stop at the minimum
/// * `n_points`: number of points to compute.
///
/// Results:
///
/// * `x`: array of x values returned.
/// * `y`: array of y values returned.
///
pub fn streamr(q: f64, rad: f64, n_points: usize) -> Result<(Vec<f64>, Vec<f64>), RocheError> {
    if n_points < 2 {
        return Err(RocheError::ParameterError(
            "Need at least 2 points in the stream.".to_string(),
        ));
    }

    if q <= 0.0 {
        return Err(RocheError::ParameterError("q = {} <= 0".to_string()));
    }

    const EPS: f64 = 1.0e-8;

    let mut x_arr: Vec<f64> = vec![];
    let mut y_arr: Vec<f64> = vec![];

    // Initialise stream
    let rl1: f64 = x_l1(q)?;
    let (mut r, mut v) = strinit(q)?;
    let rs = r;
    let vs = v;
    strmnx(q, &mut r, &mut v, EPS)?;
    let rmin = if r.length() > rad { r.length() } else { rad };

    r = rs;
    v = vs;
    x_arr.push(r.x);
    y_arr.push(r.y);
    let mut rnext: f64;
    for i in 1..n_points {
        rnext = rl1 + (rmin - rl1) * (i as f64) / (n_points as f64 - 1.0);
        stradv(q, &mut r, &mut v, rnext, 1.0e-6, 1.0e-4)?;
        x_arr.push(r.x);
        y_arr.push(r.y);
    }

    Ok((x_arr, y_arr))
}

///
/// streamr works by integrating the equations of motion for the Roche
/// potential using Burlisch-Stoer integration. It stops when the stream
/// reaches a target radius or a minimum radius, whichever is the larger.
///
/// Arguments:
///
/// * `q`: mass ratio = M2/M1. Stream flows from star 2 to 1.
/// * `rad`: Radius to aim for. If this is less than the minimum, the stream will stop at the minimum
/// * `n_points`: number of points to compute.
///
/// Results:
///
/// * `x`: array of x values returned.
/// * `y`: array of y values returned.
///
#[pyfunction]
#[pyo3(name = "streamr", signature = (q, rad, n_points=200))]
pub fn streamr_py(py: Python, q: f64, rad: f64, n_points: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (x_arr, y_arr) = streamr(q, rad, n_points)?;
    Ok((x_arr.into_pyarray(py).unbind(), y_arr.into_pyarray(py).unbind()))
}

///
/// stradv advances a particle of given position and velocity until
/// it reaches a specified radius. It then returns with updated position and
/// velocity. It is up to the user not to request a value that cannot be reached.
///
/// Arguments:
///
/// * `q`:    mass ratio = M2/M1
/// * `r`:    Initial and final position
/// * `v`:    Initial and final velocity
/// * `rad`:  Radius to aim for
/// * `acc`:  Accuracy with which to place output point at rad.
/// * `smax`: Largest time step allowed. It is possible that the
///   routine could take such a large step that it misses
///   the point when the stream is inside the requested
///   radius. This allows one to control this. Typical
///   value = 1.e-3.
///
/// Returns:
///
/// * time step taken
///
pub fn stradv(q: f64, r: &mut Vec3, v: &mut Vec3, rad: f64, acc: f64, smax: f64) -> Result<f64, RocheError> {
    const TMAX: f64 = 10.0;
    let t_next: f64 = 1.0e-2;

    let mut time: f64 = 0.0;

    // let to: f64;
    let mut ro = *r;
    let mut vo = *v;

    // Store initial radius
    let rinit: f64 = r.length();
    let mut rnow: f64 = rinit;

    // set up Bulirsch-Stoer integrator
    let system = OrbitalSystem { q };
    let mut integrator = Integrator::default()
        .with_abs_tol(1.0e-8)
        .with_rel_tol(1.0e-8)
        .into_adaptive();
    // Initialise arrays
    let mut y = ndarray::array![r.x, r.y, r.z, v.x, v.y, v.z];
    let mut y_next = ndarray::Array::zeros(y.raw_dim());

    let mut yo = y.clone();
    let mut delta_t = t_next.min(smax);
    // Step until radius crossed
    while (rinit > rad && rnow > rad) || (rinit < rad && rnow < rad) {
        ro = *r;
        vo = *v;
        yo = y.clone();
        integrator
            .step(&system, delta_t, y.view(), y_next.view_mut())
            .unwrap();
        y.assign(&y_next);
        r.set(y[0], y[1], y[2]);
        v.set(y[3], y[4], y[5]);
        rnow = r.length();
        time += delta_t;

        if time > TMAX {
            return Err(RocheError::ParameterError(
                "roche::stradv taken too long without crossing given radius.".to_string(),
            ));
        }
    }

    // Now refine by reinitialising and binary chopping until
    // close enough to requested radius.

    let mut lo: f64 = 0.0;
    let mut hi: f64 = delta_t;
    let mut rlo: f64 = ro.length();
    let mut rhi: f64 = rnow;
    let to: f64 = time;

    while (rhi - rlo).abs() > acc {
        delta_t = (lo + hi) / 2.0;
        y = yo.clone();
        *r = ro;
        *v = vo;
        time = to;

        integrator
            .step(&system, delta_t, y.view(), y_next.view_mut())
            .unwrap();
        y.assign(&y_next);

        r.set(y[0], y[1], y[2]);
        v.set(y[3], y[4], y[5]);
        rnow = r.length();

        if (rhi > rad && rnow > rad) || (rhi < rad && rnow < rad) {
            rhi = rnow;
            hi = delta_t;
        } else {
            rlo = rnow;
            lo = delta_t;
        }
    }

    Ok(time)
}

// wrapper for python library, avoiding mutable references

///
/// stradv advances a particle of given position and velocity until
/// it reaches a specified radius. It then returns with updated position and
/// velocity. It is up to the user not to request a value that cannot be reached.
///
/// \param q    mass ratio = M2/M1
/// \param r    Initial position
/// \param v    Initial velocity
/// \param rad  Radius to aim for
/// \param acc  Accuracy with which to place output point at rad.
/// \param smax Largest time step allowed. It is possible that the
/// routine could take such a large step that it misses
/// the point when the stream is inside the requested
/// radius. This allows one to control this. Typical
/// value = 1.e-3.
/// \returns (timestep, new position, new velocity)
///
#[pyfunction]
#[pyo3(name = "stradv")]
pub fn stradv_py(
    q: f64,
    r: &Vec3,
    v: &Vec3,
    rad: f64,
    acc: f64,
    smax: f64,
) -> Result<(f64, Vec3, Vec3), RocheError> {
    let mut r_mut = *r;
    let mut v_mut = *v;
    let timestep = stradv(q, &mut r_mut, &mut v_mut, rad, acc, smax)?;
    Ok((timestep, r_mut, v_mut))
}

///
/// rocacc calculates and returns the acceleration (in the rotating frame)
/// in a Roche potential of a particle of given position and velocity.
///
/// \param q mass ratio = M2/M1
/// \param r position, scaled in units of separation.
/// \param v velocity, scaled in units of separation
///
#[pyfunction]
pub fn rocacc(q: f64, r: &Vec3, v: &Vec3) -> (f64, f64, f64) {
    let f1: f64 = 1.0 / (1.0 + q);
    let f2: f64 = f1 * q;

    let yzsq: f64 = r.y * r.y + r.z * r.z;
    let r1sq: f64 = r.x * r.x + yzsq;
    let r2sq: f64 = (r.x - 1.0) * (r.x - 1.0) + yzsq;
    let fm1: f64 = f1 / (r1sq * (r1sq.sqrt()));
    let fm2: f64 = f2 / (r2sq * (r2sq.sqrt()));
    let fm3: f64 = fm1 + fm2;

    let x: f64 = -fm3 * r.x + fm2 + 2.0 * v.y + r.x - f2;
    let y: f64 = -fm3 * r.y - 2.0 * v.x + r.y;
    let z: f64 = -fm3 * r.z;
    (x, y, z)
}

///
/// brightspot_position runs strinit then stradv to get the coordinates of
/// of the gas stream when it reaches a given radius from the primary star.
///
/// Arguments:
///
/// * `q`:  mass ratio = M2/M1
/// * `rad`: radius from primary star
/// * `acc`: computational accuracy
/// * `smax`: maximum time step of Bulirsch-Stoer integration
///
/// Returns:
/// * `r`: Vec3 coordinates of gas stream at given radius from primary star
///
#[pyfunction]
#[pyo3(signature = (q, rad, acc=1.0e-7, smax=1.0e-2))]
pub fn brightspot_position(q: f64, rad: f64, acc: f64, smax: f64) -> Result<Vec3, RocheError> {
    let (mut r, mut v) = strinit(q)?;
    stradv(q, &mut r, &mut v, rad, acc, smax)?;

    Ok(r)
}

///
/// bspot runs strinit then stradv to get the coordinate and velocity
/// vectors of the gas stream when it reaches a given radius from the primary star.
///
/// Arguments:
///
/// * `q`:  mass ratio = M2/M1
/// * `rad`: radius from primary star
/// * `acc`: computational accuracy
/// * `smax`: maximum time step of Bulirsch-Stoer integration
///
/// Returns:
/// * `r`: Vec3 coordinates of gas stream at given radius from primary star
/// * `v`: Vec3 velocity of gas stream at given radius from primary star
///
#[pyfunction]
#[pyo3(signature = (q, rad, acc=1.0e-7, smax=1.0e-2))]
pub fn bspot(q: f64, rad: f64, acc: f64, smax: f64) -> Result<(Vec3, Vec3), RocheError> {
    let (mut r, mut v) = strinit(q)?;
    stradv(q, &mut r, &mut v, rad, acc, smax)?;

    Ok((r, v))
}

pub struct OrbitalSystem {
    pub q: f64,
}

impl bulirsch::System for OrbitalSystem {
    type Float = f64;

    fn system(
        &self,
        y: bulirsch::ArrayView1<Self::Float>,
        mut dydt: bulirsch::ArrayViewMut1<Self::Float>,
    ) {
        dydt[[0]] = y[[3]];
        dydt[[1]] = y[[4]];
        dydt[[2]] = y[[5]];
        let r = Vec3::new(y[[0]], y[[1]], y[[2]]);
        let v = Vec3::new(y[[3]], y[[4]], y[[5]]);
        (dydt[[3]], dydt[[4]], dydt[[5]]) = rocacc(self.q, &r, &v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strinit_stradv_test() -> Result<(), RocheError> {
        // Values from trm.roche.bspot
        let (mut r, mut v) = strinit(0.2)?;
        let _time: f64 = stradv(0.2, &mut r, &mut v, 0.3, 1.0e-7, 1.0e-3)?;
        assert!((r - Vec3::new(0.2660591412807423, 0.13860932478255575, 0.0)).length() < 1.0e-7);
        assert!((v - Vec3::new(-1.4769457229627583, 0.31712381217252994, 0.0)).length() < 1.0e-7);
        Ok(())
    }

    #[test]
    fn stream_test() -> Result<(), RocheError> {
        // Values from trm.roche.stream
        let (x, y) = stream(0.2, 0.01, 200)?;
        assert!((x[0] - 0.6585557).hypot(y[0] - 0.0) < 1.0e-4);
        assert!((x[50] - 0.18384902).hypot(y[50] - 0.15145306) < 1.0e-4);
        assert!((x[100] - -0.100431986).hypot(y[100] - -0.13697079) < 1.0e-4);
        assert!((x[150] - 0.21720248).hypot(y[150] - -0.4577784) < 1.0e-4);
        assert!((x[y.len() - 1] - 0.15403406).hypot(y[y.len() - 1] - 0.016731631) < 1.0e-4);
        assert!(stream(-0.2, 0.0001, 200).is_err());
        assert!(stream(0.2, 1.1, 200).is_err());
        assert!(stream(0.2, -0.1, 200).is_err());
        assert!(stream(0.2, 0.0001, 1).is_err());
        Ok(())
    }

    #[test]
    fn strmnx_test() -> Result<(), RocheError> {
        // Values from trm.roche.strmnx
        let (x, y, vx1, vy1, vx2, vy2) = strmnx_wrapper(0.2, 1, 1.0e-7)?;
        assert!(
            (x - -0.08613947462186848).hypot(y - 0.05411592729509131)
                / (-0.08613947462186848_f64).hypot(0.05411592729509131)
                < 1.0e-6
        );
        assert!(
            (vx1 - -1.9727409465489645).hypot(vy1 - -3.30679322752132)
                / (-1.9727409465489645_f64).hypot(-3.30679322752132)
                < 1.0e-6
        );
        assert!(
            (vx2 - -1.5225623467338747).hypot(vy2 - -2.5902178683586605)
                / (-1.5225623467338747_f64).hypot(-2.5902178683586605)
                < 1.0e-6
        );
        Ok(())
    }

    #[test]
    fn brightspot_position_test() -> Result<(), RocheError> {
        // Values from trm.roche.bspot
        let r = brightspot_position(0.2, 0.3, 1.0e-7, 1.0e-3)?;
        assert!((r - Vec3::new(0.2660591412807423, 0.13860932478255575, 0.0)).length() < 1.0e-7);
        Ok(())
    }

    #[test]
    fn bspot_test() -> Result<(), RocheError> {
        // Values from trm.roche.bspot
        let (r, v) = bspot(0.2, 0.3, 1.0e-7, 1.0e-3)?;
        assert!((r - Vec3::new(0.2660591412807423, 0.13860932478255575, 0.0)).length() < 1.0e-7);
        assert!((v - Vec3::new(-1.476945722613775, 0.31712381223279495, 0.0)).length() < 1.0e-6);
        Ok(())
    }
}
