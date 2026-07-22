use crate::errors::RocheError;
use crate::x_l1;
use crate::{Vec3, vel_transform, strinit, stradv, strmnx, rocacc, OrbitalSystem};
use bulirsch::{self, Integrator};
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};



///
/// vstream computes the path of the gas stream in a binary in velocity space.
/// There are a few different options for the type of stream velocities produced.
/// vstream works by integrating the equations of motion for the Roche
/// potential using Burlisch-Stoer integration. Every time the speed
/// changes by step, it interpolates and stores a new point.
///
/// The velocities are inertial frame velocities for comparison with Doppler maps.
/// 
/// Arguments:
/// 
/// * `q`: mass ratio = M2/M1. Stream flows from star 2 to 1.
/// * `step`: step between points (units of K1+K2).
/// * `n_points`: number of points to compute.
/// * `transform_type`: type of velocity, see !!ref{vtrans.html}{vtrans} for supported types.
/// 
/// Returns:
/// 
/// * `vx`: array of x velocities.
/// * `vy`: array of y velocities.
/// * `rad`: array of radii equivalent to velocities (units of a)
///
pub fn vstream(q: f64, step: f64, n_points: usize, transform_type: i32) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), RocheError> {

    if n_points < 2 {
        return Err(RocheError::ParameterError(
            "Need at least 2 points in the stream.".to_string(),
        ));
    }

    let mut vx_arr: Vec<f64> = vec![];
    let mut vy_arr: Vec<f64> = vec![];
    let mut r_arr: Vec<f64> = vec![];
    let mut frac: f64;
    let mut acc: f64;
    let mut ar_x: f64;
    let mut ar_y: f64;

    // Initialise stream
    let rl1: f64 = x_l1(q)?;
    let (mut r, mut v) = strinit(q)?;

    // Store L1 as first point
    let (mut tvx, mut tvy) = vel_transform(q, transform_type, rl1, 0.0, 0.0, 0.0)?;
    vx_arr.push(tvx);
    vy_arr.push(tvy);
    let mut lp: usize = 0;

    // Store interpolation between L1 and initial point if step
    // has been set small enough

    (tvx, tvy) = vel_transform(q, transform_type, r.x, r.y, v.x, v.y)?;
    let mut delta_velocity: f64 = (tvx - vx_arr[0]).hypot(tvy - vy_arr[0]);
    if delta_velocity > step {
        frac = step / delta_velocity;
        vx_arr.push(vx_arr[0] + (tvx - vx_arr[0]) * frac);
        vy_arr.push(vy_arr[0] + (tvy - vy_arr[0]) * frac);
        r_arr.push(r.x.hypot(r.y));
        lp += 1;
    }

    let mut delta_t: f64 = 1.0e-3;
    let smax: f64 = 1.0e-3_f64.min(step / 2.0);

    // set up Bulirsch-Stoer integrator
    let system = OrbitalSystem { q };
    let mut integrator = Integrator::default()
        .with_abs_tol(1.0e-8)
        .with_rel_tol(1.0e-8)
        .into_adaptive();
    // Initialise arrays
    let mut y = ndarray::array![r.x, r.y, r.z, v.x, v.y, v.z];
    let mut y_next = ndarray::Array::zeros(y.raw_dim());

    while lp < n_points - 1 {

        integrator
            .step(&system, delta_t, y.view(), y_next.view_mut())
            .unwrap();
        y.assign(&y_next);

        r.set(y[0], y[1], y[2]);
        v.set(y[3], y[4], y[5]);
        (tvx, tvy) = vel_transform(q, transform_type, r.x, r.y, v.x, v.y)?;
        delta_velocity = (tvx - vx_arr[lp]).hypot(tvy - vy_arr[lp]);
        if delta_velocity > step {
            frac = step / delta_velocity;
            vx_arr.push(vx_arr[lp] + (tvx - vx_arr[lp]) * frac);
            vy_arr.push(vy_arr[lp] + (tvy - vy_arr[lp]) * frac);
            r_arr.push(r.x.hypot(r.y));
            lp += 1;
        }
        (ar_x, ar_y, _) = rocacc(q, &r, &v);
        acc = ar_x.hypot(ar_y);
        delta_t = delta_t.min(smax/acc);
    }

    Ok((vx_arr, vy_arr, r_arr))

}

pub fn vstream_reg(q: f64, step: f64, n_points: usize, transform_type: i32) -> Result<(Vec<f64>, Vec<f64>), RocheError> {

    const TLOC: f64 = 1.0e-8;
    const RLOC: f64 = 1.0e-8;

    if n_points < 2 {
        return Err(RocheError::ParameterError(
            "Need at least 2 points in the stream.".to_string(),
        ));
    }

    let mut rm: Vec3;
    let mut vm: Vec3;
    let mut r_end: f64;
    let mut vx_arr: Vec<f64> = vec![];
    let mut vy_arr: Vec<f64> = vec![];

    let rl1: f64 = x_l1(q)?;

    let (mut tvx, mut tvy) = vel_transform(q, transform_type, rl1, 0.0, 0.0, 0.0)?;
    vx_arr.push(tvx);
    vy_arr.push(tvy);

    let mut np: usize = 1;
    let mut r_next: f64 = rl1 * (1.0 - step);
    let mut r_decreasing: bool = true;

    // Initialise stream
    let (mut r, mut v) = strinit(q)?;

    while np < n_points {

        // Advance one step
        stradv(q, &mut r, &mut v, r_next, RLOC, 1.0e-3)?;
        (tvx, tvy) = vel_transform(q, transform_type, r.x, r.y, v.x, v.y)?;
        vx_arr.push(tvx);
        vy_arr.push(tvy);
        np += 1;
        r_next = if r_decreasing {r_next - rl1 * step} else {r_next + rl1 * step};

        // Locate and store next turning point
        rm = r;
        vm = v;
        strmnx(q, &mut rm, &mut vm, TLOC)?;
        r_end = rm.length();

        // Loop over all radii wanted before next turning point
        while np < n_points && ((r_decreasing && r_next > r_end) || (!r_decreasing && r_next < r_end)) {
            stradv(q, &mut r, &mut v, r_next, RLOC, 1.0e-3)?;
            (tvx, tvy) = vel_transform(q, transform_type, r.x, r.y, v.x, v.y)?;
            vx_arr.push(tvx);
            vy_arr.push(tvy);
            np += 1;
            r_next = if r_decreasing {r_next - rl1 * step} else {r_next + rl1 * step};
        }

        // Change direction of search and move it to start at turning point
        r = rm;
        v = vm;
        r_decreasing = !r_decreasing;
        r_next = if r_decreasing {r_next - rl1 * step} else {r_next + rl1 * step};
    }
    
    Ok((vx_arr, vy_arr))
}


#[pyfunction]
#[pyo3(name = "vstream")]
#[pyo3(signature = (q, step=0.01, n_points=60, transform_type=1))]
pub fn vstream_reg_py(py: Python, q: f64, step: f64, n_points: usize, transform_type: i32) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (x_arr, y_arr) = vstream_reg(q, step, n_points, transform_type)?;
    Ok((x_arr.into_pyarray(py).unbind(), y_arr.into_pyarray(py).unbind()))
}
