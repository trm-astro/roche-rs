use crate::errors::RocheError;
use pyo3::prelude::*;

///
/// x_l1 calculates the x coordinate of the L1 Lagrangian point in terms
/// of the orbital separation of the two stars which have a mass ratio, q.
/// It works by solving for the root of a quintic polynomial by Newton-Raphson
/// iteration. L1 is the point in between the two stars and so will be between
/// 0 and 1.
///
/// * `q`: mass ratio = M2/M1
///
#[pyfunction]
#[pyo3(name = "xl1")]
pub fn x_l1(q: f64) -> Result<f64, RocheError> {
    const NMAX: i32 = 1000;
    const EPS: f64 = 1.0e-12;

    if q <= 0.0 {
        let message = format!("q = {} <= 0", q);
        return Err(RocheError::ParameterError(message));
    }

    let mu: f64 = q / (1.0 + q);
    let a1: f64 = -1.0 + mu;
    let a2: f64 = 2.0 - 2. * mu;
    let a3: f64 = -1.0 + mu;
    let a4: f64 = 1.0 + 2. * mu;
    let a5: f64 = -2.0 - mu;
    let a6: f64 = 1.0;
    let d1: f64 = 1.0 * a2;
    let d2: f64 = 2.0 * a3;
    let d3: f64 = 3.0 * a4;
    let d4: f64 = 4.0 * a5;
    let d5: f64 = 5.0 * a6;

    let mut n: i32 = 0;
    let mut xold: f64 = 0.;
    let mut x: f64 = 1. / (1.0 + q);
    let mut f: f64;
    let mut df: f64;
    while n < NMAX && (x - xold).abs() > EPS * x.abs() {
        xold = x;
        f = x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1;
        df = x * (x * (x * (x * d5 + d4) + d3) + d2) + d1;
        x -= f / df;
        n += 1;
    }
    Ok(x)
}

///
/// x_l1_1 calculates the x coordinate of the L1 Lagrangian point in terms
/// of the orbital separation of the two stars which have a mass ratio, q,
/// accounting for asynchronous rotation of the primary.
/// It works by solving for the root of a quintic polynomial by Newton-Raphson
/// iteration. L1 is the point in between the two stars and so will be between
/// 0 and 1.
///
///  * `q`: mass ratio = M2/M1
///  * `spin`
///
#[pyfunction]
#[pyo3(name = "xl11")]
pub fn x_l1_1(q: f64, spin: f64) -> Result<f64, RocheError> {
    const NMAX: i32 = 1000;
    const EPS: f64 = 1.0e-12;

    if q <= 0.0 {
        let message = format!("q = {} <= 0", q);
        return Err(RocheError::ParameterError(message));
    }

    let spin_squared: f64 = spin * spin;
    let mu: f64 = q / (1.0 + q);
    let a1: f64 = -1.0 + mu;
    let a2: f64 = 2.0 - 2. * mu;
    let a3: f64 = -1.0 + mu;
    let a4: f64 = spin_squared + 2. * mu;
    let a5: f64 = -2. * spin_squared - mu;
    let a6: f64 = spin_squared;
    let d1: f64 = 1. * a2;
    let d2: f64 = 2. * a3;
    let d3: f64 = 3. * a4;
    let d4: f64 = 4. * a5;
    let d5: f64 = 5. * a6;

    let mut n: i32 = 0;
    let mut xold: f64 = 0.;
    let mut x: f64 = 1. / (1.0 + q);
    let mut f: f64;
    let mut df: f64;
    while n < NMAX && (x - xold).abs() > EPS * x.abs() {
        xold = x;
        f = x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1;
        df = x * (x * (x * (x * d5 + d4) + d3) + d2) + d1;
        x -= f / df;
        n += 1;
    }
    Ok(x)
}

///
/// x_l1_2 calculates the x coordinate of the L1 Lagrangian point in terms
/// of the orbital separation of the two stars which have a mass ratio, q,
/// accounting for asynchronous rotation of the secondary.
/// It works by solving for the root of a quintic polynomial by Newton-Raphson
/// iteration. L1 is the point in between the two stars and so will be between
/// 0 and 1.
///
///  * `q`: mass ratio = M2/M1
///  * `spin`
///
#[pyfunction]
#[pyo3(name = "xl12")]
pub fn x_l1_2(q: f64, spin: f64) -> Result<f64, RocheError> {
    const NMAX: i32 = 1000;
    const EPS: f64 = 1.0e-12;

    if q <= 0.0 {
        let message = format!("q = {} <= 0", q);
        return Err(RocheError::ParameterError(message));
    }

    let spin_squared: f64 = spin * spin;
    let mu: f64 = q / (1.0 + q);
    let a1: f64 = -1.0 + mu;
    let a2: f64 = 2.0 - 2. * mu;
    let a3: f64 = -spin_squared + mu;
    let a4: f64 = 3. * spin_squared + 2. * mu - 2.;
    let a5: f64 = 1.0 - mu - 3. * spin_squared;
    let a6: f64 = spin_squared;
    let d1: f64 = 1. * a2;
    let d2: f64 = 2. * a3;
    let d3: f64 = 3. * a4;
    let d4: f64 = 4. * a5;
    let d5: f64 = 5. * a6;

    let mut n: i32 = 0;
    let mut xold: f64 = 0.;
    let mut x: f64 = 1. / (1.0 + q);
    let mut f: f64;
    let mut df: f64;
    while n < NMAX && (x - xold).abs() > EPS * x.abs() {
        xold = x;
        f = x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1;
        df = x * (x * (x * (x * d5 + d4) + d3) + d2) + d1;
        x -= f / df;
        n += 1;
    }
    Ok(x)
}

///
/// x_l2 calculates the x coordinate of the L2 Lagrangian point in terms
/// of the orbital separation of the two stars which have a mass ratio, q,
/// accounting for asynchronous rotation of the primary.
/// It works by solving for the root of a quintic polynomial by Newton-Raphson
/// iteration. L2 is the point on the side of the secondary opposite the primary,
/// ands so x_l2 > 1.
///
///  * `q`: mass ratio = M2/M1
///
#[pyfunction]
#[pyo3(name = "xl2")]
pub fn x_l2(q: f64) -> Result<f64, RocheError> {
    const NMAX: i32 = 1000;
    const EPS: f64 = 1.0e-12;

    if q <= 0.0 {
        let message = format!("q = {} <= 0", q);
        return Err(RocheError::ParameterError(message));
    }

    let mu = q / (1.0 + q);
    let a1 = -1.0 + mu;
    let a2 = 2.0 - 2. * mu;
    let a3 = -1.0 - mu;
    let a4 = 1.0 + 2. * mu;
    let a5 = -2.0 - mu;
    let a6 = 1.;
    let d1 = 1. * a2;
    let d2 = 2. * a3;
    let d3 = 3. * a4;
    let d4 = 4. * a5;
    let d5 = 5. * a6;

    let mut n: i32 = 0;
    let mut xold: f64 = 0.;
    let mut x: f64 = 1.5;
    let mut f: f64;
    let mut df: f64;
    while n < NMAX && (x - xold).abs() > EPS * x.abs() {
        xold = x;
        f = x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1;
        df = x * (x * (x * (x * d5 + d4) + d3) + d2) + d1;
        x -= f / df;
        n += 1;
    }
    Ok(x)
}

///
/// x_l3 calculates the x coordinate of the L3 Lagrangian point in terms
/// of the orbital separation of the two stars which have a mass ratio, q,
/// accounting for asynchronous rotation of the primary.
/// It works by solving for the root of a quintic polynomial by Newton-Raphson
/// iteration. L3 is the point on the side of the Primary opposite the secondary,
/// ands so x_l3 < 0.
///
///  * `q`: mass ratio = M2/M1
///
#[pyfunction]
#[pyo3(name = "xl3")]
pub fn x_l3(q: f64) -> Result<f64, RocheError> {
    const NMAX: i32 = 1000;
    const EPS: f64 = 1.0e-12;

    if q <= 0.0 {
        let message = format!("q = {} <= 0", q);
        return Err(RocheError::ParameterError(message));
    }

    let mu = q / (1.0 + q);
    let a1 = 1.0 - mu;
    let a2 = -2.0 + 2. * mu;
    let a3 = 1.0 - mu;
    let a4 = 1.0 + 2. * mu;
    let a5 = -2.0 - mu;
    let a6 = 1.;
    let d1 = 1. * a2;
    let d2 = 2. * a3;
    let d3 = 3. * a4;
    let d4 = 4. * a5;
    let d5 = 5. * a6;

    let mut n: i32 = 0;
    let mut xold: f64 = 0.;
    let mut x: f64 = -1.;
    let mut f: f64;
    let mut df: f64;
    while n < NMAX && (x - xold).abs() > EPS * x.abs() {
        xold = x;
        f = x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1;
        df = x * (x * (x * (x * d5 + d4) + d3) + d2) + d1;
        x -= f / df;
        n += 1;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xl1() -> Result<(), RocheError> {
        // Values from trm.roche.xl
        assert_eq!(x_l1(0.2)?, 0.6585556789537762);
        assert!(x_l1(-0.2).is_err());
        Ok(())
    }

    #[test]
    fn xl11() -> Result<(), RocheError> {
        // Values from trm.roche.xl11
        assert_eq!(x_l1_1(0.2, 0.5)?, 0.6906163968474123);
        assert!(x_l1_1(-0.2, 0.5).is_err());
        Ok(())
    }

    #[test]
    fn xl12() -> Result<(), RocheError> {
        // Values from trm.roche.xl12
        assert_eq!(x_l1_2(0.2, 0.5)?, 0.6403753216278784);
        assert!(x_l1_2(-0.2, 0.5).is_err());
        Ok(())
    }

    #[test]
    fn xl2() -> Result<(), RocheError> {
        // Values from trm.roche.xl2
        assert_eq!(x_l2(0.2)?, 1.4380767810774817);
        assert!(x_l2(-0.2).is_err());
        Ok(())
    }

    #[test]
    fn xl3() -> Result<(), RocheError> {
        // Values from trm.roche.xl3
        assert_eq!(x_l3(0.2)?, -0.9024984464339775);
        assert!(x_l3(-0.2).is_err());
        Ok(())
    }
}
