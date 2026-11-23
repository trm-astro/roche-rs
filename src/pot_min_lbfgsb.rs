use crate::errors::RocheError;
use crate::{Star, Vec3};
use crate::{set_earth, rpot_val_grad, rpot_val, rpot_grad};
use lbfgsb_rs_pure::{IterationInfo, LBFGSB, Status};
use std::f64::consts::TAU;
use basin::{
    BoxBounds,
    ConstraintDescription,
    ConstraintSet,
    Constraints,
    CostFunction,
    Executor,
    Gradient,
    Lbfgsb,
};
use basin::solver::lbfgs::model_strategy::ModelDirection;
use basin::{line_search::projected_armijo::ProjectedArmijo};
use basin::solver::lbfgs::model_strategy::ProjectedSearch;





///
/// The line of sight to any fixed point in a binary sweeps out a cone at the
/// binary rotates. Positions on the cone can be parameterised by the orbital
/// phase phi and the multiplier ('lambda') needed to get from the fixed point.
/// The question pot_min tries to solve is "does the cone intersect a surface
/// of fixed Roche potential lying within a Roche lobe?". It does so by
/// minimisation over a region of phi and lambda. It stops as soon as any
/// potential below a critical value is found. The initial range of phi and
/// lambda can be determined using sphere_eclipse which calculates them for a
/// sphere.
///
/// Arguments:
///
/// * `q`:  mass ratio = M2/M1.
/// * `cosi`: cosine orbital inclination.
/// * `sini`: sine orbital inclination.
/// * `star`: which star (needed for asynchronous case).
/// * `spin`: ratio of spin/orbital.
/// * `p`: point of origin.
/// * `phi1`: minimum phase within which eclipse may occur (0 - 1).
/// * `phi2`: maximum phase within which an eclipse may occur (> phi1).
/// * `lam1`: minimum multiplier line of sight crosses eclipsing star.
/// * `lam2`: maximum multiplier line of sight crosses eclipsing star.
/// * `rref`: reference radius.
/// * `pref`: reference potential.
/// * `acc`:  absolute accuracy in position to go for.
/// * `phi`:  phi at minimum potential. Ingress occurs between phi1 and phi.
///   if there is an eclipse. Egress occurs between phi and phi2.
/// * `lam`:  lambda at minimum potential.
///
/// Returns:
///
/// * true if minimum potential is below the reference.
///
pub fn pot_min_lbfgsb(
    q: f64,
    star: Star,
    spin: f64,
    cosi: f64,
    sini: f64,
    p: &Vec3,
    phi1: f64,
    phi2: f64,
    lam1: f64,
    lam2: f64,
    rref: f64,
    pref: f64,
    acc: f64,
    phi: &mut f64,
    lam: &mut f64,
) -> Result<bool, RocheError> {
    *phi = (phi1 + phi2) / 2.;
    *lam = (lam1 + lam2) / 2.;

    let rp: f64 = TAU * *phi;
    let (sinp, cosp) = rp.sin_cos();
    let earth: Vec3 = Vec3::new(sini * cosp, -sini * sinp, cosi);

    let pot = rpot_val(q, star, spin, &earth, p, *lam)?;
    
    if pot <= pref {
        return Ok(true);
    };

    let _delphi: f64 = q / (1.0 + q) * ((acc * acc) / (rref * rref)) / 2.0;

    let m = 10;
    let mut solver = LBFGSB::new(m)
        .with_pgtol(1e-5)
        .with_max_iter(200)
        .with_verbose(true);

    let mut x = vec![*phi, *lam];
    let lower = vec![phi1, lam1];
    let upper = vec![phi2, lam2];

    println!("initial x     = {:?}", x);
    println!("lower         = {:?}", lower);
    println!("upper         = {:?}", upper);


    let mut f_old: f64 = pot;

    let sol = solver
        .minimize_with_callback(
            &mut x,
            &lower,
            &upper,
            &mut |x: &[f64]| {
                let earth = set_earth(cosi, sini, x[0]);
                let (val, dphi, dlam) = rpot_val_grad(q, star, spin, &earth, p, x[1]).unwrap();
                println!(
                    "x = [{:.17e}, {:.17e}], f = {:.17e}, dphi = {:.17e}, dlam = {:.17e}",
                    x[0], x[1], val, dphi, dlam
                );
                (val, vec![dphi, dlam])
            },
            &mut |info: &IterationInfo, _x: &[f64]| {

                println!("proj_grad_norm = {:.17e}", info.proj_grad_norm);
                if info.f <= pref {
                    return lbfgsb_rs_pure::IterationControl::StopConverged;
                }

                // if info.proj_grad_norm <= 1e-8 {
                //     return lbfgsb_rs_pure::IterationControl::StopCustom;
                // }

                
                // let df = (info.f - f_old).abs();
                // println!("df = {:.17e}", df);
                // if df < delphi {
                //     return lbfgsb_rs_pure::IterationControl::StopCustom;
                // }

                f_old = info.f;
                lbfgsb_rs_pure::IterationControl::Continue
            },
        )
        .unwrap();

    

    // if sol.status == Status::LineSearchFailure && (sol.f - f_old).abs() < delphi {
    //     println!("pmin = {:.17e}, pot = {:.17e}, delphi = {:.17e}", sol.f, f_old, delphi);
    //     return Ok(false);
    // }
    match sol.status {
        Status::Converged => Ok(sol.f <= pref),
        Status::MaxIter => Ok(false),
        Status::NumericalFailure => Err(RocheError::PotminError("Failed with NumericalFailure".to_string())),
        // Status::LineSearchFailure if (sol.f - f_old).abs() < delphi => Ok(false),
        Status::LineSearchFailure => Err(RocheError::PotminError("Failed with LineSearchFailure".to_string())),
        // _ => Err(RocheError::PotminError("potmin failed".to_string()))
        Status::InvalidInput => Err(RocheError::PotminError("Failed with InvalidInput".to_string())),
    }
    
}



pub fn pot_min_basin(
    q: f64,
    star: Star,
    spin: f64,
    cosi: f64,
    sini: f64,
    p: &Vec3,
    phi1: f64,
    phi2: f64,
    lam1: f64,
    lam2: f64,
    _rref: f64,
    pref: f64,
    _acc: f64,
    phi: &mut f64,
    lam: &mut f64,
) -> Result<bool, RocheError> {
    let phi_mid = 0.5 * (phi1 + phi2);
    let lam_mid = 0.5 * (lam1 + lam2);
    *phi = phi_mid;
    *lam = lam_mid;

    let rp: f64 = TAU * *phi;
    let (sinp, cosp) = rp.sin_cos();
    let earth: Vec3 = Vec3::new(sini * cosp, -sini * sinp, cosi);
    let pot = rpot_val(q, star, spin, &earth, p, *lam)?;
    
    if pot <= pref {
        return Ok(true);
    };

    // Choose a target box half-width in solver space (e.g., 10.0 so the initial
    // unit step of 1.0 moves 10% of the distance from center to boundary).
    let box_half_width = 10.0;
    let phi_scale = (0.5 * (phi2 - phi1)) / box_half_width;
    let lam_scale = (0.5 * (lam2 - lam1)) / box_half_width;

    // Guard against degenerate zero-width intervals
    let phi_scale = if phi_scale > 0.0 { phi_scale } else { 1.0 };
    let lam_scale = if lam_scale > 0.0 { lam_scale } else { 1.0 };

    let problem = RochePotential {
        q,
        star,
        spin,
        cosi,
        sini,
        p: p.clone(),
        center: [phi_mid, lam_mid],
        scale: [phi_scale, lam_scale],
        lower: vec![-box_half_width, -box_half_width],
        upper: vec![box_half_width, box_half_width],
    };
    
    let x0 = vec![0.0, 0.0]; // Start at the center of the box in solver space
    
    let solver = Lbfgsb::<Vec<f64>>::new()
        .with_m_capacity(10)
        .with_absolute_projected_gradient_tolerance(1e-10)
        .with_relative_cost_reduction_tolerance(None);

    let result = Executor::from_start(problem, solver, x0)
        .max_iter(200)
        .target_objective(pref)
        .run()
        .map_err(|err| {
            match err {
                basin::ConstraintSolveError::Evaluation(e) => e,
                basin::ConstraintSolveError::InvalidConstraints(e) => {
                    RocheError::PotminError(
                        format!("Invalid constraints: {:?}", e)
                    )
                }
                _ => RocheError::PotminError(
                        format!("Basin solver error: {:?}", err)
                    ),
            }
        })?;
        
    let best_cost = result.best_cost();
    let best_param = result.best_param();

    // Unscale back to physical (phi, lam) coordinates
    *phi = phi_mid + phi_scale * best_param[0];
    *lam = lam_mid + lam_scale * best_param[1];

    println!("best pot = {}", best_cost);
    if best_cost <= pref {
        return Ok(true);
    }

    match &result.report.termination {
        basin::Termination::Failed(reason) => {
            Err(RocheError::PotminError(
                format!("Basin minimisation failed: {:?}", reason)
            ))
        }

        _ => Ok(false),
    }
}


struct RochePotential{
        q: f64,
        star: Star,
        spin: f64,
        cosi: f64,
        sini: f64,
        p: Vec3,
        center: [f64; 2],
        scale: [f64; 2],
        lower: Vec<f64>,
        upper: Vec<f64>,
}

impl RochePotential {
    #[inline]
    fn to_physical(&self, u: &[f64]) -> (f64, f64) {
        (
            self.center[0] + self.scale[0] * u[0],
            self.center[1] + self.scale[1] * u[1],
        )
    }
}

impl CostFunction for RochePotential {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = RocheError;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        let (phi, lam) = self.to_physical(x);
        let earth = set_earth(self.cosi, self.sini, phi);
        let val = rpot_val(self.q, self.star, self.spin, &earth, &self.p, lam)?;
        Ok(val)
    }
}

impl Gradient for RochePotential {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        let (phi, lam) = self.to_physical(x);
        let earth = set_earth(self.cosi, self.sini, phi);
        let (dphi, dlam) = rpot_grad(self.q, self.star, self.spin, &earth, &self.p, lam)?;
        Ok(vec![dphi, dlam])
    }

    fn cost_and_gradient(&self, x: &Vec<f64>) -> Result<(f64, Vec<f64>), Self::Error> {
        let (phi, lam) = self.to_physical(x);
        let earth = set_earth(self.cosi, self.sini, phi);
        let (val, dphi, dlam) = rpot_val_grad(self.q, self.star, self.spin, &earth, &self.p, lam)?;
        Ok((val, vec![dphi, dlam]))
    }
}

impl Constraints for RochePotential {
    type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

    fn constraints(&self)
    -> <Self::Set as ConstraintDescription>::Borrowed<'_>
    {
        ConstraintSet::new().with_bounds(BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}
// }