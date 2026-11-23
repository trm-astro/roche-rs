use crate::errors::RocheError;
use crate::{Star, Vec3};
use crate::{set_earth, rpot_val_grad, rpot_val, rpot_grad};
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





///
/// The line of sight to any fixed point in a binary sweeps out a cone at the
/// binary rotates. Positions on the cone can be parameterised by the orbital
/// phase phi and the multiplier ('lambda') needed to get from the fixed point.
/// The question pot_min tries to solve is "does the cone intersect a surface
/// of fixed Roche potential lying within a Roche lobe?". It does so by
/// constrained minimisation over a region of phi and lambda using the L-BFGS-B
/// algorithm with the phi and lambda ranges both rescaled to [-10, 10].
/// It stops as soon as any potential below a critical value is found.
/// The initial range of phi and lambda can be determined using sphere_eclipse
/// which calculates them for a sphere.
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