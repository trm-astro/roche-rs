use crate::errors::RocheError;
use crate::{Star, Vec3};
use crate::{set_earth, rpot_val_grad, rpot_val};
use lbfgsb_rs_pure::{IterationInfo, LBFGSB, Status};
use std::f64::consts::TAU;






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
        Status::Converged => Ok(true),
        Status::MaxIter => Ok(false),
        Status::NumericalFailure => Err(RocheError::PotminError("Failed with NumericalFailure".to_string())),
        // Status::LineSearchFailure if (sol.f - f_old).abs() < delphi => Ok(false),
        Status::LineSearchFailure => Err(RocheError::PotminError("Failed with LineSearchFailure".to_string())),
        // _ => Err(RocheError::PotminError("potmin failed".to_string()))
    }
    
}


// struct RochePotential{
//     lower: Vec<f64>,
//     upper: Vec<f64>,
//     q: f64,
//     cosi: f64,
//     sini: f64,
//     star: Star,
//     spin: f64,
//     p: Vec3,
// }

// impl CostFunction for RochePotential {
//     type Param = Vec<f64>;
//     type Output = f64;
//     type Error = RocheError;
//     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
//         println!(
//             "COST x = [{:.17e}, {:.17e}]",
//             x[0], x[1]
//         );


//         let earth = set_earth(self.cosi, self.sini, x[0]);
//         let val = rpot_val(self.q, self.star, self.spin, &earth, &self.p, x[1])?;
//         Ok(val)
//     }
// }

// impl Gradient for RochePotential {
//     type Gradient = Vec<f64>;
//     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
//         let earth = set_earth(self.cosi, self.sini, x[0]);
//         let (dphi, dlam) = rpot_grad(self.q, self.star, self.spin, &earth, &self.p, x[1])?;
//         Ok(vec![dphi, dlam])
//     }
// }

// impl BoxConstraints for RochePotential {
//     fn lower(&self) -> &Vec<f64> {
//         println!("LOWER = {:?}", self.lower);
//         &self.lower
//     }

//     fn upper(&self) -> &Vec<f64> {
//         println!("UPPER = {:?}", self.upper);
//         &self.upper
//     }
// }

// pub fn pot_min_basin(
//     q: f64,
//     star: Star,
//     spin: f64,
//     cosi: f64,
//     sini: f64,
//     p: &Vec3,
//     phi1: f64,
//     phi2: f64,
//     lam1: f64,
//     lam2: f64,
//     _rref: f64,
//     pref: f64,
//     _acc: f64,
//     phi: &mut f64,
//     lam: &mut f64,
// ) -> Result<bool, RocheError> {
//     *phi = (phi1 + phi2) / 2.;
//     *lam = (lam1 + lam2) / 2.;
//     // let x0 = vec![phi, lam];
//     // let constraints = BoxConstraints::new(lower, upper);
//     // let solver = Lbfgsb::new(constraints);
//     // let p = p.clone();
//     let problem = RochePotential {
//             lower: vec![phi1, lam1],
//             upper: vec![phi2, lam2],
//             q,
//             cosi,
//             sini,
//             star,
//             spin,
//             p: p.clone()
//         };

//     // println!("x0    = {:?}", vec![*phi, *lam]);
//     // println!("lower = {:?}", lower);
//     // println!("upper = {:?}", upper);
//     let result = Executor::new(
//         problem,
//         Lbfgsb::new(),
//         LbfgsState::new(vec![*phi, *lam], 10),
//     )
//         .max_iter(200)
//         .terminate_on(ProjectedGradientTolerance::new(vec![phi1, lam1], vec![phi2, lam2], 1.0e-5))
//         .run()
//         .unwrap();

//     println!("f = {:.17e}, (phi, lam) = ({:.17e}, {:.17e}), reason = {:?}", result.cost(), result.param()[0], result.param()[1], result.reason);
//     if result.cost() <= pref {
//         println!("f = {:.17e}, (phi, lam) = ({:.17e}, {:.17e})", result.cost(), result.param()[0], result.best_param()[1]);
//         Ok(true)
//     } else {
//         Ok(false)
//     }

// }