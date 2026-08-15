use crate::errors::RocheError;
use crate::{Etype, Star, Vec3};
use crate::{dbrent, pot_min, x_l1, x_l1_1, x_l1_2, x_l2, x_l3};
use crate::{drpot1, drpot2, rpot_grad, rpot_val, rpot1, rpot2};
use crate::{set_earth, sphere_eclipse, sphere_eclipse_vector};
use std::f64::consts::TAU;

///
/// RocheContext is a basic struct packaging the mass ratio, q=M2/M1,
/// the star (i.e. the Primary or Secondary), the spin of that star,
/// and the x-coordinate of the L1 point together with the functions
/// that accept these values as inputs. x_l1 is calculated on initialisation
/// with RocheContext::new(q, star, spin) and saves it being recalculated
/// over and over in loops.
///
pub struct RocheContext {
    pub q: f64,
    pub star: Star,
    pub spin: f64,
    pub x_l1: f64,
}

impl RocheContext {
    pub fn new(q: f64, star: Star, spin: f64) -> Result<Self, RocheError> {
        let x_l1: f64 = match star {
            Star::Primary => x_l1_1(q, spin)?,
            Star::Secondary => x_l1_2(q, spin)?,
        };
        Ok(Self {
            q,
            star,
            spin,
            x_l1,
        })
    }

    pub fn potential(&self, earth: &Vec3, p: &Vec3, lam: f64) -> Result<f64, RocheError> {
        rpot_val(self.q, self.star, self.spin, earth, p, lam)
    }

    pub fn gradient(&self, earth: &Vec3, p: &Vec3, lam: f64) -> Result<(f64, f64), RocheError> {
        let (dp, dl) = rpot_grad(self.q, self.star, self.spin, earth, p, lam)?;
        Ok((dp, dl))
    }

    pub fn potential_grad(
        &self,
        earth: &Vec3,
        p: &Vec3,
        lam: f64,
    ) -> Result<(f64, f64, f64), RocheError> {
        let f = self.potential(earth, p, lam)?;
        let (dp, dl) = rpot_grad(self.q, self.star, self.spin, earth, p, lam)?;
        Ok((f, dp, dl))
    }

    pub fn ref_sphere(&self, ffac: f64) -> Result<(f64, f64), RocheError> {
        let tref: f64;
        let rref: f64;
        let pref: f64;
        if self.star == Star::Primary {
            tref = self.x_l1;
            rref = tref * 1.0_f64.min(1.001 * ffac);
            pref = rpot1(
                self.q,
                self.spin,
                &Vec3 {
                    x: ffac * tref,
                    y: 0.0,
                    z: 0.0,
                },
            )?;
            Ok((rref, pref))
        } else if self.star == Star::Secondary {
            tref = 1.0 - self.x_l1;
            rref = tref * 1.0_f64.min(1.001 * ffac);
            pref = rpot2(
                self.q,
                self.spin,
                &Vec3 {
                    x: 1.0 - ffac * tref,
                    y: 0.0,
                    z: 0.0,
                },
            )?;
            Ok((rref, pref))
        } else {
            let message = format!("{:?} is not and instance of Star.", self.star);
            Err(RocheError::ParameterError(message))
        }
    }

    pub fn fblink(&self, ffac: f64, acc: f64, earth: &Vec3, p: &Vec3) -> Result<bool, RocheError> {
        let (rref, pref) = self.ref_sphere(ffac)?;

        let cofm: Vec3 = match self.star {
            Star::Primary => Vec3::cofm1(),
            Star::Secondary => Vec3::cofm2(),
        };

        // First compute the multipliers cutting the reference sphere (if any)
        let mut lam1 = 0.0;
        let mut lam2 = 0.0;
        if !sphere_eclipse_vector(earth, p, &cofm, rref, &mut lam1, &mut lam2) {
            return Ok(false);
        }
        if lam1 == 0.0 {
            return Ok(true);
        }

        // Create function objects for 1D minimisation in lambda direction
        let func = |lam: f64| self.potential(earth, p, lam);

        // Now try to bracket a minimum. We just crudely compute function at regularly spaced intervals filling in the
        // gaps until the step size between the points drops below the threshold. Take every opportunity to jump out early
        // either if the potential is below the threshold or if we have bracketed a minimum.
        let mut nstep: i32 = 1;
        let mut step: f64 = lam2 - lam1;

        let mut f1: f64 = 0.0;
        let mut f2: f64 = 0.0;
        let mut flam: f64 = 1.0;
        let mut lam: f64 = lam1;

        while step > acc {
            lam = lam1 + step / 2.0;

            for _ in 0..nstep {
                flam = func(lam)?;
                if flam <= pref {
                    return Ok(true);
                }

                // Calculate these as late as possible because they may often not be needed
                if nstep == 1 {
                    f1 = func(lam1)?;
                    f2 = func(lam2)?;
                }

                if flam < f1 && flam < f2 {
                    break;
                }

                lam += step;
            }
            if flam < f1 && flam < f2 {
                break;
            }
            step /= 2.0;
            nstep *= 2;
        }

        if flam < f1 && flam < f2 {
            // OK, minimum bracketted, so finally pin it down accurately
            // Possible that multiple minima could cause problems but I have
            // never seen this in practice.
            let dfunc = |lam: f64| {
                let (_dp, dl) = self.gradient(earth, p, lam)?;
                Ok(dl)
            };

            let (_xmin, flam) =
                dbrent(lam1, lam, lam2, func, dfunc, acc, true, pref)?;

            Ok(flam < pref)
        } else {
            // Not bracketted even after a detailed search, and we have not jumped
            // out either, so assume no eclipse
            Ok(false)
        }
    }

    pub fn face(
        &self,
        direction: Vec3,
        rref: f64,
        pref: f64,
        acc: f64,
    ) -> Result<(Vec3, Vec3, f64, f64), RocheError> {
        let mut pvec: Vec3;
        let mut r: f64;

        let cofm: Vec3 = match self.star {
            Star::Primary => Vec3::cofm1(),
            Star::Secondary => Vec3::cofm2(),
        };

        let rp: fn(f64, f64, &Vec3) -> Result<f64, RocheError> = match self.star {
            Star::Primary => rpot1,
            Star::Secondary => rpot2,
        };

        let drp: fn(f64, f64, &Vec3) -> Result<Vec3, RocheError> = match self.star {
            Star::Primary => drpot1,
            Star::Secondary => drpot2,
        };

        let mut tref: f64 = rp(self.q, self.spin, &(cofm + rref * direction))?;
        if tref < pref {
            let message = format!(
                "point at reference radius {} appears to be at lower potential {} than the reference potential {}",
                rref, tref, pref
            );
            return Err(RocheError::FaceError(message));
        }

        let mut r1: f64 = rref / 2.;
        let mut r2: f64 = rref;
        tref = pref + 1.;

        const MAXSEARCH: i32 = 30;
        let mut i: i32 = 0;
        while i < MAXSEARCH && tref > pref {
            r1 = r2 / 2.;
            tref = rp(self.q, self.spin, &(cofm + r1 * direction))?;
            if tref > pref {
                r2 = r1;
            }
            i += 1;
        }
        if tref > pref {
            let message = "could not find a radius with a potential below the reference potential; probably bad inputs.";
            return Err(RocheError::FaceError(message.to_string()));
        }

        const MAXCHOP: i32 = 100;
        let mut nchop: i32 = 0;
        while r2 - r1 > acc && nchop < MAXCHOP {
            r = (r1 + r2) / 2.;
            pvec = cofm + r * direction;
            if rp(self.q, self.spin, &pvec)? < pref {
                r1 = r;
            } else {
                r2 = r;
            }
            nchop += 1;
        }
        if nchop == MAXCHOP {
            return Err(RocheError::FaceError(
                "reached maximum number of binary chops".to_string(),
            ));
        }
        r = (r1 + r2) / 2.;
        pvec = cofm + r * direction;
        let mut dvec: Vec3 = drp(self.q, self.spin, &pvec)?;
        let g = dvec.length();
        dvec /= g;
        Ok((pvec, dvec, r, g))
    }

    pub fn ingress_egress(
        &self,
        ffac: f64,
        iangle: f64,
        delta: f64,
        r: &Vec3,
        ingress: &mut f64,
        egress: &mut f64,
    ) -> Result<bool, RocheError> {
        let rref: f64;
        let pref: f64;
        (rref, pref) = self.ref_sphere(ffac)?;
        let ri: f64 = iangle.to_radians();
        let (sini, cosi) = ri.sin_cos();

        let cofm: Vec3 = match self.star {
            Star::Primary => Vec3::cofm1(),
            Star::Secondary => Vec3::cofm2(),
        };

        let mut phi1: f64 = 0.0;
        let mut phi2: f64 = 0.0;
        let mut lam1: f64 = 0.0;
        let mut lam2: f64 = 0.0;
        let mut phi: f64 = 0.0;
        let mut lam: f64 = 0.0;

        if sphere_eclipse(
            cosi, sini, r, &cofm, rref, &mut phi1, &mut phi2, &mut lam1, &mut lam2,
        ) {
            let acc: f64 = 2. * (2. * TAU * (lam2 - lam1) * delta).sqrt();

            if self.pot_min(
                cosi, sini, r, phi1, phi2, lam1, lam2, rref, pref, acc, &mut phi, &mut lam,
            )? {
                let mut pin: f64 = phi;
                let mut pout: f64 = phi1;
                let mut pmid: f64;

                while (pin - pout).abs() > delta {
                    pmid = (pin + pout) / 2.0;
                    if self
                        .fblink(ffac, acc, &set_earth(cosi, sini, pmid), r)?
                    {
                        pin = pmid;
                    } else {
                        pout = pmid;
                    }
                }
                *ingress = (pin + pout) / 2.0;
                *ingress -= ingress.floor();

                pin = phi;
                pout = phi2;
                while (pin - pout).abs() > delta {
                    pmid = (pin + pout) / 2.;
                    if self
                        .fblink(ffac, acc, &set_earth(cosi, sini, pmid), r)?
                    {
                        pin = pmid;
                    } else {
                        pout = pmid;
                    }
                }
                *egress = (pin + pout) / 2.0;
                *egress -= egress.floor();
                if *egress < *ingress {
                    *egress += 1.0;
                }
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    pub fn star_eclipse(
        &self,
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
        if (roche && self.ingress_egress(ffac, iangle, delta, posn, &mut ingress, &mut egress)?)
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

    pub fn pot_min(
        &self,
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
        pot_min(
            self.q, self.star, self.spin, cosi, sini, p, phi1, phi2, lam1, lam2, rref, pref, acc,
            phi, lam,
        )
    }

    pub fn x_l1(&self) -> Result<f64, RocheError> {
        x_l1(self.q)
    }

    pub fn x_l1_asyncronous(&self) -> Result<f64, RocheError> {
        match self.star {
            Star::Primary => Ok(x_l1_1(self.q, self.spin)?),
            Star::Secondary => Ok(x_l1_2(self.q, self.spin)?),
        }
    }

    pub fn x_l2(&self) -> Result<f64, RocheError> {
        x_l2(self.q)
    }

    pub fn x_l3(&self) -> Result<f64, RocheError> {
        x_l3(self.q)
    }
}
