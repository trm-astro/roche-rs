use pyo3::prelude::*;
use std::fmt;
use crate::Etype;
use crate::Vec3;

///
/// Struct defining a point on the surface of a model grid (e.g. of a star or disc etc.)
/// A `Point` has a position, a direction which is the surface normal, an area,
/// a relative gravity, a vector of phase pairs defining when the point is eclipsed by
/// another model component, and a flux.
///
#[pyclass(skip_from_py_object, module = "roche")]
#[derive(Clone, Debug)]
pub struct Point {
    #[pyo3(get)]
    pub position: Vec3,

    #[pyo3(get)]
    pub direction: Vec3,

    #[pyo3(get)]
    pub area: f32,

    #[pyo3(get)]
    pub gravity: f32,

    #[pyo3(get)]
    pub eclipse: Etype,

    #[pyo3(get)]
    pub flux: f32,
}


#[pymethods]
impl Point {
    
    ///
    /// Creates a new Point.
    ///
    #[new]
    pub fn new(position: Vec3, direction: Vec3, area: f64, gravity: f64, eclipse: Etype) -> Self {
        Self {
            position,
            direction,
            area: area as f32,
            gravity: gravity as f32,
            eclipse,
            flux: 0.0,
        }
    }
    // pickle support: constructor args, then flux restored via state
    fn __getnewargs__(&self) -> (Vec3, Vec3, f64, f64, Etype) {
        (
            self.position,
            self.direction,
            self.area as f64,
            self.gravity as f64,
            self.eclipse.clone(),
        )
    }

    fn __getstate__(&self) -> f32 {
        self.flux
    }

    fn __setstate__(&mut self, flux: f32) {
        self.flux = flux;
    }

    ///
    /// sets the point's flux.
    ///
    pub fn set_flux(&mut self, flux: f32) {
        self.flux = flux;
    }
    
    ///
    ///checks that the given phase is not during one of the
    /// phase ranges when the point is eclipsed.
    ///  
    pub fn is_visible(&self, phase: f64) -> bool {
        let phi: f64 = phase - phase.floor();
        for &(p1, p2) in &self.eclipse {
            if (phi >= p1 && phi <= p2) || phi <= p2 - 1.0 {
                return false;
            }
        }
        true
    }
    
    ///
    /// This version of is_visible will not correct for phases outside
    /// of expected range to speed up large loops.
    /// run phase = phase - phase.floor();
    /// outside of loop beforehand
    ///
    pub fn is_visible_phase_normed(&self, phase: f64) -> bool {
        for &(p1, p2) in &self.eclipse {
            if (phase >= p1 && phase <= p2) || phase <= p2 - 1.0 {
                return false;
            }
        }
        true
    }

    fn __repr__(&self) -> String {
        // We use the `format!` macro to create a string. Its first argument is a
        // format string, followed by any number of parameters which replace the
        // `{}`'s in the format string.
        format!("Point({}, {}, {}, {}, {:?}, {})", self.position, self.direction, self.area, self.gravity, self.eclipse, self.flux)
    }
}

impl Default for Point {
    fn default() -> Self {
        Self::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            0.0,
            0.0,
            vec![(0.0, 0.0)],
        )
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Point({}, {}, {}, {}, {:?}, {})",
            self.position,
            self.direction,
            self.area,
            self.gravity,
            self.eclipse,
            self.flux
        )
    }
}


