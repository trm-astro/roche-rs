use pyo3::prelude::*;
use std::{ops, fmt};

// useful for defining python operators
#[derive(FromPyObject)]
enum Vec3OrF64 {
    Vec3(Vec3),
    F64(f64),
}

#[pyclass(from_py_object)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec3 {
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
    #[pyo3(get, set)]
    pub z: f64,
}

#[pymethods]
impl Vec3 {
    #[new]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// 3D vector representation of the centre of mass of the primary star.
    #[staticmethod]
    pub fn cofm1() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// 3D vector representation of the centre of mass of the secondary star.
    #[staticmethod]
    pub fn cofm2() -> Self {
        Self {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn set(&mut self, x: f64, y: f64, z: f64) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    /// Normalises the vector in place
    pub fn unit(&mut self) {
        let norm = self.length();
        self.x /= norm;
        self.y /= norm;
        self.z /= norm;
    }

    /// Returns a normalised version of the vector
    pub fn norm(&self) -> Self {
        let norm = self.length();
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    // For `__repr__` we want to return a string that Python code could use to recreate
    // the `Vec3`, like `Vec3(5, 6, 7)` for example.
    fn __repr__(&self) -> String {
        // We use the `format!` macro to create a string. Its first argument is a
        // format string, followed by any number of parameters which replace the
        // `{}`'s in the format string.
        format!("Vec3({}, {}, {})", self.x, self.y, self.z)
    }

    /// Returns the length of the vector
    pub fn length(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    /// Returns the squared length of the vector
    pub fn sqr(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2) + self.z.powi(2)
    }

    /// Returns the dot product of two vectors
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns the cross product of two vectors
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        let temp_x = self.y * other.z - self.z * other.y;
        let temp_y = self.z * other.x - self.x * other.z;
        let temp_z = self.x * other.y - self.y * other.x;
        Vec3::new(temp_x, temp_y, temp_z)
    }

    // Python dunder methods for arithmetic
    fn __add__(&self, other: Vec3OrF64) -> Self {
        match other {
            Vec3OrF64::Vec3(v) => *self + v,
            Vec3OrF64::F64(f) => *self + f,
        }
    }
    fn __radd__(&self, other: Vec3OrF64) -> Self {
        match other {
            Vec3OrF64::Vec3(v) => v + *self,
            Vec3OrF64::F64(f) => f + *self,
        }
    }
    fn __sub__(&self, other: Vec3OrF64) -> Self {
        match other {
            Vec3OrF64::Vec3(v) => *self - v,
            Vec3OrF64::F64(f) => *self - f,
        }
    }
    fn __rsub__(&self, other: Vec3OrF64) -> Self {
        match other {
            Vec3OrF64::Vec3(v) => v - *self,
            Vec3OrF64::F64(f) => f - *self,
        }
    }
    fn __mul__(&self, other: f64) -> Self {
        *self * other
    }
    fn __rmul__(&self, other: f64) -> Self {
        other * *self
    }
    fn __truediv__(&self, other: f64) -> Self {
        *self / other
    }
    fn __rtruediv__(&self, other: f64) -> Self {
        other / *self
    }
    fn __neg__(&self) -> Self {
        -*self
    }
}

// in-place multiplication by f64
impl ops::MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

// in-place division by f64
impl ops::DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

// in-place addition by f64
impl ops::AddAssign<f64> for Vec3 {
    fn add_assign(&mut self, rhs: f64) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

// in-place subtraction by f64
impl ops::SubAssign<f64> for Vec3 {
    fn sub_assign(&mut self, rhs: f64) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
    }
}

// in-place addition of vector
impl ops::AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

// in-place subtraction of vector
impl ops::SubAssign<Vec3> for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

// Sum of two vectors
impl ops::Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

// Sum of vector and f64
impl ops::Add<f64> for Vec3 {
    type Output = Self;
    fn add(self, other: f64) -> Self {
        Self {
            x: self.x + other,
            y: self.y + other,
            z: self.z + other,
        }
    }
}

// Sum of f64 and vector
impl ops::Add<Vec3> for f64 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self + other.x,
            y: self + other.y,
            z: self + other.z,
        }
    }
}

// Difference between two vectors
impl ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

// Difference between vector and f64
impl ops::Sub<f64> for Vec3 {
    type Output = Self;
    fn sub(self, other: f64) -> Self {
        Self {
            x: self.x - other,
            y: self.y - other,
            z: self.z - other,
        }
    }
}

// Difference between f64 and vector
impl ops::Sub<Vec3> for f64 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self - other.x,
            y: self - other.y,
            z: self - other.z,
        }
    }
}

// Multiplication of Vec3 by f64
impl ops::Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

// Multiplication of f64 by Vec3
impl ops::Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

// Division of Vec3 by f64
impl ops::Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

// Division of f64 by Vec3
impl ops::Div<Vec3> for f64 {
    type Output = Vec3;

    fn div(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self / rhs.x,
            y: self / rhs.y,
            z: self / rhs.z,
        }
    }
}

// Negative of a vector
impl ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Vec3({}, {}, {})",
            self.x,
            self.y,
            self.z
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set() {
        let mut v1 = Vec3::new(0.0, 0.0, 0.0);
        v1.set(1.0, 1.0, 1.0);
        assert_eq!(v1, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn unit() {
        let mut v1 = Vec3::new(2.0, 3.0, 6.0);
        v1.unit();
        assert_eq!(v1, Vec3::new(2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0));
    }

    #[test]
    fn norm() {
        let v1 = Vec3::new(2.0, 3.0, 6.0);
        assert_eq!(v1.norm(), Vec3::new(2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0));
    }

    #[test]
    fn length() {
        let v1 = Vec3::new(2.0, 3.0, 6.0);
        assert_eq!(v1.length(), 7.0);
    }

    #[test]
    fn sqr() {
        let v1 = Vec3::new(2.0, 3.0, 6.0);
        assert_eq!(v1.sqr(), 49.0);
    }

    #[test]
    fn dot() {
        let v1 = Vec3::new(1.0, 3.0, -5.0);
        let v2 = Vec3::new(4.0, -2.0, -1.0);
        assert_eq!(v1.dot(&v2), 3.0);
        assert_eq!(v2.dot(&v1), 3.0);
    }

    #[test]
    fn cross() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        let v2 = Vec3::new(4.0, 9.0, 2.0);
        assert_eq!(v1.cross(&v2), Vec3::new(-15.0, -2.0, 39.0));
    }

    #[test]
    fn mulassign_f64() {
        let mut v1 = Vec3::new(3.0, -3.0, 1.0);
        v1 *= 2.0;
        assert_eq!(v1, Vec3::new(6.0, -6.0, 2.0));
    }

    #[test]
    fn addassign_f64() {
        let mut v1 = Vec3::new(3.0, -3.0, 1.0);
        v1 += 2.0;
        assert_eq!(v1, Vec3::new(5.0, -1.0, 3.0));
    }

    #[test]
    fn subassign_f64() {
        let mut v1 = Vec3::new(3.0, -3.0, 1.0);
        v1 -= 2.0;
        assert_eq!(v1, Vec3::new(1.0, -5.0, -1.0));
    }

    #[test]
    fn addassign_vec3() {
        let mut v1 = Vec3::new(3.0, -3.0, 1.0);
        let v2 = Vec3::new(1.0, 2.0, -5.0);
        v1 += v2;
        assert_eq!(v1, Vec3::new(4.0, -1.0, -4.0));
    }

    #[test]
    fn subassign_vec3() {
        let mut v1 = Vec3::new(3.0, -3.0, 1.0);
        let v2 = Vec3::new(1.0, 2.0, -5.0);
        v1 -= v2;
        assert_eq!(v1, Vec3::new(2.0, -5.0, 6.0));
    }

    #[test]
    fn add_vec3vec3() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        let v2 = Vec3::new(1.0, 2.0, -5.0);
        assert_eq!(v1 + v2, Vec3::new(4.0, -1.0, -4.0));
    }

    #[test]
    fn sub_vec3vec3() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        let v2 = Vec3::new(1.0, 2.0, -5.0);
        assert_eq!(v1 - v2, Vec3::new(2.0, -5.0, 6.0));
    }

    #[test]
    fn divassign_f64() {
        let mut v1 = Vec3::new(4.0, -2.0, 1.0);
        v1 /= 2.0;
        assert_eq!(v1, Vec3::new(2.0, -1.0, 0.5));
    }

    #[test]
    fn div_vec3f64() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(v1 / 2.0, Vec3::new(1.5, -1.5, 0.5));
    }

    #[test]
    fn div_f64vec3() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(9.0 / v1, Vec3::new(3.0, -3.0, 9.0));
    }

    #[test]
    fn mul_vec3f64() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(v1 * 2.0, Vec3::new(6.0, -6.0, 2.0));
    }

    #[test]
    fn mul_f64vec3() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(2.0 * v1, Vec3::new(6.0, -6.0, 2.0));
    }

    #[test]
    fn add_vec3f64() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(v1 + 2.0, Vec3::new(5.0, -1.0, 3.0));
    }

    #[test]
    fn add_f64vec3() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(2.0 + v1, Vec3::new(5.0, -1.0, 3.0));
    }

    #[test]
    fn sub_vec3f64() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(v1 - 2.0, Vec3::new(1.0, -5.0, -1.0));
    }

    #[test]
    fn sub_f64vec3() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(2.0 - v1, Vec3::new(-1.0, 5.0, 1.0));
    }

    #[test]
    fn neg() {
        let v1 = Vec3::new(3.0, -3.0, 1.0);
        assert_eq!(-v1, Vec3::new(-3.0, 3.0, -1.0));
    }
}
