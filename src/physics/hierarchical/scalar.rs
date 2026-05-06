use core::ops::{Add, Div, Mul, Sub};

/// Scalar type supported by the generic dual-tree infrastructure.
pub trait DualTreeScalar:
    Copy
    + Clone
    + Default
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;

    fn from_f64(value: f64) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
}

impl DualTreeScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
}

impl DualTreeScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
}
