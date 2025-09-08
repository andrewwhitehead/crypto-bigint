//! [`BoxedUint`] multiplication operations.

use crate::{
    BoxedUint, CheckedMul, ConcatenatingMul, Limb, Uint, UintRef, Wrapping, WrappingMul, Zero,
    mul::karatsuba::{wrapping_mul, wrapping_square},
    uint::mul::{
        karatsuba::{MIN_STARTING_LIMBS, widening_mul, widening_square},
        schoolbook,
    },
};
use core::ops::{Mul, MulAssign};
use subtle::{Choice, CtOption};

impl BoxedUint {
    /// Multiply `self` by `rhs`.
    ///
    /// Returns a widened output with a limb count equal to the sums of the input limb counts.
    pub fn mul(&self, rhs: &Self) -> Self {
        self.widening_mul_limbs(rhs.as_limbs())
    }

    /// Multiply `self` by `rhs`.
    ///
    /// Returns a widened output with a limb count equal to the sums of the input limb counts.
    pub(crate) fn mul_uint<const LIMBS: usize>(&self, rhs: &Uint<LIMBS>) -> Self {
        self.widening_mul_limbs(rhs.as_limbs())
    }

    #[inline(always)]
    fn widening_mul_limbs(&self, rhs: &[Limb]) -> Self {
        let size = self.nlimbs() + rhs.len();
        let overlap = self.nlimbs().min(rhs.len());

        if overlap >= MIN_STARTING_LIMBS {
            let mut limbs = vec![Limb::ZERO; size + overlap * 2];
            let (out, scratch) = UintRef::new_mut(limbs.as_mut_slice()).split_at_mut(size);
            widening_mul(&self.limbs, rhs, out, scratch);
            limbs.truncate(size);
            return limbs.into();
        }

        let mut limbs = vec![Limb::ZERO; size];
        let (lo, hi) = limbs.split_at_mut(self.nlimbs());
        schoolbook::mul_wide(&self.limbs, rhs, lo, hi);
        limbs.into()
    }

    /// Perform wrapping multiplication, wrapping to the width of `self`.
    pub fn wrapping_mul(&self, rhs: &Self) -> Self {
        self.wrapping_mul_limbs(rhs.as_limbs())
    }

    #[inline(always)]
    fn wrapping_mul_limbs(&self, rhs: &[Limb]) -> Self {
        // Perform a widening Karatsuba multiplication and truncate
        // for very large numbers, where the performance is better.
        if self.nlimbs().min(rhs.len()) > 16 {
            let size = self.nlimbs();
            let mut limbs = vec![Limb::ZERO; size * 2];
            let (out, scratch) = UintRef::new_mut(limbs.as_mut_slice()).split_at_mut(size);
            wrapping_mul(&self.limbs, rhs, out, scratch);
            limbs.truncate(size);
            return limbs.into();
        }

        let mut limbs = vec![Limb::ZERO; self.nlimbs()];
        schoolbook::wrapping_mul(&self.limbs, rhs, &mut limbs);
        limbs.into()
    }

    /// Multiply `self` by itself.
    pub fn square(&self) -> Self {
        let size = self.nlimbs() * 2;

        if self.nlimbs() >= MIN_STARTING_LIMBS * 2 {
            let mut limbs = vec![Limb::ZERO; size * 2];
            let (out, scratch) = UintRef::new_mut(limbs.as_mut_slice()).split_at_mut(size);
            widening_square(&self.limbs, out, scratch);
            limbs.truncate(size);
            return limbs.into();
        }

        let mut limbs = vec![Limb::ZERO; size];
        let (lo, hi) = limbs.split_at_mut(self.nlimbs());
        schoolbook::square_wide(&self.limbs, lo, hi);
        limbs.into()
    }

    /// Multiply `self` by itself, wrapping to the width of `self`.
    pub fn wrapping_square(&self) -> Self {
        let size = self.nlimbs();

        if size > 16 {
            let mut limbs = vec![Limb::ZERO; size * 2];
            let (out, scratch) = UintRef::new_mut(limbs.as_mut_slice()).split_at_mut(size);
            wrapping_square(&self.limbs, out, scratch);
            limbs.truncate(size);
            return limbs.into();
        }

        let mut limbs = vec![Limb::ZERO; size];
        schoolbook::wrapping_square(&self.limbs, &mut limbs);
        limbs.into()
    }
}

impl CheckedMul for BoxedUint {
    fn checked_mul(&self, rhs: &BoxedUint) -> CtOption<Self> {
        let product = self.mul(rhs);

        // Ensure high limbs are all zero
        let is_some = product.limbs[self.nlimbs()..]
            .iter()
            .fold(Choice::from(1), |choice, limb| choice & limb.is_zero());

        let mut limbs = product.limbs.into_vec();
        limbs.truncate(self.nlimbs());
        CtOption::new(limbs.into(), is_some)
    }
}

impl Mul<BoxedUint> for BoxedUint {
    type Output = BoxedUint;

    fn mul(self, rhs: BoxedUint) -> Self {
        BoxedUint::mul(&self, &rhs)
    }
}

impl Mul<&BoxedUint> for BoxedUint {
    type Output = BoxedUint;

    fn mul(self, rhs: &BoxedUint) -> Self {
        BoxedUint::mul(&self, rhs)
    }
}

impl Mul<BoxedUint> for &BoxedUint {
    type Output = BoxedUint;

    fn mul(self, rhs: BoxedUint) -> Self::Output {
        BoxedUint::mul(self, &rhs)
    }
}

impl Mul<&BoxedUint> for &BoxedUint {
    type Output = BoxedUint;

    fn mul(self, rhs: &BoxedUint) -> Self::Output {
        self.checked_mul(rhs)
            .expect("attempted to multiply with overflow")
    }
}

impl MulAssign<BoxedUint> for BoxedUint {
    fn mul_assign(&mut self, rhs: BoxedUint) {
        self.mul_assign(&rhs)
    }
}

impl MulAssign<&BoxedUint> for BoxedUint {
    fn mul_assign(&mut self, rhs: &BoxedUint) {
        *self = self.clone().mul(rhs)
    }
}

impl MulAssign<Wrapping<BoxedUint>> for Wrapping<BoxedUint> {
    fn mul_assign(&mut self, other: Wrapping<BoxedUint>) {
        *self = Wrapping(self.0.wrapping_mul(&other.0));
    }
}

impl MulAssign<&Wrapping<BoxedUint>> for Wrapping<BoxedUint> {
    fn mul_assign(&mut self, other: &Wrapping<BoxedUint>) {
        *self = Wrapping(self.0.wrapping_mul(&other.0));
    }
}

impl ConcatenatingMul for BoxedUint {
    type Output = Self;

    #[inline]
    fn concatenating_mul(&self, rhs: BoxedUint) -> Self {
        self.mul(&rhs)
    }
}

impl ConcatenatingMul<&BoxedUint> for BoxedUint {
    type Output = Self;

    #[inline]
    fn concatenating_mul(&self, rhs: &BoxedUint) -> Self {
        self.mul(rhs)
    }
}

impl WrappingMul for BoxedUint {
    fn wrapping_mul(&self, v: &Self) -> Self {
        self.wrapping_mul(v)
    }
}

#[cfg(test)]
mod tests {
    use crate::BoxedUint;

    #[test]
    fn mul_zero_and_one() {
        assert!(bool::from(
            BoxedUint::zero().mul(&BoxedUint::zero()).is_zero()
        ));
        assert!(bool::from(
            BoxedUint::zero().mul(&BoxedUint::one()).is_zero()
        ));
        assert!(bool::from(
            BoxedUint::one().mul(&BoxedUint::zero()).is_zero()
        ));
        assert_eq!(BoxedUint::one().mul(&BoxedUint::one()), BoxedUint::one());
    }

    #[test]
    fn mul_primes() {
        let primes: &[u32] = &[3, 5, 17, 257, 65537];

        for &a_int in primes {
            for &b_int in primes {
                let actual = BoxedUint::from(a_int).mul(&BoxedUint::from(b_int));
                let expected = BoxedUint::from(a_int as u64 * b_int as u64);
                assert_eq!(actual, expected);
            }
        }
    }

    #[cfg(feature = "rand_core")]
    #[test]
    fn mul_cmp() {
        use crate::{RandomBits, Resize};
        use rand_core::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);

        for i in 0..50 {
            let a = BoxedUint::random_bits(&mut rng, 4096);
            assert_eq!(a.mul(&a), a.square(), "a={a}, i={i}");
            assert_eq!(a.wrapping_mul(&a), a.wrapping_square(), "a={a}, i={i}");
        }

        for i in 0..50 {
            let a = BoxedUint::random_bits(&mut rng, 4096);
            let b = BoxedUint::random_bits(&mut rng, 5000);
            let expect = a.mul(&b);
            assert_eq!(b.mul(&a), expect, "a={a}, b={b}, i={i}");
            assert_eq!(
                a.wrapping_mul(&b),
                expect.clone().resize_unchecked(a.bits_precision()),
                "a={a}, b={b}, i={i}"
            );
            assert_eq!(
                b.wrapping_mul(&a),
                expect.clone().resize_unchecked(b.bits_precision()),
                "a={a}, b={b}, i={i}"
            );
        }
    }
}
