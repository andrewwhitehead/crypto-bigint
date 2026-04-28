//! GCD support for [`Limb`].

use super::Limb;
use crate::{Choice, Gcd, NonZero, Odd, Word, primitives::u32_min, word};

impl Limb {
    /// Compute the greatest common divisor of `self` and `rhs`.
    #[must_use]
    pub const fn gcd(self, rhs: Self) -> Self {
        let (self_nz, self_is_nz) = self.to_nz_or_one();
        Limb::select(rhs, self_nz.gcd_unsigned(rhs).get_copy(), self_is_nz)
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[must_use]
    pub const fn gcd_vartime(self, rhs: Self) -> Self {
        if self.is_zero_vartime() {
            return rhs;
        }
        NonZero::new_unchecked(self)
            .gcd_unsigned_vartime(rhs)
            .get_copy()
    }
}

impl NonZero<Limb> {
    /// Compute the greatest common divisor of `self` and `rhs`.
    #[must_use]
    pub const fn gcd_unsigned(self, rhs: Limb) -> Self {
        let lhs = self.get_copy();

        // Note the following two GCD identity rules:
        // 1) gcd(2a, 2b) = 2 · gcd(a, b), and
        // 2) gcd(a, 2b) = gcd(a, b) if a is odd.
        //
        // Combined, these rules imply that
        // 3) gcd(2^i · a, 2^j · b) = 2^k · gcd(a, b), with k = min(i, j).
        //
        // However, to save ourselves having to divide out 2^j, we also note that
        // 4) 2^k · gcd(a, b) = 2^k · gcd(a, 2^j · b)

        let i = lhs.trailing_zeros();
        let j = rhs.trailing_zeros();
        let k = u32_min(i, j);

        let odd_lhs = Odd::new_unchecked(lhs.shr(i));
        let gcd_div_2k = odd_lhs.gcd_unsigned(rhs);
        NonZero::new_unchecked(gcd_div_2k.as_ref().shl(k))
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[must_use]
    pub const fn gcd_unsigned_vartime(self, rhs: Limb) -> Self {
        let lhs = self.as_ref();

        let i = lhs.trailing_zeros();
        let j = rhs.trailing_zeros();
        let k = u32_min(i, j);

        let odd_lhs = Odd::new_unchecked(lhs.shr(i));
        let gcd_div_2k = odd_lhs.gcd_unsigned_vartime(rhs);
        NonZero::new_unchecked(gcd_div_2k.get_copy().shl(k))
    }
}

impl Odd<Limb> {
    /// The minimal number of binary GCD iterations required to guarantee successful completion.
    pub(crate) const MINIMAL_BINGCD_ITERATIONS: u32 = 2 * Self::BITS - 1;

    /// Compute the greatest common divisor of `self` and `rhs`.
    #[inline]
    pub(crate) const fn gcd_unsigned(self, rhs: Limb) -> Self {
        self.bingcd(rhs).0
    }

    /// Computes `gcd(self, rhs)`, leveraging (a constant time implementation of) the classic
    /// Binary GCD algorithm.
    ///
    /// Ref: Pornin, Optimized Binary GCD for Modular Inversion, Algorithm 1.
    /// <https://eprint.iacr.org/2020/972.pdf>
    ///
    /// This method returns a pair consisting of the GCD and the sign of the Jacobi symbol,
    /// 0 for positive and 1 for negative.
    #[inline(always)]
    pub(super) const fn bingcd(self, rhs: Limb) -> (Self, Word) {
        // (self, rhs) corresponds to (m, y) in the Algorithm 1 notation.
        let (mut a, mut b) = (rhs, self.get_copy());
        let mut i = 0;
        let mut jacobi_neg = 0;

        while i < Self::MINIMAL_BINGCD_ITERATIONS {
            jacobi_neg ^= bingcd_step(&mut a, &mut b).2;
            i += 1;
        }

        let gcd = b
            .to_odd()
            .expect_copied("gcd of an odd value with something else is always odd");

        (gcd, jacobi_neg)
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[inline]
    pub(crate) const fn gcd_unsigned_vartime(self, rhs: Limb) -> Self {
        self.bingcd_vartime(rhs).0
    }

    /// Variable time equivalent of [`Self::bingcd`].
    #[inline(always)]
    pub(super) const fn bingcd_vartime(self, rhs: Limb) -> (Self, Word) {
        // (self, rhs) corresponds to (m, y) in the Algorithm 1 notation.
        let (mut a, mut b) = (rhs, self.get_copy());
        let mut jacobi_neg = 0;

        while !a.is_zero_vartime() {
            jacobi_neg ^= bingcd_step(&mut a, &mut b).2;
        }

        let gcd = b
            .to_odd()
            .expect_copied("gcd of an odd value with something else is always odd");

        (gcd, jacobi_neg)
    }
}

impl Gcd for Limb {
    type Output = Limb;

    fn gcd(&self, rhs: &Self) -> Self::Output {
        Limb::gcd(*self, *rhs)
    }

    fn gcd_vartime(&self, rhs: &Self) -> Self::Output {
        Limb::gcd_vartime(*self, *rhs)
    }
}

impl Gcd<Limb> for NonZero<Limb> {
    type Output = Limb;

    fn gcd(&self, rhs: &Limb) -> Self::Output {
        self.gcd_unsigned(*rhs).get_copy()
    }

    fn gcd_vartime(&self, rhs: &Limb) -> Self::Output {
        self.gcd_unsigned_vartime(*rhs).get_copy()
    }
}

impl Gcd<Limb> for Odd<Limb> {
    type Output = Limb;

    fn gcd(&self, rhs: &Limb) -> Self::Output {
        self.gcd_unsigned(*rhs).get_copy()
    }

    fn gcd_vartime(&self, rhs: &Limb) -> Self::Output {
        self.gcd_unsigned_vartime(*rhs).get_copy()
    }
}

/// Binary GCD update step.
///
/// This is a condensed, constant time execution of the following algorithm:
/// ```text
/// if a mod 2 == 1
///    if a < b
///        (a, b) ← (b, a)
///    a ← a - b
/// a ← a/2
/// ```
///
/// Note: assumes `b` to be odd. Might yield an incorrect result if this is not the case.
///
/// Ref: Pornin, Algorithm 1, L3-9, <https://eprint.iacr.org/2020/972.pdf>.
#[inline(always)]
pub(crate) const fn bingcd_step(a: &mut Limb, b: &mut Limb) -> (Choice, Choice, Word) {
    let a_b_mod_4 = (a.0 & b.0) & 3;

    let a_odd = a.is_odd();
    let (a_sub_b, borrow) = a.borrowing_sub(Limb::select(Limb::ZERO, *b, a_odd), Limb::ZERO);
    let swap = borrow.is_nonzero();
    *b = Limb::select(*b, *a, swap);
    *a = Limb::select(a_sub_b, a_sub_b.wrapping_neg(), swap).shr1().0;

    // (b|a) = -(a|b) iff a = b = 3 mod 4 (quadratic reciprocity)
    let mut jacobi_neg = word::select(0, a_b_mod_4 & (a_b_mod_4 >> 1) & 1, swap);

    // (2a|b) = -(a|b) iff b = ±3 mod 8
    // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
    let b_lo = b.0;
    jacobi_neg ^= ((b_lo >> 1) ^ (b_lo >> 2)) & 1;

    (a_odd, swap, jacobi_neg)
}

#[cfg(test)]
mod tests {
    use crate::{Gcd, Limb, Odd};

    #[test]
    fn gcd_expected() {
        let f = Odd::<Limb>::new(Limb::from(61u32 * 71)).expect("ensured odd");
        let g = Limb::from(59u32 * 61);

        assert_eq!(Limb::from(61u32), f.gcd(&g));
        assert_eq!(Limb::from(61u32), f.gcd_vartime(&g));

        let f = f.as_nz_ref();
        assert_eq!(Limb::from(61u32), f.gcd(&g));
        assert_eq!(Limb::from(61u32), f.gcd_vartime(&g));

        let f = f.get();
        assert_eq!(Limb::from(61u32), Gcd::gcd(&f, &g));
        assert_eq!(Limb::from(61u32), Gcd::gcd_vartime(&f, &g));
    }
}
