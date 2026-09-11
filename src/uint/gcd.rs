//! This module implements Binary (Extended) GCD for [`Uint`].

use crate::{
    Choice, Gcd, Int, Limb, NonZeroUint, Odd, OddUint, Uint, UintRef, Xgcd, modular::gcd,
    primitives::u32_min,
};

impl<const LIMBS: usize> Uint<LIMBS> {
    /// Compute the greatest common divisor of `self` and `rhs`.
    #[must_use]
    pub const fn gcd(&self, rhs: &Self) -> Self {
        let (self_nz, self_is_nz) = self.to_nz_or_one();
        let gcd = self_nz.gcd_unsigned(rhs).get_copy();
        Uint::select(rhs, &gcd, self_is_nz)
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[must_use]
    pub const fn gcd_vartime(&self, rhs: &Self) -> Self {
        let (mut a, mut b) = (*self, *rhs);
        let gcd_is_a = gcd::gcd_vartime(a.as_mut_uint_ref(), b.as_mut_uint_ref());
        if gcd_is_a { a } else { b }
    }

    /// Executes the Extended GCD algorithm.
    ///
    /// Given `(self, rhs)`, computes `(g, x, y)`, s.t. `self * x + rhs * y = g = gcd(self, rhs)`.
    #[must_use]
    pub const fn xgcd(&self, rhs: &Self) -> UintXgcdOutput<LIMBS> {
        let (self_nz, self_is_nz) = self.to_nz_or_one();
        let nz_output = self_nz.xgcd_unsigned(rhs);
        nz_output.to_uint_output(rhs, self_is_nz)
    }

    /// Executes the Extended GCD algorithm.
    ///
    /// Given `(self, rhs)`, computes `(g, x, y)`, s.t. `self * x + rhs * y = g = gcd(self, rhs)`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[must_use]
    pub fn xgcd_vartime(&self, rhs: &Self) -> UintXgcdOutput<LIMBS> {
        let (self_nz, self_is_nz) = self.to_nz_or_one();
        let nz_output = self_nz.xgcd_unsigned_vartime(rhs);
        nz_output.to_uint_output(rhs, self_is_nz)
    }
}

impl<const LIMBS: usize> NonZeroUint<LIMBS> {
    /// Compute the greatest common divisor of `self` and `rhs`.
    #[must_use]
    pub const fn gcd_unsigned(&self, rhs: &Uint<LIMBS>) -> Self {
        let lhs = self.as_ref();

        // Factor out common factors of two and ensure `f` is odd.
        // There is no need to strip the zeros from `g`, as
        // `gcd(a, b) == gcd(a, b•2^k)` for odd `a`.
        let lhs_tz = lhs.trailing_zeros();
        let k = u32_min(lhs_tz, rhs.trailing_zeros());

        let odd_lhs = Odd::new_unchecked(lhs.shr(lhs_tz));
        let gcd_div_2k = odd_lhs.gcd_unsigned(rhs);
        gcd_div_2k
            .as_ref()
            .shl(k)
            .to_nz()
            .expect_copied("expected non-zero GCD")
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[must_use]
    pub const fn gcd_unsigned_vartime(&self, rhs: &Uint<LIMBS>) -> Self {
        self.as_ref()
            .gcd_vartime(rhs)
            .to_nz()
            .expect_copied("expected non-zero GCD")
    }

    /// Execute the Extended GCD algorithm.
    ///
    /// Given `(self, rhs)`, computes `(g, x, y)` s.t. `self * x + rhs * y = g = gcd(self, rhs)`.
    #[must_use]
    pub const fn xgcd(&self, rhs: &Self) -> NonZeroUintXgcdOutput<LIMBS> {
        self.xgcd_unsigned(rhs.as_ref())
    }

    /// Execute the Extended GCD algorithm.
    ///
    /// Given `(self, rhs)`, computes `(g, x, y)` s.t. `self * x + rhs * y = g = gcd(self, rhs)`.
    #[must_use]
    pub const fn xgcd_unsigned(&self, rhs: &Uint<LIMBS>) -> NonZeroUintXgcdOutput<LIMBS> {
        let (mut lhs, mut rhs) = (*self.as_ref(), *rhs);

        // Observe that gcd(2^i · a, 2^j · b) = 2^k * gcd(2^(i-k)·a, 2^(j-k)·b), with k = min(i,j).
        let i = lhs.trailing_zeros();
        let j = rhs.trailing_zeros();
        let k = u32_min(i, j);
        lhs = lhs.shr(k);
        rhs = rhs.shr(k);

        // At this point, either lhs or rhs is odd (or both); swap to make sure lhs is odd.
        let swap = Choice::from_u32_lt(j, i);
        Uint::conditional_swap(&mut lhs, &mut rhs, swap);
        let lhs = lhs.to_odd().expect_copied("odd by construction");

        lhs.xgcd_unsigned(&rhs).to_nz_output(k, swap)
    }

    /// Execute the Extended GCD algorithm.
    ///
    /// Given `(self, rhs)`, computes `(g, x, y)` s.t. `self * x + rhs * y = g = gcd(self, rhs)`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[must_use]
    pub fn xgcd_unsigned_vartime(&self, rhs: &Uint<LIMBS>) -> NonZeroUintXgcdOutput<LIMBS> {
        let (mut lhs, mut rhs) = (*self.as_ref(), *rhs);

        let i = lhs.trailing_zeros_vartime();
        let j = rhs.trailing_zeros_vartime();
        let k = u32_min(i, j);
        lhs = lhs.shr_vartime(k);
        rhs = rhs.shr_vartime(k);

        // At this point, either lhs or rhs is odd (or both); swap to make sure lhs is odd.
        let swap = Choice::from_u32_lt(j, i);
        Uint::conditional_swap(&mut lhs, &mut rhs, swap);
        let lhs = lhs.to_odd().expect_copied("odd by construction");

        lhs.xgcd_unsigned_vartime(&rhs).to_nz_output(k, swap)
    }
}

impl<const LIMBS: usize> OddUint<LIMBS> {
    /// Compute the greatest common divisor of `self` and `rhs`.
    #[inline(always)]
    #[must_use]
    pub const fn gcd_unsigned(&self, rhs: &Uint<LIMBS>) -> Self {
        let mut a = *rhs;
        let mut b = self.get_copy();

        if const { LIMBS <= gcd::SMALL_THRESHOLD_LIMBS } {
            gcd::gcd_odd_small(a.as_mut_uint_ref(), b.as_mut_uint_ref());
        } else {
            gcd::gcd_odd(a.as_mut_uint_ref(), b.as_mut_uint_ref());
        }

        b.to_odd().expect_copied("expected odd GCD")
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[inline(always)]
    #[must_use]
    pub const fn gcd_unsigned_vartime(&self, rhs: &Uint<LIMBS>) -> Self {
        self.as_ref()
            .gcd_vartime(rhs)
            .to_odd()
            .expect_copied("expected odd GCD")
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    #[deprecated(since = "0.8.0")]
    #[doc(hidden)]
    #[must_use]
    pub const fn bingcd(&self, rhs: &Uint<LIMBS>) -> Self {
        self.gcd_unsigned(rhs)
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[deprecated(since = "0.8.0")]
    #[doc(hidden)]
    #[must_use]
    pub const fn bingcd_vartime(&self, rhs: &Uint<LIMBS>) -> Self {
        self.gcd_unsigned_vartime(rhs)
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    #[deprecated(since = "0.8.0")]
    #[doc(hidden)]
    #[must_use]
    pub const fn safegcd(&self, rhs: &Uint<LIMBS>) -> Self {
        self.gcd_unsigned(rhs)
    }

    /// Compute the greatest common divisor of `self` and `rhs`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[deprecated(since = "0.8.0")]
    #[doc(hidden)]
    #[must_use]
    pub const fn safegcd_vartime(&self, rhs: &Uint<LIMBS>) -> Self {
        self.gcd_unsigned_vartime(rhs)
    }

    /// Execute the Extended GCD algorithm.
    ///
    /// Given `(self, rhs)`, computes `(g, x, y)` s.t. `self * x + rhs * y = g = gcd(self, rhs)`.
    #[inline]
    #[must_use]
    pub const fn xgcd(&self, rhs: &Self) -> OddUintXgcdOutput<LIMBS> {
        self.xgcd_unsigned(rhs.as_ref())
    }

    /// Execute the Extended GCD algorithm.
    /// Given `(self, rhs)`, computes `(g, x, y)` s.t. `self * x + rhs * y = g = gcd(self, rhs)`.
    #[must_use]
    pub const fn xgcd_unsigned(&self, rhs: &Uint<LIMBS>) -> OddUintXgcdOutput<LIMBS> {
        let mut x = *rhs;
        let mut y = self.get_copy();
        let mut gcd = Uint::<LIMBS>::ZERO;
        let mut buf = [[Limb::ZERO; LIMBS]; 3];
        let buf = UintRef::new_flattened_mut(&mut buf);
        let mut a = Uint::<LIMBS>::ZERO;
        let mut b = Uint::<LIMBS>::ONE;

        gcd::xgcd_odd(
            x.as_mut_uint_ref(),
            y.as_mut_uint_ref(),
            gcd.as_mut_uint_ref(),
            a.as_mut_uint_ref(),
            b.as_mut_uint_ref(),
            buf,
        );
        let gcd = gcd.to_odd().expect_copied("expected odd GCD");

        // `a` was specifically chosen to be non-zero so that `b` is positive, `ax - by = gcd`
        // we minimize both `|a|` and `|b|` by allowing for negative `a`, and conditionally
        // subtracting `y/gcd` from `a` when `2•a > y/gcd`
        let (a_dbl, a_dbl_hi) = a.shl1_with_carry(Limb::ZERO);
        let swap = a_dbl_hi.is_nonzero().or(Uint::gt(&a_dbl, &y));
        let a = Uint::select(&a, &a.wrapping_sub(&y), swap);
        let b = Uint::select(&b.wrapping_neg(), &x.wrapping_sub(&b), swap);

        OddUintXgcdOutput {
            gcd,
            lhs_on_gcd: y,
            rhs_on_gcd: x,
            x: *b.as_int(),
            y: *a.as_int(),
        }
    }

    /// Execute the Extended GCD algorithm.
    /// Given `(self, rhs)`, computes `(g, x, y)` s.t. `self * x + rhs * y = g = gcd(self, rhs)`.
    ///
    /// Executes in variable time w.r.t. all input parameters.
    #[must_use]
    pub fn xgcd_unsigned_vartime(&self, rhs: &Uint<LIMBS>) -> OddUintXgcdOutput<LIMBS> {
        let mut x = *rhs;
        let mut y = self.get_copy();
        let mut gcd = Uint::<LIMBS>::ZERO;
        let mut buf = [[Limb::ZERO; LIMBS]; 3];
        let buf = UintRef::new_flattened_mut(&mut buf);
        let mut a = Uint::<LIMBS>::ZERO;
        let mut b = Uint::<LIMBS>::ONE;

        gcd::xgcd_vartime(
            x.as_mut_uint_ref(),
            y.as_mut_uint_ref(),
            gcd.as_mut_uint_ref(),
            a.as_mut_uint_ref(),
            b.as_mut_uint_ref(),
            buf,
        );
        let gcd = gcd.to_odd().expect_copied("expected odd GCD");

        // Mirrors `safexgcd`'s exact sign-minimization post-processing (same arithmetic, vartime
        // comparisons here): `a` was specifically chosen to be non-zero so that `b` is positive;
        // minimize both `|a|` and `|b|` by allowing negative `a`, conditionally subtracting
        // `y/gcd` from `a` when `2*a > y/gcd`.
        let (a_dbl, a_dbl_hi) = a.shl1_with_carry(Limb::ZERO);
        let swap = a_dbl_hi.is_nonzero().to_bool_vartime() || a_dbl.cmp_vartime(&y).is_gt();
        let (a, b) = if swap {
            (a.wrapping_sub(&y), x.wrapping_sub(&b))
        } else {
            (a, b.wrapping_neg())
        };

        OddUintXgcdOutput {
            gcd,
            lhs_on_gcd: y,
            rhs_on_gcd: x,
            x: *b.as_int(),
            y: *a.as_int(),
        }
    }
}

pub type UintXgcdOutput<const LIMBS: usize> = XgcdOutput<LIMBS, Uint<LIMBS>>;
pub type NonZeroUintXgcdOutput<const LIMBS: usize> = XgcdOutput<LIMBS, NonZeroUint<LIMBS>>;
pub type OddUintXgcdOutput<const LIMBS: usize> = XgcdOutput<LIMBS, OddUint<LIMBS>>;

/// Container for the processed output of the Binary XGCD algorithm.
#[derive(Debug, Copy, Clone)]
pub struct XgcdOutput<const LIMBS: usize, GCD: Copy> {
    /// Greatest common divisor
    pub gcd: GCD,
    /// x;
    pub x: Int<LIMBS>,
    /// y;
    pub y: Int<LIMBS>,
    /// lhs / gcd
    pub lhs_on_gcd: Uint<LIMBS>,
    /// rhs / gcd
    pub rhs_on_gcd: Uint<LIMBS>,
}

impl<const LIMBS: usize, GCD: Copy> XgcdOutput<LIMBS, GCD> {
    /// The greatest common divisor stored in this object.
    pub const fn gcd(&self) -> GCD {
        self.gcd
    }

    /// Obtain a copy of the Bézout coefficients.
    pub const fn bezout_coefficients(&self) -> (Int<LIMBS>, Int<LIMBS>) {
        (self.x, self.y)
    }

    /// Obtain a copy of the quotients `lhs/gcd` and `rhs/gcd`.
    pub const fn quotients(&self) -> (Uint<LIMBS>, Uint<LIMBS>) {
        (self.lhs_on_gcd, self.rhs_on_gcd)
    }
}

impl<const LIMBS: usize> NonZeroUintXgcdOutput<LIMBS> {
    #[inline(always)]
    pub(crate) const fn to_uint_output(
        self,
        rhs: &Uint<LIMBS>,
        lhs_is_nz: Choice,
    ) -> UintXgcdOutput<LIMBS> {
        UintXgcdOutput {
            // Correct the gcd in case lhs was zero
            gcd: Uint::select(rhs, self.gcd.as_ref(), lhs_is_nz),
            // Correct the Bézout coefficients in case lhs was zero.
            x: Int::select(&Int::ZERO, &self.x, lhs_is_nz),
            y: Int::select(&Int::ONE, &self.y, lhs_is_nz),
            // Correct the quotients in case lhs was zero.
            lhs_on_gcd: Uint::select(&Uint::ZERO, &self.lhs_on_gcd, lhs_is_nz),
            rhs_on_gcd: Uint::select(&Uint::ONE, &self.rhs_on_gcd, lhs_is_nz),
        }
    }
}

impl<const LIMBS: usize> OddUintXgcdOutput<LIMBS> {
    #[inline(always)]
    pub(crate) const fn to_nz_output(self, k: u32, swap: Choice) -> NonZeroUintXgcdOutput<LIMBS> {
        let Self {
            ref gcd,
            mut x,
            mut y,
            mut lhs_on_gcd,
            mut rhs_on_gcd,
        } = self;

        // Apply the removed factor 2^k back to the gcd
        let gcd = gcd
            .as_ref()
            .shl(k)
            .to_nz()
            .expect_copied("is non-zero by construction");
        Int::conditional_swap(&mut x, &mut y, swap);
        Uint::conditional_swap(&mut lhs_on_gcd, &mut rhs_on_gcd, swap);

        NonZeroUintXgcdOutput {
            gcd,
            x,
            y,
            lhs_on_gcd,
            rhs_on_gcd,
        }
    }
}

macro_rules! impl_gcd {
    ($slf:ty, [$($rhs:ty),+]) => {
        $(
            impl_gcd!($slf, $rhs, $rhs);
        )+
    };
    ($slf:ty, $rhs:ty, $out:ty) => {
        impl<const LIMBS: usize> Gcd<$rhs> for $slf {
            type Output = $out;

            #[inline]
            fn gcd(&self, rhs: &$rhs) -> Self::Output {
                rhs.gcd(self)
            }

            #[inline]
            fn gcd_vartime(&self, rhs: &$rhs) -> Self::Output {
                rhs.gcd_vartime(self)
            }
        }
    };
}

macro_rules! impl_gcd_unsigned_lhs {
    ($slf:ty, [$($rhs:ty),+]) => {
        $(
            impl_gcd_unsigned_lhs!($slf, $rhs, $slf);
        )+
    };
    ($slf:ty, $rhs:ty, $out:ty) => {
        impl<const LIMBS: usize> Gcd<$rhs> for $slf {
            type Output = $out;

            #[inline]
            fn gcd(&self, rhs: &$rhs) -> Self::Output {
                self.gcd_unsigned(&rhs)
            }

            #[inline]
            fn gcd_vartime(&self, rhs: &$rhs) -> Self::Output {
                self.gcd_unsigned_vartime(&rhs)
            }
        }
    };
}

macro_rules! impl_gcd_unsigned_rhs {
    ($slf:ty, [$($rhs:ty),+]) => {
        $(
            impl_gcd_unsigned_rhs!($slf, $rhs, $rhs);
        )+
    };
    ($slf:ty, $rhs:ty, $out:ty) => {
        impl<const LIMBS: usize> Gcd<$rhs> for $slf {
            type Output = $out;

            #[inline]
            fn gcd(&self, rhs: &$rhs) -> Self::Output {
                rhs.gcd_unsigned(self)
            }

            #[inline]
            fn gcd_vartime(&self, rhs: &$rhs) -> Self::Output {
                rhs.gcd_unsigned_vartime(self)
            }
        }
    };
}

pub(crate) use impl_gcd_unsigned_lhs;
pub(crate) use impl_gcd_unsigned_rhs;

impl_gcd!(
    Uint<LIMBS>,
    [Uint<LIMBS>, NonZeroUint<LIMBS>, OddUint<LIMBS>]
);
impl_gcd_unsigned_lhs!(NonZeroUint<LIMBS>, [Uint<LIMBS>]);
impl_gcd_unsigned_rhs!(
    NonZeroUint<LIMBS>,
    [NonZeroUint<LIMBS>, OddUint<LIMBS>]
);
impl_gcd_unsigned_lhs!(OddUint<LIMBS>, [Uint<LIMBS>, NonZeroUint<LIMBS>, OddUint<LIMBS>]);

impl<const LIMBS: usize> Xgcd for Uint<LIMBS> {
    type Output = UintXgcdOutput<LIMBS>;

    #[inline]
    fn xgcd(&self, rhs: &Uint<LIMBS>) -> Self::Output {
        self.xgcd(rhs)
    }

    #[inline]
    fn xgcd_vartime(&self, rhs: &Uint<LIMBS>) -> Self::Output {
        self.xgcd_vartime(rhs)
    }
}

impl<const LIMBS: usize> Xgcd for NonZeroUint<LIMBS> {
    type Output = NonZeroUintXgcdOutput<LIMBS>;

    #[inline]
    fn xgcd(&self, rhs: &NonZeroUint<LIMBS>) -> Self::Output {
        self.xgcd(rhs)
    }

    #[inline]
    fn xgcd_vartime(&self, rhs: &NonZeroUint<LIMBS>) -> Self::Output {
        self.xgcd_unsigned_vartime(rhs.as_ref())
    }
}

impl<const LIMBS: usize> Xgcd for OddUint<LIMBS> {
    type Output = OddUintXgcdOutput<LIMBS>;

    #[inline]
    fn xgcd(&self, rhs: &OddUint<LIMBS>) -> Self::Output {
        self.xgcd(rhs)
    }

    #[inline]
    fn xgcd_vartime(&self, rhs: &OddUint<LIMBS>) -> Self::Output {
        self.xgcd_unsigned_vartime(rhs.as_ref())
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    mod gcd {
        use crate::{U64, U128, U256, U512, U1024, U2048, U4096, Uint};

        fn test<const LIMBS: usize>(lhs: Uint<LIMBS>, rhs: Uint<LIMBS>, target: Uint<LIMBS>) {
            assert_eq!(lhs.gcd(&rhs), target);
            assert_eq!(lhs.gcd_vartime(&rhs), target);
        }

        fn run_tests<const LIMBS: usize>() {
            test(Uint::<LIMBS>::ZERO, Uint::ZERO, Uint::ZERO);
            test(Uint::<LIMBS>::ZERO, Uint::ONE, Uint::ONE);
            test(Uint::<LIMBS>::ZERO, Uint::MAX, Uint::MAX);
            test(Uint::<LIMBS>::ONE, Uint::ZERO, Uint::ONE);
            test(Uint::<LIMBS>::ONE, Uint::ONE, Uint::ONE);
            test(Uint::<LIMBS>::ONE, Uint::MAX, Uint::ONE);
            test(Uint::<LIMBS>::MAX, Uint::ZERO, Uint::MAX);
            test(Uint::<LIMBS>::MAX, Uint::ONE, Uint::ONE);
            test(Uint::<LIMBS>::MAX, Uint::MAX, Uint::MAX);
        }

        #[test]
        fn gcd_sizes() {
            run_tests::<{ U64::LIMBS }>();
            run_tests::<{ U128::LIMBS }>();
            run_tests::<{ U256::LIMBS }>();
            run_tests::<{ U512::LIMBS }>();
            run_tests::<{ U1024::LIMBS }>();
            run_tests::<{ U2048::LIMBS }>();
            run_tests::<{ U4096::LIMBS }>();
        }
    }

    mod xgcd {
        use crate::{Concat, Int, U64, U128, U256, U512, U1024, U2048, U4096, U8192, U16384, Uint};
        use core::ops::Div;

        fn check<const LIMBS: usize, const DOUBLE: usize>(
            lhs: Uint<LIMBS>,
            rhs: Uint<LIMBS>,
            output: crate::uint::gcd::UintXgcdOutput<LIMBS>,
        ) where
            Uint<LIMBS>: Concat<LIMBS, Output = Uint<DOUBLE>>,
        {
            assert_eq!(output.gcd, lhs.gcd(&rhs));

            if output.gcd > Uint::ZERO {
                assert_eq!(output.lhs_on_gcd, lhs.div(output.gcd.to_nz().unwrap()));
                assert_eq!(output.rhs_on_gcd, rhs.div(output.gcd.to_nz().unwrap()));
            }

            let (x, y) = output.bezout_coefficients();
            assert_eq!(
                x.concatenating_mul_unsigned(&lhs) + y.concatenating_mul_unsigned(&rhs),
                *output.gcd.resize().as_int()
            );
        }

        fn test<const LIMBS: usize, const DOUBLE: usize>(lhs: Uint<LIMBS>, rhs: Uint<LIMBS>)
        where
            Uint<LIMBS>: Concat<LIMBS, Output = Uint<DOUBLE>>,
        {
            check(lhs, rhs, lhs.xgcd(&rhs));
            check(lhs, rhs, lhs.xgcd_vartime(&rhs));
        }

        fn run_tests<const LIMBS: usize, const DOUBLE: usize>()
        where
            Uint<LIMBS>: Concat<LIMBS, Output = Uint<DOUBLE>>,
        {
            let min = Int::MIN.abs();
            test(Uint::ZERO, Uint::ZERO);
            test(Uint::ZERO, Uint::ONE);
            test(Uint::ZERO, min);
            test(Uint::ZERO, Uint::MAX);
            test(Uint::ONE, Uint::ZERO);
            test(Uint::ONE, Uint::ONE);
            test(Uint::ONE, min);
            test(Uint::ONE, Uint::MAX);
            test(min, Uint::ZERO);
            test(min, Uint::ONE);
            test(min, Int::MIN.abs());
            test(min, Uint::MAX);
            test(Uint::MAX, Uint::ZERO);
            test(Uint::MAX, Uint::ONE);
            test(Uint::MAX, min);
            test(Uint::MAX, Uint::MAX);
        }

        #[test]
        fn xgcd_sizes() {
            run_tests::<{ U64::LIMBS }, { U128::LIMBS }>();
            run_tests::<{ U128::LIMBS }, { U256::LIMBS }>();
            run_tests::<{ U256::LIMBS }, { U512::LIMBS }>();
            run_tests::<{ U512::LIMBS }, { U1024::LIMBS }>();
            run_tests::<{ U1024::LIMBS }, { U2048::LIMBS }>();
            run_tests::<{ U2048::LIMBS }, { U4096::LIMBS }>();
            run_tests::<{ U4096::LIMBS }, { U8192::LIMBS }>();
            run_tests::<{ U8192::LIMBS }, { U16384::LIMBS }>();
        }

        #[test]
        fn regression_tests() {
            // Sent in by @kayabaNerve (https://github.com/RustCrypto/crypto-bigint/pull/761#issuecomment-2771564732)
            let a = U256::from_be_hex(
                "000000000000000000000000000000000000001B5DFB3BA1D549DFAF611B8D4C",
            );
            let b = U256::from_be_hex(
                "000000000000345EAEDFA8CA03C1F0F5B578A787FE2D23B82A807F178B37FD8E",
            );
            test(a, b);

            // Sent in by @kayabaNerve (https://github.com/RustCrypto/crypto-bigint/pull/761#issuecomment-2771581512)
            let a = U256::from_be_hex(
                "000000000000000000000000000000000000001A0DEEF6F3AC2566149D925044",
            );
            let b = U256::from_be_hex(
                "000000000000072B69C9DD0AA15F135675EA9C5180CF8FF0A59298CFC92E87FA",
            );
            test(a, b);

            // Sent in by @kayabaNerve (https://github.com/RustCrypto/crypto-bigint/pull/761#issuecomment-2782912608)
            let a = U512::from_be_hex(concat![
                "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364142",
                "4EB38E6AC0E34DE2F34BFAF22DE683E1F4B92847B6871C780488D797042229E1"
            ]);
            let b = U512::from_be_hex(concat![
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD755DB9CD5E9140777FA4BD19A06C8283",
                "9D671CD581C69BC5E697F5E45BCD07C52EC373A8BDC598B4493F50A1380E1281"
            ]);
            test(a, b);
        }
    }

    mod traits {
        use crate::{Gcd, I256, U256};

        #[test]
        fn gcd_relatively_prime() {
            // Two semiprimes with no common factors
            let f = U256::from(59u32 * 67);
            let g = U256::from(61u32 * 71);
            let gcd = f.gcd(&g);
            assert_eq!(gcd, U256::ONE);
        }

        #[test]
        fn gcd_nonprime() {
            let f = U256::from(4391633u32);
            let g = U256::from(2022161u32);
            let gcd = f.gcd(&g);
            assert_eq!(gcd, U256::from(1763u32));
        }

        #[test]
        fn gcd_zero() {
            assert_eq!(U256::ZERO.gcd(&U256::ZERO), U256::ZERO);
            assert_eq!(U256::ZERO.gcd(&U256::ONE), U256::ONE);
            assert_eq!(U256::ONE.gcd(&U256::ZERO), U256::ONE);
        }

        #[test]
        fn gcd_one() {
            let f = U256::ONE;
            assert_eq!(U256::ONE, f.gcd(&U256::ONE));
            assert_eq!(U256::ONE, f.gcd(&U256::from(2u8)));
        }

        #[test]
        fn gcd_two() {
            let f = U256::from_u8(2);
            assert_eq!(f, f.gcd(&f));

            let g = U256::from_u8(4);
            assert_eq!(f, f.gcd(&g));
            assert_eq!(f, g.gcd(&f));
        }

        #[test]
        fn gcd_unsigned_int() {
            // Two numbers with a shared factor of 61
            let f = U256::from(61u32 * 71);
            let g = I256::from(59i32 * 61);

            let sixty_one = U256::from(61u32);
            assert_eq!(sixty_one, <U256 as Gcd<I256>>::gcd(&f, &g));
            assert_eq!(sixty_one, <U256 as Gcd<I256>>::gcd(&f, &g.wrapping_neg()));
        }

        #[test]
        fn xgcd_expected() {
            // Two numbers with a shared factor of 61
            let f = U256::from(61u32 * 71);
            let g = U256::from(59u32 * 61);

            let actual = f.xgcd(&g);
            assert_eq!(U256::from(61u32), actual.gcd);
            assert_eq!(I256::from(5i32), actual.x);
            assert_eq!(I256::from(-6i32), actual.y);

            let actual = f.xgcd_vartime(&g);
            assert_eq!(U256::from(61u32), actual.gcd);
            assert_eq!(I256::from(5i32), actual.x);
            assert_eq!(I256::from(-6i32), actual.y);
        }

        #[test]
        fn xgcd_vartime_nonzero_odd_dispatch() {
            use crate::{NonZeroUint, OddUint, Xgcd};

            let f = U256::from(61u32 * 71).to_nz().unwrap();
            let g = U256::from(59u32 * 61).to_nz().unwrap();
            let expected = f.xgcd(&g);
            let actual = Xgcd::xgcd_vartime(&f, &g);
            assert_eq!(actual.gcd, expected.gcd);
            assert_eq!(actual.bezout_coefficients(), expected.bezout_coefficients());
            let _: NonZeroUint<{ U256::LIMBS }> = actual.gcd;

            let f = U256::from(61u32 * 71).to_odd().unwrap();
            let g = U256::from(59u32 * 61).to_odd().unwrap();
            let expected = f.xgcd(&g);
            let actual = Xgcd::xgcd_vartime(&f, &g);
            assert_eq!(actual.gcd, expected.gcd);
            assert_eq!(actual.bezout_coefficients(), expected.bezout_coefficients());
            let _: OddUint<{ U256::LIMBS }> = actual.gcd;
        }
    }
}
