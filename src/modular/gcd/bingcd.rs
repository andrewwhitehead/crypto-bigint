use crate::{
    Choice, JacobiSymbol, Limb, Odd, Uint, UintRef, WideWord, Word,
    primitives::{u32_max, u32_min},
    word,
};
use compact::compact_pair;

#[cfg(feature = "alloc")]
mod boxed;
mod matrix;
mod vartime;

pub(super) use matrix::BingcdMatrix;
pub(super) mod compact;

/// The minimal number of binary GCD iterations required to guarantee successful completion.
pub const fn iterations(bits_precision: u32) -> u32 {
    2 * bits_precision - 1
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
#[allow(clippy::cast_possible_truncation)]
const fn bingcd_step_word(mut a: Word, mut b: Word) -> ((Word, Word), Choice, Choice, Word) {
    let a_b = a & b;

    let a_odd = word::choice_from_lsb(a);
    let (a_sub_b, borrow) = a.overflowing_sub(word::select(0, b, a_odd));
    let swap = Choice::from_u8_lsb(borrow as u8);
    b = word::select(b, a, swap);
    a = word::select(a_sub_b, a_sub_b.wrapping_neg(), swap) >> 1;

    // (b|a) = -(a|b) iff a = b = 3 mod 4 (quadratic reciprocity)
    let mut jacobi_neg = word::select(0, a_b & (a_b >> 1) & 1, swap);

    // (2a|b) = -(a|b) iff b = ±3 mod 8
    // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
    jacobi_neg ^= ((b >> 1) ^ (b >> 2)) & 1;

    ((a, b), a_odd, swap, jacobi_neg)
}

/// [`WideWord`] variant of `bingcd_step`.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn bingcd_step_wideword(
    mut a: WideWord,
    mut b: WideWord,
) -> ((WideWord, WideWord), Choice, Choice, Word) {
    let a_b = a as Word & b as Word;

    let a_odd = word::choice_from_lsb_wide(a);
    let (a_sub_b, borrow) = a.overflowing_sub(word::select_wide(0, b, a_odd));
    let swap = Choice::from_u8_lsb(borrow as u8);
    b = word::select_wide(b, a, swap);
    a = word::select_wide(a_sub_b, a_sub_b.wrapping_neg(), swap) >> 1;

    // (b|a) = -(a|b) iff a = b = 3 mod 4 (quadratic reciprocity)
    let mut jacobi_neg = word::select(0, a_b & (a_b >> 1) & 1, swap);

    // (2a|b) = -(a|b) iff b = ±3 mod 8
    // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
    let b_lo = b as Word;
    jacobi_neg ^= ((b_lo >> 1) ^ (b_lo >> 2)) & 1;

    ((a, b), a_odd, swap, jacobi_neg)
}

#[inline(always)]
pub(crate) const fn bingcd_word_vartime(mut a: Word, mut b: Word) -> (Word, Word) {
    debug_assert!(a & b & 1 == 1, "inputs must be odd");
    let mut jacobi_neg = 0;

    loop {
        let (diff, swap) = a.overflowing_sub(b);
        (a, b) = if swap {
            let a_b = a & b;
            jacobi_neg ^= a_b & (a_b >> 1) & 1;
            (diff.wrapping_neg(), a)
        } else {
            (diff, b)
        };

        if a == 0 {
            break;
        }

        let tz = a.trailing_zeros();
        if tz & 1 == 1 {
            // (2a|b) = -(a|b) iff b = ±3 mod 8
            // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
            jacobi_neg ^= ((b >> 1) ^ (b >> 2)) & 1;
        }
        a >>= tz;
    }

    (b, jacobi_neg)
}

#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn bingcd_wideword_vartime(mut a: WideWord, mut b: WideWord) -> (WideWord, Word) {
    debug_assert!(a & b & 1 == 1, "inputs must be odd");
    let mut jacobi_neg = 0;

    loop {
        if (a | b).leading_zeros() >= Word::BITS {
            let (b, j_neg) = bingcd_word_vartime(a as Word, b as Word);
            return (b as WideWord, jacobi_neg ^ j_neg);
        }

        let (diff, swap) = a.overflowing_sub(b);
        (a, b) = if swap {
            let a_b = a as Word & b as Word;
            jacobi_neg ^= a_b & (a_b >> 1) & 1;
            (diff.wrapping_neg(), a)
        } else {
            (diff, b)
        };

        if a == 0 {
            break;
        }

        let tz = a.trailing_zeros();
        if tz & 1 == 1 {
            // (2a|b) = -(a|b) iff b = ±3 mod 8
            // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
            let b_lo = b as Word;
            jacobi_neg ^= ((b_lo >> 1) ^ (b_lo >> 2)) & 1;
        }
        a >>= tz;
    }

    (b, jacobi_neg)
}

/// Computes `gcd(self, rhs)`, leveraging (a constant time implementation of) the classic
/// Binary GCD algorithm.
///
/// This method returns a pair consisting of the GCD and the sign of the Jacobi symbol,
/// 0 for positive and 1 for negative.
#[inline(always)]
pub(crate) const fn bingcd_word(lhs: Word, rhs: Word) -> (Word, Word) {
    // (self, rhs) corresponds to (m, y) in the Algorithm 1 notation.
    let (mut a, mut b) = (lhs, rhs);
    let mut i = 0;
    let mut jacobi_neg = 0;

    while i < iterations(Word::BITS) {
        let j_neg;
        ((a, b), _, _, j_neg) = bingcd_step_word(a, b);
        jacobi_neg ^= j_neg;
        i += 1;
    }

    (b, jacobi_neg)
}

/// Computes `gcd(self, rhs)`, leveraging (a constant time implementation of) the classic
/// Binary GCD algorithm.
///
/// This method returns a pair consisting of the GCD and the sign of the Jacobi symbol,
/// 0 for positive and 1 for negative.
#[inline(always)]
pub(crate) const fn bingcd_wideword(lhs: WideWord, rhs: WideWord) -> (WideWord, Word) {
    // (self, rhs) corresponds to (m, y) in the Algorithm 1 notation.
    let (mut a, mut b) = (lhs, rhs);
    let mut i = 0;
    let mut jacobi_neg = 0;

    while i < iterations(WideWord::BITS) {
        let j_neg;
        ((a, b), _, _, j_neg) = bingcd_step_wideword(a, b);
        jacobi_neg ^= j_neg;
        i += 1;
    }

    (b, jacobi_neg)
}

#[inline(always)]
pub(crate) const fn partial_xgcd<const HALT_AT_ZERO: bool>(
    mut a: WideWord,
    mut b: WideWord,
    mut batch: u32,
) -> (BingcdMatrix, Word) {
    debug_assert!(b & 1 == 1, "b must be odd");
    let mut jacobi_neg = 0;
    let mut matrix = BingcdMatrix::UNIT;
    let mut a_nz = Choice::FALSE;

    while batch > 0 {
        if HALT_AT_ZERO {
            a_nz = a_nz.and(word::choice_from_wide_nz(a));
        }
        let (next, a_odd, swap, j_neg) = bingcd_step_wideword(a, b);
        (a, b) = next;

        // Swap if a odd and a < b
        matrix.conditional_swap_rows(swap);
        // Subtract b from a when a is odd
        matrix.conditional_subtract_bottom_row_from_top(a_odd);
        // Double the bottom row of the matrix when a was ≠ 0 and when not halting.
        if HALT_AT_ZERO {
            matrix.conditional_double_bottom_row(a_nz);
        } else {
            matrix.double_bottom_row(1);
        }
        jacobi_neg ^= j_neg;
        batch -= 1;
    }

    (matrix, jacobi_neg)
}

#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn partial_xgcd_vartime(
    mut a: WideWord,
    mut b: WideWord,
) -> (BingcdMatrix, (WideWord, WideWord), Word) {
    debug_assert!(a & b & 1 == 1, "inputs must be odd");
    let mut jacobi_neg = 0;
    let mut matrix = BingcdMatrix::UNIT;
    let mut n = Word::BITS - 1;

    while n > 0 {
        let (diff, swap) = a.overflowing_sub(b);
        (a, b) = if swap {
            let a_b = a as Word & b as Word;
            jacobi_neg ^= a_b & (a_b >> 1) & 1;
            matrix.swap_rows();
            (diff.wrapping_neg(), a)
        } else {
            (diff, b)
        };
        matrix.subtract_bottom_row_from_top();
        n -= 1;

        if a == 0 {
            break;
        }

        let tz = a.trailing_zeros();
        let tz = if tz > n { n } else { tz };
        if tz & 1 == 1 {
            // (2a|b) = -(a|b) iff b = ±3 mod 8
            // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
            let b_lo = b as Word;
            jacobi_neg ^= ((b_lo >> 1) ^ (b_lo >> 2)) & 1;
        }
        a >>= tz;
        matrix.double_bottom_row(tz);
        n -= tz;
    }

    (matrix, (a, b), jacobi_neg)
}

/// Computes `gcd(self, rhs)`, leveraging the optimized Binary GCD algorithm.
///
/// Note: this algorithm becomes more efficient than the classical algorithm for [Uint]s with
/// relatively many `LIMBS`. A best-effort threshold is presented in [`Self::bingcd`].
///
/// Note: the full algorithm has an additional parameter; this function selects the best-effort
/// value for this parameter. You might be able to further tune your performance by calling the
/// [`Self::optimized_bingcd`_] function directly.
///
/// Ref: Pornin, Optimized Binary GCD for Modular Inversion, Algorithm 2.
/// <https://eprint.iacr.org/2020/972.pdf>
#[inline(always)]
pub(crate) const fn optimized_bingcd(a: &mut UintRef, b: &mut UintRef, batch_max: u32) -> Word {
    debug_assert!(a.bits_precision() == b.bits_precision());
    debug_assert!(b.is_odd().to_bool_vartime());

    let mut steps = iterations(a.bits_precision());
    let mut jacobi_neg = 0;

    while steps > 0 {
        let batch = u32_min(steps, batch_max);
        steps -= batch;

        // Construct a_ and b_ as the summary of a and b, respectively.
        let n = u32_max(2 * Limb::BITS, u32_max(a.bits(), b.bits()));
        let (a_, b_) = compact_pair(a, b, n);

        // Compute the batch update matrix from a_ and b_.
        let (matrix, j_neg) = partial_xgcd::<false>(a_, b_, batch);
        matrix.apply_unsigned(a, b);
        jacobi_neg ^= j_neg;
    }

    jacobi_neg
}

impl<const LIMBS: usize> Uint<LIMBS> {
    /// Calculate the greatest common denominator of `a`, and `b`.
    #[inline(always)]
    pub(crate) const fn bingcd_vartime(&self, rhs: &Self) -> Self {
        let (mut a, mut b) = (*self, *rhs);
        let index = vartime::gcd_vartime(a.as_mut_uint_ref(), b.as_mut_uint_ref());
        if index { b } else { a }
    }

    /// Computes the Jacobi symbol `(self|rhs)` using binary GCD.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn bingcd_jacobi_symbol(&self, rhs: &Odd<Self>) -> JacobiSymbol {
        let (jacobi_neg, gcd_one) = if const { LIMBS == 1 } {
            let (gcd, j_neg) = bingcd_word(self.limbs[0].0, rhs.as_ref().limbs[0].0);
            (j_neg, word::choice_from_eq(gcd, 1))
        } else if const { LIMBS == 2 } {
            let (gcd, j_neg) = bingcd_wideword(
                self.as_uint_ref().to_wide_word_unchecked(),
                rhs.as_ref().as_uint_ref().to_wide_word_unchecked(),
            );
            (j_neg, word::choice_from_wide_eq(gcd, 1))
        } else {
            let (mut a, mut b) = (*self, *rhs.as_ref());
            let j_neg = optimized_bingcd(a.as_mut_uint_ref(), b.as_mut_uint_ref(), Limb::BITS - 2);
            (j_neg, Uint::eq(&b, &Uint::ONE))
        };

        JacobiSymbol::from_sign(jacobi_neg).conditional_set_zero(gcd_one.not())
    }

    /// Computes the Jacobi symbol `(a|b)` using binary GCD.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn bingcd_jacobi_symbol_vartime<const RHS_LIMBS: usize>(
        &self,
        rhs: &Odd<Uint<RHS_LIMBS>>,
    ) -> JacobiSymbol {
        let (mut a, mut b) = (*self, *rhs.as_ref());
        vartime::jacobi_symbol_vartime(a.as_mut_uint_ref(), b.as_mut_uint_ref())
    }
}

impl<const LIMBS: usize> Odd<Uint<LIMBS>> {
    /// Compute the greatest common divisor of `self` and `rhs` using the Binary GCD algorithm.
    ///
    /// This function switches between the "classic" and "optimized" algorithm at a best-effort
    /// threshold. When using [Uint]s with `LIMBS` close to the threshold, it may be useful to
    /// manually test whether the classic or optimized algorithm is faster for your machine.
    #[doc(hidden)]
    #[inline(always)]
    #[must_use]
    // TODO: remove from public API (already undocumented)
    pub const fn bingcd(&self, rhs: &Uint<LIMBS>) -> Self {
        if const { LIMBS == 1 } {
            // Classic binary GCD for a single word
            let (gcd, _) = bingcd_word(rhs.limbs[0].0, self.as_ref().limbs[0].0);
            Uint::from_word(gcd)
                .to_odd()
                .expect_copied("expected odd gcd")
        } else if const { LIMBS == 2 } {
            // Classic binary GCD for a wide word
            let (gcd, _) = bingcd_wideword(
                rhs.as_uint_ref().to_wide_word_unchecked(),
                self.as_ref().as_uint_ref().to_wide_word_unchecked(),
            );
            Uint::from_wide_word(gcd)
                .to_odd()
                .expect_copied("expected odd gcd")
        } else {
            // Optimized binary GCD
            let (mut a, mut b) = (*rhs, *self.as_ref());
            optimized_bingcd(a.as_mut_uint_ref(), b.as_mut_uint_ref(), Limb::BITS - 1);
            debug_assert!(a.is_zero_vartime());
            b.to_odd()
                .expect_copied("gcd of an odd value is always odd")
        }
    }

    /// Computes the multiplicative inverse of `value` mod `self`.
    #[must_use]
    pub const fn bingcd_invert_mod_vartime(&self, value: &Uint<LIMBS>) -> Option<Uint<LIMBS>> {
        let mut a = *value;
        let m_inv = self.as_uint_ref().invert_mod_limb();
        let mut buf = [[Limb::ZERO; LIMBS]; 3];
        let buf = UintRef::new_flattened_mut(&mut buf);

        let res = vartime::invert_vartime(a.as_mut_uint_ref(), self.as_uint_ref(), m_inv, buf);
        if res { Some(a) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Int, Uint};

    fn bingcd_test<const LIMBS: usize>(lhs: Uint<LIMBS>, rhs: Uint<LIMBS>, target: Uint<LIMBS>) {
        assert_eq!(lhs.to_odd().unwrap().bingcd(&rhs), target.to_odd().unwrap());
    }

    fn bingcd_tests<const LIMBS: usize>() {
        bingcd_test(Uint::<LIMBS>::ONE, Uint::ZERO, Uint::ONE);
        bingcd_test(Uint::<LIMBS>::ONE, Uint::ONE, Uint::ONE);
        bingcd_test(Uint::<LIMBS>::ONE, Int::MAX.abs(), Uint::ONE);
        bingcd_test(Uint::<LIMBS>::ONE, Int::MIN.abs(), Uint::ONE);
        bingcd_test(Uint::<LIMBS>::ONE, Uint::MAX, Uint::ONE);
        bingcd_test(Int::<LIMBS>::MAX.abs(), Uint::ZERO, Int::MAX.abs());
        bingcd_test(Int::<LIMBS>::MAX.abs(), Uint::ONE, Uint::ONE);
        bingcd_test(Int::<LIMBS>::MAX.abs(), Int::MAX.abs(), Int::MAX.abs());
        bingcd_test(Int::<LIMBS>::MAX.abs(), Int::MIN.abs(), Uint::ONE);
        bingcd_test(Int::<LIMBS>::MAX.abs(), Uint::MAX, Uint::ONE);
        bingcd_test(Uint::<LIMBS>::MAX, Uint::ZERO, Uint::MAX);
        bingcd_test(Uint::<LIMBS>::MAX, Uint::ONE, Uint::ONE);
        bingcd_test(Uint::<LIMBS>::MAX, Int::MAX.abs(), Uint::ONE);
        bingcd_test(Uint::<LIMBS>::MAX, Int::MIN.abs(), Uint::ONE);
        bingcd_test(Uint::<LIMBS>::MAX, Uint::MAX, Uint::MAX);
    }

    #[test]
    fn test_bingcd() {
        bingcd_tests::<1>();
        bingcd_tests::<2>();
        bingcd_tests::<3>();
        bingcd_tests::<4>();
        bingcd_tests::<8>();
        if !cfg!(miri) {
            bingcd_tests::<16>();
            bingcd_tests::<64>();
        }
    }
}
