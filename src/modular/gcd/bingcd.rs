use super::{ExtendedIntRef, SignedLimb, SignedLimbMatrix};
use crate::{Choice, Limb, UintRef, WideWord, Word, word};

/// Maximum number of elementary binary GCD steps [`partial_xgcd`] performs per batch.
pub(super) const GCD_BATCH_SIZE: u32 = Word::BITS - SPLIT_THRESHOLD_BITS;

/// Ambiguity band for [`partial_xgcd`]'s `HALT = true` divergence check.
const SPLIT_THRESHOLD_BITS: u32 = match cpubits::CPUBITS {
    32 => 4,
    64 => 5,
    _ => unreachable!(),
};

#[derive(Debug, Copy, Clone)]
pub struct BingcdMatrix {
    pub(crate) r0: (Limb, Limb),
    pub(crate) r1: (Limb, Limb),
    pub(crate) pattern: Choice,
    pub(crate) k: u32,
}

impl BingcdMatrix {
    pub const UNIT: Self = Self {
        r0: (Limb::ONE, Limb::ZERO),
        r1: (Limb::ZERO, Limb::ONE),
        pattern: Choice::TRUE,
        k: 0,
    };

    #[inline(always)]
    pub(crate) const fn signed_limb_matrix(self) -> SignedLimbMatrix {
        let pat = self.pattern;
        SignedLimbMatrix {
            r0: (
                SignedLimb::new(self.r0.0, pat.not()),
                SignedLimb::new(self.r0.1, pat),
            ),
            r1: (
                SignedLimb::new(self.r1.0, pat),
                SignedLimb::new(self.r1.1, pat.not()),
            ),
        }
    }

    #[inline(always)]
    pub const fn wrapping_apply_shift(
        &self,
        g: &mut ExtendedIntRef<'_>,
        f: &mut ExtendedIntRef<'_>,
    ) {
        let matrix = self.signed_limb_matrix();
        matrix.wrapping_apply(g, f);
        g.shr_assign_limb(self.k);
        f.shr_assign_limb(self.k);
    }

    /// Apply the matrix to `g`/`f`, leaving them in a non-negative state.
    ///
    /// Returns a pair of flags indicating whether `g` or `f` would be negative
    /// if [`Self::wrapping_apply_shift`] were used instead.
    #[inline(always)]
    pub const fn wrapping_apply_unsigned_shift(
        &self,
        g: &mut UintRef,
        f: &mut UintRef,
    ) -> (Choice, Choice) {
        let (mut g_ext, mut f_ext, g_neg, f_neg) =
            self.signed_limb_matrix().wrapping_apply_unsigned(g, f);
        g_ext.shr_assign_limb_unsigned(self.k);
        g_ext.unsigned_drop_extension();
        f_ext.shr_assign_limb_unsigned(self.k);
        f_ext.unsigned_drop_extension();
        (g_neg, f_neg)
    }

    /// Applies a matrix computed from `g`/`f`'s *magnitudes* rather than their actual (possibly
    /// negative) values directly to the signed `g`/`f` themselves, deferring renormalization to
    /// non-negative to wherever the caller chooses to pay for it.
    #[inline(always)]
    pub const fn wrapping_apply_sign_correcting_shift(
        &self,
        g: &mut ExtendedIntRef<'_>,
        f: &mut ExtendedIntRef<'_>,
    ) {
        let (g_neg, f_neg) = (g.is_negative(), f.is_negative());
        let matrix = self.column_signed_limb_matrix(g_neg, f_neg);
        matrix.wrapping_apply(g, f);
        g.shr_assign_limb(self.k);
        f.shr_assign_limb(self.k);
    }

    /// Re-derives what [`Self::signed_limb_matrix`] must have meant in terms of the values a
    /// coefficient pair `(d, e)` is actually being updated alongside -- `d`/`e` can't supply their
    /// own column sign the way a genuine signed operand can (they're coefficients, not the signed
    /// `(g, f)` the matrix was derived from), so the caller passes `g_neg`/`f_neg` in directly:
    /// column 0 (`r0.0`, `r1.0`) picks up `g`'s sign, column 1 (`r0.1`, `r1.1`) picks up `f`'s,
    /// exactly like [`Self::wrapping_apply_shift_signed`] does for `(g, f)` themselves. Capture
    /// `g_neg`/`f_neg` (e.g. from `g.is_negative()`/`f.is_negative()`) *before* applying the matrix
    /// to `(g, f)`, since that application changes their sign, and this needs the value from
    /// *before* it.
    #[inline(always)]
    pub(crate) const fn column_signed_limb_matrix(
        self,
        g_neg: Choice,
        f_neg: Choice,
    ) -> SignedLimbMatrix {
        let mut matrix = self.signed_limb_matrix();
        matrix.r0.0.sign = matrix.r0.0.sign.xor(g_neg);
        matrix.r1.0.sign = matrix.r1.0.sign.xor(g_neg);
        matrix.r0.1.sign = matrix.r0.1.sign.xor(f_neg);
        matrix.r1.1.sign = matrix.r1.1.sign.xor(f_neg);
        matrix
    }

    /// [`Self::column_signed_limb_matrix`]'s counterpart for a coefficient pair updated alongside
    /// `(g, f)` *after* an unsigned, row-combined-sign apply
    /// ([`Self::wrapping_apply_shift_unsigned`]/[`SignedLimbMatrix::wrapping_apply_unsigned`])
    /// rather than a deferred-sign one -- that shortcut forces each row's own result to its
    /// absolute value independently, silently flipping the *entire row* (not a column) whenever
    /// that row's raw computation came out negative, exactly the `(g_negated, f_negated)`
    /// `wrapping_apply_shift_unsigned` itself already returns. A coefficient pair updated via a
    /// *column*-adjusted matrix (as if `g`/`f` had never needed correcting) tracks the
    /// pre-correction, occasionally wrong-signed row result instead -- confirmed directly via the
    /// `d * a ≡ f (mod y)` trace on a real failing case: `d`/`e`'s own row-0/row-1 entries came out
    /// exactly negated relative to what the stored (always non-negative) `g`/`f` actually equal,
    /// whenever `wrapping_apply_shift_unsigned`'s returned flag for that row was `true`. Row 0
    /// (`r0.0`, `r0.1`) flips on `g_negated`, row 1 (`r1.0`, `r1.1`) on `f_negated` -- the
    /// mirror-image split of `column_signed_limb_matrix`'s column-based one. Capture
    /// `g_negated`/`f_negated` from the *same* `wrapping_apply_shift_unsigned` call this matrix
    /// was just used for -- feeding it a mismatched matrix/flags pair silently miscorrects.
    #[inline(always)]
    pub(crate) const fn row_signed_limb_matrix(
        self,
        g_negated: Choice,
        f_negated: Choice,
    ) -> SignedLimbMatrix {
        let mut signed = self.signed_limb_matrix();
        signed.r0.0.sign = signed.r0.0.sign.xor(g_negated);
        signed.r0.1.sign = signed.r0.1.sign.xor(g_negated);
        signed.r1.0.sign = signed.r1.0.sign.xor(f_negated);
        signed.r1.1.sign = signed.r1.1.sign.xor(f_negated);
        signed
    }
}

impl PartialEq for BingcdMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.signed_limb_matrix().eq(&other.signed_limb_matrix())
    }
}

/// The minimal number of binary GCD iterations required to guarantee successful completion.
#[inline(always)]
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
pub(super) const fn step_word(mut a: Word, mut b: Word) -> ((Word, Word), Choice, Choice, Word) {
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

/// [`WideWord`] variant of [`step_word`].
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub(super) const fn step_wide_word(
    mut a: WideWord,
    mut b: WideWord,
) -> ((WideWord, WideWord), Choice, Choice, Word) {
    let a_b = a as Word & b as Word;

    let a_odd = word::choice_from_lsb(a as Word);
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

/// Compute `gcd(a, b)` as well as the Jacobi symbol `(a|b)` in variable-time
/// using the classic binary GCD.
#[inline(always)]
#[allow(trivial_numeric_casts)]
pub(crate) const fn gcd_word_vartime(mut a: Word, mut b: Word) -> (Word, Word) {
    debug_assert!(b & 1 == 1, "b must be odd");
    let mut jacobi_neg = 0;

    while a != 0 {
        let tz = a.trailing_zeros();
        a >>= tz;
        // (2a|b) = -(a|b) iff b = ±3 mod 8
        // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
        jacobi_neg ^= tz as Word & ((b >> 1) ^ (b >> 2));

        let (diff, swap) = a.overflowing_sub(b);
        if swap {
            let a_b = a & b;
            jacobi_neg ^= a_b & (a_b >> 1);
            (a, b) = (diff.wrapping_neg(), a);
        } else {
            a = diff;
        }
    }

    (b, jacobi_neg & 1)
}

/// Compute `gcd(a, b)` as well as the Jacobi symbol `(a|b)` in variable-time
/// using the classic binary GCD.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
#[allow(trivial_numeric_casts)]
pub(crate) const fn gcd_wide_word_vartime(mut a: WideWord, mut b: WideWord) -> (WideWord, Word) {
    debug_assert!(b & 1 == 1, "b must be odd");
    let mut jacobi_neg = 0;

    while a != 0 {
        let tz = a.trailing_zeros();
        a >>= tz;
        let b_lo = b as Word;
        // (2a|b) = -(a|b) iff b = ±3 mod 8
        // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
        jacobi_neg ^= tz as Word & ((b_lo >> 1) ^ (b_lo >> 2));

        if (a | b) >> Word::BITS == 0 {
            let (gcd, j2) = gcd_word_vartime(a as Word, b as Word);
            return (gcd as WideWord, jacobi_neg ^ j2);
        }

        let (diff, swap) = a.overflowing_sub(b);
        if swap {
            let a_b = a as Word & b as Word;
            jacobi_neg ^= a_b & (a_b >> 1);
            (a, b) = (diff.wrapping_neg(), a);
        } else {
            a = diff;
        }
    }

    (b, jacobi_neg & 1)
}

#[inline(always)]
#[must_use]
pub const fn partial_xgcd<const HALT: bool>(
    (mut a_lo, mut a_hi): (Word, Word),
    (mut b_lo, mut b_hi): (Word, Word),
    exact: Choice,
    steps: u32,
) -> (BingcdMatrix, Word, Choice) {
    debug_assert!(b_lo & 1 == 1, "b_lo must be odd");

    let mut m = BingcdMatrix::UNIT;
    let mut i = 0;
    let mut jacobi_neg = 0;
    let mut active = Choice::TRUE;

    while i < steps {
        let a_b = a_lo & b_lo;
        let a_odd = word::choice_from_lsb(a_lo);
        let apply_sub = a_odd.and(active);

        let (hi_diff_raw, hi_borrow) = a_hi.overflowing_sub(b_hi);
        let swap_raw = Choice::from_u8_lsb(hi_borrow as u8);
        let apply_swap = swap_raw.and(apply_sub);
        let hi_diff = word::select(hi_diff_raw, hi_diff_raw.wrapping_neg(), swap_raw);

        let above_threshold = if HALT {
            word::choice_from_nz(hi_diff >> SPLIT_THRESHOLD_BITS).or(exact.or(a_odd.not()))
        } else {
            Choice::TRUE
        };

        let new_b_hi = word::select(b_hi, a_hi, apply_swap);
        let new_a_hi = word::select(a_hi, hi_diff, apply_sub) >> 1;

        // `lo` chain: the same shape of update, sharing `hi`'s raw swap decision for its own
        // magnitude normalization (no borrow ever crosses from this subtraction back into the
        // `hi` chain) and the final gated `swap`/`apply` for which values actually get kept.
        let lo_diff_raw = a_lo.wrapping_sub(b_lo);
        // let lo_diff = word::select(a_lo.wrapping_sub(b_lo), b_lo.wrapping_sub(a_lo), swap_raw);
        let lo_diff = word::select(lo_diff_raw, lo_diff_raw.wrapping_neg(), swap_raw);
        let new_b_lo = word::select(b_lo, a_lo, apply_swap);
        let new_a_lo = word::select(a_lo, lo_diff, apply_sub) >> 1;

        (a_hi, b_hi) = (new_a_hi, new_b_hi);
        (a_lo, b_lo) = (new_a_lo, new_b_lo);

        (m.r0, m.r1, m.pattern) = (
            (
                Limb::select(m.r0.0, m.r0.0.wrapping_add(m.r1.0), apply_sub),
                Limb::select(m.r0.1, m.r0.1.wrapping_add(m.r1.1), apply_sub),
            ),
            (
                Limb::select(m.r1.0, m.r0.0, apply_swap).shl(1),
                Limb::select(m.r1.1, m.r0.1, apply_swap).shl(1),
            ),
            m.pattern.xor(apply_swap),
        );

        i += 1;
        m.k = active.select_u32(m.k, i);

        active = active.and(above_threshold);

        // `(b|a) = -(a|b) iff a = b = 3 mod 4` (quadratic reciprocity) when a swap occurred.
        jacobi_neg ^= word::select(0, a_b & (a_b >> 1), apply_swap);

        // `(2a|b) = -(a|b) iff b = ±3 mod 8` we always strip a zero from `a` unless we fell below the threshold.
        jacobi_neg ^= word::select(0, (b_lo >> 1) ^ (b_lo >> 2), active);
    }

    m.r0.0 = m.r0.0.shl(steps - m.k);
    m.r0.1 = m.r0.1.shl(steps - m.k);
    m.k = steps;

    (m, jacobi_neg & 1, active)
}

#[inline(always)]
#[must_use]
#[allow(trivial_numeric_casts)]
pub const fn partial_xgcd_vartime(
    mut a: (Word, Word),
    mut b: (Word, Word),
    max_batch: u32,
    exact: bool,
) -> (BingcdMatrix, Word) {
    assert!(b.0 & 1 == 1);

    let mut m = BingcdMatrix::UNIT;
    let mut jacobi_neg = 0;
    let mut steps_remain = max_batch;
    let mut abort = false;
    let mut pattern = 1;
    let threshold = if exact { 0 } else { max_batch as Word };

    loop {
        let a_tz = a.0.trailing_zeros();
        let tz = if a_tz < steps_remain {
            a_tz
        } else {
            steps_remain
        };
        a.0 >>= tz;
        a.1 >>= tz;
        m.r1.0 = m.r1.0.shl(tz);
        m.r1.1 = m.r1.1.shl(tz);
        steps_remain -= tz;
        jacobi_neg ^= tz as Word & ((b.0 >> 1) ^ (b.0 >> 2));

        if steps_remain == 0 || abort {
            break;
        }

        let (diff_hi, swap) = a.1.overflowing_sub(b.1);
        let swap_mask = (swap as Word).wrapping_neg();
        let abs_diff_hi = (diff_hi ^ swap_mask).wrapping_sub(swap_mask);
        let a_b = a.0 & b.0;
        jacobi_neg ^= swap_mask & a_b & (a_b >> 1);

        (a, b, m.r0, m.r1, pattern, abort) = (
            (
                (a.0.wrapping_sub(b.0) ^ swap_mask).wrapping_sub(swap_mask),
                abs_diff_hi,
            ),
            (
                (b.0 ^ (swap_mask & (a.0 ^ b.0))),
                (b.1 ^ (swap_mask & (a.1 ^ b.1))),
            ),
            (m.r0.0.wrapping_add(m.r1.0), m.r0.1.wrapping_add(m.r1.1)),
            (
                Limb(m.r1.0.0 ^ (swap_mask & (m.r0.0.0 ^ m.r1.0.0))),
                Limb(m.r1.1.0 ^ (swap_mask & (m.r0.1.0 ^ m.r1.1.0))),
            ),
            pattern ^ swap_mask,
            abs_diff_hi <= threshold,
        );
    }

    m.k = max_batch - steps_remain;
    m.pattern = word::choice_from_lsb(pattern);
    (m, jacobi_neg & 1)
}

#[cfg(test)]
mod tests {
    /// Validates `partial_xgcd` against a trusted reference built by literally
    /// running `step_word` `GCD_BATCH_SIZE` times and accumulating a matrix
    /// the same way `partial_xgcd` does -- for `a`, `b` that fit in a single word
    /// (so `a_hi=a_lo=a`, `b_hi=b_lo=b` is the correct split-form representation,
    /// the two should be identical.
    #[cfg(feature = "rand_core")]
    #[test]
    fn partial_xgcd_matches_elementary_steps() {
        use crate::{Choice, Limb, Random, Uint, Word, modular::gcd::bingcd::GCD_BATCH_SIZE};
        use chacha20::ChaCha8Rng;
        use rand_core::SeedableRng;

        fn trusted_matrix(mut a: Word, mut b: Word) -> super::BingcdMatrix {
            let mut m = super::BingcdMatrix::UNIT;
            m.k = GCD_BATCH_SIZE;
            let mut i = 0;
            while i < m.k {
                let (new_ab, a_odd, swap, _) = super::step_word(a, b);
                (a, b) = new_ab;
                (m.r0, m.r1, m.pattern) = (
                    (
                        m.r0.0.wrapping_add(Limb::select(Limb::ZERO, m.r1.0, a_odd)),
                        m.r0.1.wrapping_add(Limb::select(Limb::ZERO, m.r1.1, a_odd)),
                    ),
                    (
                        Limb::select(m.r1.0, m.r0.0, swap).shl(1),
                        Limb::select(m.r1.1, m.r0.1, swap).shl(1),
                    ),
                    m.pattern.xor(swap),
                );
                i += 1;
            }
            m
        }

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..10_000 {
            let a = Uint::<1>::random_from_rng(&mut rng).limbs[0].0;
            let b = Uint::<1>::random_from_rng(&mut rng).limbs[0].0 | 1;

            let expected = trusted_matrix(a, b);

            // `HALT = false`: unconditional, matches the trusted reference unconditionally.
            let (actual, _, _) =
                super::partial_xgcd::<false>((a, a), (b, b), Choice::FALSE, GCD_BATCH_SIZE);
            assert_eq!(
                (expected.r0, expected.r1, expected.pattern.to_bool_vartime()),
                (actual.r0, actual.r1, actual.pattern.to_bool_vartime()),
                "HALT=false a={a:#x} b={b:#x}"
            );

            // `HALT = true`, `exact = TRUE`: single-word inputs are always the `s == 0` case (the
            // whole value fits its register, `hi == lo` exactly), so `above_threshold` should hold
            // every step regardless of the margin -- output should match the trusted reference exactly,
            // and `active` should end up `TRUE` (never froze).
            let (actual_halt, _j, active) =
                super::partial_xgcd::<true>((a, a), (b, b), Choice::TRUE, GCD_BATCH_SIZE);
            assert!(
                active.to_bool_vartime(),
                "a={a:#x} b={b:#x}: froze despite exact=TRUE"
            );
            assert_eq!(
                (expected.r0, expected.r1, expected.pattern.to_bool_vartime()),
                (
                    actual_halt.r0,
                    actual_halt.r1,
                    actual_halt.pattern.to_bool_vartime()
                ),
                "HALT=true a={a:#x} b={b:#x}"
            );
        }
    }
}
