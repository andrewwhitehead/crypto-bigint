use super::{ExtendedIntRef, SPLIT_THRESHOLD_BITS, SignedLimb, SignedLimbMatrix};
use crate::{Choice, Limb, UintRef, WideWord, Word, word};

/// A 2x2 matrix of non-negative [`Limb`] entries accumulated by a batch of elementary binary-GCD
/// update steps ([`partial_xgcd`], [`partial_xgcd_word`], [`partial_xgcd_vartime`]), together with
/// the shift and sign bookkeeping needed to apply it to the evolving `(a, b)` pair.
///
/// Every entry these batches produce follows a fixed checkerboard sign pattern relative to a
/// single shared bit -- see [`Self::signed_limb_matrix`] -- so only that one bit needs tracking
/// instead of four independent signs, unlike the general [`SignedLimbMatrix`] this converts into
/// before actually being applied.
#[derive(Debug, Copy, Clone)]
pub struct BingcdMatrix {
    /// Top row: unsigned coefficients for the new `a`.
    pub(crate) r0: (Limb, Limb),
    /// Bottom row: unsigned coefficients for the new `b`.
    pub(crate) r1: (Limb, Limb),
    /// The shared sign bit each entry's true sign is derived from; see [`Self::signed_limb_matrix`].
    pub(crate) pattern: Choice,
    /// The number of elementary steps this matrix represents; applying it shifts the result right
    /// by this many bits.
    pub(crate) k: u32,
}

impl BingcdMatrix {
    /// The identity matrix.
    pub const UNIT: Self = Self {
        r0: (Limb::ONE, Limb::ZERO),
        r1: (Limb::ZERO, Limb::ONE),
        pattern: Choice::TRUE,
        k: 0,
    };

    /// Expands the compact `(r0, r1, pattern)` representation into a full [`SignedLimbMatrix`],
    /// assigning each entry its true sign from the shared `pattern` bit: within each row the two
    /// columns get opposite signs, and the two rows are likewise each other's opposite.
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

    /// Apply the matrix to `a`/`b`, leaving them in a non-negative state.
    ///
    /// Returns a pair of flags indicating whether `a` or `b` would be negative
    /// if [`Self::wrapping_apply_shift`] were used instead.
    #[inline(always)]
    pub const fn wrapping_apply_unsigned_shift(
        &self,
        a: &mut UintRef,
        b: &mut UintRef,
    ) -> (Choice, Choice) {
        let (mut a_ext, mut b_ext, a_neg, b_neg) =
            self.signed_limb_matrix().wrapping_apply_unsigned(a, b);
        a_ext.shr_assign_limb_unsigned(self.k);
        a_ext.unsigned_drop_extension();
        b_ext.shr_assign_limb_unsigned(self.k);
        b_ext.unsigned_drop_extension();
        (a_neg, b_neg)
    }

    /// Applies a matrix computed from `a`/`b`'s *magnitudes* rather than their actual (possibly
    /// negative) values directly to the signed `a`/`b` themselves, deferring renormalization to
    /// non-negative to wherever the caller chooses to pay for it.
    #[inline(always)]
    pub const fn wrapping_apply_sign_correcting_shift(
        &self,
        a: &mut ExtendedIntRef<'_>,
        b: &mut ExtendedIntRef<'_>,
    ) {
        let (a_neg, b_neg) = (a.is_negative(), b.is_negative());
        let matrix = self.column_signed_limb_matrix(a_neg, b_neg);
        matrix.wrapping_apply(a, b);
        a.shr_assign_limb(self.k);
        b.shr_assign_limb(self.k);
    }

    /// Re-derives what [`Self::signed_limb_matrix`] must have meant in terms of the values a
    /// coefficient pair `(d, e)` is actually being updated alongside -- `d`/`e` can't supply their
    /// own column sign the way a genuine signed operand can (they're coefficients, not the signed
    /// `(a, b)` the matrix was derived from), so the caller passes `a_neg`/`b_neg` in directly:
    /// column 0 (`r0.0`, `r1.0`) picks up `a`'s sign, column 1 (`r0.1`, `r1.1`) picks up `b`'s,
    /// exactly like [`Self::wrapping_apply_shift_signed`] does for `(a, b)` themselves. Capture
    /// `a_neg`/`b_neg` (e.g. from `a.is_negative()`/`b.is_negative()`) *before* applying the matrix
    /// to `(a, b)`, since that application changes their sign, and this needs the value from
    /// *before* it.
    #[inline(always)]
    pub(crate) const fn column_signed_limb_matrix(
        self,
        a_neg: Choice,
        b_neg: Choice,
    ) -> SignedLimbMatrix {
        let mut matrix = self.signed_limb_matrix();
        matrix.r0.0.sign = matrix.r0.0.sign.xor(a_neg);
        matrix.r1.0.sign = matrix.r1.0.sign.xor(a_neg);
        matrix.r0.1.sign = matrix.r0.1.sign.xor(b_neg);
        matrix.r1.1.sign = matrix.r1.1.sign.xor(b_neg);
        matrix
    }

    /// [`Self::column_signed_limb_matrix`]'s counterpart for a coefficient pair updated alongside
    /// `(a, b)` *after* an unsigned, row-combined-sign apply
    /// ([`Self::wrapping_apply_shift_unsigned`]/[`SignedLimbMatrix::wrapping_apply_unsigned`])
    /// rather than a deferred-sign one -- that shortcut forces each row's own result to its
    /// absolute value independently, silently flipping the *entire row* (not a column) whenever
    /// that row's raw computation came out negative, exactly the `(a_neg, b_neg)`
    /// `wrapping_apply_shift_unsigned` itself already returns. A coefficient pair updated via a
    /// *column*-adjusted matrix (as if `a`/`b` had never needed correcting) tracks the
    /// pre-correction, occasionally wrong-signed row result instead -- confirmed directly via the
    /// `d * a ≡ f (mod y)` trace on a real failing case: `d`/`e`'s own row-0/row-1 entries came out
    /// exactly negated relative to what the stored (always non-negative) `a`/`b` actually equal,
    /// whenever `wrapping_apply_shift_unsigned`'s returned flag for that row was `true`. Row 0
    /// (`r0.0`, `r0.1`) flips on `a_neg`, row 1 (`r1.0`, `r1.1`) on `b_neg` -- the
    /// mirror-image split of `column_signed_limb_matrix`'s column-based one. Capture
    /// `a_neg`/`b_neg` from the *same* `wrapping_apply_shift_unsigned` call this matrix
    /// was just used for -- feeding it a mismatched matrix/flags pair silently miscorrects.
    #[inline(always)]
    pub(crate) const fn row_signed_limb_matrix(
        self,
        a_neg: Choice,
        b_neg: Choice,
    ) -> SignedLimbMatrix {
        let mut signed = self.signed_limb_matrix();
        signed.r0.0.sign = signed.r0.0.sign.xor(a_neg);
        signed.r0.1.sign = signed.r0.1.sign.xor(a_neg);
        signed.r1.0.sign = signed.r1.0.sign.xor(b_neg);
        signed.r1.1.sign = signed.r1.1.sign.xor(b_neg);
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
pub(super) const fn step_word(
    mut a: Word,
    mut b: Word,
    mut jacobi_neg: Word,
) -> ((Word, Word), Choice, Choice, Word) {
    let a_b = a & b;
    let apply_sub = word::choice_from_lsb(a);
    let (diff, borrow) = a.overflowing_sub(word::select(0, b, apply_sub));
    let apply_swap = Choice::from_u8_lsb(borrow as u8);
    (a, b) = (
        word::select(diff, diff.wrapping_neg(), apply_swap) >> 1,
        word::select(b, a, apply_swap),
    );

    // (b|a) = -(a|b) iff a = b = 3 mod 4 (quadratic reciprocity)
    jacobi_neg ^= word::select(0, a_b & (a_b >> 1) & 1, apply_swap);
    // (2a|b) = -(a|b) iff b = ±3 mod 8
    // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
    jacobi_neg ^= ((b >> 1) ^ (b >> 2)) & 1;

    ((a, b), apply_sub, apply_swap, jacobi_neg)
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
    let mut jacobi_neg = word::select(0, a_b & (a_b >> 1), swap);

    // (2a|b) = -(a|b) iff b = ±3 mod 8
    // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
    let b_lo = b as Word;
    jacobi_neg ^= (b_lo >> 1) ^ (b_lo >> 2);

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
        let swap_mask = (swap as Word).wrapping_neg();
        let a_b = a & b;
        jacobi_neg ^= swap_mask & a_b & (a_b >> 1);
        (a, b) = (
            (diff ^ swap_mask).wrapping_sub(swap_mask),
            b ^ (swap_mask & (a ^ b)),
        );
    }

    (b, jacobi_neg & 1)
}

/// Runs a constant-time batch of `steps` elementary binary-GCD update steps ([`step_word`]'s
/// split-representation counterpart) over `a`/`b`, each given as a `(lo, hi)` word pair -- `lo`
/// the low word of the currently tracked window, `hi` the top-aligned word swap/subtract
/// decisions are actually made from -- and accumulates the resulting [`BingcdMatrix`].
///
/// `exact` marks whether `hi` can be trusted to reflect every comparison exactly for the whole
/// batch (e.g. because the tracked value fits entirely within it). When `HALTING` is `false`, all
/// `steps` iterations run unconditionally, exactly as `exact = TRUE` would. When `HALTING` is
/// `true` and `exact` is not set, each step additionally checks whether the shrinking `hi`
/// magnitude has dropped below [`SPLIT_THRESHOLD_BITS`]; once it has, further subtract/swap
/// decisions on `a`/`b` stop for the remainder of the batch (though `a`'s residual trailing-zero
/// shift keeps draining into the matrix) -- the caller must then re-extract a fresh window and
/// retry with a smaller batch.
///
/// Returns `(matrix, jacobi_neg, unhalted)`: `jacobi_neg` is the low bit of the accumulated Jacobi
/// symbol sign flips, and `unhalted` reports whether the batch ran to completion (`TRUE`) rather
/// than halting early (`FALSE`, `HALTING` only).
#[inline(always)]
#[must_use]
pub const fn partial_xgcd<const HALTING: bool>(
    (mut a_lo, mut a_hi): (Word, Word),
    (mut b_lo, mut b_hi): (Word, Word),
    exact: Choice,
    steps: u32,
) -> (BingcdMatrix, Word, Choice) {
    debug_assert!(b_lo & 1 == 1, "b_lo must be odd");

    let mut m = BingcdMatrix::UNIT;
    let mut i = 0;
    let mut jacobi_neg = 0;
    let mut real_steps = 0;
    let mut unhalted = Choice::TRUE;

    while i < steps {
        let a_b = a_lo & b_lo;
        let a_odd = word::choice_from_lsb(a_lo);
        let apply_sub = if HALTING { a_odd.and(unhalted) } else { a_odd };

        let (hi_diff, borrow) = a_hi.overflowing_sub(word::select(0, b_hi, apply_sub));
        let apply_swap = Choice::from_u8_lsb(borrow as u8);
        let lo_diff = a_lo.wrapping_sub(word::select(0, b_lo, apply_sub));
        let (abs_lo_diff, abs_hi_diff) = (
            word::select(lo_diff, lo_diff.wrapping_neg(), apply_swap),
            word::select(hi_diff, hi_diff.wrapping_neg(), apply_swap),
        );

        (a_lo, a_hi, b_lo, b_hi) = (
            abs_lo_diff >> 1,
            abs_hi_diff >> 1,
            word::select(b_lo, a_lo, apply_swap),
            word::select(b_hi, a_hi, apply_swap),
        );

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

        // `(b|a) = -(a|b) iff a = b = 3 mod 4` (quadratic reciprocity) when a swap occurred.
        jacobi_neg ^= word::select(0, a_b & (a_b >> 1), apply_swap);

        // `(2a|b) = -(a|b) iff b = ±3 mod 8` we always strip a zero from `a` unless we fell below the threshold.
        // NB: it is valid to keep updating this after `a` hits zero, as a GCD of 1 means that the sign stays
        // the same, and a larger GCD means the correct symbol is zero.
        jacobi_neg ^= word::select(0, (b_lo >> 1) ^ (b_lo >> 2), unhalted);

        i += 1;

        if HALTING {
            real_steps = unhalted.select_u32(real_steps, i);
            let above_threshold =
                word::choice_from_nz(abs_hi_diff >> SPLIT_THRESHOLD_BITS).or(exact.or(a_odd.not()));
            unhalted = unhalted.and(above_threshold);
        }
    }

    if HALTING {
        m.r0.0 = m.r0.0.shl(steps - real_steps);
        m.r0.1 = m.r0.1.shl(steps - real_steps);
    }
    m.k = steps;

    (m, jacobi_neg & 1, unhalted)
}

/// [`partial_xgcd`]'s single-word tail variant: runs `steps` elementary [`step_word`] updates
/// unconditionally (no halting -- callers only reach this once the tracked window already fits a
/// single word) and accumulates the resulting [`BingcdMatrix`].
///
/// Returns `(b, matrix, jacobi_neg)`: `b` is the resulting value of the second operand (`gcd(a,
/// b)` once `steps` covers the whole remaining reduction), and `jacobi_neg` is the low bit of the
/// accumulated Jacobi symbol sign flips.
#[inline(always)]
#[must_use]
pub const fn partial_xgcd_word(mut a: Word, mut b: Word, steps: u32) -> (Word, BingcdMatrix, Word) {
    debug_assert!(b & 1 == 1, "b must be odd");

    let mut m = BingcdMatrix::UNIT;
    let mut i = 0;
    let mut jacobi_neg = 0;

    while i < steps {
        let (apply_sub, apply_swap);
        ((a, b), apply_sub, apply_swap, jacobi_neg) = step_word(a, b, jacobi_neg);

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
    }

    m.k = steps;
    (b, m, jacobi_neg & 1)
}

/// [`partial_xgcd`]'s variable-time counterpart, used by the vartime GCD/XGCD/Jacobi-symbol paths:
/// consumes up to `max_batch` elementary steps from a top-bit-aligned `(a, b)` window (each given
/// as a `(lo, hi)` pair, `hi` the top-aligned word swap/subtract decisions are made from),
/// skipping whole runs of `a`'s trailing zero bits at once via `trailing_zeros` rather than
/// stepping through them one at a time.
///
/// Stops early -- before exhausting `max_batch` -- once `hi`'s magnitude drops to fit within
/// `threshold_bits` (`0` when `exact` is set, i.e. the window already fits a single limb;
/// [`SPLIT_THRESHOLD_BITS`] otherwise), since a window that thin can no longer be trusted to make
/// correct swap decisions for further steps.
///
/// Returns `(matrix, jacobi_neg)`, where `matrix.k` records the number of steps actually consumed
/// (which may be less than `max_batch`) and `jacobi_neg` is the low bit of the accumulated Jacobi
/// symbol sign flips.
#[inline(always)]
#[must_use]
#[allow(trivial_numeric_casts)]
pub const fn partial_xgcd_vartime(
    (mut a_lo, mut a_hi): (Word, Word),
    (mut b_lo, mut b_hi): (Word, Word),
    max_batch: u32,
    exact: bool,
) -> (BingcdMatrix, Word) {
    assert!(b_lo & 1 == 1);

    let mut m = BingcdMatrix::UNIT;
    let mut jacobi_neg = 0;
    let mut steps_remain = max_batch;
    let mut abort = false;
    let mut pattern = true;
    let threshold_bits = if exact { 0 } else { SPLIT_THRESHOLD_BITS };

    loop {
        let a_tz = a_lo.trailing_zeros();
        let tz = if a_tz < steps_remain {
            a_tz
        } else {
            steps_remain
        };
        a_lo >>= tz;
        a_hi >>= tz;
        m.r1.0 = m.r1.0.shl(tz);
        m.r1.1 = m.r1.1.shl(tz);
        steps_remain -= tz;
        jacobi_neg ^= tz as Word & ((b_lo >> 1) ^ (b_lo >> 2));

        if steps_remain == 0 || abort {
            break;
        }

        let a_b = a_lo & b_lo;
        let (hi_diff, borrow) = a_hi.overflowing_sub(b_hi);
        let apply_swap = Choice::from_u8_lsb(borrow as u8);
        let lo_diff = a_lo.wrapping_sub(b_lo);
        let (abs_lo_diff, abs_hi_diff) = (
            word::select(lo_diff, lo_diff.wrapping_neg(), apply_swap),
            word::select(hi_diff, hi_diff.wrapping_neg(), apply_swap),
        );

        (a_lo, a_hi, b_lo, b_hi) = (
            abs_lo_diff,
            abs_hi_diff,
            word::select(b_lo, a_lo, apply_swap),
            word::select(b_hi, a_hi, apply_swap),
        );

        (m.r0, m.r1, pattern, abort) = (
            (m.r0.0.wrapping_add(m.r1.0), m.r0.1.wrapping_add(m.r1.1)),
            (
                Limb::select(m.r1.0, m.r0.0, apply_swap),
                Limb::select(m.r1.1, m.r0.1, apply_swap),
            ),
            pattern ^ borrow,
            abs_hi_diff >> threshold_bits == 0,
        );

        jacobi_neg ^= borrow as Word & (a_b >> 1);
    }

    m.k = max_batch - steps_remain;
    m.pattern = if pattern { Choice::TRUE } else { Choice::FALSE };
    (m, jacobi_neg & 1)
}

#[cfg(test)]
mod tests {
    /// Validates `partial_xgcd` against a trusted reference built by literally
    /// running `step_word` `SPLIT_BATCH_SIZE` times and accumulating a matrix
    /// the same way `partial_xgcd` does -- for `a`, `b` that fit in a single word
    /// (so `a_hi=a_lo=a`, `b_hi=b_lo=b` is the correct split-form representation,
    /// the two should be identical.
    #[cfg(feature = "rand_core")]
    #[test]
    fn partial_xgcd_matches_elementary_steps() {
        use crate::{Choice, Limb, Random, Uint, Word, modular::gcd::SPLIT_BATCH_SIZE};
        use chacha20::ChaCha8Rng;
        use rand_core::SeedableRng;

        fn trusted_matrix(mut a: Word, mut b: Word) -> super::BingcdMatrix {
            let mut m = super::BingcdMatrix::UNIT;
            m.k = SPLIT_BATCH_SIZE;
            let mut i = 0;
            while i < m.k {
                let (new_ab, a_odd, swap, _) = super::step_word(a, b, 0);
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
                super::partial_xgcd::<false>((a, a), (b, b), Choice::FALSE, SPLIT_BATCH_SIZE);
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
                super::partial_xgcd::<true>((a, a), (b, b), Choice::TRUE, SPLIT_BATCH_SIZE);
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
