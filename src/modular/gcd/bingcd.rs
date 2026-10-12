use super::{ExtendedIntRef, SPLIT_THRESHOLD_BITS, SignedLimb, SignedLimbMatrix};
use crate::{Choice, Limb, UintRef, Word, word};

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

    /// Assembles a batch's matrix from its unsigned entries, deriving `pattern` from them rather
    /// than tracking it step by step.
    ///
    /// Each step multiplies the unsigned entries by `[[1, 1], [2, 0]]` when it swaps (determinant
    /// `-2`) and by `[[1, 1], [0, 2]]` or `[[1, 0], [0, 2]]` otherwise (determinant `2`), so
    /// `r0.0 * r1.1 - r0.1 * r1.0 = ±2^k`, positive exactly when `pattern` is set. With
    /// `k <= Word::BITS - 2` the two cases already differ in the top bit of the wrapping product.
    /// (A halted [`partial_xgcd`] batch must call this before its final top-row shift, which
    /// scales the determinant beyond that range.)
    #[inline(always)]
    pub(crate) const fn from_entries(r0: (Word, Word), r1: (Word, Word), k: u32) -> Self {
        debug_assert!(k <= Word::BITS - 2);
        let det = (r0.0.wrapping_mul(r1.1)).wrapping_sub(r0.1.wrapping_mul(r1.0));
        Self {
            r0: (Limb(r0.0), Limb(r0.1)),
            r1: (Limb(r1.0), Limb(r1.1)),
            pattern: word::choice_from_msb(det).not(),
            k,
        }
    }

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
    /// Returns a pair of flags indicating whether `a` or `b` was negated, i.e. whether that row's
    /// signed result came out negative.
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
    /// exactly like [`Self::wrapping_apply_sign_correcting_shift`] does for `(a, b)` themselves.
    /// Capture `a_neg`/`b_neg` (e.g. from `a.is_negative()`/`b.is_negative()`) *before* applying the matrix
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
    /// ([`Self::wrapping_apply_unsigned_shift`]/[`SignedLimbMatrix::wrapping_apply_unsigned`])
    /// rather than a deferred-sign one. That apply forces each row's result to its absolute value
    /// independently, flipping the *entire row* (not a column) whenever that row's signed result
    /// came out negative -- exactly the `(a_neg, b_neg)` it returns. A coefficient pair must track
    /// the stored (non-negative) `a`/`b`, so row 0 (`r0.0`, `r0.1`) flips on `a_neg` and row 1
    /// (`r1.0`, `r1.1`) on `b_neg`: the mirror image of `column_signed_limb_matrix`'s
    /// column-based split. Capture `a_neg`/`b_neg` from the *same* `wrapping_apply_unsigned_shift`
    /// call this matrix was just used for -- a mismatched matrix/flags pair silently miscorrects.
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
/// Returns the updated `(a, b)`, the masks (`0` or `Word::MAX`) for whether the subtraction and
/// the swap were applied, and the updated Jacobi sign accumulator. The decisions are derived
/// directly as masks, once each: a `Choice` converted to a mask on every use is an optimization
/// barrier.
///
/// Note: assumes `b` to be odd. Might yield an incorrect result if this is not the case.
///
/// Ref: Pornin, Algorithm 1, L3-9, <https://eprint.iacr.org/2020/972.pdf>.
#[inline(always)]
pub(super) const fn step_word(
    a: Word,
    b: Word,
    mut jacobi_neg: Word,
) -> ((Word, Word), Word, Word, Word) {
    let a_b = a & b;
    let sub = (a & 1).wrapping_neg();
    let (diff, borrow) = a.overflowing_sub(b & sub);
    let swap = (borrow as Word).wrapping_neg();

    // (b|a) = -(a|b) iff a = b = 3 mod 4 (quadratic reciprocity)
    jacobi_neg ^= swap & a_b & (a_b >> 1) & 1;
    let (a_next, b_next) = ((diff ^ swap).wrapping_sub(swap) >> 1, b ^ (swap & (a ^ b)));
    // (2a|b) = -(a|b) iff b = ±3 mod 8
    // b is always odd, so we ignore the lower bit and check that bits 2 and 3 are '01' or '10'
    jacobi_neg ^= ((b_next >> 1) ^ (b_next >> 2)) & 1;

    ((a_next, b_next), sub, swap, jacobi_neg)
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
/// `true` and `exact` is not set, each subtract step additionally checks whether the absolute
/// high-word difference `|ah - bh|` has fallen below `2^SPLIT_THRESHOLD_BITS` (a banded step).
/// Right after the first such step the batch halts: no further subtractions are made, and both
/// matrix rows are scaled by `2^(steps - real_steps)` (the bottom row by its per-step doubling,
/// the top row by a final shift), which the `>> matrix.k` on application divides back out. The
/// batch is therefore still charged as `steps` steps (`matrix.k == steps`) while leaving
/// `(a, b)` exactly where the halting step left them; there is no retry.
///
/// Returns `(matrix, jacobi_neg, unhalted)`: `jacobi_neg` is the low bit of the accumulated Jacobi
/// symbol sign flips, and `unhalted` reports whether the batch ran to completion (`TRUE`) rather
/// than halting early (`FALSE`, `HALTING` only).
#[inline(always)]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub const fn partial_xgcd<const HALTING: bool>(
    (mut a_lo, mut a_hi): (Word, Word),
    (mut b_lo, mut b_hi): (Word, Word),
    exact: Choice,
    steps: u32,
) -> (BingcdMatrix, Word, Choice) {
    debug_assert!(b_lo & 1 == 1, "b_lo must be odd");

    // All decisions are carried as full-width masks, each derived once per step, rather than as
    // `Choice` values: every conversion of a `Choice` to a mask is an optimization barrier.
    let (mut r0_0, mut r0_1, mut r1_0, mut r1_1): (Word, Word, Word, Word) = (1, 0, 0, 1);
    let mut i = 0;
    let mut jacobi_neg: Word = 0;
    let mut real_steps = 0;
    let mut unhalted = Word::MAX;
    let exact = word::choice_to_mask(exact);

    while i < steps {
        let a_b = a_lo & b_lo;
        let a_odd = (a_lo & 1).wrapping_neg();
        let sub = if HALTING { a_odd & unhalted } else { a_odd };

        let (hi_diff, borrow) = a_hi.overflowing_sub(b_hi & sub);
        let swap = (borrow as Word).wrapping_neg();
        let lo_diff = a_lo.wrapping_sub(b_lo & sub);
        let (abs_lo_diff, abs_hi_diff) = (
            (lo_diff ^ swap).wrapping_sub(swap),
            (hi_diff ^ swap).wrapping_sub(swap),
        );

        (a_lo, a_hi, b_lo, b_hi) = (
            abs_lo_diff >> 1,
            abs_hi_diff >> 1,
            b_lo ^ (swap & (a_lo ^ b_lo)),
            b_hi ^ (swap & (a_hi ^ b_hi)),
        );

        (r0_0, r0_1, r1_0, r1_1) = (
            r0_0.wrapping_add(r1_0 & sub),
            r0_1.wrapping_add(r1_1 & sub),
            (r1_0 ^ (swap & (r1_0 ^ r0_0))) << 1,
            (r1_1 ^ (swap & (r1_1 ^ r0_1))) << 1,
        );

        // `(b|a) = -(a|b) iff a = b = 3 mod 4` (quadratic reciprocity) when a swap occurred.
        jacobi_neg ^= swap & a_b & (a_b >> 1);

        // `(2a|b) = -(a|b) iff b = ±3 mod 8` we always strip a zero from `a` unless we fell below the threshold.
        // NB: it is valid to keep updating this after `a` hits zero, as a GCD of 1 means that the sign stays
        // the same, and a larger GCD means the correct symbol is zero.
        jacobi_neg ^= unhalted & ((b_lo >> 1) ^ (b_lo >> 2));

        i += 1;

        if HALTING {
            let unhalted_32 = unhalted as u32;
            real_steps ^= unhalted_32 & (real_steps ^ i);
            let over = abs_hi_diff >> SPLIT_THRESHOLD_BITS;
            let over_nz = (over | over.wrapping_neg()) >> (Word::BITS - 1);
            // Barrier on the mask straight out of the comparison: without it the compiler can
            // turn the threshold test into a branch.
            let above_threshold = core::hint::black_box(over_nz.wrapping_neg());
            unhalted &= above_threshold | exact | !a_odd;
        }
    }

    // `pattern` is derived before a halted batch's top-row shift: at this point the determinant
    // is `±2^steps` (the skipped steps having doubled only the bottom row), whereas the shift
    // scales it again and can push it past the top bit `from_entries` reads.
    let mut m = BingcdMatrix::from_entries((r0_0, r0_1), (r1_0, r1_1), steps);
    if HALTING {
        m.r0 = (
            m.r0.0.shl(steps - real_steps),
            m.r0.1.shl(steps - real_steps),
        );
    }
    (m, jacobi_neg & 1, Choice::from_u8_lsb(unhalted as u8))
}

/// [`partial_xgcd`]'s single-word tail variant: runs `steps` elementary [`step_word`] updates
/// unconditionally (no halting -- callers only reach this once the tracked window already fits a
/// single word) and accumulates the resulting [`BingcdMatrix`].
///
/// Returns `(b, matrix)`: `b` is the resulting value of the second operand (`gcd(a, b)` once
/// `steps` covers the whole remaining reduction).
#[inline(always)]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub const fn partial_xgcd_word(mut a: Word, mut b: Word, steps: u32) -> (Word, BingcdMatrix) {
    debug_assert!(b & 1 == 1, "b must be odd");

    let (mut r0_0, mut r0_1, mut r1_0, mut r1_1): (Word, Word, Word, Word) = (1, 0, 0, 1);
    let mut i = 0;

    while i < steps {
        let (sub, swap);
        ((a, b), sub, swap, _) = step_word(a, b, 0);

        (r0_0, r0_1, r1_0, r1_1) = (
            r0_0.wrapping_add(r1_0 & sub),
            r0_1.wrapping_add(r1_1 & sub),
            (r1_0 ^ (swap & (r1_0 ^ r0_0))) << 1,
            (r1_1 ^ (swap & (r1_1 ^ r0_1))) << 1,
        );

        i += 1;
    }

    (
        b,
        BingcdMatrix::from_entries((r0_0, r0_1), (r1_0, r1_1), steps),
    )
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
        let swap_mask = (borrow as Word).wrapping_neg();
        let lo_diff = a_lo.wrapping_sub(b_lo);
        let (abs_lo_diff, abs_hi_diff) = (
            (lo_diff ^ swap_mask).wrapping_sub(swap_mask),
            (hi_diff ^ swap_mask).wrapping_sub(swap_mask),
        );

        (a_lo, a_hi, b_lo, b_hi) = (
            abs_lo_diff,
            abs_hi_diff,
            b_lo ^ (swap_mask & (a_lo ^ b_lo)),
            b_hi ^ (swap_mask & (a_hi ^ b_hi)),
        );

        (m.r0, m.r1, pattern, abort) = (
            (m.r0.0.wrapping_add(m.r1.0), m.r0.1.wrapping_add(m.r1.1)),
            (
                if borrow { m.r0.0 } else { m.r1.0 },
                if borrow { m.r0.1 } else { m.r1.1 },
            ),
            pattern ^ borrow,
            abs_hi_diff >> threshold_bits == 0,
        );

        jacobi_neg ^= borrow as Word & (a_b >> 1);
    }

    // Unlike the constant-time batches, `pattern` is tracked per step here: this loop only
    // iterates on subtract steps and its batches are often short, so the per-batch determinant
    // in `BingcdMatrix::from_entries` would cost more than it saves.
    m.k = max_batch - steps_remain;
    m.pattern = if pattern { Choice::TRUE } else { Choice::FALSE };
    (m, jacobi_neg & 1)
}

#[cfg(test)]
mod tests {
    /// Validates the batch functions (`partial_xgcd`, `partial_xgcd_word` and
    /// `partial_xgcd_vartime`) against a trusted reference built by literally
    /// running `step_word` `steps` times and accumulating a matrix the same way
    /// `partial_xgcd` does -- for `a`, `b` that fit in a single word (so
    /// `a_hi=a_lo=a`, `b_hi=b_lo=b` is the correct split-form representation, and
    /// the window is exact), the results should be identical.
    #[cfg(feature = "rand_core")]
    #[test]
    fn partial_xgcd_matches_elementary_steps() {
        use crate::{Choice, Limb, Random, Uint, Word, modular::gcd::SPLIT_BATCH_SIZE};
        use chacha20::ChaCha8Rng;
        use rand_core::SeedableRng;

        /// Returns the accumulated matrix, the final `b`, and the low bit of the Jacobi sign.
        fn trusted_matrix(
            mut a: Word,
            mut b: Word,
            steps: u32,
        ) -> (super::BingcdMatrix, Word, Word) {
            let mut m = super::BingcdMatrix::UNIT;
            m.k = steps;
            let mut jacobi_neg = 0;
            let mut i = 0;
            while i < m.k {
                use crate::word;

                let (new_ab, a_odd, swap);
                (new_ab, a_odd, swap, jacobi_neg) = super::step_word(a, b, jacobi_neg);
                let (a_odd, swap) = (word::choice_from_lsb(a_odd), word::choice_from_lsb(swap));
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
            (m, b, jacobi_neg & 1)
        }

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..10_000 {
            let a = Uint::<1>::random_from_rng(&mut rng).limbs[0].0;
            let b = Uint::<1>::random_from_rng(&mut rng).limbs[0].0 | 1;

            let (expected, _, expected_j) = trusted_matrix(a, b, SPLIT_BATCH_SIZE);

            // `HALT = false`: unconditional, matches the trusted reference unconditionally.
            let (actual, j, _) =
                super::partial_xgcd::<false>((a, a), (b, b), Choice::FALSE, SPLIT_BATCH_SIZE);
            assert_eq!(
                (
                    expected.r0,
                    expected.r1,
                    expected.pattern.to_bool(),
                    expected_j
                ),
                (actual.r0, actual.r1, actual.pattern.to_bool(), j),
                "HALT=false a={a:#x} b={b:#x}"
            );

            // `HALT = true`, `exact = TRUE`: single-word inputs are always the `s == 0` case (the
            // whole value fits its register, `hi == lo` exactly), so `above_threshold` should hold
            // every step regardless of the margin -- output should match the trusted reference exactly,
            // and `active` should end up `TRUE` (never froze).
            let (actual_halt, j, active) =
                super::partial_xgcd::<true>((a, a), (b, b), Choice::TRUE, SPLIT_BATCH_SIZE);
            assert!(
                active.to_bool(),
                "a={a:#x} b={b:#x}: froze despite exact=TRUE"
            );
            assert_eq!(
                (
                    expected.r0,
                    expected.r1,
                    expected.pattern.to_bool(),
                    expected_j
                ),
                (
                    actual_halt.r0,
                    actual_halt.r1,
                    actual_halt.pattern.to_bool(),
                    j
                ),
                "HALT=true a={a:#x} b={b:#x}"
            );

            // Vartime, `exact = true`: skips runs of trailing zeros at once, but an exact window
            // never stops early, so it consumes the full batch with the same decisions.
            let (actual_vt, j) =
                super::partial_xgcd_vartime((a, a), (b, b), SPLIT_BATCH_SIZE, true);
            assert_eq!(
                (
                    expected.r0,
                    expected.r1,
                    expected.pattern.to_bool(),
                    expected.k,
                    expected_j
                ),
                (
                    actual_vt.r0,
                    actual_vt.r1,
                    actual_vt.pattern.to_bool(),
                    actual_vt.k,
                    j
                ),
                "vartime a={a:#x} b={b:#x}"
            );

            // Single-word tail: unconditional steps up to its own longer batch size.
            for steps in [SPLIT_BATCH_SIZE, super::super::WORD_BATCH_SIZE] {
                let (expected, expected_b, _) = trusted_matrix(a, b, steps);
                let (actual_b, actual_w) = super::partial_xgcd_word(a, b, steps);
                assert_eq!(
                    (
                        expected.r0,
                        expected.r1,
                        expected.pattern.to_bool(),
                        expected.k,
                        expected_b
                    ),
                    (
                        actual_w.r0,
                        actual_w.r1,
                        actual_w.pattern.to_bool(),
                        actual_w.k,
                        actual_b
                    ),
                    "word steps={steps} a={a:#x} b={b:#x}"
                );
            }
        }
    }

    /// Checks the `pattern` that `BingcdMatrix::from_entries` derives against the sign of the
    /// exact determinant, for `HALTING` batches that actually halt. The single-word inputs above
    /// are exact and never halt, so `a_hi` is drawn close to `b_hi` here to land subtract steps in
    /// the ambiguity band.
    #[cfg(feature = "rand_core")]
    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn partial_xgcd_halting_pattern() {
        use crate::{Choice, Random, Uint, Word, modular::gcd::SPLIT_BATCH_SIZE};
        use chacha20::ChaCha8Rng;
        use rand_core::SeedableRng;

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut word = || Uint::<1>::random_from_rng(&mut rng).limbs[0].0;
        let mut halted = 0;
        for _ in 0..100_000 {
            let (a_lo, b_lo, b_hi, r) = (word(), word() | 1, word(), word());
            // Low bits of `r` pick the case; the rest supply a small offset or a fresh word.
            let a_hi = if r & 1 == 0 {
                b_hi.wrapping_add(r >> (Word::BITS - 24 + ((r >> 1) as u32 & 15)))
            } else {
                word()
            };
            let exact = Choice::from_u8_lsb((r >> 8) as u8 & (r >> 9) as u8);

            let (m, _, unhalted) =
                super::partial_xgcd::<true>((a_lo, a_hi), (b_lo, b_hi), exact, SPLIT_BATCH_SIZE);
            let det = i128::from(m.r0.0.0) * i128::from(m.r1.1.0)
                - i128::from(m.r0.1.0) * i128::from(m.r1.0.0);
            assert!(det.unsigned_abs().is_power_of_two());
            assert_eq!(
                det > 0,
                m.pattern.to_bool(),
                "a=({a_lo:#x},{a_hi:#x}) b=({b_lo:#x},{b_hi:#x})"
            );
            halted += u32::from(!unhalted.to_bool());
        }
        assert!(halted > 1_000, "only {halted} batches halted");
    }
}
