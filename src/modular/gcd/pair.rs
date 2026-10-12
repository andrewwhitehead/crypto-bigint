use super::{
    CofactorPair, ExtendedIntRef, SMALL_THRESHOLD_LIMBS, SPLIT_BATCH_SIZE, WORD_BATCH_SIZE, bingcd,
};
use crate::{Choice, Limb, UintRef, Word, modular::gcd::bingcd::BingcdMatrix, word};

/// The running state of a binary GCD reduction.
#[derive(Debug)]
pub struct GcdPair<'a> {
    /// The shrinking operand; reaches (and stays at) zero once the gcd has been found.
    a: &'a mut UintRef,
    /// The odd operand; ends up holding the gcd.
    b: &'a mut UintRef,
    /// Current width (in limbs) of the `a`/`b` tracking window; starts at `a.nlimbs()` and only ever
    /// shrinks.
    len: usize,
    /// Current number of elementary steps still owed, which shrinks by the size of each batch applied.
    k_remain: u32,
    /// The value of `k_remain` at which we can reduce the size of the tracking window (`self.len`).
    window_shrink_at: u32,
}

impl<'a> GcdPair<'a> {
    /// Starts tracking `a`, `b` at their full width, with the step budget Pornin's Phi bound
    /// requires for operands of that width.
    #[inline(always)]
    pub const fn new(a: &'a mut UintRef, b: &'a mut UintRef) -> Self {
        let k_remain = bingcd::iterations(a.bits_precision());
        Self::new_with_budget(a, b, k_remain)
    }

    /// [`Self::new`] with the step budget supplied rather than derived: the caller must have
    /// `bitlen(a), bitlen(b) <= k_remain` (Pornin's Phi bound).
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn new_with_budget(a: &'a mut UintRef, b: &'a mut UintRef, k_remain: u32) -> Self {
        let len = a.nlimbs();
        assert!(b.nlimbs() == len, "a and b must have the same width");
        assert!(b.limbs[0].0 & 1 == 1, "b must be odd");
        let window_shrink_at = (len as u32 - 1) << Limb::LOG2_BITS;
        Self {
            a,
            b,
            len,
            k_remain,
            window_shrink_at,
        }
    }
}

impl GcdPair<'_> {
    /// Reduces `a`/`b` down to their gcd using a deferred-sign reduction loop with a scheduled
    /// high word extraction before handing off to [`Self::gcd_small`] once the
    /// tracked window narrows to `SMALL_THRESHOLD_LIMBS`.
    ///
    /// Spends the step budget `self.k_remain`, leaving it at zero.
    ///
    /// Outputs:
    /// - `b` receives `gcd(a, b)` and `a` is left zeroed.
    ///
    /// # Approach
    ///
    /// Each round reads a top-bit-aligned magnitude window of `a`/`b` via
    /// [`scheduled_extract_compact_pair`] (deferred sign -- `a`/`b` are tracked as
    /// two's-complement values across rounds for efficiency),
    /// turns it into a `SPLIT_BATCH_SIZE`-step matrix via [`bingcd::partial_xgcd`], and applies
    /// it to the inputs. Once the window size drops to the small size threshold, the inputs
    /// are renormalized to non-negative, then [`Self::gcd_small`] takes over
    /// using an exact (non-scheduled) extraction.
    ///
    /// The extraction position is `E - W`, where `E = (k_remain + 1) / 2` is read before each
    /// batch is charged, so it depends only on the public step budget.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn gcd<const JACOBI: bool>(&mut self) -> Word {
        let mut jacobi_neg = 0;
        let (mut a_hi, mut b_hi) = (Limb::ZERO, Limb::ZERO);
        let mut b_true_limbs = self.len;

        // Stage 1 batches: extract according the fixed schedule, build the transition matrix, apply it to
        // `(a, b)` and update the schedule.
        while self.len > SMALL_THRESHOLD_LIMBS {
            // The schedule position `E = (k_remain + 1) / 2` is taken before this batch is
            // charged, and the extraction reads from a full word below it.
            let extract_pos = ((self.k_remain + 1) >> 1) - Limb::BITS;
            self.k_remain -= SPLIT_BATCH_SIZE;

            let (mut a_ext, mut b_ext) = (
                ExtendedIntRef::new(self.a.leading_mut(self.len), a_hi),
                ExtendedIntRef::new(self.b.leading_mut(self.len), b_hi),
            );
            (_, jacobi_neg) =
                gcd_reduce_large::<JACOBI>(&mut a_ext, &mut b_ext, extract_pos, jacobi_neg);
            (a_hi, b_hi) = (a_ext.hi, b_ext.hi);

            if self.k_remain <= self.window_shrink_at {
                self.len -= 1;
                self.window_shrink_at -= Limb::BITS;
            }

            let a_nonzero = self.a.limbs[0].is_nonzero();
            b_true_limbs = a_nonzero.select_u32(b_true_limbs as u32, self.len as u32) as usize;
        }

        // Restore `a`/`b` to non-negative before handing off: `a` is dropped back to
        // its absolute value directly (its true value always fits within `self.len`);
        // `b` is sign-extended from `b_true_limbs` and negated in place if needed.
        if self.b.nlimbs() > SMALL_THRESHOLD_LIMBS {
            ExtendedIntRef::new(self.a.leading_mut(self.len), a_hi).abs_drop_extension();

            let b_sign = b_hi.bit(Limb::HI_BIT);
            let b_true_limbs = b_true_limbs as u32;
            let b_neg_mask = Limb::choice_to_mask(b_sign);
            let mut carry = b_neg_mask.wrapping_neg();
            let mut i = 0;
            while i < self.len {
                (self.b.limbs[i], carry) =
                    self.b.limbs[i].bitxor(b_neg_mask).overflowing_add(carry);
                i += 1;
            }
            while i < self.b.nlimbs() {
                (self.b.limbs[i], carry) =
                    self.b.limbs[i].bitxor(b_neg_mask).overflowing_add(carry);
                let keep = Choice::from_u32_lt(i as u32, b_true_limbs);
                self.b.limbs[i] = Limb::select(Limb::ZERO, self.b.limbs[i], keep);
                i += 1;
            }
        }

        // Stage 2: batched matrix reduction down to a single-word remainder, then word steps.
        jacobi_neg ^ self.gcd_small::<JACOBI>()
    }

    /// Reduces `a`/`b` down to their gcd using the exact, non-deferred-sign counterpart of
    /// [`Self::gcd`]'s reduction loop, then finishing with single-`Word` elementary steps.
    ///
    /// Spends the step budget `self.k_remain`, leaving it at zero.
    ///
    /// Outputs:
    /// - `b` receives `gcd(a, b)` and `a` is left zeroed.
    ///
    /// The batched matrix reduction operates on `a`/`b` a `SPLIT_BATCH_SIZE`-step matrix at a
    /// time, until at most `WORD_BATCH_SIZE` steps of budget remain. At that point the operands
    /// (which occupy at most `k_remain` bits) fit a single `Word`, and the `Word` step function
    /// finishes the job. An operand that starts at a single limb skips the batches entirely.
    ///
    /// Each stage spends a fixed, input-independent number of steps per round --
    /// `SPLIT_BATCH_SIZE` for a batch, one for a `Word` step -- so how the budget
    /// splits across the two is pure arithmetic on `self.k_remain`, settled before any
    /// reduction runs. Each stage is then a counted loop over a value the compiler can fold to a
    /// constant whenever `self.k_remain` is one, as it is for the fixed-`LIMBS` entry points.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn gcd_small<const JACOBI: bool>(&mut self) -> Word {
        let mut jacobi_neg = 0;

        // The batches narrow the operands until the remaining budget fits a single `Word`: with
        // `k_remain` steps left to spend the tracked values occupy at most `k_remain` bits, so the
        // last `WORD_BATCH_SIZE` or fewer steps are a single `Word`'s. A single limb is stepped
        // directly. The step counts are pure arithmetic on `self.k_remain`, settled before any
        // reduction runs, so each stage is a counted loop the compiler can fold to a constant
        // whenever `self.k_remain` is one, as it is for the fixed-`LIMBS` entry points.
        //
        // The tracked window deliberately stays at its starting width: at these sizes only the
        // last few batches could run narrower, and a constant width lets the compiler unroll the
        // extraction and matrix application, which measured faster than shrinking.
        let batches = if self.len == 1 {
            0
        } else {
            self.k_remain
                .saturating_sub(WORD_BATCH_SIZE)
                .div_ceil(SPLIT_BATCH_SIZE)
        };
        let word_steps = self.k_remain - batches * SPLIT_BATCH_SIZE;

        // Batched matrix reduction down to what a single `Word` can finish.
        let mut i = 0;
        while i < batches {
            (_, _, jacobi_neg) = gcd_reduce_small::<JACOBI>(
                self.a.leading_mut(self.len),
                self.b.leading_mut(self.len),
                jacobi_neg,
            );
            i += 1;
        }

        // Single-`Word` finish for the remaining reduction steps.
        let (mut a, mut b) = (self.a.limbs[0].0, self.b.limbs[0].0);
        let mut i = 0;
        while i < word_steps {
            ((a, b), _, _, jacobi_neg) = bingcd::step_word(a, b, jacobi_neg);
            i += 1;
        }
        self.b.limbs[0] = Limb(b);
        self.k_remain = 0;

        jacobi_neg
    }

    /// Core signed extended-gcd engine shared by `xgcd_odd` and `invert_odd_mod`: reduces `b` (a
    /// copy of `cofactors.y`) and `a` down to their gcd using
    /// [`Self::gcd`]'s binary-GCD method (magnitude-comparison-based batched
    /// matrices, deferred sign), applying the identical sequence of step matrices to a
    /// caller-supplied [`CofactorPair`] alongside `(a, b)`.
    ///
    /// Inputs:
    /// - `cofactors.y` must be odd and the same width (`nlimbs`) as `x`, `cofactors.u`,
    ///   `cofactors.v`, and `gcd`.
    ///
    /// Outputs:
    /// - `self.b` receives `gcd(self.a, cofactors.y)`.
    /// - `cofactors.v` ends up satisfying `v * a ≡ gcd(a, b) (mod y)`; call
    ///   [`CofactorPair::finalize`] to reduce it into `[0, y)`.
    ///
    /// # Approach
    ///
    /// Two stages, matching [`Self::gcd`]'s own split:
    ///
    /// Stage 1 (`self.len > SMALL_THRESHOLD_LIMBS`) is the deferred-sign loop. Each batch extracts
    /// a top-bit-aligned magnitude window of `a`/`b` via [`scheduled_extract_compact_pair`] at the
    /// scheduled position `E - W`. It then turns the extraction into a
    /// `SPLIT_BATCH_SIZE`-step [`bingcd::BingcdMatrix`] via [`bingcd::partial_xgcd`], and applies it
    /// to `(a, b)`. `(u, v)` are updated by the same column-sign-adjusted matrix (captured as
    /// `a_neg`/`b_neg` before that update changes `a`/`b`'s own sign).
    ///
    /// Stage 2 (`self.len <= SMALL_THRESHOLD_LIMBS`) is [`Self::raw_xgcd_small`],
    /// handed whatever step budget Stage 1 left.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn raw_xgcd(&mut self, cofactors: &mut CofactorPair<'_>) {
        assert!(cofactors.u.nlimbs() >= self.len);

        let (mut a_hi, mut b_hi) = (Limb::ZERO, Limb::ZERO);
        let mut b_true_limbs = self.len;

        // Stage 1 batches: extract, build the transition matrix, apply it to `(a, b)` and
        // `(u, v)`, and update the schedule.
        while self.len > SMALL_THRESHOLD_LIMBS {
            // The schedule position `E = (k_remain + 1) / 2` is taken before this batch is
            // charged, and the extraction reads from a full word below it.
            let extract_pos = ((self.k_remain + 1) >> 1) - Limb::BITS;
            self.k_remain -= SPLIT_BATCH_SIZE;

            let (mut a_ext, mut b_ext) = (
                ExtendedIntRef::new(self.a.leading_mut(self.len), a_hi),
                ExtendedIntRef::new(self.b.leading_mut(self.len), b_hi),
            );
            let matrix;
            (matrix, _) = gcd_reduce_large::<false>(&mut a_ext, &mut b_ext, extract_pos, 0);

            cofactors.apply_matrix(
                matrix.column_signed_limb_matrix(a_hi.bit(Limb::HI_BIT), b_hi.bit(Limb::HI_BIT)),
                matrix.k,
            );
            (a_hi, b_hi) = (a_ext.hi, b_ext.hi);

            if self.k_remain <= self.window_shrink_at {
                self.len -= 1;
                self.window_shrink_at -= Limb::BITS;
            }

            let a_nonzero = self.a.limbs[0].is_nonzero();
            b_true_limbs = a_nonzero.select_u32(b_true_limbs as u32, self.len as u32) as usize;
        }

        // Stage 1 -> Stage 2 transition: normalize `(a, b)`. Stage 2 needs both operands non-negative
        // to safely use the exact, non-deferred-sign `extract_compact_pair` extraction. Skipped when
        // Stage 1 did not run at all as the terms are guaranteed non-negative.
        if self.b.nlimbs() > self.len {
            let a_ext = ExtendedIntRef::new(self.a.leading_mut(self.len), a_hi);
            cofactors.negate_u_if(a_ext.is_negative());
            a_ext.abs_drop_extension();

            let b_sign = b_hi.bit(Limb::HI_BIT);
            cofactors.negate_v_if(b_sign);

            let b_true_limbs = b_true_limbs as u32;
            let b_mask = Limb::choice_to_mask(b_sign);
            let mut carry = b_mask.wrapping_neg();
            let mut i = 0;
            while i < self.len {
                (self.b.limbs[i], carry) = self.b.limbs[i].bitxor(b_mask).overflowing_add(carry);
                i += 1;
            }
            while i < self.b.nlimbs() {
                (self.b.limbs[i], carry) = self.b.limbs[i].bitxor(b_mask).overflowing_add(carry);
                let keep = Choice::from_u32_lt(i as u32, b_true_limbs);
                self.b.limbs[i] = Limb::select(Limb::ZERO, self.b.limbs[i], keep);
                i += 1;
            }
        }

        // Pass off to perform the final reduction for the remaining budget, updating `(a, b)` and
        // `(u, v)` in place.
        self.raw_xgcd_small(cofactors);
    }

    /// [`Self::raw_xgcd`]'s Stage 2, split out so that the pair mirrors the way
    /// [`Self::gcd`] hands off to [`Self::gcd_small`]: the exact,
    /// non-deferred-sign counterpart of that method's own reduction loop, carrying a
    /// [`CofactorPair`] alongside `(a, b)`.
    ///
    /// Inputs:
    /// - `self.k_remain` is the step budget left to spend -- whatever [`Self::raw_xgcd`]'s Stage 1
    ///   did not consume, which is the whole budget when Stage 1 never ran at all.
    /// - `(a, b)` must already be non-negative, which is what Stage 1's transition guarantees and
    ///   what the exact extraction below relies on.
    ///
    /// Outputs:
    /// - `self.b` receives `gcd(self.a, cofactors.y)`.
    /// - `cofactors.v` ends up satisfying `v * a ≡ gcd(a, b) (mod y)`; call
    ///   [`CofactorPair::finalize`] to reduce it into `[0, y)`.
    ///
    /// Like [`Self::gcd_small`], the tracked window stays at the width Stage 1 left it at.
    ///
    /// `(a, b)` stay non-negative throughout (every round's `wrapping_apply_unsigned_shift`
    /// unconditionally re-corrects both) but `(u, v)`'s own update still needs the *row*-sign
    /// adjustment that call reports back (`a_negated`, `b_negated`).
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn raw_xgcd_small(&mut self, cofactors: &mut CofactorPair<'_>) {
        let batches = self
            .k_remain
            .saturating_sub(WORD_BATCH_SIZE)
            .div_ceil(SPLIT_BATCH_SIZE);
        let word_steps = self.k_remain - batches * SPLIT_BATCH_SIZE;

        let mut i = 0;
        while i < batches {
            let (matrix, (a_neg, b_neg), _) = gcd_reduce_small::<false>(
                self.a.leading_mut(self.len),
                self.b.leading_mut(self.len),
                0,
            );
            let signed = matrix.row_signed_limb_matrix(a_neg, b_neg);

            cofactors.apply_matrix(signed, matrix.k);
            i += 1;
        }

        let (b, matrix) =
            bingcd::partial_xgcd_word(self.a.limbs[0].0, self.b.limbs[0].0, word_steps);
        self.b.limbs[0] = Limb(b);
        self.k_remain = 0;
        cofactors.half_apply_matrix(matrix.signed_limb_matrix(), matrix.k);
    }
}

#[inline(always)]
const fn gcd_reduce_large<const JACOBI: bool>(
    a: &mut ExtendedIntRef<'_>,
    b: &mut ExtendedIntRef<'_>,
    extract_pos: u32,
    mut jacobi_neg: Word,
) -> (BingcdMatrix, Word) {
    debug_assert!(a.lo.nlimbs() == b.lo.nlimbs() && a.lo.nlimbs() >= 4);

    let (a_, b_, _shift) = scheduled_extract_compact_pair(a, b, extract_pos);

    let (matrix, j_neg, unhalted) =
        bingcd::partial_xgcd::<JACOBI>(a_, b_, Choice::FALSE, SPLIT_BATCH_SIZE);
    matrix.wrapping_apply_sign_correcting_shift(a, b);

    if JACOBI {
        let a_neg = a.hi.shr(Limb::HI_BIT);

        // If `a` is negated, then the divergence should be detected during the batch update
        debug_assert!(
            a_neg.lsb_to_choice().not().or(unhalted.not()).to_bool(),
            "a went negative without detection"
        );
        // `b` would only become negative after a previous undetected divergence
        debug_assert!(!b.is_negative().to_bool(), "b should not go negative");

        jacobi_neg ^= j_neg ^ (a_neg.0 & (b.lo.limbs[0].0 >> 1));
    }

    (matrix, jacobi_neg)
}

#[inline(always)]
const fn gcd_reduce_small<const JACOBI: bool>(
    a: &mut UintRef,
    b: &mut UintRef,
    mut jacobi_neg: Word,
) -> (BingcdMatrix, (Choice, Choice), Word) {
    let (a_, b_, exact) = extract_compact_pair(a, b);
    let (matrix, j_neg, unhalted) = bingcd::partial_xgcd::<JACOBI>(a_, b_, exact, SPLIT_BATCH_SIZE);
    let (a_neg, b_neg) = matrix.wrapping_apply_unsigned_shift(a, b);

    if JACOBI {
        // If `a` is negated, then the divergence should be detected during the batch update
        debug_assert!(
            a_neg.not().or(unhalted.not()).to_bool(),
            "a went negative without detection"
        );
        // `b` would only become negative after a previous undetected divergence
        debug_assert!(!b_neg.to_bool(), "b should not go negative");

        jacobi_neg ^= j_neg;
        jacobi_neg ^= word::choice_to_mask(a_neg) & (b.limbs[0].0 >> 1);
    }

    (matrix, (a_neg, b_neg), jacobi_neg)
}

/// Perform a single low-to-high scan latching the `(previous, current)` limb pair of `a`, `b`
/// and extracting a single aligned top word for each input.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn extract_compact_pair(a: &UintRef, b: &UintRef) -> ((Word, Word), (Word, Word), Choice) {
    debug_assert!(a.nlimbs() == b.nlimbs(), "mismatched sizes");

    // The latch is driven by a full-width mask rather than a `Choice`: every conversion of a
    // `Choice` to a mask is an optimization barrier, and this runs once per limb.
    let (mut a_prev, mut b_prev) = (0, 0);
    let (mut a_lo, mut a_hi, mut b_lo, mut b_hi) = (0, 0, 0, 0);
    let mut top_index = 0;

    let mut i = 0;
    while i < a.nlimbs() {
        let (a_i, b_i) = (a.limbs[i].0, b.limbs[i].0);
        let nz = nonzero_mask(a_i | b_i);
        a_lo ^= nz & (a_lo ^ a_prev);
        a_hi ^= nz & (a_hi ^ a_i);
        b_lo ^= nz & (b_lo ^ b_prev);
        b_hi ^= nz & (b_hi ^ b_i);
        top_index ^= nz as u32 & (top_index ^ i as u32);

        (a_prev, b_prev) = (a_i, b_i);
        i += 1;
    }

    let (a_top, b_top) = top_window_words((a_lo, b_lo), (a_hi, b_hi));
    // When the operands fit the top limb the window holds `a * 2^lz` and `b * 2^lz` with nothing
    // dropped, and every step keeps it that way: an even `a` halves exactly, and a subtract step
    // leaves `a - b` even (both odd). So the window compares exactly for the whole batch.
    let exact = Choice::from_u32_nz(top_index).not();
    ((a.limbs[0].0, a_top), (b.limbs[0].0, b_top), exact)
}

/// Assembles `(a_word, b_word)` out of the latched `(previous, current)` words of `a` and `b`
/// (each pair given as `(a, b)`) as produced by a single low-to-high scan over `a`/`b`: trims the
/// shared leading zeros of the current words, filling from the previous words.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn top_window_words(lo: (Word, Word), hi: (Word, Word)) -> (Word, Word) {
    // `hi_lz <= W`, and the wide shift reads the previous word in full when `hi_lz = W`.
    let hi_lz = (hi.0 | hi.1).leading_zeros();
    (
        ((word::join(lo.0, hi.0) << hi_lz) >> Word::BITS) as Word,
        ((word::join(lo.1, hi.1) << hi_lz) >> Word::BITS) as Word,
    )
}

/// Extract a comparable pair of compact high words from two signed operands.
///
/// `n` is the bit position `E - W`; the window is the three limbs starting at
/// limb containing this position, `[base, base + 3W)` with
/// `base = W * floor(n / W)`. This places the dominance threshold `E + S` inside
/// the window.
///
/// `trip_and_overflow` reduces each operand to a magnitude, so the two trips are
/// comparable regardless of sign. They are normalised by a single shared shift,
/// `min(clz(a_trip | b_trip), 2W)`.
///
/// The cap at `2W` limits the shift; without it, operands below `W` would be
/// normalised past their own base, with the low bits filled by fabricated zeros.
/// With the cap the extraction error is under one ulp, or under two for a
/// negative operand, whose one's-complement magnitude is `|x| - 1`.
///
/// # Overflow
///
/// `over` is set for an operand whose magnitude has significant bits above the
/// window. When either flag is set, both compact words are discarded and
/// replaced by the flags widened to full-word masks — `MAX` for the flagged
/// operand, zero for its partner — so the batch compares nonzero against zero
/// rather than two magnitudes.
///
/// Under the invariant `Phi <= 2E` the smaller operand satisfies `m <= E`,
/// which is below the window top, so at most one flag can fire and the
/// substitution always yields exactly one `MAX` and one zero. That ordering
/// then holds for every step of the batch rather than just the first: a halving
/// keeps a zero word zero, and a subtract either reduces the marker's word or
/// swaps the zero word, unchanged, into `b`. The marker loses at most one bit per
/// step, so at the batch's last comparison it is still at least
/// `MAX >> (S - 1) = 2^(q+2) - 1`, above the halt threshold `2^q`.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub const fn scheduled_extract_compact_pair(
    a: &ExtendedIntRef<'_>,
    b: &ExtendedIntRef<'_>,
    n: u32,
) -> ((Word, Word), (Word, Word), u32) {
    let extract_limb = (n >> Limb::LOG2_BITS) as usize;

    // All decisions are carried as full-width masks rather than `Choice` values: every conversion
    // of a `Choice` to a mask is an optimization barrier.
    let (a_neg, b_neg) = (msb_mask(a.hi.0), msb_mask(b.hi.0));
    let (a_trip, a_over) = trip_and_overflow(a.lo.trailing(extract_limb), a_neg);
    let (b_trip, b_over) = trip_and_overflow(b.lo.trailing(extract_limb), b_neg);
    debug_assert!(
        a_over & b_over == 0,
        "invalid extraction position, m <= E violation"
    );

    // `lz = min(clz(a_trip | b_trip), 2W)`: a zero word has `W` leading zeros, so adding the
    // middle word's count only when the top word is zero reaches exactly `2W` once both are.
    let (top, mid) = (a_trip[2] | b_trip[2], a_trip[1] | b_trip[1]);
    let lz = top.leading_zeros() + (!nonzero_mask(top) as u32 & mid.leading_zeros());
    let (a_top, b_top) = (shifted_top_word(a_trip, lz), shifted_top_word(b_trip, lz));

    let any_over = a_over | b_over;
    let (a0, b0) = (a.lo.limbs[0].0, b.lo.limbs[0].0);
    (
        (
            (a0 ^ a_neg).wrapping_sub(a_neg),
            a_top ^ (any_over & (a_top ^ a_over)),
        ),
        (
            (b0 ^ b_neg).wrapping_sub(b_neg),
            b_top ^ (any_over & (b_top ^ b_over)),
        ),
        Limb::BITS * 2 - (lz & !(any_over as u32)),
    )
}

/// The per-operand half of [`scheduled_extract_compact_pair`], called once for each
/// of `a`, `b`: extract a 3-word non-negative window from the start of `x` (one's complemented
/// when `neg` is set, and padded with zeros past the end of `x`), as well as a mask that is set
/// when the magnitude has significant bits above the window.
#[inline(always)]
const fn trip_and_overflow(x: &UintRef, neg: Word) -> ([Word; 3], Word) {
    let mut trip = [0; 3];
    let mut over = 0;
    let mut i = 0;
    while i < x.nlimbs() {
        let w = x.limbs[i].0 ^ neg;
        if i < 3 {
            trip[i] = w;
        } else {
            over |= w;
        }
        i += 1;
    }
    (trip, nonzero_mask(over))
}

/// The top word of the 3-word value `t` shifted left by `lz <= 2W`: below `W` it straddles the top
/// two words, from `W` up it straddles the lower two.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn shifted_top_word(t: [Word; 3], lz: u32) -> Word {
    // `lz >= W` exactly when `W - 1 - lz` borrows, setting the top bit of the `u32` difference.
    let ge = (((Limb::BITS - 1).wrapping_sub(lz) >> (u32::BITS - 1)) as Word).wrapping_neg();
    let hi = t[2] ^ (ge & (t[2] ^ t[1]));
    let lo = t[1] ^ (ge & (t[1] ^ t[0]));
    let s = lz - (Limb::BITS & ge as u32);
    ((word::join(lo, hi) << s) >> Word::BITS) as Word
}

/// `Word::MAX` if the top bit of `x` is set, zero otherwise.
#[inline(always)]
const fn msb_mask(x: Word) -> Word {
    (x >> (Word::BITS - 1)).wrapping_neg()
}

/// `Word::MAX` if `x` is nonzero, zero otherwise.
#[inline(always)]
const fn nonzero_mask(x: Word) -> Word {
    msb_mask(x | x.wrapping_neg())
}

#[cfg(all(test, feature = "rand_core"))]
#[allow(
    clippy::cast_possible_truncation,
    clippy::integer_division_remainder_used,
    reason = "test"
)]
mod tests {
    use super::{ExtendedIntRef, extract_compact_pair, scheduled_extract_compact_pair};
    use crate::{Limb, Random, Uint, UintRef, Word};
    use chacha20::ChaCha8Rng;
    use rand_core::{Rng, SeedableRng};

    /// The straightforward `Uint<3>` formulation of [`scheduled_extract_compact_pair`].
    fn reference(
        a: (&[Limb], Limb),
        b: (&[Limb], Limb),
        n: u32,
    ) -> ((Word, Word), (Word, Word), u32) {
        let extract_limb = (n / Limb::BITS) as usize;
        let trip = |x: &[Limb], hi: Limb| {
            let neg = if hi.0 >> (Word::BITS - 1) == 1 {
                Word::MAX
            } else {
                0
            };
            let x = &x[extract_limb..];
            let mut words = [neg; 3];
            for (w, l) in words.iter_mut().zip(x) {
                *w = l.0;
            }
            let over = x.iter().skip(3).any(|l| l.0 ^ neg != 0);
            (Uint::<3>::from_words(words.map(|w| w ^ neg)), over, neg)
        };
        let ((a_trip, a_over, a_neg), (b_trip, b_over, b_neg)) = (trip(a.0, a.1), trip(b.0, b.1));
        let lz = a_trip.bitor(&b_trip).leading_zeros().min(2 * Limb::BITS);
        let any_over = a_over || b_over;
        let top = |t: Uint<3>, over: bool| {
            if any_over {
                if over { Word::MAX } else { 0 }
            } else {
                t.shl(lz).limbs[2].0
            }
        };
        let low = |x: &[Limb], neg: Word| {
            if neg != 0 {
                x[0].0.wrapping_neg()
            } else {
                x[0].0
            }
        };
        (
            (low(a.0, a_neg), top(a_trip, a_over)),
            (low(b.0, b_neg), top(b_trip, b_over)),
            if any_over {
                2 * Limb::BITS
            } else {
                2 * Limb::BITS - lz
            },
        )
    }

    /// Random non-negative value of `len` limbs (in a fixed buffer) with a random bit length.
    fn magnitude(rng: &mut ChaCha8Rng, len: usize) -> [Limb; 12] {
        let bits = rng.next_u32() % (len as u32 * Limb::BITS + 1);
        let mut limbs = [Limb::ZERO; 12];
        for (i, limb) in limbs.iter_mut().enumerate().take(len) {
            let word = Uint::<1>::random_from_rng(rng).limbs[0].0;
            let keep = bits.saturating_sub(i as u32 * Limb::BITS).min(Limb::BITS);
            *limb = Limb(if keep == 0 {
                0
            } else {
                word >> (Limb::BITS - keep)
            });
        }
        limbs
    }

    /// Random signed operand of `len` limbs: a [`magnitude`], stored one's complemented with an
    /// all-ones `hi` when negative.
    fn operand(rng: &mut ChaCha8Rng, len: usize) -> ([Limb; 12], Limb) {
        let mut limbs = magnitude(rng, len);
        if rng.next_u32() & 1 == 1 {
            for limb in limbs.iter_mut().take(len) {
                *limb = limb.not();
            }
            (limbs, Limb::MAX)
        } else {
            (limbs, Limb::ZERO)
        }
    }

    #[test]
    fn scheduled_extract_matches_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (mut below_w, mut above_w, mut capped, mut over, mut neg) = (0, 0, 0, 0, 0);
        for _ in 0..50_000 {
            let len = 4 + (rng.next_u32() % 9) as usize;
            let (a_lo, a_hi) = operand(&mut rng, len);
            let (b_lo, b_hi) = operand(&mut rng, len);
            let n = rng.next_u32() % (len as u32 * Limb::BITS);

            let expected = reference((&a_lo[..len], a_hi), (&b_lo[..len], b_hi), n);
            let (a_over, b_over) = (expected.0.1 == Word::MAX, expected.1.1 == Word::MAX);
            if a_over && b_over {
                // Excluded by the schedule's invariant (and by a debug assertion).
                continue;
            }

            let (mut a_buf, mut b_buf) = (a_lo, b_lo);
            let a = ExtendedIntRef::new(UintRef::new_mut(&mut a_buf[..len]), a_hi);
            let b = ExtendedIntRef::new(UintRef::new_mut(&mut b_buf[..len]), b_hi);
            let actual = scheduled_extract_compact_pair(&a, &b, n);
            assert_eq!(
                actual,
                expected,
                "len={len} n={n} a={:?}/{a_hi:?} b={:?}/{b_hi:?}",
                &a_lo[..len],
                &b_lo[..len]
            );

            let lz = 2 * Limb::BITS - expected.2;
            match () {
                _ if expected.2 == 2 * Limb::BITS && (a_over || b_over) => over += 1,
                _ if lz == 2 * Limb::BITS => capped += 1,
                _ if lz >= Limb::BITS => above_w += 1,
                _ => below_w += 1,
            }
            neg += usize::from(a_hi == Limb::MAX || b_hi == Limb::MAX);
        }
        for (case, count) in [
            ("lz < W", below_w),
            ("W <= lz < 2W", above_w),
            ("lz = 2W", capped),
            ("overflow", over),
            ("negative", neg),
        ] {
            assert!(count > 1_000, "only {count} cases with {case}");
        }
    }

    #[test]
    fn extract_matches_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let (mut single, mut multi) = (0, 0);
        for _ in 0..50_000 {
            let len = 1 + (rng.next_u32() % 8) as usize;
            let (a, mut b) = (magnitude(&mut rng, len), magnitude(&mut rng, len));
            b[0].0 |= 1;

            // Reference: the top nonzero limb of `a | b` and the one below it.
            let t = (0..len)
                .rev()
                .find(|&i| (a[i].0 | b[i].0) != 0)
                .unwrap_or(0);
            let prev = |x: &[Limb; 12]| if t == 0 { 0 } else { x[t - 1].0 };
            let lz = (a[t].0 | b[t].0).leading_zeros();
            let top = |x: &[Limb; 12]| {
                (((u128::from(x[t].0) << Word::BITS | u128::from(prev(x))) << lz) >> Word::BITS)
                    as Word
            };
            let expected = ((a[0].0, top(&a)), (b[0].0, top(&b)), t == 0);

            let ((a0, a_top), (b0, b_top), is_exact) =
                extract_compact_pair(UintRef::new(&a[..len]), UintRef::new(&b[..len]));
            assert_eq!(
                ((a0, a_top), (b0, b_top), is_exact.to_bool()),
                expected,
                "len={len} a={:?} b={:?}",
                &a[..len],
                &b[..len]
            );

            if t == 0 {
                single += 1;
            } else {
                multi += 1;
            }
        }
        for (case, count) in [("single limb", single), ("multi limb", multi)] {
            assert!(count > 1_000, "only {count} {case} cases");
        }
    }
}
