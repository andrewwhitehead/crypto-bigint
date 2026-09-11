use super::{
    CofactorPair, ExtendedIntRef, SMALL_THRESHOLD_LIMBS, SPLIT_BATCH_SIZE, WORD_BATCH_SIZE, bingcd,
};
use crate::{
    Choice, Limb, Uint, UintRef, WideWord, Word, modular::gcd::bingcd::BingcdMatrix,
    primitives::u32_min, word,
};

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
}

impl<'a> GcdPair<'a> {
    /// Starts tracking `a`, `b` at their full width, with the step budget Pornin's Phi bound
    /// requires for operands of that width.
    pub const fn new(a: &'a mut UintRef, b: &'a mut UintRef) -> Self {
        let k_remain = bingcd::iterations(a.bits_precision());
        Self::new_with_budget(a, b, k_remain)
    }

    /// [`Self::new`] with the step budget supplied rather than derived: the caller must have
    /// `bitlen(a), bitlen(b) <= k_remain` (Pornin's Phi bound).
    pub const fn new_with_budget(a: &'a mut UintRef, b: &'a mut UintRef, k_remain: u32) -> Self {
        let len = a.nlimbs();
        assert!(b.nlimbs() == len, "a and b must have the same width");
        assert!(b.limbs[0].0 & 1 == 1, "b must be odd");
        Self {
            a,
            b,
            len,
            k_remain,
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
    /// [`Self::scheduled_extract_compact_pair`] (deferred sign -- `a`/`b` are tracked as
    /// two's-complement values across rounds for efficiency),
    /// turns it into a `SPLIT_BATCH_SIZE`-step matrix via [`bingcd::partial_xgcd`], and applies
    /// it to the inputs. Once the window size drops to the small size threshold, the inputs
    /// are renormalized to non-negative, then [`Self::gcd_small`] takes over
    /// using an exact (non-scheduled) extraction.
    ///
    /// The extraction position walks a schedule that starts one batch's worth of bits below full
    /// width and falls by half of whatever's been consumed so far every round unconditionally.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn gcd<const JACOBI: bool>(&mut self) -> Word {
        let mut jacobi_neg = 0;
        let mut window_shrink_at = (self.len as u32 - 1) << Limb::LOG2_BITS;
        let (mut a_hi, mut b_hi) = (Limb::ZERO, Limb::ZERO);
        let mut b_true_limbs = self.len;

        // Stage 1 batches: extract according the fixed schedule, build the transition matrix, apply it to
        // `(a, b)` and update the schedule.
        while self.len > SMALL_THRESHOLD_LIMBS {
            self.k_remain -= SPLIT_BATCH_SIZE;
            let extract_pos = (self.k_remain + 1) >> 1;

            let (mut a_ext, mut b_ext) = (
                ExtendedIntRef::new(self.a.leading_mut(self.len), a_hi),
                ExtendedIntRef::new(self.b.leading_mut(self.len), b_hi),
            );
            (_, jacobi_neg) =
                gcd_reduce_large::<JACOBI>(&mut a_ext, &mut b_ext, extract_pos, jacobi_neg);
            (a_hi, b_hi) = (a_ext.hi, b_ext.hi);

            if self.k_remain <= window_shrink_at {
                self.len -= 1;
                window_shrink_at -= Limb::BITS;
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

        // Stage 2: batched matrix reduction down to a 2-limb window.
        while self.len > 2 {
            let matrix;
            (matrix, _, jacobi_neg) = gcd_reduce_small::<JACOBI>(
                self.a.leading_mut(self.len),
                self.b.leading_mut(self.len),
                jacobi_neg,
            );
            self.k_remain -= matrix.k;

            if self.k_remain <= window_shrink_at {
                self.len -= 1;
                window_shrink_at -= Limb::BITS;
            }
        }

        // Stage 3: reduce by the remaining budget using register-only reduction.
        jacobi_neg ^= self.gcd_fixed::<JACOBI, 2>();

        jacobi_neg
    }

    /// Reduces `a`/`b` down to their gcd using the exact, non-deferred-sign counterpart of
    /// [`Self::gcd`]'s reduction loop, then using specialized reductions for 2 and 1 limb sizes.
    ///
    /// Spends the step budget `self.k_remain`, leaving it at zero.
    ///
    /// Outputs:
    /// - `b` receives `gcd(a, b)` and `a` is left zeroed.
    ///
    /// The batched matrix reduction operates on `a`/`b` a `SPLIT_BATCH_SIZE`-step matrix at a
    /// time, narrowing `self.len` via the same `window_shrink_at` formula [`Self::gcd`] uses for
    /// its own reduction loop, until it reaches 2 limbs. The reduction of the final limbs uses
    /// step functions optimized for these sizes.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn gcd_fixed<const JACOBI: bool, const LIMBS: usize>(&mut self) -> Word {
        let mut jacobi_neg = 0;

        assert!(self.len <= LIMBS);

        // Batched matrix reduction down to a 2-limb window.
        if self.len > 2 {
            let (mut a, mut b) = (
                self.a.to_uint_resize::<LIMBS>(),
                self.b.to_uint_resize::<LIMBS>(),
            );
            let (ua, ub) = (a.as_mut_uint_ref(), b.as_mut_uint_ref());

            while self.k_remain > WideWord::BITS {
                let m;
                (m, _, jacobi_neg) = gcd_reduce_small::<JACOBI>(ua, ub, jacobi_neg);
                self.k_remain -= m.k;
            }
            self.a.leading_mut(self.len).copy_from(ua.leading(self.len));
            self.b.leading_mut(self.len).copy_from(ub.leading(self.len));
            self.len = 2;
        }

        // Perform `WideWord` elementary steps until `remaining_steps` proves a single `Word` suffices.
        if self.len == 2 {
            let (a2, b2) = (self.a.leading_mut(2), self.b.leading_mut(2));
            let mut a = a2.to_wide_word_unchecked();
            let mut b = b2.to_wide_word_unchecked();
            while self.k_remain > Limb::BITS {
                let j_neg;
                ((a, b), _, _, j_neg) = bingcd::step_wide_word(a, b);
                jacobi_neg ^= j_neg;
                self.k_remain -= 1;
            }
            a2.set_from_wide_word(a);
            b2.set_from_wide_word(b);
        }

        // Single-`Word` finish for the remaining reduction steps.
        let (mut a, mut b) = (self.a.limbs[0].0, self.b.limbs[0].0);
        while self.k_remain != 0 {
            ((a, b), _, _, jacobi_neg) = bingcd::step_word(a, b, jacobi_neg);
            self.k_remain -= 1;
        }
        self.b.limbs[0] = Limb(b);

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
    /// a top-bit-aligned magnitude window of `a`/`b` via [`Self::scheduled_extract_compact_pair`],
    /// walking an unconditionally-moving schedule position. It them turns the extraction into a
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

        let mut window_shrink_at = (self.len as u32 - 1) << Limb::LOG2_BITS;

        let (mut a_hi, mut b_hi) = (Limb::ZERO, Limb::ZERO);
        let mut b_true_limbs = self.len;

        // Stage 1 batches: extract, build the transition matrix, apply it to `(a, b)` and
        // `(u, v)`, and update the schedule.
        while self.len > SMALL_THRESHOLD_LIMBS {
            self.k_remain -= SPLIT_BATCH_SIZE;
            let extract_pos = (self.k_remain + 1) >> 1;

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

            if self.k_remain <= window_shrink_at {
                self.len -= 1;
                window_shrink_at -= Limb::BITS;
            }

            let a_nonzero = self.a.limbs[0].is_nonzero();
            b_true_limbs = a_nonzero.select_u32(b_true_limbs as u32, self.len as u32) as usize;
        }

        // Stage 1 -> Stage 2 transition: normalize `(a, b)`. Stage 2 needs both operands non-negative
        // to safely use the exact, non-deferred-sign `top_window_pair` extraction. Skipped when
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
    /// `window_shrink_at` is re-derived from `self.len` rather than threaded in: Stage 1 pairs
    /// every `self.len -= 1` with a `Limb::BITS` decrement, so `(self.len - 1) * Limb::BITS` is
    /// exactly the value it would have handed over.
    ///
    /// `(a, b)` stay non-negative throughout (every round's `wrapping_apply_unsigned_shift`
    /// unconditionally re-corrects both) but `(u, v)`'s own update still needs the *row*-sign
    /// adjustment that call reports back (`a_negated`, `b_negated`).
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn raw_xgcd_small(&mut self, cofactors: &mut CofactorPair<'_>) {
        let mut window_shrink_at = (self.len as u32 - 1) << Limb::LOG2_BITS;

        while self.k_remain > WORD_BATCH_SIZE {
            let (matrix, (a_neg, b_neg), _) = gcd_reduce_small::<false>(
                self.a.leading_mut(self.len),
                self.b.leading_mut(self.len),
                0,
            );
            let signed = matrix.row_signed_limb_matrix(a_neg, b_neg);

            self.k_remain -= matrix.k;
            cofactors.apply_matrix(signed, matrix.k);

            if self.len != 1 && self.k_remain <= window_shrink_at {
                self.len -= 1;
                window_shrink_at -= Limb::BITS;
            }
        }

        let (b, matrix, _) =
            bingcd::partial_xgcd_word(self.a.limbs[0].0, self.b.limbs[0].0, self.k_remain);
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
            a_neg
                .lsb_to_choice()
                .not()
                .or(unhalted.not())
                .to_bool_vartime(),
            "a went negative without detection"
        );
        // `b` would only become negative after a previous undetected divergence
        debug_assert!(
            !b.is_negative().to_bool_vartime(),
            "b should not go negative"
        );

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
            a_neg.not().or(unhalted.not()).to_bool_vartime(),
            "a went negative without detection"
        );
        // `b` would only become negative after a previous undetected divergence
        debug_assert!(!b_neg.to_bool_vartime(), "b should not go negative");

        jacobi_neg ^= j_neg;
        jacobi_neg ^= word::choice_to_mask(a_neg) & (b.limbs[0].0 >> 1);
    }

    (matrix, (a_neg, b_neg), jacobi_neg)
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
/// The cap at `2W` limits the shift; without it, operands below `W` would normalised
/// past their own base and the low bits filled with fabricated zeros. With the cap
/// the extraction error is under one ulp, however full the window is under two for
/// a negative operand, whose one's-complement magnitude is `|x| - 1`.
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
/// then holds for every step of the batch rather than just the first: a step
/// replaces the larger operand and leaves the other untouched, so nothing can
/// make the zero side nonzero, and `MAX` survives `S` halvings for any
/// `S <= W - 1`.
#[inline(always)]
pub const fn scheduled_extract_compact_pair(
    a: &ExtendedIntRef<'_>,
    b: &ExtendedIntRef<'_>,
    n: u32,
) -> ((Word, Word), (Word, Word), u32) {
    let extract_limb = (n >> Limb::LOG2_BITS) as usize;

    let (a_neg, b_neg) = (a.is_negative(), b.is_negative());
    let (a_trip, a_over) = trip_and_overflow(a.lo.trailing(extract_limb), a_neg);
    let (b_trip, b_over) = trip_and_overflow(b.lo.trailing(extract_limb), b_neg);

    let lz = u32_min(a_trip.bitor(&b_trip).leading_zeros(), Limb::BITS * 2);

    let (a_top, b_top) = (a_trip.shl(lz).limbs[2].0, b_trip.shl(lz).limbs[2].0);
    debug_assert!(
        !a_over.and(b_over).to_bool_vartime(),
        "invalid extraction position, m <= E violation"
    );
    let (a0, b0) = (a.lo.limbs[0].0, b.lo.limbs[0].0);
    let any_over = a_over.or(b_over);
    (
        (
            word::select(a0, a0.wrapping_neg(), a_neg),
            word::select(a_top, word::choice_to_mask(a_over), any_over),
        ),
        (
            word::select(b0, b0.wrapping_neg(), b_neg),
            word::select(b_top, word::choice_to_mask(b_over), any_over),
        ),
        any_over.select_u32(Limb::BITS * 2 - lz, Limb::BITS * 2),
    )
}

/// Perform a single low-to-high scan latching the `(previous, current)` limb pair of `a`, `b`
/// and extracting a single aligned top word for each input.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn extract_compact_pair(a: &UintRef, b: &UintRef) -> ((Word, Word), (Word, Word), Choice) {
    debug_assert!(a.nlimbs() == b.nlimbs(), "mismatched sizes");
    let mut lo_pair = 0;
    let (mut top_lo_pair, mut top_hi_pair) = (0, 0);
    let mut top_index = 0;

    let mut i = 0;
    while i < a.nlimbs() {
        let (a_i, b_i) = (a.limbs[i], b.limbs[i]);
        let hi_pair = word::join(a_i.0, b_i.0);
        let nz = a_i.bitor(b_i).is_nonzero();
        top_lo_pair = word::select_wide(top_lo_pair, lo_pair, nz);
        top_hi_pair = word::select_wide(top_hi_pair, hi_pair, nz);
        top_index = nz.select_u32(top_index, i as u32);

        lo_pair = hi_pair;
        i += 1;
    }

    let (a_top, b_top, shift) =
        top_window_words(word::split_wide(top_lo_pair), word::split_wide(top_hi_pair));
    let exact = Choice::from_u32_nz(top_index | shift).not();
    ((a.limbs[0].0, a_top), (b.limbs[0].0, b_top), exact)
}

/// Assembles `(a_word, b_word, shift)` out of the latched `(previous, current)` `Word` pairs
/// (`a` in the low half, `b` in the high half) as produced by a single low-to-high scan
/// over `a`/`b`. Trims leading zeros from the wide word values and returns the high
/// word along with the shift value.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn top_window_words(lo: (Word, Word), hi: (Word, Word)) -> (Word, Word, u32) {
    let hi_lz = (hi.0 | hi.1).leading_zeros();
    let shift = Word::BITS - hi_lz;
    (
        Limb(lo.0).unbounded_shr(shift).0 | (hi.0 << hi_lz),
        Limb(lo.1).unbounded_shr(shift).0 | (hi.1 << hi_lz),
        shift,
    )
}

/// The per-operand half of [`scheduled_extract_compact_pair`], called once for each
/// of `a`, `b`: extract a 3-word non-negative window from the start of `x`, as well as a flag
/// indicating whether the absolute value of the upper words is non-zero.
#[inline(always)]
const fn trip_and_overflow(x: &UintRef, neg: Choice) -> (Uint<3>, Choice) {
    let neg_mask = Limb::choice_to_mask(neg);

    // Extract 3 words from the beginning of `x`, padding with the sign word if necessary.
    let mut trip_raw = Uint::<3>::from_words([neg_mask.0; 3]);
    let len = if x.nlimbs() > 3 { 3 } else { x.nlimbs() };
    trip_raw
        .as_mut_uint_ref()
        .leading_mut(len)
        .copy_from(x.leading(len));
    // Swap for the one's complement `trip_raw.not()` if negative.
    let trip = trip_raw.bitxor_limb(neg_mask);

    // Reduce the limbs above `len` determining whether they are all sign bits.
    let mut over_acc = Limb::ZERO;
    let mut i = len;
    while i < x.nlimbs() {
        over_acc = over_acc.bitor(x.limbs[i].bitxor(neg_mask));
        i += 1;
    }
    let over = over_acc.is_nonzero();
    (trip, over)
}
