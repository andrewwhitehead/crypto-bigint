use super::{CofactorPair, ExtendedIntRef, GCD_BATCH_SIZE, SignedLimbMatrix, bingcd};
use crate::{JacobiSymbol, Limb, UintRef, Word};

/// The running state of a vartime binary GCD reduction.
#[derive(Debug)]
pub struct GcdPairVartime<'a> {
    /// The shrinking operand; reaches (and stays at) zero once the gcd has been found.
    a: &'a mut UintRef,
    /// The odd operand; ends up holding the gcd.
    b: &'a mut UintRef,
    /// `a`'s current bit length, re-measured after every round. `0` is both the loop
    /// termination condition and (combined with `b` staying odd) the signal that `b`
    /// now holds the final gcd.
    a_bits: u32,
    /// Current width (in limbs) of the `a`/`b` tracking window; starts at `a.nlimbs()` and only
    /// ever shrinks, via [`Self::truncate`] confirming the top limb of both is actually zero.
    len: usize,
}

impl<'a> GcdPairVartime<'a> {
    /// Starts tracking `a`, `b` at their full width.
    pub const fn new(a: &'a mut UintRef, b: &'a mut UintRef) -> Self {
        let len = a.nlimbs();
        assert!(b.nlimbs() == len, "a and b must have the same width");
        assert!(b.limbs[0].0 & 1 == 1, "b must be odd");
        let a_bits = a.bits_vartime();
        Self { a, a_bits, b, len }
    }
}

impl GcdPairVartime<'_> {
    /// Reduces `(a, b)` to `gcd(a, b)` in place.
    ///
    /// Returns the gcd in `b` (`a` ends at zero).
    #[inline(always)]
    pub const fn gcd_odd(&mut self) {
        while self.a_bits != 0 {
            self.strip_trailing_zeros();
            self.truncate();

            if self.len == 1 {
                let (gcd, _) = bingcd::gcd_word_vartime(self.a.limbs[0].0, self.b.limbs[0].0);
                self.a.set_from_limb(Limb::ZERO);
                self.a_bits = 0;
                self.b.set_from_limb(Limb(gcd));
                break;
            }

            self.reduce_bingcd(GCD_BATCH_SIZE);
        }
    }

    /// Computes the Jacobi symbol `(a|b)`.
    #[inline(always)]
    pub const fn jacobi_symbol(&mut self) -> JacobiSymbol {
        let mut jacobi_neg = 0;

        while self.a_bits != 0 {
            let tz = self.strip_trailing_zeros();
            if tz & 1 == 1 {
                let b_lo = self.b.limbs[0].0;
                jacobi_neg ^= (b_lo >> 1) ^ (b_lo >> 2);
            }
            self.truncate();

            if self.len == 1 {
                let (gcd, j) = bingcd::gcd_word_vartime(self.a.limbs[0].0, self.b.limbs[0].0);
                self.a.set_from_limb(Limb::ZERO);
                self.a_bits = 0;
                self.b.set_from_limb(Limb(gcd));
                jacobi_neg ^= j;
                break;
            }

            let (_, _, j) = self.reduce_bingcd(GCD_BATCH_SIZE);
            jacobi_neg ^= j;
        }

        if self.b.leading(self.len).is_one().to_bool_vartime() {
            JacobiSymbol::from_sign(jacobi_neg & 1)
        } else {
            JacobiSymbol::Zero
        }
    }

    /// Reduces `(a, b)` to `gcd(a, b)` in `b` exactly like [`Self::gcd_odd`], but additionally
    /// accumulates coefficients into `cofactors` -- tracked relative to `cofactors`'s own fixed
    /// odd modulus `y`, used only for its on-demand reductions, not part of the core gcd
    /// computation itself -- so that they end up satisfying the same `coefficient * a_orig ≡ gcd
    /// (mod y)` relationship `raw_xgcd_odd`'s `d` does. Unlike `gcd_odd`, this does *not* fast-path
    /// through `strip_trailing_zeros` every round: only a single upfront strip (`init_k`) happens,
    /// with its count folded directly into `cofactors.k` (cheaper than a full matrix round for a
    /// pure shift); every factor of two after that is left for `reduce_bingcd`'s own batched
    /// matrix to account for, since a bare shift on `a` alone -- correct for `gcd_odd`'s
    /// magnitude-only goal -- would silently desynchronize `cofactors` from `(a, b)`'s evolving
    /// linear relationship. Also skips the `len <= 2` closed-form fast path `gcd_odd` uses, since
    /// that path doesn't produce a matrix for `cofactors` to consume.
    #[inline(always)]
    pub const fn raw_xgcd<'a>(&mut self, cofactors: &mut CofactorPair<'a>) {
        let init_k = self.strip_trailing_zeros();

        while self.a_bits != 0 {
            self.truncate();

            let (m, mk, _) = self.reduce_bingcd(GCD_BATCH_SIZE);
            cofactors.apply_matrix_vartime(m, mk);
        }

        cofactors.defer_k(init_k);
    }

    /// Runs one batched round of `max_batch` (or fewer, if the top-window comparison stops being
    /// trustworthy early) elementary binary-GCD steps, applies the resulting matrix to the full
    /// tracked `(a, b)` window, and re-measures `a`'s bit length.
    ///
    /// The matrix itself is built from a top-bit-aligned window of `a`/`b` (`a_top`/`b_top`,
    /// normalized so their combined leading-zero count is `0` by shifting in bits from the next
    /// limb down when the current top limb is short) paired with their exact low limb, fed to
    /// [`bingcd::partial_xgcd_vartime`] -- the same
    /// top-bit-windowing approach [`jacobi_symbol`](super::jacobi_symbol) uses for
    /// its own constant-time batches, but with an `exact` flag driven by whether `top` itself fits
    /// a single limb (`top < Limb::BITS`) rather than the constant-time versions' schedule-derived
    /// `exact`/`above_threshold` machinery, since vartime execution can just re-measure the window fresh
    /// every round instead of trusting a stale one.
    ///
    /// After applying, `a`/`b` are independently renormalized to non-negative (each possibly
    /// needing its own row of the matrix negated to match, so a caller composing matrices across
    /// rounds -- like [`Self::raw_xgcd`]'s `cofactors` -- stays correct), shifted right by the
    /// round's consumed step count, and the returned `SignedLimbMatrix` reflects those same
    /// per-row negations. Returns `(matrix, steps_consumed, jacobi_neg)`; `gcd_odd`/`raw_xgcd`
    /// ignore `jacobi_neg`, `jacobi_symbol` folds it into its running sign.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    const fn reduce_bingcd(&mut self, max_batch: u32) -> (SignedLimbMatrix, u32, Word) {
        let (a_, b_) = self.extract_pair();
        let (m, j_neg1) = bingcd::partial_xgcd_vartime(a_, b_, max_batch, self.len == 1);
        let (m2, j_neg2) = self.apply_matrix(m);
        (m2, m.k, j_neg1 ^ j_neg2)
    }

    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    const fn apply_matrix(&mut self, matrix: bingcd::BingcdMatrix) -> (SignedLimbMatrix, Word) {
        let mut jacobi_neg = 0;

        let (a, b) = (self.a.leading_mut(self.len), self.b.leading_mut(self.len));
        let mut m2 = matrix.signed_limb_matrix();
        let (mut ae, mut be) = (
            ExtendedIntRef::new(a, Limb::ZERO),
            ExtendedIntRef::new(b, Limb::ZERO),
        );
        m2.wrapping_apply(&mut ae, &mut be);

        if be.is_negative_vartime() {
            m2.negate_bottom_row();
            be.wrapping_neg_assign();
        }
        be.shr_assign_limb_unsigned(matrix.k);
        be.unsigned_drop_extension();
        let b_lo = b.limbs[0];
        debug_assert!(b_lo.0 & 1 == 1, "b must be odd");

        if ae.is_negative_vartime() {
            m2.negate_top_row();
            ae.wrapping_neg_assign();
            jacobi_neg ^= b_lo.0 >> 1;
        }
        ae.shr_assign_limb_unsigned(matrix.k);
        ae.unsigned_drop_extension();

        self.a_bits = a.bits_vartime();

        (m2, jacobi_neg)
    }

    #[inline(always)]
    const fn extract_pair(&self) -> ((Word, Word), (Word, Word)) {
        let top = self.len - 1;
        let (mut a_top, mut b_top) = (self.a.limbs[top], self.b.limbs[top]);
        let lz = a_top.bitor(b_top).leading_zeros();
        if lz != 0 && top != 0 {
            (a_top, b_top) = (
                a_top
                    .shl(lz)
                    .bitor(self.a.limbs[top - 1].shr(Limb::BITS - lz)),
                b_top
                    .shl(lz)
                    .bitor(self.b.limbs[top - 1].shr(Limb::BITS - lz)),
            );
        };
        ((self.a.limbs[0].0, a_top.0), (self.b.limbs[0].0, b_top.0))
    }

    /// Divides `a` by its own trailing-zero-bit factor of two (a no-op, returning `0`, once `a` has
    /// already reached zero), shrinking `a_bits` to match, and returns the count stripped. Sound
    /// as a magnitude-only operation (`gcd_odd`) because `b` stays odd throughout, so factors of
    /// two can only ever come from `a`; callers that also need a coefficient consistent with the
    /// stripped value (`raw_xgcd`) must fold the returned count in themselves rather than treating
    /// this as free.
    #[inline(always)]
    const fn strip_trailing_zeros(&mut self) -> u32 {
        if self.a_bits == 0 {
            0
        } else {
            let a = self.a.leading_mut(self.len);
            let tz = a.trailing_zeros_vartime();
            a.unbounded_shr_assign_vartime(tz);
            self.a_bits -= tz;
            tz
        }
    }

    /// Shrinks the tracked window while both `a`'s and `b`'s current top limb are exactly zero
    /// (vartime checks against the *actual* data, unlike `safegcd`'s public-step-count-only
    /// schedule). Never grows `len` back -- only ever called after a round has just narrowed `a`,
    /// so re-widening isn't needed here the way [`CofactorPair`]'s own growth needs it.
    #[inline(always)]
    const fn truncate(&mut self) {
        while self.len > 1
            && self.a.limbs[self.len - 1]
                .bitor(self.b.limbs[self.len - 1])
                .is_zero_vartime()
        {
            self.len -= 1;
        }
    }
}
