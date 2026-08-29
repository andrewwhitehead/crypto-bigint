use super::{CofactorPair, ExtendedIntRef, SignedLimbMatrix};
use crate::{JacobiSymbol, Limb, Odd, UintRef, Word};

/// The running state of a vartime binary GCD reduction.
#[derive(Debug)]
struct GcdPairVartime<'a> {
    /// The shrinking operand; reaches (and stays at) zero once the gcd has been found.
    a: &'a mut UintRef,
    /// The odd operand; ends up holding the gcd.
    b: &'a mut UintRef,
    /// `a`'s current bit length, re-measured after every round. `0` is both the loop
    /// termination condition and (combined with `b` staying odd) the signal that `b`
    /// now holds the final gcd.
    a_bits: u32,
    /// Current width (in limbs) of the live `a`/`b` window; starts at `a.nlimbs()` and only ever
    /// shrinks, via [`Self::truncate`] confirming the top limb of both is actually zero.
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

            if self.len <= 2 {
                let (gcd, _) = super::bingcd::gcd_wide_word_vartime(
                    self.a.to_wide_word_unchecked(),
                    self.b.to_wide_word_unchecked(),
                );
                self.a.set_from_limb(Limb::ZERO);
                self.a_bits = 0;
                self.b.set_from_wide_word(gcd);
                break;
            }

            self.reduce_bingcd(super::bingcd::GCD_BATCH_SIZE);
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

            let (_, _, j) = self.reduce_bingcd(super::bingcd::GCD_BATCH_SIZE);
            jacobi_neg ^= j;
        }

        if self.b.is_one().to_bool_vartime() {
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

            let (m, mk, _) = self.reduce_bingcd(super::bingcd::GCD_BATCH_SIZE);
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
    /// [`bingcd::partial_xgcd_vartime`](super::bingcd::partial_xgcd_vartime) -- the same
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
        let (a_, b_) = ((self.a.limbs[0].0, a_top.0), (self.b.limbs[0].0, b_top.0));

        let (matrix, mut jacobi_neg) =
            super::bingcd::partial_xgcd_vartime(a_, b_, max_batch, top >> Limb::LOG2_BITS == 0);
        let mk = matrix.k;

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

        (m2, mk, jacobi_neg)
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
            && self.a.limbs[self.len - 1].is_zero_vartime()
            && self.b.limbs[self.len - 1].is_zero_vartime()
        {
            self.len -= 1;
        }
    }
}

/// Computes `gcd(a, b)`, leaving it in whichever of `a`/`b` the returned `bool` names (`true` for
/// `a`, `false` for `b`) -- the other buffer's contents are unspecified. `a`/`b` must be the same
/// width; either may be zero (including both at once, giving `gcd(0, 0) = 0`).
pub const fn gcd_vartime(a: &mut UintRef, b: &mut UintRef) -> bool {
    assert!(a.nlimbs() == b.nlimbs(), "inputs must be the same size");

    // GCD is `b` if `a` is zero, `a` if `b` is zero
    let (az, bz) = (a.is_zero_vartime(), b.is_zero_vartime());
    if az || bz {
        return bz;
    }

    let (atz, btz) = (a.trailing_zeros_vartime(), b.trailing_zeros_vartime());
    // GCD shift is the minimum of the trailing zeros (common factors of two)
    let k = if atz < btz { atz } else { btz };
    // Ensure `b` is odd
    b.unbounded_shr_assign_vartime(btz);

    // Compute GCD in place
    let mut pair = GcdPairVartime::new(a, b);
    pair.gcd_odd();

    // Apply common factors of two
    if k != 0 {
        b.unbounded_shl_assign_vartime(k);
    }
    false
}

/// Computes the Jacobi symbol `(a/b)` for odd `b`. Vartime analogue of
/// [`jacobi_symbol`](super::jacobi_symbol) (the constant-time version), built on
/// [`GcdPairVartime::jacobi_symbol`] instead.
pub const fn jacobi_symbol_vartime(a: &mut UintRef, b: &mut UintRef) -> JacobiSymbol {
    assert!(b.is_odd().to_bool_vartime(), "denominator must be odd");
    assert!(a.nlimbs() == b.nlimbs(), "inputs must be the same size");

    let mut pair = GcdPairVartime::new(a, b);
    pair.jacobi_symbol()
}

/// Computes the modular inverse of `x` modulo odd `y`, returning whether one exists (`gcd(x, y) ==
/// 1` and `x != 0`) -- vartime analogue of `invert_odd_mod`, built on
/// [`GcdPairVartime::raw_xgcd`] instead of safegcd divsteps.
///
/// `x` must be no narrower than `y`; `y_inv` is `y`'s mod-limb inverse (`Odd<UintRef>::
/// invert_mod_limb`). `buf` must be at least `3 * x.nlimbs()` limbs (`y`'s copy, plus the `u`/`v`
/// coefficient pair, each `x.nlimbs()`-wide).
///
/// On success (`true`), `x` is overwritten with `x⁻¹ mod y`; on failure (`false`, including `x ==
/// 0`), `x` is left unspecified.
pub const fn invert_odd_mod_vartime<'a>(
    x: &'a mut UintRef,
    y: &'a Odd<UintRef>,
    y_inv: Limb,
    buf: &'a mut UintRef,
) -> bool {
    assert!(x.nlimbs() >= y.as_ref().nlimbs());
    let a = &mut *x;
    if a.is_zero_vartime() {
        // No reciprocal for zero
        return false;
    }

    let limbs = a.nlimbs();
    let (b, buf) = buf.split_at_mut(limbs);
    b.copy_from(y.as_ref());
    let (u, buf) = buf.split_at_mut(limbs);
    let v = buf.leading_mut(limbs);

    let (mut pair, mut cofactors) = (
        GcdPairVartime::new(a, b),
        CofactorPair::new(u, v, y, y_inv, None),
    );
    pair.raw_xgcd(&mut cofactors);
    if !b.is_one().to_bool_vartime() {
        return false;
    }

    let inv = cofactors.finalize_vartime();
    a.copy_from(inv.leading(limbs));

    true
}

/// Calculate the greatest common divisor of `x` and odd `y`, and the Bezout coefficient `a` such
/// that `a*x - b*y = gcd` for some (unreturned) `b`. Vartime analogue of
/// [`xgcd_odd`](super::xgcd_odd), built on [`GcdPairVartime::raw_xgcd`] (the
/// vartime binary-GCD building block) instead of safegcd divsteps.
///
/// `x` must be at least as wide as `y`. `buf` must be at least `3 * x.nlimbs()` limbs (`x`'s
/// working copy, plus the `u`/`v` coefficient pair, each `max(y.nlimbs(), x.nlimbs())`-wide).
pub const fn xgcd_vartime(
    x: &mut UintRef,
    y: &mut UintRef,
    gcd: &mut UintRef,
    a: &mut UintRef,
    b: &mut UintRef,
    buf: &mut UintRef,
) {
    assert!(y.is_odd().to_bool_vartime(), "y must be odd");
    let y_limbs = y.nlimbs();
    let y_odd = Odd::new_ref_unchecked(y);
    let y_inv = y_odd.invert_mod_limb();

    let limbs = x.nlimbs();
    // `u`/`v` (`CofactorPair`'s tracked coefficient pair) need to be at least `y_limbs`
    // limbs wide, since that's what the final, fully-mod-`y`-reduced result needs, and at least
    // `limbs` wide so `x_copy`/`u_scratch`/`v_scratch` can share this one flat `buf` at a uniform
    // stride below. `max` covers both requirements at once (and, as a side effect, guards the case
    // where `x` is wider than `y`).
    let wide_len = if y_limbs > limbs { y_limbs } else { limbs };
    let (x_copy, rest) = buf.split_at_mut(limbs);
    x_copy.copy_from(x);
    // Dedicated scratch for `u`/`v`, kept entirely separate from `a`/`b`: `CofactorPair`
    // is `finalize_vartime`d into a `&mut UintRef` borrowed from `v_scratch`'s own storage, and
    // reading that result into `a` needs `a`/`b` to still be independent, unaliased buffers at
    // that point.
    let (u, rest) = rest.split_at_mut(wide_len);
    let v = rest.leading_mut(wide_len);

    // Half xgcd (vartime): compute gcd(x_copy, y) and `a` such that `a*x ≡ gcd (mod y)`,
    // mirroring `raw_xgcd_odd`'s contract but built on `GcdPairVartime::raw_xgcd`. `gcd`'s
    // buffer holds a copy of `y` going in (matching `raw_xgcd_odd`'s `f`).
    gcd.copy_from(y_odd.as_ref());

    // `GcdPairVartime::raw_xgcd`'s binary-GCD elementary steps assume both operands are
    // genuinely nonzero going in (unlike `raw_xgcd_odd`'s fixed-iteration safegcd divsteps,
    // which handle `x = 0` for free since multiplying an unchanging `0` by any matrix stays
    // `0`) -- `x_copy = 0` needs handling explicitly: `gcd(0, y) = y` (already sitting in `gcd`
    // from the copy above) and the coefficient of `x` is `0` (already `a`'s initial value).
    if x.is_zero_vartime() {
        a.fill(Limb::ZERO);
    } else {
        let mut pair = GcdPairVartime::new(x_copy, gcd);
        let mut cofactors = CofactorPair::new(u, v, y_odd, y_inv, None);
        pair.raw_xgcd(&mut cofactors);

        let v_final = cofactors.finalize_vartime();
        // `finalize_vartime` reduces into `[0, y)`, which (given `y_limbs <= limbs`) fits in `limbs`
        // limbs with room to spare -- `v_final` itself is `wide_len >= limbs` limbs wide, so
        // anything beyond the first `limbs` of it must be zero.
        debug_assert!(v_final.trailing(limbs).is_zero().to_bool_vartime());
        a.copy_from(v_final.leading(limbs));
    };

    // Post-processing (mirrors `xgcd_odd`'s tail exactly, using vartime primitives): derive the
    // second Bezout coefficient `b` algebraically from `a` via `a*(x/gcd) - b*(y/gcd) = 1`.
    x.div_exact_vartime(gcd);
    y.div_exact_vartime(gcd);

    // Reduce `a` modulo `y/gcd`.
    b.copy_from(y);
    a.div_rem_vartime(b);
    a.copy_from(b);
    // Set `a` to `y/gcd` if it became zero: this ensures a positive value for `b`.
    if a.is_zero_vartime() {
        a.copy_from(y);
    }

    // Compute b: ax - by = gcd => b = (a(x/gcd) - 1) / (y/gcd)
    let bp = buf.leading_mut(limbs * 2);
    bp.fill(Limb::MAX); // set to -1
    a.wrapping_mul_add(x, bp);
    let exact = bp.div_exact_vartime(y);
    b.copy_from(bp.leading(limbs));
    debug_assert!(exact.to_bool_vartime());
}
