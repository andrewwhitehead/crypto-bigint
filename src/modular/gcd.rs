mod bingcd;
mod cofactor_pair;
mod extended_int_ref;
mod matrix;
mod vartime;

use crate::{
    Choice, JacobiSymbol, Limb, Odd, Uint, UintRef, WideWord, Word, primitives::u32_min, word,
};

pub(super) use self::{
    cofactor_pair::CofactorPair,
    extended_int_ref::ExtendedIntRef,
    matrix::{SignedLimb, SignedLimbMatrix},
};

pub(crate) use self::vartime::{
    gcd_vartime, invert_odd_mod_vartime, jacobi_symbol_vartime, xgcd_vartime,
};

/// Define the threshold at which we switch to the reduction optimized for
/// smaller integers. This must be ≥4 to avoid the extraction index falling
/// behind the tracking window.
pub const SMALL_THRESHOLD_LIMBS: usize = 8;

/// Compute the modular inverse of `x` modulo odd `y`, plus a `Choice` reporting whether that
/// inverse exists (`gcd(x, y) == 1` and `x != 0`). Thin wrapper around
/// [`raw_xgcd_odd`] that adds the "does the inverse exist" check on top of its
/// `v * x ≡ gcd(x, y) (mod y)` result -- when `gcd == 1`, that congruence is exactly `v = x^-1
/// mod y`.
///
/// Inputs:
/// - `x`: value to invert, same width as `y`; consumed as scratch and left in an unspecified state.
/// - `y`: odd modulus.
/// - `y_inv`: `y`'s limb-sized modular inverse, as returned by `Odd<UintRef>::invert_mod_limb`.
/// - `buf`: scratch, at least `3 * x.nlimbs()` limbs.
/// - `monty_form_r2`: seeds [`CofactorPair`]'s `u` coefficient with a Montgomery `R^2` instead of
///   the plain-inverse default of `1` -- see `invert_mod_precomputed`.
///
/// Outputs:
/// - Returns a `Choice` that is true iff `x` is invertible mod `y` (`gcd(x, y) == 1 && x != 0`).
/// - `x` holds `x^-1 mod y` when the returned `Choice` is true, otherwise its state is unspecified.
#[inline(always)]
pub const fn invert_odd_mod<'a>(
    x: &'a mut UintRef,
    y: &'a Odd<UintRef>,
    y_inv: Limb,
    buf: &'a mut UintRef,
    monty_form_r2: Option<&UintRef>,
) -> Choice {
    let x_nonzero = x.is_nonzero();
    let limbs = x.nlimbs();
    let (gcd, buf) = buf.split_at_mut(limbs);
    let (u, buf) = buf.split_at_mut(limbs);
    let v = buf.leading_mut(limbs);
    let mut cofactors = CofactorPair::new(u, v, y, y_inv, monty_form_r2);
    raw_xgcd_odd(x, gcd, &mut cofactors);
    x.copy_from(cofactors.finalize());
    gcd.is_one().and(x_nonzero)
}

/// Compute `gcd(x, y)` together with non-negative Bezout coefficients `a`, `b` satisfying
/// `a*x - b*y = gcd(x, y)`, for odd `y`. Builds on [`raw_xgcd_odd`], adding the
/// post-processing needed to turn its single reduced-mod-`y` coefficient into a full two-sided
/// Bezout identity; this is the shared engine behind
/// `Uint::xgcd_unsigned`/`BoxedUint::xgcd_unsigned`.
///
/// Inputs:
/// - `x`, `y`: same width (`nlimbs`); `y` must be odd (asserted).
/// - `gcd`: output buffer, same width; initial contents ignored.
/// - `a`, `b`: output buffers, same width; initial contents ignored (unlike
///   [`raw_xgcd_odd`]'s own `(u, v)`, which callers seed themselves via
///   [`CofactorPair::new`], this function seeds and discards its own internal pair -- `a`/`b`
///   here are pure outputs).
/// - `buf`: scratch, at least `3 * x.nlimbs()` limbs.
///
/// Outputs:
/// - `gcd` receives `gcd(x, y)`.
/// - `x`, `y` are divided in place by `gcd` (left holding `x/gcd`, `y/gcd`).
/// - `a`, `b` are left non-negative and satisfy `a*x_orig - b*y_orig = gcd`, with `a` reduced
///   modulo `y/gcd` (and bumped up to `y/gcd` itself if that reduction would otherwise leave it
///   `0`, so `a` stays nonzero).
pub const fn xgcd_odd(
    x: &mut UintRef,
    y: &mut UintRef,
    gcd: &mut UintRef,
    a: &mut UintRef,
    b: &mut UintRef,
    buf: &mut UintRef,
) {
    assert!(y.is_odd().to_bool_vartime(), "y must be odd");
    let y_odd = Odd::new_ref_unchecked(y);
    let y_inv = y_odd.invert_mod_limb();

    let limbs = x.nlimbs();
    let (x_copy, rest) = buf.split_at_mut(limbs);
    x_copy.copy_from(x);
    let (u, rest) = rest.split_at_mut(limbs);
    let (v, _) = rest.split_at_mut(limbs);

    let mut cofactors = CofactorPair::new(u, v, y_odd, y_inv, None);
    raw_xgcd_odd(x_copy, gcd, &mut cofactors);
    a.copy_from(cofactors.finalize());

    // Replace `x`, `y` with `x/gcd`, `y/gcd`
    x.div_exact(gcd);
    y.div_exact(gcd);

    // Reduce `a` modulo `y/gcd`
    b.copy_from(y);
    a.div_rem(b);
    a.copy_from(b);
    // Set a to `y/gcd` if it became zero: this ensures a positive value for `b`
    a.conditional_copy_from(y, a.is_zero());

    // Compute b: `ax - by = gcd ==> b = (a(x/gcd) - 1) / (y/gcd)`
    let bp = buf.leading_mut(limbs * 2);
    bp.fill(Limb::MAX); // set to -1
    a.wrapping_mul_add(x, bp);
    let exact = bp.div_exact(y);
    b.copy_from(bp.leading(limbs));
    debug_assert!(exact.to_bool_vartime());
}

/// Core signed extended-gcd engine shared by [`xgcd_odd`] and [`invert_odd_mod`]: reduces `f` (a
/// copy of `cofactors.y`) and `g` (`x`) down to their gcd using
/// [`gcd_odd_with_budget`]'s binary-GCD method (magnitude-comparison-based batched
/// matrices, deferred sign), applying the identical sequence of step matrices to a
/// caller-supplied [`CofactorPair`] alongside `(f, g)`.
///
/// Inputs:
/// - `x` doubles as scratch (`g` in the steps below) and is left in an unspecified state.
/// - `cofactors.y` must be odd and the same width (`nlimbs`) as `x`, `cofactors.u`,
///   `cofactors.v`, and `gcd`.
///
/// Outputs:
/// - `gcd` receives `gcd(x, cofactors.y)`.
/// - `cofactors.v` ends up satisfying `v * x ≡ gcd(x, y) (mod y)`; call
///   [`CofactorPair::finalize`] to reduce it into `[0, y)`.
///
/// # Approach
///
/// Two stages, matching `gcd_odd_with_budget`'s own split:
///
/// Stage 1 (`window_limbs > SMALL_THRESHOLD_LIMBS`) is the deferred-sign loop. Each batch extracts
/// a top-bit-aligned magnitude window of `f`/`g` via
/// [`extract_pair_vartime_signed`]/[`ExtendedIntRef::abs_low_limb`], walking an
/// unconditionally-moving schedule position. It them turns the extraction into a
/// `GCD_BATCH_SIZE`-step [`bingcd::BingcdMatrix`] via [`bingcd::partial_xgcd`], and applies it
/// to `(g, f)`. `(d, e)` are updated by the same column-sign-adjusted matrix (captured as
/// `g_neg`/`f_neg` before that update changes `g`/`f`'s own sign).
///
/// Stage 2 (`window_limbs <= SMALL_THRESHOLD_LIMBS`) switches to the same exact, non-deferred-sign
/// extraction [`gcd_odd_small_with_budget`]'s own Stage 1 uses -- `top_window_pair` plus the
/// unsigned `wrapping_apply_shift_unsigned` -- instead of continuing Stage 1's tracked-position
/// `extract_pair_vartime_signed` down to a 1-limb window. Unlike [`gcd_odd_small_with_budget`]'s own
/// tail, which hands the rest off wholesale to [`gcd_odd_tiny_with_budget`], Stage 2 here runs
/// the same loop shape inline all the way to convergence (`k_remain == 0`) since `(d, e)` need
/// updating every round.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
#[allow(unused_assignments)]
const fn raw_xgcd_odd<'a>(x: &mut UintRef, gcd: &mut UintRef, cofactors: &mut CofactorPair<'a>) {
    let f = &mut *gcd;
    f.copy_from(cofactors.y.as_ref());
    let len = f.nlimbs();

    let g = &mut *x;
    assert!(len == g.nlimbs());
    assert!(cofactors.u.nlimbs() == len && cofactors.v.nlimbs() == len);

    let total_steps = bingcd::iterations(f.bits_precision());
    let mut k_remain = total_steps;
    let mut window_limbs = len;
    let mut window_shrink_at = (window_limbs as u32 - 1) << Limb::LOG2_BITS;
    let mut extract_pos = (f.bits_precision() - bingcd::GCD_BATCH_SIZE) << 1;

    let (mut f_hi, mut g_hi) = (Limb::ZERO, Limb::ZERO);
    let mut f_true_limbs = window_limbs;

    // Stage 1 batches: extract, build the transition matrix, apply it to `(g, f)` and
    // `(d, e)`, and update the schedule.
    while window_limbs > SMALL_THRESHOLD_LIMBS {
        let (mut g_ext, mut f_ext) = (
            ExtendedIntRef::new(g.leading_mut(window_limbs), g_hi),
            ExtendedIntRef::new(f.leading_mut(window_limbs), f_hi),
        );

        let (g_hi_mag, f_hi_mag, _shift) =
            extract_pair_vartime_signed(&g_ext, &f_ext, extract_pos >> 1);
        let (g_lo_mag, f_lo_mag) = (g_ext.abs_low_limb(), f_ext.abs_low_limb());
        let (g_neg, f_neg) = (g_ext.is_negative(), f_ext.is_negative());

        let (matrix, _jacobi_neg, _active) = bingcd::partial_xgcd::<false>(
            (g_lo_mag, g_hi_mag),
            (f_lo_mag, f_hi_mag),
            Choice::FALSE,
            bingcd::GCD_BATCH_SIZE,
        );
        matrix.wrapping_apply_sign_correcting_shift(&mut g_ext, &mut f_ext);
        (g_hi, f_hi) = (g_ext.hi, f_ext.hi);

        cofactors.apply_matrix(matrix.column_signed_limb_matrix(g_neg, f_neg), matrix.k);

        k_remain -= matrix.k;
        if k_remain <= window_shrink_at {
            window_limbs -= 1;
            window_shrink_at -= Limb::BITS;
        }
        extract_pos -= bingcd::GCD_BATCH_SIZE;

        let g_nonzero = g.limbs[0].is_nonzero();
        f_true_limbs = g_nonzero.select_u32(f_true_limbs as u32, window_limbs as u32) as usize;
    }

    // Stage 1 -> Stage 2 transition: normalize `(g, f)`. Stage 2 needs both operands non-negative
    // to safely use the exact, non-deferred-sign `top_window_pair` extraction. Skipped when
    // Stage 1 did not run at all as the terms are guaranteed non-negative.
    if len > SMALL_THRESHOLD_LIMBS {
        let g_ext = ExtendedIntRef::new(g.leading_mut(window_limbs), g_hi);
        cofactors.negate_u_if(g_ext.is_negative());
        g_ext.abs_drop_extension();

        let f_sign = f_hi.bit(Limb::HI_BIT);
        cofactors.negate_v_if(f_sign);

        let f_true_limbs_u32 = f_true_limbs as u32;
        let f_mask = Limb::choice_to_mask(f_sign);
        let mut carry = f_mask.wrapping_neg();
        let mut i = 0;
        while i < window_limbs {
            (f.limbs[i], carry) = f.limbs[i].bitxor(f_mask).overflowing_add(carry);
            i += 1;
        }
        while i < len {
            (f.limbs[i], carry) = f.limbs[i].bitxor(f_mask).overflowing_add(carry);
            let keep = Choice::from_u32_lt(i as u32, f_true_limbs_u32);
            f.limbs[i] = Limb::select(Limb::ZERO, f.limbs[i], keep);
            i += 1;
        }
    }

    // Stage 2 (`window_limbs <= SMALL_THRESHOLD_LIMBS`): matching `gcd_odd_small_with_budget`'s own Stage 1.
    // `(g, f)` stay non-negative throughout (every round's `wrapping_apply_shift_unsigned` unconditionally
    // re-corrects both) but `(d, e)`'s own update still needs the *row*-sign adjustment
    // `wrapping_apply_shift_unsigned` itself reports back (`g_negated`, `f_negated`).
    while k_remain != 0 {
        let batch_size = if k_remain > bingcd::GCD_BATCH_SIZE {
            bingcd::GCD_BATCH_SIZE
        } else {
            k_remain
        };
        let (g_window, f_window) = (g.leading_mut(window_limbs), f.leading_mut(window_limbs));
        let (g_hi_word, f_hi_word, _exact) = top_window_pair(g_window, f_window);

        let (matrix, _jacobi_neg, _active) = bingcd::partial_xgcd::<false>(
            (g_window.limbs[0].0, g_hi_word),
            (f_window.limbs[0].0, f_hi_word),
            Choice::FALSE,
            batch_size,
        );
        let (g_negated, f_negated) = matrix.wrapping_apply_unsigned_shift(g_window, f_window);
        cofactors.apply_matrix(
            matrix.row_signed_limb_matrix(g_negated, f_negated),
            matrix.k,
        );

        k_remain -= matrix.k;
        if window_limbs != 1 && k_remain <= window_shrink_at {
            window_limbs -= 1;
            window_shrink_at -= Limb::BITS;
        }
    }
}

/// Calculates the greatest common denominator of odd `f` and `g`, using the optimized
/// (batched) Binary GCD algorithm. Thin wrapper around [`gcd_odd_with_budget`], always
/// deriving its step budget from `f`'s own width.
pub const fn gcd_odd(f: &mut UintRef, g: &mut UintRef) {
    gcd_odd_with_budget(f, g, bingcd::iterations(f.bits_precision()));
}

/// Reduces `f`/`g` down to their gcd using a deferred-sign reduction loop with a scheduled
/// high word extraction before handing off to [`gcd_odd_small_with_budget`] once the tracked
/// window narrows to `SMALL_THRESHOLD_LIMBS`.
///
/// Inputs:
/// - `f`/`g` must be the same width (`nlimbs`) and `f` must be odd.
/// - `total_steps` is the step budget: the caller must have `bitlen(f), bitlen(g) <= k_remain`
///   (Pornin's Phi bound).
///
/// Outputs:
/// - `f` receives `gcd(f, g)` and `g` is left zeroed.
///
/// # Approach
///
/// Each round reads a top-bit-aligned magnitude window of `f`/`g` via
/// [`extract_pair_vartime_signed`]/[`ExtendedIntRef::abs_low_limb`] (deferred
/// sign -- `f`/`g` are tracked as two's-complement values across rounds for efficiency),
/// turns it into a `GCD_BATCH_SIZE`-step matrix via [`bingcd::partial_xgcd`], and applies
/// it to the inputs. Once the window size drops to the small size threshold, the inputs
/// are renormalized to non-negative, then [`gcd_odd_small_with_budget`] takes over with
/// using an exact (non-scheduled) extraction.
///
/// The extraction position walks a schedule that starts one batch's worth of bits below full
/// width and falls by half of whatever's been consumed so far every round unconditionally.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub const fn gcd_odd_with_budget(f: &mut UintRef, g: &mut UintRef, total_steps: u32) {
    let len = f.nlimbs();
    assert!(len == g.nlimbs());
    assert!(f.is_odd().to_bool_vartime(), "f must be odd");

    let mut k_remain = total_steps;
    let mut window_limbs = len;
    let mut window_shrink_at = (window_limbs as u32 - 1) << Limb::LOG2_BITS;
    let mut extract_pos = (f.bits_precision() - bingcd::GCD_BATCH_SIZE) << 1;

    let (mut f_hi, mut g_hi) = (Limb::ZERO, Limb::ZERO);
    let mut f_true_limbs = window_limbs;

    // Stage 1 batches: extract according the fixed schedule, build the transition matrix, apply it to
    // `(g, f)` and update the schedule.
    while window_limbs > SMALL_THRESHOLD_LIMBS {
        let (mut g_ext, mut f_ext) = (
            ExtendedIntRef::new(g.leading_mut(window_limbs), g_hi),
            ExtendedIntRef::new(f.leading_mut(window_limbs), f_hi),
        );

        let (g_hi_mag, f_hi_mag, _shift) =
            extract_pair_vartime_signed(&g_ext, &f_ext, extract_pos >> 1);
        let (g_lo_mag, f_lo_mag) = (g_ext.abs_low_limb(), f_ext.abs_low_limb());

        let (matrix, _jacobi_neg, _active) = bingcd::partial_xgcd::<false>(
            (g_lo_mag, g_hi_mag),
            (f_lo_mag, f_hi_mag),
            Choice::FALSE,
            bingcd::GCD_BATCH_SIZE,
        );

        matrix.wrapping_apply_sign_correcting_shift(&mut g_ext, &mut f_ext);
        (g_hi, f_hi) = (g_ext.hi, f_ext.hi);

        k_remain -= matrix.k;
        if k_remain <= window_shrink_at {
            window_limbs -= 1;
            window_shrink_at -= Limb::BITS;
        }
        extract_pos -= bingcd::GCD_BATCH_SIZE;

        let g_nonzero = g.limbs[0].is_nonzero();
        f_true_limbs = g_nonzero.select_u32(f_true_limbs as u32, window_limbs as u32) as usize;
    }

    // Restore `f`/`g` to non-negative before handing off: `g` is dropped back to
    // its absolute value directly (its true value always fits within `window_limbs`);
    // `f` is sign-extended from `f_true_limbs` and negated in place if needed.
    if len > SMALL_THRESHOLD_LIMBS {
        ExtendedIntRef::new(g.leading_mut(window_limbs), g_hi).abs_drop_extension();

        let f_sign = f_hi.bit(Limb::HI_BIT);
        let f_true_limbs = f_true_limbs as u32;
        let f_mask = Limb::choice_to_mask(f_sign);
        let mut carry = f_mask.wrapping_neg();
        let mut i = 0;
        while i < window_limbs {
            (f.limbs[i], carry) = f.limbs[i].bitxor(f_mask).overflowing_add(carry);
            i += 1;
        }
        while i < len {
            (f.limbs[i], carry) = f.limbs[i].bitxor(f_mask).overflowing_add(carry);
            let keep = Choice::from_u32_lt(i as u32, f_true_limbs);
            f.limbs[i] = Limb::select(Limb::ZERO, f.limbs[i], keep);
            i += 1;
        }
    }

    // Pass off to perform the final reduction for `k_remain` steps, updating `f` and `g` in place
    gcd_odd_small_with_budget(
        f.leading_mut(window_limbs),
        g.leading_mut(window_limbs),
        k_remain,
    );
}

/// Calculates the greatest common denominator of odd `f` and `g`, using the optimized
/// (batched) Binary GCD algorithm. Thin wrapper around [`gcd_odd_small_with_budget`], always
/// deriving its step budget from `f`'s own width.
#[inline(always)]
pub const fn gcd_odd_small(f: &mut UintRef, g: &mut UintRef) {
    gcd_odd_small_with_budget(f, g, bingcd::iterations(f.bits_precision()));
}

/// Reduces `f`/`g` down to their gcd using the exact, non-deferred-sign counterpart of
/// [`gcd_odd_with_budget`]'s reduction loop, then hands off to [`gcd_odd_tiny_with_budget`]
/// once the tracked window narrows to 2 limbs.
///
/// Inputs:
/// - `f`/`g` must be the same width (`nlimbs`) and `f` must be odd.
/// - `total_steps` is the step budget: the caller must have `bitlen(f), bitlen(g) <= k_remain`
///   (Pornin's Phi bound), the same requirement [`gcd_odd_with_budget`]'s own `total_steps`
///   documents.
///
/// Outputs:
/// - `f` receives `gcd(f, g)` and `g` is left zeroed.
///
/// The batched matrix reduction operates on `f`/`g` a `GCD_BATCH_SIZE`-step matrix at a
/// time, narrowing `window_limbs` via the same `window_shrink_at` formula [`gcd_odd_with_budget`]
/// uses for its own reduction loop, until it reaches 2 limbs. The reduction of the final limbs
/// is delegated to [`gcd_odd_tiny_with_budget`].
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub const fn gcd_odd_small_with_budget(f: &mut UintRef, g: &mut UintRef, total_steps: u32) {
    let len = f.nlimbs();
    debug_assert!(g.nlimbs() == len, "f/g must be the same width");

    // Stage 1: batched matrix reduction down to a 2-limb window.
    let mut k_remain = total_steps;
    let mut window_limbs = len;
    let mut window_shrink_at = (window_limbs as u32 - 1) << Limb::LOG2_BITS;

    while window_limbs > 2 {
        let (g_window, f_window) = (g.leading_mut(window_limbs), f.leading_mut(window_limbs));
        let (g_hi, f_hi, _exact) = top_window_pair(g_window, f_window);
        let (matrix, _jneg, _active) = bingcd::partial_xgcd::<false>(
            (g_window.limbs[0].0, g_hi),
            (f_window.limbs[0].0, f_hi),
            Choice::TRUE,
            bingcd::GCD_BATCH_SIZE,
        );
        matrix.wrapping_apply_unsigned_shift(g_window, f_window);
        k_remain -= matrix.k;
        if k_remain <= window_shrink_at {
            window_limbs -= 1;
            window_shrink_at -= Limb::BITS;
        }
    }

    // Pass off to perform the final reduction for `k_remain` steps, updating `f` and `g` in place
    gcd_odd_tiny_with_budget(
        f.leading_mut(window_limbs),
        g.leading_mut(window_limbs),
        k_remain,
    );
}

/// Finishes an in-progress `gcd_odd_small`/`jacobi_symbol` run in one register-only call,
/// once the schedule-derived operand ceiling (`window_limbs`) has fallen to at most two
/// limbs.
///
/// Writes the gcd back into `f`'s low limb(s) and zeros `g`'s; returns the accumulated
/// quadratic-reciprocity sign for the finished portion (`gcd_odd_small` ignores it,
/// `jacobi_symbol` XORs it into the running `jacobi_neg`).
#[inline(always)]
const fn gcd_odd_tiny_with_budget(f: &mut UintRef, g: &mut UintRef, total_steps: u32) -> Word {
    let len = f.nlimbs();
    assert!(len == g.nlimbs() && len <= 2);
    let mut jacobi_neg = 0;
    let mut k_remain = total_steps;

    // Perform `WideWord` elementary steps until `remaining_steps` proves a single `Word` suffices.
    if len == 2 {
        let (g2, f2) = (g.leading_mut(2), f.leading_mut(2));
        let mut a = g2.to_wide_word_unchecked();
        let mut b = f2.to_wide_word_unchecked();
        while k_remain > Limb::BITS {
            let j_neg;
            ((a, b), _, _, j_neg) = bingcd::step_wide_word(a, b);
            jacobi_neg ^= j_neg;
            k_remain -= 1;
        }
        g2.set_from_wide_word(a);
        f2.set_from_wide_word(b);
    }

    // Single-`Word` finish for the remaining reduction steps.
    let (mut a, mut b) = (g.limbs[0].0, f.limbs[0].0);
    while k_remain != 0 {
        let j_neg;
        ((a, b), _, _, j_neg) = bingcd::step_word(a, b);
        jacobi_neg ^= j_neg;
        k_remain -= 1;
    }
    f.limbs[0] = Limb(b);
    g.limbs[0] = Limb::ZERO;
    jacobi_neg
}

#[allow(clippy::cast_possible_truncation)]
pub const fn jacobi_symbol(a: &mut UintRef, b: &mut UintRef) -> JacobiSymbol {
    assert!(b.is_odd().to_bool_vartime(), "denominator must be odd");
    assert!(a.nlimbs() == b.nlimbs(), "inputs must be the same size");

    let mut jacobi_neg = 0;
    let mut k_remain = bingcd::iterations(a.bits_precision());
    let mut window_limbs = a.nlimbs();
    let mut window_shrink_at = (window_limbs as u32 - 1) << Limb::LOG2_BITS;

    // Seed the first round's top-bit-aligned window directly
    let (a_hi, b_hi, mut exact) = top_window_pair(a.leading(window_limbs), b.leading(window_limbs));
    let (mut a_, mut b_) = ((a.limbs[0].0, a_hi), (b.limbs[0].0, b_hi));

    while window_limbs > 2 {
        let (matrix, j, active) =
            bingcd::partial_xgcd::<true>(a_, b_, exact, bingcd::GCD_BATCH_SIZE);
        jacobi_neg ^= j;

        let (mut a_ext, mut b_ext) = (
            ExtendedIntRef::new(a.leading_mut(window_limbs), Limb::ZERO),
            ExtendedIntRef::new(b.leading_mut(window_limbs), Limb::ZERO),
        );
        matrix.wrapping_apply_shift(&mut a_ext, &mut b_ext);
        let a_negated = a_ext.is_negative();

        // If `a` is negated, then that possibility should be detected during the batch update
        debug_assert!(
            !a_negated.to_bool_vartime() || !active.to_bool_vartime(),
            "a went negative without detection"
        );
        // `b` would only become negative after a previous undetected divergence
        debug_assert!(
            !b_ext.is_negative().to_bool_vartime(),
            "b must never come out negative"
        );

        // Correct `a` and `b` `a` came out negative from the transition (this essentially
        // reverses the last swap decision), fused with extracting the next round's seed.
        let b_lo_pre = b.limbs[0].0;
        (a_, b_, exact) = jacobi_correct_and_extract(a, a_negated, b);
        let b_lo_post = b_.0;

        let eps = (b_lo_pre & b_lo_post) >> 1;
        let jj = word::select(
            (b_lo_post >> 1) ^ (b_lo_post >> 2) ^ word::select(0, eps, a_negated),
            0,
            active,
        ) & 1;
        jacobi_neg ^= jj;

        k_remain -= matrix.k;
        if k_remain <= window_shrink_at {
            window_limbs -= 1;
            window_shrink_at -= Limb::BITS;
        }
    }

    // Reduce the remaining 1-2 word window and update the Jacobi symbol negation flag.
    jacobi_neg ^= gcd_odd_tiny_with_budget(
        b.leading_mut(window_limbs),
        a.leading_mut(window_limbs),
        k_remain,
    );

    JacobiSymbol::from_sign(jacobi_neg).zero_if(b.is_one().not())
}

/// Fused replacement for `jacobi_symbol`'s per-round `a = |a|, b -= 2|a|` swap-correction
/// that, in the same full-width pass over `a`/`b`, also latches the top-bit-aligned `(lo, hi)`
/// word pair each of them will need as input for the *next* round.
///
/// Negates `a` in-place and sets `b -= (a << 1)` when `a_neg` is truthy, leaving `a` and `b`
/// unchanged otherwise.
///
/// Returns `((a_lo, a_hi), (b_lo, b_hi), exact)`, ready to hand straight to
/// `partial_xgcd` for the next round.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn jacobi_correct_and_extract(
    a: &mut UintRef,
    a_neg: Choice,
    b: &mut UintRef,
) -> ((Word, Word), (Word, Word), Choice) {
    let len = a.nlimbs();
    debug_assert!(len == b.nlimbs());

    let mask = Limb::choice_to_mask(a_neg);

    // `(a, b)` pairs bundled into a single `WideWord`
    let mut lo_pair: WideWord = 0;
    let (mut top_lo_pair, mut top_hi_pair) = (0, 0);
    let mut top_index: u32 = 0;

    let mut neg_carry = mask.shr(Limb::HI_BIT);
    let mut shl_carry = Limb::ZERO;
    let mut borrow = Limb::ZERO;

    let mut i = 0;
    while i < len {
        let (new_a_i, new_b_i);

        // Conditionally negate a
        (new_a_i, neg_carry) = a.limbs[i].bitxor(mask).overflowing_add(neg_carry);
        a.limbs[i] = new_a_i;

        // `2*a`, computed via shift-with-carry instead of a multiply, masked to zero unless
        // `a_neg` is set.
        let doubled = a.limbs[i].shl(1).bitor(shl_carry).bitand(mask);
        shl_carry = a.limbs[i].shr(Limb::HI_BIT);

        (new_b_i, borrow) = b.limbs[i].borrowing_sub(doubled, borrow);
        b.limbs[i] = new_b_i;

        let hi_pair = word::join(new_a_i.0, new_b_i.0);
        let nz = new_a_i.bitor(new_b_i).is_nonzero();
        top_lo_pair = word::select_wide(top_lo_pair, lo_pair, nz);
        top_hi_pair = word::select_wide(top_hi_pair, hi_pair, nz);
        top_index = nz.select_u32(top_index, i as u32);

        lo_pair = hi_pair;
        i += 1;
    }

    // The true (unbounded) result must fit back within `len` limbs: neither the subtraction's
    // own borrow chain nor a masked carry-out from doubling `a`'s own top bit may survive past
    // the top real limb.
    if cfg!(debug_assertions) {
        let ovf = shl_carry.bitand(mask).is_nonzero();
        assert!(
            borrow.is_zero().and(ovf.not()).to_bool_vartime(),
            "overflow"
        );
    }

    let (a_hi, b_hi, shift) =
        top_window_words(word::split_wide(top_lo_pair), word::split_wide(top_hi_pair));
    let exact = Choice::from_u32_nz(top_index | shift).not();
    ((a.limbs[0].0, a_hi), (b.limbs[0].0, b_hi), exact)
}

/// Perform a single low-to-high scan latching the `(prev, cur)` limb pair of `a`, `b`
/// and extracting a single aligned top word for each input.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn top_window_pair(a: &UintRef, b: &UintRef) -> (Word, Word, Choice) {
    let len = a.nlimbs();
    debug_assert!(len == b.nlimbs());

    let mut lo_pair: WideWord = 0;
    let (mut top_lo_pair, mut top_hi_pair) = (0, 0);
    let mut top_index = 0;

    let mut i = 0;
    while i < len {
        let (a_i, b_i) = (a.limbs[i], b.limbs[i]);
        let hi_pair = word::join(a_i.0, b_i.0);
        let nz = a_i.bitor(b_i).is_nonzero();
        top_lo_pair = word::select_wide(top_lo_pair, lo_pair, nz);
        top_hi_pair = word::select_wide(top_hi_pair, hi_pair, nz);
        top_index = nz.select_u32(top_index, i as u32);

        lo_pair = hi_pair;
        i += 1;
    }

    let (a_hi, b_hi, shift) =
        top_window_words(word::split_wide(top_lo_pair), word::split_wide(top_hi_pair));
    let exact = Choice::from_u32_nz(top_index | shift).not();
    (a_hi, b_hi, exact)
}

/// Assembles `(a_word, b_word, shift)` out of the latched `(prev, cur)` `Word` pairs
/// (`a` in the low half, `b` in the high half) as produced by a single low-to-high scan
/// over `a`/`b`. Trims leading zeros from the wide word values and returns the high
/// word along with the shift value.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
const fn top_window_words(lo: (Word, Word), hi: (Word, Word)) -> (Word, Word, u32) {
    let (a, b) = (word::join(lo.0, hi.0), word::join(lo.1, hi.1));

    let hi_lz = (hi.0 | hi.1).leading_zeros();
    let shift = Word::BITS - hi_lz;

    ((a >> shift) as Word, (b >> shift) as Word, shift)
}

/// Extract a comparable pair of compact high words from two signed operands.
///
/// `n` is the bit position `E - S`; the window is the three limbs starting at
/// limb containing this position, `[base, base + 3W)` with
/// `base = W * floor(n / W)`. This places the dominance threshold `E + S` inside
/// the window: the window top is at least `E - S + 2W + 1`.
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
#[allow(clippy::cast_possible_truncation)]
pub const fn extract_pair_vartime_signed(
    a: &ExtendedIntRef<'_>,
    b: &ExtendedIntRef<'_>,
    n: u32,
) -> (Word, Word, u32) {
    let extract_limb = (n >> Limb::LOG2_BITS) as usize;

    let (a_trip, a_over) = trip_and_overflow(a.lo.trailing(extract_limb), a.is_negative());
    let (b_trip, b_over) = trip_and_overflow(b.lo.trailing(extract_limb), b.is_negative());

    let lz = u32_min(a_trip.bitor(&b_trip).leading_zeros(), Limb::BITS * 2);

    let (a_hi, b_hi) = (a_trip.shl(lz).limbs[2], b_trip.shl(lz).limbs[2]);
    debug_assert!(
        !a_over.and(b_over).to_bool_vartime(),
        "invalid extraction position"
    );
    let any_over = a_over.or(b_over);
    (
        Limb::select(a_hi, Limb::choice_to_mask(a_over), any_over).0,
        Limb::select(b_hi, Limb::choice_to_mask(b_over), any_over).0,
        any_over.select_u32(Limb::BITS * 2 - lz, Limb::BITS * 2),
    )
}

/// The per-operand half of [`extract_pair_vartime_signed`], called once for each
/// of `a`/`b`: extract a 3-word non-negative window from the start of `x`, as well as a flag
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
