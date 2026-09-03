mod bingcd;
mod cofactor_pair;
mod extended_int_ref;
mod matrix;
mod pair;
mod pair_vartime;

use crate::{Choice, JacobiSymbol, Limb, Odd, UintRef, Word};

pub(super) use self::{
    cofactor_pair::CofactorPair,
    extended_int_ref::ExtendedIntRef,
    matrix::{SignedLimb, SignedLimbMatrix},
    pair::GcdPair,
    pair_vartime::GcdPairVartime,
};

/// Maximum number of elementary binary GCD steps [`partial_xgcd`] performs per batch.
pub(super) const GCD_BATCH_SIZE: u32 = Word::BITS - SPLIT_THRESHOLD_BITS - 1;

/// Ambiguity band for [`partial_xgcd`]'s `HALT = true` divergence check.
const SPLIT_THRESHOLD_BITS: u32 = match cpubits::CPUBITS {
    32 => 4,
    64 => 5,
    _ => unreachable!(),
};

/// Define the threshold at which we switch to the reduction optimized for
/// smaller integers. This must be ≥4 to avoid the extraction index falling
/// behind the tracking window.
pub const SMALL_THRESHOLD_LIMBS: usize = 8;

/// Calculates the greatest common denominator of `a` and odd `b`, using the optimized
/// batched Binary GCD algorithm.
///
/// Thin wrapper around [`GcdPair::gcd_odd_with_budget`], always deriving its step budget
/// from `b`'s own width.
pub const fn gcd_odd(a: &mut UintRef, b: &mut UintRef) {
    let total_steps = bingcd::iterations(b.bits_precision());
    let mut pair = GcdPair::new(a, b);
    pair.gcd_with_budget::<false>(total_steps);
}

/// Calculates the greatest common denominator of `a` and odd `b`, using the optimized
/// batched Binary GCD algorithm.
///
/// Thin wrapper around [`GcdPair::gcd_small_with_budget`], always deriving its step
/// budget from `b`'s own width.
#[inline(always)]
pub const fn gcd_odd_small(a: &mut UintRef, b: &mut UintRef) {
    let total_steps = bingcd::iterations(b.bits_precision());
    let mut pair = GcdPair::new(a, b);
    pair.gcd_small_with_budget::<false>(total_steps);
}

/// Computes `gcd(a, b)`, leaving it in whichever of `a`/`b` the returned `bool` names (`true` for
/// `a`, `false` for `b`) -- the other buffer's contents are unspecified. `a`/`b` must be the same
/// width; either may be zero (including both at once, giving `gcd(0, 0) = 0`).
///
/// # Panics
/// If `a` and `b` are not the same width.
pub const fn gcd_vartime(a: &mut UintRef, b: &mut UintRef) -> bool {
    assert!(a.nlimbs() == b.nlimbs(), "inputs must be the same width");

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

/// Computes the Jacobi symbol `(a|b)` for odd `b`, specialized for small operands.
///
/// Uses [`GcdPair::gcd_small_with_budget`]'s small-width reduction loop directly, skipping
/// the deferred-sign Stage 1 loop [`jacobi_symbol`] uses for wider operands.
///
/// Inputs:
/// - `a`, `b`: same width (`nlimbs`); `b` must be odd.
///
/// Outputs:
/// - `a`, `b` are consumed as scratch and left in an unspecified state.
/// - Returns [`JacobiSymbol::Zero`] if `gcd(a, b) != 1` (detected via `b` failing to reduce to
///   `1`), otherwise the sign accumulated by the underlying reduction's quadratic-reciprocity
///   flips, folded together via [`JacobiSymbol::from_sign`].
///
/// # Panics
/// If `a` and `b` are not the same width or `b` is not odd.
#[inline(always)]
pub const fn jacobi_symbol_small(a: &mut UintRef, b: &mut UintRef) -> JacobiSymbol {
    let total_steps = bingcd::iterations(a.bits_precision());
    let mut pair = GcdPair::new(a, b);
    let jacobi_neg = pair.gcd_small_with_budget::<true>(total_steps);
    JacobiSymbol::from_sign(jacobi_neg & 1).zero_if(b.is_one().not())
}

/// Computes the Jacobi symbol `(a|b)` for odd `b`, using the full deferred-sign
/// [`GcdPair::gcd_with_budget`] reduction loop.
///
/// Inputs:
/// - `a`, `b`: same width (`nlimbs`); `b` must be odd.
///
/// Outputs:
/// - `a`, `b` are consumed as scratch and left in an unspecified state.
/// - Returns [`JacobiSymbol::Zero`] if `gcd(a, b) != 1` (detected via `b` failing to reduce to
///   `1`), otherwise the sign accumulated by the underlying reduction's quadratic-reciprocity
///   flips, folded together via [`JacobiSymbol::from_sign`].
///
/// # Panics
/// If `a` and `b` are not the same width or `b` is not odd.
pub const fn jacobi_symbol(a: &mut UintRef, b: &mut UintRef) -> JacobiSymbol {
    let total_steps = bingcd::iterations(a.bits_precision());
    let mut pair = GcdPair::new(a, b);
    let jacobi_neg = pair.gcd_with_budget::<true>(total_steps);
    JacobiSymbol::from_sign(jacobi_neg & 1).zero_if(b.is_one().not())
}

/// Computes the Jacobi symbol `(a|b)` for odd `b`.
///
/// Variable-time analog of [`jacobi_symbol`].
///
/// # Panics
/// If `a` and `b` are not the same width or `b` is not odd.
pub const fn jacobi_symbol_vartime(a: &mut UintRef, b: &mut UintRef) -> JacobiSymbol {
    assert!(b.is_odd().to_bool_vartime(), "denominator must be odd");
    assert!(a.nlimbs() == b.nlimbs(), "inputs must be the same width");

    let mut pair = GcdPairVartime::new(a, b);
    pair.jacobi_symbol()
}

/// Compute the modular inverse of `x` modulo odd `y`, plus a `Choice` reporting whether that
/// inverse exists (`gcd(x, y) == 1` and `x != 0`)
///
/// This is a thin wrapper around [`GcdPair::raw_xgcd`] that adds the "does the inverse exist" check on
/// top of its `v * x ≡ gcd(x, y) (mod y)` result -- when `gcd == 1`, that congruence is exactly `v = x^-1
/// mod y`.
///
/// Inputs:
/// - `x`: value to invert, same width as `y`.
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
    gcd.copy_from(y.as_ref());
    let mut pair = GcdPair::new(x, gcd);

    let (u, buf) = buf.split_at_mut(limbs);
    let v = buf.leading_mut(limbs);
    let mut cofactors = CofactorPair::new(u, v, y, y_inv, monty_form_r2);

    pair.raw_xgcd(&mut cofactors);

    let inv = cofactors.finalize_vartime();
    x.copy_from(inv);

    gcd.is_one().and(x_nonzero)
}

/// Computes the modular inverse of `x` modulo odd `y`, returning whether one exists (`gcd(x, y) ==
/// 1` and `x != 0`). Variable-time analog of [`invert_odd_mod`].
///
/// Inputs:
/// - `x`: value to invert, must be no narrower than `y`.
/// - `y`: odd modulus.
/// - `y_inv`: `y`'s limb-sized modular inverse, as returned by `Odd<UintRef>::invert_mod_limb`.
/// - `buf`: scratch, at least `3 * x.nlimbs()` limbs.
///
/// Outputs:
/// - Returns a `bool` that is true iff `x` is invertible mod `y` (`gcd(x, y) == 1 && x != 0`).
/// - `x` holds `x^-1 mod y` when the return value is true, otherwise its state is unspecified.
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
    let mut pair = GcdPairVartime::new(a, b);

    let (u, buf) = buf.split_at_mut(limbs);
    let v = buf.leading_mut(limbs);
    let mut cofactors = CofactorPair::new(u, v, y, y_inv, None);

    pair.raw_xgcd(&mut cofactors);
    if !b.is_one().to_bool_vartime() {
        return false;
    }

    let inv = cofactors.finalize_vartime();
    a.copy_from(inv.leading(limbs));

    true
}

/// Compute `gcd(x, y)` together with non-negative Bezout coefficients `a`, `b` satisfying
/// `a*x - b*y = gcd(x, y)`, for odd `y`.
///
/// Builds on [`GcdPair::raw_xgcd`], adding the post-processing needed to turn its single
/// reduced-mod-`y` coefficient into a full two-sided Bezout identity.
///
/// Inputs:
/// - `x`, `y`: same width (`nlimbs`); `y` must be odd.
/// - `gcd`: output buffer, same width; initial contents ignored.
/// - `a`, `b`: output buffers, same width; initial contents ignored.
/// - `buf`: scratch, at least `3 * x.nlimbs()` limbs.
///
/// Outputs:
/// - `gcd` receives `gcd(x, y)`.
/// - `x`, `y` are divided in place by `gcd` (left holding `x/gcd`, `y/gcd`).
/// - `a`, `b` are left non-negative and satisfy `a*x_orig - b*y_orig = gcd`, with `a` reduced
///   modulo `y/gcd` (and bumped up to `y/gcd` itself if that reduction would otherwise leave it
///   `0`, so `a` stays nonzero).
///
/// # Panics
/// If `y` is not odd.
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
    gcd.copy_from(y);
    let mut pair = GcdPair::new(x_copy, gcd);

    let (u, rest) = rest.split_at_mut(limbs);
    let (v, _) = rest.split_at_mut(limbs);
    let mut cofactors = CofactorPair::new(u, v, y_odd, y_inv, None);

    // Perform the partial XGCD, populating `a`
    pair.raw_xgcd(&mut cofactors);
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

/// Calculate the greatest common divisor of `x` and odd `y`, and the Bezout coefficients
/// `a` and `b` such that `a*x - b*y = gcd`. Variable-time analog of [`xgcd_odd`].
///
/// Inputs:
/// - `x`, `y`: same width (`nlimbs`); `y` must be odd.
/// - `gcd`: output buffer, same width; initial contents ignored.
/// - `a`, `b`: output buffers, same width; initial contents ignored.
/// - `buf`: scratch, at least `3 * x.nlimbs()` limbs.
///
/// Outputs:
/// - `gcd` receives `gcd(x, y)`.
/// - `x`, `y` are divided in place by `gcd` (left holding `x/gcd`, `y/gcd`).
/// - `a`, `b` are left non-negative and satisfy `a*x_orig - b*y_orig = gcd`, with `a` reduced
///   modulo `y/gcd` (and bumped up to `y/gcd` itself if that reduction would otherwise leave it
///   `0`, so `a` stays nonzero).
///
/// # Panics
/// If `y` is not odd.
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
    let wide_len = if y_limbs > limbs { y_limbs } else { limbs };

    let (x_copy, rest) = buf.split_at_mut(limbs);
    x_copy.copy_from(x);
    gcd.copy_from(y_odd.as_ref());
    let mut pair = GcdPairVartime::new(x_copy, gcd);

    let (u, rest) = rest.split_at_mut(wide_len);
    let v = rest.leading_mut(wide_len);
    let mut cofactors = CofactorPair::new(u, v, y_odd, y_inv, None);

    // Perform the partial XGCD, populating `a`
    pair.raw_xgcd(&mut cofactors);
    let v_final = cofactors.finalize_vartime();
    debug_assert!(v_final.trailing(limbs).is_zero().to_bool_vartime());
    a.copy_from(v_final.leading(limbs));

    // Replace `x`, `y` with `x/gcd`, `y/gcd`
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
