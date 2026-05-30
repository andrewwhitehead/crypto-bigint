use crate::{
    Choice, CtOption, Limb, Odd, Uint, UintRef, Word, modular::gcd::ExtendedIntRef,
    uint::gcd::OddUintXgcdOutput, word,
};

use self::matrix::SafegcdMatrix;

#[cfg(feature = "alloc")]
mod boxed;
mod matrix;

const GCD_BATCH_SIZE: u32 = Limb::BITS - 2;

/// Calculate the maximum number of iterations required according to
/// safegcd-bounds: <https://github.com/sipa/safegcd-bounds>
// NOTE: the division is non-constant-time, but this is used to compute the number of iterations we
// perform which is leaked in timing information
#[allow(clippy::integer_division_remainder_used, reason = "public parameter")]
const fn iterations(bits: u32) -> u32 {
    (45907 * bits + 30179) / 19929
}

/// Calculate the greatest common denominator of odd `f`, and `g`.
#[inline(always)]
pub const fn gcd_odd(f: &mut UintRef, g: &mut UintRef) {
    let len = f.nlimbs();
    assert!(len == g.nlimbs());
    assert!(f.is_odd().to_bool_vartime(), "f must be odd");

    let mut steps = iterations(f.bits_precision());
    let mut delta = 1;
    let mut matrix;
    let (mut f, mut g) = (
        ExtendedIntRef::new(f, Limb::ZERO),
        ExtendedIntRef::new(g, Limb::ZERO),
    );

    while steps > GCD_BATCH_SIZE {
        (matrix, delta) = partial_xgcd(f.low_limb(), g.low_limb(), delta, GCD_BATCH_SIZE);
        matrix.wrapping_apply_shift(&mut f, &mut g);
        steps -= GCD_BATCH_SIZE;
    }

    if steps != 0 {
        (matrix, _) = partial_xgcd(f.low_limb(), g.low_limb(), delta, steps);
        matrix.wrapping_half_apply_shift(&mut f, &g);
    }

    f.abs_drop_extension();
}

/// Calculate the greatest common denominator of odd `f`, and `g`.
#[inline(always)]
pub const fn invert_odd_mod(
    a: &mut UintRef,
    m: &Odd<UintRef>,
    m_inv: Limb,
    d: &mut UintRef,
    e: &mut UintRef,
    gcd: &mut UintRef,
) -> Choice {
    let a_nonzero = a.is_nonzero();
    half_xgcd_odd(a, m, m_inv, d, e, gcd);
    gcd.is_one().and(a_nonzero)
}

#[inline(always)]
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
    let (x_copy, _) = buf.split_at_mut(limbs);
    x_copy.copy_from(x);

    half_xgcd_odd(x_copy, y_odd, y_inv, a, b, gcd);

    // Replace x, y with x/gcd, y/gcd
    x.div_exact(gcd);
    y.div_exact(gcd);

    // Reduce a modulo y/gcd
    b.copy_from(y);
    a.div_rem(b);
    a.copy_from(b);
    // Set a to y/gcd if it became zero: this ensures a positive value for b
    a.conditional_copy_from(y, a.is_zero());

    // Compute b: ax - by = gcd => b = (a(x/gcd) - 1) / (y/gcd)
    let bp = buf.leading_mut(limbs * 2);
    bp.fill(Limb::MAX); // set to -1
    a.wrapping_mul_add(x, bp);
    let exact = bp.div_exact(y);
    b.copy_from(bp.leading(limbs));
    debug_assert!(exact.to_bool_vartime());
}

#[inline(always)]
const fn half_xgcd_odd(
    x: &mut UintRef,
    y: &Odd<UintRef>,
    y_inv: Limb,
    d: &mut UintRef,
    e: &mut UintRef,
    gcd: &mut UintRef,
) {
    assert!(x.nlimbs() == y.as_ref().nlimbs());

    let f = &mut *gcd;
    f.copy_from(y.as_ref());

    let g = &mut *x;

    let mut steps = iterations(f.bits_precision());
    let mut delta = 1;
    let mut matrix;

    let (mut f, mut g, mut d, mut e) = (
        ExtendedIntRef::new(f, Limb::ZERO),
        ExtendedIntRef::new(g, Limb::ZERO),
        ExtendedIntRef::new(d, Limb::ZERO),
        ExtendedIntRef::new(e, Limb::ZERO),
    );

    while steps > GCD_BATCH_SIZE {
        (matrix, delta) = partial_xgcd(f.low_limb(), g.low_limb(), delta, GCD_BATCH_SIZE);
        matrix.wrapping_apply_shift(&mut f, &mut g);
        matrix.wrapping_apply_div2k_mod(&mut d, &mut e, y, y_inv);
        steps -= GCD_BATCH_SIZE;
    }

    if steps != 0 {
        (matrix, _) = partial_xgcd(f.low_limb(), g.low_limb(), delta, steps);
        matrix.wrapping_half_apply_shift(&mut f, &g);
        matrix.wrapping_half_apply_div2k_mod(&mut d, &e, y, y_inv);
    }

    let (_, f_sign) = f.abs_drop_extension();
    d.conditional_wrapping_neg_assign(f_sign);
    d.try_reduce_mod(y.as_nz_ref());
    d.conditional_wrapping_add_assign_unsigned(y.as_ref(), d.is_negative());
    debug_assert!(!d.is_negative().to_bool_vartime());
}

/// Perform `batch` steps of the gcd reduction process on signed tail values `f` and `g`.
#[inline(always)]
const fn partial_xgcd(
    mut f: Limb,
    mut g: Limb,
    mut delta: i64,
    mut batch: u32,
) -> (SafegcdMatrix, i64) {
    debug_assert!(f.is_odd().to_bool_vartime(), "f must be odd");
    let mut matrix = SafegcdMatrix::UNIT;
    while batch != 0 {
        (f, g, matrix, delta) = gcd_step(f, g, matrix, delta);
        batch -= 1;
    }
    (matrix, delta)
}

/// This follows the half-delta variant of safegcd-bounds which reduces the round count.
/// <https://github.com/sipa/safegcd-bounds>
#[inline(always)]
#[allow(clippy::cast_sign_loss)]
const fn gcd_step(
    f: Limb,
    g: Limb,
    t: SafegcdMatrix,
    delta: i64,
) -> (Limb, Limb, SafegcdMatrix, i64) {
    let d_gtz = word::choice_from_nz((delta & !(delta >> 63)) as Word);
    let g_odd = Limb(g.0 & 1);
    let f_adj = Limb::select(g_odd, g_odd.wrapping_neg(), d_gtz);
    let swap = d_gtz.and(g_odd.lsb_to_choice());
    (
        Limb::select(f, g, swap),
        g.wrapping_add(f.wrapping_mul(f_adj)).shr1().0,
        SafegcdMatrix {
            m0: (
                Limb::select(t.m0.0, t.m1.0, swap).shl(1),
                Limb::select(t.m0.1, t.m1.1, swap).shl(1),
            ),
            m1: (
                t.m1.0.wrapping_add(t.m0.0.wrapping_mul(f_adj)),
                t.m1.1.wrapping_add(t.m0.1.wrapping_mul(f_adj)),
            ),
            k: t.k + 1,
        },
        swap.select_i64(2i64.wrapping_add(delta), 2i64.wrapping_sub(delta)),
    )
}

impl<const LIMBS: usize> Odd<Uint<LIMBS>> {
    /// Computes the greatest common divisor of `self` and `rhs`.
    #[doc(hidden)]
    #[inline(always)]
    #[must_use]
    // TODO: remove from public API (already undocumented)
    pub const fn safegcd(&self, rhs: &Uint<LIMBS>) -> Self {
        let mut f = self.get_copy();
        let mut g = *rhs;
        gcd_odd(f.as_mut_uint_ref(), g.as_mut_uint_ref());
        Odd::new_unchecked(f)
    }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn safexgcd(&self, rhs: &Uint<LIMBS>) -> OddUintXgcdOutput<LIMBS> {
        let mut x = *rhs;
        let mut y = self.get_copy();
        let mut gcd = Uint::<LIMBS>::ZERO;
        let mut buf = [[Limb::ZERO; LIMBS]; 2];
        let buf = UintRef::new_flattened_mut(&mut buf);
        let mut a = Uint::<LIMBS>::ZERO;
        let mut b = Uint::<LIMBS>::ONE;

        xgcd_odd(
            x.as_mut_uint_ref(),
            y.as_mut_uint_ref(),
            gcd.as_mut_uint_ref(),
            a.as_mut_uint_ref(),
            b.as_mut_uint_ref(),
            buf,
        );
        let gcd = Odd::new_unchecked(gcd);

        // `a` was specifically chosen to be non-zero so that `b` is positive, `ax - by = gcd`
        // we minimize both `|a|` and `|b|` by allowing for negative `a`, and conditionally
        // subtracting `y/gcd` from `a` when `2•a > y/gcd`
        let (a_dbl, a_dbl_hi) = a.shl1_with_carry(Limb::ZERO);
        let swap = a_dbl_hi.is_nonzero().or(Uint::gt(&a_dbl, &y));
        let a = Uint::select(&a, &a.wrapping_sub(&y), swap);
        let b = Uint::select(&b.wrapping_neg(), &x.wrapping_sub(&b), swap);

        OddUintXgcdOutput {
            gcd,
            lhs_on_gcd: y,
            rhs_on_gcd: x,
            x: *b.as_int(),
            y: *a.as_int(),
        }
    }

    /// Computes the multiplicative inverse of `value` mod `self`.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn safegcd_invert_mod(&self, value: &Uint<LIMBS>) -> CtOption<Uint<LIMBS>> {
        let self_inv = self.as_uint_ref().invert_mod_limb();
        self.safegcd_invert_mod_precomp(value, self_inv, &Uint::ONE)
    }

    /// Computes the multiplicative inverse of `value` mod `self`.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn safegcd_invert_mod_precomp(
        &self,
        value: &Uint<LIMBS>,
        self_inv: Limb,
        monty_form_r2: &Uint<LIMBS>,
    ) -> CtOption<Uint<LIMBS>> {
        let mut a = *value;
        let m = self.as_uint_ref();
        let mut d = Uint::<LIMBS>::ZERO;
        let mut e = *monty_form_r2;
        let mut gcd = Uint::<LIMBS>::ZERO;

        let is_some = invert_odd_mod(
            a.as_mut_uint_ref(),
            m,
            self_inv,
            d.as_mut_uint_ref(),
            e.as_mut_uint_ref(),
            gcd.as_mut_uint_ref(),
        );

        CtOption::new(d, is_some)
    }
}
