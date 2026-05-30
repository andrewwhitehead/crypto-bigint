use crate::{BoxedUint, CtOption, Limb, NonZero, Odd, Resize, primitives::u32_min};

impl BoxedUint {
    /// Computes the greatest common divisor of `self` and odd `rhs`.
    #[inline(always)]
    #[must_use]
    pub(crate) fn safegcd(self, rhs: &BoxedUint) -> Self {
        let mut lhs = self;
        let lhs_zero = lhs.is_zero();
        lhs.as_mut_uint_ref().conditional_set_one(lhs_zero);
        lhs = NonZero::new_unchecked(lhs).safegcd(rhs).get();
        lhs.as_mut_uint_ref()
            .conditional_copy_from(rhs.as_uint_ref(), lhs_zero);
        lhs
    }
}

impl NonZero<BoxedUint> {
    /// Computes the greatest common divisor of `self` and odd `rhs`.
    #[inline(always)]
    #[must_use]
    pub(crate) fn safegcd(self, rhs: &BoxedUint) -> Self {
        let mut lhs = self.get();
        // Factor out common factors of two and ensure `lhs` is odd.
        // There is no need to strip the zeros from `rhs`, as
        // `gcd(a, b) == gcd(a, b•2^k)` for odd `a`.
        let lhs_tz = lhs.trailing_zeros();
        let k = u32_min(lhs_tz, rhs.trailing_zeros());
        lhs.shr_assign(lhs_tz);
        lhs = Odd::new_unchecked(lhs).safegcd(rhs).get();
        lhs.shl_assign(k);
        NonZero::new(lhs).expect("ensured non-zero")
    }
}

impl Odd<BoxedUint> {
    /// Computes the greatest common divisor of `self` and odd `rhs`.
    #[inline(always)]
    #[must_use]
    pub(crate) fn safegcd(self, rhs: &BoxedUint) -> Self {
        assert!(self.bits_precision() >= rhs.bits_precision());
        let mut f = self.get();
        let mut g = rhs.resize(f.bits_precision());
        super::gcd_odd(f.as_mut_uint_ref(), g.as_mut_uint_ref());
        Odd::new(f).expect("ensured odd")
    }

    /// Calculate the multiplicative inverse of `value` modulo `self`.
    pub(crate) fn safegcd_invert_mod(&self, value: &BoxedUint) -> CtOption<BoxedUint> {
        let self_inv = self.as_uint_ref().invert_mod_limb();
        self.safegcd_invert_mod_precomp(value, self_inv, None)
    }

    /// Calculate the multiplicative inverse of `a` modulo `m`.
    pub(crate) fn safegcd_invert_mod_precomp(
        &self,
        value: &BoxedUint,
        self_inv: Limb,
        monty_form_r2: Option<&BoxedUint>,
    ) -> CtOption<BoxedUint> {
        let bits_precision = self.bits_precision();
        let limbs = self.nlimbs();
        assert!(value.bits_precision() <= bits_precision);

        let mut d = BoxedUint::zero_with_precision(bits_precision);
        let mut buf = BoxedUint::zero_with_precision(bits_precision * 3);
        let buf = buf.as_mut_uint_ref();
        let (g, buf) = buf.split_at_mut(limbs);
        g.leading_mut(value.nlimbs()).copy_from(value.as_uint_ref());
        let (e, gcd) = buf.split_at_mut(limbs);
        if let Some(r2) = monty_form_r2 {
            e.copy_from(r2.as_uint_ref());
        } else {
            e.limbs[0] = Limb::ONE;
        }

        let is_some =
            super::invert_odd_mod(g, self.as_uint_ref(), self_inv, d.as_mut_uint_ref(), e, gcd);
        CtOption::new(d, is_some)
    }
}
