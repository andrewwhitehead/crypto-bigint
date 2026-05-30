use crate::{BoxedUint, Odd};

impl BoxedUint {
    /// Calculate the greatest common denominator of `a` and `b`.
    #[inline(always)]
    pub(crate) fn bingcd_vartime(&self, rhs: &Self) -> Self {
        let (mut a, mut b) = (self.clone(), rhs.clone());
        let index = super::vartime::gcd_vartime(a.as_mut_uint_ref(), b.as_mut_uint_ref());
        if index { b } else { a }
    }
}

impl Odd<BoxedUint> {
    /// Computes the multiplicative inverse of `value` mod `self`.
    #[inline(always)]
    #[must_use]
    pub(crate) fn bingcd_invert_mod_vartime(&self, value: &BoxedUint) -> Option<BoxedUint> {
        let bits_precision = self.bits_precision();
        assert!(value.bits_precision() <= bits_precision);

        let mut a = value.clone();
        let self_inv = self.as_uint_ref().invert_mod_limb();
        let mut buf = BoxedUint::zero_with_precision(bits_precision * 3);

        let res = super::vartime::invert_vartime(
            a.as_mut_uint_ref(),
            self.as_uint_ref(),
            self_inv,
            buf.as_mut_uint_ref(),
        );
        if res { Some(a) } else { None }
    }
}
