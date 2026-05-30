use super::UintRef;
use crate::{Choice, Limb};

impl UintRef {
    /// Conditionally perform carrying negation: map `self` to `(self ^ 1111...1111) + 0000...0001`.
    /// Return whether the carry flag is set on the addition.
    ///
    /// Note: the carry is set if and only if `self == Self::ZERO`.
    #[inline]
    pub(crate) const fn wrapping_neg_assign(&mut self) -> Limb {
        let mut carry = Limb::ONE;
        let mut i = 0;
        while i < self.limbs.len() {
            (self.limbs[i], carry) = self.limbs[i].not().overflowing_add(carry);
            i += 1;
        }
        carry
    }

    /// Conditionally perform carrying negation: map `self` to `(self ^ 1111...1111) + 0000...0001`.
    /// Return whether the carry flag is set on the addition.
    ///
    /// Note: the carry is set if and only if `self == Self::ZERO`.
    #[inline]
    pub(crate) const fn conditional_wrapping_neg_assign(&mut self, apply: Choice) -> Limb {
        let mut carry = Limb::ONE;
        let mut neg;
        let mut i = 0;
        while i < self.limbs.len() {
            (neg, carry) = self.limbs[i].not().overflowing_add(carry);
            self.limbs[i] = Limb::select(self.limbs[i], neg, apply);
            i += 1;
        }
        Limb::select(Limb::ZERO, carry, apply)
    }
}
