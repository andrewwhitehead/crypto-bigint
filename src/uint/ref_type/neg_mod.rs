use super::UintRef;
use crate::{Choice, Limb, NonZero, word};

impl UintRef {
    /// Computes `-self mod p`.
    /// Assumes `self` is in `[0, p)`.
    #[inline]
    #[track_caller]
    pub const fn wrapping_neg_mod_assign(&mut self, p: &NonZero<Self>) {
        self.conditional_wrapping_neg_mod_assign(p, Choice::TRUE);
    }

    /// Computes `-a mod p`.
    /// Assumes `self` is in `[0, p)`.
    #[inline]
    #[track_caller]
    pub const fn conditional_wrapping_neg_mod_assign(&mut self, p: &NonZero<Self>, apply: Choice) {
        let p = p.as_ref();
        debug_assert!(self.bits_precision() >= p.bits_precision());
        // If self is zero, then leave it untouched.
        // Otherwise self would become `p` which is too large.
        let apply = apply.and(self.is_nonzero());

        let mut i = 0;
        let mask = Limb(word::choice_to_mask(apply));
        let mut carry = Limb::ONE;

        while i < p.limbs.len() {
            (self.limbs[i], carry) = self.limbs[i]
                .not()
                .carrying_add(p.limbs[i].bitand(mask), carry);
            i += 1;
        }
        while i < self.limbs.len() {
            (self.limbs[i], carry) = Limb::MAX.overflowing_add(carry);
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::U256;

    #[test]
    fn neg_mod_random() {
        let x =
            U256::from_be_hex("8d16e171674b4e6d8529edba4593802bf30b8cb161dd30aa8e550d41380007c2");
        let p =
            U256::from_be_hex("928334a4e4be0843ec225a4c9c61df34bdc7a81513e4b6f76f2bfa3148e2e1b5")
                .to_nz()
                .unwrap();

        let mut actual = x;
        actual
            .as_mut_uint_ref()
            .wrapping_neg_mod_assign(p.as_uint_ref());
        let expected =
            U256::from_be_hex("056c53337d72b9d666f86c9256ce5f08cabc1b63b207864ce0d6ecf010e2d9f3");
        assert_eq!(expected, actual);
    }

    #[test]
    fn neg_mod_zero() {
        let p =
            U256::from_be_hex("928334a4e4be0843ec225a4c9c61df34bdc7a81513e4b6f76f2bfa3148e2e1b5")
                .to_nz()
                .unwrap();

        let mut actual = U256::ZERO;
        actual
            .as_mut_uint_ref()
            .wrapping_neg_mod_assign(p.as_uint_ref());
        assert_eq!(U256::ZERO, actual);
    }
}
