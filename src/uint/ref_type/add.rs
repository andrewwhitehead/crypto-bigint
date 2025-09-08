use super::UintRef;
use crate::{ConstChoice, Limb};

impl UintRef {
    /// Perform an in-place carrying add of a limb, returning the carried limb value.
    #[inline]
    pub const fn add_assign_limb(&mut self, mut rhs: Limb) -> Limb {
        let mut i = 0;
        while i < self.0.len() {
            (self.0[i], rhs) = self.0[i].overflowing_add(rhs);
            i += 1;
        }
        rhs
    }

    /// Perform an in-place carrying add of another [`UintRef`], returning the carried limb value.
    #[inline]
    pub const fn carrying_add_assign(&mut self, rhs: &Self, carry: Limb) -> Limb {
        self.carrying_add_assign_slice(&rhs.0, carry)
    }

    /// Perform an in-place carrying add of another limb slice, returning the carried limb value.
    #[inline]
    pub const fn carrying_add_assign_slice(&mut self, rhs: &[Limb], mut carry: Limb) -> Limb {
        assert!(self.0.len() == rhs.len(), "length mismatch");
        let mut i = 0;
        while i < self.0.len() {
            (self.0[i], carry) = self.0[i].carrying_add(rhs[i], carry);
            i += 1;
        }
        carry
    }

    /// Perform a conditional in-place carrying add of another limb slice, returning the carried limb value.
    #[inline]
    pub const fn conditional_carrying_add_assign(
        &mut self,
        rhs: &Self,
        carry: Limb,
        choice: ConstChoice,
    ) -> Limb {
        self.conditional_carrying_add_assign_slice(&rhs.0, carry, choice)
    }

    /// Perform a conditional in-place carrying add of another limb slice, returning the carried limb value.
    #[inline]
    pub const fn conditional_carrying_add_assign_slice(
        &mut self,
        rhs: &[Limb],
        mut carry: Limb,
        choice: ConstChoice,
    ) -> Limb {
        assert!(self.0.len() == rhs.len(), "length mismatch");
        let mut i = 0;
        while i < self.0.len() {
            (self.0[i], carry) =
                self.0[i].carrying_add(Limb::select(Limb::ZERO, rhs[i], choice), carry);
            i += 1;
        }
        carry
    }
}
