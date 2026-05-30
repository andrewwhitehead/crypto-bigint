//! Multiplicative inverses of boxed integers in Montgomery form.

use super::BoxedMontyForm;
use crate::{Choice, CtOption, Invert};

impl BoxedMontyForm {
    /// Computes `self^-1` representing the multiplicative inverse of `self`,
    /// i.e. `self * self^-1 = 1`.
    #[must_use]
    pub fn invert(&self) -> CtOption<Self> {
        let maybe_inverse = self.params.modulus().safegcd_invert_mod_precomp(
            &self.montgomery_form,
            self.params.mod_inv().limbs[0],
            Some(self.params.r2()),
        );
        maybe_inverse.map(|inv| Self::from_montgomery(inv, &self.params))
    }

    /// Computes `self^-1` representing the multiplicative inverse of `self`,
    /// i.e. `self * self^-1 = 1`.
    ///
    /// This version is variable-time with respect to the self of `self`, but constant-time with
    /// respect to `self`'s `params`.
    #[must_use]
    pub fn invert_vartime(&self) -> CtOption<Self> {
        let maybe_inverse = self
            .params
            .modulus()
            .bingcd_invert_mod_vartime(&self.retrieve());

        if let Some(inv) = maybe_inverse {
            CtOption::some(Self::new(inv, &self.params))
        } else {
            CtOption::new(Self::zero(&self.params), Choice::FALSE)
        }
    }
}

impl Invert for BoxedMontyForm {
    type Output = CtOption<Self>;

    fn invert(&self) -> Self::Output {
        self.invert()
    }

    fn invert_vartime(&self) -> Self::Output {
        self.invert_vartime()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        BoxedUint, Invert,
        modular::{BoxedMontyForm, BoxedMontyParams},
    };
    use hex_literal::hex;

    fn monty_params() -> BoxedMontyParams {
        BoxedMontyParams::new(
            BoxedUint::from_be_slice(
                &hex!("15477BCCEFE197328255BFA79A1217899016D927EF460F4FF404029D24FA4409"),
                256,
            )
            .unwrap()
            .to_odd()
            .unwrap(),
        )
    }

    #[test]
    fn test_self_inverse() {
        let params = monty_params();
        let x = BoxedUint::from_be_slice(
            &hex!("77117F1273373C26C700D076B3F780074D03339F56DD0EFB60E7F58441FD3685"),
            256,
        )
        .unwrap();
        let x_mod = BoxedMontyForm::new(x, &params);

        let inv = x_mod.invert().unwrap();
        let res = &x_mod * &inv;

        assert!(bool::from(res.retrieve().is_one()));

        let inv_trait = Invert::invert(&x_mod).unwrap();
        assert_eq!(inv_trait, inv);
    }
}
