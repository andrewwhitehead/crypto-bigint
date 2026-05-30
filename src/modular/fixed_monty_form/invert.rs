//! Multiplicative inverses of integers in Montgomery form with a modulus set at runtime.

use super::FixedMontyForm;
use crate::{Choice, CtOption, traits::Invert};

impl<const LIMBS: usize> FixedMontyForm<LIMBS> {
    /// Computes `self^-1` representing the multiplicative inverse of `self`.
    /// i.e. `self * self^-1 = 1`.
    ///
    /// If the number was invertible, the second element of the tuple is the truthy value,
    /// otherwise it is the falsy value (in which case the first element's value is unspecified).
    #[deprecated(since = "0.7.0", note = "please use `invert` instead")]
    #[must_use]
    pub const fn inv(&self) -> CtOption<Self> {
        self.invert()
    }

    /// Computes `self^-1` representing the multiplicative inverse of `self`.
    /// i.e. `self * self^-1 = 1`.
    ///
    /// If the number was invertible, the second element of the tuple is the truthy value,
    /// otherwise it is the falsy value (in which case the first element's value is unspecified).
    #[must_use]
    pub const fn invert(&self) -> CtOption<Self> {
        let params = &self.params;
        let maybe_inverse = params.modulus().safegcd_invert_mod_precomp(
            &self.montgomery_form,
            params.mod_inv.limbs[0],
            &params.r2,
        );

        let ret = Self {
            montgomery_form: maybe_inverse.to_inner_unchecked(),
            params: self.params,
        };

        CtOption::new(ret, maybe_inverse.is_some())
    }

    /// Computes `self^-1` representing the multiplicative inverse of `self`.
    /// i.e. `self * self^-1 = 1`.
    ///
    /// If the number was invertible, the second element of the tuple is the truthy value,
    /// otherwise it is the falsy value (in which case the first element's value is unspecified).
    ///
    /// This version is variable-time with respect to the value of `self`, but constant-time with
    /// respect to `self`'s `params`.
    #[deprecated(since = "0.7.0", note = "please use `invert_vartime` instead")]
    #[must_use]
    pub const fn inv_vartime(&self) -> CtOption<Self> {
        self.invert_vartime()
    }

    /// Computes `self^-1` representing the multiplicative inverse of `self`.
    /// i.e. `self * self^-1 = 1`.
    ///
    /// If the number was invertible, the second element of the tuple is the truthy value,
    /// otherwise it is the falsy value (in which case the first element's value is unspecified).
    ///
    /// This version is variable-time with respect to the value of `self`, but constant-time with
    /// respect to `self`'s `params`.
    #[must_use]
    pub const fn invert_vartime(&self) -> CtOption<Self> {
        let maybe_inverse = self
            .params
            .modulus()
            .bingcd_invert_mod_vartime(&self.retrieve());

        if let Some(inv) = maybe_inverse {
            CtOption::some(Self::new(&inv, &self.params))
        } else {
            CtOption::new(Self::zero(&self.params), Choice::FALSE)
        }
    }
}

impl<const LIMBS: usize> Invert for FixedMontyForm<LIMBS> {
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
    use crate::modular::{FixedMontyForm, FixedMontyParams};
    use crate::{Invert, Odd, U256};

    fn params() -> FixedMontyParams<{ U256::LIMBS }> {
        FixedMontyParams::new_vartime(Odd::<U256>::from_be_hex(
            "15477BCCEFE197328255BFA79A1217899016D927EF460F4FF404029D24FA4409",
        ))
    }

    #[test]
    fn test_self_inverse() {
        let params = params();
        let x =
            U256::from_be_hex("77117F1273373C26C700D076B3F780074D03339F56DD0EFB60E7F58441FD3685");
        let x_monty = FixedMontyForm::new(&x, &params);

        let inv = x_monty.invert().unwrap();
        let res = x_monty * inv;

        assert_eq!(res.retrieve(), U256::ONE);

        let inv_trait = Invert::invert(&x_monty).unwrap();
        assert_eq!(inv_trait, inv);
    }
}
