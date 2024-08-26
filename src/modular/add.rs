use crate::{Odd, Uint};

#[inline]
pub(crate) const fn add_montgomery_form<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    b: &Uint<LIMBS>,
    modulus: &Odd<Uint<LIMBS>>,
) -> Uint<LIMBS> {
    a.add_mod(b, &modulus.0)
}

#[inline]
pub(crate) const fn add_montgomery_form_unsaturated<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    b: &Uint<LIMBS>,
    modulus: &Odd<Uint<LIMBS>>,
) -> Uint<LIMBS> {
    a.add_mod_unsaturated(b, &modulus.0)
}

#[inline]
pub(crate) const fn double_montgomery_form<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    modulus: &Odd<Uint<LIMBS>>,
) -> Uint<LIMBS> {
    a.double_mod(&modulus.0)
}

#[inline]
pub(crate) const fn double_montgomery_form_unsaturated<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    modulus: &Odd<Uint<LIMBS>>,
) -> Uint<LIMBS> {
    a.double_mod_unsaturated(&modulus.0)
}
