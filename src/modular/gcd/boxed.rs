use crate::{BoxedUint, Odd, Resize};

/// Calculate the greatest common denominator of `a`, and `b`.
pub fn gcd_vartime(a: &BoxedUint, b: &BoxedUint) -> BoxedUint {
    let bits_precision = a.bits_precision().max(b.bits_precision());

    if b.is_zero_vartime() {
        a.resize(bits_precision)
    } else if a.is_zero_vartime() {
        b.resize(bits_precision)
    } else {
        let (mut a, mut b) = (a.resize(bits_precision), b.resize(bits_precision));
        let index = super::gcd_vartime(a.as_mut_uint_ref(), b.as_mut_uint_ref());
        if index { b } else { a }
    }
}

pub fn invert_odd_mod_vartime(a: &BoxedUint, m: &Odd<BoxedUint>) -> Option<BoxedUint> {
    let bits_precision = a.bits_precision().max(m.as_ref().bits_precision());

    let mut a = a.resize(bits_precision);
    let m_inv = m.as_uint_ref().invert_mod_limb();
    let mut buf = BoxedUint::zero_with_precision(bits_precision * 3);
    let res = super::invert_vartime(
        a.as_mut_uint_ref(),
        m.as_uint_ref(),
        m_inv,
        buf.as_mut_uint_ref(),
    );
    if res { Some(a) } else { None }
}
