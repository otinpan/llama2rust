// math calc
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    assert_eq!(out.len(), x.len());
    assert_eq!(x.len(), weight.len());

    let mut ss: f32 = 0.0;
    let size = out.len() as f32;

    for &value in x {
        ss += value * value;
    }
    ss /= size;
    ss += 1e-5f32;
    let scale = 1.0 / ss.sqrt();

    for i in 0..out.len() {
        out[i] = weight[i] * (scale * x[i]);
    }
}


// @trace-pilot 93839955e758894f661e2d4e97ab9f903eb85509
// void matmul(f
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    assert_eq!(out.len(), d);
    assert_eq!(x.len(), n);
    assert_eq!(w.len(), d * n);

    // W(d,n) @ x(n,) -> out(d,)
    for i in 0..d {
        let mut val: f32=0.0;
        for j in 0..n {
            val+=w[i*n+j]*x[j];
        }
        out[i]=val;
    }
}

// @trace-pilot 62b1c67d0b2702e0f89dabf1587d7075f7383340
// void softmax(f
pub fn softmax(x: &mut [f32]){
    let max=x.iter().copied().fold(f32::NEG_INFINITY,f32::max);

    let mut sum=0.0;
    for v in x.iter_mut(){
        *v=(*v-max).exp();
        sum+=*v;
    }

    for v in x.iter_mut(){
        *v/=sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_slice_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (*a - *e).abs() <= tol,
                "expected {e}, got {a}, tol {tol}"
            );
        }
    }

    #[test]
    fn rmsnorm_scales_input_by_rms_and_weight() {
        let x = [3.0, 4.0];
        let weight = [1.0, 0.5];
        let mut out = [0.0, 0.0];

        rmsnorm(&mut out, &x, &weight);

        let ss = (3.0f32 * 3.0 + 4.0 * 4.0) / 2.0 + 1e-5f32;
        let scale = 1.0 / ss.sqrt();
        let expected = [scale * 3.0, scale * 4.0 * 0.5];
        assert_slice_close(&out, &expected, 1e-6);
    }

    #[test]
    fn matmul_multiplies_matrix_and_vector() {
        let x = [2.0, -1.0, 3.0];
        let w = [
            1.0, 0.0, 2.0,
            -1.0, 4.0, 0.5,
        ];
        let mut out = [0.0, 0.0];

        matmul(&mut out, &x, &w, 3, 2);

        let expected = [8.0, -4.5];
        assert_slice_close(&out, &expected, 1e-6);
    }

    #[test]
    fn softmax_normalizes_values() {
        let mut x = [1.0, 2.0, 3.0];

        softmax(&mut x);

        let e0 = (-2.0f32).exp();
        let e1 = (-1.0f32).exp();
        let e2 = 1.0f32;
        let sum = e0 + e1 + e2;
        let expected = [e0 / sum, e1 / sum, e2 / sum];
        assert_slice_close(&x, &expected, 1e-6);
    }
}
