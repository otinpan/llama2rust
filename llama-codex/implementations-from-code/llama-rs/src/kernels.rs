pub const RMS_NORM_EPS: f32 = 1.0e-5;

pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    assert_eq!(out.len(), x.len(), "out and x must have the same length");
    assert_eq!(
        x.len(),
        weight.len(),
        "x and weight must have the same length"
    );

    let mean_square = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + RMS_NORM_EPS).sqrt();

    for ((dst, src), w) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *dst = w * (src * scale);
    }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;

    for value in x.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }

    for value in x.iter_mut() {
        *value /= sum;
    }
}

pub fn matmul(out: &mut [f32], weight: &[f32], x: &[f32], rows: usize, cols: usize) {
    assert_eq!(out.len(), rows, "out length must match row count");
    assert_eq!(
        weight.len(),
        rows * cols,
        "weight size must match matrix dimensions"
    );
    assert_eq!(x.len(), cols, "x length must match column count");

    for (row, dst) in out.iter_mut().enumerate() {
        let start = row * cols;
        let weights = &weight[start..start + cols];
        *dst = weights.iter().zip(x.iter()).map(|(w, xv)| w * xv).sum();
    }
}

pub fn accum(x: &mut [f32], delta: &[f32]) {
    assert_eq!(x.len(), delta.len(), "accum inputs must have the same length");
    for (xv, dv) in x.iter_mut().zip(delta.iter()) {
        *xv += dv;
    }
}

pub fn silu(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = *value / (1.0 + (-*value).exp());
    }
}

pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    assert_eq!(out.len(), gate.len(), "out and gate must have the same length");
    assert_eq!(gate.len(), up.len(), "gate and up must have the same length");

    for ((dst, gate_value), up_value) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        let silu_gate = *gate_value / (1.0 + (-*gate_value).exp());
        *dst = silu_gate * *up_value;
    }
}

pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_size: usize) {
    assert!(
        q.len().is_multiple_of(head_size),
        "q length must be divisible by head_size"
    );
    assert!(
        k.len().is_multiple_of(head_size),
        "k length must be divisible by head_size"
    );
    assert!(head_size.is_multiple_of(2), "head_size must be even");

    apply_rope_to_slice(q, pos, head_size);
    apply_rope_to_slice(k, pos, head_size);
}

fn apply_rope_to_slice(x: &mut [f32], pos: usize, head_size: usize) {
    for head in x.chunks_exact_mut(head_size) {
        for pair_index in (0..head_size).step_by(2) {
            let freq = 1.0_f32 / 10000.0_f32.powf(pair_index as f32 / head_size as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let x0 = head[pair_index];
            let x1 = head[pair_index + 1];
            head[pair_index] = x0 * cos - x1 * sin;
            head[pair_index + 1] = x0 * sin + x1 * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{accum, apply_rope, matmul, rmsnorm, silu, softmax, swiglu};

    fn approx_eq(left: &[f32], right: &[f32], tol: f32) {
        assert_eq!(left.len(), right.len(), "length mismatch");
        for (index, (l, r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (*l - *r).abs() <= tol,
                "index {index} differs: left={l}, right={r}, tol={tol}"
            );
        }
    }

    #[test]
    fn applies_rmsnorm() {
        let x = [1.0, 2.0];
        let weight = [1.0, 1.5];
        let mut out = [0.0; 2];

        rmsnorm(&mut out, &x, &weight);

        approx_eq(&out, &[0.6324543, 1.8973628], 1.0e-6);
    }

    #[test]
    fn normalizes_softmax() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        approx_eq(&x, &[0.09003057, 0.24472848, 0.66524094], 1.0e-6);
    }

    #[test]
    fn multiplies_matrix_and_vector() {
        let weight = [
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0,
        ];
        let x = [1.0, 0.5, -1.0];
        let mut out = [0.0; 2];

        matmul(&mut out, &weight, &x, 2, 3);

        approx_eq(&out, &[-1.0, 0.5], 1.0e-6);
    }

    #[test]
    fn accumulates_residuals() {
        let mut x = [1.0, 2.0, 3.0];
        accum(&mut x, &[0.25, -0.5, 1.0]);
        approx_eq(&x, &[1.25, 1.5, 4.0], 1.0e-6);
    }

    #[test]
    fn applies_silu_in_place() {
        let mut x = [-1.0, 0.0, 1.0];
        silu(&mut x);
        approx_eq(&x, &[-0.26894143, 0.0, 0.7310586], 1.0e-6);
    }

    #[test]
    fn applies_swiglu() {
        let mut out = [0.0; 3];
        swiglu(&mut out, &[1.0, -1.0, 0.5], &[2.0, 3.0, 4.0]);
        approx_eq(&out, &[1.4621172, -0.8068243, 1.2449187], 1.0e-6);
    }

    #[test]
    fn applies_rope_per_head() {
        let mut q = [1.0, 0.0, 0.0, 1.0];
        let mut k = [1.0, 0.0, 0.0, 1.0];

        apply_rope(&mut q, &mut k, 1, 2);

        approx_eq(&q, &[0.5403023, 0.84147096, -0.84147096, 0.5403023], 1.0e-6);
        approx_eq(&k, &[0.5403023, 0.84147096, -0.84147096, 0.5403023], 1.0e-6);
    }
}
