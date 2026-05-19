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
