// Configure
// @trace-pilot 33f1d51036a7ed5e018c718c65b39335f621ca42
#[derive(Debug,Clone)]
pub struct Config{
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}
