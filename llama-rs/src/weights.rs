// Weights

#[derive(Debug, Clone, Copy)]
pub struct TransformerWeights<'a> {
    // token embedding table
    pub token_embedding_table: &'a [f32],

    // rmsnorm
    pub rms_att_weight: &'a [f32],
    pub rms_ffn_weight: &'a [f32],

    // attention
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],

    // feed forward
    pub w1: &'a [f32],
    pub w2: &'a [f32],
    pub w3: &'a [f32],

    // final rmsnorm
    pub rms_final_weight: &'a [f32],

    // classifier
    pub wcls: &'a [f32],
}
