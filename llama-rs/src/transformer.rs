use crate::config::Config;
use crate::checkpoint::Checkpoint;
use crate::state::RunState;
use crate::weights::TransformerWeights;
use crate::ops;

use rayon::prelude::*;
use std::io;
use std::path::Path;

// transformerはweightsと同じ寿命
// weightsは循環参照になるから置かない
pub struct Transformer{
    pub checkpoint: Checkpoint,
    pub state: RunState,
}

impl Transformer{
    // @trace-pilot 2b5c2081e889b97ee7e465303bef5eb6a1909ca3
    // void build_transformer
    pub fn new(path: impl AsRef<Path>) -> io::Result<Self>{
        let checkpoint = Checkpoint::open(path)?;
        let state = RunState::new(checkpoint.config());
        Ok(Self { checkpoint, state })
    }

    pub fn config(&self) -> &Config {
        self.checkpoint.config()
    }

    pub fn weights(&self) -> io::Result<TransformerWeights<'_>>{
        self.checkpoint.weights()
    }

    // @trace-pilot 0c396301b935032c4a4f350961d51d8b5c958369
    // override to ~max length
    pub fn clamp_steps(&self, steps: usize)->usize{
        let max_steps=self.config().seq_len;
        if steps==0 || steps>max_steps{
            max_steps
        }else{
            steps
        }
    }

    // @trace-pilot 46d46cff98fd65977c69e8031d32d44c07d2b3de
    // float* forward(
    pub fn forward(&mut self,token: u32,pos: usize)->Vec<f32>{
        let checkpoint = &self.checkpoint;
        let state = &mut self.state;
        let weights = checkpoint.weights().expect("failed to load transformer weights");
        let p = checkpoint.config();

        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let hidden_dim = p.hidden_dim;
        let vocab_size = p.vocab_size;
        let seq_len = p.seq_len;
        let n_layers = p.n_layers;

// @trace-pilot ce1e032e4c1f426c708f5b316f12bc451aeb997c
// Self-Attention

// @trace-pilot fbf2b58b13ae6a66996ea0ed348c7de6a44fc84c
// Embedding Vector
        state.x = Self::embedding(&weights, dim, token);

        for l in 0..n_layers {
            // @trace-pilot f2f6f0819adbc404aef31ab85d3099c042829dfc
            // Q/K/V生成
            let att_weight = &weights.rms_att_weight[l * dim..(l + 1) * dim];
            // 正規化
            ops::rmsnorm(
                &mut state.xb,
                &state.x,
                att_weight);

            let loff = l * seq_len * kv_dim;
            let cache_offset = loff + pos * kv_dim;
            let cache_end = cache_offset + kv_dim;
            // @trace-pilot 04abe07bcf4f298e4692bcce03d84966fd63802c
            let k_cache = &mut state.key_cache[cache_offset..cache_end];
            let v_cache = &mut state.value_cache[cache_offset..cache_end];

            let wq=&weights.wq[l*dim*dim..(l+1)*dim*dim];
            let wk=&weights.wk[l*dim*kv_dim..(l+1)*dim*kv_dim];
            let wv=&weights.wv[l*dim*kv_dim..(l+1)*dim*kv_dim];

            // q,k,vを求める
            // q(dim,1)=wq(dim,dim)@x(dim,1)
            ops::matmul(&mut state.q,&state.xb,wq,dim,dim);
            ops::matmul(k_cache,&state.xb,wk,dim,kv_dim);
            ops::matmul(v_cache,&state.xb,wv,dim,kv_dim);

// @trace-pilot 93becf09202f2eab3a160e0864aa39c26ca7b65c
// Positional Encoding 
            // @trace-pilot 993369b863e534622167f6526969feb115f3c057
            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            Self::rope(&mut state.q, k_cache, p, pos);

// @trace-pilot cb32729cef8f59f29731e24720c1b5ccdf8de675
// Attention Score 計算
            // @trace-pilot 42e44b2b990c1e63daf572d919cc01eb139bd189
            // multihead attention. iterate over all heads
            // 各stepで0..posまで(各tokenごと)のattentionを計算し、x=attention*vで更新
            Self::score_heads(
                &state.q,
                &state.key_cache,
                &state.value_cache,
                &mut state.att,
                &mut state.xb,
                p,
                l,
                pos,
            );

            // @trace-pilot d02ef69da4e6445a14d252da2eca1dd51c51b1a4
            // final matmul to get the output of the attention
            // xb2=wo(dim,dim) @ xb(dim,1)
            let wo=&weights.wo[l*dim*dim..(l+1)*dim*dim];
            crate::ops::matmul(&mut state.xb2,&state.xb,wo,dim,dim);

            for i in 0..dim{
                state.x[i] += state.xb2[i];
            }
// @trace-pilot de5230f566396a244451229e6eaec7205e34ed0e
// Feed Forward Network
            let ffn_weight = &weights.rms_ffn_weight[l * dim..(l + 1) * dim];
            crate::ops::rmsnorm(&mut state.xb, &state.x, ffn_weight);

            let w1=&weights.w1[l*dim*hidden_dim..(l+1)*dim*hidden_dim];
            let w2=&weights.w2[l*dim*hidden_dim..(l+1)*dim*hidden_dim];
            let w3=&weights.w3[l*dim*hidden_dim..(l+1)*dim*hidden_dim];

            // feed-forward network (FFN)
            crate::ops::matmul(&mut state.hb,&state.xb,w1,dim,hidden_dim);
            crate::ops::matmul(&mut state.hb2,&state.xb,w3,dim,hidden_dim);
            
            // @trace-pilot f628e6e14ddffa24c1d49c441f241ccce9be531d
            // SwiGLU non-linearity
            Self::swiglu(&mut state.hb, &state.hb2);

            crate::ops::matmul(&mut state.xb,&state.hb,w2,hidden_dim,dim);

            for i in 0..dim{
                state.x[i] += state.xb[i];
            }
        }

        let x = state.x.clone();
        crate::ops::rmsnorm(&mut state.x, &x, weights.rms_final_weight);
        crate::ops::matmul(&mut state.logits, &state.x, weights.wcls, dim, vocab_size);
        state.logits.clone()
    }

    fn swiglu(hb: &mut [f32], hb2: &[f32]) {
        assert_eq!(hb.len(), hb2.len());

        for i in 0..hb.len() {
            let mut val = hb[i];
            // @trace-pilot ed9dbf85a2572707d853e063c10ed796e7fca788
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= 1.0 / (1.0 + (-val).exp());
            val *= hb2[i];
            hb[i] = val;
        }
    }

    fn embedding(weights: &TransformerWeights<'_>, dim: usize, token: u32)->Vec<f32>{
        let token = token as usize;
        let start = token * dim;
        let end = start + dim;

        weights.token_embedding_table[start..end].to_vec()
    }

    // q,kの要素をposが増加するにつれて回転させる
    fn rope(q: &mut [f32],k: &mut [f32],config: &Config,pos: usize) {
        let dim = config.dim;
        let head_size = dim / config.n_heads;
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

        for i in (0..dim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.0f32 / 10000.0f32.powf(head_dim as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();

            let rotn = if i < kv_dim { 2 } else { 1 };
            for v in 0..rotn {
                let vec = if v == 0 { &mut *q } else { &mut *k };
                let v0 = vec[i];
                let v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }

    // 各headのscoreを並列に出す
    // softmax(q*k/sqrt(head_size)) を計算し、attention*v を xb に書く
    fn score_heads(
        q_all: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        att_all: &mut [f32],
        xb_all: &mut [f32],
        config: &Config,
        layer: usize,
        pos: usize,
    ) {
        let dim = config.dim;
        let head_size = dim / config.n_heads;
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        let kv_mul = config.n_heads / config.n_kv_heads;
        let seq_len = config.seq_len;
        let loff = layer * seq_len * kv_dim;

        att_all
            .par_chunks_mut(seq_len)
            .zip(xb_all.par_chunks_mut(head_size))
            .enumerate()
            .for_each(|(head, (att, xb))| {
                let q_start = head * head_size;
                let q_end = q_start + head_size;
                let q = &q_all[q_start..q_end];

                for t in 0..=pos {
                    let k_start = loff + t * kv_dim + (head / kv_mul) * head_size;
                    let k_end = k_start + head_size;
                    let k = &key_cache[k_start..k_end];

                    let mut score = 0.0f32;
                    for i in 0..head_size {
                        score += q[i] * k[i];
                    }
                    att[t] = score / (head_size as f32).sqrt();
                }

                ops::softmax(&mut att[..=pos]);

                xb.fill(0.0);
                for t in 0..=pos {
                    let v_start = loff + t * kv_dim + (head / kv_mul) * head_size;
                    let v_end = v_start + head_size;
                    let v = &value_cache[v_start..v_end];
                    let a = att[t];

                    for i in 0..head_size {
                        xb[i] += a * v[i];
                    }
                }
            });
    }
}
