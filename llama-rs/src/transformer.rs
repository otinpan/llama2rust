use crate::config::Config;
use crate::checkpoint::Checkpoint;
use crate::weights::TransformerWeights;
use crate::state::RunState;
use crate::ops;

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
        let _kv_mul = p.n_heads / p.n_kv_heads;
        let _hidden_dim = p.hidden_dim;
        let _head_size = dim / p.n_heads;
        let seq_len = p.seq_len;
        let n_layers = p.n_layers;

        state.x = Self::embedding(&weights, dim, token);

        for l in 0..n_layers {
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
            ops::matmul(&mut state.q,&state.xb,wq,dim,dim);
            ops::matmul(k_cache,&state.xb,wk,dim,kv_dim);
            ops::matmul(v_cache,&state.xb,wv,dim,kv_dim);
            

            for i in 0..dim{
            }

        }

        let _ = pos;
        todo!()
    }

    fn embedding(weights: &TransformerWeights<'_>, dim: usize, token: u32)->Vec<f32>{
        let token = token as usize;
        let start = token * dim;
        let end = start + dim;

        weights.token_embedding_table[start..end].to_vec()
    }
}
