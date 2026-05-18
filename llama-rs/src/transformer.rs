use crate::config::Config;
use crate::checkpoint::Checkpoint;
use crate::weights::TransformerWeights;
use crate::state::RunState;

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
}
