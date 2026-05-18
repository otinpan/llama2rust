// @trace-pilot 98257553dbb6647925fa2cc6bf95febb5a1373f2
// Sampler;

#[derive(Debug,Clone,Copy)]
pub struct ProbIndex{
    pub prob: f32,
    pub index: usize,
}

#[derive(Debug)]
pub struct Sampler{
    pub vocab_size: usize,
    pub prob_index: Vec<ProbIndex>,
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
}

impl Sampler{
    // @trace-pilot a954bf9684037b17f19305e5ef58919a83204148
    // void build_sampler
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> Self{
        let prob_index=vec![
            ProbIndex{
                prob: 0.0,
                index: 0,
            };
            vocab_size
        ];

        Self{
            vocab_size,
            prob_index,
            temperature,
            topp,
            rng_state: rng_seed,
        }
    }
}

