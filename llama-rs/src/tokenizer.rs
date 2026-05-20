// @trace-pilot a753c602b9fc1a8a17f6019e74599aef2b3cc035
// Tokenizer

// @trace-pilot 6884d3a2b2265d5e4d213741554073e7f4dad7e5
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;


#[derive(Debug,Clone)]
struct TokenIndex{
    vocab: String,
    id: usize,
}

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub vocab_scores: Vec<f32>, 
    pub sorted_vocab: Option<Vec<TokenIndex>>,
    pub max_token_length: usize,
    pub vocab_size: usize, // token idがいくつ存在するか
}

impl Tokenizer {
    // @trace-pilot 6a5e681f77038ebba2c8b5ec7e66767ffb3503d3
    // void build_tokenizer
    pub fn new(path: impl AsRef<Path>, vocab_size: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let max_token_length = read_u32(&mut reader)? as usize;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);

        for _ in 0..vocab_size {
            let score = read_f32(&mut reader)?;
            let piece_len = read_u32(&mut reader)? as usize;
            let mut bytes = vec![0_u8; piece_len];
            reader.read_exact(&mut bytes)?;

            let piece = String::from_utf8(bytes).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid tokenizer entry: {err}"),
                )
            })?;

            vocab_scores.push(score);
            vocab.push(piece);
        }

        let sorted_vocab: Option<Vec<TokenIndex>>=None;

        Ok(Self {
            vocab,
            vocab_scores,
            sorted_vocab,
            max_token_length,
            vocab_size,
        })
    }

    // promptをtoken化する
    // // @trace-pilot 2fc39f491ccdb51bdc1f168582930830ff1cec61
    // void encode(
    pub fn encode(&mut self,prompt: Option<&str>,bos: bool, eos: bool) -> std::io::Result<Vec<u32>>{
        let prompt=prompt.ok_or_else(||{
            io::Error::new(io::ErrorKind::InvalidInput, "prompt is None")
        })?;
        
        if self.sorted_vocab.is_none(){
            self.initialize_sort_vocab();
        }

        let mut tokens: Vec<u32>=Vec::new();

        if bos{
            tokens.push(1);
        }

        if !prompt.is_empty() {
            if let Some(dummy_prefix)=self.str_lookup(" "){
                tokens.push(dummy_prefix);
            }
        }

        // 1文字ずつtoken化
        self.tokenize_per_char(prompt,&mut tokens);

        // @trace-pilot 3bef21aa67ab579865476baf3cfe842cec58d5eb
        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        self.merge_tokens(&mut tokens);

        if eos {
            tokens.push(2);
        }

        Ok(tokens)
    }

    pub fn decode(&self, prev_token: u32, token: u32) -> String {
        let mut piece = self
            .vocab
            .get(token as usize)
            .cloned()
            .unwrap_or_default();

        if prev_token == 1 && piece.starts_with(' ') {
            piece.remove(0);
        }

        if let Some(byte_val) = parse_byte_fallback(&piece) {
            return char::from(byte_val).to_string();
        }

        piece
    }

    // @trace-pilot e27df3610106a9695676f753f61f8349a9381cab
    // void safe_printf(
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    pub fn safe_piece<'a>(&self, piece: &'a str) -> Option<&'a str> {
        let _ = self;
        if piece.is_empty() {
            return None;
        }

        if piece.len() == 1 {
            let byte_val = piece.as_bytes()[0];
            if !(byte_val.is_ascii_graphic() || byte_val.is_ascii_whitespace()) {
                return None;
            }
        }

        Some(piece)
    }

    // @trace-pilot 3f084a2526336d5d95d55cd7dd9e8ef85c016f0c
    // int str_lookup
    // 単語をトークン(id) に変換する
    fn str_lookup(&self,str: &str) -> Option<u32>{
        let sorted_vocab=self.sorted_vocab.as_ref()?;

        sorted_vocab
            .binary_search_by(|token| token.vocab.as_str().cmp(str))
            .ok()
            .map(|idx| sorted_vocab[idx].id as u32)
        
    }

    

    fn tokenize_per_char(&self,prompt: &str,tokens: &mut Vec<u32>){
        let bytes=prompt.as_bytes();
        let mut str_buffer: Vec<u8>=Vec::new();
        let mut i=0;

        while i<bytes.len(){
            let b=bytes[i];

            if (b&0xC0)!=0x80{
                str_buffer.clear();
            }

            str_buffer.push(b);

            // continuation byteかチェック
            // 10...... ならcontinuation byte
            if i+1<bytes.len() && (bytes[i+1] & 0xC0)==0x80 && str_buffer.len()<4{
                i+=1;
                continue;
            }

            let piece=std::str::from_utf8(&str_buffer).ok();

            if let Some(piece)=piece{
                if let Some(id)=self.str_lookup(piece){
                    tokens.push(id);
                }else{
                    // @trace-pilot c450e58b93fae971b974a9707775c6aae6fc4cd3
                    // byte_fallback encoding: just encode each byte as a token
                    // +3 is here because the first 3 vocab elements are <unk>, <s>, </s> so the individual bytes only start at index 3
                    for &byte in &str_buffer{
                        tokens.push(byte as u32+3);
                    }
                }
            }else{
                for &byte in &str_buffer{
                    tokens.push(byte as u32+3);
                }
            }

            str_buffer.clear();
            i+=1;
        }
    }

    fn merge_tokens(&self,tokens: &mut Vec<u32>){
        loop {
            let mut best_score: f32 = -1e10;
            let mut best_id: Option<u32> = None;
            let mut best_idx: Option<usize> = None;

            for i in 0..tokens.len().saturating_sub(1) {
                let merged_vocab = format!(
                    "{}{}",
                    self.vocab[tokens[i] as usize],
                    self.vocab[tokens[i + 1] as usize]
                );

                if let Some(id) = self.str_lookup(&merged_vocab) {
                    let score = self.vocab_scores[id as usize];
                    if score > best_score {
                        crate::log_info!("merged vocab: {}",merged_vocab);
                        best_score = score;
                        best_id = Some(id);
                        best_idx = Some(i);
                    }
                }
            }

            let (best_id, best_idx) = match (best_id, best_idx) {
                (Some(id), Some(idx)) => (id, idx),
                _ => break,
            };

            tokens[best_idx] = best_id;
            tokens.remove(best_idx + 1);
        }
    }

    // @trace-pilot 0fe18937a63947ac3b1eb25bbb83d2b0a69fe204
    // lazily malloc and sort the vocabulary
    pub fn initialize_sort_vocab(&mut self){
        let mut sorted_vocab: Vec<TokenIndex>=self.vocab
            .iter()
            .cloned()
            .enumerate()
            .map(|(id,vocab)| TokenIndex {vocab,id})
            .collect();
        sorted_vocab.sort_by(|a,b| a.vocab.cmp(&b.vocab));
        self.sorted_vocab=Some(sorted_vocab);
    }

    
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn parse_byte_fallback(piece: &str) -> Option<u8> {
    if piece.len() != 6 || !piece.starts_with("<0x") || !piece.ends_with('>') {
        return None;
    }

    u8::from_str_radix(&piece[3..5], 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokenizer() -> Tokenizer {
        let vocab = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            " ".to_string(),
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
        ];
        let vocab_scores = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0];

        let mut tokenizer = Tokenizer {
            vocab,
            vocab_scores,
            sorted_vocab: None,
            max_token_length: 2,
            vocab_size: 7,
        };
        tokenizer.initialize_sort_vocab();
        tokenizer
    }

    #[test]
    fn str_lookup_finds_known_token() {
        let tokenizer = make_tokenizer();

        assert_eq!(tokenizer.str_lookup("a"), Some(4));
        assert_eq!(tokenizer.str_lookup("ab"), Some(6));
        assert_eq!(tokenizer.str_lookup("x"), None);
    }

    #[test]
    fn tokenize_per_char_pushes_known_tokens() {
        let tokenizer = make_tokenizer();
        let mut tokens = Vec::new();

        tokenizer.tokenize_per_char("ab", &mut tokens);

        assert_eq!(tokens, vec![4, 5]);
    }

    #[test]
    fn tokenize_per_char_falls_back_to_bytes() {
        let tokenizer = make_tokenizer();
        let mut tokens = Vec::new();

        tokenizer.tokenize_per_char("A", &mut tokens);

        assert_eq!(tokens, vec![68]);
    }

    #[test]
    fn merge_tokens_merges_best_pair() {
        let tokenizer = make_tokenizer();
        let mut tokens = vec![4, 5];

        tokenizer.merge_tokens(&mut tokens);

        assert_eq!(tokens, vec![6]);
    }

    #[test]
    fn encode_adds_bos_dummy_prefix_and_eos() {
        let mut tokenizer = make_tokenizer();

        let tokens = tokenizer.encode(Some("ab"), true, true).unwrap();

        assert_eq!(tokens, vec![1, 3, 6, 2]);
    }

    #[test]
    fn decode_strips_leading_space_after_bos() {
        let tokenizer = make_tokenizer();

        assert_eq!(tokenizer.decode(1, 3), "");
    }

    #[test]
    fn safe_piece_rejects_non_printable_single_byte() {
        let tokenizer = make_tokenizer();

        assert_eq!(tokenizer.safe_piece("\u{0001}"), None);
        assert_eq!(tokenizer.safe_piece("\n"), Some("\n"));
    }

    #[test]
    fn parse_byte_fallback_reads_hex_token() {
        assert_eq!(parse_byte_fallback("<0x0A>"), Some(0x0A));
        assert_eq!(parse_byte_fallback("hello"), None);
    }
}
