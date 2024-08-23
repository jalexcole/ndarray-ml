pub trait NGram {
    /// Compile the n-gram counts for the text(s) in `corpus_fp`.
    fn train(&self, corpus_fp: &str, n: usize);
    /// Return the distribution over proposed next words under the `N`-gram language model.
    fn completion(&self, words: &Vec<String>, N: usize) -> Vec<(String, f64)>;
    /// Use the `N`-gram language model to generate sentences.
    fn generate(&self, n: usize, seed_words: &Vec<String>, sentences: usize) -> String;
    /// Calculate the model perplexity on a sequence of words.
    fn perplexity(&self, n: usize, words: &Vec<String>) -> f64 {
        self.cross_entropy(words, n).exp()
    }
    /// Calculate the model cross-entropy on a sequence of words against the empirical distribution of words in a sample.
    fn cross_entropy(&self, words: &Vec<String>, n: usize) -> f64;
    /// Compute the log probability of a sequence of words under the unsmoothed, maximum-likelihood `N`-gram language model.
    fn log_prob(&self, words: &Vec<String>, n: usize) -> f64;
}

trait NGramHidden: NGram {
    fn generate(&self, n: usize, seed_words: &Vec<String>, sentences: usize) -> String;
}

pub struct MLENGram;

impl NGram for MLENGram {
    fn train(&self, corpus_fp: &str, n: usize) {
        todo!()
    }

    fn completion(&self, words: &Vec<String>, N: usize) -> Vec<(String, f64)> {
        todo!()
    }

    fn generate(&self, n: usize, seed_words: &Vec<String>, sentences: usize) -> String {
        todo!()
    }

    fn perplexity(&self, n: usize, words: &Vec<String>) -> f64 {
        todo!()
    }

    fn cross_entropy(&self, words: &Vec<String>, n: usize) -> f64 {
        todo!()
    }

    fn log_prob(&self, words: &Vec<String>, n: usize) -> f64 {
        todo!()
    }
}

pub struct AdditiveNGram;

impl NGram for AdditiveNGram {
    fn train(&self, corpus_fp: &str, n: usize) {
        todo!()
    }

    fn completion(&self, words: &Vec<String>, N: usize) -> Vec<(String, f64)> {
        todo!()
    }

    fn generate(&self, n: usize, seed_words: &Vec<String>, sentences: usize) -> String {
        todo!()
    }

    fn perplexity(&self, n: usize, words: &Vec<String>) -> f64 {
        todo!()
    }

    fn cross_entropy(&self, words: &Vec<String>, n: usize) -> f64 {
        todo!()
    }

    fn log_prob(&self, words: &Vec<String>, n: usize) -> f64 {
        todo!()
    }
}

pub struct GoodTuringNGram;

impl NGram for GoodTuringNGram {
    fn train(&self, corpus_fp: &str, n: usize) {
        todo!()
    }

    fn completion(&self, words: &Vec<String>, N: usize) -> Vec<(String, f64)> {
        todo!()
    }

    fn generate(&self, n: usize, seed_words: &Vec<String>, sentences: usize) -> String {
        todo!()
    }

    fn perplexity(&self, n: usize, words: &Vec<String>) -> f64 {
        todo!()
    }

    fn cross_entropy(&self, words: &Vec<String>, n: usize) -> f64 {
        todo!()
    }

    fn log_prob(&self, words: &Vec<String>, n: usize) -> f64 {
        todo!()
    }
}

pub enum NGrams {
    MLENGram(MLENGram),
    AdditiveNGram(AdditiveNGram),
    GoodTuringNGram(GoodTuringNGram),
}

impl NGram for NGrams {
    fn completion(&self, words: &Vec<String>, N: usize) -> Vec<(String, f64)> {
        todo!()
    }

    fn generate(&self, n: usize, seed_words: &Vec<String>, sentences: usize) -> String {
        todo!()
    }

    fn perplexity(&self, n: usize, words: &Vec<String>) -> f64 {
        todo!()
    }

    fn cross_entropy(&self, words: &Vec<String>, n: usize) -> f64 {
        todo!()
    }

    fn log_prob(&self, words: &Vec<String>, n: usize) -> f64 {
        match self {
            NGrams::MLENGram(_) => todo!(),
            NGrams::AdditiveNGram(_) => todo!(),
            NGrams::GoodTuringNGram(_) => todo!(),
        }
    }

    fn train(&self, corpus_fp: &str, n: usize) {
        match self {
            NGrams::MLENGram(_) => todo!(),
            NGrams::AdditiveNGram(_) => todo!(),
            NGrams::GoodTuringNGram(_) => todo!(),
        }
    }
}
