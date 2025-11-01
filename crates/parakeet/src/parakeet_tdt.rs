use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Axis};
use ort::session::Session;
use std::path::{Path, PathBuf};

/// TDT model configs
#[derive(Debug, Clone)]
pub struct TDTModelConfig {
    pub vocab_size: usize,
    pub blank_idx: usize,
    pub max_tokens_per_step: usize,
}

impl Default for TDTModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8193,
            blank_idx: 8192, // vocab_size - 1
            max_tokens_per_step: 10,
        }
    }
}

#[derive(Clone)]
pub struct State {
    h: Array3<f32>,
    c: Array3<f32>,
}

pub struct ParakeetTDTModel {
    preprocessor: Session,
    encoder: Session,
    decoder_joint: Session,
    config: TDTModelConfig,
}

impl ParakeetTDTModel {
    /// Load TDT model from directory containing encoder and decoder_joint ONNX files
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Find encoder and decoder_joint files
        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_joint_path = Self::find_decoder_joint(model_dir)?;
        let prepocessor_path = Self::find_preprocessor(model_dir)?;

        let config = TDTModelConfig::default();

        // Load encoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        // Load decoder_joint
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_joint = builder.commit_from_file(&decoder_joint_path)?;

        // Load preprocessor
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let preprocessor = builder.commit_from_file(&prepocessor_path)?;

        let state = State {
            h: Array3::<f32>::zeros((2, 1, 640)),
            c: Array3::<f32>::zeros((2, 1, 640)),
        };

        Ok(Self {
            preprocessor,
            encoder,
            decoder_joint,
            config,
        })
    }

    fn find_preprocessor(dir: &Path) -> Result<PathBuf> {
        let candidates = ["nemo128.onnx"];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No encoder model found in {}",
            dir.display()
        )))
    }

    fn find_encoder(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "encoder-model.onnx",
            "encoder-model.int8.onnx",
            "encoder.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No encoder model found in {}",
            dir.display()
        )))
    }

    fn find_decoder_joint(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "decoder_joint-model.onnx",
            "decoder_joint-model.int8.onnx",
            "decoder_joint.onnx",
            "decoder-model.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No decoder_joint model found in {}",
            dir.display()
        )))
    }

    /// Run greedy decoding - returns list of (token_ids, timestamps) for each sequence
    pub fn forward(
        &mut self,
        waves: Array2<f32>,
        waves_lens: Array1<i64>,
    ) -> Result<Vec<(Vec<i32>, Vec<usize>)>> {
        // Run preprocessor
        let (features, features_len) = self.preprocess(waves, waves_lens)?;
        // Run encoder
        let (encoder_out, encoder_len) = self.encode(features, features_len)?;

        // Run greedy decoding with decoder_joint
        let results = self.decoding(encoder_out, encoder_len)?;

        Ok(results)
    }

    pub fn preprocess(
        &mut self,
        wave: Array2<f32>,
        lens: Array1<i64>,
    ) -> Result<(Array3<f32>, Array1<i64>)> {
        let outputs = self.preprocessor.run(ort::inputs![
            "waveforms"=>ort::value::Value::from_array(wave)?,
            "waveforms_lens"=>ort::value::Value::from_array(lens)?,
        ])?;

        let features = outputs["features"].try_extract_tensor::<f32>()?;
        let shape = features.0.as_ref();

        if shape.len() != 3 {
            return Err(Error::Model(format!(
                "Expected 3D encoder output, got shape: {shape:?}"
            )));
        }
        let features = Array3::from_shape_vec(
            [shape[0] as usize, shape[1] as usize, shape[2] as usize],
            features.1.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to create features array: {e}")))?;

        let features_lens = outputs["features_lens"].try_extract_tensor::<i64>()?;
        let shape = features_lens.0.as_ref();

        if shape.len() != 1 {
            return Err(Error::Model(format!(
                "Expected 3D encoder output, got shape: {shape:?}"
            )));
        }

        let features_lens =
            Array1::from_shape_vec([shape[0] as usize], features_lens.1.to_vec())
                .map_err(|e| Error::Model(format!("Failed to create features_lens array: {e}")))?;

        Ok((features, features_lens))
    }

    pub fn encode(
        &mut self,
        features: Array3<f32>,
        features_lens: Array1<i64>,
    ) -> Result<(Array3<f32>, Array1<i64>)> {
        // Pass features and lengths directly to encoder, matching Python implementation
        let outputs = self.encoder.run(ort::inputs!(
            "audio_signal" => ort::value::Value::from_array(features)?,
            "length" => ort::value::Value::from_array(features_lens)?
        ))?;

        let encoder_out = &outputs["outputs"];
        let encoder_lens = &outputs["encoded_lengths"];

        let (shape, data) = encoder_out
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let shape_dims = shape.as_ref();
        if shape_dims.len() != 3 {
            return Err(Error::Model(format!(
                "Expected 3D encoder output, got shape: {shape_dims:?}"
            )));
        }

        let b = shape_dims[0] as usize;
        let t = shape_dims[1] as usize;
        let d = shape_dims[2] as usize;

        // Create encoder output array
        let encoder_out = Array3::from_shape_vec((b, t, d), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create encoder array: {e}")))?
            .permuted_axes((0, 2, 1))
            .to_owned();

        // The encoder already outputs in (batch, features, time) format, no transpose needed

        let (shape, lens_data) = encoder_lens
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder lengths: {e}")))?;

        let shape_dims = shape.as_ref();
        if shape_dims.len() != 1 {
            return Err(Error::Model(format!(
                "Expected 1D encoder lengths, got shape: {shape_dims:?}"
            )));
        }

        let encoder_out_lens = Array1::from_shape_vec([shape_dims[0] as usize], lens_data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create encoder lengths array: {e}")))?;

        Ok((encoder_out, encoder_out_lens))
    }

    /// Decode a single step - returns (probs, step, state)
    fn decode(
        &mut self,
        tokens: &[i32],
        prev_state: State,
        encoding: Array1<f32>,
    ) -> Result<(Array1<f32>, usize, State)> {
        let encoder_dim = encoding.len();

        // Reshape encoding to [1, encoder_dim, 1] for decoder input - matches Python: encoder_out[None, :, None]
        let frame_reshaped = encoding
            .insert_axis(Axis(0))
            .insert_axis(Axis(2))
            //.to_shape((1, encoder_dim, 1))
            //.map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
            .to_owned();

        // Use last token or blank if no tokens yet
        let last_token = tokens
            .last()
            .copied()
            .unwrap_or(self.config.blank_idx as i32);
        let targets = Array2::from_shape_vec((1, 1), vec![last_token])
            .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;

        // Run decoder_joint
        let outputs = self.decoder_joint.run(ort::inputs!(
            "encoder_outputs" => ort::value::Value::from_array(frame_reshaped)?,
            "targets" => ort::value::Value::from_array(targets)?,
            "target_length" => ort::value::Value::from_array(Array1::from_vec(vec![1i32]))?,
            "input_states_1" => ort::value::Value::from_array(prev_state.h.clone())?,
            "input_states_2" => ort::value::Value::from_array(prev_state.c.clone())?
        ))?;

        // Extract logits
        let (_, logits_data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

        // Get vocab probabilities (first vocab_size elements)
        let vocab_logits: Vec<f32> = logits_data
            .iter()
            .take(self.config.vocab_size)
            .copied()
            .collect();
        let probs = Array1::from_vec(vocab_logits);

        // Get duration step (from duration logits) - TDT specific
        let duration_logits: Vec<f32> = logits_data
            .iter()
            .skip(self.config.vocab_size)
            .copied()
            .collect();
        let step = if !duration_logits.is_empty() {
            duration_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx) // TDT uses argmax directly, not +1
                .unwrap_or(0)
        } else {
            0 // Return 0 for no step, will be handled in decoding loop
        };

        // Extract new states
        let (h_shape, h_data) = outputs["output_states_1"].try_extract_tensor::<f32>()?;
        let dims = h_shape.as_ref();
        let state_h = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            h_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to update state_h: {e}")))?;

        let (c_shape, c_data) = outputs["output_states_2"].try_extract_tensor::<f32>()?;
        let dims = c_shape.as_ref();
        let state_c = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            c_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to update state_c: {e}")))?;

        let new_state = State {
            h: state_h,
            c: state_c,
        };

        Ok((probs, step, new_state))
    }

    /// Create initial state
    fn create_state() -> State {
        State {
            h: Array3::<f32>::zeros((2, 1, 640)),
            c: Array3::<f32>::zeros((2, 1, 640)),
        }
    }

    /// Decoding function that matches the Python implementation
    pub fn decoding(
        &mut self,
        encoder_out: Array3<f32>,
        encoder_out_lens: Array1<i64>,
    ) -> Result<Vec<(Vec<i32>, Vec<usize>)>> {
        let mut results = Vec::new();

        for (encodings, &encodings_len) in
            encoder_out.axis_iter(Axis(0)).zip(encoder_out_lens.iter())
        {
            let mut prev_state = Self::create_state();
            let mut tokens: Vec<i32> = Vec::new();
            let mut timestamps: Vec<usize> = Vec::new();
            let mut t = 0;
            let mut emitted_tokens = 0;

            while t < encodings_len as usize {
                // Get encoding at time t - encodings shape is (features, time)
                let encoding = encodings.slice(ndarray::s![t, ..]).to_owned();

                let (probs, step, state) = self.decode(&tokens, prev_state.clone(), encoding)?;

                // Ensure probs shape is valid
                if probs.len() > self.config.vocab_size {
                    return Err(Error::Model(format!(
                        "Probs shape {} exceeds vocab size {}",
                        probs.len(),
                        self.config.vocab_size
                    )));
                }

                // Get token with highest probability
                let token = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(self.config.blank_idx) as i32;

                println!("{}", token);

                if token != self.config.blank_idx as i32 {
                    prev_state = state;
                    tokens.push(token);
                    timestamps.push(t);
                    emitted_tokens += 1;
                }

                // Handle time advancement - matches Python reference logic exactly
                if step > 0 {
                    t += step;
                    emitted_tokens = 0;
                } else if token == self.config.blank_idx as i32
                    || emitted_tokens == self.config.max_tokens_per_step
                {
                    t += 1;
                    emitted_tokens = 0;
                }
            }

            results.push((tokens, timestamps));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {

    use ndarray::{Array1, Array2};
    use ort::session::Session;

    #[test]
    fn test_preprocessor() {
        let mut preprocessor = Session::builder()
            .expect("")
            .with_inter_threads(1)
            .expect("")
            .with_intra_threads(1)
            .expect("")
            .commit_from_file("../../models/nemo128.onnx")
            .expect("");

        let silence = vec![0.1f32; 180 * 100];
        let silence = Array2::<f32>::from_shape_vec([1, silence.len()], silence).expect("");
        let lens = Array1::<i64>::from_shape_vec([1], vec![180 * 100]).expect("");

        let inps = ort::inputs![
            ort::value::Tensor::from_array(silence).expect(""),
            ort::value::Tensor::from_array(lens).expect("")
        ];

        let res = preprocessor.run(inps).expect("");

        let features = res["features"].try_extract_array::<f32>().expect("msg");

        let features_lens = res["features_lens"]
            .try_extract_array::<i64>()
            .expect("msg");

        println!("lens: {:?}", features_lens);
        println!("features: {:?}", features)
    }
}
