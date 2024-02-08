use std::path::{Path, PathBuf};
use std::fs::File;
use std::io;

use candle_core::{shape::Dim, DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Dropout, VarBuilder, Activation, embedding, layer_norm, Module, Linear, linear};
use tokenizers::{PaddingParams, Tokenizer};
use serde::{Serialize, Deserialize};
use serde_json::from_reader;

/// Loads a model and tokenizer from the specified folder.
///
/// This function takes a path to a folder containing the model and tokenizer files.
/// It constructs the paths to the weight and tokenizer files, ensures they exist,
/// and then loads the weights, tokenizer, and model configuration.
///
/// # Model Structure
///
/// The `MPNetModel` structure is as follows:
///
/// ```plaintext
/// MPNetModel(
///     (embeddings): MPNetEmbeddings(
///         (word_embeddings): Embedding(30527, 768, padding_idx=1)
///         (position_embeddings): Embedding(512, 768, padding_idx=1)
///         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
///         (dropout): Dropout(p=0.1, inplace=False)
///     )
///     (encoder): MPNetEncoder(
///         (layer): ModuleList(
///             (0-11): 12 x MPNetLayer(
///                 (attention): MPNetAttention(
///                     (attn): MPNetSelfAttention(
///                         (q): Linear(in_features=768, out_features=768, bias=True)
///                         (k): Linear(in_features=768, out_features=768, bias=True)
///                         (v): Linear(in_features=768, out_features=768, bias=True)
///                         (o): Linear(in_features=768, out_features=768, bias=True)
///                         (dropout): Dropout(p=0.1, inplace=False)
///                     )
///                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
///                     (dropout): Dropout(p=0.1, inplace=False)
///              )
///                 (intermediate): MPNetIntermediate(
///                     (dense): Linear(in_features=768, out_features=3072, bias=True)
///                     (intermediate_act_fn): GELUActivation()
///                     )
///                 (output): MPNetOutput(
///                     (dense): Linear(in_features=3072, out_features=768, bias=True)
///                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
///                     (dropout): Dropout(p=0.1, inplace=False)
///                 )
///             )
///         )
///         (relative_attention_bias): Embedding(32, 12)
///     )
///     (pooler): MPNetPooler(
///         (dense): Linear(in_features=768, out_features=768, bias=True)
///         (activation): Tanh()
///     )
/// )
/// ```
///
/// # Arguments
///
/// * `path_to_check_points_folder` - A string that holds the path to the folder containing the model and tokenizer files.
///
/// # Returns
///
/// * `Ok((model, tokenizer))` - A tuple containing the loaded model and tokenizer.
/// * `Err` - An error if the specified paths do not exist, or if there is an issue loading the weights, tokenizer, or model configuration.
///
/// # How to use
///
/// ```plaintext
/// use patentpick::mpnet::load_model;
/// let (model, tokenizer) = load_model("/path/to/model/and/tokenizer").unwrap();
/// ```
pub fn load_model(path_to_check_points_folder:String) -> Result<(MPNetModel, Tokenizer)>{
    // Construct the paths to the weight and tokenizer files
    let path_to_torch_weights = Path::new(&path_to_check_points_folder).join("pytorch_model.bin");
    let path_to_safetensors = Path::new(&path_to_check_points_folder).join("model.safetensors");
    // let path_to_weight = Path::new(&path_to_check_points_folder).join("model.safetensors");

    let path_to_tokenizer = Path::new(&path_to_check_points_folder).join("tokenizer.json");
    let path_to_config = Path::new(&path_to_check_points_folder).join("config.json");

    // Ensure the paths exist
    if !path_to_safetensors.exists() || !path_to_tokenizer.exists() || !path_to_config.exists(){
        Err::<MPNetModel, _>(io::Error::new(io::ErrorKind::NotFound, "The specified paths do not exist."));
    }

    let weights = candle_core::safetensors::load(&path_to_safetensors, &Device::Cpu)?;
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let config = MPNetConfig::load(&path_to_config)?;
    let tokenizer = Tokenizer::from_file(path_to_tokenizer).unwrap();
    let model = MPNetModel::load(vb, &config)?;

    Ok((model, tokenizer))
}

#[derive(Serialize, Deserialize)]
pub struct MPNetConfig {
    _name_or_path: String,
    architectures: Vec<String>,
    attention_probs_dropout_prob: f32,
    bos_token_id: u32,
    eos_token_id: u32,
    hidden_act: String,
    hidden_dropout_prob: f32,
    hidden_size: usize,
    initializer_range: f64,
    intermediate_size: usize,
    layer_norm_eps: f64,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: u32,
    pad_token_id: u32,
    relative_attention_num_buckets: usize,
    torch_dtype: String,
    transformers_version: String,
    vocab_size: usize,
}
impl Default for MPNetConfig {
    fn default() -> Self {
        Self {
            _name_or_path: "/home/ubuntu/.cache/torch/sentence_transformers/sentence-transformers_multi-qa-mpnet-base-dot-v1/".to_string(),
            architectures: vec!["MPNetModel".to_string()],
            attention_probs_dropout_prob: 0.1,
            bos_token_id: 0,
            eos_token_id: 2,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            initializer_range: 0.02,
            intermediate_size: 3072,
            layer_norm_eps: 1e-05,
            max_position_embeddings: 514,
            model_type: "mpnet".to_string(),
            num_attention_heads: 12,
            num_hidden_layers: 12,
            pad_token_id: 1,
            relative_attention_num_buckets: 32,
            torch_dtype: "f64".to_string(),
            transformers_version: "4.11.2".to_string(),
            vocab_size: 30527
        }
    }
}
impl MPNetConfig {
    pub fn load(path: &PathBuf) -> Result<Self> {
        // Open the file in read-only mode.
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);

        // Deserialize the JSON string into an instance of `MPNetConfig`.
        let config = from_reader(reader).unwrap();

        Ok(config)
    }
}

pub struct PoolingConfig{
    word_embedding_dimension: u32,
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool

}

impl Default for PoolingConfig{
    fn default() -> Self {
        Self{
            word_embedding_dimension: 768,
            pooling_mode_cls_token: true,
            pooling_mode_mean_tokens: false,
            pooling_mode_max_tokens: false,
            pooling_mode_mean_sqrt_len_tokens: false
        }
    }
}

pub struct MPNetModel {
    embeddings: MPNetEmbeddings,
    encoder: MPNetEncoder,
    pub device: Device,
}

impl MPNetModel {
    pub fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let (embeddings, encoder) = match (
            MPNetEmbeddings::load(vb.pp("embeddings"), &config), // MPNetEmbeddings(config)
            MPNetEncoder::load(vb.pp("encoder"), &config), // MPNetEncoder(config)
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let model_type= &config.model_type {

                    if let (Ok(embeddings), Ok(encoder)) = (
                        MPNetEmbeddings::load(vb.pp(&format!("{model_type}.embeddings")), config),
                        MPNetEncoder::load(vb.pp(&format!("{model_type}.encoder")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };

        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, is_train:bool) -> Result<Tensor> {
        let embedding_output = self.embeddings .forward(input_ids, None, None, is_train)?;
        let sequence_output = self.encoder.forward(&embedding_output, is_train)?;
        Ok(sequence_output)
    }
}


struct MPNetEncoder{
    layers: Vec<MPNetLayer>,
    relative_attention_bias: Embedding,
}
impl MPNetEncoder {
    fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        // nn.ModuleList([MPNetLayer(config) for _ in range(config.num_hidden_layers)])
        let layers = (0..config.num_hidden_layers)
            .map(|index| MPNetLayer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let relative_attention_bias = embedding(config.relative_attention_num_buckets, config.num_attention_heads,  vb.pp("relative_attention_bias"))?;
        Ok(MPNetEncoder {
            layers,
            relative_attention_bias,
        })
    }

    fn forward(&self, hidden_states: &Tensor, is_train:bool) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        //for i, layer_module in enumerate(self.layer):
        //  layer_outputs = layer_module(hidden_states)

        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, is_train)?;
        }
        Ok(hidden_states)
    }
}

struct MPNetLayer {
    attention: MPNetAttention,
    intermediate: MPNetIntermediate,
    output: MPNetOutput,
}

impl MPNetLayer {
    fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let attention = MPNetAttention::load(vb.pp("attention"), config)?;
        let intermediate = MPNetIntermediate::load(vb.pp("intermediate"), config)?;
        let output = MPNetOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, is_train:bool) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, is_train)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output, is_train)?;
        Ok(layer_output)
    }
}

struct MPNetOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl MPNetOutput {
    fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor, is_train:bool) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, is_train)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct MPNetIntermediate {
    dense: Linear,
    intermediate_act: Activation,
}

impl MPNetIntermediate {
    fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        // nn.Linear(config.hidden_size, config.intermediate_size)
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: Activation::Gelu,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}
struct MPNetAttention {
    attn : MPNetSelfAttention,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl MPNetAttention {
    fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let attn = MPNetSelfAttention::load(vb.pp("attn"), config)?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            attn,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, is_train:bool) -> Result<Tensor> {
        let self_outputs = self.attn.forward(hidden_states, is_train)?;

        let dropped = self.dropout.forward(&self_outputs, is_train)?;
        let attention_output = self.layer_norm.forward(&(dropped + hidden_states)?)?;

        Ok(attention_output)
    }
}

pub struct MPNetSelfAttention{
    num_attention_heads: usize,
    attention_head_size: usize,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    dropout: Dropout,
}

impl MPNetSelfAttention{
    pub fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self>{
        if config.hidden_size % config.num_attention_heads != 0 {
            panic!("The hidden size ({}) is not a multiple of the number of attention heads ({})",
                   config.hidden_size, config.num_attention_heads);
        }

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        let dropout = Dropout::new(config.attention_probs_dropout_prob);

        let q = linear(config.hidden_size, all_head_size, vb.pp("q"))?;
        let k = linear(config.hidden_size, all_head_size, vb.pp("k"))?;
        let v = linear(config.hidden_size, all_head_size, vb.pp("v"))?;
        let o = linear(config.hidden_size, config.hidden_size,vb.pp("o"))?;

        Ok(Self{
            num_attention_heads:config.num_attention_heads,
            attention_head_size,
            q,
            k,
            v,
            o,
            dropout,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);

        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, is_train:bool) -> Result<Tensor> {
        let query_layer = self.q.forward(hidden_states)?;
        let key_layer = self.k.forward(hidden_states)?;
        let value_layer = self.v.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_probs = {candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)?};
        let attention_probs = self.dropout.forward(&attention_probs, is_train)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        let output = self.o.forward(&context_layer)?;

        Ok(output)
    }
}



pub struct MPNetEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    dropout: Dropout,
    pub padding_idx: u32,
}

impl MPNetEmbeddings {
    /// Loads the `MPNetEmbeddings` from the given `VarBuilder` and `MPNetConfig`.
    ///
    /// # Arguments
    ///
    /// * `vb` - A `VarBuilder` used to construct the embeddings.
    /// * `config` - The `MPNetConfig` that holds the configuration for the embeddings.
    ///
    /// # Returns
    ///
    /// * `Self` - The constructed `MPNetEmbeddings`.
    pub fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let word_embeddings = embedding(config.vocab_size, config.hidden_size,  vb.pp("word_embeddings"))?;
        let position_embeddings = embedding(config.max_position_embeddings, config.hidden_size, vb.pp("position_embeddings"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let padding_idx = config.pad_token_id;

        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            layer_norm,
            dropout,
            padding_idx,
        })
    }

    /// Performs a forward pass of the `MPNetEmbeddings`.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - The input tensor.
    /// * `position_ids` - The position ids tensor.
    /// * `inputs_embeds` - The inputs embeddings tensor.
    /// * `is_train` - A boolean indicating whether the model is in training mode.
    ///
    /// # Returns
    ///
    /// * `Tensor` - The result tensor after the forward pass.
    pub fn forward(&self, input_ids: &Tensor, position_ids: Option<&Tensor>, inputs_embeds: Option<&Tensor>, is_train:bool) -> Result<Tensor> {
        let position_ids = match position_ids {
            Some(ids) => ids.to_owned(),
            None => {
                if Option::is_some(&inputs_embeds){
                    let position_ids = self.create_position_ids_from_input_embeds(inputs_embeds.unwrap())?; //
                    position_ids
                } else {
                    let position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)?;
                    position_ids
                }
            }
        };

        let inputs_embeds : Tensor = match inputs_embeds {
            Some(embeds) => embeds.to_owned(),
            None => {
                // self.word_embeddings(input_ids)
                let embeds = self.word_embeddings.forward(input_ids)?;
                embeds
            }
        };
        let mut embeddings = inputs_embeds;


        if let Some(position_embeddings) = &self.position_embeddings {
            // embeddings + self.position_embeddings(position_ids)
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }

        // self.LayerNorm(embeddings)
        let embeddings = self.layer_norm.forward(&embeddings)?;
        // self.dropout(embeddings)
        let embeddings = self.dropout.forward(&embeddings, is_train)?;

        Ok(embeddings)

    }

    /// Creates position ids from the input embeddings.
    ///
    /// # Arguments
    ///
    /// * `input_embeds` - The input embeddings tensor.
    ///
    /// # Returns
    ///
    /// * `Tensor` - The position ids tensor.
    pub fn create_position_ids_from_input_embeds(&self, input_embeds: &Tensor) -> Result<Tensor> {
        // In candle, we use dims3() for getting the size of a 3-dimensional tensor
        let input_shape = input_embeds.dims3()?;
        let seq_length = input_shape.1;

        let mut position_ids = Tensor::arange(
            self.padding_idx + 1,
            seq_length as u32 + self.padding_idx + 1,
            &Device::Cpu,
        )?;

        position_ids = position_ids
            .unsqueeze(0)?
            .expand((input_shape.0, input_shape.1))?;
        Ok(position_ids)
    }
}

/// Creates position ids from the input ids.
///
/// # Arguments
///
/// * `input_ids` - The input ids tensor.
/// * `padding_idx` - The padding index.
///
/// # Returns
///
/// * `Tensor` - The position ids tensor.
pub fn create_position_ids_from_input_ids(input_ids: &Tensor, padding_idx: u32) -> Result<Tensor> {
    // println!("input_ids: {:?}", input_ids.to_vec2::<u32>()?);
    let mask = input_ids.ne(padding_idx)?.to_dtype(input_ids.dtype())?;
    // println!("mask: {:?}", mask.to_vec2::<u8>()?);

    let incremental_indices = cumsum(&mask, 1).unwrap();
    let incremental_indices = incremental_indices.broadcast_add(&Tensor::new(&[padding_idx], input_ids.device())?)?;

    Ok(incremental_indices)
}


/// Returns the cumulative sum of elements of input in the dimension dim.
///
/// [https://pytorch.org/docs/stable/generated/torch.cumsum.html](https://pytorch.org/docs/stable/generated/torch.cumsum.html)
pub fn cumsum<D: Dim>(input: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(input.shape(), "cumsum")?;
    let dim_size = input.dim(dim)?;

    let mut tensors = Vec::with_capacity(dim_size);

    let mut a = input.clone();
    for i in 0..dim_size {
        if i > 0 {
            a = a.narrow(dim, 1, dim_size - i)?;
            let b = input.narrow(dim, 0, dim_size - i)?;
            a = (a + b)?;
        }
        tensors.push(a.narrow(dim, 0, 1)?);
    }
    let cumsum = Tensor::cat(&tensors, dim)?;
    Ok(cumsum)
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::fmt::Pointer;
    use super::*;
    #[test]
    fn test_transpose_for_scores() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let config = MPNetConfig::default();

        let mpnet_self_attention = MPNetSelfAttention::load(vb, &config).unwrap();
        let xs = Tensor::randn(0f32, 1f32,(1, 2, config.hidden_size), &Device::Cpu).unwrap();

        let result = mpnet_self_attention.transpose_for_scores(&xs).unwrap();
        let expected_shape = vec![1, config.num_attention_heads, 2, config.hidden_size/config.num_attention_heads];
        assert_eq!(result.shape().dims().to_vec(), expected_shape);
    }

    #[test]
    fn test_self_attention_forward() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let config = MPNetConfig::default();

        let mpnet_self_attention = MPNetSelfAttention::load(vb, &config).unwrap();
        let hidden_states = Tensor::randn(0f32, 1f32,(1, 2, config.hidden_size), &Device::Cpu).unwrap();

        let result = mpnet_self_attention.forward(&hidden_states, false);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dims().to_vec(), &[1, 2, config.hidden_size]);
    }

    #[test]
    fn test_attention_forward() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let config = MPNetConfig::default();

        let mpnet_attention = MPNetAttention::load(vb, &config).unwrap();

        let hidden_states = Tensor::randn(0f32, 1f32,(1, 2, config.hidden_size), &Device::Cpu).unwrap();

        let result = mpnet_attention.forward(&hidden_states, false);

        // Check if the output is Ok and if the size is as expected
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dims().to_vec(), &[1, 2, config.hidden_size]);
    }

    #[test]
    fn test_intermediate_forward() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let config = MPNetConfig::default();

        let mpnet_intermediate = MPNetIntermediate::load(vb, &config).unwrap();
        let hidden_states = Tensor::randn(0f32, 1f32,(1, 2, config.hidden_size), &Device::Cpu).unwrap();

        let result = mpnet_intermediate.forward(&hidden_states);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dims().to_vec(), &[1, 2, config.intermediate_size]);
    }
    #[test]
    fn test_output_forward() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let config = MPNetConfig::default();

        let mpnet_output = MPNetOutput::load(vb, &config).unwrap();
        let hidden_states = Tensor::randn(0f32, 1f32,(1, 2, config.intermediate_size), &Device::Cpu).unwrap();
        let input_tensor = Tensor::randn(0f32, 1f32,(1, 2, config.hidden_size), &Device::Cpu).unwrap();

        let result = mpnet_output.forward(&hidden_states, &input_tensor, false);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dims().to_vec(), &[1, 2, config.hidden_size]);
    }

    #[test]
    fn test_mpnet_layer_forward() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let config = MPNetConfig::default();

        let mpnet_layer = MPNetLayer::load(vb, &config).unwrap();
        let hidden_states = Tensor::randn(0f32, 1f32,(10, 32, config.hidden_size), &Device::Cpu).unwrap();

        let result = mpnet_layer.forward(&hidden_states, false);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dims().to_vec(), &[10, 32, config.hidden_size]);
    }

    #[test]
    fn test_mpnet_encoder_forward() {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let config = MPNetConfig::default();

        let encoder = MPNetEncoder::load(vb, &config).unwrap();
        let hidden_states = Tensor::randn(0f32, 1f32,(10, 32, config.hidden_size), &Device::Cpu).unwrap();

        let result = encoder.forward(&hidden_states, false);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dims().to_vec(), &[10, 32, config.hidden_size]);
    }
}
