use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Dropout, VarBuilder, embedding, layer_norm, Module};
use serde::Deserialize;

// MPNetModel(
//     (embeddings): MPNetEmbeddings(
//         (word_embeddings): Embedding(30527, 768, padding_idx=1)
//         (position_embeddings): Embedding(512, 768, padding_idx=1)
//         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
//         (dropout): Dropout(p=0.1, inplace=False)
//     )
//     (encoder): MPNetEncoder(
//         (layer): ModuleList(
//             (0-11): 12 x MPNetLayer(
//                 (attention): MPNetAttention(
//                     (attn): MPNetSelfAttention(
//                         (q): Linear(in_features=768, out_features=768, bias=True)
//                         (k): Linear(in_features=768, out_features=768, bias=True)
//                         (v): Linear(in_features=768, out_features=768, bias=True)
//                         (o): Linear(in_features=768, out_features=768, bias=True)
//                         (dropout): Dropout(p=0.1, inplace=False)
//                     )
//                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
//                     (dropout): Dropout(p=0.1, inplace=False)
//              )
//                 (intermediate): MPNetIntermediate(
//                     (dense): Linear(in_features=768, out_features=3072, bias=True)
//                     (intermediate_act_fn): GELUActivation()
//                     )
//                 (output): MPNetOutput(
//                     (dense): Linear(in_features=3072, out_features=768, bias=True)
//                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
//                     (dropout): Dropout(p=0.1, inplace=False)
//                 )
//             )
//         )
//         (relative_attention_bias): Embedding(32, 12)
//     )
//     (pooler): MPNetPooler(
//         (dense): Linear(in_features=768, out_features=768, bias=True)
//         (activation): Tanh()
//     )
// )
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
    intermediate_size: u32,
    layer_norm_eps: f64,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: u32,
    num_hidden_layers: u32,
    pad_token_id: u32,
    relative_attention_num_buckets: u32,
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
    let mask = input_ids.ne(padding_idx)?;
    let incremental_indices = (mask.cumsum(1) * mask).unwrap();

    // let position_ids = incremental_indices + padding_idx;
    let incremental_indices = incremental_indices.broadcast_add(&Tensor::new(&[padding_idx], input_ids.device())?)?;

    Ok(incremental_indices)
}


pub fn load_model(){
    let safetensor_path = "D:/RustWorkspace/patentpick/resources/checkpoints/AI-Growth-Lab_PatentSBERTa/model.safetensors".to_string();
    // let api = Api::new().unwrap();
    // let repo = api.model();

    let weights = candle_core::safetensors::load(safetensor_path, &Device::Cpu);
    println!("done!");

}



# [cfg (test)]
mod tests {
    use candle_nn::VarMap;
    use super::*;
    #[test]
    fn test_model_load() {
        load_model();

    }

    #[test]
    fn test_create_position_ids_from_input_embeds() -> Result<()> {


        let config = MPNetConfig::default();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let embeddings_module = MPNetEmbeddings::load(vb, &config).unwrap();

        let input_embeds = Tensor::randn(0f32, 1f32, (2, 4, 30), &Device::Cpu).unwrap();
        let position_ids = embeddings_module.create_position_ids_from_input_embeds(&input_embeds);

        let expected_tensor: &[[u32; 4]; 2] = &[
            [0 + embeddings_module.padding_idx + 1, 1 + embeddings_module.padding_idx + 1, 2 + embeddings_module.padding_idx + 1, 3 + embeddings_module.padding_idx + 1,],
            [0 + embeddings_module.padding_idx + 1, 1 + embeddings_module.padding_idx + 1, 2 + embeddings_module.padding_idx + 1, 3 + embeddings_module.padding_idx + 1,]
        ];

        assert_eq!(position_ids.unwrap().to_vec2::<u32>()?, expected_tensor);
        Ok(())

    }
}
