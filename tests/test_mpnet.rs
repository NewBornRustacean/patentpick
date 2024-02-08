use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder,  Module};

use patentpick::mpnet::{ MPNetEmbeddings, MPNetConfig, create_position_ids_from_input_ids, cumsum, load_model};


#[test]
fn test_model_load() ->Result<()>{
    let HIDDEN_SIZE = 768 as usize;
    fn round_to_decimal_places(n: f32, places: u32) -> f32 {
        let multiplier: f32 = 10f32.powi(places as i32);
        (n * multiplier).round() / multiplier
    }

    let path_to_checkpoints_folder = "D:/RustWorkspace/patentpick/resources/checkpoints/AI-Growth-Lab_PatentSBERTa".to_string();
    let (model, mut tokenizer) = load_model(path_to_checkpoints_folder).unwrap();

    let input_ids = &[[0u32, 30500, 232, 328, 740, 1140, 12695, 69, 30237, 1588, 2]];
    let input_ids = Tensor::new(input_ids, &model.device).unwrap();

    let output = model.forward(&input_ids, false)?;

    let expected_shape = [1, 11, HIDDEN_SIZE];

    assert_eq!(output.shape().dims(), &expected_shape);

    Ok(())

}

#[test]
fn test_create_position_ids_from_input_embeds() -> candle_core::Result<()> {


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

#[test]
fn test_create_position_ids_from_input_ids() -> candle_core::Result<()> {

    let config = MPNetConfig::default();

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let embeddings_module = MPNetEmbeddings::load(vb, &config).unwrap();

    let input_ids = &[[0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]];
    let input_ids = Tensor::new(input_ids, &Device::Cpu)?;

    let position_ids = create_position_ids_from_input_ids(&input_ids, embeddings_module.padding_idx)?;

    let expected_tensor = &[[2u8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]];

    println!("position_ids: {:?}", position_ids);
    // assert_eq!(position_ids.to_vec2::<u8>()?, expected_tensor);
    Ok(())
}

#[test]
fn test_cumsum() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let a = Tensor::new(&[1u32, 2, 3], &device)?;
    let b = cumsum(&a, 0)?;
    assert_eq!(b.to_vec1::<u32>()?, &[1, 3, 6]);
    Ok(())
}

