use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder,  Module};

use patentpick::mpnet::{ MPNetEmbeddings, MPNetConfig, create_position_ids_from_input_ids, cumsum, load_model, get_embeddings, normalize_l2};


#[test]
fn test_model_load() ->Result<()>{
    let HIDDEN_SIZE = 768 as usize;
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
fn test_get_embeddings() ->Result<()>{
    let path_to_checkpoints_folder = "D:/RustWorkspace/patentpick/resources/checkpoints/AI-Growth-Lab_PatentSBERTa".to_string();

    let (model, mut tokenizer) = load_model(path_to_checkpoints_folder).unwrap();

    let sentences = vec![
        "an invention that targets GLP-1",
        "new chemical that targets glucagon like peptide-1 ",
        "de novo chemical that targets GLP-1",
        "invention about GLP-1 receptor",
        "new chemical synthesis for glp-1 inhibitors",
        "It feels like I'm in America",
        "It's rainy. all day long.",
    ];
    let n_sentences = sentences.len();

    let embeddings = get_embeddings(&model, &tokenizer, &sentences).unwrap();

    // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    println!("_n_sentence, n_tokens, _hidden_size: {:?}, {:?}, {:?}", _n_sentence, n_tokens, _hidden_size);

    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    let l2norm_embeds = normalize_l2(&embeddings).unwrap();
    println!("pooled embeddings {:?}", l2norm_embeds.shape());

    let mut similarities = vec![];
    for i in 0..n_sentences {
        let e_i = l2norm_embeds.get(i)?;
        for j in (i + 1)..n_sentences {
            let e_j = l2norm_embeds.get(j)?;
            let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
            let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

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

#[test]
fn test_normalize_l2() {
    let v = Tensor::new(&[[1f32, 2.0, 3.0, 4.0, 5.0]], &Device::Cpu).unwrap();
    let result = normalize_l2(&v).unwrap();

    let sum_of_squares = result.sqr().unwrap().sum(1).unwrap();
    assert!((sum_of_squares.get(0).unwrap().to_vec0::<f32>().unwrap() - 1.0f32).abs() < 1e-5, "The tensor is not normalized");
}