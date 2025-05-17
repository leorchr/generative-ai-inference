#include "M2M100Manager.h"

#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <chrono>
#include <iostream>

M2M100Manager::M2M100Manager(bool useCuda) : ModelManager(useCuda){}

bool M2M100Manager::Load()
{
	ModelManager::Load();
	const auto start{std::chrono::steady_clock::now()};
	
	const auto status = sp.Load("./ressources/m2m100_418M/sentencepiece.bpe.model");
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return false;
	}
	
	int vocab_size = sp.GetPieceSize();
	std::cout << "Taille du vocab: " << vocab_size << std::endl;


	/*for (int id = 0; id < vocab_size; id++) {
		std::string token = sp.IdToPiece(id);
		std::cout << id << " : " << token << std::endl;
	}*/

	
	// Check the providers

	/*
	auto providers = Ort::GetAvailableProviders();
	for (auto provider : providers) {
		std::cout << provider << std::endl;
	}
	*/

	env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
	
	try{
		// Model path is const wchar_t*
		encoder_session = Ort::Session(env, L"./ressources/m2m100_418M/encoder_model.onnx", sessionOptions);
		decoder_session = Ort::Session(env, L"./ressources/m2m100_418M/decoder_model.onnx", sessionOptions);
		decoder_wp_session = Ort::Session(env, L"./ressources/m2m100_418M/decoder_with_past_model.onnx", sessionOptions);
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}

	try {
		memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}


	const auto end{std::chrono::steady_clock::now()};
	
	const std::chrono::duration<float> elapsedSeconds{end - start};
	std::cout << "Load time: " << elapsedSeconds.count() << "s\n";

	return true;
}

void M2M100Manager::Run(std::string inputText)
{
	const auto startInference{std::chrono::steady_clock::now()};
	
	Ort::AllocatorWithDefaultOptions allocator = Ort::AllocatorWithDefaultOptions();
	std::vector<int> generated_token_ids;
	
	// Encoder part
	
	std::string input_text = "__en__ Hello World </s>";

	std::vector<int> input_text_ids = sp.EncodeAsIds(input_text);
	//int source_language_id = sp.PieceToId("en");
	//int eos_id = sp.PieceToId("</s>");

	
	std::vector<int64_t> input_ids{};
	input_ids.insert(input_ids.end(), input_text_ids.begin(), input_text_ids.end());
	//input_ids.insert(input_ids.end(), input_text_ids.begin(), input_text_ids.end());
	//input_ids.push_back(eos_id);

	std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

	Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
	memory_info,
	input_ids.data(),
	input_ids.size(),
	input_shape.data(),
	input_shape.size()
	);


	std::vector<int64_t> encoder_attention_mask(input_ids.size(), 1);

	std::vector<int64_t> encoder_attention_mask_shape = {1, static_cast<int>(encoder_attention_mask.size())};

	Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
	memory_info,
	encoder_attention_mask.data(),
	encoder_attention_mask.size(),
	encoder_attention_mask_shape.data(),
	encoder_attention_mask_shape.size()
	);
	
	size_t encoder_input_count = encoder_session.GetInputCount();
	std::vector<std::string> encoder_input_names_str = GetInputNames(&encoder_session,(int)encoder_input_count, allocator);
	std::vector<const char*> encoder_input_names;

	encoder_input_names.reserve(encoder_input_names_str.size());
	for (auto& s : encoder_input_names_str) {
		encoder_input_names.emplace_back(s.c_str());
	}
	
	size_t encoder_output_count = encoder_session.GetOutputCount();
	std::vector<std::string> encoder_output_names_str = GetOutputNames(&encoder_session, (int)encoder_output_count, allocator);

	std::vector<const char*> encoder_output_names;

	encoder_output_names.reserve(encoder_output_names_str.size());
	for (auto& s : encoder_output_names_str) {
		encoder_output_names.emplace_back(s.c_str());
	}

	Ort::Value values[2] = {std::move(input_tensor), std::move(attention_mask_tensor)};

	std::vector<Ort::Value> output_tensors;
	try {
		output_tensors = encoder_session.Run(
			Ort::RunOptions{nullptr},
			encoder_input_names.data(), values, encoder_input_count,
			encoder_output_names.data(), encoder_output_count
		);
	} catch (const Ort::Exception& e) {
		std::cerr << "Erreur ONNX Runtime: " << e.what() << std::endl;
	}


	// Decoder part
	
	Ort::Value& encoder_hidden_states = output_tensors[0];

	std::vector<int> decoder_input = sp.EncodeAsIds("</s>");
	std::vector<int64_t> decoder_input_ids{};
	decoder_input_ids.insert(decoder_input_ids.end(), decoder_input.begin(), decoder_input.end());
	
	std::vector<int64_t> decoder_input_shape = {1, static_cast<int64_t>(decoder_input_ids.size())};
	Ort::Value decoder_input_tensor = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		decoder_input_ids.data(),
		decoder_input_ids.size(),
		decoder_input_shape.data(),
		decoder_input_shape.size()
	);

	Ort::Value encoder_attention_mask_tensor_for_decoder = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		encoder_attention_mask.data(),
		encoder_attention_mask.size(),
		encoder_attention_mask_shape.data(),
		encoder_attention_mask_shape.size()
		);
	

	
	Ort::Value decoder_inputs[3] = { std::move(encoder_attention_mask_tensor_for_decoder),std::move(decoder_input_tensor), std::move(encoder_hidden_states) };
	
	size_t decoder_input_count = decoder_session.GetInputCount();
	std::vector<std::string> decoder_input_names_str = GetInputNames(&decoder_session,(int)decoder_input_count, allocator);
	std::vector<const char*> decoder_input_names;
	decoder_input_names.reserve(decoder_input_names_str.size());
	for (auto& s : decoder_input_names_str) {
		decoder_input_names.emplace_back(s.c_str());
	}
	
	
	size_t decoder_output_count = decoder_session.GetOutputCount();
	std::vector<std::string> decoder_output_names_str = GetOutputNames(&decoder_session, (int)decoder_output_count, allocator);
	std::vector<const char*> decoder_output_names;
	decoder_output_names.reserve(decoder_output_names_str.size());
	for (auto& s : decoder_output_names_str) {
		decoder_output_names.emplace_back(s.c_str());
	}

	std::vector<Ort::Value> decoder_outputs;
	try {
		decoder_outputs = decoder_session.Run(
			Ort::RunOptions{nullptr},
				decoder_input_names.data(),
				decoder_inputs, 
				decoder_input_count,
				decoder_output_names.data(),
				decoder_output_count
		);
	} catch (const Ort::Exception& e) {
		std::cerr << "Erreur ONNX Runtime: " << e.what() << std::endl;
	}

	// Get Generated Token

	int64_t nextToken = GetNextToken(decoder_outputs[0]);
	//int64_t nextToken = sp.PieceToId("__fr__");
	generated_token_ids.push_back(static_cast<int>(nextToken));
	
	
	
	// Decoder with past part

	size_t decoder_wp_input_count = decoder_wp_session.GetInputCount();
    std::vector<std::string> decoder_wp_input_names_str = GetInputNames(&decoder_wp_session,(int)decoder_wp_input_count, allocator);
    std::vector<const char*> decoder_wp_input_names;
    decoder_wp_input_names.reserve(decoder_wp_input_names_str.size());
    for (auto& s : decoder_wp_input_names_str) {
    	decoder_wp_input_names.emplace_back(s.c_str());
    }


    size_t decoder_wp_output_count = decoder_wp_session.GetOutputCount();
    std::vector<std::string> decoder_wp_output_names_str = GetOutputNames(&decoder_wp_session, (int)decoder_wp_output_count, allocator);
    std::vector<const char*> decoder_wp_output_names;
    decoder_wp_output_names.reserve(decoder_wp_output_names_str.size());
    for (auto& s : decoder_wp_output_names_str) {
    	decoder_wp_output_names.emplace_back(s.c_str());
    }

	std::vector<int64_t> decoder_wp_input_shape = {1, 1};


	// Decoder with past first pass

	Ort::Value decoder_wp_input_tensor = Ort::Value::CreateTensor<int64_t>(
			memory_info,
			&nextToken,
			1,
			decoder_wp_input_shape.data(),
			decoder_wp_input_shape.size()
		);

	Ort::Value encoder_attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
		memory_info,
		encoder_attention_mask.data(),
		encoder_attention_mask.size(),
		encoder_attention_mask_shape.data(),
		encoder_attention_mask_shape.size()
		);
	
	std::vector<Ort::Value> decoder_wp_inputs;
	decoder_wp_inputs.reserve(decoder_wp_input_count);

	decoder_wp_inputs.emplace_back(std::move(encoder_attention_mask_tensor));
	decoder_wp_inputs.emplace_back(std::move(decoder_wp_input_tensor));

	for (size_t i = 2; i < decoder_wp_input_count; i++)
	{
		decoder_wp_inputs.emplace_back(std::move(decoder_outputs[i-1]));
	}

	std::vector<Ort::Value> decoder_wp_outputs;
	
	try {
	decoder_wp_outputs = decoder_wp_session.Run(
		Ort::RunOptions{nullptr},
		decoder_wp_input_names.data(),
		decoder_wp_inputs.data(),
		decoder_wp_input_count,
		decoder_wp_output_names.data(),
		decoder_wp_output_count
	);
	} catch (const Ort::Exception& e) {
		std::cerr << "Erreur ONNX Runtime: " << e.what() << std::endl;
	}

	nextToken = GetNextToken(decoder_wp_outputs[0]);
	generated_token_ids.push_back(static_cast<int>(nextToken));
	
	
	int max_length = 50;
    for (int step = 0; step < max_length; step++) {
    	
        decoder_wp_input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            &nextToken,
            1,
            decoder_wp_input_shape.data(),
            decoder_wp_input_shape.size()
        );

    	decoder_wp_inputs[1] = std::move(decoder_wp_input_tensor);

    	for (size_t i = 0; i < (decoder_wp_outputs.size() - 1) / 2; i ++)
    	{
    		decoder_wp_inputs[2+i*4] = std::move(decoder_wp_outputs[i*2+1]);
    		decoder_wp_inputs[3+i*4] = std::move(decoder_wp_outputs[i*2+2]);
    	}	

        try
        {
	        decoder_wp_outputs = decoder_wp_session.Run(
				Ort::RunOptions{nullptr},
				decoder_wp_input_names.data(),
				decoder_wp_inputs.data(),
				decoder_wp_input_count,
				decoder_wp_output_names.data(),
				decoder_wp_output_count
			);
        } catch (const Ort::Exception& e){
			std::cerr << "Erreur ONNX Runtime: " << e.what() << std::endl;
        }
        
    	nextToken = GetNextToken(decoder_wp_outputs[0]);
    	generated_token_ids.push_back(static_cast<int>(nextToken));
        
        if (nextToken == sp.PieceToId("</s>")) {
            break;
        }
   }


	

	const auto endInference{std::chrono::steady_clock::now()};
	
	const std::chrono::duration<float> elapsedSecondsInference{endInference - startInference};
	std::cout << "Inference duration : " << elapsedSecondsInference.count() << "s\n";

	
    // ----- Partie 5 : Détokénisation de la sortie -----
    std::string output_text = sp.DecodeIds(generated_token_ids);
    std::cout << "Texte généré: " << output_text << std::endl;
}

std::vector<std::string> M2M100Manager::GetInputNames(Ort::Session* session, int inputCount, Ort::AllocatorWithDefaultOptions allocator)
{
	std::vector<std::string> input_name_strings;
	
	input_name_strings.reserve(inputCount);
	
	for (size_t i = 0; i < inputCount; i++)
	{
		auto name_ptr = session->GetInputNameAllocated(i, allocator);
		input_name_strings.emplace_back(name_ptr.get());
	}

	return input_name_strings;
}

std::vector<std::string> M2M100Manager::GetOutputNames(Ort::Session* session, int outputCount, Ort::AllocatorWithDefaultOptions allocator)
{
	std::vector<std::string> output_name_strings;
	
	output_name_strings.reserve(outputCount);
	
	for (size_t i = 0; i < outputCount; i++)
	{
		auto name_ptr = session->GetOutputNameAllocated(i, allocator);
		output_name_strings.emplace_back(name_ptr.get());
	}

	return output_name_strings;
}

int64_t M2M100Manager::GetNextToken(Ort::Value& logitTensor)
{
	auto logits_info = logitTensor.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> logits_shape = logits_info.GetShape();
	
	int64_t current_length = logits_shape[1];
	int64_t vocab_size = logits_shape[2];
 
	float* logits_data = logitTensor.GetTensorMutableData<float>();
	
	size_t offset = (current_length - 1) * vocab_size;
	int64_t next_token = 0;
	float max_logit = -std::numeric_limits<float>::infinity();
	for (int64_t j = 0; j < vocab_size; j++) {
		float logit = logits_data[offset + j];
		if (logit > max_logit) {
			max_logit = logit;
			next_token = j;
		}
	}

	return next_token;
}