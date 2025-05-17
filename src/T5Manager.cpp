#include "T5Manager.h"

#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <chrono>
#include <iostream>

T5Manager::T5Manager(bool useCuda) : ModelManager(useCuda) {}

bool T5Manager::Load()
{
	ModelManager::Load();
	const auto start{std::chrono::steady_clock::now()};
	
	const auto status = sp.Load("./ressources/t5_base/spiece.model");
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return false;
	}
	
	//int vocab_size = sp.GetPieceSize();
	//std::cout << "Taille du vocab: " << vocab_size << std::endl;
	
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
		encoder_session = Ort::Session(env, L"./ressources/t5_base/t5-encoder-12.onnx", sessionOptions);
		decoder_session = Ort::Session(env, L"./ressources/t5_base/t5-decoder-with-lm-head-12.onnx", sessionOptions);
		//std::cout << "Nombre d'entrées: " << encoder_session.GetInputCount() << std::endl;
		//std::cout << "Nombre d'entrées: " << decoder_session.GetInputCount() << std::endl;
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
	std::cout << "\nThis models shows a translation from english to german." << std::endl;
	std::cout << "Load time: " << elapsedSeconds.count() << "s\n\n";

	return true;
}

void T5Manager::Run(std::string inputText)
{
	const auto startInference{std::chrono::steady_clock::now()};
	
	std::string input_text = "translate English to German: " + inputText;

	std::vector<int> ids = sp.EncodeAsIds(input_text);
	std::vector<int64_t> input_ids(ids.begin(), ids.end());

	std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

	Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
	memory_info,
	input_ids.data(),
	input_ids.size(),
	input_shape.data(),
	input_shape.size()
	);

	const char* encoder_input_names[] = {"input_ids"};
	const char* encoder_output_names[] = {"hidden_states"};

	auto output_tensors = encoder_session.Run(
		Ort::RunOptions{nullptr},
		encoder_input_names, &input_tensor, 1,   // 1 entrée
		encoder_output_names, 1                  // 1 sortie
	);
	
	
	Ort::Value& encoder_hidden_states = output_tensors[0];

	int start_token = sp.PieceToId("<pad>");
	std::vector<int64_t> decoder_input_ids = { start_token };
	std::vector<int> generated_token_ids;

	
	std::array<Ort::Value, 2> decoder_inputs = { Ort::Value(), std::move(encoder_hidden_states) };
    
    int max_length = 50;  // nombre maximal de tokens à générer
    for (int step = 0; step < max_length; step++) {
        // Préparer le tenseur d'entrée pour le décodeur : forme [1, current_length]
        std::vector<int64_t> decoder_input_shape = {1, static_cast<int64_t>(decoder_input_ids.size())};
        Ort::Value decoder_input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            decoder_input_ids.data(),
            decoder_input_ids.size(),
            decoder_input_shape.data(),
            decoder_input_shape.size()
        );
        
        // Le décodeur reçoit deux entrées :
        // - "input_ids" : la séquence de tokens générée jusqu'à présent
        // - "encoder_hidden_states" : la représentation obtenue par l'encodeur
        const char* decoder_input_names[] = {"input_ids", "encoder_hidden_states"};
        // On prépare un tableau d'inputs pour le décodeur

    	decoder_inputs[0] = std::move(decoder_input_tensor);
    	
        const char* decoder_output_names[] = {"hidden_states"};
        
        // Exécuter l'inférence du décodeur
        auto decoder_outputs = decoder_session.Run(
            Ort::RunOptions{nullptr},
            decoder_input_names,
            decoder_inputs.data(),
            2,
            decoder_output_names,
            1
        );
        
        // On récupère les logits de sortie (supposés de forme [1, seq_len, vocab_size])
        Ort::Value& logits_tensor = decoder_outputs[0];
        auto logits_info = logits_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> logits_shape = logits_info.GetShape();
        // On attend une forme de type [1, current_length, vocab_size]
        int64_t current_length = logits_shape[1];
        int64_t vocab_size = logits_shape[2];

        // On récupère un pointeur sur les données de type float (les logits)
        float* logits_data = logits_tensor.GetTensorMutableData<float>();
        // On choisit le token avec la valeur max parmi les logits du dernier token généré.
        // Calculer l'offset dans le tenseur pour le dernier token (indice current_length - 1)
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
        
        // Ajouter le token généré à la séquence du décodeur
        decoder_input_ids.push_back(next_token);
        generated_token_ids.push_back(static_cast<int>(next_token));
        
        // Vérifier la condition d'arrêt (par exemple, token de fin "</s>")
        if (next_token == sp.PieceToId("</s>")) {
            break;
        }
    }

	const auto endInference{std::chrono::steady_clock::now()};
	
	const std::chrono::duration<float> elapsedSecondsInference{endInference - startInference};
	std::cout << "Inference duration : " << elapsedSecondsInference.count() << "s\n";

	
    // ----- Partie 5 : Détokénisation de la sortie -----
    std::string output_text = sp.DecodeIds(generated_token_ids);
    std::cout << "Generated Text : " << output_text << std::endl;
}