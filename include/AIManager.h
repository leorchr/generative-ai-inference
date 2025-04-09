#pragma once
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>


class AIManager {
public:
	AIManager();
        
    bool Load();
    void Run();


private:
	
	sentencepiece::SentencePieceProcessor sp;

	Ort::Env env;
	Ort::SessionOptions sessionOptions;

	Ort::Session encoder_session{nullptr};	
	Ort::Session decoder_session{nullptr};

	Ort::MemoryInfo memory_info{ nullptr };     // Used to allocate memory for input

};
