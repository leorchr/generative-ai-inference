#pragma once
#include "ModelManager.h"
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>


class M2M100Manager : public ModelManager {
public:
	M2M100Manager(bool useCuda = false);
        
    virtual bool Load() override;
	std::vector<std::string> GetInputNames(Ort::Session* session, int inputCount, Ort::AllocatorWithDefaultOptions allocator);
	std::vector<std::string> GetOutputNames(Ort::Session* session, int outputCount, Ort::AllocatorWithDefaultOptions allocator);
	int64_t GetNextToken(Ort::Value& logitTensor);
    virtual void Run() override;


private:
	
	sentencepiece::SentencePieceProcessor sp;

	Ort::Env env;
	Ort::SessionOptions sessionOptions;

	Ort::Session encoder_session{nullptr};	
	Ort::Session decoder_session{nullptr};
	Ort::Session decoder_wp_session{nullptr};

	Ort::MemoryInfo memory_info{ nullptr };     // Used to allocate memory for input

};
