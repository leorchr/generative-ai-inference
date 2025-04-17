#pragma once
#include "ModelManager.h"
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>


class T5Manager : public ModelManager {
public:
	T5Manager(bool useCuda = false);
        
    virtual bool Load() override;
    virtual void Run() override;


private:
	
	sentencepiece::SentencePieceProcessor sp;

	Ort::Env env;
	Ort::SessionOptions sessionOptions;

	Ort::Session encoder_session{nullptr};	
	Ort::Session decoder_session{nullptr};

	Ort::MemoryInfo memory_info{ nullptr };     // Used to allocate memory for input

};
