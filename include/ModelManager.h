#pragma once
#include <string>

class ModelManager
{
public:
	ModelManager(bool useCuda = false);
	virtual ~ModelManager() = default;
	
	virtual bool Load() = 0;
	virtual void Run(std::string inputText) = 0;
	static void LoadCuda();

private:
	bool useCuda{};
};
