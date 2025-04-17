#include "ModelManager.h"

ModelManager::ModelManager(bool useCuda) : useCuda(useCuda){}

bool ModelManager::Load()
{
	if (useCuda) LoadCuda();
	return false;
}

void ModelManager::LoadCuda()
{
	
}
