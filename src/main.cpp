#include "M2M100Manager.h"
#include "T5Manager.h"

int main(int argc, char** argv)
{
	ModelManager* manager = new T5Manager();
	if (!manager->Load())
	{
		return 1;
	};
	manager->Run();
	return 0;
}