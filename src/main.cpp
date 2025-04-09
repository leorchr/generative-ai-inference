#include <iostream>

#include "Engine.h"
#include "AIManager.h"

int main(int argc, char** argv)
{
	AIManager manager = AIManager();
	manager.Load();
	manager.Run();
	// Engine::getInstance().run();
	// Engine::getInstance().update();
	// Engine::getInstance().close();
	return 0;
}