#include "Engine.h"

#include "Window.h"
#include "RendererOGL.h"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

Engine* Engine::instance = nullptr;

void Engine::run()
{
	context = std::make_unique<Context>();

	context->window = std::make_unique<Window>();
	context->window->initialize();

	context->renderer = std::make_unique<RendererOGL>();
	context->renderer->initialize(context->window.get());
}

void Engine::close()
{
	context->renderer->close();
	context->window->close();
}

void Engine::update()
{
	try{
		while (!context->window->shouldClose()) {


			context->renderer->render();

			// Pour l'instant doit inclure glfw.h
			glfwPollEvents();  // Traiter les événements (entrée utilisateur, etc.)
		}
	}
	catch (std::exception e) {
		spdlog::error("{}", e.what());
	}
}

Context& Engine::getContext()
{
	if (!context)
	{
		context = std::make_unique<Context>();
	}
	return *context;
}

Engine& Engine::getInstance()
{
	if (!instance)
		instance = new Engine();

	return *instance;
}