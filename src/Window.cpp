#include "Window.h"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

Window::Window() : glfwWindow(nullptr), width(1920), height(1080)
{
}

bool Window::initialize()
{
	if (!glfwInit())
	{
		spdlog::error("{} Unable to initialize glfw", __FILENAME__);
		return false;
	}

	glfwWindow = glfwCreateWindow(width, height, "GameEngine", NULL, NULL);
	if (!glfwWindow)
	{
		spdlog::error("{} Failed to create window", __FILENAME__);
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(glfwWindow);

	return true;
}

void Window::close()
{
	if (glfwWindow != nullptr) glfwDestroyWindow(glfwWindow);
	glfwTerminate();
}

void Window::update()
{
	while (!glfwWindowShouldClose(glfwWindow))
	{
		glfwSwapBuffers(glfwWindow);
		glfwPollEvents();
	}
}

bool Window::shouldClose() const
{
	return glfwWindowShouldClose(glfwWindow);
}
