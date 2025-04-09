#include "RendererOGL.h"

#include "Window.h"

#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

RendererOGL::RendererOGL() : window(NULL) {}

RendererOGL::~RendererOGL(){}

bool RendererOGL::initialize(Window* window)
{
	this->window = window;
	// GLFW

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	glfwWindowHint(GLFW_ALPHA_BITS, 8);


	glfwWindowHint(GLFW_DEPTH_BITS, 24);

	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

	// GLEW

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		spdlog::error("{} : Failed to init GLEW", __FILENAME__);
		return false;
	}

	// On some platforms, GLEW will emit a benign error code, so clear it
	glGetError();

	// Enable debugging
	glEnable(GL_DEBUG_OUTPUT);

	return true;
}

void RendererOGL::close()
{
}

void RendererOGL::render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	beginDraw();
	draw();
	endDraw();

	glfwSwapBuffers(window->getGLFWWindow());
}

void RendererOGL::beginDraw()
{
}

void RendererOGL::draw()
{
}

void RendererOGL::endDraw()
{
}

IRenderer::Type RendererOGL::type()
{
	return IRenderer::Type::OpenGL;
}
