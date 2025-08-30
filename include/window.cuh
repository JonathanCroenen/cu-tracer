#pragma once

// Include glad first so it loads the GL headers
#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include "framebuffer.cuh"

namespace rt {

#define GL_CHECK()                                                                         \
    do {                                                                                   \
        GLenum err = glGetError();                                                         \
        if (err != GL_NO_ERROR) {                                                          \
            std::cerr << "OpenGL error at " << __FILE__ << ":" << __LINE__ << " - " << err \
                      << std::endl;                                                        \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

class OpenGLCudaWindow {
   public:
    int width;
    int height;

    OpenGLCudaWindow(int width, int height);
    ~OpenGLCudaWindow();

    bool Init();
    void Cleanup();
    void SwapBuffers();
    bool ShouldClose() const;
    void PollEvents();
    bool IsKeyPressed(int key) const;
    Framebuffer& GetFramebuffer() {
        return _framebuffer;
    }

   private:
    GLFWwindow* _window;
    GLuint _texture;
    GLuint _vao;
    GLuint _vbo;
    GLuint _shader_program;
    cudaGraphicsResource_t _cuda_texture_resource;
    Framebuffer _framebuffer;

    bool _CreateShaders();
    void _CreateFullscreenQuad();
    bool _CreateTexture();
};

}  // namespace rt
