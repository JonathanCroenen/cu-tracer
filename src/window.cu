#include "window.cuh"

namespace rt {

OpenGLCudaWindow::OpenGLCudaWindow(int w, int h)
    : width(w), height(h), _window(nullptr), _texture(0), _vao(0), _vbo(0), _shader_program(0),
      _cuda_texture_resource(nullptr), _current_time(0.0), _last_time(0.0), _delta_time(0.0) {}

OpenGLCudaWindow::~OpenGLCudaWindow() {
    Cleanup();
}

bool OpenGLCudaWindow::Init() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _window = glfwCreateWindow(width, height, "CUDA Ray Tracer", nullptr, nullptr);
    if (!_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(_window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    if (!_CreateTexture()) return false;
    _CreateFullscreenQuad();
    if (!_CreateShaders()) return false;

    return true;
}

void OpenGLCudaWindow::Cleanup() {
    if (_cuda_texture_resource) {
        cudaGraphicsUnregisterResource(_cuda_texture_resource);
    }

    if (_texture) glDeleteTextures(1, &_texture);
    if (_vao) glDeleteVertexArrays(1, &_vao);
    if (_vbo) glDeleteBuffers(1, &_vbo);
    if (_shader_program) glDeleteProgram(_shader_program);
    if (_window) glfwDestroyWindow(_window);
    glfwTerminate();
}

bool OpenGLCudaWindow::_CreateTexture() {
    glGenTextures(1, &_texture);
    glBindTexture(GL_TEXTURE_2D, _texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Set up CUDA OpenGL interop
    cudaError_t interop_result = cudaGraphicsGLRegisterImage(
        &_cuda_texture_resource, _texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    if (interop_result != cudaSuccess) {
        std::cerr << "CUDA OpenGL interop failed: " << cudaGetErrorString(interop_result)
                  << std::endl;
        return false;
    }

    _framebuffer = Framebuffer(_cuda_texture_resource);

    return true;
}

void OpenGLCudaWindow::_CreateFullscreenQuad() {
    // clang-format off
    float vertices[] = {
        // positions          // texture coords
        -1.0f,  1.0f,  0.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 
         1.0f, -1.0f,  0.0f,  1.0f, 0.0f, 
         1.0f,  1.0f,  0.0f,  1.0f, 1.0f
    };
    // clang-format on

    unsigned int indices[] = {0, 1, 2, 0, 2, 3};

    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    unsigned int ebo;
    glGenBuffers(1, &ebo);

    glBindVertexArray(_vao);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

bool OpenGLCudaWindow::_CreateShaders() {
    const char* vertex_shader_source = R"(
        #version 410 core

        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        void main() {
            gl_Position = vec4(aPos, 1.0);
            TexCoord = aTexCoord;
        }
    )";

    const char* fragment_shader_source = R"(
        #version 410 core

        out vec4 FragColor;
        in vec2 TexCoord;

        uniform sampler2D texture1;

        void main() {
            FragColor = texture(texture1, TexCoord);
        }
    )";

    unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
    glCompileShader(vertex_shader);

    unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
    glCompileShader(fragment_shader);

    _shader_program = glCreateProgram();
    glAttachShader(_shader_program, vertex_shader);
    glAttachShader(_shader_program, fragment_shader);
    glLinkProgram(_shader_program);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return true;
}

void OpenGLCudaWindow::SwapBuffers() {
    _last_time = _current_time;
    _current_time = glfwGetTime();
    _delta_time = _current_time - _last_time;

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(_shader_program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _texture);

    glBindVertexArray(_vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(_window);
}

bool OpenGLCudaWindow::ShouldClose() const {
    return glfwWindowShouldClose(_window);
}

void OpenGLCudaWindow::PollEvents() {
    glfwPollEvents();
}

bool OpenGLCudaWindow::IsKeyPressed(int key) const {
    return glfwGetKey(_window, key) == GLFW_PRESS;
}

}  // namespace rt
