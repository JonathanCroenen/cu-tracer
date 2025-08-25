#include <iostream>
#include "camera.hu"
#include "material.hu"
#include "renderer.hu"
#include "sphere.hu"
#include "window.hu"

// Constants
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

int main() {
    rt::OpenGLCudaWindow window(WINDOW_WIDTH, WINDOW_HEIGHT);
    if (!window.Init()) {
        std::cerr << "Failed to initialize OpenGL renderer" << std::endl;
        return -1;
    }

    rt::Renderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT, 8);
    if (!renderer.Init()) {
        std::cerr << "Failed to initialize CUDA renderer" << std::endl;
        return -1;
    }

    std::cout << "CUDA Ray Tracer initialized successfully!" << std::endl;
    std::cout << "Press ESC to exit" << std::endl;

    // Set up camera and sphere
    rt::Camera camera(rt::Vec3f(0, 0, 1),                         // look from
                      rt::Vec3f(0, 0, -1),                        // look at
                      rt::Vec3f(0, 1, 0),                         // up vector
                      90.0f,                                      // vertical FOV
                      float(WINDOW_WIDTH) / float(WINDOW_HEIGHT)  // aspect ratio
    );

    // Use CUDA managed memory for spheres
    rt::Material* metal = new rt::Material(rt::Metal{rt::Vec3f(1.0f, 1.0f, 1.0f), 0.0f});
    rt::Material* metal2 = new rt::Material(rt::Metal{rt::Vec3f(0.0f, 1.0f, 1.0f), 1.0f});

    const int num_spheres = 3;
    rt::Sphere* spheres = nullptr;
    cudaMallocManaged(&spheres, num_spheres * sizeof(rt::Sphere));
    spheres[0] = rt::Sphere(rt::Vec3f(-0.7f, 0.3f, -1.0f), 0.5f, metal);
    spheres[1] = rt::Sphere(rt::Vec3f(0.5f, 0, -1.0f), 0.5f, metal);
    spheres[2] = rt::Sphere(rt::Vec3f(0.0f, 20.0f, -1.0f), 20.0f, metal2);

    // Main render loop
    while (!window.ShouldClose()) {
        window.PollEvents();

        // Handle input
        if (window.IsKeyPressed(GLFW_KEY_ESCAPE)) {
            break;
        }

        if (window.IsKeyPressed(GLFW_KEY_R)) {
            renderer.ClearAccumulator();
            std::cout << "Reset renderer accumulator" << std::endl;
        }

        // Render frame with CUDA
        renderer.RenderFrame(window.GetFramebuffer(), camera, spheres, num_spheres);

        window.SwapBuffers();
    }

    // Free device memory
    cudaFree(spheres);

    return 0;
}