#include <iostream>
#include "camera.cuh"
#include "material.cuh"
#include "object.cuh"
#include "renderer.cuh"
#include "scene.cuh"
#include "window.cuh"

// Constants
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

std::unique_ptr<rt::Scene> CreateRTInOneWeekendScene() {
    auto scene = rt::Scene::Create();
    rt::MaterialRef ground = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.8f, 0.8f, 0.0f)));
    rt::MaterialRef center = scene->AddMaterial(rt::Metal(rt::Vec3f(0.1f, 0.2f, 0.5f), 1.0f));
    // Clear glass with slight surface imperfections
    rt::MaterialRef left = scene->AddMaterial(rt::Dielectric(1.5f, 0.02f, 0.01f));
    // Bubble with very smooth surfaces
    rt::MaterialRef bubble = scene->AddMaterial(rt::Dielectric(1.0f / 1.5f, 0.0f, 0.0f));
    rt::MaterialRef right = scene->AddMaterial(rt::Metal(rt::Vec3f(0.8f, 0.6f, 0.2f), 0.0f));

    scene->AddObject(rt::Sphere(rt::Vec3f(0.0f, -100.5f, -1.0f), 100.0f), ground);
    scene->AddObject(rt::Sphere(rt::Vec3f(0.0f, 0.0f, -1.2f), 0.5f), center);
    scene->AddObject(rt::Sphere(rt::Vec3f(-1.0f, 0.0f, -1.0f), 0.5f), left);
    scene->AddObject(rt::Sphere(rt::Vec3f(-1.0f, 0.0f, -1.0f), 0.4f), bubble);
    scene->AddObject(rt::Sphere(rt::Vec3f(1.0f, 0.0f, -1.0f), 0.5f), right);

    scene->SetCamera(rt::Camera(rt::Vec3f(-2.0f, 2.0f, 1.0f),               // look from
                                rt::Vec3f(0, 0, -1),                        // look at
                                rt::Vec3f(0, 1, 0),                         // up vector
                                90.0f,                                      // vertical FOV
                                float(WINDOW_WIDTH) / float(WINDOW_HEIGHT)  // aspect ratio
                                ));

    return scene;
}

std::unique_ptr<rt::Scene> CreateCornellBoxScene() {
    auto scene = rt::Scene::Create();

    rt::MaterialRef red = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.65f, 0.05f, 0.05f)));
    rt::MaterialRef green = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.12f, 0.45f, 0.15f)));
    rt::MaterialRef white = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.73f, 0.73f, 0.73f)));
    rt::MaterialRef light = scene->AddMaterial(rt::Emissive(rt::Vec3f(1.0f, 1.0f, 1.0f), 10.0f));
    rt::MaterialRef metal = scene->AddMaterial(rt::Metal(rt::Vec3f(0.8f, 0.85f, 0.88f), 0.0f));
    // Clear glass with slight surface roughness
    rt::MaterialRef glass = scene->AddMaterial(rt::Dielectric(1.5f, 0.03f, 0.02f));
    rt::MaterialRef fuzz = scene->AddMaterial(rt::Metal(rt::Vec3f(0.8f, 0.6f, 0.2f), 0.6f));

    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, -0.5f, 0.0f), rt::Vec3f(0.0f, 1.0f, 0.0f)), white);
    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, 0.5f, 0.0f), rt::Vec3f(0.0f, -1.0f, 0.0f)), white);
    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, 0.0f, -1.0f), rt::Vec3f(0.0f, 0.0f, 1.0f)), white);
    scene->AddObject(rt::Plane(rt::Vec3f(-0.5f, 0.0f, 0.0f), rt::Vec3f(1.0f, 0.0f, 0.0f)), red);
    scene->AddObject(rt::Plane(rt::Vec3f(0.5f, 0.0f, 0.0f), rt::Vec3f(-1.0f, 0.0f, 0.0f)), green);

    scene->AddObject(rt::Sphere(rt::Vec3f(0.0f, 1.09f, 0.0f), 0.6f), light);
    scene->AddObject(rt::Sphere(rt::Vec3f(0.0f, 0.2f, -0.6f), 0.1f), metal);
    scene->AddObject(rt::Sphere(rt::Vec3f(-0.2f, -0.15f, 0.0f), 0.12f), glass);
    scene->AddObject(rt::Sphere(rt::Vec3f(0.0f, 0.0f, 1.8f), 0.12f), glass);
    scene->AddObject(rt::Sphere(rt::Vec3f(0.2f, -0.35f, -0.3f), 0.1f), fuzz);
    scene->SetCamera(rt::Camera(rt::Vec3f(0.0f, 0.0f, 2.5f),  // look from
                                rt::Vec3f(0.0f, 0.0f, 0.0f),  // look at
                                rt::Vec3f(0.0f, 1.0f, 0.0f),  // up
                                30.0f,                        // vfov
                                float(WINDOW_WIDTH) / float(WINDOW_HEIGHT)));

    return scene;
}

int main() {
    rt::OpenGLCudaWindow window(WINDOW_WIDTH, WINDOW_HEIGHT);
    if (!window.Init()) {
        std::cerr << "Failed to initialize OpenGL renderer" << std::endl;
        return -1;
    }

    rt::Renderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT, 16, 8);
    if (!renderer.Init()) {
        std::cerr << "Failed to initialize CUDA renderer" << std::endl;
        return -1;
    }

    std::cout << "CUDA Ray Tracer initialized successfully!" << std::endl;
    std::cout << "Press ESC to exit" << std::endl;

    // auto scene = CreateRTInOneWeekendScene();
    auto scene = CreateCornellBoxScene();
    scene->Build();

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
        renderer.RenderFrame(window.GetFramebuffer(), *scene);

        window.SwapBuffers();
    }

    return 0;
}