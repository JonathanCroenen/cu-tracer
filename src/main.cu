#include <iostream>
#include "camera.cuh"
#include "importers/obj.h"
#include "material.cuh"
#include "object.cuh"
#include "renderer.cuh"
#include "scene.cuh"
#include "window.cuh"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

std::unique_ptr<rt::Scene> CreateCornellBoxSpheres() {
    auto scene = rt::Scene::Create();

    rt::MaterialRef red = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.65f, 0.05f, 0.05f)));
    rt::MaterialRef green = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.12f, 0.45f, 0.15f)));
    rt::MaterialRef white = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.73f, 0.73f, 0.73f)));
    rt::MaterialRef light = scene->AddMaterial(rt::Emissive(rt::Vec3f(1.0f, 1.0f, 1.0f), 10.0f));
    rt::MaterialRef metal = scene->AddMaterial(rt::Metal(rt::Vec3f(0.8f, 0.85f, 0.88f), 0.0f));
    rt::MaterialRef glass = scene->AddMaterial(rt::Dielectric(1.5f, 0.03f, 0.02f));
    rt::MaterialRef colored_glass =
        scene->AddMaterial(rt::Dielectric(1.5f, 0.01f, 0.01f, rt::Vec3f(0.1f, 0.05f, 1.0f), 9.0f));
    rt::MaterialRef fuzz = scene->AddMaterial(rt::Metal(rt::Vec3f(0.8f, 0.6f, 0.2f), 0.6f));

    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, 1.0f, 0.0f)), white,
                     rt::Transformf::Translation(0.0f, -0.5f, 0.0f));
    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, -1.0f, 0.0f)), white,
                     rt::Transformf::Translation(0.0f, 0.5f, 0.0f));
    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, 0.0f, 1.0f)), white,
                     rt::Transformf::Translation(0.0f, 0.0f, -1.0f));
    scene->AddObject(rt::Plane(rt::Vec3f(1.0f, 0.0f, 0.0f)), red,
                     rt::Transformf::Translation(-0.5f, 0.0f, 0.0f));
    scene->AddObject(rt::Plane(rt::Vec3f(-1.0f, 0.0f, 0.0f)), green,
                     rt::Transformf::Translation(0.5f, 0.0f, 0.0f));

    scene->AddObject(
        rt::Sphere(1.0f), light,
        rt::Transformf::Translation(0.0f, 0.5f, 0.0f) * rt::Transformf::Scaling(0.2f, 0.02f, 0.2f));
    scene->AddObject(rt::Sphere(0.1f), white,
                     rt::Transformf::Translation(-0.3f, 0.2f, -0.6f) *
                         rt::Transformf::Scaling(1.3f, 1.0f, 1.0f));
    scene->AddObject(rt::Sphere(0.12f), glass,
                     rt::Transformf::Translation(-0.2f, -0.15f, 0.0f) *
                         rt::Transformf::RotationY(-0.6f) *
                         rt::Transformf::Scaling(1.4f, 1.4f, 1.0f));
    scene->AddObject(rt::Sphere(0.1f), colored_glass,
                     rt::Transformf::Translation(0.0f, -0.4f, 0.5f));
    scene->AddObject(rt::Sphere(0.1f), fuzz, rt::Transformf::Translation(0.2f, -0.35f, -0.3f));

    scene->SetCamera(rt::Camera(rt::Vec3f(0.0f, 0.0f, 2.5f),  // look from
                                rt::Vec3f(0.0f, 0.0f, 0.0f),  // look at
                                rt::Vec3f(0.0f, 1.0f, 0.0f),  // up
                                30.0f,                        // vfov
                                float(WINDOW_WIDTH) / float(WINDOW_HEIGHT)));

    return scene;
}

std::unique_ptr<rt::Scene> CreateCornellBoxMesh() {
    rt::ObjFile obj_file = rt::ParseObjFile("assets/teapot.obj");

    auto scene = rt::Scene::Create();

    rt::MaterialRef red = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.65f, 0.05f, 0.05f)));
    rt::MaterialRef green = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.12f, 0.45f, 0.15f)));
    rt::MaterialRef white = scene->AddMaterial(rt::Lambertian(rt::Vec3f(0.73f, 0.73f, 0.73f)));
    rt::MaterialRef light = scene->AddMaterial(rt::Emissive(rt::Vec3f(1.0f, 1.0f, 1.0f), 15.0f));
    rt::MaterialRef glass = scene->AddMaterial(rt::Dielectric(1.5f, 0.03f, 0.02f));

    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, 1.0f, 0.0f)), white,
                     rt::Transformf::Translation(0.0f, -0.5f, 0.0f));
    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, -1.0f, 0.0f)), white,
                     rt::Transformf::Translation(0.0f, 0.5f, 0.0f));
    scene->AddObject(rt::Plane(rt::Vec3f(0.0f, 0.0f, 1.0f)), white,
                     rt::Transformf::Translation(0.0f, 0.0f, -1.0f));
    scene->AddObject(rt::Plane(rt::Vec3f(1.0f, 0.0f, 0.0f)), red,
                     rt::Transformf::Translation(-0.5f, 0.0f, 0.0f));
    scene->AddObject(rt::Plane(rt::Vec3f(-1.0f, 0.0f, 0.0f)), green,
                     rt::Transformf::Translation(0.5f, 0.0f, 0.0f));

    scene->AddObject(
        rt::Sphere(1.0f), light,
        rt::Transformf::Translation(0.0f, 0.5f, 0.0f) * rt::Transformf::Scaling(0.2f, 0.02f, 0.2f));

    // for (const auto& face : obj_file.faces) {
    //     rt::Vec3f v0 = obj_file.vertices[face.face_infos[0].vertex_index];
    //     rt::Vec3f v1 = obj_file.vertices[face.face_infos[1].vertex_index];
    //     rt::Vec3f v2 = obj_file.vertices[face.face_infos[2].vertex_index];

    //     scene->AddObject(
    //         rt::Triangle(v0, v1, v2), white,
    //         rt::Transformf::Translation(0.0f, 0.0f, -0.4f) * rt::Transformf::Scaling(0.1f));
    // }

    rt::Vec3f v0(0.0f, 0.0f, -0.4f);
    rt::Vec3f v1(-0.2f, -0.2f, -0.4f);
    rt::Vec3f v2(0.2f, -0.2f, -0.4f);
    scene->AddObject(rt::Triangle(v0, v1, v2), glass);

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

    rt::Renderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT, 4, 4);
    if (!renderer.Init()) {
        std::cerr << "Failed to initialize CUDA renderer" << std::endl;
        return -1;
    }

    std::cout << "CUDA Ray Tracer initialized successfully!" << std::endl;
    std::cout << "Press ESC to exit" << std::endl;

    auto scene = CreateCornellBoxMesh();
    // auto scene = CreateCornellBoxSpheres();
    scene->Build();

    double timer = 0.0;
    while (!window.ShouldClose()) {
        window.PollEvents();

        if (window.IsKeyPressed(GLFW_KEY_ESCAPE)) {
            break;
        }

        if (window.IsKeyPressed(GLFW_KEY_R)) {
            renderer.ClearAccumulator();
            std::cout << "Reset renderer accumulator" << std::endl;
        }

        renderer.RenderFrame(window.GetFramebuffer(), *scene, rt::RenderMode::PATH_TRACING);
        window.SwapBuffers();

        if (timer >= 1.0) {
            std::cout << "frame time: " << window.GetDeltaTime() * 1000.0 << " ms, ";
            std::cout << "fps: " << window.GetFPS() << std::endl;
            timer = 0.0;
        }

        timer += window.GetDeltaTime();
    }

    return 0;
}