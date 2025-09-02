#pragma once

#include <string>
#include <vector>
#include "../math.cuh"

namespace rt {

struct FaceInfo {
    int vertex_index;
    int uv_index;
    int normal_index;
};

struct Face {
    std::vector<FaceInfo> face_infos;
};

struct ObjFile {
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;
    std::vector<Vec2f> uvs;
    std::vector<Face> faces;
};

ObjFile ParseObjFile(const std::string& filename);

}  // namespace rt
