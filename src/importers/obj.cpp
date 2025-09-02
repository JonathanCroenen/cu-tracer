#include "importers/obj.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace rt {

Face ParseFace(std::istringstream& iss) {
    Face face;
    std::string tuple;

    while (iss >> tuple) {
        FaceInfo face_info;
        std::istringstream tuple_iss(tuple);
        std::string part;

        std::getline(tuple_iss, part, '/');
        face_info.vertex_index = std::stoi(part) - 1;

        if (std::getline(tuple_iss, part, '/') && !part.empty()) {
            face_info.uv_index = std::stoi(part) - 1;
        }

        if (std::getline(tuple_iss, part, '/') && !part.empty()) {
            face_info.normal_index = std::stoi(part) - 1;
        }

        face.face_infos.push_back(face_info);
    }

    return face;
}

ObjFile ParseObjFile(const std::string& filename) {
    ObjFile obj_file;

    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type == "v") {
            Vec3f vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            obj_file.vertices.push_back(vertex);
        } else if (type == "vt") {
            Vec2f uv;
            iss >> uv.x >> uv.y;
            obj_file.uvs.push_back(uv);
        } else if (type == "vn") {
            Vec3f normal;
            iss >> normal.x >> normal.y >> normal.z;
            obj_file.normals.push_back(normal);
        } else if (type == "f") {
            Face face = ParseFace(iss);
            obj_file.faces.push_back(face);
        }
    }

    std::cout << "Number of vertices: " << obj_file.vertices.size() << std::endl;

    return obj_file;
}

}  // namespace rt