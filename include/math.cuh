#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "common.cuh"

namespace rt {

constexpr float epsilon = 1e-6f;

// Forward declarations
template <typename T>
struct Vec2;
template <typename T>
struct Vec3;
template <typename T>
struct Vec4;
template <typename T>
struct Mat3;
template <typename T>
struct Mat4;
template <typename T>
struct Transform;

// Type aliases for common use cases
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;
using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;

using Mat3f = Mat3<float>;
using Mat4f = Mat4<float>;
using Mat3d = Mat3<double>;
using Mat4d = Mat4<double>;

using Transformf = Transform<float>;
using Transformd = Transform<double>;

// ============================================================================
// VECTOR CLASSES
// ============================================================================

template <typename T>
struct Vec2 {
    T x, y;

    DEVICE_HOST Vec2() : x(0), y(0) {}
    DEVICE_HOST Vec2(T x, T y) : x(x), y(y) {}
    DEVICE_HOST explicit Vec2(T v) : x(v), y(v) {}

    // Arithmetic operators
    DEVICE_HOST Vec2<T> operator+(const Vec2<T>& other) const {
        return Vec2<T>(x + other.x, y + other.y);
    }
    DEVICE_HOST Vec2<T> operator-(const Vec2<T>& other) const {
        return Vec2<T>(x - other.x, y - other.y);
    }
    DEVICE_HOST Vec2<T> operator*(const Vec2<T>& other) const {
        return Vec2<T>(x * other.x, y * other.y);
    }
    DEVICE_HOST Vec2<T> operator/(const Vec2<T>& other) const {
        return Vec2<T>(x / other.x, y / other.y);
    }

    DEVICE_HOST Vec2<T> operator+(T scalar) const {
        return Vec2<T>(x + scalar, y + scalar);
    }
    DEVICE_HOST Vec2<T> operator-(T scalar) const {
        return Vec2<T>(x - scalar, y - scalar);
    }
    DEVICE_HOST Vec2<T> operator*(T scalar) const {
        return Vec2<T>(x * scalar, y * scalar);
    }
    DEVICE_HOST Vec2<T> operator/(T scalar) const {
        return Vec2<T>(x / scalar, y / scalar);
    }

    // Compound assignment
    DEVICE_HOST Vec2<T>& operator+=(const Vec2<T>& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    DEVICE_HOST Vec2<T>& operator-=(const Vec2<T>& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    DEVICE_HOST Vec2<T>& operator*=(const Vec2<T>& other) {
        x *= other.x;
        y *= other.y;
        return *this;
    }
    DEVICE_HOST Vec2<T>& operator/=(const Vec2<T>& other) {
        x /= other.x;
        y /= other.y;
        return *this;
    }

    // Compound assignment with scalar
    DEVICE_HOST Vec2<T>& operator+=(T scalar) {
        x += scalar;
        y += scalar;
        return *this;
    }
    DEVICE_HOST Vec2<T>& operator-=(T scalar) {
        x -= scalar;
        y -= scalar;
        return *this;
    }
    DEVICE_HOST Vec2<T>& operator*=(T scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    DEVICE_HOST Vec2<T>& operator/=(T scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    // Unary operators
    DEVICE_HOST Vec2<T> operator-() const {
        return Vec2<T>(-x, -y);
    }

    // Access operators
    DEVICE_HOST T& operator[](int i) {
        return (&x)[i];
    }
    DEVICE_HOST const T& operator[](int i) const {
        return (&x)[i];
    }

    // Utility functions
    DEVICE_HOST T Dot(const Vec2<T>& other) const {
        return x * other.x + y * other.y;
    }
    DEVICE_HOST T LengthSquared() const {
        return x * x + y * y;
    }
    DEVICE_HOST T Length() const {
        return sqrt(LengthSquared());
    }
    DEVICE_HOST Vec2<T> Normalized() const {
        T len = Length();
        return len > 0 ? *this / len : *this;
    }

    // Static constructors
    DEVICE_HOST static Vec2<T> Zero() {
        return Vec2<T>(0, 0);
    }
    DEVICE_HOST static Vec2<T> One() {
        return Vec2<T>(1, 1);
    }
};

template <typename T>
struct Vec3 {
    T x, y, z;

    DEVICE_HOST Vec3() : x(0), y(0), z(0) {}
    DEVICE_HOST Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
    DEVICE_HOST explicit Vec3(T v) : x(v), y(v), z(v) {}
    DEVICE_HOST Vec3(const Vec2<T>& xy, T z) : x(xy.x), y(xy.y), z(z) {}

    // Arithmetic operators
    DEVICE_HOST Vec3<T> operator+(const Vec3<T>& other) const {
        return Vec3<T>(x + other.x, y + other.y, z + other.z);
    }
    DEVICE_HOST Vec3<T> operator-(const Vec3<T>& other) const {
        return Vec3<T>(x - other.x, y - other.y, z - other.z);
    }
    DEVICE_HOST Vec3<T> operator*(const Vec3<T>& other) const {
        return Vec3<T>(x * other.x, y * other.y, z * other.z);
    }
    DEVICE_HOST Vec3<T> operator/(const Vec3<T>& other) const {
        return Vec3<T>(x / other.x, y / other.y, z / other.z);
    }

    DEVICE_HOST Vec3<T> operator+(T scalar) const {
        return Vec3<T>(x + scalar, y + scalar, z + scalar);
    }
    DEVICE_HOST Vec3<T> operator-(T scalar) const {
        return Vec3<T>(x - scalar, y - scalar, z - scalar);
    }
    DEVICE_HOST Vec3<T> operator*(T scalar) const {
        return Vec3<T>(x * scalar, y * scalar, z * scalar);
    }
    DEVICE_HOST Vec3<T> operator/(T scalar) const {
        return Vec3<T>(x / scalar, y / scalar, z / scalar);
    }

    // Compound assignment
    DEVICE_HOST Vec3<T>& operator+=(const Vec3<T>& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    DEVICE_HOST Vec3<T>& operator-=(const Vec3<T>& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    DEVICE_HOST Vec3<T>& operator*=(const Vec3<T>& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }
    DEVICE_HOST Vec3<T>& operator/=(const Vec3<T>& other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        return *this;
    }

    // Compound assignment with scalar
    DEVICE_HOST Vec3<T>& operator+=(T scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        return *this;
    }
    DEVICE_HOST Vec3<T>& operator-=(T scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        return *this;
    }
    DEVICE_HOST Vec3<T>& operator*=(T scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    DEVICE_HOST Vec3<T>& operator/=(T scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    // Unary operators
    DEVICE_HOST Vec3<T> operator-() const {
        return Vec3<T>(-x, -y, -z);
    }

    // Access operators
    DEVICE_HOST T& operator[](int i) {
        return (&x)[i];
    }
    DEVICE_HOST const T& operator[](int i) const {
        return (&x)[i];
    }

    // Utility functions
    DEVICE_HOST T Dot(const Vec3<T>& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    DEVICE_HOST Vec3<T> Cross(const Vec3<T>& other) const {
        return Vec3<T>(y * other.z - z * other.y, z * other.x - x * other.z,
                       x * other.y - y * other.x);
    }
    DEVICE_HOST T LengthSquared() const {
        return x * x + y * y + z * z;
    }
    DEVICE_HOST T Length() const {
        return sqrt(LengthSquared());
    }
    DEVICE_HOST Vec3<T> Normalized() const {
        T len = Length();
        return len > 0 ? *this / len : *this;
    }

    DEVICE_HOST Vec3<T> Reflect(const Vec3<T>& normal) const {
        return *this - 2 * this->Dot(normal) * normal;
    }

    // Static constructors
    DEVICE_HOST static Vec3<T> Zero() {
        return Vec3<T>(0, 0, 0);
    }
    DEVICE_HOST static Vec3<T> One() {
        return Vec3<T>(1, 1, 1);
    }
    DEVICE_HOST static Vec3<T> UnitX() {
        return Vec3<T>(1, 0, 0);
    }
    DEVICE_HOST static Vec3<T> UnitY() {
        return Vec3<T>(0, 1, 0);
    }
    DEVICE_HOST static Vec3<T> UnitZ() {
        return Vec3<T>(0, 0, 1);
    }
};

template <typename T>
struct Vec4 {
    T x, y, z, w;

    DEVICE_HOST Vec4() : x(0), y(0), z(0), w(0) {}
    DEVICE_HOST Vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    DEVICE_HOST explicit Vec4(T v) : x(v), y(v), z(v), w(v) {}
    DEVICE_HOST Vec4(const Vec3<T>& xyz, T w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}
    DEVICE_HOST Vec4(const Vec2<T>& xy, const Vec2<T>& zw) : x(xy.x), y(xy.y), z(zw.x), w(zw.y) {}

    // Arithmetic operators
    DEVICE_HOST Vec4<T> operator+(const Vec4<T>& other) const {
        return Vec4<T>(x + other.x, y + other.y, z + other.z, w + other.w);
    }
    DEVICE_HOST Vec4<T> operator-(const Vec4<T>& other) const {
        return Vec4<T>(x - other.x, y - other.y, z - other.z, w - other.w);
    }
    DEVICE_HOST Vec4<T> operator*(const Vec4<T>& other) const {
        return Vec4<T>(x * other.x, y * other.y, z * other.z, w * other.w);
    }
    DEVICE_HOST Vec4<T> operator/(const Vec4<T>& other) const {
        return Vec4<T>(x / other.x, y / other.y, z / other.z, w / other.w);
    }

    DEVICE_HOST Vec4<T> operator+(T scalar) const {
        return Vec4<T>(x + scalar, y + scalar, z + scalar, w + scalar);
    }
    DEVICE_HOST Vec4<T> operator-(T scalar) const {
        return Vec4<T>(x - scalar, y - scalar, z - scalar, w - scalar);
    }
    DEVICE_HOST Vec4<T> operator*(T scalar) const {
        return Vec4<T>(x * scalar, y * scalar, z * scalar, w * scalar);
    }
    DEVICE_HOST Vec4<T> operator/(T scalar) const {
        return Vec4<T>(x / scalar, y / scalar, z / scalar, w / scalar);
    }

    // Compound assignment
    DEVICE_HOST Vec4<T>& operator+=(const Vec4<T>& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }
    DEVICE_HOST Vec4<T>& operator-=(const Vec4<T>& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }
    DEVICE_HOST Vec4<T>& operator*=(const Vec4<T>& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        w *= other.w;
        return *this;
    }
    DEVICE_HOST Vec4<T>& operator/=(const Vec4<T>& other) {
        x /= other.x;
        y /= other.y;
        z /= other.z;
        w /= other.w;
        return *this;
    }

    // Compound assignment with scalar
    DEVICE_HOST Vec4<T>& operator+=(T scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        w += scalar;
        return *this;
    }
    DEVICE_HOST Vec4<T>& operator-=(T scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        w -= scalar;
        return *this;
    }
    DEVICE_HOST Vec4<T>& operator*=(T scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }
    DEVICE_HOST Vec4<T>& operator/=(T scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        return *this;
    }

    // Unary operators
    DEVICE_HOST Vec4<T> operator-() const {
        return Vec4<T>(-x, -y, -z, -w);
    }

    // Access operators
    DEVICE_HOST T& operator[](int i) {
        return (&x)[i];
    }
    DEVICE_HOST const T& operator[](int i) const {
        return (&x)[i];
    }

    // Utility functions
    DEVICE_HOST T Dot(const Vec4<T>& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }
    DEVICE_HOST T LengthSquared() const {
        return x * x + y * y + z * z + w * w;
    }
    DEVICE_HOST T Length() const {
        return sqrt(LengthSquared());
    }
    DEVICE_HOST Vec4<T> Normalized() const {
        T len = Length();
        return len > 0 ? *this / len : *this;
    }

    // Static constructors
    DEVICE_HOST static Vec4<T> Zero() {
        return Vec4<T>(0, 0, 0, 0);
    }
    DEVICE_HOST static Vec4<T> One() {
        return Vec4<T>(1, 1, 1, 1);
    }
    DEVICE_HOST static Vec4<T> UnitX() {
        return Vec4<T>(1, 0, 0, 0);
    }
    DEVICE_HOST static Vec4<T> UnitY() {
        return Vec4<T>(0, 1, 0, 0);
    }
    DEVICE_HOST static Vec4<T> UnitZ() {
        return Vec4<T>(0, 0, 1, 0);
    }
    DEVICE_HOST static Vec4<T> UnitW() {
        return Vec4<T>(0, 0, 0, 1);
    }
};

// ============================================================================
// MATRIX CLASSES
// ============================================================================

template <typename T>
struct Mat3 {
    T m[9];  // Column-major storage

    DEVICE_HOST Mat3() {
        for (int i = 0; i < 9; i++)
            m[i] = 0;
        m[0] = m[4] = m[8] = 1;  // Identity matrix
    }

    DEVICE_HOST Mat3(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22) {
        m[0] = m00;
        m[3] = m01;
        m[6] = m02;
        m[1] = m10;
        m[4] = m11;
        m[7] = m12;
        m[2] = m20;
        m[5] = m21;
        m[8] = m22;
    }

    DEVICE_HOST T& operator()(int row, int col) {
        return m[col * 3 + row];
    }
    DEVICE_HOST const T& operator()(int row, int col) const {
        return m[col * 3 + row];
    }

    // Matrix operations
    DEVICE_HOST Mat3<T> operator*(const Mat3<T>& other) const {
        Mat3<T> result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = 0;
                for (int k = 0; k < 3; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    DEVICE_HOST Vec3<T> operator*(const Vec3<T>& vec) const {
        return Vec3<T>((*this)(0, 0) * vec.x + (*this)(0, 1) * vec.y + (*this)(0, 2) * vec.z,
                       (*this)(1, 0) * vec.x + (*this)(1, 1) * vec.y + (*this)(1, 2) * vec.z,
                       (*this)(2, 0) * vec.x + (*this)(2, 1) * vec.y + (*this)(2, 2) * vec.z);
    }

    // Static constructors
    DEVICE_HOST static Mat3<T> Identity() {
        return Mat3<T>();
    }

    DEVICE_HOST static Mat3<T> RotationX(T angle) {
        T c = cos(angle);
        T s = sin(angle);
        return Mat3<T>(1, 0, 0, 0, c, -s, 0, s, c);
    }

    DEVICE_HOST static Mat3<T> RotationY(T angle) {
        T c = cos(angle);
        T s = sin(angle);
        return Mat3<T>(c, 0, s, 0, 1, 0, -s, 0, c);
    }

    DEVICE_HOST static Mat3<T> RotationZ(T angle) {
        T c = cos(angle);
        T s = sin(angle);
        return Mat3<T>(c, -s, 0, s, c, 0, 0, 0, 1);
    }

    DEVICE_HOST static Mat3<T> Scaling(T sx, T sy, T sz) {
        return Mat3<T>(sx, 0, 0, 0, sy, 0, 0, 0, sz);
    }
};

template <typename T>
struct Mat4 {
    T m[16];  // Column-major storage

    DEVICE_HOST Mat4() {
        for (int i = 0; i < 16; i++)
            m[i] = 0;
        m[0] = m[5] = m[10] = m[15] = 1;  // Identity matrix
    }

    DEVICE_HOST Mat4(T m00, T m01, T m02, T m03, T m10, T m11, T m12, T m13, T m20, T m21, T m22,
                     T m23, T m30, T m31, T m32, T m33) {
        m[0] = m00;
        m[4] = m01;
        m[8] = m02;
        m[12] = m03;
        m[1] = m10;
        m[5] = m11;
        m[9] = m12;
        m[13] = m13;
        m[2] = m20;
        m[6] = m21;
        m[10] = m22;
        m[14] = m23;
        m[3] = m30;
        m[7] = m31;
        m[11] = m32;
        m[15] = m33;
    }

    DEVICE_HOST T& operator()(int row, int col) {
        return m[col * 4 + row];
    }
    DEVICE_HOST const T& operator()(int row, int col) const {
        return m[col * 4 + row];
    }

    // Matrix operations
    DEVICE_HOST Mat4<T> operator*(const Mat4<T>& other) const {
        Mat4<T> result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result(i, j) = 0;
                for (int k = 0; k < 4; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    DEVICE_HOST Vec4<T> operator*(const Vec4<T>& vec) const {
        return Vec4<T>((*this)(0, 0) * vec.x + (*this)(0, 1) * vec.y + (*this)(0, 2) * vec.z +
                           (*this)(0, 3) * vec.w,
                       (*this)(1, 0) * vec.x + (*this)(1, 1) * vec.y + (*this)(1, 2) * vec.z +
                           (*this)(1, 3) * vec.w,
                       (*this)(2, 0) * vec.x + (*this)(2, 1) * vec.y + (*this)(2, 2) * vec.z +
                           (*this)(2, 3) * vec.w,
                       (*this)(3, 0) * vec.x + (*this)(3, 1) * vec.y + (*this)(3, 2) * vec.z +
                           (*this)(3, 3) * vec.w);
    }

    DEVICE_HOST Vec3<T> TransformPoint(const Vec3<T>& point) const {
        Vec4<T> p(point, 1);
        Vec4<T> result = (*this) * p;
        return Vec3<T>(result.x / result.w, result.y / result.w, result.z / result.w);
    }

    DEVICE_HOST Vec3<T> TransformVector(const Vec3<T>& vector) const {
        Vec4<T> v(vector, 0);
        Vec4<T> result = (*this) * v;
        return Vec3<T>(result.x, result.y, result.z);
    }

    DEVICE_HOST Mat4<T> Transpose() const {
        Mat4<T> result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result(i, j) = (*this)(j, i);
            }
        }
        return result;
    }

    // Static constructors
    DEVICE_HOST static Mat4<T> Identity() {
        return Mat4<T>();
    }

    DEVICE_HOST static Mat4<T> Translation(T x, T y, T z) {
        Mat4<T> m = Identity();
        m(0, 3) = x;
        m(1, 3) = y;
        m(2, 3) = z;
        return m;
    }

    DEVICE_HOST static Mat4<T> RotationX(T angle) {
        T c = cos(angle);
        T s = sin(angle);
        Mat4<T> m = Identity();
        m(1, 1) = c;
        m(1, 2) = -s;
        m(2, 1) = s;
        m(2, 2) = c;
        return m;
    }

    DEVICE_HOST static Mat4<T> RotationY(T angle) {
        T c = cos(angle);
        T s = sin(angle);
        Mat4<T> m = Identity();
        m(0, 0) = c;
        m(0, 2) = s;
        m(2, 0) = -s;
        m(2, 2) = c;
        return m;
    }

    DEVICE_HOST static Mat4<T> RotationZ(T angle) {
        T c = cos(angle);
        T s = sin(angle);
        Mat4<T> m = Identity();
        m(0, 0) = c;
        m(0, 1) = -s;
        m(1, 0) = s;
        m(1, 1) = c;
        return m;
    }

    DEVICE_HOST static Mat4<T> Scaling(T sx, T sy, T sz) {
        Mat4<T> m = Identity();
        m(0, 0) = sx;
        m(1, 1) = sy;
        m(2, 2) = sz;
        return m;
    }

    DEVICE_HOST static Mat4<T> LookAt(const Vec3<T>& eye, const Vec3<T>& target,
                                      const Vec3<T>& up) {
        Vec3<T> z = (eye - target).Normalized();
        Vec3<T> x = up.Cross(z).Normalized();
        Vec3<T> y = z.Cross(x);

        Mat4<T> m = Identity();
        m(0, 0) = x.x;
        m(0, 1) = x.y;
        m(0, 2) = x.z;
        m(1, 0) = y.x;
        m(1, 1) = y.y;
        m(1, 2) = y.z;
        m(2, 0) = z.x;
        m(2, 1) = z.y;
        m(2, 2) = z.z;
        m(0, 3) = -x.Dot(eye);
        m(1, 3) = -y.Dot(eye);
        m(2, 3) = -z.Dot(eye);
        return m;
    }

    DEVICE_HOST static Mat4<T> Perspective(T fov, T aspect, T near, T far) {
        T f = 1.0f / tan(fov * 0.5f);
        Mat4<T> m;
        m(0, 0) = f / aspect;
        m(1, 1) = f;
        m(2, 2) = (far + near) / (near - far);
        m(2, 3) = (2 * far * near) / (near - far);
        m(3, 2) = -1;
        m(3, 3) = 0;
        return m;
    }

    DEVICE_HOST static Mat4<T> Orthographic(T left, T right, T bottom, T top, T near, T far) {
        Mat4<T> m = Identity();
        m(0, 0) = 2.0f / (right - left);
        m(1, 1) = 2.0f / (top - bottom);
        m(2, 2) = -2.0f / (far - near);
        m(0, 3) = -(right + left) / (right - left);
        m(1, 3) = -(top + bottom) / (top - bottom);
        m(2, 3) = -(far + near) / (far - near);
        return m;
    }
};

// ============================================================================
// TRANSFORM CLASS
// ============================================================================

template <typename T>
struct Transform {
    Mat4<T> matrix;
    Mat4<T> inverse_matrix;

    DEVICE_HOST Transform() : matrix(Mat4<T>::Identity()), inverse_matrix(Mat4<T>::Identity()) {}
    DEVICE_HOST Transform(const Mat4<T>& m)
        : matrix(m), inverse_matrix(m) { /* TODO: compute inverse */ }

    // Transform operations
    DEVICE_HOST Vec3<T> TransformPoint(const Vec3<T>& point) const {
        return matrix.TransformPoint(point);
    }
    DEVICE_HOST Vec3<T> TransformVector(const Vec3<T>& vector) const {
        return matrix.TransformVector(vector);
    }
    DEVICE_HOST Vec3<T> TransformNormal(const Vec3<T>& normal) const {
        return inverse_matrix.Transpose().TransformVector(normal).normalized();
    }

    // Static constructors
    DEVICE_HOST static Transform<T> Identity() {
        return Transform<T>();
    }

    DEVICE_HOST static Transform<T> Translation(T x, T y, T z) {
        return Transform<T>(Mat4<T>::Translation(x, y, z));
    }

    DEVICE_HOST static Transform<T> RotationX(T angle) {
        return Transform<T>(Mat4<T>::RotationX(angle));
    }

    DEVICE_HOST static Transform<T> RotationY(T angle) {
        return Transform<T>(Mat4<T>::RotationY(angle));
    }

    DEVICE_HOST static Transform<T> RotationZ(T angle) {
        return Transform<T>(Mat4<T>::RotationZ(angle));
    }

    DEVICE_HOST static Transform<T> Scaling(T sx, T sy, T sz) {
        return Transform<T>(Mat4<T>::Scaling(sx, sy, sz));
    }

    DEVICE_HOST static Transform<T> LookAt(const Vec3<T>& eye, const Vec3<T>& target,
                                           const Vec3<T>& up) {
        return Transform<T>(Mat4<T>::LookAt(eye, target, up));
    }

    // Composition
    DEVICE_HOST Transform<T> operator*(const Transform<T>& other) const {
        return Transform<T>(matrix * other.matrix);
    }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Scalar operations
template <typename T>
DEVICE_HOST T Clamp(T value, T min, T max) {
    return std::max(min, std::min(max, value));
}

template <typename T>
DEVICE_HOST T Lerp(T a, T b, T t) {
    return a + t * (b - a);
}

template <typename T>
DEVICE_HOST T Smoothstep(T edge0, T edge1, T x) {
    T t = Clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
}

// Random number generation
inline DEVICE float RandomFloat(curandState* random_state) {
    return curand_uniform(random_state);
}

inline DEVICE float RandomFloat(curandState* random_state, float min, float max) {
    return min + RandomFloat(random_state) * (max - min);
}

inline DEVICE Vec3f RandomUnitVector(curandState* random_state) {
    float a = RandomFloat(random_state, 0, 2 * M_PI);
    float z = RandomFloat(random_state, -1, 1);
    float r = sqrt(1 - z * z);
    return Vec3f(r * cos(a), r * sin(a), z);
}

inline DEVICE Vec3f RandomInUnitSphere(curandState* random_state) {
    Vec3f p;
    do {
        p = Vec3f(RandomFloat(random_state, -1, 1), RandomFloat(random_state, -1, 1),
                  RandomFloat(random_state, -1, 1));
    } while (p.LengthSquared() >= 1);
    return p;
}

inline DEVICE Vec3f RandomInUnitDisk(curandState* random_state) {
    Vec3f p;
    do {
        p = Vec3f(RandomFloat(random_state, -1, 1), RandomFloat(random_state, -1, 1), 0);
    } while (p.LengthSquared() >= 1);
    return p;
}

// ============================================================================
// OPERATOR OVERLOADS FOR SCALAR OPERATIONS
// ============================================================================

template <typename T>
DEVICE_HOST Vec2<T> operator+(T scalar, const Vec2<T>& vec) {
    return vec + scalar;
}
template <typename T>
DEVICE_HOST Vec2<T> operator-(T scalar, const Vec2<T>& vec) {
    return Vec2<T>(scalar - vec.x, scalar - vec.y);
}
template <typename T>
DEVICE_HOST Vec2<T> operator*(T scalar, const Vec2<T>& vec) {
    return vec * scalar;
}
template <typename T>
DEVICE_HOST Vec2<T> operator/(T scalar, const Vec2<T>& vec) {
    return Vec2<T>(scalar / vec.x, scalar / vec.y);
}

template <typename T>
DEVICE_HOST Vec3<T> operator+(T scalar, const Vec3<T>& vec) {
    return vec + scalar;
}
template <typename T>
DEVICE_HOST Vec3<T> operator-(T scalar, const Vec3<T>& vec) {
    return Vec3<T>(scalar - vec.x, scalar - vec.y, scalar - vec.z);
}
template <typename T>
DEVICE_HOST Vec3<T> operator*(T scalar, const Vec3<T>& vec) {
    return vec * scalar;
}
template <typename T>
DEVICE_HOST Vec3<T> operator/(T scalar, const Vec3<T>& vec) {
    return Vec3<T>(scalar / vec.x, scalar / vec.y, scalar / vec.z);
}

template <typename T>
DEVICE_HOST Vec4<T> operator+(T scalar, const Vec4<T>& vec) {
    return vec + scalar;
}
template <typename T>
DEVICE_HOST Vec4<T> operator-(T scalar, const Vec4<T>& vec) {
    return Vec4<T>(scalar - vec.x, scalar - vec.y, scalar - vec.z, scalar - vec.w);
}
template <typename T>
DEVICE_HOST Vec4<T> operator*(T scalar, const Vec4<T>& vec) {
    return vec * scalar;
}
template <typename T>
DEVICE_HOST Vec4<T> operator/(T scalar, const Vec4<T>& vec) {
    return Vec4<T>(scalar / vec.x, scalar / vec.y, scalar / vec.z, scalar / vec.w);
}

// ============================================================================
// OUTPUT OPERATORS
// ============================================================================

template <typename T>
HOST std::ostream& operator<<(std::ostream& os, const Vec2<T>& v) {
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

template <typename T>
HOST std::ostream& operator<<(std::ostream& os, const Vec3<T>& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

template <typename T>
HOST std::ostream& operator<<(std::ostream& os, const Vec4<T>& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return os;
}

template <typename T>
HOST std::ostream& operator<<(std::ostream& os, const Mat3<T>& m) {
    os << "[" << m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << "]\n";
    os << "[" << m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << "]\n";
    os << "[" << m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << "]";
    return os;
}

template <typename T>
HOST std::ostream& operator<<(std::ostream& os, const Mat4<T>& m) {
    for (int i = 0; i < 4; ++i) {
        os << "[";
        for (int j = 0; j < 4; ++j) {
            os << m(i, j);
            if (j < 3) os << " ";
        }
        os << "]\n";
    }
    return os;
}

}  // namespace rt
