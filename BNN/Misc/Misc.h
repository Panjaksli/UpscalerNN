#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <cmath>
#include <syncstream>

#if defined(__clang__)
#define restrict __restrict
#elif defined(__GNUC__) || defined(__GNUG__)
#define restrict __restrict__
#elif defined(_MSC_VER)
#define restrict __restrict
#endif
//stupid C++ doesnt provide a way for automatic spacing in ostream...
#define SPC <<' '<<
namespace BNN {
	using std::vector;
	using std::min;
	using std::max;
	inline float fast_sin(float x) {
		float x2 = x * x;
		return x + x * x2 * (-1.6666656684e-1f + x2 * (8.3330251389e-3f + x2 * (-1.9807418727e-4f + x2 * 2.6019030676e-6f)));
	}
	inline float mod(float u, float v) {
		return u - v * floorf(u / v);
	}
	inline float fsin(float x) {
		constexpr float pi = 3.1415926535;
		constexpr float pi2 = 2.0 * 3.1415926535;
		constexpr float hpi = 0.5 * 3.1415926535;
		x = mod(x - hpi, pi2);
		x = fabsf(x - pi) - hpi;
		return fast_sin(x);
	}
	inline float sinc(float x) {
		return fsin(x) / x;
	}
	inline float nsinc(float x) {
		return sinc(x * 3.1415926535);
	}
	template <class T>
	inline T clamp(const T& x, const T& _min, const T& _max) {
		return min(max(x, _min), _max);
	}
	template <class T>
	inline T lerp(const T& x, const T& y, const auto t) {
		return (1 - t) * x + t * y;
	}
	template <class T>
	inline T cerp(const T* p, T x) {
		return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
	}
	template <class T>
	inline T bicerp(const T* p, T x, T y) {
		T tmp[4];
		tmp[0] = cerp(p, x);
		tmp[1] = cerp(p + 4, x);
		tmp[2] = cerp(p + 8, x);
		tmp[3] = cerp(p + 12, x);
		return cerp(tmp, y);
	}
	
	inline float lanc3(float x) {
		return abs(x) < 3.f ? nsinc(x) * nsinc(x * (1.f / 3.f)) : 0.f;
	}

	template <class T>
	inline T lanczos3(const T* p, T x0, T y0) {
		float res = 0;
		for(int y = 0; y < 6; y++) {
			for(int x = 0; x < 6; x++) {
				res += lanc3(y - 3 + y0) * lanc3(x - 3 + x0) * p[y * 7 + x];
			}
		}
		return res;
	}
	enum Interpol {
		Nearest,
		Linear,
		Cubic,
		Lanczos
	};
	inline const char* to_cstr(Interpol i) {
		switch(i) {
			case Nearest: return "Nearest";
			case Linear: return "Linear";
			case Cubic: return "Cubic";
			default: return "Null";
		}
	}
	using uchar = uint_fast8_t;
	using uint = uint_fast32_t;
	inline uint xorshift32() {
		thread_local static uint x = 0x6969;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		return x;
	}
	inline uint fastrand() {
		thread_local static uint x = 0x6969;
		x = (214013U * x + 2531011U);
		return x;
	}

	inline float rafl() {
		uint x = 0x3f800000 | (xorshift32() & 0x007FFFFF);
		return *(float*)&x - 1.f;
	}

	inline float rafl(float min, float max) {
		return rafl() * (max - min) + min;
	}
	inline int raint(int min, int max) {
		return int(rafl() * int(max - min + 1)) + min;
	}
	template <typename T>
	T saturate(const T& x) {
		return min(max(x, T(0)), T(1));
	}
	inline float decay_rate(float alpha, float iter, float end_iter) {
		return alpha / (1.f + powf(iter/ end_iter,2));
	}
	template <class T = int>
	inline vector<T> shuffled(int n) {
		vector<T> shuff(n);
		for(int i = 0; i < n; i++)
			shuff[i] = i;
		for(int i = 0; i < n - 1; i++)
			std::swap(shuff[i], shuff[raint(i, n - 1)]);
		return shuff;
	}
	template <class T = int>
	inline void shuffle(vector<T> &data) {
		int n = data.size();
		for(int i = 0; i < n - 1; i++)
			std::swap(data[i], data[raint(i, n - 1)]);
	}
	inline double timer() {
		auto t = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double>(t.time_since_epoch()).count();
	}
	inline double timer(double t1) {
		return timer() - t1;
	}
#define cout std::osyncstream(std::cout)
	template <class T>
	inline void print(T t) {
		cout << t << ' ';
	}
	template <class T, class ...Ts>
	inline void print(T t, Ts... ts) {
		print(t);
		print(ts...);
	}
	template <class T>
	inline void printr(T t) {
		cout << t << "\r";
	}
	template <class T, class ...Ts>
	inline void printr(T t, Ts... ts) {
		print(t);
		printr(ts...);
	}
	template <class T>
	inline void println(T t) {
		cout << t << "\n";
	}
	template <class T, class ...Ts>
	inline void println(T t, Ts... ts) {
		print(t);
		println(ts...);
	}
	template <class T>
	inline void printlns(T t) {
		println(t);
	}
	template <class T, class ...Ts>
	inline void printlns(T t, Ts... ts) {
		println(t);
		printlns(ts...);
	}
#undef cout
}