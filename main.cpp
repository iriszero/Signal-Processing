#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <math.h>
#define N_HEIGHT 25
using namespace std;

const double e = exp(1.0);
const double PI = acos(-1.0);

class Complex {
public:
	double a, b; //real part, imaginary part
	Complex(double _a, double _b): a(_a), b(_b) {}
	Complex operator+(const Complex &other) {
		Complex res(a, b);
		res.a += other.a;
		res.b += other.b;
		return res;
	}
	Complex operator-(const Complex &other) {
		Complex res(a, b);
		res.a -= other.a;
		res.b -= other.b;
		return res;
	}
	Complex operator* (const Complex &other) const {
		return Complex(a*other.a-b*other.b, a*other.b + b*other.a);
	}

	Complex& operator=(const Complex &other) {
		return (*this);
	}
	Complex& operator+=(const Complex &other) {
		a += other.a;
		b += other.b;
		return (*this);
	}

	double getSize() const {
		return sqrt(a*a+b*b);
	}
};

Complex operator *(const double c, const Complex &other) {
	return Complex(c * other.a, c*other.b);
}
Complex operator/(const Complex &c, const double k) {
	return Complex(c.a /k ,c.b/k);
}
Complex ej(double theta) {
	return Complex(cos(theta), sin(theta));
}

//X[k] = 1/N sigma n=0 to 31 x[n]e^(-jk(2pi/N)n)
vector<Complex> getXaxisCoeiff(const cv::Mat &img, const cv::Rect &rect) {
	vector<Complex> X(rect.height, {0, 0});
	const int N = rect.width;
	for (int i=0; i<rect.height; i++) {
		int cy = rect.y + i;
		for (int k=0; k<N; k++) {
			for (int n=0; n<N; n++) {
				int cx = rect.x + n;
				X[k] += double(img.at<unsigned char>(cx, cy)) * ej(-k * 2* PI / double(N) * n)
				                                              / double(N*rect.height);
			}
		}
	}

	return X;
}
vector<Complex> getYaxisCoeiff(const cv::Mat &img, const cv::Rect &rect) {
	vector<Complex> X(rect.width, {0, 0});
	const int N = rect.height;
	for (int i=0; i<rect.width; i++) {
		int cx = rect.x + i;
		for (int k=0; k<N; k++) {
			for (int n=0; n<N; n++) {
				int cy = rect.y + n;
				X[k] += double(img.at<unsigned char>(cx, cy)) * ej(-k * 2* PI / double(N) * n) / double(N*rect.height);
			}
		}
	}

	return X;
}
void printBar(const vector<Complex> &vec) {
	for (int i=N_HEIGHT; i>=1; i--) {
		for (int j=0; j<vec.size(); j++) {
			if (vec[j].getSize() >= (double(i) / double(N_HEIGHT)) * 4.0) {
				printf("%6s", "*");
			}
			else {
				printf("%6s", "");
			}
		}
		printf("\n");
	}
	for (int j=0; j<vec.size(); j++) {
		printf("%6d", j);
	}
	printf("\n");

	for (int j=0; j<vec.size(); j++) {
		printf("%6.2f", vec[j].getSize());
	}
	printf("\n");
}
int main(int argc, const char** argv) {
	if (argc < 2 ) {
		printf("Usage : %s\n", "[Image File], ...");
	}

	cout << "OpenCV Version : " << CV_VERSION << std::endl;
	for (int i=1; i<argc; i++) {
		cv::Mat img, grayImg;
		img = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "[!] Image load fail!" << std::endl;
			return -1;
		}
		printf("Image Loaded : %s\n", argv[i]);
		cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

		cv::Rect rects[] = {
				cv::Rect(50, 90, 32, 32),
				cv::Rect(230, 120,32,32),
				cv::Rect(440, 110, 32, 32),
				cv::Rect(90, 400, 32, 32),
				cv::Rect(290, 450, 32, 32)
		};

		for (const cv::Rect& rect : rects ) {
			vector<Complex> vX = getXaxisCoeiff(grayImg, rect);
			vector<Complex> vY = getYaxisCoeiff(grayImg, rect);

			printf("=============== (%d, %d) X[k] ===============\n", rect.x, rect.y);
			printBar(vX);
			printf("\n");

			printf("=============== (%d, %d) Y[k] ===============\n", rect.x, rect.y);
			printBar(vY);
		}
		printf("\n\n");
	}

	return 0;
}


