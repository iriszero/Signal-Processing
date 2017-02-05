#include <iostream>
#include <opencv2/opencv.hpp>
#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>
#include <set>
#include <string>
#include <ctime>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <functional>

#define N_HEIGHT 25
const double DOUBLE_INF = 1e20;
const double PI = acos(-1.0);
using namespace std;

complex<double> ej(const double theta) {
	return complex<double>(cos(theta), sin(theta));
}

double getEuclideanDist(const vector<double> &left, const vector<double> &right) {
	assert(left.size() == right.size());
	double distSquare = 0;
	for (int d=0; d<left.size(); d++ ){
		distSquare += (left[d] - right[d]) * (left[d] - right[d]);
	}
	return distSquare;
}

int getType(const char* imgpath) {
	assert(strlen(imgpath) >= 3);
	return (imgpath[0]-'0') * 10 + (imgpath[1] - '0');
}

void get2dDTFS(const cv::Mat &img, vector<vector<complex<double>>> &X) {
	const int M = img.cols;
	const int N = img.rows;
	X = vector<vector<complex<double>>>(M, vector<complex<double>>(N, {0, 0}));

	for (int n=0; n<N; n++) {
		for (int m=0; m<M; m++) {
			for (int y=0; y<N; y++) {
				for (int x=0; x<M; x++) {
					X[m][n] += (ej(- ( double(m) * 2.0 * PI / double(M) * double(x) +
							double(n) * 2.0 * PI / double(N) * double(y))) )
					* double(img.at<unsigned char>(x,y)) / (double(N * M));

				}

			}

		}
	}
	/*for (int i=0; i<M; i++) {
		for (int j=0;j<N; j++) {
			printf("X[%2d][%2d] = %10.5g\t", i, j, abs(X[i][j]));
		}
		printf("\n");
	}*/
	return ;
}

/* Param : points, nCluster
 * Return: vecClusteredSet, centers
 */
void KMeansClustering(const vector<vector<double>> &points,
                      const int nCluster,
                      vector<vector<int>>    &vecClusteredSet,
                      vector<vector<double>> &centers) {
	srand(time(NULL));
	if (points.empty() || nCluster ==0 || nCluster > points.size()) return;

	const int dim = points.back().size();
	vecClusteredSet.resize(nCluster);
	centers.clear();

	set<int> randomIdxSet;
	vector<int> clusterNo(points.size(), -1);

	while (randomIdxSet.size() < nCluster) {
		randomIdxSet.insert(rand() % nCluster);
	}
	for (const int n: randomIdxSet) {
		centers.push_back(points[n]);
	}

	while(1) {
		bool isConvergent = true;
		for (int i=0; i<points.size(); i++) {
			const auto &point = points[i];
			double minDistSquare= DOUBLE_INF; int minIdx = -1;

			for (int j=0; j<centers.size(); j++) {
				const auto &center = centers[j];

				double distSquare = getEuclideanDist(point, center);
				if (minDistSquare > distSquare) {
					minDistSquare = distSquare;
					minIdx =j;
				}
			}
			if (clusterNo[i] != minIdx) {
				clusterNo[i]= minIdx;
				isConvergent = true;
			}
		}

		vector<vector<double>> adjustedCenters(nCluster, vector<double>(dim, 0));
		vector<int> clusterCounts(nCluster ,0);
		for (int i=0; i<points.size(); i++) {
			for (int d=0; d<dim; d++) {
				adjustedCenters[clusterNo[i]][d] += points[i][d];
				clusterCounts[clusterNo[i]]++;
			}
		}
		for (int i=0; i< nCluster; i++) {
			for (int d=0; d<dim; d++ ) {
				adjustedCenters[i][d] /= double(clusterCounts[i]);
			}
		}

		if (isConvergent) break;
	}
	for (int i=0; i<clusterNo.size(); i++) {
		vecClusteredSet[clusterNo[i]].push_back(i);
	}
}

int getNumCluster(const vector<int> &type) {
	set<int> s;
	for (auto t : type) {
		s.insert(t);
	}
	return s.size();
}
vector<int> getMostFrequentType(const vector<int> &type, const vector<vector<int>> &vecClusteredSet) {
	const int nCluster = vecClusteredSet.size();
	vector<int> res(nCluster);
	vector<bool> taken(nCluster, false);

	for (int i=0; i<nCluster; i++) {
		const auto& cSet = vecClusteredSet[i];

		vector<int> counts(nCluster, 0);
		for (int j=0; j<cSet.size(); j++) {
			counts[type[cSet[j]]]++;
		}

		vector<pair<int, int>> vecPIdxCount;
		for (int j=0; j<counts.size(); j++) {
			vecPIdxCount.push_back({counts[j], j});
		}
		sort(vecPIdxCount.begin(), vecPIdxCount.end(), greater<pair<int, int>>());
		for (const auto &pIdxCount : vecPIdxCount) {
			if (taken[pIdxCount.second] == false) {
				res[i] = pIdxCount.second;
				taken[pIdxCount.second] = true;
				break;
			}
		}
	}
	return res;
}

int getNearestCenterIdx(const vector<double> &coord, const vector<vector<double>> &centers) {
	double minDistSquare = DOUBLE_INF; int minIdx = -1;
	for (int i=0; i<centers.size(); i++) {
		double distSquare = getEuclideanDist(coord, centers[i]);
		if (minDistSquare > distSquare) {
			minDistSquare = distSquare;
			minIdx = i;
		}
	}
	return minIdx;
}
void get2dDTFSHistogram() {
	FILE *fpInfo = fopen("../img_info.txt", "r");
	char imgname[256]; int x, y, width, height;

	while ( fscanf(fpInfo, "%s %d %d %d %d", imgname, &x, &y, &width, &height) != EOF) {
		string imgpath = string("../img/") + string(imgname) + string(".jpg");
		cv::Mat img = cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())
		{
			std::cout << "[!] Image load fail! : " << imgpath << endl;
			exit(1);
		}
		const int M = img.cols;
		const int N = img.rows;

		vector<vector<complex<double>>> v;
		get2dDTFS(img, v);

		cv::Mat hist(N, M, CV_8UC3);
		double min = 1e10, max = 1e-10;
		for (int i=0; i<v.size(); i++) {
			for (int j=0; j<v.back().size(); j++) {
				//ignore DC component
				if (i==0 && j==0) continue;
				//[Linear] if (max < (abs(v[i][j]))) max = (abs(v[i][j]));
				if (max < log(abs(v[i][j]))) max = log(abs(v[i][j]));
				if (min > log(abs(v[i][j]))) min = log(abs(v[i][j]));
			}
		}

		for (int i=0; i<v.size(); i++) {
			for (int j=0; j<v.back().size(); j++) {
				//[Linear]hist.at<cv::Vec3b>(i, j)[0] = int( ( 1.0- ( abs(v[i][j]) / max )) * 90.0);
				hist.at<cv::Vec3b>(i, j)[0] = int(( 1.0- ( log(abs(v[i][j])) -min) / (max-min) ) * 90.0);
				hist.at<cv::Vec3b>(i, j)[1] = 255;
				hist.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
		cv::Mat output;
		cv::cvtColor(hist, output, CV_HSV2BGR);
		cv::imwrite("../img_hist/" + string(imgname) + string(".jpg"), output);
	}
	fclose(fpInfo);
}
void extractElement(const vector<vector<complex<double>>> &src, vector<double> &dst) {
	double sum = 0.0;
	dst.clear();
	int cx= src.size()/2, cy = src.back().size()/2, r = src.size()/2;
	for (int i=0; i< (src.size()+1) / 2; i++) {
		for (int j=0; j<src.back().size(); j++) {
			if ( (i-cx) * (i-cx) + (j-cy) * (j-cy) < (src.size()/2 - 9) * (src.size()*2 - 9) || (i==0 && j==0)) {
				continue;
			}
			dst.push_back(abs(src[i][j]));
			sum += abs(src[i][j]) * abs(src[i][j]);
		}
	}
	for (auto &d : dst) {
		d /= sqrt(sum) / dst.size();
	}
}
void adjustCenter(const vector<vector<double>> &points,
                  const int nCluster,
                  vector<vector<int>>    &vecClusteredSet,
                  vector<vector<double>> &centers) {
	for (int i=0; i<nCluster; i++) {
		vector<int> &clusteredSet = vecClusteredSet[i];
		vector<double> &center = centers[i];

		vector<pair<double, int>> vecPDistIdx;
		for (int j=0; j<clusteredSet.size(); j++) {
			vecPDistIdx.push_back({getEuclideanDist(points[clusteredSet[j]], center),
			                       clusteredSet[j]});
		}
		sort(vecPDistIdx.begin(), vecPDistIdx.end(), greater<pair<double, int>>());

		clusteredSet.clear();
		center = vector<double>(center.size(), 0.0);
		const int dim = center.size();

		for (int j=vecPDistIdx.size()/10; j<vecPDistIdx.size() ;j++) {
			clusteredSet.push_back(vecPDistIdx[j].second);

			for (int d=0; d<dim; d++) {
				center[d] += points[vecPDistIdx[j].second][d];
			}
		}
		for (int d=0; d<dim; d++) {
			center[d] /= clusteredSet.size();
		}
	}
}
void carRecognition(const clock_t &start_at) {
	FILE *fpInfo = fopen("../img_info.txt", "r");
	FILE *fpSample = fopen("../img_sample.txt", "r");
	char imgname[256]; int x, y, width, height;

	vector<vector<double>> points;
	vector<int> type;

	int countImgAnalysis = 0;
	while ( fscanf(fpInfo, "%s %d %d %d %d", imgname, &x, &y, &width, &height) != EOF) {
		countImgAnalysis++;
		string imgpath = string("../img/") + string(imgname) + string(".jpg");
		cv::Mat img = cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())
		{
			std::cout << "[!] Image load fail! : " << imgpath << endl;
			exit(1);
		}

		type.push_back(getType(imgname)-1);

		vector<vector<complex<double>>> v;
		vector<double> vec;
		get2dDTFS(img, v);
		extractElement(v, vec);

		points.push_back(vec);
	}
	printf("[CPU Time]= %12.6lf sec. Analysing %4d Images Completed.\n",
	       double(clock()-start_at)/CLOCKS_PER_SEC, countImgAnalysis);

	const int nCluster = getNumCluster(type);
	vector<vector<int>> vecClusteredSet; vector<vector<double>> centers;
	KMeansClustering(points, nCluster, vecClusteredSet, centers);
	adjustCenter(points, nCluster, vecClusteredSet, centers);
	printf("[CPU Time]= %12.6lf sec. Clustering Completed. \n", double(clock()-start_at)/CLOCKS_PER_SEC);

	// type predicated by the occurence of each clustered set
	vector<int> frequentType = getMostFrequentType(type, vecClusteredSet);

	vector<int> success(nCluster, 0);
	vector<int> total(nCluster, 0);
	vector<vector<int>> result(nCluster, vector<int>(nCluster,0));
	int countImgSample = 0;
	while( fscanf(fpSample, "%s %d %d %d %d", imgname, &x, &y, &width, &height) != EOF) {
		countImgSample++;
		string imgpath = string("../img/") + string(imgname) + string(".jpg");
		cv::Mat img = cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())
		{
			std::cout << "[!] Image load fail! : " << imgpath << endl;
			exit(1);
		}
		vector<vector<complex<double>>> v;
		vector<double> vec;
		get2dDTFS(img, v);
		extractElement(v, vec);

		int actualType = getType(imgname)-1;
		int estimatedType = frequentType[getNearestCenterIdx(vec, centers)];

		result[actualType][estimatedType]++;
		if (actualType == estimatedType) {
			success[actualType] = success[actualType] + 1;
			total[actualType] = total[actualType] + 1;
		}
		else {

			total[actualType] = total[actualType] + 1;
		}
	}
	printf("[CPU Time]= %12.6lf sec. Sampling %4d Images Completed.\n",
	       double(clock()-start_at)/CLOCKS_PER_SEC, countImgSample);

	for (int i=0; i<nCluster; i++) {
		printf("Type = %02d, Success Rate = %6.2lf%% (%2d / %2d)\n",
		       i+1, double(success[i]) / double(total[i]) * 100.0, success[i], total[i]);
	}
	
	for (int i=0; i<nCluster; i++) {
		printf("Type %02d, ", i+1);
		for (int j=0; j<nCluster; j++) {
			printf("%2d", result[i][j]);
			if (j!=nCluster-1) printf(" / ");
		}
		printf("\n");
	}

	int sumSucess=0, sumTotal = 0;
	for (int i=0; i<nCluster; i++) {
		sumSucess += success[i];
		sumTotal += total[i];
	}

	printf("Total Sucess Rate = %6.2lf%% (%2d / %2d)\n",
	       double(sumSucess) / double(sumTotal) * 100.0, sumSucess, sumTotal);


	printf("Centroids of the cluster\n");
	for (int i=0; i<centers.size(); i++) {
		const auto &center = centers[i];
		printf("( ");
		for (int d=0; d<center.size(); d++) {
			printf("%10.6g", center[d]);
			if (d!=center.size()-1) printf(", ");
		}
		printf(" )\n");
	}


	fclose(fpInfo);
}
int main() {
	cout << "OpenCV Version : " << CV_VERSION << endl;
	//get2dDTFSHistogram();
	carRecognition(clock());

	return 0;
}
