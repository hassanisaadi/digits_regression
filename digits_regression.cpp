// logistic_regression.cpp : Defines the entry point for the console application.
//

#include <algorithm>
#include <random>

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <iomanip>

#include <sstream>
#include <string>
#include <algorithm>
#include <ctgmath>

using namespace std;

vector<vector<double>> my_feed_forward(vector<vector<double>> W, vector<vector<double>> b, vector<vector<double>>X);
vector<vector<double>> one_hot(vector<vector<double>> y, int NCLASS);
vector<vector<double>> computeError(vector<vector<double>> y, vector<vector<double>> y_pred);
//double computeCost(vector<vector<double>>y, vector<vector<double>>y_pred);
vector<vector<double>> softmax_label(vector<vector<double>>y);
double accuracy(vector<vector<double>> y_pred, vector<vector<double>> y);
vector<vector<double>> update_w(vector<vector<double>> W, double alpha, vector<vector<double>> err, vector<vector<double>>X);
//vector<vector<double>> update_b(vector<vector<double>> b, double alpha, vector<vector<double>> err);
vector<vector<double>> matmul(vector<vector<double>>a, vector<vector<double>>b);
vector<vector<double>> transpose(vector<vector<double>> a);
vector<vector<double>> mult_scalar(vector<vector<double>> a, double alpha);
vector<vector<double>> matadd(vector<vector<double>> a, vector<vector<double>> b);
vector<vector<double>> matsub(vector<vector<double>> a, vector<vector<double>> b);

vector<vector<double>> matadd(vector<vector<double>> a, vector<vector<double>> b) {
	vector<vector<double>> c(a.size(), vector<double>(a[0].size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
	return b;
}

vector<vector<double>> matsub(vector<vector<double>> a, vector<vector<double>> b) {
	vector<vector<double>> c(a.size(), vector<double>(a[0].size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			c[i][j] = a[i][j] - b[i][j];
		}
	}
	return b;
}


vector<vector<double>> mult_scalar(vector<vector<double>> a, double alpha) {
	vector<vector<double>> b(a.size(), vector<double>(a[0].size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			b[i][j] = alpha * a[i][j];
		}
	}
	return b;
}

vector<vector<double>> matmul(vector<vector<double>>a, vector<vector<double>>b) {
	assert(a[0].size() == b.size());
	vector<vector<double>> m(a.size(), vector<double>(b[0].size(),0));
	for (int i = 0; i < a.size(); ++i)
		for (int j = 0; j < b[0].size(); ++j)
			for (int k = 0; k < b.size(); ++k){
				m[i][j] += a[i][k] * b[k][j];
			}
	return m;
}

double accuracy(vector<vector<double>> y_pred, vector<vector<double>> y) {
	//y = nx1
	//y_pred = nx1
	double acc = 0;
	for (int i = 0; i < y.size(); i++) {
		if (y[i][0] == y_pred[i][0])
			acc++;
	}
	acc /= y.size();
	return acc;
}

vector<vector<double>> transpose(vector<vector<double>> a) {
	vector<vector<double>> outtrans(a[0].size(), vector<double>(a.size()));
	for (int i = 0; i < a.size(); ++i)
		for (int j = 0; j < a[0].size(); ++j)
			outtrans[j][i] = a[i][j];
	return outtrans;
}

vector<vector<double>> my_feed_forward(vector<vector<double>> W, vector<vector<double>> b, vector<vector<double>>X) {
	vector<vector<double>>Y(b.size(), vector<double>(X.size(),0));
	//X = nx64
	//W = 10x64
	//b = 10x1
	//Y = 10xn
	for (int c = 0; c < W.size(); c++) {
		for (int n = 0; n < X.size(); n++) {
			vector<vector<double>> XX(1, vector<double>(X[0].size(),0));
			XX.push_back(X[n]);  //1x64
			vector<vector<double>> WXX = matmul(W, transpose(XX));
			Y[c][n] = WXX[c][0] + b[c][0];
		}
	}
	return Y;
}

vector<vector<double>> one_hot(vector<vector<double>> y, int NCLASS) {
	//y = nx1
	//y_one_hot = 10xn
	vector<vector<double>> y_one_hot(NCLASS, vector<double>(y.size(),0));
	
	for (int sample = 0; sample < y.size(); sample++) {
		if      (y[sample][0] == 0)
			y_one_hot[0][sample] = 1;
		else if (y[sample][0] == 1)
			y_one_hot[1][sample] = 1;
		else if (y[sample][0] == 2)
			y_one_hot[2][sample] = 1;
		else if (y[sample][0] == 3)
			y_one_hot[3][sample] = 1;
		else if (y[sample][0] == 4)
			y_one_hot[4][sample] = 1;
		else if (y[sample][0] == 5)
			y_one_hot[5][sample] = 1;
		else if (y[sample][0] == 6)
			y_one_hot[6][sample] = 1;
		else if (y[sample][0] == 7)
			y_one_hot[7][sample] = 1;
		else if (y[sample][0] == 8)
			y_one_hot[8][sample] = 1;
		else if (y[sample][0] == 9)
			y_one_hot[9][sample] = 1;
	}
	return y_one_hot;

}


vector<vector<double>> computeError(vector<vector<double>> y, vector<vector<double>> y_pred) {
	//y = 10xn
	//y_pred = 10xn
	//err = 10xn
	vector<vector<double>>err(y.size(), vector<double>(y[0].size(),0));
	for (int c = 0; c < y.size(); c++) {
		for (int n = 0; n < y[0].size(); n++) {
			err[c][n] = 0.5 * pow(y[c][n] - y_pred[c][n], 2);
		}
	}
	return err;
}

/*double computeCost(vector<vector<double>>y, vector<vector<double>>y_pred) {
	//y = nx1
	//y_pred = nx1
	double c = 0;
	for (int n = 0; n < y.size(); n++) {
		c += abs(y[c][0] - y_pred[c][0]);
	}
	return c;
}*/

vector<vector<double>> softmax_label(vector<vector<double>>y) {
	//y = 10xn
	//label = nx1
	//return the predicted LABEL
	vector<vector<double>> label(y[0].size(), vector<double>(1,0));
	for (int n = 0; n < y[0].size(); n++) {
		for (int c = 0; c < y.size(); c++) {
			double max_score = 0;
			if (y[c][n] > max_score)
				max_score = y[c][n];
				label[n][0] = c;
		}
	}
	return label;
}

vector<vector<double>> update_w(vector<vector<double>> W, double alpha, vector<vector<double>> err, vector<vector<double>>X) {
	//W = 10x64
	//err = 10xn
	//X = nx64
	vector<vector<double>> w_new(err.size(), vector<double>(W[0].size(),0));
	vector<vector<double>> errX = matmul(err, X);  //10x64
	errX = mult_scalar(errX, alpha);
	w_new = matsub(W, errX);
	return w_new;
}

vector<vector<double>> update_b(vector<vector<double>> b, double alpha, vector<vector<double>> err) {
	//b = 10x1
	//err = 10xn
	vector<vector<double>> b_new(b.size(), vector<double>(1,0));
	vector<vector<double>> err_m(b.size(), vector<double>(1,0)); //10x1

	for (int i = 0; i < b.size(); i++) {
		err_m[i][0] = 0;
		for (int j = 0; j < err[0].size(); j++) {
			err_m[i][0] += err[i][j];
		}
		err_m[i][0] /= err[0].size();
	}

	err_m = mult_scalar(err_m, alpha);
	b_new = matsub(b, err_m);
	return b_new;
}

int main() {
	//loading data
	vector<vector<double>> X;
	ifstream in("digits.csv");
	if (!in)
		cout << "File not found" << endl;

	string line, sector;
	while (getline(in, line))
	{
		vector<double> v;
		stringstream ss(line);
		while (getline(ss, sector, ','))
			v.push_back(stod(sector));
		X.push_back(v);
	}

	//shuffling data
	random_shuffle(std::begin(X), std::end(X));

	//normalizing data
	for (int i = 0; i < X.size(); i++) {
		for (int j = 0; j < X[0].size() - 1; j++) {
			X[i][j] /= 16;
		}
	}

	//train and test dataset
	int cidx = int(0.8*X.size());
	vector<vector<double>> x_train;
	vector<vector<double>> y_train;
	vector<vector<double>> x_test;
	vector<vector<double>> y_test;

	for (int i = 0; i < cidx; i++) {
		vector <double> row;
		vector <double> label;
		for (int j = 0; j<64; j++)
			row.push_back(X[i][j]);
		label.push_back(X[i][64]);
		x_train.push_back(row);
		y_train.push_back(label);
	}

	for (int i = cidx; i < X.size(); i++) {
		vector <double> row;
		vector <double> label;
		for (int j = 0; j<64; j++)
			row.push_back(X[i][j]);
		label.push_back(X[i][64]);
		x_test.push_back(row);
		y_test.push_back(label);
	}

	//training
	int MAX_ITER = 4;
	int NCLASS = 10;
	double LEARNINGRATE = 0.01;
	vector<vector<double>> W(NCLASS, vector<double>(x_train[0].size(),0.01));
	vector<vector<double>> b(NCLASS, vector<double>(1,0));
	vector<vector<double>> x_train_shf(x_train.size(), vector<double>(x_train[0].size(), 0));
	vector<vector<double>> y_train_shf(y_train.size(), vector<double>(y_train[0].size(), 0));
	for (int iter = 0; iter < MAX_ITER; iter++) {
		//shuffle data
		std::vector<int> indexes;
		indexes.reserve(x_train.size());
		for (int i = 0; i < x_train.size(); ++i)
			indexes.push_back(i);
		std::random_shuffle(indexes.begin(), indexes.end());
		int p = 0;
		for (std::vector<int>::iterator it1 = indexes.begin(); it1 != indexes.end(); ++it1) {
			x_train_shf[p] = x_train[*it1];
			y_train_shf[p] = y_train[*it1];
			p++;
		}

		vector<vector<double>> y_one_hot = one_hot(y_train_shf, NCLASS); // 10xn

		vector<vector<double>> y_pred = my_feed_forward(W, b, x_train_shf); //10xn

		vector<vector<double>> err = computeError(y_one_hot, y_pred); //10xn
		
		double tr_acc = accuracy(softmax_label(y_pred), y_train_shf);
		cout << "Accuracy for iteration " << iter << " is " << tr_acc << endl;

		W = update_w(W, LEARNINGRATE, err, x_train_shf);
		b = update_b(b, LEARNINGRATE, err);
				
	}

	getchar();
	return 0;
}