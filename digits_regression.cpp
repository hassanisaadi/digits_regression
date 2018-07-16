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
double computeCost(vector<vector<double>>y, vector<vector<double>>y_pred);
vector<vector<double>> softmax_label(vector<vector<double>>y);
double accuracy(vector<vector<double>> y_pred, vector<vector<double>> y);
vector<vector<double>> update_w(vector<vector<double>> W, double alpha, vector<vector<double>> err, vector<vector<double>>X);
vector<vector<double>> update_b(vector<vector<double>> b, double alpha, vector<vector<double>> err);
vector<vector<double>> matmul(vector<vector<double>>a, vector<vector<double>>b);
vector<vector<double>> transpose(vector<vector<double>> a);
vector<vector<double>> mult_scalar(vector<vector<double>> a, double alpha);
vector<vector<double>> matadd(vector<vector<double>> a, vector<vector<double>> b);
vector<vector<double>> matsub(vector<vector<double>> a, vector<vector<double>> b);
vector<vector<double>> activation(vector<vector<double>> y_pred);

vector<vector<double>> activation(vector<vector<double>> y_pred) {
	//y_pred = 10xn
	vector<vector<double>> y_act(y_pred.size(), vector<double>(y_pred[0].size(), 0));
	for (int c = 0; c < y_pred.size(); c++) {
		for (int n = 0; n < y_pred[0].size(); n++) {
			if (y_pred[c][n] >= 0)
				y_act[c][n] = y_pred[c][n];
			else
				y_act[c][n] = 0.0;
		}
	}
	return y_act;
}

vector<vector<double>> matadd(vector<vector<double>> a, vector<vector<double>> b) {
	vector<vector<double>> c(a.size(), vector<double>(a[0].size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
	return c;
}

vector<vector<double>> matsub(vector<vector<double>> a, vector<vector<double>> b) {
	vector<vector<double>> c(a.size(), vector<double>(a[0].size(), 0));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[0].size(); j++) {
			c[i][j] = a[i][j] - b[i][j];
		}
	}
	return c;
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
		//if (9*y[i][0] - y_pred[i][0] < 0.4)
		if (y[i][0] - y_pred[i][0] < 0.4)
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
			for (int j = 0; j < X[0].size(); j++) {
				XX[0][j] = X[n][j];
			}
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
	assert(y.size() == y_pred.size());
	assert(y[0].size() == y_pred[0].size());
	//y = 10xn
	//y_pred = 10xn
	//err = 10xn
	vector<vector<double>> err(y.size(), vector<double>(y[0].size(),0));
	for (int c = 0; c < y.size(); c++) {
		for (int n = 0; n < y[0].size(); n++) {
			err[c][n] = 0.5 * pow(y[c][n] - y_pred[c][n], 2);
		}
	}
	return err;
}

double computeCost(vector<vector<double>>y, vector<vector<double>>y_pred) {
	//y = nx1
	//y_pred = nx1
	double c = 0;
	for (int n = 0; n < y.size(); n++) {
		c += 0.5 * pow(y[n][0] - y_pred[n][0], 2);
	}
	return c;
}

vector<vector<double>> softmax_label(vector<vector<double>>y) {
	//y = 10xn
	//label = nx1
	//return the predicted LABEL
	vector<vector<double>> label(y[0].size(), vector<double>(1,0));
	for (int n = 0; n < y[0].size(); n++) {
		double max_score = -1000000.0;
		for (int c = 0; c < y.size(); c++) {
			if (y[c][n] > max_score) {
				max_score = y[c][n];
				label[n][0] = c;
			}
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
	vector<vector<double>> errXalpha = mult_scalar(errX, alpha); 
	w_new = matsub(W, errXalpha);
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
		//err_m[i][0] /= err[0].size();
	}

	vector<vector<double>> err_malpha = mult_scalar(err_m, alpha);
	b_new = matsub(b, err_malpha);
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

	cout << X.size() << " sample loaded." << endl;

	//shuffling data
	random_shuffle(std::begin(X), std::end(X));

	//normalizing data
	for (int i = 0; i < X.size(); i++) {
		for (int j = 0; j < X[0].size() - 1; j++) {
			X[i][j] = 2*(X[i][j] / 16) - 1;
		}
		//X[i][X[0].size()-1] /= 9;
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
	int MAX_ITER = 10;
	int NCLASS = 10;
	double LEARNINGRATE = 0.001;
	int BATCHSIZE = 140;
	vector<vector<double>> W(NCLASS, vector<double>(x_train[0].size(),0));
	vector<vector<double>> b(NCLASS, vector<double>(1,0));

	//random W initialization
	for (int c = 0; c < NCLASS; c++) {
		for (int s = 0; s < x_train[0].size(); s++) {
			W[c][s] = 2 * ((double)rand() / RAND_MAX) - 1;
		}
	}

	vector<vector<double>> x_train_shf(x_train.size(), vector<double>(x_train[0].size(), 0));
	vector<vector<double>> y_train_shf(y_train.size(), vector<double>(y_train[0].size(), 0));
	int iter = 0;
	while(iter < MAX_ITER){
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

		vector<vector<double>> x_train_shf_b(BATCHSIZE, vector<double>(x_train_shf[0].size(), 0));
		vector<vector<double>> y_train_shf_b(BATCHSIZE, vector<double>(y_train_shf[0].size(), 0));

		for (int bs = 0; bs < int(x_train_shf.size() / BATCHSIZE); bs++) {
			for (int idx = 0; idx < BATCHSIZE; idx++) {
				x_train_shf_b[idx] = x_train_shf[bs*BATCHSIZE+idx];
				y_train_shf_b[idx] = y_train_shf[bs*BATCHSIZE+idx];
			}

			vector<vector<double>> y_one_hot = one_hot(y_train_shf_b, NCLASS); // 10xn

			vector<vector<double>> y_pred = my_feed_forward(W, b, x_train_shf_b); //10xn

			vector<vector<double>> y_act = activation(y_pred); //ReLU

			vector<vector<double>> err = computeError(y_one_hot, y_pred); //10xn

			vector<vector<double>> y_pred_f = softmax_label(y_pred); //nx1

			double tr_acc = accuracy(y_pred_f, y_train_shf_b);
			double cost = computeCost(y_train_shf_b, y_pred_f);
			cout << "Accuracy and cost for iteration " << iter << " and batch " << bs << " are " << tr_acc << "\t" << cost << endl;

			W = update_w(W, LEARNINGRATE, err, x_train_shf_b);
			b = update_b(b, LEARNINGRATE, err);
		}
		iter++;
	}
	cout << "Training Done!" << endl;

	vector<vector<double>> y_pred_test = my_feed_forward(W, b, x_test); //10xn
	double te_acc = accuracy(softmax_label(y_pred_test), y_test);
	cout << "Accuracy for Test is " << te_acc << endl;

	getchar();
	return 0;
}
