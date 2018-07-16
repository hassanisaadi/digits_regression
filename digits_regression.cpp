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
//vector<vector<double>> computeError(vector<vector<double>> y, vector<vector<double>> y_pred);
//double computeCost(vector<vector<double>>y, vector<vector<double>>y_pred);
double computeLoss(vector<vector<double>> Y, vector<vector<double>> Y_hat);
vector<vector<double>> thr_label(vector<vector<double>>y);
double accuracy(vector<vector<double>> y_pred, vector<vector<double>> y);
vector<vector<double>> update_w(vector<vector<double>> W, double alpha, vector<vector<double>> dW);
vector<vector<double>> update_b(vector<vector<double>> b, double alpha, vector<vector<double>> db);
vector<vector<double>> activation(vector<vector<double>> y_pred);
double inner_product(vector<double> W, vector<double> X);
vector<vector<double>> gradient_w(vector<vector<double>> X, vector<vector<double>> Y, vector<vector<double>> Y_hat);
vector<vector<double>> gradient_b(vector<vector<double>> Y, vector<vector<double>> Y_hat);

vector<vector<double>> matmul(vector<vector<double>>a, vector<vector<double>>b);
vector<vector<double>> transpose(vector<vector<double>> a);
vector<vector<double>> mult_scalar(vector<vector<double>> a, double alpha);
vector<vector<double>> matadd(vector<vector<double>> a, vector<vector<double>> b);
vector<vector<double>> matsub(vector<vector<double>> a, vector<vector<double>> b);

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
			X[i][j] = 2 * (X[i][j] / 16) - 1;
			//X[i][j] = X[i][j] / 16;
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
	double LEARNINGRATE = 0.0001;
	int BATCHSIZE = 1;
	vector<vector<double>> W(NCLASS, vector<double>(x_train[0].size(),0)); //10x64
	vector<vector<double>> b(NCLASS, vector<double>(1,0.01)); //10x1

	//random W initialization
	for (int c = 0; c < NCLASS; c++) {
		for (int s = 0; s < x_train[0].size(); s++) {
			W[c][s] = 2 * ((double)rand() / RAND_MAX) - 1;
			//W[c][s] = (double)rand() / RAND_MAX;
		}
	}

	vector<vector<double>> x_train_shf(x_train.size(), vector<double>(x_train[0].size(), 0));
	vector<vector<double>> y_train_shf(y_train.size(), vector<double>(y_train[0].size(), 0));
	int iter = 0;
	double avg_tr_acc = 0;
	int num_tr_acc = 0;
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
			vector<vector<double>> Z = my_feed_forward(W, b, x_train_shf_b); //10xn
			vector<vector<double>> Y_pred = activation(Z); //10xn
			vector<vector<double>> y_one_hot = one_hot(y_train_shf_b, NCLASS); // 10xn
			vector<vector<double>> dW = gradient_w(x_train_shf_b, y_one_hot, Y_pred); //10x64
			vector<vector<double>> db = gradient_b(y_one_hot, Y_pred); //10x1
			W = update_w(W, LEARNINGRATE, dW);
			b = update_b(b, LEARNINGRATE, db);
			double loss = computeLoss(y_one_hot, Y_pred);
			vector<vector<double>> y_pred_f = thr_label(Y_pred);
			double tr_acc = accuracy(y_pred_f, y_train_shf_b);
			avg_tr_acc += tr_acc;
			num_tr_acc++;
			printf("iter = %2d \t batch = %4d \t acc = %.3f \t loss = %.4f\n", iter, bs, tr_acc, loss);
			//cout << "iter = " << iter << "\t batch = " << bs << "\t acc = " << tr_acc << "\t loss = " << loss << endl;
		}
		iter++;
	}
	avg_tr_acc /= num_tr_acc;
	std::cout << "Average train accuracy = " << avg_tr_acc << endl;
	std::cout << "Training Done!" << endl;

	vector<vector<double>> Z_test = my_feed_forward(W, b, x_test); //10xn
	vector<vector<double>> y_pred_test = activation(Z_test); //10xn
	vector<vector<double>> y_pred_f_test = thr_label(y_pred_test);
	double te_acc = accuracy(y_pred_f_test, y_test);
	cout << "Test accuracy = " << te_acc << endl;

	getchar();
	return 0;
}


vector<vector<double>> activation(vector<vector<double>> Z) {
	//Z = 10xn
	//y_act = 10xn
	vector<vector<double>> y_act(Z.size(), vector<double>(Z[0].size(), 0));
	for (int c = 0; c < Z.size(); c++) {
		for (int n = 0; n < Z[0].size(); n++) {
			y_act[c][n] = 1 / (1 + exp(Z[c][n]));
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
	vector<vector<double>> m(a.size(), vector<double>(b[0].size(), 0));
	for (int i = 0; i < a.size(); ++i)
		for (int j = 0; j < b[0].size(); ++j)
			for (int k = 0; k < b.size(); ++k) {
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

double inner_product(vector<double> W, vector<double> X) {
	assert(W.size() == X.size());
	double p = 0;
	for (int i = 0; i < W.size(); i++) {
		p += W[i] * X[i];
	}
	return p;
}

vector<vector<double>> my_feed_forward(vector<vector<double>> W, vector<vector<double>> b, vector<vector<double>>X) {
	vector<vector<double>>Z(b.size(), vector<double>(X.size(), 0));
	//X = nx64
	//W = 10x64
	//b = 10x1
	//Z = 10xn
	for (int c = 0; c < W.size(); c++) {
		for (int n = 0; n < X.size(); n++) {
			Z[c][n] = inner_product(W[c], X[n]) + b[c][0];
		}
	}
	return Z;
}

vector<vector<double>> one_hot(vector<vector<double>> y, int NCLASS) {
	//y = nx1
	//y_one_hot = 10xn
	vector<vector<double>> y_one_hot(NCLASS, vector<double>(y.size(), 0));

	for (int sample = 0; sample < y.size(); sample++) {
		if (y[sample][0] == 0)
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
		else
			cerr << "Invalid Label" << endl;
	}
	return y_one_hot;

}


/*vector<vector<double>> computeError(vector<vector<double>> y, vector<vector<double>> y_pred) {
	assert(y.size() == y_pred.size());
	assert(y[0].size() == y_pred[0].size());
	//y = 10xn
	//y_pred = 10xn
	//err = 10xn
	vector<vector<double>> err(y.size(), vector<double>(y[0].size(), 0));
	for (int c = 0; c < y.size(); c++) {
		for (int n = 0; n < y[0].size(); n++) {
			err[c][n] = 0.5 * pow(y[c][n] - y_pred[c][n], 2);
		}
	}
	return err;
}*/

/*double computeCost(vector<vector<double>>y, vector<vector<double>>y_pred) {
	//y = nx1
	//y_pred = nx1
	double c = 0;
	for (int n = 0; n < y.size(); n++) {
		c += 0.5 * pow(y[n][0] - y_pred[n][0], 2);
	}
	return c;
}*/

vector<vector<double>> thr_label(vector<vector<double>>y) {
	//This function threshold the output value (0.5) and reverse one_hot.
	//y = 10xn
	//label = nx1
	vector<vector<double>> label(y[0].size(), vector<double>(1, 0));
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

vector<vector<double>> update_w(vector<vector<double>> W, double alpha, vector<vector<double>> dW) {
	//W = 10x64
	//dW = 10x64
	vector<vector<double>> dW_alpha = mult_scalar(dW, alpha);
	vector<vector<double>> w_new = matsub(W, dW_alpha);
	return w_new;
}

vector<vector<double>> update_b(vector<vector<double>> b, double alpha, vector<vector<double>> db) {
	//b = 10x1
	//db = 10x1
	vector<vector<double>> db_alpha = mult_scalar(db, alpha);
	vector<vector<double>> b_new = matsub(b, db_alpha);
	return b_new;
}

vector<vector<double>> gradient_w(vector<vector<double>> X, vector<vector<double>> Y, vector<vector<double>> Y_hat) {
	//X = nx64
	//Y = 10xn (one_hot)
	//Y_hat = 10xn
	//dW = 10x64
	vector<vector<double>> dW(Y.size(), vector<double>(X[0].size(),0));
	for (int c = 0; c < Y.size(); c++) {
		for (int f = 0; f < X[0].size(); f++) {
			for (int n = 0; n < Y[0].size(); n++) {
				dW[c][f] += (Y[c][n] - Y_hat[c][n]) * (Y_hat[c][n]) * (1 - Y_hat[c][n]) * X[n][f];
			}
		}
	}
	return dW;
}
vector<vector<double>> gradient_b(vector<vector<double>> Y, vector<vector<double>> Y_hat) {
	//Y = 10xn (one_hot)
	//Y_hat = 10xn
	//db = 10x1
	vector<vector<double>> db(Y.size(), vector<double>(1, 0));
	for (int c = 0; c < Y.size(); c++) {
		for (int n = 0; n < Y[0].size(); n++) {
			db[c][0] += (Y[c][n] - Y_hat[c][n]) * (Y_hat[c][n]) * (1 - Y_hat[c][n]);
		}
	}
	return db;
}

double computeLoss(vector<vector<double>> Y, vector<vector<double>> Y_hat) {
	//Y and Y_hat = 10xn
	double loss = 0;
	for (int c = 0; c < Y.size(); c++) {
		for (int n = 0; n < Y[0].size(); n++) {
			loss += pow(Y[c][n]-Y_hat[c][n], 2);
		}
	}
	loss /= 0.5;
	return loss;
}
