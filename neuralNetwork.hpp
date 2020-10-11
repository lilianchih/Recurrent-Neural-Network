#ifndef __NEURALNETWORK__HPP__
#define __NEURALNETWORK__HPP__

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
using namespace Eigen;

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <bitset>
#include <math.h>
#include <complex>
#include <cmath>
#include <vector>
using namespace std;

#include <omp.h>
#define EIGEN_DONT_PARALLELIZE
#define DBL(i) ((i).cast<double>())

class Model{
public:
    Model(int d, int l, int o); //dimension of (input, hidden, output)
    MatrixXd Whx;
    MatrixXd Whh;
    MatrixXd Wyh;
    VectorXd bh;
    VectorXd by;
    Model(const Model& model);
};

Model::Model(const Model& model){
    Whx = model.Whx;
    Whh = model.Whh;
    Wyh = model.Wyh;
    bh = model.bh;
    by = model.by;
}

class Cache{
public:
    Cache(int d, int l, int o);
    VectorXi input;
    VectorXd prev_hidden;
    VectorXd hidden;
    VectorXd output;
    VectorXd gh;
};

class History{
public:
    History(int d, int l, int o, int T);
    MatrixXi input_t;
    MatrixXd hidden_t;
    MatrixXd output_t;
};


class Gradient{
public:
    Gradient(int d, int l, int o);
    MatrixXd gWhx;
    MatrixXd gWhh;
    MatrixXd gWyh;
    VectorXd gbh;
    VectorXd gby;
    void step_backward(const Model& model, Cache& cache, int y);
    Gradient operator+(Gradient& temp);
    Gradient operator-(Gradient& temp);
    Gradient operator/(int N);
    Gradient operator*(double c);
    Gradient& operator=(Gradient temp);
};

Gradient Gradient::operator+(Gradient& temp){
    Gradient result(gWhx.cols(), gbh.size(), gby.size());
    result.gWhx = gWhx + temp.gWhx;
    result.gWhh = gWhh + temp.gWhh;
    result.gWyh = gWyh + temp.gWyh;
    result.gbh = gbh + temp.gbh;
    result.gby = gby + temp.gby;
    return result;
}

Gradient Gradient::operator-(Gradient& temp){
    Gradient result(gWhx.cols(), gbh.size(), gby.size());
    result.gWhx = gWhx - temp.gWhx;
    result.gWhh = gWhh - temp.gWhh;
    result.gWyh = gWyh - temp.gWyh;
    result.gbh = gbh - temp.gbh;
    result.gby = gby - temp.gby;
    return result;
}

Gradient Gradient::operator/(int N){
    Gradient result(gWhx.cols(), gbh.size(), gby.size());
    result.gWhx = gWhx/N;
    result.gWhh = gWhh/N;
    result.gWyh = gWyh/N;
    result.gbh = gbh/N;
    result.gby = gby/N;
    return result;
}

Gradient Gradient::operator*(double c){
    Gradient result(gWhx.cols(), gbh.size(), gby.size());
    result.gWhx = gWhx*c;
    result.gWhh = gWhh*c;
    result.gWyh = gWyh*c;
    result.gbh = gbh*c;
    result.gby = gby*c;
    return result;
}

Gradient& Gradient::operator=(Gradient temp){
    gWhx = temp.gWhx;
    gWhh = temp.gWhh;
    gWyh = temp.gWyh;
    gbh = temp.gbh;
    gby = temp.gby;
    return *this;
}


#endif
