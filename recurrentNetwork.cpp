#include "neuralNetwork.hpp"

//probability distribution (d-dim, time)

MatrixXi batch(int M, const MatrixXd& data){
    int sampling_step = 5;
    int T = data.rows()/sampling_step;
    MatrixXi samples(M, T);
    MatrixXd r = 0.5*(MatrixXd::Random(M, T) + MatrixXd::Ones(M, T));
    int d = 7;
    
    for (int m=0; m<M; m++){
        for (int t=0; t<T; t++){
            double sum = 0;
            int ind = 0;
            while (r(m, t) > sum){
                ind ++;
                sum += data(t*sampling_step, ind);
            }
            samples(m, t) = ind-1;
        }
    }
    return samples;
}

Model::Model(int d, int l, int o){
    srand((unsigned int) time(0));
    Whx = MatrixXd::Random(l, d)*0.5;
    Whh = MatrixXd::Random(l, l)*0.5;
    Wyh = MatrixXd::Random(o, l)*0.5;
    bh = VectorXd::Random(l)*2;
    by = VectorXd::Random(o)*2;
}

Cache::Cache(int d, int l, int o){
    input = VectorXi::Zero(d);
    prev_hidden = VectorXd::Zero(l);
    hidden = VectorXd::Zero(l);
    output = VectorXd::Zero(o);
    gh = VectorXd::Zero(l);
}

History::History(int d, int l, int o, int T){
    input_t = MatrixXi::Zero(d, T);
    hidden_t = MatrixXd::Zero(l, T);
    output_t = MatrixXd::Zero(o, T);
}

Gradient::Gradient(int d, int l, int o){
    gWhx = MatrixXd::Zero(l, d);
    gWhh = MatrixXd::Zero(l, l);
    gWyh = MatrixXd::Zero(o, l);
    gbh = VectorXd::Zero(l);
    gby = VectorXd::Zero(o);
}

VectorXd softmax(VectorXd x){
    VectorXd exponential = (x.array()).exp();
    return exponential / exponential.sum();
}

MatrixXi ind2vec(const VectorXi& x, int dim){
    MatrixXi result = MatrixXi::Zero(dim, x.size());
    for (int i=0; i<x.size(); i++){
        result(x(i), i) = 1;
    }
    return result;
}

VectorXd forward(const Model& model, const VectorXi& input, VectorXd& hidden){
    VectorXd theta = model.Whx * DBL(input) + model.Whh * hidden + model.bh;
    hidden = theta.array().tanh();
    VectorXd z = model.Wyh * hidden + model.by;
    return softmax(z);
}

double cross_entropy(int y, VectorXd P){
    //VectorXd ret = (P.array()).log();
    return -log(P(y));
}

void Gradient::step_backward(const Model& model, Cache& cache, int y){
    VectorXd gz = cache.output;
    gz(y) -= 1;
    VectorXd gtheta = VectorXd::Ones(cache.hidden.size()) - cache.hidden.cwiseAbs2();
    cache.gh += (model.Wyh.transpose() * gz);
    gtheta = cache.gh.cwiseProduct(gtheta);
    
    gWyh = gz * cache.hidden.transpose();
    gWhx = gtheta * DBL(cache.input.transpose());
    gWhh = gtheta * cache.prev_hidden.transpose();
    gby = gz;
    gbh = gtheta;
    cache.gh = model.Whh.transpose() * gtheta;
}

Gradient backward(const Model& model, const History& history, const VectorXi& sample){
    int T = sample.size();
    Gradient result(model.Whx.cols(), model.bh.size(), model.by.size());
    Gradient step(model.Whx.cols(), model.bh.size(), model.by.size());
    Cache cache(model.Whx.cols(), model.bh.size(), model.by.size());
    for (int t=T-2; t>=0; t--){
        cache.input = history.input_t.col(t);
        if (t>0)
            cache.prev_hidden = history.hidden_t.col(t-1);
        else
            cache.prev_hidden = VectorXd::Zero(model.bh.size());
        cache.hidden = history.hidden_t.col(t);
        cache.output = history.output_t.col(t);
        step.step_backward(model, cache, sample(t+1));

        result = result + step;
    }
    return result;
}

void descent(Model& model, double rate, Gradient& grad){
    model.Whx -= rate * grad.gWhx;
    model.Whh -= rate * grad.gWhh;
    model.Wyh -= rate * grad.gWyh;
    model.bh -= rate * grad.gbh;
    model.by -= rate * grad.gby;
}

MatrixXd prediction(const Model& model, const VectorXd& initial, int T, int M){
    int d = initial.size();
    MatrixXd result = MatrixXd::Zero(d, T);
    MatrixXd r = 0.5*(MatrixXd::Random(M, T) + MatrixXd::Ones(M, T));
    VectorXd prob(d);
    MatrixXi traj(d, T);
    double ri;
    for (int m=0; m<M; m++){
        VectorXi input = VectorXi::Zero(d);
        ri = rand();
        double sum = 0;
        int ind = 0;
        while (ri < sum){
            sum += initial(ind);
            ind ++;
        }
        input(ind-1) = 1;
        
        VectorXd hidden = VectorXd::Zero(model.bh.size());
        for (int t=0; t<T; t++){
            prob = forward(model, input, hidden);
            input = VectorXi::Zero(d);
            double sum = 0;
            int ind = 0;
            while (r(m, t) > sum){
                sum += prob(ind);
                ind ++;
            }
            input(ind-1) = 1;
            
            traj.col(t) = input;
        }
        result += DBL(traj);
    }
    result /= M;
    return result;
}

MatrixXd loadParam(string filename, int N, int M){
    MatrixXd param = MatrixXd::Zero(N, M);
    ifstream inparam;
    inparam.open(filename);
    
    for(int i=0; i<N; i++){
        for (int j=0; j<M; j++){
            inparam >> param(i, j);
        }
    }
    return param;
}
template <class T>
void storeResult(T& result, string filename){
    ofstream outFile;
    outFile.open(filename);
    outFile<<result;
    outFile.close();
}

int main(){  
    
    int input_size = 7;
    int hidden_size = 21;
    int output_size = 7;
    Model model(input_size, hidden_size, output_size);
    
    //model.Whx = loadParam("Whx", hidden_size, input_size);
    //model.Whh = loadParam("Whh", hidden_size, hidden_size);
    //model.Wyh = loadParam("Wyh", output_size, hidden_size);
    //model.bh = loadParam("bh", hidden_size, 1);
    //model.by = loadParam("by", output_size, 1);
    
    MatrixXd data = loadParam("data2.txt", 225, 8); //450
    //VectorXd time = data.col(0);
    int batch_size = 20000;
    MatrixXi samples = batch(batch_size, data);
    MatrixXd batch_avg = MatrixXd::Zero(input_size, samples.cols());
    for (int m=0; m<batch_size; m++){
        batch_avg += DBL(ind2vec(samples.row(m).transpose(), input_size));
    }
    batch_avg /= batch_size;
    storeResult(batch_avg, "batch");
    
    double lr = 0.0001;
    
    double error;
    double total_loss_t[8];
    double total_loss;
    int epochs = 1000;

    for (int i=0; i<epochs; i++){ //update parameters
        vector<Gradient> parallel(8, Gradient(input_size, hidden_size, output_size));
        
#pragma omp parallel num_threads(8)
{
    int tid = omp_get_thread_num();
    total_loss_t[tid] = 0;
    //Model model_t(model);
    Gradient temp_g(input_size, hidden_size, output_size);
#pragma omp for 
        for (int m=0; m<batch_size; m++){
            History history(input_size, hidden_size, output_size, samples.cols()-1);
            VectorXd temp_h = VectorXd::Zero(hidden_size);
            history.input_t = ind2vec(samples.block(m, 0, 1, samples.cols()-1).transpose(), input_size);
            for (int t=0; t<samples.cols()-1; t++){
                history.output_t.col(t) = forward(model, history.input_t.col(t), temp_h);
                history.hidden_t.col(t) = temp_h;
                total_loss_t[tid] += cross_entropy(samples(m, t+1), history.output_t.col(t));
            }
            temp_g = backward(model, history, samples.row(m));
            parallel[tid] = parallel[tid] + temp_g;
        }
}
        Gradient avg_g(input_size, hidden_size, output_size);
        total_loss = 0;
        for (int tid=0; tid<8; tid++){
             avg_g = avg_g + parallel[tid];
             total_loss += total_loss_t[tid];
        } 
        avg_g = avg_g / batch_size;
        
        descent(model, lr, avg_g);
        
        if ((i+1)%10 == 0){
            cout<<"Epoch "<<(i+1)<<"\t";
            error = avg_g.gWhx.cwiseAbs().maxCoeff();
            cout<<setprecision(6)<<fixed;
            cout<<"Errors "<<error<<"\t";
            error = avg_g.gWhh.cwiseAbs().maxCoeff();
            cout<<error<<"\t";
            error = avg_g.gWyh.cwiseAbs().maxCoeff();
            cout<<error<<"\t";
            total_loss /= batch_size;
            cout<<"Loss "<<total_loss<<endl;
            
        }
        if ((i+1)%1000 == 0){
            storeResult(model.Whx, "Whx");
            storeResult(model.Whh, "Whh");
            storeResult(model.Wyh, "Wyh");
            storeResult(model.bh, "bh");
            storeResult(model.by, "by");
        }
        lr = max(lr*0.9995, 0.0005);
    }
    
    
    int traj_num = 100000;
    int T_pred = samples.cols()*1;
    VectorXd initial = data.block(0, 1, 1, 7).transpose();
    
    MatrixXd P_pred = prediction(model, initial, T_pred, traj_num);
    storeResult(P_pred, "prediction");
 
        
    return 0;
}
    
