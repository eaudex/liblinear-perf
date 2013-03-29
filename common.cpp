#include <iostream>
#include <cmath>
#include <cassert>
#include <queue>
#include <vector>
#include <unordered_map>
#include <utility>
#include "linear.h"
#include "common.h"

void print_null(const char *s) {}

double dot(feature_node * xi, feature_node * xt) {
    if(xi==NULL || xt==NULL) return 0.0;

    double prod = 0.0;
    while(xi->index!=-1 && xt->index!=-1) {
        if(xi->index == xt->index) {
            prod += xi->value*xt->value;
            ++xi; ++xt;
        }
        else if(xi->index > xt->index)
            ++xt;
        else
            ++xi;
    }
    return prod;
}

double distance(feature_node * xi, feature_node * xt) {
    static std::unordered_map<feature_node*,double> hTable;

    double dot_prod = dot(xi, xt);

    double norm2_xi;
    if( hTable.find(xi)!=hTable.end() ) {
        //cout<< "got xi" <<endl;
        norm2_xi = hTable[xi];
    }
    else {
        norm2_xi = 0.0;
        feature_node * x = xi;
        while(x->index !=-1) {
            norm2_xi += x->value*x->value;
            ++x;
        }
        hTable[xi] = norm2_xi;
    }

    double norm2_xt;
    if( hTable.find(xt)!=hTable.end() ) {
        //cout<< "got xt" <<endl;
        norm2_xt = hTable[xt];
    }
    else {
        norm2_xt = 0.0;
        feature_node * x = xt;
        while(x->index !=-1) {
            norm2_xt += x->value*x->value;
            ++x;
        }
        hTable[xt] = norm2_xt;
    }

    return sqrt(norm2_xi+norm2_xt-2*dot_prod);
}


// xt : test instance
// prob_est : probability output, if not NULL
int knn_predict(problem * prob, parameter * param, feature_node * xt, double * prob_est) {
    assert(prob !=NULL && param !=NULL && xt !=NULL);
    typedef std::pair<double,int> tuple;

    int l = prob->l;
    int k = (int)param->C;
    if(k > l) {
        std::cout<< "Warning: k is too large. It must be less than or equal to #training instances " << l <<std::endl;
        std::cout<< "k <-- " << l <<std::endl;
        k = l;
    }
    else if(k < 1) {
        std::cout<< "Warning: k is too small. It must be greater than or equal to 1" <<std::endl;
        std::cout<< "k <-- 1" <<std::endl;
        k = 1;
    }

    std::priority_queue<tuple, std::vector<tuple>, std::greater<tuple> > Q;
    for(int i=0; i<l; ++i) {
        feature_node * xi = prob->x[i];
        double d = distance(xi, xt);
        Q.push( tuple(d,prob->y[i]) );
    }

    std::unordered_map<int,int> votes;
    for(int i=0; i<k; ++i) {
        tuple nn = Q.top();
        int label = nn.second;
        if( votes.find(label)!=votes.end() )
            votes[label] += 1;
        else
            votes[label] = 1;
        Q.pop();
    }

    int major_label = -1;
    int major_count = -1;
    TRVS(it, votes) {
        if((*it).second > major_count) {
            major_label = (*it).first;
            major_count = (*it).second;
        }
    }
    //std::cout<< major_label << ", " << major_count <<std::endl;

    if(prob_est !=NULL) {
        *prob_est = (double)major_count /k;
    }
    return major_label;
}
