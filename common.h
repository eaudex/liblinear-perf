#ifndef _COMMON_H
#define _COMMON_H

#include <cstdlib>
#include <cmath>
#include <vector>
#include "linear.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#define TRVS(it,C) for(typeof(C.begin())it=C.begin(); it!=C.end(); ++it)

typedef std::vector<double> dvec_t;
typedef std::vector<int> ivec_t;

extern void print_null(const char *s);

//
double dot(feature_node * xi, feature_node * xt);
double distance(feature_node * xi, feature_node * xt);
int knn_predict(problem * prob, parameter * param, feature_node * xt, double * prob_est=NULL);

#endif /* _COMMON_H */
