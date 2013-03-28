#ifndef _EVAL_H
#define _EVAL_H

#include <vector>
#include "linear.h"

typedef std::vector<double> dvec_t;
typedef std::vector<int> ivec_t;

extern model * model_;
//void exit_input_error(int line_num);

/* evaluation functions for binary classification */
double accuracy(const dvec_t & dec_values, const dvec_t & ty);
double auc(const dvec_t & dec_values, const dvec_t & ty);
double bac(const dvec_t & dec_values, const dvec_t & ty);
double fscore(const dvec_t & dec_values, const dvec_t & ty);
double precision(const dvec_t & dec_values, const dvec_t & ty);
double recall(const dvec_t & dec_values, const dvec_t & ty);

/* evaluation functions for regression */
double mean_absolute_error(const dvec_t & pred_values, const dvec_t & true_values);
double mean_squared_error(const dvec_t & pred_values, const dvec_t & true_values);
double r_squared(const dvec_t & pred_values, const dvec_t & true_values);

/* cross validation function */
double binary_class_cross_validation(const problem * prob, const parameter * param, int nr_fold);
double regression_cross_validation(const problem * prob, const parameter * param, int nr_fold);

#endif
