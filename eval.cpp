#include <cstdio>
#include <cassert>
#include <cmath>
#include <algorithm>
#include "linear.h"
#include "eval.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//static enum measure {
//    ACC=0, AUC, BAC, F_SCORE, PRECISION, RECALL, M_SQ_ERR=11, M_ABS_ERR, R_SQ
//};
static const char * measure_name[] = {
        "accuracy", "AUC", "BAC", "F score", "precision", "recall", "", "", "", "", "",
        "mean squared error", "mean absolute error", "squared correlation coefficient"};

// XXX
// evaluation function pointer
// You can assign this pointer to any above prototype
double (*validation_function)(const dvec_t&, const dvec_t&) = logloss;
double (*validation_function_regression)(const dvec_t&, const dvec_t&) = mean_squared_error;


// ty[i] is either in {1,-1} or {1,0}.
// dec_values[i] is the decision value, w^T xi + bias
double logloss(const dvec_t & dec_values, const dvec_t & ty) {
    assert(dec_values.size() == ty.size());

    double logloss = 0.0;
	size_t total = ty.size();

	for(size_t i=0; i<total; ++i) {
        double prob_est = 1.0/(1.0+exp(-dec_values[i]));
        if(ty[i] > 0)
            logloss += log(prob_est+1e-6);      //add small value to prevent -inf
        else //ty[i]<=0
            logloss += log(1.0-prob_est+1e-6);  //add small value to prevent -inf
    }
    logloss /= total;

	printf("Logloss %g\n", logloss);
	return logloss;
}


// ty[i] is either in {1,-1} or {1,0}.
// dec_values[i] is the decision value, w^T xi + bias
double accuracy(const dvec_t & dec_values, const dvec_t & ty) {
    double acc = 0.0;
	size_t correct = 0;
	size_t total = ty.size();

	for(size_t i=0; i<total; ++i) {
        if(dec_values[i]>0 && ty[i]>0)
            ++correct;
        else if(dec_values[i]<=0 && ty[i]<=0)
            ++correct;
    }
    acc = (double)correct / total;

	printf("Accuracy %g (%d/%d)\n", acc, correct,total);
	return acc;
}

double precision(const dvec_t& dec_values, const dvec_t& ty){
	size_t size = dec_values.size();
	size_t i;
	int    tp, fp;
	double precision;

	tp = fp = 0;

	for(i = 0; i < size; ++i) if(dec_values[i] >= 0){
		if(ty[i] == 1) ++tp;
		else           ++fp;
	}

	if(tp + fp == 0){
		fprintf(stderr, "warning: No postive predict label.\n");
		precision = 0;
	}else
		precision = tp / (double) (tp + fp);
	printf("Precision %g%% (%d/%d)\n", 100.0 * precision, tp, tp + fp);
	
	return precision;
}


double recall(const dvec_t& dec_values, const dvec_t& ty){
	size_t size = dec_values.size();
	size_t i;
	int    tp, fn; // true_positive and false_negative
	double recall;

	tp = fn = 0;

	for(i = 0; i < size; ++i) if(ty[i] == 1){ // true label is 1
		if(dec_values[i] >= 0) ++tp; // predict label is 1
		else                   ++fn; // predict label is -1
	}

	if(tp + fn == 0){
		fprintf(stderr, "warning: No postive true label.\n");
		recall = 0;
	}else
		recall = tp / (double) (tp + fn);
	// print result in case of invocation in prediction
	printf("Recall %g%% (%d/%d)\n", 100.0 * recall, tp, tp + fn);
	
	return recall; // return the evaluation value
}


double fscore(const dvec_t& dec_values, const dvec_t& ty){
	size_t size = dec_values.size();
	size_t i;
	int    tp, fp, fn;
	double precision, recall;
	double fscore;

	tp = fp = fn = 0;

	for(i = 0; i < size; ++i) 
		if(dec_values[i] >= 0 && ty[i] == 1) ++tp;
		else if(dec_values[i] >= 0 && ty[i] == -1) ++fp;
		else if(dec_values[i] <  0 && ty[i] == 1) ++fn;

	if(tp + fp == 0){
		fprintf(stderr, "warning: No postive predict label.\n");
		precision = 0;
	}else
		precision = tp / (double) (tp + fp);
	if(tp + fn == 0){
		fprintf(stderr, "warning: No postive true label.\n");
		recall = 0;
	}else
		recall = tp / (double) (tp + fn);

	
	if(precision + recall == 0){
		fprintf(stderr, "warning: precision + recall = 0.\n");
		fscore = 0;
	}else
		fscore = 2 * precision * recall / (precision + recall);

	printf("F-score %g\n", fscore);
	
	return fscore;
}


double bac(const dvec_t& dec_values, const dvec_t& ty){
	size_t size = dec_values.size();
	size_t i;
	int    tp, fp, fn, tn;
	double specificity, recall;
	double bac;

	tp = fp = fn = tn = 0;

	for(i = 0; i < size; ++i) 
		if(dec_values[i] >= 0 && ty[i] == 1) ++tp;
		else if(dec_values[i] >= 0 && ty[i] == -1) ++fp;
		else if(dec_values[i] <  0 && ty[i] == 1)  ++fn;
		else ++tn;

	if(tn + fp == 0){
		fprintf(stderr, "warning: No negative true label.\n");
		specificity = 0;
	}else
		specificity = tn / (double)(tn + fp);
	if(tp + fn == 0){
		fprintf(stderr, "warning: No positive true label.\n");
		recall = 0;
	}else
		recall = tp / (double)(tp + fn);

	bac = (specificity + recall) / 2;
	printf("BAC %g\n", bac);
	
	return bac;
}



// only for auc
class Comp{
	const double *dec_val;
	public:
	Comp(const double *ptr): dec_val(ptr){}
	bool operator()(int i, int j) const{
		return dec_val[i] > dec_val[j];
	}
};

double auc(const dvec_t& dec_values, const dvec_t& ty){
	double roc  = 0;
//	size_t size = dec_values.size();
//	size_t i;
//	std::vector<size_t> indices(size);

	int i;
	int size = (int)dec_values.size();
	ivec_t indices(size);

	for(i = 0; i < size; ++i) indices[i] = i;

	std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));

	int tp = 0,fp = 0;
	for(i = 0; i < size; i++) {
		if(ty[indices[i]] == 1) tp++;
		else if(ty[indices[i]] == -1) {
			roc += tp;
			fp++;
		}
	}

	if(tp == 0 || fp == 0)
	{
		fprintf(stderr, "warning: Too few postive true labels or negative true labels\n");
		roc = 0;
	}
	else
		roc = roc / tp / fp;

	printf("AUC %g\n", roc);

	return roc;
}





double binary_class_cross_validation(const problem *prob, const parameter *param, int nr_fold)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);
	int * labels;
	dvec_t dec_values;
	dvec_t ty;

	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		std::swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int                begin   = fold_start[i];
		int                end     = fold_start[i+1];
		int                j,k;
		struct problem subprob;

		subprob.n = prob->n;
		subprob.bias = prob->bias;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);

		labels = Malloc(int, get_nr_class(submodel));
		get_labels(submodel, labels);

		if(get_nr_class(submodel) > 2) 
		{
			fprintf(stderr,"Error: the number of class is not equal to 2\n");
			exit(-1);
		}

		dec_values.resize(end);
		ty.resize(end);

		for(j=begin; j<end; ++j) {
			predict_values(submodel, prob->x[perm[j]], &dec_values[j]);
			ty[j] = ((int)prob->y[perm[j]]==labels[0])?(+1):(-1);
		}

//		if(labels[0] <= 0) {
//			for(j=begin;j<end;j++)
//				dec_values[j] *= -1;
//		}
	
		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
		free(labels);
	}		

	free(perm);
	free(fold_start);

	return validation_function(dec_values, ty);	
}



//XXX
double r_squared(const dvec_t & pred_values, const dvec_t & true_values) {
    assert(pred_values.size() == true_values.size());

    size_t l = pred_values.size();
    double sumv=0.0, sumy=0.0;
    double sumvv=0.0, sumyy=0.0, sumvy=0.0;

    for(size_t i=0; i<l; ++i) {
        double y = pred_values[i];
        double v = true_values[i];
        sumv += v;
        sumy += y;
        sumvv += v*v;
        sumyy += y*y;
        sumvy += v*y;
    }

    double ll = (double)l;
    double rsq = ((ll*sumvy-sumv*sumy)*(ll*sumvy-sumv*sumy))
                    / ((ll*sumvv-sumv*sumv)*(ll*sumyy-sumy*sumy));
	printf("Squared correlation coefficient %g\n", rsq);

    return rsq;
}

double mean_absolute_error(const dvec_t & pred_values, const dvec_t & true_values) {
    assert(pred_values.size() == true_values.size());

    size_t l = pred_values.size();
    double total_error = 0.0;

    for(size_t i=0; i<l; ++i) {
        double diff = pred_values[i]-true_values[i];
        total_error += fabs(diff);
    }

	printf("Mean absolute error %g\n", total_error/l);

    return total_error/l;
}

double mean_squared_error(const dvec_t & pred_values, const dvec_t & true_values) {
    assert(pred_values.size() == true_values.size());

    size_t l = pred_values.size();
    double total_error = 0.0;

    for(size_t i=0; i<l; ++i) {
        double diff = pred_values[i]-true_values[i];
        total_error += diff*diff;
    }

	printf("Mean squared error %g\n", total_error/l);

    return total_error/l;
}


double regression_cross_validation(const problem * prob, const parameter * param, int nr_fold)
{
	int i;
	int * fold_start = Malloc(int, nr_fold+1);
	int l = prob->l;
	int * perm = Malloc(int,l);
	dvec_t pred_values;  //predicted
	dvec_t true_values; //actual

	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i + rand()%(l-i);
		std::swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i] = i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int                begin   = fold_start[i];
		int                end     = fold_start[i+1];
		int                j,k;
		struct problem subprob;

		subprob.n = prob->n;
		subprob.bias = prob->bias;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);

		pred_values.resize(end);
		true_values.resize(end);

		for(j=begin;j<end;j++) {
			predict_values(submodel, prob->x[perm[j]], &pred_values[j]);
			true_values[j] = prob->y[perm[j]];
		}
	
		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}

	free(perm);
	free(fold_start);

	return validation_function_regression(pred_values, true_values);
}

