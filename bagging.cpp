#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include <algorithm>
#include "linear.h"
#include "eval.h"
#include "common.h"

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"  for regression\n"
	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n"
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
	"		where f is the primal function and pos/neg are # of\n"
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
	"	-s 1, 3, 4, and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

//char * measure_name;

// lookup table to get the default eps for a solver, using `enum solver_type'
static const double default_eps[] = { 0.01, 0.1, 0.01, 0.1, 0.1, 0.01, 0.01, 0.1, INF,INF,INF, 0.001, 0.1, 0.1 };
// lookup table to get the solver's name, using `enum solver_type'
static const char * solver_names[] = { "L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS", "L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL", "","","", "L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL" };

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int nr_fold;
double bias;

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

    //bagging: bootstrap aggregating
    int base_solvers[] = {L2R_LR, L2R_L2LOSS_SVC, L1R_LR, L1R_L2LOSS_SVC};
    int num_base_solvers = (int)sizeof(base_solvers)/sizeof(base_solvers[0]);

    int start_logC = -3;
    int end_logC = 10;
    int num_Cs = end_logC - start_logC +1;
    double * Cs = new double[num_Cs];
    for(int i=start_logC; i<=end_logC; ++i) {
        Cs[i-start_logC] = pow(2.0, i);
    }

    for(int i=0; i<num_base_solvers; ++i) {

        std::cout<< "Current base solver " << solver_names[base_solvers[i]] <<std::endl;

        std::cout<< "Bootstrap sample as sub-problem" <<std::endl;
        int l = (int)prob.l;        //total # training instances
        int subl = (int)(0.6*l);    //# training instances in the subset
        int indices[subl];
        for(int i=0; i<subl; ++i) {
            indices[i] = rand()%l;
        }
        /*
        int indices[l];
        for(int i=0; i<l; ++i) {
            indices[i] = i;
        }
        for(int i=0; i<subl; ++i) {
            int j = i + rand()%(l-i);
            std::swap(indices[i],indices[j]);
        }
        */

        problem subprob;
        subprob.n = prob.n;
        subprob.bias = prob.bias;
        subprob.l = subl;
        subprob.x = new feature_node*[subl];
        subprob.y = new double[subl];
        for(int i=0; i<subl; ++i) {
            subprob.x[i] = prob.x[indices[i]];
            subprob.y[i] = prob.y[indices[i]];
        }
        //
        parameter subparam(param); //copy
        subparam.solver_type = base_solvers[i];         //set the current base solver
        subparam.eps = default_eps[param.solver_type];  //set the default eps for the current base solver

        std::cout<< "Grid search" <<std::endl;

        double bestC = -1.0;
        double bestCV = -1.0;
        for(int i=0; i<num_Cs; ++i) {
            std::cout<< "C " <<  Cs[i] << " ";
            subparam.C = Cs[i];
            double cv =  binary_class_cross_validation(&subprob, &subparam, nr_fold);
            if(cv > bestCV) {
                bestC = subparam.C;
                bestCV = cv;
            }
            //printf("Cross validation %g at C %g (bestCV %g, bestC %g)\n", cv,subparam.C,bestCV,bestC);
        }
        std::cout<< "Best cross validation " << bestCV << " at C " << bestC <<std::endl;
        subparam.C = bestC; //update C to the best


        std::cout<< "Train sub-model with bestC" <<std::endl;

        char submodel_file_name[1024];
        sprintf(submodel_file_name, "%s.%s", model_file_name,solver_names[base_solvers[i]]);
        //XXX start training
		model_ = train(&subprob, &subparam); 
        //XXX finish training
        std::cout<< "Save sub-model as file " << submodel_file_name <<std::endl<<std::endl;
		if(save_model(submodel_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}

//        double dec_value;
//        feature_node * xt;
//        vectorize(line, xt);
//        int pred_label = predict_values(model_, xt, &dec_value);

		free_and_destroy_model(&model_);
	    destroy_param(&subparam);
        delete [] subprob.x;
        delete [] subprob.y;
    }

//	if(flag_cross_validation)
//	{
//        if(param.solver_type == L2R_L2LOSS_SVR ||
//           param.solver_type == L2R_L1LOSS_SVR_DUAL ||
//           param.solver_type == L2R_L2LOSS_SVR_DUAL)
//        {
//            double cv = regression_cross_validation(&prob, &param, nr_fold);
//            printf("Cross validation = #%g#\n", cv);
//        }
//        else {
//            double cv =  binary_class_cross_validation(&prob, &param, nr_fold);
//            printf("Cross validation = #%g#\n", cv);
//        }
//	}
//	else
//	{
//		model_=train(&prob, &param);
//		if(save_model(model_file_name, model_))
//		{
//			fprintf(stderr,"can't save model to file %s\n",model_file_name);
//			exit(1);
//		}
//		free_and_destroy_model(&model_);
//	}

	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

    delete [] Cs;

	return 0;
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	//void (*print_func)(const char*) = NULL;	// default printing to stdout
	void (*print_func)(const char*) = print_null;	// default printing to null

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	flag_cross_validation = 1;
    nr_fold = 5;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}
