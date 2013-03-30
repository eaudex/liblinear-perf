#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <iostream>
#include "linear.h"
#include "common.h"

using namespace std;

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"	 0 -- kNN\n"
	"-c k: set the parameter C (default 1)\n"
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

struct feature_node *x_space;
struct parameter param;
struct problem prob;
int flag_cross_validation;
int nr_fold;
double bias;
long nnz;


int main(int argc, char *argv[])
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

    // select the best K
    int startK = 1;
    int endK = 4*prob.l/nr_fold;

    double bestC = -INF;
    double bestCV = -INF;
    dvec_t pred_values(prob.l,0.0);
    dvec_t prob_values(prob.l,0.0);
    dvec_t true_values(prob.l,0.0);

    for(int K=startK; K<endK; K+=10) {
        param.C = K;
        cout<< "K " <<  K << ", ";

        //XXX cross validation
        int i;
        int * fold_start = new int[nr_fold+1];
        int l = prob.l;
        int * perm = new int[l];

        for(i=0;i<l;i++) perm[i]=i;
        for(i=0;i<l;i++)
        {
            int j = i+rand()%(l-i);
            swap(perm[i],perm[j]);
        }
        for(i=0;i<=nr_fold;i++)
            fold_start[i]=i*l/nr_fold;

        for(i=0;i<nr_fold;i++)
        {
            int                begin   = fold_start[i];
            int                end     = fold_start[i+1];
            int                j,k;
            problem subprob;

            subprob.n = prob.n;
            subprob.bias = prob.bias;
            subprob.l = l-(end-begin);
            subprob.x = new feature_node*[subprob.l];
            subprob.y = new double[subprob.l];

            k=0;
            for(j=0;j<begin;j++)
            {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            for(j=end;j<l;j++)
            {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }

            for(j=begin; j<end; ++j) {
                pred_values[j] = (double)knn_predict(&subprob, &param, prob.x[perm[j]], &prob_values[j]);
                true_values[j] = prob.y[perm[j]];
            }
        
            delete [] subprob.x;
            delete [] subprob.y;
        }
        delete [] perm;
        delete [] fold_start;

        //XXX
        int correct = 0;
        int total = prob.l;
        double logloss = 0.0;

        for(int i=0; i<total; ++i) {
            if(true_values[i] > 0) {
                if(pred_values[i] > 0) {
                    ++correct;
                    logloss += log(prob_values[i]+1e-6);
                }
                else
                    logloss += log(1-prob_values[i]+1e-6);
            }
            else {
                if(pred_values[i] <= 0) {
                    correct++;
                    logloss += log(prob_values[i]+1e-6);
                }
                else
                    logloss += log(1-prob_values[i]+1e-6);
            }
        }
        logloss /= total;

        cout<< "Cross validation logloss " << logloss << ", acc " << (double)correct/total <<endl;

        if(logloss > bestCV) {
            bestC = K;
            bestCV = logloss;
        }
    }
    std::cout<< "Best cross validation " << bestCV << " at C " << bestC <<std::endl;
    param.C = bestC;


	FILE * test_input = fopen(model_file_name, "r");
	if(test_input == NULL)
	{
		fprintf(stderr,"can't open test input file %s\n", model_file_name);
		exit(1);
	}

    //XXX predict
    int total = 0;
    int correct = 0;
    int max_nr_attr = (int)nnz/prob.l;
    feature_node * x = Malloc(feature_node, max_nr_attr);
	while(readline(test_input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			// feature indices larger than those in training are not used
			if(x[i].index <= prob.n)
				++i;
		}
		x[i].index = -1;

        double prob_est;
        predict_label = (double)knn_predict(&prob, &param, x, &prob_est);
        cout<< predict_label <<endl;

        ++total;
		if(predict_label == target_label)
			++correct;
	}
    cout<< (double)correct/total << " (" << correct << "/" << total << ")" <<endl;
    fclose(test_input);

	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
	free(x);

	return EXIT_SUCCESS;
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

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

    //XXX
    nnz = elements+prob.l;

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
