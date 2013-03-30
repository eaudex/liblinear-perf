#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <errno.h>
#include "linear.h"
#include "eval.h"


int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

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

void exit_with_help()
{
	printf(
	"Usage: predict [options] test_file model_file output_file\n"
	"options:\n"
    "-o output_option: 0* for label, 1 for decision value, 2 for probability estimate\n"
	"-q : quiet mode (no outputs) (verbose mode by default)\n"
    " `*' indicates the default options\n"
	);
	exit(1);
}


struct feature_node * x;
int max_nr_attr = 64;

struct model * model_;
int flag_predict_probability = 0;
int output_option = 0;

void do_predict(FILE *input, FILE *output);


int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
//			case 'b':
//				flag_predict_probability = atoi(argv[i]);
//				break;
			case 'o':
				output_option = atoi(argv[i]);
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	if(i>=argc)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model_=load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
	do_predict(input, output);
	free_and_destroy_model(&model_);
	free(line);
	free(x);
	fclose(input);
	fclose(output);
	return 0;
}


void do_predict(FILE *input, FILE *output)
{
    std::vector<double> pred_values; //store decision values
    std::vector<double> true_values; //store true values

	int total = 0;
	int nr_class = get_nr_class(model_);
	int * labels = Malloc(int, nr_class);
    get_labels(model_, labels);
	double * prob_estimates = NULL;
	int j, n;
	int nr_feature = get_nr_feature(model_);
	if(model_->bias >=0)
		n = nr_feature+1;
	else
		n = nr_feature;

    // not yet support multiclass
    assert(nr_class==2);

    //print out header...
    if(output_option ==2) {
		prob_estimates = Malloc(double, nr_class);
		fprintf(output,"labels");
		for(j=0;j<nr_class;j++)
			fprintf(output," %d",labels[j]);
		fprintf(output,"\n");
    }

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
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
			if(x[i].index <= nr_feature)
				++i;
		}

		if(model_->bias>=0)
		{
			x[i].index = n;
			x[i].value = model_->bias;
			i++;
		}
		x[i].index = -1;

        //XXX
        double dec_value;
        predict_label = predict_values(model_, x, &dec_value);

        if(output_option==0) {
		    //predict_label = predict(model_, x);
			fprintf(output,"%g\n", predict_label);
        }
        else if(output_option==1) {
			fprintf(output,"%g\n", dec_value);
        }
        else if(output_option==2) {
//            if(target_label==model_->label[0]) {
//			    fprintf(output,"%g %g\n", predict_label,1.0/(1.0+exp(-dec_value)) );
//            }
//            else {
//			    fprintf(output,"%g %g\n", predict_label,1.0/(1.0+exp(-(1-dec_value))) );
//            }
			predict_label = predict_probability(model_, x, prob_estimates);
			fprintf(output,"%g", predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
        }
        else {
            //unexpected output option
        }

        // store for evaluating model performance
        pred_values.push_back( dec_value );
        true_values.push_back( (target_label==model_->label[0])?(+1):(-1) );
	}

	if(model_->param.solver_type==L2R_L2LOSS_SVR ||
	   model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
	   model_->param.solver_type==L2R_L2LOSS_SVR_DUAL)
	{
        validation_function_regression(pred_values, true_values);
    }
    else
    {
        validation_function(pred_values, true_values);
    }

	free(labels);

//	if(flag_predict_probability)
	if(output_option==2)
		free(prob_estimates);

}
