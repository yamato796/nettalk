#include"map.h"
#include<math.h>
#include<string.h>

#define numLayers 3
#define nodesInLayer_0 203
#define nodesInLayer_1 80
#define nodesInLayer_2 26
int nodesInLayer[3];

void init_allocate(double**** &w, double** &value, double**** &delta ,FILE* fp){
	int i, j, k, l;
w = (double****)malloc(sizeof(double***)*numLayers);
	delta = (double****)malloc(sizeof(double***)*numLayers);
	for(i=0; i<numLayers; ++i){
		w[i] = (double***)malloc(sizeof(double**)*nodesInLayer[i]);
		delta[i] = (double***)malloc(sizeof(double**)*nodesInLayer[i]);
		for(j=0; j<nodesInLayer[i]; ++j){
			w[i][j] = (double**)malloc(sizeof(double*)*(i+1));
			delta[i][j] = (double**)malloc(sizeof(double*)*(i+1));
			for(k=0; k<=i; ++k){
				w[i][j][k] = (double*)malloc(sizeof(double)*nodesInLayer[k]);
				delta[i][j][k] = (double*)malloc(sizeof(double)*nodesInLayer[k]);
				for(l=0; l < nodesInLayer[k]; ++l){
				
					w[i][j][k][l] = 0.f;
					delta[i][j][k][l] = 0.f;
				}
			}
		}
	}
	
	
	for(i=1; i<numLayers; ++i){
		for(j=0; j<nodesInLayer[i]; ++j){
			for(k=0; k<i; ++k){
				for(l=0; l < nodesInLayer[k]; ++l){
					fscanf(fp, "%lf ", &w[i][j][k][l]);
				}
			}
		}
	}
	
	
	//printf("init weight done\n");
	value = (double**)malloc(sizeof(double*)*numLayers);
	for(i=0; i<numLayers; ++i){
		value[i] = (double*)malloc(sizeof(double)*nodesInLayer[i]);
		for(j=0; j<nodesInLayer[i]; ++j){
			value[i][j] = 0.f;
		}
	}
}

double sigmoid(double value){
	return 1.f/(1.f + exp(-value));
}

double dsigmoid(double value){
	double x  = 1.f*exp(-value);
	return 1.f*x / ((x+1.f)*(x+1.f));
}

void update(double**** &w, double** &value){
	int i, j, k, l;
	for(i=1; i<numLayers; ++i){
		for(j=0; j<nodesInLayer[i]; ++j){
			double sum = 0.f;
			for(k=0; k<i; ++k){
				for(l=0; l < nodesInLayer[k]; ++l){
					//printf("update 3: %f, %f %f\n", value[k][l], w[i][j][k][l], sigmoid(value[k][l]));
					sum += sigmoid(value[k][l])*w[i][j][k][l];
				}
			}
			value[i][j] = sum;
			//if (i==1) printf("v: %f\n", value[i][j]);
		}
	}
	//printf("\n");
}

void dump_value(double** &value){
	int i, j;
	double sum= 0.f;
	int count = 0;
	//printf("value: \n");
	for(i=1; i<2; ++i){
		//printf("layer: %d, nodeNum: %d\n", i, nodesInLayer[i]);
		for(j=0; j<nodesInLayer[i]; ++j){
			if(j==40) printf("\n");
			printf("%f ", sigmoid(value[i][j]));
			if(sigmoid(value[i][j]> 0.5)) count++;
			sum += sigmoid(value[i][j]);
		}
		printf("\n\n");
	}
	printf("%f\n%d\n", sum, count);
}

void backpropagate(double** &value, double* &desired, double**** &w, double** &gradient){
	int i, j;
	//printf("gradient2\n");
	for(i=0; i<nodesInLayer[2]; ++i){
		gradient[2][i] = (desired[i] - sigmoid(value[2][i]))*dsigmoid(value[2][i]);
	//	printf("%f ", gradient[2][i]);
	}//printf("\n");
	//printf("gradient10\n");
	for(i=0; i<nodesInLayer[1]; ++i){
		double sum= 0.f;
		for(j=0; j<nodesInLayer[2]; ++j){
		
			sum += gradient[2][j]*w[2][j][1][i]*dsigmoid(value[1][i]);
				//printf("%f,%f,%f,%f ",gradient[2][j], w[2][j][1][i], value[1][i], sum);
		}

		gradient[1][i] = sum;
		
	}
	//printf("\n\n\n");
}

double error(double* output, double* desired){
	double err = 0.f;
	int i;
	//printf("%d\n", nodesInLayer[2]);
	for(i=0; i<nodesInLayer[2]; ++i){
		double x = 1.f*desired[i] - sigmoid(output[i]);
		//printf("ee\n");
		//printf("%f, %f, %f\n", desired[i], sigmoid(output[i]), x);
		err += x*x;
	//	printf("%f ", err);
	}
	return err;
}

void compute_delta(double**** &delta, double** &gradient, double** &value, double alpha){
	//	printf("\n\ndelta:\n");
	int i, j, k, l;
	for(i=1; i<numLayers; ++i){
		for(j=0; j<nodesInLayer[i]; ++j){
			for(k=0; k<i; ++k){
				for(l=0; l<nodesInLayer[k]; ++l){
					delta[i][j][k][l] = delta[i][j][k][l]*alpha + ((1-alpha)*gradient[i][j]*sigmoid(value[k][l]));
	//				printf("%f ", delta[i][j][k][l]);
				}
			}
		}
	}
	//printf("\n\n\n");
}

void update_weight(double**** &delta, double**** &weights, double eta){
	int i,j,k,l;
	for(i=1; i<numLayers; ++i){
		for(j=0; j<nodesInLayer[i]; ++j){
			for(k=0; k<i; ++k){
				for(l=0; l<nodesInLayer[k]; ++l){
					weights[i][j][k][l] += delta[i][j][k][l]*eta;
					//delta[i][j][k][l] = delta[i][j][k][l]*a + ((1 - a) * gradient[i][j] * output[i][k])
				}
			}
		}
	}
}





int main(int argc, char** argv){
	nodesInLayer[0] = nodesInLayer_0;
	nodesInLayer[1] = nodesInLayer_1;
	nodesInLayer[2] = nodesInLayer_2;
	
	
	FILE* fp1 = fopen(argv[1], "r");
	
	
	double eta = .1f;
	double alpha = 0.9f;
	int offset = 0;
	int groupSize = 29;
	int margin = 3;
	int frame = 2*margin + 1;
	int input = frame * groupSize;
	double**** weight;
	double** value;
	double* desired;
	double**** delta;
	int* converted;
	
	init_allocate(weight, value, delta, fp1);
	desired = (double*)malloc(sizeof(double)*nodesInLayer[2]);
	
	int charIndex[input/groupSize];
	int i, j;
	for(i=0, j=0; i<input; i+=groupSize, ++j){
		charIndex[j] = i;
	}
	
	
	char wordtest[128], phonemetest[128], stresstest[128];
	int pos;
	
	printf("input (pos word phoneme stress)\n");
	scanf("%d %s %s %s", &pos, wordtest, phonemetest, stresstest);
	//printf("%d %s\n", pos, wordtest);
	int strlength = strlen(wordtest);
	//	}
	int numCorrect = 0;
	int total = 0;
	double avgError = 0;
	
	
		int start = pos - margin;
		int end = pos + margin + 1;
		int index = 0;

		//printf("%d, %d\n", start, end);
		char c;
		for(j=start; j<end; ++j){
			if(j<0||j>=strlength){
				c = '-';
			}else{
				c = wordtest[j];
			}

			converted = map_char(c); //malloc
				

				
			int cIndex = 0;
			int k;
			//printf("v0 :\n");
			for(k=charIndex[index]; k<charIndex[index]+groupSize; ++k){
						
					value[0][k] = 1.f*converted[cIndex];
			//		printf("%lf", value[0][k]);
					cIndex++;
						
			}//printf("\n num: %d\n", k);
			index += 1;
		}
		
		update(weight, value);

		
		
		int* pVector = map_phoneme(phonemetest[pos]);

		int* sVector = map_stress(stresstest[pos]);
		


		for(i=0; i<21; ++i)	{
			
			desired[i] = pVector[i];
			//printf("%f", desired[i]);
		}
		for(j=0; j<5; ++j, ++i){
			desired[i] = sVector[j];
			//printf("%f", sVector[j]);
		}

					//dump_weight(weights);
		//printf("===========================================================>\n\n\n");
				
		double err = error(value[2], desired);
		
		dump_value(value);

	printf("%f %s", err, wordtest);
	
	return 0;
}