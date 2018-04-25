#include"map.h"
//#include<stdio.h>
//#include<stdlib.h>
#include<vector>
#include<math.h>
#include<time.h>
#include<string.h>
using namespace std;

#define numLayers 3
#define nodesInLayer_0 203
#define nodesInLayer_1 80
#define nodesInLayer_2 26
int nodesInLayer[3];


void load_nettalk_data(char* filename, vector<char*> &vword, vector<char*> &vphoneme, vector<char*> &vstress){
	printf("%s\n", filename);
	FILE* fp = fopen(filename, "r");
	char readtemp[3][128];
	int i;
	while(fscanf(fp, "%s %s %s %d", readtemp[0], readtemp[1], readtemp[2], &i)!=EOF){
		vword.push_back(readtemp[0]);
		vphoneme.push_back(readtemp[1]);
		vstress.push_back(readtemp[2]);
		//printf("%s, %s, %s\n", vword.back(), vphoneme.back(), vstress.back());
	}
	printf("allocate done\n");
}

void allocate_lists(double*** &topology, double** &value, double**** &weights, double**** &delta, double** &gradient, double* &desired){
	int i, j, k, l;
	srand(time(NULL));
	printf("allocate");
	nodesInLayer[0] = nodesInLayer_0;
	nodesInLayer[1] = nodesInLayer_1;
	nodesInLayer[2] = nodesInLayer_2;
	
	desired = (double*)malloc(sizeof(double)*nodesInLayer[2]);
	
	//topology allocate
	topology = (double***)malloc(sizeof(double**)*nodesInLayer[0]);
	for(i=0; i<nodesInLayer[0]; ++i){
		topology[i] = (double**)malloc(sizeof(double*)*nodesInLayer[1]);
		for(j=0; j<nodesInLayer[1]; ++j){
			topology[i][j] = (double*)malloc(sizeof(double)*nodesInLayer[2]);
		}
	}
	
	//gradient & value allocate
	gradient = (double**)malloc(sizeof(double*)*numLayers);
	value = (double**)malloc(sizeof(double*)*numLayers);
	for(i=0; i<numLayers; ++i){
		gradient[i] = (double*)malloc(sizeof(double)*nodesInLayer[i]);
		value[i] = (double*)malloc(sizeof(double)*nodesInLayer[i]);
		for(j=0; j<nodesInLayer[i]; ++j){
			gradient[i][j] = 0.f;
			value[i][j] = 0.f;
		}
	}
	
	//weights & delta allocate
	weights = (double****)malloc(sizeof(double***)*numLayers);
	delta = (double****)malloc(sizeof(double***)*numLayers);
	for(i=0; i<numLayers; ++i){
		weights[i] = (double***)malloc(sizeof(double**)*nodesInLayer[i]);
		delta[i] = (double***)malloc(sizeof(double**)*nodesInLayer[i]);
		for(j=0; j<nodesInLayer[i]; ++j){
			weights[i][j] = (double**)malloc(sizeof(double*)*(i+1));
			delta[i][j] = (double**)malloc(sizeof(double*)*(i+1));
			for(k=0; k<=i; ++k){
				weights[i][j][k] = (double*)malloc(sizeof(double)*nodesInLayer[k]);
				delta[i][j][k] = (double*)malloc(sizeof(double)*nodesInLayer[k]);
				for(l=0; l < nodesInLayer[k]; ++l){
					double num = 1.f*rand()/RAND_MAX;
					double sign = 1.f*rand()/RAND_MAX;
					if(sign<0.5f) num *= -1;
					
					weights[i][j][k][l] = num;
					delta[i][j][k][l] = 0.f;
				}
			}
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
			
				}
			}
		}
	}
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
			//printf("v: %f\n", value[i][j]);
		}
	}
	//printf("\n");
} 
FILE *fp = fopen("weight_7000_v2.out","w");
void dump_weight(double**** weights){
	int i,j,k,l;
	for(i=1; i<numLayers; ++i){
		for(j=0; j<nodesInLayer[i]; ++j){
			for(k=0; k<i; ++k){
				for(l=0; l<nodesInLayer[k]; ++l){
					fprintf(fp,"%f ", weights[i][j][k][l]);
				}
				fprintf(fp,"\n");
			}
		}
	}
}




int main(int argc, char** argv){
	FILE* wf = fopen("result0.out","w");
	int passes = atoi ( argv[2]);
	int numWords = atoi (argv[3]);
	printf("passes:%d\n",passes);
	double max[3]={0.0,0.0,0.0};
	double eta = .1f;
	double alpha = 0.9f;
	//int passes = 7000;
	//int numWords = 50;
	int offset = 0;
	int groupSize = 29;
	int margin = 3;
	int frame = 2*margin + 1;
	int input = frame * groupSize;
	double*** topology;
	double** values;
	double**** weights;
	double**** delta;
	double** gradient;
	double* desired;
	//int numWords = 1024;
	int charIndex[input/groupSize];
	int i, j;
	for(i=0, j=0; i<input; i+=groupSize, ++j){
		charIndex[j] = i;
	}
	vector<char*> word;
	vector<char*> phoneme;
	vector<char*> stress;
	
	printf("fsdfsfsdf");
	allocate_lists(topology,values,weights,delta,gradient, desired);
	
	//load_nettalk_data(argv[1], word, phoneme, stress);
	
	printf("%s\n", argv[1]);
	FILE* fp = fopen(argv[1], "r");
	char** readtemp;
	do{
		readtemp = (char**)malloc(sizeof(char*)*3);
		for(int i=0; i<3; ++i){
			readtemp[i] = (char*)malloc(sizeof(char)*128);
		}
		if(fscanf(fp, "%s %s %s %d", readtemp[0], readtemp[1], readtemp[2], &i)!=EOF){
			word.push_back(readtemp[0]);
			phoneme.push_back(readtemp[1]);
			stress.push_back(readtemp[2]);
			//numWords++;
		}else{
			break;
		}
	}while(1);
	
		//printf("%s, %s, %s\n", vword.back(), vphoneme.back(), vstress.back());
	
	printf("load done %d\n",numWords);
	
	//for(int i=0; i<word.size(); ++i){
		//printf("%s %s %s\n", word.at(i), phoneme.at(i), stress.at(i));
//	}
	int numCorrect = 0;
	int total = 0;
	double avgError = 0;

	
	int round;
	
	int* converted;
	
	for(round=0; round<passes; ++round){
		srand(711);
		numCorrect = 0;
		total = 0;
		avgError = 0.f;
		int widx;
		int t;
		//for(widx = 0; widx <numWords; ++widx){
		for(t = 0; t<numWords; ++t){
			widx = rand()%20007;
			//printf("widx: %d %s %s %s\n",widx,word.at(widx),phoneme.at(widx),stress.at(widx));
			int strlength = strlen(word.at(widx));
			int pos;
			for(pos=0; pos< strlength; ++pos){
				int start = pos - margin;
				int end = pos + margin + 1;
				int index = 0;
				int i, j;
				char c;
				for(j=start; j<end; ++j){
					if(j<0||j>strlength){
						c = '-';
					}else{
						c = word.at(widx)[j];
					}
					converted = map_char(c); //malloc
					
					//printf("[");
					//for(i=0; i<29; ++i){
					//	printf("%d ", converted[i]);
					//}printf("]\n");				
					
					int cIndex = 0;
					int k;
					for(k=charIndex[index]; k<charIndex[index]+groupSize; ++k){
							
							values[0][k] = converted[cIndex];
							cIndex++;
							
					}
					index += 1;
				}

			//	printf("%c, %c\n", phoneme.at(widx)[pos], stress.at(widx)[pos]);
				int* pVector = map_phoneme(phoneme.at(widx)[pos]);
				int* sVector = map_stress(stress.at(widx)[pos]);
				

				for(i=0; i<21; ++i)		desired[i] = pVector[i];

				for(j=0; j<5; ++j, ++i) desired[i] = sVector[j];

				//dump_weight(weights);
			//	printf("===========================================================>\n\n\n");
				update(weights, values);
			
				double err = error(values[2], desired);
				//printf("err: %f\n", err);
				avgError += err;
				if(err > .1){
					backpropagate(values,desired,weights,gradient);
					compute_delta(delta,gradient,values,alpha);
					update_weight(delta,weights,eta);
					//	dump_weight(weights);
				}else{
					numCorrect += 1;
				}
				total+=1;
			}
		}
		avgError = avgError / total;
		printf("Correctly classified %d of %d phonemes after %d passes, average error: %f\n", numCorrect, total, round, avgError);
		double temp = numCorrect; 
		fprintf(wf,"%lf\n",temp/total);
		printf("%lf\n",temp/total);
		if(avgError<max[2]){
			max[0] = numCorrect;
			max[1] = total;
			max[2] = avgError;
		}
		numCorrect = 0;
	}
	
	dump_weight(weights);
	printf("best correct %f,passes %f average error %f \n",max[0],max[1],max[2] );
	fprintf(fp,"best correct %lf,passes %lf average error %lf \n",max[0],max[1],max[2] );
	
	return 0;
}