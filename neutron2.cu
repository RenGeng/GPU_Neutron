/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define OUTPUT_FILE "/tmp/absorbed.dat"


#define NB_BLOCK 256
#define NB_THREAD 256




#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)


// Déclaration dans le mémoire RAM du GPU
__device__ int device_r;
__device__ int device_b;
__device__ int device_t;


char info[] = "\
Usage:\n\
    neutron-seq H Nb C_c C_s\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
\n\
Exemple d'execution : \n\
    neutron-seq 1.0 500000000 0.5 0.5\n\
";

__global__ void setup_kernel(curandState *state){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;

  // On initialise chaque générateur avec une graine différente
  curand_init(idx, 0, 0, &state[idx]);

  /*On initialise chaque générateur avec la même graine mais avec une séquence différente
  Les générateur donneront pas les mêmes chiffres car chaque séquence est séparé de 2^67 nombres*/
  // curand_init(666, idx, 0, &state[idx]);
}

/*
 * notre gettimeofday()
 */
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}


__global__ void neutron_gpu(curandState *state, float h, int n, float c_c, float c_s, float *result)
{

  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t;
  r = b = t = 0;

  // Tableau pour l'écriture de chaque thread
  __shared__ int R[NB_THREAD];
  __shared__ int B[NB_THREAD];
  __shared__ int T[NB_THREAD];


  float c;
  c = c_c + c_s;

  // distance parcourue par le neutron avant la collision
  float L;
  // direction du neutron (0 <= d <= PI)
  float d;
  // variable aléatoire uniforme
  float u;
  // position de la particule (0 <= x <= h)
  float x;

  int idx;
  idx = threadIdx.x + blockIdx.x * blockDim.x;
  // On copie le générateur sur le registre pour plus d'efficacité
  curandState localState = state[idx];

  /* code GPU */
  while(idx < n)
  {
    d = 0.0;
    x = 0.0;

    while(1)
    {
      u = curand_uniform(&localState);
      L = -(1 / c) * log(u);
      x = x + L * cos(d);
      if (x < 0)
      {
        r++;
        break;
      }
      else if (x >= h)
      {
        t++;
        break;
      }
      else if ((u = curand_uniform(&localState)) < c_c / c)
      {
        b++;
        result[idx] = x;
        break;
      }
      else
      {
        u = curand_uniform(&localState);
        d = u * M_PI;
      }
    }

    idx+= blockDim.x * gridDim.x;
  }


  // On stock r,b,t dans le tableau
  R[threadIdx.x] = r;
  B[threadIdx.x] = b;
  T[threadIdx.x] = t;

  // Synchronisation avant qu'un thread calcule la somme totale
  __syncthreads();

  // Reduction des tableaux
  for(unsigned int s = blockDim.x/2; s > 0; s = s/2)
  {
    if(threadIdx.x < s)
    {
      R[threadIdx.x] += R[threadIdx.x + s];
      B[threadIdx.x] += B[threadIdx.x + s];
      T[threadIdx.x] += T[threadIdx.x + s];
    }
    __syncthreads();
  }


  // Seul le thread 0 d'une bloc va additionner l'ensemble des valeurs
  if(threadIdx.x == 0)
  {
    atomicAdd(&device_r,R[0]);
    atomicAdd(&device_b,B[0]);
    atomicAdd(&device_t,T[0]);
  }
}

/*
 * main()
 */
int main(int argc, char *argv[]) {

  // La distance moyenne entre les interactions neutron/atome est 1/c. 
  // c_c et c_s sont les composantes absorbantes et diffusantes de c. 
  float c_c, c_s;
  // épaisseur de la plaque
  float h;
  // nombre d'échantillons
  int n;
  // chronometrage
  double start, finish;

  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;
  c_c = 0.5;
  c_s = 0.5;

  // recuperation des parametres
  if (argc > 1)
    h = atof(argv[1]);
  if (argc > 2)
    n = atoi(argv[2]);
  if (argc > 3)
    c_c = atof(argv[3]);
  if (argc > 4)
    c_s = atof(argv[4]);

  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);

  //Allocation mémoire du résultat côté CPU
  float *host_absorbed;
  host_absorbed = (float *) calloc(n, sizeof(float));

  int r,b,t;

  //Allocation mémoire du résultat côté GPU
  float *device_absorbed;
  cudaMalloc((void **)&device_absorbed, n*sizeof(float));
  cudaMemset(device_absorbed,0,n*sizeof(float));

  // Allocation mémoire par le CPU du tableau de générateur pseudo-aléatoire
  curandState *d_state;
  CUDA_CALL(cudaMalloc((void **)&d_state, NB_BLOCK*NB_THREAD*sizeof(curandState)));

  // debut du chronometrage
  start = my_gettimeofday();
  
  // On initialise les générateurs
  setup_kernel<<<NB_BLOCK,NB_THREAD>>>(d_state);

  neutron_gpu<<<NB_BLOCK,NB_THREAD>>>(d_state, h, n, c_c, c_s, device_absorbed);

  cudaMemcpy(host_absorbed,device_absorbed,n*sizeof(float),cudaMemcpyDeviceToHost);

  cudaMemcpyFromSymbol(&r, device_r, sizeof(int),0);  
  cudaMemcpyFromSymbol(&b, device_b, sizeof(int),0); 
  cudaMemcpyFromSymbol(&t, device_t, sizeof(int),0); 

  // fin du chronometrage
  finish = my_gettimeofday();

  printf("r=%d, b=%d, t=%d\n",r,b,t);
  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);

  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

  // // ouverture du fichier pour ecrire les positions des neutrons absorbés
  // FILE *f_handle = fopen(OUTPUT_FILE, "w");
  // if (!f_handle) {
  //   fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
  //   exit(EXIT_FAILURE);
  // }

  // for (int j = 0; j < n; j++)
  //   if(host_absorbed[j]!=0.0) fprintf(f_handle, "%f\n", host_absorbed[j]);

  // // fermeture du fichier
  // fclose(f_handle);
  // printf("Result written in " OUTPUT_FILE "\n"); 

  cudaFree(d_state);
  cudaFree(device_absorbed);
  free(host_absorbed);

  return EXIT_SUCCESS;
}

