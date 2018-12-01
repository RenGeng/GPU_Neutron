/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version CPU+GPU
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <omp.h>

#define OUTPUT_FILE "/tmp/absorbed.dat"


#define NB_BLOCK 256
#define NB_THREAD 256
#define CHARGE_GPU 0.75



#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)


// Déclaration dans le mémoire RAM
__device__ int device_r;
__device__ int device_b;
__device__ int device_t;
__device__ int device_j=0;


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

struct drand48_data alea_buffer;

void init_uniform_random_number() {
  srand48_r(0 + omp_get_thread_num(), &alea_buffer);
}

float uniform_random_number() {
  double res = 0.0; 
  drand48_r(&alea_buffer,&res);
  return res;
}


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
  float c, c_c, c_s;
  // épaisseur de la plaque
  float h;
  // distance parcourue par le neutron avant la collision
  float L;
  // direction du neutron (0 <= d <= PI)
  float d;
  // variable aléatoire uniforme
  float u;
  // position de la particule (0 <= x <= h)
  float x;
  // nombre d'échantillons
  int n;
  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t;
  // nombre de neutrons refléchis, absorbés et transmis
  int rh, bh, th;
  // chronometrage
  double start, finish;

  int i,j=0; // compteurs 

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

  r = b = t = 0;
  c = c_c + c_s;


  // Le GPU récupère la plus part du travail
  int taille_gpu = ceil(CHARGE_GPU * n);
  // Le reste est pour le CPU
  int taille_cpu = n - taille_gpu;

  printf("taill gpu: %d et taille_cpu : %d",taille_gpu,taille_cpu);

  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);

  //Allocation mémoire du résultat côté CPU
  float *host_absorbed;
  host_absorbed = (float *) calloc(n, sizeof(float));

    //Allocation mémoire du résultat côté GPU
  float *device_absorbed;
  cudaMalloc((void **)&device_absorbed, taille_gpu*sizeof(float));
  cudaMemset(device_absorbed,0,taille_gpu*sizeof(float));

  // Allocation mémoire par le CPU du tableau de générateur pseudo-aléatoire
  curandState *d_state;
  CUDA_CALL(cudaMalloc((void **)&d_state, NB_BLOCK*NB_THREAD*sizeof(curandState)));

  // debut du chronometrage
  start = my_gettimeofday();

  #pragma omp parallel num_threads(4)
  {
    // un seul thread appel le kernel
    #pragma omp master
    {
    // On initialise les générateurs
    setup_kernel<<<NB_BLOCK,NB_THREAD>>>(d_state);

    neutron_gpu<<<NB_BLOCK,NB_THREAD>>>(d_state, h, taille_gpu, c_c, c_s, device_absorbed);

    cudaMemcpy(host_absorbed+taille_cpu,device_absorbed,taille_gpu*sizeof(float),cudaMemcpyDeviceToHost);

    cudaMemcpyFromSymbol(&rh, device_r, sizeof(int),0);  
    cudaMemcpyFromSymbol(&bh, device_b, sizeof(int),0); 
    cudaMemcpyFromSymbol(&th, device_t, sizeof(int),0); 

    }
    // tous les autres threads calculent les neutrons
    {
      init_uniform_random_number();
      // Faire partir chaque i avec le numéro de thread for(i = num_thread; i< taille_cpu/nb_thread; i++) un truc comme ça
      #pragma omp for reduction(+:r,b,t) private(u,L,x,d)
      for (i = 0; i < taille_cpu; i++) {
        d = 0.0;
        x = 0.0;
        // printf("thread %d dans le else i = %d;\n",omp_get_thread_num(),i);
        while (1){
          
          u = uniform_random_number();
          L = -(1 / c) * log(u);
          x = x + L * cos(d);
          if (x < 0) {
    	r++;
    	break;
          } else if (x >= h) {
    	t++;
    	break;
          } else if ((u = uniform_random_number()) < c_c / c) {
    	b++;
      j++;
    	host_absorbed[j] = x;
    	break;
          } else {
    	u = uniform_random_number();
    	d = u * M_PI;
          }
        }
      }
    }
    // On s'assure que tous les threads ont terminé leurs tâches
    // #pragma omp barrier
}


  // printf("rh=%d, bh=%d, th=%d\n",rh,bh,th);
  // printf("r=%d, b=%d, t=%d, nb total neutron traités = %d\n",r,b,t,r+b+t);
  r = r + rh;
  b = b + bh;
  t = t + th;
  // fin du chronometrage
  finish = my_gettimeofday();

  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);

  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  FILE *f_handle = fopen(OUTPUT_FILE, "w");
  if (!f_handle) {
    fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
    exit(EXIT_FAILURE);
  }

  for (int j = 0; j < b; j++)
    fprintf(f_handle, "%f\n", host_absorbed[j]);

  // fermeture du fichier
  fclose(f_handle);
  printf("Result written in " OUTPUT_FILE "\n"); 

  cudaFree(d_state);
  cudaFree(device_absorbed);
  free(host_absorbed);

  return EXIT_SUCCESS;
}
