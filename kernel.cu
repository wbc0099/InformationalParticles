#include <cufftXt.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
#include <math.h>
#include <random> 
#include <iomanip>
#include "cuda_runtime.h"
#include <cuda.h>
#include <curand_kernel.h>

//if no run in a long time ,maybe beacause box is too small

//Definitions=======================================================================
// Define the precision of real numbers, could be real/double.
#define real double
#define Pi 3.1415926535897932384626433832795
#define Zero 0
using namespace std;

struct Particle {
    real* x;//save x position in GPU
    real* y;//save y position in GPU
    int* cellX;//save xth cell of nth particle
    int* cellY;//save yth cell of nth particle
    int* cellList;//cell particle id for all particle, as [maxParticlePerCell*id + offsetsCL]?????
    int* cellOffsetsCL;//offset of every cell list to save particle number in this cell 
    int* particleAroundId;//save ids around this on particle, use rd to judge wether is "around"????????
    int* particleAroundFlagX;//mask whether cell of idth particle at the edge of box
    int* particleAroundFlagY;//mask whether cell of idth particle at the edge of box
    int* offsetsAL;//offset of every particle's around list
    int* offsetsNL;//offset of every particle's neighbor list to save neighbor particle id
    int* NeighborList;//neighbor list
    int* NeighborListFlagX;//translate from particleAroundFlagX
    int* NeighborListFlagY;//translate from particleAroundFlagY
    real* fx;//force on the x direction
    real* fy;//force on the y direction
    real* x0ToUpdateHybridList;//save xGpu[id] to judge whether update hybrid list 
    real* y0ToUpdateHybridList;//save yGpu[id] to judge whether update hybrid list
    curandState* state;//save random number generator
    int* aroundNum;
    real* kBT;
} PT, pt;//pt is saved in CPU and PT in GPU

struct Parameter {
    real boxX;//box size X
    real boxY;//box size Y
    real cellSizeX;//cell size in the x direction
    real cellSizeY;//cell size in the y direction
    int cellNumX;//num of cell in the x direction
    int cellNumY;//num of cell in the y direction
    int maxParticlePerCell;//theory maxmum particle number in one cell
    real rd;//deadline distance to get NeighborList
    real miniInstanceBetweenParticle;//theory minimum distance from two particle
    real r0;//balance position
    real epsilon;//coefficient of force
    float kBT;//kB*T
    real gammaValue;//Viscosity coefficien
    real rOutUpdateList;//update hybrid list when any one particle move a distance greater than rOutUpdateList
    int particleNum; //particle number 
    real tStart;//start time
    real tStop;//stop time
    real tStep;//step time
    real tExpo;//export data every tExpo
    unsigned long long seed;//generate random number seed
    int blockNum;//num of block in one grid
    int threadNum;//num of thread in one block
    int nthGPU;//which GPU to use
    real rOff;//offset distance to judge whether particle is "around"
    real rOffIn;//Inner diameter of circular ring
    real N;//coefficient of temperture change
    real forceCoefficient;//coefficient of force
    real rNeighborList;//size neighbor
    int kBTChangeMode;
    int kBTChangePM0;
    real visionConeXLen;
} PM;

__device__ int updateListFlag = 0;
__device__ int wrongFlag = 0;
int updateListFlagHost = 0;
int wrongFlagHost = 0;

void ExpoConf(const std::string& str_t);
void MemFree();
void getInput();
void MemAlloc();
void printInput();
void Init_Coords(int flag, Particle pt, Parameter PM);
void initAroundNum(Particle PT, Parameter PM);
void InitOffset();
void HostUpdataToDevice();
void DeviceUpdataToHost();
void listUpdate(Particle PT,Parameter PM);
void forceAndPositionUpdate(Particle PT, Parameter PM);
void iterate(Particle PT,Parameter PM);
void initBlockAndThreadNum();
void showProgress(real tNow, real tStart, real tStop, clock_t clockNow, clock_t clockStart);
int setDevice(int n);
int printGpuError();
int initAll();
__global__ void initState(curandState* state,unsigned long long seed, int particleNum);
__global__ void getCellList(Particle PT, Parameter PM);
__global__ void getAroundCellParticleId(Particle PT, Parameter PM);
__global__ void saveXY0ToUpdateHybridList(Particle PT, Parameter PM);
__global__ void checkUpdate(Particle PT, Parameter PM);
__global__ void getForce (Particle PT, Parameter PM);
__global__ void updatePosition(Particle PT, Parameter PM);
__device__ int getNeighborListTry(real x0, real y0, real x1, real y1, Parameter PM);
__device__ int sign(real x);
__device__ int sign01(real x);
__device__ real force (real forceCoefficient,real dr,real rd);
__device__ real generateNormal(curandState* state);
__device__ void updateKBT(Particle PT, Parameter PM, int id);

int main()
{
    real tNow = PM.tStart;
    if(!initAll())return 0;    
    ExpoConf("0");

    listUpdate(PT, PM);
    cudaDeviceSynchronize();

    if(!printGpuError())return 0; 

    clock_t clockStart = clock();
    for (tNow = PM.tStart;tNow < PM.tStop;tNow += PM.tStep) {
        iterate(PT, PM);

        cudaMemcpyFromSymbol(&wrongFlagHost, wrongFlag, sizeof(int));
        if (wrongFlagHost == 1)return 0;

        if (floor(tNow / PM.tExpo) > floor((tNow - PM.tStep) / PM.tExpo)) {
            showProgress(tNow, PM.tStart, PM.tStop, clock(), clockStart);
            DeviceUpdataToHost();//下载数据到主机
            int te = floor(tNow / PM.tExpo) + 1;
            string str_t = to_string(te);
            ExpoConf(str_t);
        }
    }

    if(!printGpuError())return 0; 

    MemFree();//释放内存
    cudaDeviceReset();
    return 0; // 返回成功状态
}

int initAll(){
    getInput();
    if(!setDevice(PM.nthGPU))return 0;
    MemAlloc();
    printInput();
    Init_Coords(1, pt, PM);
    initBlockAndThreadNum();
    InitOffset();
    initState << <PM.blockNum, PM.threadNum >> > (PT.state, PM.seed, PM.particleNum);
    cudaDeviceSynchronize();
    HostUpdataToDevice();
    PM.seed = static_cast<unsigned long long>(time(0));
    printf("seed:%d\n", PM.seed);
    return 1;
}

void getInput() {
    std::ifstream InputFile("input.dat");

    if (!InputFile.is_open()) {
        std::cerr << "Error opening input file!" << std::endl;
        return; // 退出函数
    }

    std::string line;
    int lineCount = 0;

    while (std::getline(InputFile, line)) {
        // 检查是否为注释行
        if (line.empty() || line.find('#') != std::string::npos) {
            continue; // 跳过空行和注释行
        }

        std::istringstream iss(line);
        switch (lineCount) {
        case 0: iss >> PM.boxX; break;
        case 1: iss >> PM.boxY; break;
        case 2: iss >> PM.cellSizeX; break;
        case 3: iss >> PM.cellSizeY; break;
        case 4: iss >> PM.cellNumX; break;
        case 5: iss >> PM.cellNumY; break;
        case 6: iss >> PM.maxParticlePerCell; break;
        case 7: iss >> PM.rd; break;
        case 8: iss >> PM.miniInstanceBetweenParticle; break;
        case 9: iss >> PM.r0; break;
        case 10: iss >> PM.epsilon; break;
        case 11: iss >> PM.kBT; break;
        case 12: iss >> PM.gammaValue; break;
        case 13: iss >> PM.rOutUpdateList; break;
        case 14: iss >> PM.particleNum; break;
        case 15: iss >> PM.tStart; break;
        case 16: iss >> PM.tStop; break;
        case 17: iss >> PM.tStep; break;
        case 18: iss >> PM.tExpo; break;
	    case 19: iss >> PM.nthGPU; break;
        case 20: iss >> PM.rOff; break;
        case 21: iss >> PM.rOffIn; break;
        case 22: iss >> PM.N; break;
        case 23: iss >> PM.forceCoefficient; break;
        case 24: iss >> PM.rNeighborList; break;
        case 25: iss >> PM.kBTChangeMode; break;
        case 26: iss >> PM.kBTChangePM0; break;
        case 27: iss >> PM.visionConeXLen; break;
        default: break; // 超过预期行数时不处理
        }
        lineCount++;
    }

    InputFile.close();
}

int setDevice(int n){
    cudaError_t err = cudaSetDevice(n);
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return 1;
}

void printInput() {
    std::cout << "Box X: " << PM.boxX << std::endl;
    std::cout << "Box Y: " << PM.boxY << std::endl;
    std::cout << "Cell size X: " << PM.cellSizeX << std::endl;
    std::cout << "Cell size Y: " << PM.cellSizeY << std::endl;
    std::cout << "Cell num X: " << PM.cellNumX << std::endl;
    std::cout << "Cell num Y: " << PM.cellNumY << std::endl;
    std::cout << "Max particle per cell: " << PM.maxParticlePerCell << std::endl;
    std::cout << "Deadline distance: " << PM.rd << std::endl;
    std::cout << "Mini instance between particle: " << PM.miniInstanceBetweenParticle << std::endl;
    std::cout << "Equilibrium position: " << PM.r0 << std::endl;
    std::cout << "Epsilon: " << PM.epsilon << std::endl;
    std::cout << "kBT: " << PM.kBT << std::endl;
    std::cout << "Gamma value: " << PM.gammaValue << std::endl;
    std::cout << "Update list distance threshold: " << PM.rOutUpdateList << std::endl;
    std::cout << "Particle num: " << PM.particleNum << std::endl;
    std::cout << "Start time: " << PM.tStart << std::endl;
    std::cout << "Stop time: " << PM.tStop << std::endl;
    std::cout << "Time step: " << PM.tStep << std::endl;
    std::cout << "TExpo: " << PM.tExpo << std::endl;
    std::cout << "nthGPU: " << PM.nthGPU << std::endl;
    std::cout << "rOff: " << PM.rOff << std::endl;
    std::cout << "rOffInner: " << PM.rOffIn << std::endl;
    std::cout << "N: " << PM.N << std::endl;
    std::cout << "forceCoefficient: " << PM.forceCoefficient<< std::endl;
    std::cout << "rNeighborList: " << PM.rNeighborList<< std::endl;
    std::cout << "kBTChangeMode: " << PM.kBTChangeMode<< std::endl;
    std::cout << "kBTChangePM0: " << PM.kBTChangePM0<< std::endl;
    std::cout << "visionConeXLen: " << PM.visionConeXLen<< std::endl;
}

void Init_Coords(int flag, Particle pt, Parameter PM) {
    memset(pt.cellList, 0, PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell * sizeof(int));
    memset(pt.cellOffsetsCL, 0, PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell * sizeof(int));
    int N=PM.particleNum;
    real xBox = PM.boxX;
    real yBox = PM.boxY;
    std::default_random_engine e;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    e.seed(time(0));
    real x0, y0, dx, dy;
    int cellX,cellY,cellX1,cellY1,cellAround;
    int xFlag,yFlag;
    int wrongFlag=0;
    for(int i=0;i<N;i++){
        while(1){
            x0=u(e)*xBox;
            y0=u(e)*yBox;
            //printf("id:%d,x:%f,y:%f\n",i,x0,y0);
            wrongFlag=0;
            cellX=std::floor(x0/PM.cellSizeX);
            cellY=std::floor(y0/PM.cellSizeY);
            for(int x=-1;x<=1;x++){
                for(int y=-1;y<=1;y++){
                    if(cellX+x==-1){
                        cellX1=PM.cellNumX-1;
                        xFlag=1;
                    }else if(cellX+x==PM.cellNumX){
                        cellX1=0;
                        xFlag=-1;
                    }else{
                        cellX1=cellX+x;
                        xFlag=0;
                    }
                    if(cellY+y==-1){
                        cellY1=PM.cellNumY-1;
                        yFlag=1;
                    }else if(cellY+y==PM.cellNumY){                    
                        cellY1=0;
                        yFlag=-1;
                    }else{
                        cellY1=cellY+y;
                        yFlag=0;
                    }
                    cellAround=cellX1+cellY1*PM.cellNumX;
                    for(int j=0;j<pt.cellOffsetsCL[cellAround];j++){
                        //printf("cell:%d,cellAround:%d,j:%d,x0:%f,y0:%f,x:%f,y:%f\n",cellX+cellY*PM.cellNumX,cellAround,j,x0,y0,pt.x[pt.cellList[cellAround*PM.maxParticlePerCell+j]],pt.y[pt.cellList[cellAround*PM.maxParticlePerCell+j]]);
                        dx=(x0-pt.x[pt.cellList[cellAround*PM.maxParticlePerCell+j]])+xFlag*PM.boxX;
                        dy=(y0-pt.y[pt.cellList[cellAround*PM.maxParticlePerCell+j]])+yFlag*PM.boxY;
                        if(dx*dx+dy*dy<PM.r0*PM.r0){
                            wrongFlag=1;
                            break;
                        }
                    }
                    if(wrongFlag==1){
                        break;
                    }
                }
                if(wrongFlag==1){ 
                    break;
                }
            }
            if(wrongFlag==0){
                break;
            }else continue;
        }        
        pt.x[i]=x0;
        pt.y[i]=y0;
        pt.cellList[(cellX+cellY*PM.cellNumX)*PM.maxParticlePerCell+pt.cellOffsetsCL[cellX+cellY*PM.cellNumX]]=i;
        pt.cellOffsetsCL[cellX+cellY*PM.cellNumX]++;
    }
}

__global__ void initState(curandState* state,unsigned long long seed, int particleNum) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particleNum)return;
    curand_init(seed, id, 0, &state[id]);
}

void initBlockAndThreadNum() {
    PM.threadNum = 256;
    PM.blockNum = (PM.particleNum + PM.threadNum - 1) / PM.threadNum;
    printf("blockNum:%d,threadNum:%d\n", PM.blockNum, PM.threadNum);
}

void InitOffset() {
    cudaMemset(PT.cellOffsetsCL, 0, sizeof(int) * PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell);
    cudaMemset(PT.offsetsNL, 0, sizeof(int) * PM.particleNum);
    cudaMemset(PT.offsetsAL, 0, sizeof(int) * PM.particleNum);
}

void HostUpdataToDevice() {
    cudaMemcpy(PT.x, pt.x, PM.particleNum * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(PT.y, pt.y, PM.particleNum * sizeof(real), cudaMemcpyHostToDevice);
}

void DeviceUpdataToHost() {
    cudaMemcpy(pt.x, PT.x, PM.particleNum * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(pt.y, PT.y, PM.particleNum * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(pt.kBT, PT.kBT, PM.particleNum * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(pt.aroundNum, PT.aroundNum, PM.particleNum * sizeof(int), cudaMemcpyDeviceToHost);
}

void MemAlloc() {
    // Allocate particle mem in host memory.
    pt.x = new real[PM.particleNum];
    pt.y = new real[PM.particleNum];
    pt.cellList = new int[PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell];
    pt.cellOffsetsCL = new int[PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell];
    pt.particleAroundId = new int[9 * PM.particleNum * PM.maxParticlePerCell];
    pt.particleAroundFlagX = new int[9 * PM.particleNum * PM.maxParticlePerCell];
    pt.particleAroundFlagY = new int[9 * PM.particleNum * PM.maxParticlePerCell];
    pt.offsetsNL = new int[PM.particleNum];
    pt.NeighborList = new int[PM.particleNum * PM.maxParticlePerCell];
    pt.NeighborListFlagX = new int[PM.particleNum];
    pt.NeighborListFlagY = new int[PM.particleNum];
    pt.fx = new real[PM.particleNum];
    pt.fy = new real[PM.particleNum];
    pt.x0ToUpdateHybridList = new real[PM.particleNum];
    pt.y0ToUpdateHybridList = new real[PM.particleNum];
    pt.state = new curandState[PM.particleNum];
    pt.aroundNum = new int[PM.particleNum];
    pt.kBT = new real[PM.particleNum];


    // Allocate memory of fields in device.
    cudaMalloc((void**)&PT.x, PM.particleNum * sizeof(real));
    cudaMalloc((void**)&PT.y, PM.particleNum * sizeof(real));
    cudaMalloc((void**)&PT.cellX, PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.cellY, PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.cellList, PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell * sizeof(int));
    cudaMalloc((void**)&PT.cellOffsetsCL, PM.cellNumX * PM.cellNumY * PM.maxParticlePerCell * sizeof(int));
    cudaMalloc((void**)&PT.particleAroundId, 9 * PM.maxParticlePerCell * PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.particleAroundFlagX, 9 * PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.particleAroundFlagY, 9 * PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.offsetsAL, PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.offsetsNL, PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.NeighborList, PM.particleNum * PM.maxParticlePerCell * sizeof(int));
    cudaMalloc((void**)&PT.NeighborListFlagX, PM.particleNum * PM.maxParticlePerCell * sizeof(int));
    cudaMalloc((void**)&PT.NeighborListFlagY, PM.maxParticlePerCell * PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.fx, PM.particleNum * sizeof(real));
    cudaMalloc((void**)&PT.fy, PM.particleNum * sizeof(real));
    cudaMalloc((void**)&PT.x0ToUpdateHybridList, PM.particleNum * sizeof(real));
    cudaMalloc((void**)&PT.y0ToUpdateHybridList, PM.particleNum * sizeof(real));
    cudaMalloc((void**)&PT.state, PM.particleNum * sizeof(curandState));
    cudaMalloc((void**)&PT.aroundNum, PM.particleNum * sizeof(int));
    cudaMalloc((void**)&PT.kBT, PM.particleNum * sizeof(real));

    cudaMemset(PT.aroundNum, PM.kBT, PM.particleNum * sizeof(int));
    
}

void ExpoConf(const std::string& str_t) {
    std::ofstream ConfFile;
    //设置输出精度
    int PrecData = 8;

    // 文件名
    std::string ConfFileName = "conf_" + str_t + ".dat";
    ConfFile.open(ConfFileName.c_str());

    if (!ConfFile.is_open()) {
        std::cerr << "无法打开文件: " << ConfFileName << std::endl;
        return;
    }
    for (int idx = 0; idx < PM.particleNum; idx++) {
        // 使用固定格式和精度输出数据
        ConfFile << std::fixed << std::setprecision(PrecData)
            << pt.x[idx] << ' '
            << pt.y[idx] << ' '
            << pt.kBT[idx] << ' '
            << pt.aroundNum[idx];
        ConfFile << std::endl; // 换行
    }
    ConfFile.close();
}

void listUpdate(Particle PT,Parameter PM) {
    InitOffset();
    getCellList << <PM.blockNum, PM.threadNum >> > (PT, PM);
    cudaDeviceSynchronize();
    getAroundCellParticleId << <PM.blockNum, PM.threadNum >> > (PT, PM);
    cudaDeviceSynchronize();
    saveXY0ToUpdateHybridList << <PM.blockNum, PM.threadNum >> > (PT, PM);
    cudaDeviceSynchronize();
}

__global__ void getCellList(Particle PT, Parameter PM) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= PM.particleNum)return;
    PT.cellX[id] = std::floor(PT.x[id] / PM.cellSizeX);
    PT.cellY[id] = std::floor(PT.y[id] / PM.cellSizeY);
    int cellId = PT.cellY[id] * PM.cellNumX + PT.cellX[id];
    int offsetsCL = atomicAdd(&PT.cellOffsetsCL[cellId], 1);
    if (offsetsCL < PM.maxParticlePerCell) {
        PT.cellList[cellId * PM.maxParticlePerCell + offsetsCL] = id;
    }
    else {
        printf("wrong: offsetsCL is greater than maxParticlePerCell");//append cout error later
        wrongFlag=1;
    }
}

__global__ void getAroundCellParticleId(Particle PT, Parameter PM) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= PM.particleNum)return;
    int offsetPAI = 0;//particleAroundId put particleId in PAI
    int periodicBoundaryFlagX, periodicBoundaryFlagY;
    int cellXAround, cellYAround;
    int cellAroundId;
    for (int x = -1;x <= 1;x++) {
        for (int y = -1;y <= 1;y++) {
            if (PT.cellX[id] + x == -1) {
                cellXAround = PM.cellNumX - 1;
                periodicBoundaryFlagX = 1;
            }
            else if (PT.cellX[id] + x == PM.cellNumX) {
                cellXAround = 0;
                periodicBoundaryFlagX = -1;

            }
            else {
                cellXAround = PT.cellX[id] + x;
                periodicBoundaryFlagX = 0;
            }
            if (PT.cellY[id] + y == -1) {
                cellYAround = PM.cellNumY - 1;
                periodicBoundaryFlagY = 1;
            }
            else if (PT.cellY[id] + y == PM.cellNumY) {
                cellYAround = 0;
                periodicBoundaryFlagY = -1;
            }
            else {
                cellYAround = PT.cellY[id] + y;
                periodicBoundaryFlagY = 0;
            }
            int cellAroundId = cellYAround * PM.cellNumX + cellXAround;

            for (int i = 0;i < PT.cellOffsetsCL[cellAroundId];i++) {
                if (PT.cellList[cellAroundId * PM.maxParticlePerCell + i] == id)continue;
                int ifNeighbor = getNeighborListTry(PT.x[id], PT.y[id], PT.x[PT.cellList[cellAroundId * PM.maxParticlePerCell + i]]\
                    , PT.y[PT.cellList[cellAroundId * PM.maxParticlePerCell + i]], PM);
                if (ifNeighbor) {
                    PT.NeighborList[id * PM.maxParticlePerCell + PT.offsetsNL[id]] = PT.cellList[cellAroundId * PM.maxParticlePerCell + i];
                    PT.NeighborListFlagX[id * PM.maxParticlePerCell + PT.offsetsNL[id]] = periodicBoundaryFlagX;
                    PT.NeighborListFlagY[id * PM.maxParticlePerCell + PT.offsetsNL[id]] = periodicBoundaryFlagY;//nodebug
                    atomicAdd(&PT.offsetsNL[id], 1);
                }
            }
        }
    }
}

__device__ int getNeighborListTry(real x0, real y0, real x1, real y1, Parameter PM) {
    real dx = sign(x1 - x0) * (x1 - x0);
    real dy = sign(y1 - y0) * (y1 - y0);
    dx = sign01(0.5 * PM.boxX - dx) * dx + sign01(dx - 0.5 * PM.boxX) * (PM.boxX - dx);
    dy = sign01(0.5 * PM.boxY - dy) * dy + sign01(dy - 0.5 * PM.boxY) * (PM.boxY - dy);
    real dr2 = dx * dx + dy * dy;
    if (dr2 < PM.rNeighborList * PM.rNeighborList) return 1;
    else return 0;
}

__global__ void saveXY0ToUpdateHybridList(Particle PT, Parameter PM) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= PM.particleNum)return;
    PT.x0ToUpdateHybridList[id] = PT.x[id];
    PT.y0ToUpdateHybridList[id] = PT.y[id];
}

int printGpuError(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

void iterate(Particle PT,Parameter PM) {
    forceAndPositionUpdate(PT,PM);
    checkUpdate << <PM.blockNum, PM.threadNum >> > (PT, PM);
    cudaMemcpyFromSymbol(&updateListFlagHost, updateListFlag, sizeof(int));
    if (updateListFlagHost){
        listUpdate(PT, PM);
        updateListFlagHost = 0;
        cudaMemcpyToSymbol(updateListFlag, &updateListFlagHost, sizeof(int));
    }
}

void forceAndPositionUpdate(Particle PT, Parameter PM) {
    initAroundNum(PT, PM);
    getForce << <PM.blockNum, PM.threadNum >> > (PT, PM);
    cudaDeviceSynchronize();
    updatePosition << <PM.blockNum, PM.threadNum >> > (PT, PM);
    cudaDeviceSynchronize();
}

void initAroundNum(Particle PT, Parameter PM) {
    cudaMemset(PT.aroundNum, 0, sizeof(int) * PM.particleNum);
}

__global__ void getForce (Particle PT, Parameter PM) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= PM.particleNum)return;
    real x0, y0, x1, y1, dx, dy, dr, f12;
    PT.fx[id] = 0;
    PT.fy[id] = 0;
    int i;
    PT.aroundNum[id] = 0;
    for (i = 0;i < PT.offsetsNL[id];i++) {
        real disp=0.5;
        //x0 = PT.x[id]<PM.boxX-disp ? PT.x[id]+disp:PT.x[id]+disp-PM.boxX;
        x0 = PT.x[id];
        y0 = PT.y[id];
        x1 = PT.x[PT.NeighborList[id * PM.maxParticlePerCell + i]];
        y1 = PT.y[PT.NeighborList[id * PM.maxParticlePerCell + i]];
        dx = sign01(0.5 * PM.boxX - x0 + x1) * sign01(0.5 * PM.boxX + x0 - x1) * (x0 - x1) + \
            sign01(sign(x0 - x1) * (x0 - x1) - 0.5 * PM.boxX) * -sign(x0 - x1) * (PM.boxX - sign(x0 - x1) * (x0 - x1));
        dy = sign01(0.5 * PM.boxY - y0 + y1) * sign01(0.5 * PM.boxY + y0 - y1) * (y0 - y1) + \
            sign01(sign(y0 - y1) * (y0 - y1) - 0.5 * PM.boxY) * -sign(y0 - y1) * (PM.boxY - sign(y0 - y1) * (y0 - y1));
        dr = sqrt(dx * dx + dy * dy);
        
        real x01 = PT.x[id]<PM.boxX-disp ? PT.x[id]+disp:PT.x[id]+disp-PM.boxX;
        real x02 = PT.x[id]<PM.boxX-2*disp ? PT.x[id]+2*disp:PT.x[id]+2*disp-PM.boxX;
        real dx1 = sign01(0.5 * PM.boxX - x01 + x1) * sign01(0.5 * PM.boxX + x01 - x1) * (x01 - x1) + \
            sign01(sign(x01 - x1) * (x01 - x1) - 0.5 * PM.boxX) * -sign(x01 - x1) * (PM.boxX - sign(x01 - x1) * (x01 - x1));
        real dx2 = sign01(0.5 * PM.boxX - x02 + x1) * sign01(0.5 * PM.boxX + x02 - x1) * (x02 - x1) + \
            sign01(sign(x02 - x1) * (x02 - x1) - 0.5 * PM.boxX) * -sign(x02 - x1) * (PM.boxX - sign(x02 - x1) * (x02 - x1));
        real dr1 = sqrt(dx1 * dx1 + dy * dy);
        real dr2 = sqrt(dx2 * dx2 + dy * dy);
        
        //if(dr<PM.rOff && dx>(dr*PM.visionConeXLen) && dr>PM.rOffIn){
        //if(dx>-4 && dx<4 && dy>-0.5 && dy<0.5){
        if(dr<1 || (dr1<2 && dr2>1 && dy>0)){
            PT.aroundNum[id] += 1;
        }

       f12=0;
        
        if (PT.fx[id] > 10000 || PT.fx[id] < -10000 || PT.fy[id] > 10000 || PT.fy[id] < -10000) {
            break;
        }
    }
    if (PT.fx[id] > 10000 || PT.fx[id] < -10000) {
        printf("wrong!!!!!!!!!id:%d,fx:%f,fy:%f,dx:%f,dy:%f,x0:%f,x1:%f,y0:%f,y1:%f,NLFX:%d\n", id,PT.fx[id], PT.fy[id], dx,dy,x0,x1, y0, y1, PT.NeighborListFlagX[id * PM.maxParticlePerCell + i]);
        wrongFlag = 1;
    }
    if (PT.fy[id] > 10000 || PT.fy[id] < -10000) {
        printf("wrong!!!!!!!!!id:%d,fx:%f,fy:%f,dx:%f,dy:%f,y0:%f,y1:%f,x0:%f,x1:%f,NLFY:%d\n", id, PT.fx[id], PT.fy[id],dx,dy,y0,y1, x0, x1, PT.NeighborListFlagY[id * PM.maxParticlePerCell + i]);
        wrongFlag = 1;
    }
}

__device__ real force (real forceCoefficient,real dr,real rd){
    return forceCoefficient*(rd-dr)*(rd-dr);
}

__device__ void updateKBT(Particle PT, Parameter PM, int id){
    if (PM.kBTChangeMode == 1){
        PT.kBT[id]=PM.kBT+PM.N*(PT.aroundNum[id]-PM.kBTChangePM0)*(PT.aroundNum[id]-PM.kBTChangePM0)/10;
    } else if (PM.kBTChangeMode == 2){
        PT.kBT[id]=PM.kBT+PM.N*sign(PT.aroundNum[id]-PM.kBTChangePM0)*(PT.aroundNum[id]-PM.kBTChangePM0);
    } else if (PM.kBTChangeMode == 3){
        PT.kBT[id]=PM.kBT+PM.N*(sin((PT.aroundNum[id]-PM.kBTChangePM0)*Pi/3-Pi/2)+1);
    } 
}

__global__ void updatePosition(Particle PT, Parameter PM) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= PM.particleNum)return;
    updateKBT(PT,PM,id);
    real fT = sqrt(2 * PT.kBT[id] * PM.gammaValue * PM.tStep);
    real FRx = generateNormal(&PT.state[id]);
    real FRy = generateNormal(&PT.state[id]);
    //PT.x[id] = fmod(PT.x[id] + (PT.fx[id] * PM.tStep + fT * FRx) / PM.gammaValue + PM.boxX, PM.boxX);
    //PT.y[id] = fmod(PT.y[id] + (PT.fy[id] * PM.tStep + fT * FRy) / PM.gammaValue + PM.boxY, PM.boxY);
    PT.x[id] = fmod(PT.x[id] + (fT * FRx) / PM.gammaValue + PM.boxX, PM.boxX);
    PT.y[id] = fmod(PT.y[id] + (fT * FRy) / PM.gammaValue + PM.boxY, PM.boxY);

}

__device__ real generateNormal(curandState* state) {
    return curand_normal(&(*state));
}

__device__ int sign(real x) {
    return -(x < 0.f) + (x > 0.f);
}

__device__ int sign01(real x) {
    return (sign(x) + 1) / 2;
}

__global__ void checkUpdate(Particle PT, Parameter PM) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= PM.particleNum)return;
    real x1 = PT.x[id], x0 = PT.x0ToUpdateHybridList[id];
    real y1 = PT.y[id], y0 = PT.y0ToUpdateHybridList[id];
    real dx = sign(x1 - x0) * (x1 - x0);
    real dy = sign(y1 - y0) * (y1 - y0);
    dx = sign01(0.5 * PM.boxX - dx) * dx + sign01(dx - 0.5 * PM.boxX) * (PM.boxX - dx);
    dy = sign01(0.5 * PM.boxY - dy) * dy + sign01(dy - 0.5 * PM.boxY) * (PM.boxY - dy);
    if ((dx * dx + dy * dy) > PM.rOutUpdateList * PM.rOutUpdateList) atomicExch(&updateListFlag, 1);
}

void showProgress(real tNow, real tStart, real tStop, clock_t clockNow, clock_t clockStart) {
    real progress = (tNow - tStart) / (tStop - tStart);
    real tUsed = double(clockNow - clockStart) / CLOCKS_PER_SEC;
    real tUsePrediction = (tStop - tNow) * tUsed / (tNow - tStart);
    printf("First Particle(test Error): %.8f,%.8f\t", pt.x[0], pt.y[0]);
    printf("  Progress:%.4f\%,Prediction:%.1fs\t\r", progress*100, tUsePrediction);
    fflush(stdout);
}

void MemFree() {
    // Free host memory
    delete[] pt.x;
    delete[] pt.y;
    delete[] pt.cellList;
    delete[] pt.cellOffsetsCL;
    delete[] pt.particleAroundId;
    delete[] pt.particleAroundFlagX;
    delete[] pt.particleAroundFlagY;
    delete[] pt.offsetsNL;
    delete[] pt.offsetsAL;
    delete[] pt.NeighborList;
    delete[] pt.NeighborListFlagX;
    delete[] pt.NeighborListFlagY;
    delete[] pt.fx;
    delete[] pt.fy;
    delete[] pt.x0ToUpdateHybridList;
    delete[] pt.y0ToUpdateHybridList;
    delete[] pt.state;
    delete[] pt.aroundNum;
    delete[] pt.kBT;

    // Free device memory
    cudaFree(pt.x);
    cudaFree(pt.y);
    cudaFree(pt.cellX);


    // Free device memory
    cudaFree(PT.x);
    cudaFree(PT.y);
    cudaFree(PT.cellX);
    cudaFree(PT.cellY);
    cudaFree(PT.cellList);
    cudaFree(PT.cellOffsetsCL);
    cudaFree(PT.particleAroundId);
    cudaFree(PT.particleAroundFlagX);
    cudaFree(PT.particleAroundFlagY);
    cudaFree(PT.offsetsAL);
    cudaFree(PT.offsetsNL);
    cudaFree(PT.NeighborList);
    cudaFree(PT.NeighborListFlagX);
    cudaFree(PT.NeighborListFlagY);
    cudaFree(PT.fx);
    cudaFree(PT.fy);
    cudaFree(PT.x0ToUpdateHybridList);
    cudaFree(PT.y0ToUpdateHybridList);
    cudaFree(PT.state);
    cudaFree(PT.aroundNum);
    cudaFree(PT.kBT);
}
