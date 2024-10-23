#include "kseq/kseq.h"
#include "common.h"

__global__ void matchKernel(char* sample, char* sign, int* match, double* score, char* qual,int n, int m, int sample_time, int sign_time, int sample_batch, int sign_batch) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_time*sample_batch+blockIdx.x < n && sign_time*sign_batch+threadIdx.x < m) {
        int ans = -1;
        int lps[10000];
        int prevlps = 0;
        lps[0] = 0;
        int i = 1;
        int l = 0;
        while (i < 10000) {
            if (sign[threadIdx.x*10001+i] == sign[threadIdx.x*10001+prevlps]) {
                ++prevlps;
                lps[i] = prevlps;
                ++i;
            }
            else if (prevlps == 0) {
                lps[i] = 0;
                ++i;
            }
            else{
                prevlps = lps[prevlps-1];
            }
        }
        i = 0;
        int j = 0;
        while (i <= 200000-500+1) {
            if (sample[blockIdx.x*200001+i] == 0 && sign[threadIdx.x*10001 + j] != 0 ) {
                break;
            }
            else if (sign[threadIdx.x*10001 + j] == 0) {
                l = j;
                ans = i - j;
                break;
            }
            else if (sample[blockIdx.x*200001+i] == sign[threadIdx.x*10001 + j]) {
                ++i;
                ++j;
            }
            else if (j == 0) {
                ++i;
            }
            else {
                j = lps[j-1];
            }
        }
        match[tid] = ans;
        if (ans != -1) {
            double sco = 0;
            for (int k = 0; k < l; ++k) {
                sco += qual[blockIdx.x*200001+ans+k] - 33;
            }
            sco /= l;
            score[tid] = sco;
        }
    }
    else {
        match[tid] = -1;
    }
}


void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
    int n = samples.size();
    int m = signatures.size();
    const int sample_batch = 2048;
    const int sign_batch = 1024;
    int sample_time = (n+sample_batch-1)/sample_batch;
    int sign_time = (m+sign_batch-1)/sign_batch;
    if (n <= sample_batch) {
        sample_time = 1;
    }
    else{
        sample_time = (n+sample_batch-1)/sample_batch;
    }
    if (m <= sign_batch) {
        sign_time = 1;
    }
    else {
        sign_time = (m+sign_batch-1)/sign_batch;
    }
    char* d_sign_seq = nullptr;
    char* d_sample_seq = nullptr;
    char* d_qual = nullptr;
    int* d_match = nullptr;
    int* match = new int[sample_batch * sign_batch];
    double* score = new double[sample_batch*sign_batch];
    double* d_score = nullptr;
    cudaStream_t stream0;
    cudaStream_t stream1;
    int sm = 0;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStream_t streams[2] = {stream0, stream1};
    cudaMalloc(&d_sign_seq, sizeof(char) * sign_batch * 10001);
    cudaMalloc(&d_sample_seq, sizeof(char) * sample_batch * 200001);
    cudaMalloc(&d_qual, sizeof(char) * sample_batch * 200001);
    cudaMalloc(&d_match, sizeof(int)*sample_batch*sign_batch);
    cudaMalloc(&d_score, sizeof(double)*sample_batch*sign_batch);
    for (int i = 0; i < sample_time; ++i) {
        for (int s = 0; s < sample_batch; ++s) {
            if (i*sample_batch+s < n) {
                cudaMemcpyAsync(d_sample_seq + 200001 * s, samples[i*sample_batch+s].seq.c_str(), sizeof(char)*200000, cudaMemcpyHostToDevice, streams[(sm)%2]);
                cudaMemcpyAsync(d_qual + 200001 * s, samples[i*sample_batch+s].qual.c_str(), sizeof(char)*200000, cudaMemcpyHostToDevice, streams[(sm)%2]);
                // cudaMemcpy(d_sample_seq + 200001 * s, samples[i*sample_batch+s].seq.c_str(), sizeof(char)*200000, cudaMemcpyHostToDevice);
                // cudaMemcpy(d_qual + 200001 * s, samples[i*sample_batch+s].qual.c_str(), sizeof(char)*200000, cudaMemcpyHostToDevice);
            }
        }
        for (int j = 0; j < sign_time; ++j) {
            for (int t = 0; t < sign_batch; ++t) {
                if (j*sign_batch+t < m) {
                    cudaMemcpyAsync(d_sign_seq + 10001 * t, signatures[j*sign_batch+t].seq.c_str(), sizeof(char)*10000, cudaMemcpyHostToDevice, streams[(sm)%2]);
                    // cudaMemcpy(d_sign_seq + 10001 * t, signatures[j*sign_batch+t].seq.c_str(), sizeof(char)*10000, cudaMemcpyHostToDevice);
                }
            }
            int grid = sample_batch;
            int block = sign_batch;
            matchKernel<<<grid,block,0,streams[(sm)%2]>>>(d_sample_seq,d_sign_seq,d_match,d_score,d_qual,n,m,i,j,sample_batch,sign_batch);
            // matchKernel<<<sample_batch,sign_batch,0>>>(d_sample_seq,d_sign_seq,d_match,d_score,d_qual,n,m,i,j,sample_batch,sign_batch);
            // cudaDeviceSynchronize();
            cudaMemcpyAsync(match, d_match, sizeof(int)*sample_batch*sign_batch, cudaMemcpyDeviceToHost,streams[(sm)%2]);
            cudaMemcpyAsync(score, d_score, sizeof(double)*sample_batch*sign_batch, cudaMemcpyDeviceToHost,streams[(sm)%2]);
            // cudaMemcpy(match, d_match, sizeof(int)*sample_batch*sign_batch, cudaMemcpyDeviceToHost);
            // cudaMemcpy(score, d_score, sizeof(double)*sample_batch*sign_batch, cudaMemcpyDeviceToHost);
            ++sm;
            for (int ii = 0; ii < sample_batch; ++ii) {
                for (int jj = 0; jj < sign_batch; ++jj) {
                    int ans = match[ii*sign_batch+jj];
                    if (ans != -1) {
                        matches.push_back({samples[i*sample_batch+ii].name, signatures[j*sign_batch+jj].name, score[ii*sign_batch+jj]});
                    }
                }
            }                                                                    
        }
    }
    cudaFree(d_sample_seq);
    cudaFree(d_match);
    cudaFree(d_sign_seq);
    cudaFree(d_score);
    cudaFree(d_qual);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    delete[] match;
    delete[] score;
}
