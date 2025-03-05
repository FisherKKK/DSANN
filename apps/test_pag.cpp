//
// Created by ECNU on 2024/9/10.
//
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>

#include "pa_graph.h"

#define TEST


#ifdef TEST
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <mkl.h>
#include <boost/program_options.hpp>
#include <unordered_map>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#ifdef _WINDOWS
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#include "filter_utils.h"
#include "utils.h"

// WORKS FOR UPTO 2 BILLION POINTS (as we use INT INSTEAD OF UNSIGNED)

#define PARTSIZE 10000000
#define ALIGNMENT 512

// custom types (for readability)
typedef tsl::robin_set<std::string> label_set;
typedef std::string path;

namespace po = boost::program_options;

template <class T> T div_round_up(const T numerator, const T denominator)
{
    return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}

using pairIF = std::pair<size_t, float>;
struct cmpmaxstruct
{
    bool operator()(const pairIF &l, const pairIF &r)
    {
        return l.second < r.second;
    };
};

using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

template <class T> T *aligned_malloc(const size_t n, const size_t alignment)
{
#ifdef _WINDOWS
    return (T *)_aligned_malloc(sizeof(T) * n, alignment);
#else
    return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
}

inline bool custom_dist(const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b)
{
    return a.second < b.second;
}

void compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const uint64_t dim)
{
    assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
    for (int64_t d = 0; d < num_points; ++d)
        points_l2sq[d] = cblas_sdot((int64_t)dim, matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1,
                                    matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1);
}

void distsq_to_points(const size_t dim,
                      float *dist_matrix, // Col Major, cols are queries, rows are points
                      size_t npoints, const float *const points,
                      const float *const points_l2sq, // points in Col major
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq, // queries in Col major
                      float *ones_vec = NULL)          // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-2.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, points_l2sq, npoints,
                ones_vec, nqueries, (float)1.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, ones_vec, npoints,
                queries_l2sq, nqueries, (float)1.0, dist_matrix, npoints);
    if (ones_vec_alloc)
        delete[] ones_vec;
}

void inner_prod_to_points(const size_t dim,
                          float *dist_matrix, // Col Major, cols are queries, rows are points
                          size_t npoints, const float *const points, size_t nqueries, const float *const queries,
                          float *ones_vec = NULL) // Scratchspace of num_data size and init to 1.0
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-1.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);

    if (ones_vec_alloc)
        delete[] ones_vec;
}

void exact_knn(const size_t dim, const size_t k,
               size_t *const closest_points,     // k * num_queries preallocated, col
                                                 // major, queries columns
               float *const dist_closest_points, // k * num_queries
                                                 // preallocated, Dist to
                                                 // corresponding closes_points
               size_t npoints,
               float *points_in, // points in Col major
               size_t nqueries, float *queries_in,
               diskann::Metric metric = diskann::Metric::L2) // queries in Col major
{
    float *points_l2sq = new float[npoints];
    float *queries_l2sq = new float[nqueries];
    compute_l2sq(points_l2sq, points_in, npoints, dim);
    compute_l2sq(queries_l2sq, queries_in, nqueries, dim);

    float *points = points_in;
    float *queries = queries_in;

    if (metric == diskann::Metric::COSINE)
    { // we convert cosine distance as
      // normalized L2 distnace
        points = new float[npoints * dim];
        queries = new float[nqueries * dim];
#pragma omp parallel for schedule(static, 4096)
        for (int64_t i = 0; i < (int64_t)npoints; i++)
        {
            float norm = std::sqrt(points_l2sq[i]);
            if (norm == 0)
            {
                norm = std::numeric_limits<float>::epsilon();
            }
            for (uint32_t j = 0; j < dim; j++)
            {
                points[i * dim + j] = points_in[i * dim + j] / norm;
            }
        }

#pragma omp parallel for schedule(static, 4096)
        for (int64_t i = 0; i < (int64_t)nqueries; i++)
        {
            float norm = std::sqrt(queries_l2sq[i]);
            if (norm == 0)
            {
                norm = std::numeric_limits<float>::epsilon();
            }
            for (uint32_t j = 0; j < dim; j++)
            {
                queries[i * dim + j] = queries_in[i * dim + j] / norm;
            }
        }
        // recalculate norms after normalizing, they should all be one.
        compute_l2sq(points_l2sq, points, npoints, dim);
        compute_l2sq(queries_l2sq, queries, nqueries, dim);
    }

    std::cout << "Going to compute " << k << " NNs for " << nqueries << " queries over " << npoints << " points in "
              << dim << " dimensions using";
    if (metric == diskann::Metric::INNER_PRODUCT)
        std::cout << " MIPS ";
    else if (metric == diskann::Metric::COSINE)
        std::cout << " Cosine ";
    else
        std::cout << " L2 ";
    std::cout << "distance fn. " << std::endl;

    size_t q_batch_size = (1 << 9);
    float *dist_matrix = new float[(size_t)q_batch_size * (size_t)npoints];

    for (size_t b = 0; b < div_round_up(nqueries, q_batch_size); ++b)
    {
        int64_t q_b = b * q_batch_size;
        int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

        if (metric == diskann::Metric::L2 || metric == diskann::Metric::COSINE)
        {
            distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, q_e - q_b,
                             queries + (ptrdiff_t)q_b * (ptrdiff_t)dim, queries_l2sq + q_b);
        }
        else
        {
            inner_prod_to_points(dim, dist_matrix, npoints, points, q_e - q_b,
                                 queries + (ptrdiff_t)q_b * (ptrdiff_t)dim);
        }
        std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")" << std::endl;

#pragma omp parallel for schedule(dynamic, 16)
        for (long long q = q_b; q < q_e; q++)
        {
            maxPQIFCS point_dist;
            for (size_t p = 0; p < k; p++)
                point_dist.emplace(p, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
            for (size_t p = k; p < npoints; p++)
            {
                if (point_dist.top().second > dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints])
                    point_dist.emplace(p, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
                if (point_dist.size() > k)
                    point_dist.pop();
            }
            for (ptrdiff_t l = 0; l < (ptrdiff_t)k; ++l)
            {
                closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = point_dist.top().first;
                dist_closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = point_dist.top().second;
                point_dist.pop();
            }
            assert(std::is_sorted(dist_closest_points + (ptrdiff_t)q * (ptrdiff_t)k,
                                  dist_closest_points + (ptrdiff_t)(q + 1) * (ptrdiff_t)k));
        }
        std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
    }

    delete[] dist_matrix;

    delete[] points_l2sq;
    delete[] queries_l2sq;

    if (metric == diskann::Metric::COSINE)
    {
        delete[] points;
        delete[] queries;
    }
}

#endif

static void fvecs_read(const char *path, float vecs[], int n, int d) {
    std::fstream fin(path, std::ios_base::binary | std::ios_base::in);
    assert(fin);

    int df; // read dimension in the file
    fin.read((char*)&df, sizeof(df));
    assert(df == d);

    int dummy; // dummy for one
    for (int i = 0; i < n; i++) {
        float *vec_i = vecs + i * d;
        fin.read((char*)vec_i, sizeof(*vecs) * d);
        fin.read((char*)&dummy, sizeof(dummy));
    }
}

// x是rs, y是gt
float calculate_recall_k(uint32_t *x, size_t *y, int k) {
    float number = 0.f;
    std::unordered_set<size_t> set;
    for (size_t i = 0; i != k; i++) set.insert(x[i]);
    for (size_t i = 0; i != k; i++)
        if (set.count(y[i])) number += 1;
    return number / k;
}


int main() {
//    int d = 128;      // dimension
//    int nb = 1000000; // database size
//    int nq = 10000;  // nb of queries

int d = 128;      // dimension
int nb = 10000; // database size
int nq = 100;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);

    omp_set_num_threads(1);
    fvecs_read("/home/dataset/siftsmall/siftsmall_base.fvecs", xb.data(), nb, d);
    fvecs_read("/home/dataset/siftsmall/siftsmall_query.fvecs", xq.data(), nq, d);

//    fvecs_read("/home/dataset/sift/sift_base.fvecs", xb, nb, d);
//    fvecs_read("/home/dataset/sift/sift_query.fvecs", xq, nq, d);

//    for (int i = 0; i < nb; i++) {
//        for (int j = 0; j < d; j++)
//            xb[d * i + j] = distrib(rng);
//        xb[d * i] += i / 1000.;
//    }
//
//    for (int i = 0; i < nq; i++) {
//        for (int j = 0; j < d; j++)
//            xq[d * i + j] = distrib(rng);
//        xq[d * i] += i / 100.;
//    }

    int k = 10;
    std::vector<size_t> gt(nq * k);
    std::vector<float> gt_dist(nq * k);
    {
        exact_knn(d, k, gt.data(), gt_dist.data(), nb, xb.data(), nq, xq.data());
    }

    {
//        omp_set_num_threads(1);
        auto pag = std::make_unique<diskann::PointAggregationGraph>();
        pag->BuildInDisk("/home/dataset/siftsmall/siftsmall_base.fbin", "/data/siftsmall/pag_disk/index_pag");
        std::cout << "Complete index build" << std::endl;

        float recall = 0.f;
        std::vector<uint32_t> result(k);
        std::vector<float> dists(k);
        omp_set_num_threads(1);
        for (size_t i = 0 ; i < nq; i++) {
            float *q = xq.data() + i * d;
            pag->SearchInDiskByRoute(q, k, 30, result.data(), dists.data());
            recall += calculate_recall_k(result.data(), gt.data() + k * i, k);
        }

        diskann::cout << recall / nq << "\n";

    }
}