// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <limits>
#include <malloc.h>
#include <math_utils.h>
#include <mkl.h>
#include "logger.h"
#include "utils.h"

namespace math_utils
{

float calc_distance(float *vec_1, float *vec_2, size_t dim)
{
    float dist = 0;
    for (size_t j = 0; j < dim; j++)
    {
        dist += (vec_1[j] - vec_2[j]) * (vec_1[j] - vec_2[j]);
    }
    return dist;
}

// compute l2-squared norms of data stored in row major num_points * dim,
// needs
// to be pre-allocated
// 计算向量的l2-square范数
void compute_vecs_l2sq(float *vecs_l2sq, float *data, const size_t num_points, const size_t dim)
{
#pragma omp parallel for schedule(static, 8192)
    for (int64_t n_iter = 0; n_iter < (int64_t)num_points; n_iter++)
    {
        vecs_l2sq[n_iter] = cblas_snrm2((MKL_INT)dim, (data + (n_iter * dim)), 1);
        vecs_l2sq[n_iter] *= vecs_l2sq[n_iter];
    }
}

void rotate_data_randomly(float *data, size_t num_points, size_t dim, float *rot_mat, float *&new_mat,
                          bool transpose_rot)
{
    CBLAS_TRANSPOSE transpose = CblasNoTrans;
    if (transpose_rot)
    {
        diskann::cout << "Transposing rotation matrix.." << std::flush;
        transpose = CblasTrans;
    }
    diskann::cout << "done Rotating data with random matrix.." << std::flush;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, transpose, (MKL_INT)num_points, (MKL_INT)dim, (MKL_INT)dim, 1.0, data,
                (MKL_INT)dim, rot_mat, (MKL_INT)dim, 0, new_mat, (MKL_INT)dim);

    diskann::cout << "done." << std::endl;
}

// calculate k closest centers to data of num_points * dim (row major)
// centers is num_centers * dim (row major)
// data_l2sq has pre-computed squared norms of data
// centers_l2sq has pre-computed squared norms of centers
// pre-allocated center_index will contain id of nearest center
// pre-allocated dist_matrix shound be num_points * num_centers and contain
// squared distances
// Default value of k is 1

// Ideally used only by compute_closest_centers
// 计算距离矩阵, center_index保存了最近邻的k个点
void compute_closest_centers_in_block(const float *const data, const size_t num_points, const size_t dim,
                                      const float *const centers, const size_t num_centers,
                                      const float *const docs_l2sq, const float *const centers_l2sq,
                                      uint32_t *center_index, float *const dist_matrix, size_t k)
{
    if (k > num_centers)
    {
        diskann::cout << "ERROR: k (" << k << ") > num_center(" << num_centers << ")" << std::endl;
        return;
    }

    float *ones_a = new float[num_centers];
    float *ones_b = new float[num_points];

    for (size_t i = 0; i < num_centers; i++)
    {
        ones_a[i] = 1.0;
    }
    for (size_t i = 0; i < num_points; i++)
    {
        ones_b[i] = 1.0;
    }

    // 采用 |a - b|^2 = a^2 + b^2 - 2ab
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)num_points, (MKL_INT)num_centers, (MKL_INT)1, 1.0f,
                docs_l2sq, (MKL_INT)1, ones_a, (MKL_INT)1, 0.0f, dist_matrix, (MKL_INT)num_centers);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)num_points, (MKL_INT)num_centers, (MKL_INT)1, 1.0f,
                ones_b, (MKL_INT)1, centers_l2sq, (MKL_INT)1, 1.0f, dist_matrix, (MKL_INT)num_centers);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)num_points, (MKL_INT)num_centers, (MKL_INT)dim, -2.0f,
                data, (MKL_INT)dim, centers, (MKL_INT)dim, 1.0f, dist_matrix, (MKL_INT)num_centers);
    // k = 1计算最近邻, 保存在center_index中
    if (k == 1)
    {
#pragma omp parallel for schedule(static, 8192)
        for (int64_t i = 0; i < (int64_t)num_points; i++)
        {
            // dist_matrix形状为 num_points x num_centers
            float min = std::numeric_limits<float>::max();
            float *current = dist_matrix + (i * num_centers);
            for (size_t j = 0; j < num_centers; j++)
            {
                if (current[j] < min)
                {
                    // 保存最近
                    center_index[i] = (uint32_t)j;
                    min = current[j];
                }
            }
        }
    }
    else
    {
#pragma omp parallel for schedule(static, 8192)
        // 这里采用优先队列保存最近邻的k个点
        for (int64_t i = 0; i < (int64_t)num_points; i++)
        {
            std::priority_queue<PivotContainer> top_k_queue;
            float *current = dist_matrix + (i * num_centers);
            // push操作
            for (size_t j = 0; j < num_centers; j++)
            {
                PivotContainer this_piv(j, current[j]);
                top_k_queue.push(this_piv);
            }
            // 取top-k操作
            for (size_t j = 0; j < k; j++)
            {
                PivotContainer this_piv = top_k_queue.top();
                center_index[i * k + j] = (uint32_t)this_piv.piv_id;
                top_k_queue.pop();
            }
        }
    }
    delete[] ones_a;
    delete[] ones_b;
}

// Given data in num_points * new_dim row major
// Pivots stored in full_pivot_data as num_centers * new_dim row major
// Calculate the k closest pivot for each point and store it in vector
// closest_centers_ivf (row major, num_points*k) (which needs to be allocated
// outside) Additionally, if inverted index is not null (and pre-allocated),
// it
// will return inverted index for each center, assuming each of the inverted
// indices is an empty vector. Additionally, if pts_norms_squared is not null,
// then it will assume that point norms are pre-computed and use those values

// 为每个vector计算最近邻的k个pivot, 结果保存在inverted_index中
void compute_closest_centers(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers,
                             size_t k, uint32_t *closest_centers_ivf, std::vector<size_t> *inverted_index,
                             float *pts_norms_squared)
{
    if (k > num_centers)
    {
        diskann::cout << "ERROR: k (" << k << ") > num_center(" << num_centers << ")" << std::endl;
        return;
    }

    bool is_norm_given_for_pts = (pts_norms_squared != NULL);

    float *pivs_norms_squared = new float[num_centers];
    if (!is_norm_given_for_pts)
        pts_norms_squared = new float[num_points];

    size_t PAR_BLOCK_SIZE = num_points;
    size_t N_BLOCKS =
        (num_points % PAR_BLOCK_SIZE) == 0 ? (num_points / PAR_BLOCK_SIZE) : (num_points / PAR_BLOCK_SIZE) + 1;

    if (!is_norm_given_for_pts)
        math_utils::compute_vecs_l2sq(pts_norms_squared, data, num_points, dim);
    math_utils::compute_vecs_l2sq(pivs_norms_squared, pivot_data, num_centers, dim);
    // 保存最近邻的k个pivot, 每个节点
    uint32_t *closest_centers = new uint32_t[PAR_BLOCK_SIZE * k];
    // 每个center到points的距离, 这里的矩阵形状是pts * cpt
    float *distance_matrix = new float[num_centers * PAR_BLOCK_SIZE];

    // 对所有的block进行遍历
    for (size_t cur_blk = 0; cur_blk < N_BLOCKS; cur_blk++)
    {
        // 当前block的数据
        float *data_cur_blk = data + cur_blk * PAR_BLOCK_SIZE * dim;
        size_t num_pts_blk = std::min(PAR_BLOCK_SIZE, num_points - cur_blk * PAR_BLOCK_SIZE);
        // 当前block的l2-square
        float *pts_norms_blk = pts_norms_squared + cur_blk * PAR_BLOCK_SIZE;

        // 以块为单位计算最近, 计算结果保存再distance_matrix, closest_centers中
        math_utils::compute_closest_centers_in_block(data_cur_blk, num_pts_blk, dim, pivot_data, num_centers,
                                                     pts_norms_blk, pivs_norms_squared, closest_centers,
                                                     distance_matrix, k);

        // 这里就相当于将刚刚的计算结果保存到inverted_index和closest_centers_ivf中
#pragma omp parallel for schedule(static, 1)
        for (int64_t j = cur_blk * PAR_BLOCK_SIZE;
             j < std::min((int64_t)num_points, (int64_t)((cur_blk + 1) * PAR_BLOCK_SIZE)); j++)
        {
            for (size_t l = 0; l < k; l++)
            {
                size_t this_center_id = closest_centers[(j - cur_blk * PAR_BLOCK_SIZE) * k + l];
                closest_centers_ivf[j * k + l] = (uint32_t)this_center_id;
                if (inverted_index != NULL)
                {
#pragma omp critical
                    inverted_index[this_center_id].push_back(j);
                }
            }
        }
    }
    delete[] closest_centers;
    delete[] distance_matrix;
    delete[] pivs_norms_squared;
    if (!is_norm_given_for_pts)
        delete[] pts_norms_squared;
}

// if to_subtract is 1, will subtract nearest center from each row. Else will
// add. Output will be in data_load iself.
// Nearest centers need to be provided in closst_centers.
void process_residuals(float *data_load, size_t num_points, size_t dim, float *cur_pivot_data, size_t num_centers,
                       uint32_t *closest_centers, bool to_subtract)
{
    diskann::cout << "Processing residuals of " << num_points << " points in " << dim << " dimensions using "
                  << num_centers << " centers " << std::endl;
#pragma omp parallel for schedule(static, 8192)
    for (int64_t n_iter = 0; n_iter < (int64_t)num_points; n_iter++)
    {
        for (size_t d_iter = 0; d_iter < dim; d_iter++)
        {
            if (to_subtract == 1)
                data_load[n_iter * dim + d_iter] =
                    data_load[n_iter * dim + d_iter] - cur_pivot_data[closest_centers[n_iter] * dim + d_iter];
            else
                data_load[n_iter * dim + d_iter] =
                    data_load[n_iter * dim + d_iter] + cur_pivot_data[closest_centers[n_iter] * dim + d_iter];
        }
    }
}

} // namespace math_utils

namespace kmeans
{

// run Lloyds one iteration
// Given data in row major num_points * dim, and centers in row major
// num_centers * dim And squared lengths of data points, output the closest
// center to each data point, update centers, and also return inverted index.
// If
// closest_centers == NULL, will allocate memory and return. Similarly, if
// closest_docs == NULL, will allocate memory and return.

// 运行一次Lloyds迭代
float lloyds_iter(float *data, size_t num_points, size_t dim, float *centers, size_t num_centers, float *docs_l2sq,
                  std::vector<size_t> *closest_docs, uint32_t *&closest_center)
{
    // 这里本质上就是计算冗余量
    bool compute_residual = true;
    // Timer timer;

    if (closest_center == NULL)
        closest_center = new uint32_t[num_points];
    if (closest_docs == NULL)
        closest_docs = new std::vector<size_t>[num_centers];
    else // 相当于对closest_docs进行初始化
        for (size_t c = 0; c < num_centers; ++c)
            closest_docs[c].clear();

    // 计算所有vector最近邻的pivot, 结果保存在closest_center, closest_docs
    math_utils::compute_closest_centers(data, num_points, dim, centers, num_centers, 1, closest_center, closest_docs,
                                        docs_l2sq);
    // 将中心点归0
    memset(centers, 0, sizeof(float) * (size_t)num_centers * (size_t)dim);

    //! 这里本质上就是计算新的center, 按照聚类后的结果
#pragma omp parallel for schedule(static, 1)
    for (int64_t c = 0; c < (int64_t)num_centers; ++c)
    {
        float *center = centers + (size_t)c * (size_t)dim;

        // cluster_sum对应的值就是所有维度的和
        double *cluster_sum = new double[dim];
        for (size_t i = 0; i < dim; i++)
            cluster_sum[i] = 0.0;

        // 获取中心点c的所有最近邻进行遍历
        for (size_t i = 0; i < closest_docs[c].size(); i++)
        {
            // 获取这个点
            float *current = data + ((closest_docs[c][i]) * dim);
            for (size_t j = 0; j < dim; j++)
            {   // 加上所有维度
                cluster_sum[j] += (double)current[j];
            }
        }
        // 这里就是取均值, 也就是现在的center是当前类的均值
        if (closest_docs[c].size() > 0)
        {
            for (size_t i = 0; i < dim; i++)
                center[i] = (float)(cluster_sum[i] / ((double)closest_docs[c].size()));
        }
        delete[] cluster_sum;
    }

    float residual = 0.0;
    if (compute_residual)
    {
        size_t BUF_PAD = 32;
        size_t CHUNK_SIZE = 2 * 8192;
        // 计算有多少个chunk, 将所有的点分块
        size_t nchunks = num_points / CHUNK_SIZE + (num_points % CHUNK_SIZE == 0 ? 0 : 1);
        std::vector<float> residuals(nchunks * BUF_PAD, 0.0);

#pragma omp parallel for schedule(static, 32)
        for (int64_t chunk = 0; chunk < (int64_t)nchunks; ++chunk)
            // 这里就相当于对这个chunk中的点进行遍历, 那么residuals[chunk*BUF_PAD]中存储的就是这个距离之和
            for (size_t d = chunk * CHUNK_SIZE; d < num_points && d < (chunk + 1) * CHUNK_SIZE; ++d)
                residuals[chunk * BUF_PAD] +=
                    math_utils::calc_distance(data + (d * dim), centers + (size_t)closest_center[d] * (size_t)dim, dim);
        // 这里就相当于将所有的residual合并在一起
        for (size_t chunk = 0; chunk < nchunks; ++chunk)
            residual += residuals[chunk * BUF_PAD];
    }

    return residual;
}

// Run Lloyds until max_reps or stopping criterion
// If you pass NULL for closest_docs and closest_center, it will NOT return
// the
// results, else it will assume appriate allocation as closest_docs = new
// vector<size_t> [num_centers], and closest_center = new size_t[num_points]
// Final centers are output in centers as row major num_centers * dim
//
// 从获得的pivot中运行k-means算法, 这个pivot可能存在重复
float run_lloyds(float *data, size_t num_points, size_t dim, float *centers, const size_t num_centers,
                 const size_t max_reps, std::vector<size_t> *closest_docs, uint32_t *closest_center)
{
    float residual = std::numeric_limits<float>::max();
    bool ret_closest_docs = true;
    bool ret_closest_center = true;
    if (closest_docs == NULL)
    {
        // 大小是num_centers的vector
        closest_docs = new std::vector<size_t>[num_centers];
        ret_closest_docs = false;
    }
    if (closest_center == NULL)
    {
        // 大小是num_points
        closest_center = new uint32_t[num_points];
        ret_closest_center = false;
    }

    // docs_l2sq保存了向量的l2 squaure
    float *docs_l2sq = new float[num_points];
    math_utils::compute_vecs_l2sq(docs_l2sq, data, num_points, dim);

    float old_residual;
    // Timer timer;
    // 迭代的最大次数
    for (size_t i = 0; i < max_reps; ++i)
    {
        old_residual = residual;

        residual = lloyds_iter(data, num_points, dim, centers, num_centers, docs_l2sq, closest_docs, closest_center);

        // 判断residual满足条件就终止
        if (((i != 0) && ((old_residual - residual) / residual) < 0.00001) ||
            (residual < std::numeric_limits<float>::epsilon()))
        {
            diskann::cout << "Residuals unchanged: " << old_residual << " becomes " << residual
                          << ". Early termination." << std::endl;
            break;
        }
    }
    delete[] docs_l2sq;
    if (!ret_closest_docs)
        delete[] closest_docs;
    if (!ret_closest_center)
        delete[] closest_center;
    return residual;
}

// assumes memory allocated for pivot_data as new
// float[num_centers*dim]
// and select randomly num_centers points as pivots
void selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers)
{
    //	pivot_data = new float[num_centers * dim];

    std::vector<size_t> picked;
    std::random_device rd;
    auto x = rd();
    std::mt19937 generator(x);
    std::uniform_int_distribution<size_t> distribution(0, num_points - 1);

    size_t tmp_pivot;
    for (size_t j = 0; j < num_centers; j++)
    {
        tmp_pivot = distribution(generator);
        if (std::find(picked.begin(), picked.end(), tmp_pivot) != picked.end())
            continue;
        picked.push_back(tmp_pivot);
        std::memcpy(pivot_data + j * dim, data + tmp_pivot * dim, dim * sizeof(float));
    }
}

/** k-means++挑选pivot的逻辑流程:
 *      1. 随机选择初始点
 *      2. 计算所有点到初始点的距离保存到dist中
 *      3. 随机选择一个阈值threshold = p * sum(dist), 找到临界点作为另一个pivot
 *      4. 计算所有点到新pivot的距离, 采用min更新, 重新回到第3步
 */
void kmeanspp_selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers)
{
    if (num_points > 1 << 23)
    {
        diskann::cout << "ERROR: n_pts " << num_points
                      << " currently not supported for k-means++, maximum is "
                         "8388608. Falling back to random pivot "
                         "selection."
                      << std::endl;
        selecting_pivots(data, num_points, dim, pivot_data, num_centers);
        return;
    }

    std::vector<size_t> picked;
    std::random_device rd;
    auto x = rd();
    std::mt19937 generator(x);
    std::uniform_real_distribution<> distribution(0, 1);
    std::uniform_int_distribution<size_t> int_dist(0, num_points - 1);
    // 随机挑选的初始变量
    size_t init_id = int_dist(generator);
    size_t num_picked = 1;

    picked.push_back(init_id);
    std::memcpy(pivot_data, data + init_id * dim, dim * sizeof(float));

    // 记录每个点的距离
    float *dist = new float[num_points];

#pragma omp parallel for schedule(static, 8192)
    // 计算train中每个点到init_d的距离
    for (int64_t i = 0; i < (int64_t)num_points; i++)
    {
        dist[i] = math_utils::calc_distance(data + i * dim, data + init_id * dim, dim);
    }

    double dart_val;
    size_t tmp_pivot;
    bool sum_flag = false;

    while (num_picked < num_centers)
    {
        dart_val = distribution(generator);

        double sum = 0;
        // sum所有点的距离
        for (size_t i = 0; i < num_points; i++)
        {
            sum = sum + dist[i];
        }
        if (sum == 0)
            sum_flag = true;

        dart_val *= sum;

        // 前缀和
        double prefix_sum = 0;
        // 这个循环的含义就是如果dart_val最后一次大于了前缀和
        // 那么这个转折点作为tmp_pivot
        for (size_t i = 0; i < (num_points); i++)
        {
            tmp_pivot = i;
            if (dart_val >= prefix_sum && dart_val < prefix_sum + dist[i])
            {
                break;
            }

            prefix_sum += dist[i];
        }

        // 判断这个点是否已经被pick: 如果已经被pick, 并且sum_flag = false
        // 但是如果这个点足以使得所有的dist都变成0, 那么选择它也无妨
        if (std::find(picked.begin(), picked.end(), tmp_pivot) != picked.end() && (sum_flag == false))
            continue;
        picked.push_back(tmp_pivot);
        // 将这个点作为pivot_data
        std::memcpy(pivot_data + num_picked * dim, data + tmp_pivot * dim, dim * sizeof(float));

#pragma omp parallel for schedule(static, 8192)
        // 并行计算所有点到tmp_pivot的距离, 并更新dist[i]为其中最小距离
        for (int64_t i = 0; i < (int64_t)num_points; i++)
        {
            dist[i] = (std::min)(dist[i], math_utils::calc_distance(data + i * dim, data + tmp_pivot * dim, dim));
        }
        num_picked++;
    }
    delete[] dist;
}

} // namespace kmeans
