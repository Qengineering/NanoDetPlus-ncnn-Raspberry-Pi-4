#ifndef NANODETPLUS_H
#define NANODETPLUS_H

//
// Create by RangiLyu
// 2020 / 10 / 2
//
// modifies 6-12-2022 Q-engineering

#include <opencv2/core/core.hpp>
#include <net.h>

struct CenterPrior
{
    int x;
    int y;
    int stride;
};

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class NanoDet
{
public:
    NanoDet(const char* param, const char* bin, int Size);
    ~NanoDet();

    static NanoDet* detector;
    ncnn::Net* Net;
    static bool hasGPU;
    int Fsize;
    // modify these parameters to the same with your config if you want to use your own model
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);
private:
    void decode_infer(ncnn::Mat& feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
};
#endif // NANODETPLUS_H
