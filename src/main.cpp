#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <net.h>
#include "NanoDetPlus.h"

// modifies 6-12-2022 Q-engineering

static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    //std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        //std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        //std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    return 0;
}

void draw_bboxes(const cv::Mat& image, const std::vector<BoxInfo>& bboxes, object_rect effect_roi)
{
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;


    for (size_t i = 0; i < bboxes.size(); i++){
        const BoxInfo& bbox = bboxes[i];

        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                      cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), cv::Scalar(0,0,255));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255,255,255), -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
    }
}

const int ModelSize = 416;

int main(int argc, char** argv)
{
    float f;
    float FPS[16];
    int i,Fcnt=0;
    const char* imagepath = argv[1];

    cv::Mat frame = cv::imread(imagepath, 1);
    if(frame.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    //some timing
    std::chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    NanoDet nanodet = NanoDet("nanodet-plus-m_416.param", "nanodet-plus-m_416.bin", ModelSize);

//    cv::VideoCapture cap(0);   //Movie
//    if (!cap.isOpened()) {
//        std::cerr << "ERROR: Unable to open the camera" << std::endl;
//        return 0;
//    }
//    std::cout << "Start grabbing, press ESC on TLive window to terminate" << std::endl;

    while(1){
        frame = cv::imread(imagepath, 1);
//        cap >> frame;
        if (frame.empty()) {
            std::cerr << "End of movie" << std::endl;
            break;
        }

        Tbegin = std::chrono::steady_clock::now();

        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(frame, resized_img, cv::Size(ModelSize, ModelSize), effect_roi);
        auto results = nanodet.detect(resized_img, 0.4, 0.5);
        draw_bboxes(frame, results, effect_roi);

        Tend = std::chrono::steady_clock::now();

        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        cv::putText(frame, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

        cv::imshow("RPi4 - 1.95 GHz - 2 GB ram",frame);
        //cv::imwrite("test.jpg",frame);
        char esc = cv::waitKey(5);
        if(esc == 27) break;
    }

    return 0;
}
