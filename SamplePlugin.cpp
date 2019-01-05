/*
            RoVi1
       <Final Project>
     INTEGRATION - PLUGIN
         Haris Sestan
             and
     Mathiebhan Mahendran

Used code snippets are referenced at the places they are used (in comments)
*/
#include "SamplePlugin.hpp"

#include <rws/RobWorkStudio.hpp>

#include <QPushButton>

#include <rw/loaders/ImageLoader.hpp>
#include <rw/loaders/WorkCellFactory.hpp>

#include <functional>
#include <rw/rw.hpp>
#include <Eigen/Dense>
#include <istream>


using namespace rw::common;
using namespace rw::graphics;
using namespace rw::kinematics;
using namespace rw::loaders;
using namespace rw::models;
using namespace rw::sensor;
using namespace rwlibs::opengl;
using namespace rwlibs::simulation;
using namespace rw::math;


using namespace rws;
//using namespace cv;
using namespace std::placeholders;

// ---------GLOBAL VARIABLES--------
double delta_t=0.05; // enter value in seconds
Device::Ptr device;
std::vector<cv::Point> stored_points;
int global_counter;
rw::math::Q vel_limits;
std::ofstream myfile;
int n_points;
int marker_n;


std::vector<rw::math::Transform3D<double>> slow_transforms;
std::vector<rw::math::Transform3D<double>> medium_transforms;
std::vector<rw::math::Transform3D<double>> fast_transforms;
std::vector<rw::math::Transform3D<double>> current_transforms;


// Find edges using the Canny algorithm.
// Taken from BlackBoard solution 09_edge_detection.cpp
// https://e-learn.sdu.dk/bbcswebdav/pid-5276374-dt-content-rid-9100769_2/courses/RMROVI1-U1-1-E18/vision-exercises/solutions/09_edge_detection.cpp
cv::Mat canny(const cv::Mat& blurred)
{
    int threshold1 = 90;
    int threshold2 = 230;
    cv::Mat edges;

    cv::Canny(blurred, edges, threshold1, threshold2);


    return edges;
}


// Uses a code snippet for solving a system of 2 equations using matrices
// taken from:   https://stackoverflow.com/questions/383480/intersection-of-two-lines-defined-in-rho-theta-parameterization
std::vector<cv::Point> find_intersections(std::vector<cv::Vec2f> input_lines, cv::Mat original_image) {
    std::vector<cv::Point> intersections;
    std::vector<std::vector<cv::Point>> matrix_of_intersections (input_lines.size(),std::vector<cv::Point>(input_lines.size()));
    cv::Point intersection;
    int x;
    int y;
    int width=original_image.cols;
    int height=original_image.rows;
    cv::Point empty;
    empty.x=-1;
    empty.y=-1;
    for(int i=0; i<input_lines.size(); i++)
    {
        auto& line1 = input_lines[i];
        auto r1 = line1[0];
        auto t1 = line1[1];
        float c1=cosf(t1);     //matrix element a
        float s1=sinf(t1);     //b
        for(int j=i+1; j<input_lines.size(); j++)
        {
            auto& line2 = input_lines[j];
            auto r2 = line2[0];
            auto t2 = line2[1];
            float c2=cosf(t2);     //c
            float s2=sinf(t2);     //d
            float d=c1*s2-s1*c2;        //determinative (rearranged matrix for inverse)
            if(d!=0.0f)
            {
                x=(int)((s2*r1-s1*r2)/d);
                y=(int)((-c2*r1+c1*r2)/d);
                intersection.x=x;
                intersection.y=y;
                if(x<width && x>=0 && y<height && y>=0)
                {
                    matrix_of_intersections[i][j]=intersection;
                }
                else
                {
                    matrix_of_intersections[i][j]=empty;
                }
            }
        }
    }
    for(int i=0; i<input_lines.size(); i++)
    {
        for(int j=i+1; j<input_lines.size(); j++)
        {
            auto& line1 = input_lines[i];
            auto t1 = line1[1];
            auto& line2 = input_lines[j];
            auto t2 = line2[1];
            auto angle=t1-t2;
            if(angle<0)
                angle*=-1;
            if(angle>CV_PI/2)
                angle=CV_PI-angle;
            if(angle<1.1)
                matrix_of_intersections[i][j]=empty;
        }
    }
    int counter;
    for(int k=0; k<input_lines.size(); k++)
    {
        counter=0;
        for(int i=0; i<input_lines.size(); i++)
        {
            for(int j=i+1; j<input_lines.size(); j++)
            {
                if(i==k || j==k)
                {
                    if(matrix_of_intersections[i][j]!=empty)
                        counter++;
                }
            }
        }
        if(counter<2)
        {
            for(int i=0; i<input_lines.size(); i++)
            {
                for(int j=i+1; j<input_lines.size(); j++)
                {
                    if(i==k || j==k)
                    {
                        matrix_of_intersections[i][j]=empty;
                    }
                }
            }
        }

    }
    for(int i=0; i<input_lines.size(); i++)
    {
        for(int j=i+1; j<input_lines.size(); j++)
        {
            if(matrix_of_intersections[i][j]!=empty)
            {
                intersections.push_back(matrix_of_intersections[i][j]);
            }
        }
    }
    return intersections;
}

// Apply the standard Hough transform to the input image.
// Taken from BlackBoard example solution 10_edge_linking.cpp
// https://e-learn.sdu.dk/bbcswebdav/pid-5276369-dt-content-rid-9100764_2/courses/RMROVI1-U1-1-E18/vision-exercises/solutions/10_edge_linking.cpp
std::vector<cv::Vec2f> hough_standard(const cv::Mat& edges,cv::Mat& source)
{
    int threshold = 120;
    std::vector<cv::Vec2f> chosen_lines;

    cv::Mat out;
    cv::cvtColor(edges, out, cv::COLOR_GRAY2BGR);
    cv::Mat out2;
    cv::cvtColor(source, out2, cv::COLOR_GRAY2BGR);

    std::vector<cv::Vec2f> lines;
    bool line_found;
    cv::HoughLines(edges, lines, 1, CV_PI/180, threshold);

    for (int i=0; i<lines.size(); i++) {
        auto& line = lines[i];
        auto rho = line[0];
        auto theta = line[1];
        if(rho<0)
        {
            rho*=-1;
            theta=theta-CV_PI;
            line[0]=rho;
            line[1]=theta;
        }
        line_found=false;

        if(i==0)
        {
            chosen_lines.push_back(line);
        }
        else
        {
            for (int j=0; j<chosen_lines.size(); j++)
            {
                const auto& chosen_line = chosen_lines[j];
                auto stored_rho = chosen_line[0];
                auto stored_theta = chosen_line[1];
                if(rho<stored_rho+30 && rho>stored_rho-30 && theta<stored_theta+0.31 && theta>stored_theta-0.31)
                {
                    line_found=true;
                    break;
                }
            }
            if(line_found==false)
            {
                chosen_lines.push_back(line);
            }
        }
        if(line_found==false)
        {
            auto a = std::cos(theta);
            auto b = std::sin(theta);
            auto x0 = a * rho;
            auto y0 = b * rho;

            cv::Point pt1(std::round(x0 + 3000*(-b)), std::round(y0 + 3000*a));
            cv::Point pt2(std::round(x0 - 3000*(-b)), std::round(y0 - 3000*a));
            cv::line(out, pt1, pt2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }


    for (int i=0; i<chosen_lines.size(); i++)
    {
        auto& line = chosen_lines[i];
        if(line[1]<0)
        {
            line[0]*=-1;
            line[1]+=CV_PI;
        }
    }

//    std::vector<cv::Vec2f> sorted_lines;
//    sorted_lines=chosen_lines;

//    std::sort(sorted_lines.begin(),sorted_lines.end(),[](const cv::Vec2f& left, const cv::Vec2f& right)
//    {
//        return left[1] < right[1];
//    });

    return chosen_lines;

}


std::vector<cv::Point> find_marker(std::vector<cv::Point> intersections, cv::Mat input_image)
{
    std::vector<cv::Point> chosen_intersections;
    cv::Mat copy=input_image;
    int radius=20;
    int x,y;
    int pixelValue;

    for(int i=0; i<intersections.size(); i++)
    {
        if(intersections[i].x-radius<0 || intersections[i].x+radius>input_image.cols || intersections[i].y-radius<0 || intersections[i].y+radius>input_image.rows)
        {
            continue;
        }
        int counter=0;
        bool pass=false;
        int start=input_image.at<uchar>(intersections[i].y-radius,intersections[i].x-radius);
        int end=input_image.at<uchar>(intersections[i].y-radius,intersections[i].x-radius+1);

        y=intersections[i].x-radius;

        for(x=intersections[i].y-radius; x<intersections[i].y+radius; x++)
        {
            pixelValue = input_image.at<uchar>(x,y);
            if(pixelValue==255 && pass==false)
            {
                pass=true;
                counter++;
            }
            if(pixelValue==0)
            {
                pass=false;
            }
            copy.at<uchar>(x,y)=127;
        }

        x=intersections[i].y+radius;

        for(y=intersections[i].x-radius;y<intersections[i].x+radius; y++)
        {
            pixelValue = input_image.at<uchar>(x,y);
            if(pixelValue==255 && pass==false)
            {
                pass=true;
                counter++;
            }
            if(pixelValue==0)
            {
                pass=false;
            }
            copy.at<uchar>(x,y)=127;
        }

        y=intersections[i].x+radius;

        for(x=intersections[i].y+radius; x>intersections[i].y-radius; x--)
        {
            pixelValue = input_image.at<uchar>(x,y);
            if(pixelValue==255 && pass==false)
            {
                pass=true;
                counter++;
            }
            if(pixelValue==0)
            {
                pass=false;
            }
            copy.at<uchar>(x,y)=127;
        }

        x=intersections[i].y-radius;

        for(y=intersections[i].x+radius;y>intersections[i].x-radius; y--)
        {
            pixelValue = input_image.at<uchar>(x,y);
            if(pixelValue==255 && pass==false)
            {
                pass=true;
                counter++;
            }
            if(pixelValue==0)
            {
                pass=false;
            }
            copy.at<uchar>(x,y)=127;
        }


        if(start==end && end==255)
            counter--;
        if(counter==7 || counter==8)
        {
            chosen_intersections.push_back(intersections[i]);
        }
    }

    int x1,x2,y1,y2;
    bool closeness;
    for(int i=0; i<chosen_intersections.size(); i++)
    {
        x1=chosen_intersections[i].x;
        y1=chosen_intersections[i].y;
        closeness=false;
        for(int j=0; j<chosen_intersections.size(); j++)
        {
            if(i!=j)
            {
                x2=chosen_intersections[j].x;
                y2=chosen_intersections[j].y;
                if((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)<=400)
                {
                    chosen_intersections.erase(chosen_intersections.begin()+j);
                    j--;
                    closeness=true;
                }
            }
        }
        if(closeness==true)
            i--;
    }

    return chosen_intersections;
}


cv::Mat selective_blur(cv::Mat original, cv::Mat blurred)
{
    for(int i=0; i<original.rows; i++)
    {
        for(int j=0; j<original.cols; j++)
        {
            int original_pixel = original.at<uchar>(i,j);
            int blurred_pixel = blurred.at<uchar>(i,j);
            if(original_pixel<blurred_pixel-60)
                blurred.at<uchar>(i,j)=original_pixel;
        }
    }
    return blurred;
}

std::vector<cv::Point> find_keypoints(std::vector<cv::Point> intersections)
{
    std::vector<cv::Point> extracted_intersections;
    std::vector<float> arg_values;
    std::vector<float> arg_values_sorted;
    if(intersections.size()!=6)
        return extracted_intersections;
    float xsum=0, ysum=0;
    int x,y;
    float x1,y1;
    for(int i=0; i<intersections.size(); i++)
    {
        xsum+=intersections[i].x;
        ysum+=intersections[i].y;
    }
    x=xsum/6;
    y=ysum/6;

    for(int i=0; i<intersections.size(); i++)
    {
        x1=float(intersections[i].x-x);
        y1=float(intersections[i].y-y);
        arg_values.push_back(atan2(y1,x1));
    }
    arg_values_sorted=arg_values;
    std::sort(arg_values_sorted.begin(),arg_values_sorted.end());

    float theta1, theta2,theta3;
    int theta1_index,theta2_index;

    int counter;
    for(int i=0; i<arg_values_sorted.size(); i++)
    {
        if(i!=arg_values_sorted.size()-1)
        {
            if(arg_values_sorted[i+1]-arg_values_sorted[i]<0.5)
            {
                counter=1;
                for(int j=0; j<2; j++)
                {
                    if(i==5)
                        i=0;
                    else
                        i++;
                    if(i==5){
                        if(arg_values_sorted[0]+2*CV_PI-arg_values_sorted[i]<0.5)
                            counter++;}
                    else{
                        if(arg_values_sorted[i+1]-arg_values_sorted[i]<0.5)
                            counter++;}
                }

                if(counter==2)
                {   if(i!=5)
                        theta2=arg_values_sorted[i+1];
                    else
                        theta2=arg_values_sorted[0];

                    for(int j=0; j<2; j++)
                    {
                        if(i==0)
                            i=5;
                        else
                            i--;
                    }
                    theta1=arg_values_sorted[i];
                    break;
                }
                else
                {
                    for(int j=0; j<2; j++)
                    {
                        if(i==0)
                            i=5;
                        else
                            i--;
                    }
                }

            }

        }
        else
        {
            if(arg_values_sorted[i+1]-arg_values_sorted[i]<0.5)
            {
                counter=1;
                for(int j=0; j<2; j++)
                {
                    if(i==5)
                        i=0;
                    else
                        i++;
                    if(i==5){
                        if(arg_values_sorted[0]+2*CV_PI-arg_values_sorted[i]<0.5)
                            counter++;}
                    else{
                        if(arg_values_sorted[i+1]-arg_values_sorted[i]<0.5)
                            counter++;}
                }

                if(counter==2)
                {
                    if(i!=5)
                        theta2=arg_values_sorted[i+1];
                    else
                        theta2=arg_values_sorted[0];
                    for(int j=0; j<2; j++)
                    {
                        if(i==0)
                            i=5;
                        else
                            i--;
                    }
                    theta1=arg_values_sorted[i];
                    break;
                }
                else
                {
                    for(int j=0; j<2; j++)
                    {
                        if(i==0)
                            i=5;
                        else
                            i--;
                    }
                }

            }
        }
    }
    for(int i=0; i<arg_values_sorted.size(); i++)
    {
        if(arg_values_sorted[i]==theta1)
            theta1_index=i;
    }
    for(int i=0; i<arg_values_sorted.size(); i++)
    {
        if(arg_values_sorted[i]==theta2)
            theta2_index=i;
    }
    int theta1_unsorted_index,theta2_unsorted_index,theta3_index;

    for(int i=0; i<arg_values.size(); i++)
    {
        if(arg_values[i]==theta2)
            theta2_unsorted_index=i;
    }
    for(int i=0; i<arg_values.size(); i++)
    {
        if(arg_values[i]==theta1)
            theta1_unsorted_index=i;
    }
    extracted_intersections.push_back(intersections[theta1_unsorted_index]);
    extracted_intersections.push_back(intersections[theta2_unsorted_index]);

    for(int i=0; i<arg_values_sorted.size(); i++)
    {
        if(arg_values_sorted[i]==theta2 && i!=arg_values_sorted.size()-1)
            theta3=arg_values_sorted[i+1];
        else if(arg_values_sorted[i]==theta2 && i==arg_values_sorted.size()-1)
            theta3=arg_values_sorted[0];
    }
    for(int i=0; i<arg_values.size(); i++)
    {
        if(arg_values[i]==theta3)
            theta3_index=i;

    }
    extracted_intersections.push_back(intersections[theta3_index]);
    return extracted_intersections;
}

std::vector<cv::Point> extract_marker_2(cv::Mat src)
{
//    cv::Mat src2;
    cv::cvtColor(src, src, CV_BGR2GRAY);
 //   cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

    cv::Mat blurred;
    cv::blur(src, blurred, cv::Size(61, 61));

    blurred=selective_blur(src,blurred);

    cv::Mat canny_edges;
    canny_edges=canny(blurred);

    std::vector<cv::Vec2f> lines;
    lines=hough_standard(canny_edges,blurred);
    std::vector<cv::Point> line_intersections;
    line_intersections=find_intersections(lines,blurred);
    cv::threshold(blurred,blurred,100,255,cv::THRESH_BINARY);

    //cv::Mat kernel = cv::Mat::ones(9, 9, CV_8U);
    //cv::dilate(canny_edges, canny_edges, kernel);


    std::vector<cv::Point> chosen_intersections;
    if(line_intersections.size()>6)
        chosen_intersections=find_marker(line_intersections, canny_edges);
    else
        chosen_intersections=line_intersections;

    chosen_intersections=find_keypoints(chosen_intersections);
    return chosen_intersections;


}

// Thresholding in HSV space
// Taken from BlackBoard solution example 08_color_segmentation.cpp
// https://e-learn.sdu.dk/bbcswebdav/pid-5276375-dt-content-rid-9100770_2/courses/RMROVI1-U1-1-E18/vision-exercises/solutions/08_color_segmentation.cpp
cv::Mat color_segmentation_hsv(const cv::Mat hsv, int h_slider_min, int h_slider_max)
{
    int s_slider_min = 0;
    int s_slider_max = 255;
    int v_slider_min = 0;
    int v_slider_max = 255;
    cv::Mat dst;
    cv::inRange(hsv,
                cv::Scalar(h_slider_min, s_slider_min, v_slider_min),
                cv::Scalar(h_slider_max, s_slider_max, v_slider_max),
                dst);

    return dst;
}

std::vector<cv::Point> extract_marker_1 (cv::Mat img)
{
    // Setup SimpleBlobDetector parameters.
    cv::SimpleBlobDetector::Params params;

    params.filterByCircularity = true;
    params.minCircularity = 0.3;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.3;

    params.filterByColor = true;
    params.blobColor = 255;

    params.filterByArea = true; // filter by area of the blob
    params.minArea = 200 ;// Minimum area of the blob
    params.maxArea = 100000; // Maximum area of the blob

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::Mat kernel = cv::Mat::ones(30, 30, CV_8U);

    cv::Mat seg_red = color_segmentation_hsv(hsv,0,15);
    cv::Mat seg_blue = color_segmentation_hsv(hsv,100,131);
    cv::Mat seg_green = color_segmentation_hsv(hsv,35,90);

    cv::erode(seg_green, seg_green, kernel);
    kernel = cv::Mat::ones(130, 130, CV_8U);
    cv::dilate(seg_green, seg_green, kernel);
    cv::blur(seg_green,seg_green,cv::Size(200,200));

    std::vector<cv::KeyPoint> keypoints1;
    detector->detect( seg_red, keypoints1);

    std::vector<cv::KeyPoint> keypoints2;
    detector->detect( seg_blue, keypoints2);

    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.minArea = 500 ;// Minimum area of the blob
    params.maxArea = 1000000; // Maximum area of the blob
    params.filterByColor = true;
    params.blobColor = 255;

    cv::Ptr<cv::SimpleBlobDetector> detector1 = cv::SimpleBlobDetector::create(params);

    std::vector<cv::KeyPoint> keypoints3;
    //cv::threshold(seg_green,seg_green,254,255,cv::THRESH_BINARY);
    detector1->detect( seg_green, keypoints3);

    int red_count,blue_count,x,y,x1,y1;
    float dist,s;
    bool marker_found=false;
    std::vector<cv::KeyPoint> marker_keypoints;

    for(int i=0; i<keypoints3.size(); i++)
    {
        red_count=0;
        blue_count=0;
        x=keypoints3[i].pt.x;
        y=keypoints3[i].pt.y;
        s=keypoints3[i].size+20;
        for(int j=0; j<keypoints1.size(); j++)
        {
            x1=keypoints1[j].pt.x-x;
            y1=keypoints1[j].pt.y-y;
            dist=std::sqrt(float(x1*x1)+float(y1*y1));
            if(dist<s/2)
            {
                red_count++;
                marker_keypoints.push_back(keypoints1[j]);
            }
        }

        for(int k=0; k<keypoints2.size(); k++)
        {
            x1=keypoints2[k].pt.x-x;
            y1=keypoints2[k].pt.y-y;
            dist=std::sqrt(float(x1*x1)+float(y1*y1));
            if(dist<s/2)
            {
                blue_count++;
                marker_keypoints.push_back(keypoints2[k]);
            }
        }

        if(red_count==1 && blue_count==3)
        {
            marker_found=true;
            break;
        }
        else
        {
            marker_found=false;
            marker_keypoints.clear();
        }
    }

    std::vector<float> arg_values;
    std::vector<float> arg_values_sorted;
    std::vector<cv::Point> chosen_points;

    if(marker_found==true)
    {
        for(int i=0; i<marker_keypoints.size(); i++)
        {
            x1=float(marker_keypoints[i].pt.x-x);
            y1=float(marker_keypoints[i].pt.y-y);
            arg_values.push_back(atan2(y1,x1));
        }

        arg_values_sorted=arg_values;
        std::sort(arg_values_sorted.begin(),arg_values_sorted.end());

        float left, right;
        for(int i=0; i<arg_values_sorted.size(); i++)
        {
            if(arg_values_sorted[i]==arg_values[0])
            {
                if(i==0)
                {
                    left=arg_values_sorted[3];
                    right=arg_values_sorted[1];
                }
                else if(i==3)
                {
                    left=arg_values_sorted[2];
                    right=arg_values_sorted[0];
                }
                else
                {
                    left=arg_values_sorted[i-1];
                    right=arg_values_sorted[i+1];
                }
            }
        }

        chosen_points.push_back(marker_keypoints[0].pt);

        for(int i=0; i<arg_values.size(); i++)
        {
            if(left==arg_values[i])
                chosen_points.push_back(marker_keypoints[i].pt);
        }

        for(int i=0; i<arg_values.size(); i++)
        {
            if(right==arg_values[i])
                chosen_points.push_back(marker_keypoints[i].pt);
        }

    }


    return chosen_points;
}






SamplePlugin::SamplePlugin():
    RobWorkStudioPlugin("SamplePluginUI", QIcon(":/pa_icon.png"))
{
    setupUi(this);

    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(timer()));

    // now connect stuff from the ui component
    connect(_btn0    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    connect(_btn1    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    connect(_btn2    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    connect(_btn_slow    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    connect(_btn_medium    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    connect(_btn_fast    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    connect(_slider    ,SIGNAL(sliderReleased()), this, SLOT(btnPressed()) );
    connect(_btn_1p    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    connect(_btn_2p   ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    connect(_btn_3p    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );


    connect(_spinBox  ,SIGNAL(valueChanged(int)), this, SLOT(btnPressed()) );

    Image textureImage(300,300,Image::GRAY,Image::Depth8U);
    _textureRender = new RenderImage(textureImage);
    Image bgImage(0,0,Image::GRAY,Image::Depth8U);
    _bgRender = new RenderImage(bgImage,2.5/1000.0);
    _framegrabber = NULL;
}

SamplePlugin::~SamplePlugin()
{
    delete _textureRender;
    delete _bgRender;
}

std::vector<rw::math::Transform3D<double>> marker_transforms(int motion)
{
    std::vector<rw::math::Transform3D<double>> output_transforms;
    std::ifstream file;
    if(motion==1)
        file.open( "/home/student/Downloads/FinalProject/SamplePluginPA10/motions/MarkerMotionSlow.txt" );
    else if(motion==2)
        file.open( "/home/student/Downloads/FinalProject/SamplePluginPA10/motions/MarkerMotionMedium.txt" );
    else if(motion==3)
        file.open( "/home/student/Downloads/FinalProject/SamplePluginPA10/motions/MarkerMotionFast.txt" );


    std::vector<std::string> lines;
    std::string s;
    while(std::getline(file,s))
        lines.push_back(s);
    std::vector<std::string>::iterator i;
    for(i=lines.begin();i<lines.end(); i++)
    {
        std::istringstream s2(*i);
        double x,y,z,R,P,Y;
        s2>>x>>y>>z>>R>>P>>Y;
        rw::math::Vector3D<double> p(x,y,z);
        rw::math::RPY<double> rpy(R,P,Y);
        rw::math::Rotation3D<double> rot=rpy.toRotation3D();
        rw::math::Transform3D<double> T (p,rot);
        output_transforms.push_back(T);
        std::cout<< T<<" \n";

    }

    file.close();
    return output_transforms;

}

void SamplePlugin::initialize() {
    log().info() << "INITIALIZE" << "\n";

    getRobWorkStudio()->stateChangedEvent().add(std::bind(&SamplePlugin::stateChangedListener, this, std::placeholders::_1), this);

    // Auto load workcell
    WorkCell::Ptr wc = WorkCellLoader::Factory::load("/home/student/Downloads/FinalProject/PA10WorkCell/ScenePA10RoVi1.wc.xml");
    getRobWorkStudio()->setWorkCell(wc);
    device = wc->findDevice("PA10");

    vel_limits=device->getVelocityLimits();
    //log().info() <<"vel limits: "<< vel_limits<<" \n";
    //log().info() <<"joint limits: "<< device->getBounds().first<<" and: "<< device->getBounds().second<<" \n";

    n_points=1;
    log().info() << "Tracking 1 point.\n";

    slow_transforms=marker_transforms(1);
    medium_transforms=marker_transforms(2);
    fast_transforms=marker_transforms(3);
    current_transforms=slow_transforms;
    log().info() << "Using sequence MarkerMotionSlow.txt.\n";


    Image::Ptr image;
    image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/markers/Marker1.ppm");
    _textureRender->setImage(*image);
    getRobWorkStudio()->updateAndRepaint();
    marker_n=1;
    log().info() << "Marker 1 loaded.\n";


    // Load Lena image
    //    cv::Mat im, image2;
    //    im = cv::imread("/home/student/Downloads/FinalProject/SamplePluginPA10/src/lena.bmp", CV_LOAD_IMAGE_COLOR); // Read the file
    //    cv::cvtColor(im, image2, CV_BGR2RGB); // Switch the red and blue color channels
    //    if(! image2.data ) {
    //        RW_THROW("Could not open or find the image: please modify the file path in the source code!");
    //    }
    //    QImage img(image2.data, image2.cols, image2.rows, image2.step, QImage::Format_RGB888); // Create QImage from the OpenCV image
    //    _label->setPixmap(QPixmap::fromImage(img)); // Show the image at the label in the plugin
}

void SamplePlugin::open(WorkCell* workcell)
{
    log().info() << "OPEN" << "\n";
    _wc = workcell;
    _state = _wc->getDefaultState();

    log().info() << workcell->getFilename() << "\n";

    if (_wc != NULL) {
        // Add the texture render to this workcell if there is a frame for texture
        Frame* textureFrame = _wc->findFrame("MarkerTexture");
        if (textureFrame != NULL) {
            getRobWorkStudio()->getWorkCellScene()->addRender("TextureImage",_textureRender,textureFrame);
        }
        // Add the background render to this workcell if there is a frame for texture
        Frame* bgFrame = _wc->findFrame("Background");
        if (bgFrame != NULL) {
            getRobWorkStudio()->getWorkCellScene()->addRender("BackgroundImage",_bgRender,bgFrame);
        }

        // Create a GLFrameGrabber if there is a camera frame with a Camera property set
        Frame* cameraFrame = _wc->findFrame("CameraSim");
        if (cameraFrame != NULL) {
            if (cameraFrame->getPropertyMap().has("Camera")) {
                // Read the dimensions and field of view
                double fovy;
                int width,height;
                std::string camParam = cameraFrame->getPropertyMap().get<std::string>("Camera");
                std::istringstream iss (camParam, std::istringstream::in);
                iss >> fovy >> width >> height;
                // Create a frame grabber
                _framegrabber = new GLFrameGrabber(width,height,fovy);
                SceneViewer::Ptr gldrawer = getRobWorkStudio()->getView()->getSceneViewer();
                _framegrabber->init(gldrawer);
            }
        }
    }
}


void SamplePlugin::close() {
    log().info() << "CLOSE" << "\n";

    // Stop the timer
    _timer->stop();
    // Remove the texture render
    Frame* textureFrame = _wc->findFrame("MarkerTexture");
    if (textureFrame != NULL) {
        getRobWorkStudio()->getWorkCellScene()->removeDrawable("TextureImage",textureFrame);
    }
    // Remove the background render
    Frame* bgFrame = _wc->findFrame("Background");
    if (bgFrame != NULL) {
        getRobWorkStudio()->getWorkCellScene()->removeDrawable("BackgroundImage",bgFrame);
    }
    // Delete the old framegrabber
    if (_framegrabber != NULL) {
        delete _framegrabber;
    }
    _framegrabber = NULL;
    _wc = NULL;
}

cv::Mat SamplePlugin::toOpenCVImage(const Image& img) {
    cv::Mat res(img.getHeight(),img.getWidth(), CV_8UC3);
    res.data = (uchar*)img.getImageData();
    cv::cvtColor(res,res, cv::COLOR_RGB2BGR);
    return res;
}


void SamplePlugin::btnPressed() {
    QObject *obj = sender();
    if(obj==_btn0)
    {
        marker_n=1;
        _timer->stop();
        log().info() << "Marker 1 loaded.\n";
        // Set a new texture (one pixel = 1 mm)
        Image::Ptr image;
        image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/markers/Marker1.ppm");
        _textureRender->setImage(*image);
        getRobWorkStudio()->updateAndRepaint();
    }

    else if(obj==_btn2)
    {
        marker_n=2;
        _timer->stop();
        log().info() << "Marker 2a loaded.\n";
        // Set a new texture (one pixel = 1 mm)
        Image::Ptr image;
        image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/markers/Marker2a.ppm");
        _textureRender->setImage(*image);
        getRobWorkStudio()->updateAndRepaint();
    }

    else if(obj==_btn_1p)
    {
        _timer->stop();
        n_points=1;
        log().info() << "Tracking 1 point.\n";
    }
    else if(obj==_btn_2p)
    {
        _timer->stop();
        n_points=2;
        log().info() << "Tracking 2 points.\n";
    }
    else if(obj==_btn_3p)
    {
        _timer->stop();
        n_points=3;
        log().info() << "Tracking 3 points.\n";
    }



    else if(obj==_btn_slow)
    {
        log().info() << "Using sequence MarkerMotionSlow.txt.\n";
        _timer->stop();
        current_transforms=slow_transforms;
    }
    else if(obj==_btn_medium)
    {
        log().info() << "Using sequence MarkerMotionMedium.txt.\n";
        _timer->stop();
        current_transforms=medium_transforms;
    }
    else if(obj==_btn_fast)
    {
        log().info() << "Using sequence MarkerMotionFast.txt.\n";
        _timer->stop();
        current_transforms=fast_transforms;
    }
    else if(obj==_slider)
    {
        _timer->stop();
        delta_t=double(_slider->value())/20;
        log().info() << "Timer delta_t is set to "<<delta_t<<" seconds.\n";
    }


    else if(obj==_btn1)
    {

        stored_points.clear();
        rw::math::Q q(7,0,-0.65,0,1.8,0.0,0.42,0);
        device->setQ(q,_state);
        getRobWorkStudio()->setState(_state);

        MovableFrame* marker = _wc->findFrame<MovableFrame>("Marker");

        marker->setTransform(current_transforms[0], _state);
        getRobWorkStudio()->setState(_state);

        // Toggle the timer on and off
        if (!_timer->isActive())
        {
            myfile.open ("t-1-joint-tool.txt");
            global_counter=0;

            device = _wc->findDevice("PA10");

            _timer->start(delta_t*1000); // run 10 Hz
        }
        else
        {   myfile.close();
            stored_points.clear();
            _timer->stop();
        }
    }

    else if(obj==_spinBox){
        Frame* bgFrame = _wc->findFrame("Background");
        Image::Ptr image;
        if(_spinBox->value()==0)
        {

            if (bgFrame != NULL) {
                getRobWorkStudio()->getWorkCellScene()->setVisible(false,bgFrame);
            }
        }
        else if(_spinBox->value()==1)
        {

            log().info() << "Background color1.ppm loaded.\n";
            // Set a new texture (one pixel = 1 mm)
            image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/backgrounds/color1.ppm");
            _bgRender->setImage(*image);
            if (bgFrame != NULL) {
                getRobWorkStudio()->getWorkCellScene()->setVisible(true,bgFrame);
            }
        }
        else if(_spinBox->value()==2)
        {
            log().info() << "Background color2.ppm loaded.\n";
            // Set a new texture (one pixel = 1 mm)
            image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/backgrounds/color2.ppm");
            _bgRender->setImage(*image);
            if (bgFrame != NULL) {
                getRobWorkStudio()->getWorkCellScene()->setVisible(true,bgFrame);
            }
        }
        else if(_spinBox->value()==3)
        {
            log().info() << "Background color3.ppm loaded.\n";
            // Set a new texture (one pixel = 1 mm)
            image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/backgrounds/color3.ppm");
            _bgRender->setImage(*image);
            if (bgFrame != NULL) {
                getRobWorkStudio()->getWorkCellScene()->setVisible(true,bgFrame);
            }
        }
        else if(_spinBox->value()==4)
        {
            log().info() << "Background lines1.ppm loaded.\n";
            // Set a new texture (one pixel = 1 mm)
            image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/backgrounds/lines1.ppm");
            _bgRender->setImage(*image);
            if (bgFrame != NULL) {
                getRobWorkStudio()->getWorkCellScene()->setVisible(true,bgFrame);
            }
        }
        else if(_spinBox->value()==5)
        {
            log().info() << "Background texture1.ppm loaded.\n";
            // Set a new texture (one pixel = 1 mm)
            image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/backgrounds/texture1.ppm");
            _bgRender->setImage(*image);
            if (bgFrame != NULL) {
                getRobWorkStudio()->getWorkCellScene()->setVisible(true,bgFrame);
            }
        }
        else if(_spinBox->value()==6)
        {
            log().info() << "Background texture2.ppm loaded.\n";
            // Set a new texture (one pixel = 1 mm)
            image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/backgrounds/texture2.ppm");
            _bgRender->setImage(*image);
            if (bgFrame != NULL) {
                getRobWorkStudio()->getWorkCellScene()->setVisible(true,bgFrame);
            }
        }
        else if(_spinBox->value()==7)
        {
            log().info() << "Background texture3.ppm loaded.\n";
            // Set a new texture (one pixel = 1 mm)
            image = ImageLoader::Factory::load("/home/student/Downloads/FinalProject/SamplePluginPA10/backgrounds/texture3.ppm");
            _bgRender->setImage(*image);
            if (bgFrame != NULL) {
                getRobWorkStudio()->getWorkCellScene()->setVisible(true,bgFrame);
            }
        }
        else
            log().info() << "Invalid input. \n";
        getRobWorkStudio()->updateAndRepaint();

    }
}

rw::math::LinearAlgebra::EigenMatrix<double>::type jacobian_to_eigen(rw::math::Jacobian J)
{
    rw::math::LinearAlgebra::EigenMatrix<double>::type output_matrix(J.size1(), J.size2());
    for(int i=0; i<J.size1(); i++)
    {
        for(int j=0; j<J.size2(); j++)
        {
            output_matrix(i,j)=J.elem(i,j);
        }
    }
    return output_matrix;
}

rw::math::Jacobian eigen_to_jacobian(rw::math::LinearAlgebra::EigenMatrix<double>::type m)
{
    rw::math::Jacobian output_jacobian(m.rows(), m.cols());
    for(int i=0; i<output_jacobian.size1(); i++)
    {
        for(int j=0; j<output_jacobian.size2(); j++)
        {
            output_jacobian.elem(i,j)=m(i,j);
        }
    }
    return output_jacobian;
}




rw::math::Jacobian transpose_jacobian(rw::math::Jacobian J)
{
    rw::math::Jacobian output_jacobian(J.size2(),J.size1());
    for(int i=0; i<output_jacobian.size1(); i++)
    {
        for(int j=0; j<output_jacobian.size2(); j++)
        {
            output_jacobian.elem(i,j)=J.elem(j,i);
        }
    }
    return output_jacobian;
}





rw::math::Jacobian calculate_Zimage(std::vector<cv::Point> points, Device::Ptr device, State &state)
{
    double f=823;
    double u,v;
    double z=0.5;
    rw::math::Jacobian Jimage(2*points.size(),6);

    for(int i=0; i<points.size(); i++)
    {

        u=points[i].x-double(1024/2);
        v=points[i].y-double(768/2);


        Jimage.elem(i*2,0)=-f/z;
        Jimage.elem(i*2,1)=0;
        Jimage.elem(i*2,2)=u/z;
        Jimage.elem(i*2,3)=u*v/f;
        Jimage.elem(i*2,4)=-(f*f+u*u)/f;
        Jimage.elem(i*2,5)=v;
        Jimage.elem(i*2+1,0)=0;
        Jimage.elem(i*2+1,1)=-f/z;
        Jimage.elem(i*2+1,2)=v/z;
        Jimage.elem(i*2+1,3)=(f*f+v*v)/f;
        Jimage.elem(i*2+1,4)=-u*v/f;
        Jimage.elem(i*2+1,5)=-u;
    }

    rw::math::Jacobian S=rw::math::Jacobian::zero(6,6);
    std::cout<<S<<"\n";

    rw::math::Rotation3D<>R_base_cam=device->baseTend(state).R();
    std::cout<<R_base_cam<<"\n";

    rw::math::Rotation3D<>R_base_cam_T=R_base_cam.inverse();

    std::cout<<R_base_cam_T<<"\n";

    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            S.elem(i,j)=R_base_cam_T(i,j);
            S.elem(i+3,j+3)=R_base_cam_T(i,j);
        }

    }
    std::cout<<S<<"\n";


    rw::math::Jacobian J=device->baseJend(state);
    std::cout<<J<<"\n";

    rw::math::Jacobian Zimage=Jimage*S*J;
    std::cout<<Zimage<<"\n";


    return Zimage;

}

void SamplePlugin::timer() {
    if (_framegrabber != NULL) {
        if(global_counter==current_transforms.size())
        {
            myfile.close();
            _timer->stop();
            return;
        }
        rw::math::Transform3D<double> current_transform=current_transforms[global_counter];
        // Get the image as a RW image

//        std::vector<rw::math::Vector3D<>> p_vec;
//        rw::math::Vector3D<> p_1(0.15,0.15,0), p_2(-0.15,0.15,0), p_3(0.15,-0.15,0);
//        p_vec.push_back(p_1);
//        p_vec.push_back(p_2);
//        p_vec.push_back(p_3);


        Frame* cam = _wc->findFrame("Camera");
        MovableFrame* marker = _wc->findFrame<MovableFrame>("Marker");

        //CASTAJ MARKER BLABLA
        Frame* cameraFrame = _wc->findFrame("CameraSim");
        _framegrabber->grab(cameraFrame, _state);
        Image& image = _framegrabber->getImage();

        marker->setTransform(current_transform, _state);
        getRobWorkStudio()->setState(_state);



        // Convert to OpenCV image
        cv::Mat im = toOpenCVImage(image);
        cv::Mat imflip;
        cv::flip(im, imflip, 0);
        int t1=_timer->remainingTime();
        std::vector<cv::Point> feature_points, feature_points_reduced;
        if(marker_n==1)
        {
            feature_points=extract_marker_1(imflip);

            for(int i=0; i<n_points; i++)
                feature_points_reduced.push_back(feature_points[i]);
            for(int i=0;i<feature_points_reduced.size();i++){
                cv::Scalar color;
                if(i==0) color=cv::Scalar(0,255,255);
                if(i==1) color=cv::Scalar(203,192,255);
                if(i==2) color=cv::Scalar(0,128,255);

                cv::circle(imflip,feature_points_reduced[i],40,color,5);


            }
            if(global_counter==0) stored_points=feature_points_reduced;
        }
        else if(marker_n==2)
        {
            feature_points=extract_marker_2(imflip);
            for(int i=0; i<n_points; i++)
                feature_points_reduced.push_back(feature_points[i]);
            for(int i=0;i<feature_points_reduced.size();i++){
                cv::Scalar color;
                if(i==0) color=cv::Scalar(255,0,0);
                if(i==1) color=cv::Scalar(0,255,0);
                if(i==2) color=cv::Scalar(0,0,255);

                cv::circle(imflip,feature_points_reduced[i],40,color,5);

            }
            if(global_counter==0) stored_points=feature_points_reduced;
        }
        cv::cvtColor(imflip,imflip, cv::COLOR_BGR2RGB);

        int t2=_timer->remainingTime();
        double t_dif=double(t1-t2)/1000;
        myfile<<t_dif<<"\n";
         //Show in QLabel
        QImage img(imflip.data, imflip.cols, imflip.rows, imflip.step, QImage::Format_RGB888);
        QPixmap p = QPixmap::fromImage(img);
        unsigned int maxW = 400;
        unsigned int maxH = 800;
        _label->setPixmap(p.scaled(maxW,maxH,Qt::KeepAspectRatio));

        double f=823;
        double x,y,u,v;
        double z=0.5;

        if(feature_points_reduced.empty()==true)
        {
            global_counter++;
            return;
        }
//        rw::math::Transform3D<double> T_base_cam=device->baseTframe(cam,_state);
//        rw::math::Transform3D<double> T_base_marker=device->baseTframe(marker,_state);
//        rw::math::Transform3D<double>::invMult(T_base_cam,T_base_marker);
//        rw::math::Transform3D<double> T_cam_marker=T_base_cam;

//        std::vector<rw::math::Vector3D<>> p_cam_vector;
//        for(int i=0; i<p_vec.size(); i++)
//            p_cam_vector.push_back(T_cam_marker*p_vec[i]);

        rw::math::Jacobian d_uv (2*n_points,1);

//        std::vector<double> p_u,p_v;
//        for(int i=0; i<n_points; i++)
//        {
//            x=p_cam_vector[i][0];
//            y=p_cam_vector[i][1];

//            u=f*x/z;
//            v=f*y/z;
//            p_u.push_back(u);
//            p_v.push_back(v);
//        }

        for(int i=0; i<n_points; i++)
        {
            u=feature_points_reduced[i].x-double(1024/2);
            v=feature_points_reduced[i].y-double(768/2);
            d_uv(2*i,0)=(stored_points[i].x-double(1024/2)-u);
            d_uv(2*i+1,0)=(stored_points[i].y-double(768/2)-v);
            //myfile<<d_uv(2*i,0)<<" "<< d_uv(2*i+1,0)<<" ";

        }
        //myfile<<"\n";


        rw::math::Jacobian Zimage=calculate_Zimage(feature_points_reduced,device,_state);
        rw::math::Jacobian Zimage_T=transpose_jacobian(Zimage);
        rw::math::Jacobian Zproduct=Zimage*Zimage_T;

        rw::math::LinearAlgebra::EigenMatrix<double>::type Zproduct_m=jacobian_to_eigen(Zproduct);
        rw::math::LinearAlgebra::EigenMatrix<double>::type Zproduct_m_inv=Zproduct_m.inverse();
        rw::math::Jacobian Zproduct_inv=eigen_to_jacobian(Zproduct_m_inv);
        rw::math::Jacobian y_m=Zproduct_inv*d_uv;
        rw::math::Jacobian dq=Zimage_T*y_m;

        rw::math::Q dq_Q(7);
        dq_Q(0)=dq(0,0);
        dq_Q(1)=dq(1,0);
        dq_Q(2)=dq(2,0);
        dq_Q(3)=dq(3,0);
        dq_Q(4)=dq(4,0);
        dq_Q(5)=dq(5,0);
        dq_Q(6)=dq(6,0);
        for(int i=0; i<7; i++)
        {
            if(dq_Q(i)/(delta_t-t_dif)>vel_limits(i))
            {
                double Vdif=(dq_Q(i)/(delta_t-t_dif)) - vel_limits(i);
                double dq_dif=Vdif*(delta_t-t_dif);
                dq_Q(i)=dq_Q(i)-dq_dif;

            }
            //myfile<<dq_Q(i)/(delta_t-t_dif)<<" ";

        }
        //myfile<<"\n";
        rw::math::Q current=device->getQ(_state);
        device->setQ(current+dq_Q,_state);
        //myfile<<(current+dq_Q)(0)<<" "<<(current+dq_Q)(1)<<" "<<(current+dq_Q)(2)<<" "<<(current+dq_Q)(3)<<" "<<(current+dq_Q)(4)<<" "<<(current+dq_Q)(5)<<" "<<(current+dq_Q)(6)<<"\n ";
        getRobWorkStudio()->setState(_state);
        //log().info() << dq_Q<<" \n";
        //        log().info() << "Stored point 0 (u,v) is: "<<stored_points[0][0]<<","<<stored_points[0][1]<<" \n";
        //        log().info() << "Stored point 1 (u,v) is: "<<stored_points[1][0]<<","<<stored_points[1][1]<<" \n";
        //        log().info() << "Stored point 2 (u,v) is: "<<stored_points[2][0]<<","<<stored_points[2][1]<<" \n";


        global_counter++;

//        Frame* tool = _wc->findFrame("Tool");
//        rw::math::Transform3D<double> T_base_tool=tool->wTf(_state);
//        rw::math::Pose6D<double> pose (T_base_tool);
//        myfile<< (dq_Q+current)(0)<<" "<<(dq_Q+current)(1)<<" "<<(dq_Q+current)(2)<<" "<<(dq_Q+current)(3)<<" "<<(dq_Q+current)(4)<<" "<<(dq_Q+current)(5)<<" "<<(dq_Q+current)(6)<<" "<<pose(0)<<" "<<pose(1)<<" "<<pose(2)<<" "<<pose(3)<<" "<<pose(4)<<" "<<pose(5)<<"\n";

    }
}

void SamplePlugin::stateChangedListener(const State& state) {
    _state = state;
}
