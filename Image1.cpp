/*
             RoVi1
       Vision MiniProject

Haris Sestan and Mathiebhan Mahendran

   Code for filtering Image1.png
*/

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

// Calculates and returns a 1D histogram of 'img' with 256 bins. (Taken from the exercise solution files on BlackBoard)
cv::Mat histogram(const cv::Mat& img)
{
assert(img.type() == CV_8UC1);

cv::Mat hist;
cv::calcHist(
            std::vector<cv::Mat>{img},
{0}, // channels
            cv::noArray(), // mask
            hist, // output histogram
{256}, // histogram sizes (number of bins) in each dimension
{0, 256} // pairs of bin lower (incl.) and upper (excl.) boundaries in each dimension
            );
return hist; // returned type is 32FC1
}


// Draws the 1D histogram 'hist' and returns the image. (Taken from the exercise solution files on BlackBoard)
cv::Mat draw_histogram_image(const cv::Mat& hist)
{
int nbins = hist.rows;
double max = 0;
cv::minMaxLoc(hist, nullptr, &max);
cv::Mat img(nbins, nbins, CV_8UC1, cv::Scalar(255));

for (int i = 0; i < nbins; i++) {
    double h = nbins * (hist.at<float>(i) / max); // Normalize
    cv::line(img, cv::Point(i, nbins), cv::Point(i, nbins - h), cv::Scalar::all(0));
}

return img;
}


// Rearranges the quadrants of a Fourier image so that the origin is at the
// center of the image. (Taken from the exercise solution files on BlackBoard)
void dftshift(cv::Mat& mag)
{
int cx = mag.cols / 2;
int cy = mag.rows / 2;

cv::Mat tmp;
cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

q0.copyTo(tmp);
q3.copyTo(q0);
tmp.copyTo(q3);

q1.copyTo(tmp);
q2.copyTo(q1);
tmp.copyTo(q2);
}

// Filters by sorting pixels in a 3x3 neighborhood, ignoring pepper noise pixels
// and finding the median of the remaining pixels
// (Modified version of the exercise solution files on BlackBoard)
cv::Mat custom_filter(const cv::Mat& img)
{
// Pad original with border
cv::Mat padded;
cv::copyMakeBorder(img, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);

// Output image
cv::Mat out(img.size(), img.type());

for (int i = 1; i < padded.rows - 1; i++) {
    // Pointers to previous, current and next rows
    uchar* r[] = {
        padded.ptr<uchar>(i - 1),
        padded.ptr<uchar>(i),
        padded.ptr<uchar>(i + 1)
    };

    for (int j = 1; j < padded.cols - 1; j++) {
        // Column vector of neighboring pixels
        cv::Vec<uchar, 9> v;
        v << r[0][j-1], r[0][j], r[0][j+1],
                r[1][j-1], r[1][j], r[1][j+1],
                r[2][j-1], r[2][j], r[2][j+1];

        cv::sort(v, v, (cv::SORT_EVERY_COLUMN | cv::SORT_ASCENDING));
        // Ignore the 0's (pepper) and find the median of the remaining pixels
        // For even amount of pixels, the median is chosen by rounding down
        for(int k =0; k<9;k++)
        {
            if(v(k)!=0){
                v(8)=v((k+8)/2); // Set v(8) as the median of non-zero pixels
                break;
            }
        }

        // Assign new pixel value - will be either 0 or median determined in the previous loop
        out.at<uchar>(i - 1, j - 1) = v(8);

    }
}

return out;
}

// Transforms src, both brightness and contrast can be changed
// (Modified version of the exercise solution files on BlackBoard so that it includes
// both contrast and brightness transformation)
cv::Mat transform_intensity(const cv::Mat& src, int brightness, float contrast)
{
cv::Mat img = src.clone();

for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
        int val = img.at<uchar>(i, j);
        img.at<uchar>(i, j) = cv::saturate_cast<uchar>(contrast*val+brightness);
    }
}

return img;
}


// Spectrum visualisation taken from the exercise solution files on BlackBoard
int main(int argc, char* argv[])
{
cv::CommandLineParser parser(argc, argv,
                             "{help   |            | print this message}"
                             "{@image | ./Image1.png | image path}"
                             );

if (parser.has("help")) {
    parser.printMessage();
    return 0;
}

// Load Image1.png as grayscale
std::string filepath = parser.get<std::string>("@image");
cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);

if (img.empty()) {
    std::cout << "Input image not found at '" << filepath << "'\n";
    return 1;
}

cv::imshow("Image 1: Input image", img);

cv::Mat hist = histogram(img);
cv::imshow("Histogram of input image", draw_histogram_image(hist));

// Expand the image to an optimal size
//<<border>>
cv::Mat padded;
int opt_rows = cv::getOptimalDFTSize(img.rows * 2 - 1);
int opt_cols = cv::getOptimalDFTSize(img.cols * 2 - 1);
cv::copyMakeBorder(img,
                   padded,
                   0,
                   opt_rows - img.rows,
                   0,
                   opt_cols - img.cols,
                   cv::BORDER_CONSTANT,
                   cv::Scalar::all(0)
                   );
//<<->>

// Make place for both the real and complex values by merging planes into a
// cv::Mat with 2 channels.
// Use float element type because frequency domain ranges are large.
cv::Mat planes[] = {
    cv::Mat_<float>(padded),
    cv::Mat_<float>::zeros(padded.size())
};
cv::Mat complex;
cv::merge(planes, 2, complex);

// Compute DFT
cv::dft(complex, complex);

// Split real and complex planes
cv::split(complex, planes);

// Compute the magnitude and phase
cv::Mat mag, phase;
cv::cartToPolar(planes[0], planes[1], mag, phase);

// Shift quadrants so the Fourier image origin is in the center of the image
dftshift(mag);

// Switch to logarithmic scale to be able to display on screen
mag += cv::Scalar::all(1);
cv::log(mag, mag);

// Transform the matrix with float values into a viewable image form (float
// values between 0 and 1) and show the result.
cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
cv::imshow("Magnitude of input image", mag);

cv::Mat my_filtered = custom_filter(img); // Apply custom_filter
cv::imshow("Image 2: Result after applying the filter", my_filtered );
cv::Mat my_filtered_1 = custom_filter(my_filtered); // Apply custom_filter again
cv::imshow("Image 3: Result after applying the filter 2 times", my_filtered_1);

cv::Mat histogram1 = histogram(my_filtered_1 );
cv::imshow("Histogram of image 3", draw_histogram_image(histogram1));

cv::Mat my_filtered_2=transform_intensity(my_filtered_1, 0, 1.5); // Multiply every pixel by 1.5 to increase contrast
cv::imshow("Image 4: intensity transformation applied to Image 3", my_filtered_2);

cv::Mat histogram2 = histogram(my_filtered_2 );
cv::imshow("Histogram of image 4", draw_histogram_image(histogram2));

while (cv::waitKey() != 27) // Waiting for Esc to be pressed
    ;

return 0;
}
