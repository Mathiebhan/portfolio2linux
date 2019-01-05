/*
             RoVi1
       Vision MiniProject

Haris Sestan and Mathiebhan Mahendran

   Code for filtering Image4_1.png
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

// Returns a complex Butterworth high-pass filter of order 'n', diameter 'd0'
// and size 'size', and center shifted by 'px' and 'py'.
// (Modified version of the exercise solution files on BlackBoard)
cv::Mat butter_highpass(float d0, int n, cv::Size size, int px, int py)
{
    cv::Mat_<cv::Vec2f> hpf(size);
    cv::Point2f c = cv::Point2f(size) / 2;

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            float d = std::sqrt((i - c.y - py) * (i - c.y - py) + (j - c.x - px) * (j - c.x - px));

            // Real part
            if (std::abs(d) < 1.e-9f) // Avoid division by zero
                hpf(i, j)[0] = 0;
            else {
                hpf(i, j)[0] = 1 / (1 + std::pow(d0 / d, n));
            }

            // Imaginary part
            hpf(i, j)[1] = 0;
        }
    }

    return hpf;
}

// Function for displaying spectrum magnitude, taken from http://breckon.eu/toby/teaching/dip/opencv/lecture_demos/c++/butterworth_lowpass.cpp
cv::Mat create_spectrum_magnitude_display(cv::Mat& complexImg, bool rearrange)
{
    cv::Mat planes[2];

    // compute magnitude spectrum (N.B. for display)
    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))

    cv::split(complexImg, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);

    cv::Mat mag = (planes[0]).clone();
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);

    if (rearrange)
    {
        // re-arrange the quaderants
        dftshift(mag);
    }

    normalize(mag, mag, 0, 1, CV_MINMAX);

    return mag;

}

// Spectrum visualisation and modification taken from the exercise solution files on BlackBoard
int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv,
                                 "{help   |            | print this message}"
                                 "{@image | ./Image4_1.png | image path}"
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
    cv::Mat planes[] = {
        cv::Mat_<float>(padded),
        cv::Mat_<float>::zeros(padded.size())
    };
    cv::Mat complex;
    cv::merge(planes, 2, complex);

    // Compute DFT of image
    cv::dft(complex, complex);

    // Shift quadrants to center
    dftshift(complex);

    // Create 4 complex Butterworth filters (a pair of those is a notch filter), shifted to cover noise sources in the spectrum magnitude.
    cv::Mat filter;
    filter = butter_highpass(150, 2, complex.size(),200,207);
    cv::mulSpectrums(complex, filter, complex, 0);

    filter = butter_highpass(150, 2, complex.size(),-200,-207);
    cv::mulSpectrums(complex, filter, complex, 0);

    filter = butter_highpass(150, 2, complex.size(),-604,622);
    cv::mulSpectrums(complex, filter, complex, 0);

    filter = butter_highpass(150, 2, complex.size(),604,-622);
    cv::mulSpectrums(complex, filter, complex, 0);

    // Shift back
    dftshift(complex);
    cv::imshow("Image 2: Spectrum magnitude after applying notch filters", create_spectrum_magnitude_display(complex,true));

    // Compute inverse DFT
    cv::Mat filtered;
    cv::idft(complex, filtered, (cv::DFT_SCALE | cv::DFT_REAL_OUTPUT));

    // Crop image (remove padded borders)
    filtered = cv::Mat(filtered, cv::Rect(cv::Point(0, 0), img.size()));

    cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Image 3: Filtered image", filtered);

    while (cv::waitKey() != 27) // Waiting for Esc to be pressed
        ;

    return 0;
}
